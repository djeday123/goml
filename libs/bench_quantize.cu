#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>

extern "C"
{
    int fp8_quantize_fp16(const void *input_fp16, void *output_fp8, float *scale_out, int n);
    int fp8_dequantize_to_fp16(const void *input_fp8, void *output_fp16, const float *scale_in, int n);
    int cuda_device_sync2(void);
}

int main()
{
    printf("=== FP8 Quantization Benchmark ===\n\n");

    // Test correctness first with small tensor
    {
        printf("--- Correctness Test (1024 elements) ---\n");
        int n = 1024;
        std::vector<half> h_input(n);
        srand(42);
        float true_max = 0;
        for (int i = 0; i < n; i++)
        {
            float v = ((float)rand() / RAND_MAX - 0.5f) * 200.0f; // [-100, 100]
            h_input[i] = __float2half(v);
            true_max = fmaxf(true_max, fabsf(v));
        }

        half *d_input;
        uint8_t *d_fp8;
        half *d_output;
        float *d_scale;

        cudaMalloc(&d_input, n * sizeof(half));
        cudaMalloc(&d_fp8, n);
        cudaMalloc(&d_output, n * sizeof(half));
        cudaMalloc(&d_scale, sizeof(float));

        cudaMemcpy(d_input, h_input.data(), n * sizeof(half), cudaMemcpyHostToDevice);

        // Quantize
        fp8_quantize_fp16(d_input, d_fp8, d_scale, n);
        cuda_device_sync2();

        // Check scale
        float h_scale;
        cudaMemcpy(&h_scale, d_scale, sizeof(float), cudaMemcpyDeviceToHost);
        printf("  True absmax:  %.4f\n", true_max);
        printf("  GPU scale:    %.4f (absmax/448)\n", h_scale);
        printf("  Expected:     %.4f\n", true_max / 448.0f);
        printf("  Scale match:  %s\n", fabsf(h_scale - true_max / 448.0f) < 0.01f ? "OK" : "FAIL");

        // Dequantize
        fp8_dequantize_to_fp16(d_fp8, d_output, d_scale, n);
        cuda_device_sync2();

        // Check roundtrip error
        std::vector<half> h_output(n);
        cudaMemcpy(h_output.data(), d_output, n * sizeof(half), cudaMemcpyDeviceToHost);

        double mse = 0, max_err = 0;
        for (int i = 0; i < n; i++)
        {
            float orig = __half2float(h_input[i]);
            float recon = __half2float(h_output[i]);
            double err = fabs(orig - recon);
            mse += err * err;
            if (err > max_err)
                max_err = err;
        }
        mse /= n;
        printf("  Roundtrip MSE:    %.6f\n", mse);
        printf("  Roundtrip MaxErr: %.4f\n", max_err);
        printf("  Quality:          %s\n\n", max_err < 5.0f ? "OK" : "FAIL");

        cudaFree(d_input);
        cudaFree(d_fp8);
        cudaFree(d_output);
        cudaFree(d_scale);
    }

    // Benchmark throughput
    {
        printf("--- Throughput Benchmark ---\n");
        printf("%-12s %-12s %-12s %-12s\n", "Elements", "Quant(us)", "Dequant(us)", "GB/s");
        printf("------------------------------------------------\n");

        int sizes[] = {65536, 262144, 1048576, 4194304, 16777216, 67108864};
        int nsizes = 6;
        int iters = 100;
        int warmup = 20;

        for (int si = 0; si < nsizes; si++)
        {
            int n = sizes[si];

            half *d_input;
            uint8_t *d_fp8;
            half *d_output;
            float *d_scale;

            cudaMalloc(&d_input, n * sizeof(half));
            cudaMalloc(&d_fp8, n);
            cudaMalloc(&d_output, n * sizeof(half));
            cudaMalloc(&d_scale, sizeof(float));

            // Fill with random data
            std::vector<half> h_data(n);
            for (int i = 0; i < n; i++)
                h_data[i] = __float2half(((float)rand() / RAND_MAX - 0.5f) * 100.0f);
            cudaMemcpy(d_input, h_data.data(), n * sizeof(half), cudaMemcpyHostToDevice);

            // Warmup
            for (int i = 0; i < warmup; i++)
            {
                fp8_quantize_fp16(d_input, d_fp8, d_scale, n);
                fp8_dequantize_to_fp16(d_fp8, d_output, d_scale, n);
            }
            cudaDeviceSynchronize();

            // Benchmark quantize
            cudaEvent_t t0, t1;
            cudaEventCreate(&t0);
            cudaEventCreate(&t1);

            cudaEventRecord(t0);
            for (int i = 0; i < iters; i++)
                fp8_quantize_fp16(d_input, d_fp8, d_scale, n);
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            float q_ms;
            cudaEventElapsedTime(&q_ms, t0, t1);
            float q_us = (q_ms / iters) * 1000.0f;

            // Benchmark dequantize
            cudaEventRecord(t0);
            for (int i = 0; i < iters; i++)
                fp8_dequantize_to_fp16(d_fp8, d_output, d_scale, n);
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            float dq_ms;
            cudaEventElapsedTime(&dq_ms, t0, t1);
            float dq_us = (dq_ms / iters) * 1000.0f;

            // GB/s for quantize: read n*2 bytes (FP16) + write n bytes (FP8)
            double gbytes = (double)n * 3.0 / 1e9;
            double gbs = gbytes / (q_us / 1e6);

            printf("%-12d %-12.1f %-12.1f %-12.1f\n", n, q_us, dq_us, gbs);

            cudaEventDestroy(t0);
            cudaEventDestroy(t1);
            cudaFree(d_input);
            cudaFree(d_fp8);
            cudaFree(d_output);
            cudaFree(d_scale);
        }
    }

    printf("\nReference: RTX 4090 memory bandwidth = 1008 GB/s\n");
    return 0;
}