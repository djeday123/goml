#include <cublasLt.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
    cublasLtHandle_t handle;
    int32_t M, N, K, _pad;
    const void *A, *B;
    void *C;
    const float *alpha, *beta;
} Fp8MatmulArgs;

typedef struct
{
    cublasLtHandle_t handle;
    int32_t M, N, K, epilogue;
    const void *A, *B;
    void *C;
    const float *alpha, *beta;
    const void *bias;
} Fp8FusedArgs;

extern int fp8_matmul_wrapper(Fp8MatmulArgs *a);
extern int fp8_matmul_fused(Fp8FusedArgs *a);
extern int cuda_device_sync(void);

// Simple ReLU kernel
__attribute__((weak)) void relu_on_gpu(void *data, int n);

int main()
{
    cublasLtHandle_t handle;
    cublasLtCreate(&handle);

    int sizes[] = {1024, 2048, 4096, 8192};
    int nsizes = 4;
    int iters = 50, warmup = 10;
    float alpha = 1.0f, beta = 0.0f;

    printf("=== Fused Epilogue Benchmark (FP8 MatMul) ===\n\n");

    // Test which epilogues are supported
    printf("--- Epilogue Support ---\n");
    {
        int M = 1024, N = 1024, K = 1024;
        void *dA, *dB, *dC, *dBias;
        cudaMalloc(&dA, M * K);
        cudaMalloc(&dB, K * N);
        cudaMalloc(&dC, M * N * 2);
        cudaMalloc(&dBias, N * 2); // FP16 bias, size N
        cudaMemset(dBias, 0, N * 2);

        const char *names[] = {"DEFAULT", "RELU", "GELU", "BIAS", "RELU+BIAS", "GELU+BIAS"};
        for (int epi = 0; epi <= 5; epi++)
        {
            Fp8FusedArgs fa;
            memset(&fa, 0, sizeof(fa));
            fa.handle = handle;
            fa.M = M;
            fa.N = N;
            fa.K = K;
            fa.epilogue = epi;
            fa.A = dA;
            fa.B = dB;
            fa.C = dC;
            fa.alpha = &alpha;
            fa.beta = &beta;
            fa.bias = (epi >= 3) ? dBias : NULL;

            int ret = fp8_matmul_fused(&fa);
            cudaDeviceSynchronize();
            printf("  %-12s: %s\n", names[epi], ret == 0 ? "SUPPORTED" : "NOT SUPPORTED");
        }
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        cudaFree(dBias);
    }

    // Benchmark: unfused vs fused
    printf("\n--- Performance Comparison ---\n");
    printf("%-8s %-14s %-14s %-14s %-10s\n", "Size", "MatMul(ms)", "MM+GELU(ms)", "Fused(ms)", "Savings");
    printf("--------------------------------------------------------------\n");

    for (int si = 0; si < nsizes; si++)
    {
        int M = sizes[si], N = sizes[si], K = sizes[si];

        void *dA, *dB, *dC, *dC2, *dBias;
        cudaMalloc(&dA, (size_t)M * K);
        cudaMalloc(&dB, (size_t)K * N);
        cudaMalloc(&dC, (size_t)M * N * 2);
        cudaMalloc(&dC2, (size_t)M * N * 2);
        cudaMalloc(&dBias, (size_t)N * 2);

        // Fill random
        size_t maxsz = (size_t)M * K;
        if ((size_t)K * N > maxsz)
            maxsz = (size_t)K * N;
        uint8_t *tmp = (uint8_t *)malloc(maxsz);
        for (size_t i = 0; i < maxsz; i++)
            tmp[i] = rand() & 0x3F;
        cudaMemcpy(dA, tmp, M * K, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, tmp, K * N, cudaMemcpyHostToDevice);
        cudaMemset(dBias, 0, N * 2);
        free(tmp);

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);

        // 1. MatMul only (baseline)
        Fp8MatmulArgs ma;
        memset(&ma, 0, sizeof(ma));
        ma.handle = handle;
        ma.M = M;
        ma.N = N;
        ma.K = K;
        ma.A = dA;
        ma.B = dB;
        ma.C = dC;
        ma.alpha = &alpha;
        ma.beta = &beta;

        for (int i = 0; i < warmup; i++)
            fp8_matmul_wrapper(&ma);
        cudaDeviceSynchronize();

        cudaEventRecord(t0, 0);
        for (int i = 0; i < iters; i++)
            fp8_matmul_wrapper(&ma);
        cudaEventRecord(t1, 0);
        cudaEventSynchronize(t1);
        float ms_mm;
        cudaEventElapsedTime(&ms_mm, t0, t1);
        float avg_mm = ms_mm / iters;

        // 2. Fused MatMul + GELU + BIAS
        Fp8FusedArgs fa;
        memset(&fa, 0, sizeof(fa));
        fa.handle = handle;
        fa.M = M;
        fa.N = N;
        fa.K = K;
        fa.epilogue = 5; // GELU+BIAS
        fa.A = dA;
        fa.B = dB;
        fa.C = dC2;
        fa.alpha = &alpha;
        fa.beta = &beta;
        fa.bias = dBias;

        int ret = fp8_matmul_fused(&fa);
        cudaDeviceSynchronize();

        float avg_fused = -1;
        if (ret == 0)
        {
            for (int i = 0; i < warmup; i++)
                fp8_matmul_fused(&fa);
            cudaDeviceSynchronize();

            cudaEventRecord(t0, 0);
            for (int i = 0; i < iters; i++)
                fp8_matmul_fused(&fa);
            cudaEventRecord(t1, 0);
            cudaEventSynchronize(t1);
            float ms_f;
            cudaEventElapsedTime(&ms_f, t0, t1);
            avg_fused = ms_f / iters;
        }

        // Estimate unfused: matmul + separate bias + separate GELU
        // Each separate op = read+write M*N FP16 = M*N*4 bytes
        // At 1008 GB/s: time = M*N*4 / 1008e9
        double mem_ops_ns = ((double)M * N * 4.0 / 1008e9) * 1e3; // in ms
        float avg_unfused = avg_mm + (float)(mem_ops_ns * 2);     // bias + gelu

        if (avg_fused > 0)
        {
            float savings = (1.0f - avg_fused / avg_unfused) * 100.0f;
            printf("%-8d %-14.3f %-14.3f %-14.3f %.1f%%\n",
                   M, avg_mm, avg_unfused, avg_fused, savings);
        }
        else
        {
            printf("%-8d %-14.3f %-14.3f %-14s NOT SUPPORTED\n",
                   M, avg_mm, avg_unfused, "-");
        }

        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        cudaFree(dC2);
        cudaFree(dBias);
    }

    cublasLtDestroy(handle);
    return 0;
}