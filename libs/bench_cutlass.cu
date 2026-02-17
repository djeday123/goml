#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

// Forward declare our wrapper
struct Fp8Args
{
    void *handle;
    int M, N, K;
    int _pad;
    const void *A;
    const void *B;
    void *C;
    const float *alpha;
    const float *beta;
};

// Will be loaded dynamically or linked
extern "C" int cutlass_fp8_gemm(Fp8Args *a);
extern "C" int cuda_device_sync(void);

// Simple FP8 E4M3 converter (truncate)
static unsigned char float_to_fp8_e4m3(float v)
{
    if (v != v)
        return 0x7F; // NaN
    if (v == 0.0f)
        return 0;
    unsigned char sign = (v < 0) ? 0x80 : 0;
    v = fabsf(v);
    if (v > 448.0f)
        v = 448.0f;
    // Use CUDA's conversion via half
    // Simplified: just store small random values
    int exp;
    float frac = frexpf(v, &exp); // v = frac * 2^exp, 0.5 <= frac < 1
    exp += 6;                     // bias=7, and frexp gives 0.5-based
    if (exp < 0)
        return sign;
    if (exp > 15)
        return sign | 0x7E;                             // max
    int mantissa = (int)((frac - 0.5f) * 16.0f + 0.5f); // 3 bit mantissa
    if (mantissa > 7)
        mantissa = 7;
    return sign | ((exp & 0xF) << 3) | (mantissa & 0x7);
}

int main(int argc, char **argv)
{
    int sizes[] = {512, 1024, 2048, 4096, 8192};
    int nsizes = 5;
    int iters = 50;
    int warmup = 10;

    printf("=== CUTLASS FP8 E4M3 Benchmark (Ada SM89) ===\n");
    printf("%-8s %-10s %-10s %-12s\n", "Size", "Time(ms)", "TFLOPS", "Status");
    printf("----------------------------------------------\n");

    for (int si = 0; si < nsizes; si++)
    {
        int M = sizes[si], N = sizes[si], K = sizes[si];
        size_t sizeA = (size_t)M * K;
        size_t sizeB = (size_t)K * N;
        size_t sizeC = (size_t)M * N;

        // Host data
        std::vector<unsigned char> hA(sizeA), hB(sizeB);
        std::vector<unsigned short> hC(sizeC, 0); // FP16 output

        // Fill with small random FP8 values
        srand(42 + si);
        for (size_t i = 0; i < sizeA; i++)
        {
            float v = ((float)(rand() % 200) - 100) / 100.0f; // [-1, 1]
            hA[i] = float_to_fp8_e4m3(v);
        }
        // B is column-major [K, N], so fill K*N elements
        for (size_t i = 0; i < sizeB; i++)
        {
            float v = ((float)(rand() % 200) - 100) / 100.0f;
            hB[i] = float_to_fp8_e4m3(v);
        }

        // Device alloc
        void *dA, *dB, *dC;
        cudaMalloc(&dA, sizeA);
        cudaMalloc(&dB, sizeB);
        cudaMalloc(&dC, sizeC * 2); // FP16 = 2 bytes

        cudaMemcpy(dA, hA.data(), sizeA, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), sizeB, cudaMemcpyHostToDevice);
        cudaMemset(dC, 0, sizeC * 2);

        float alpha = 1.0f, beta = 0.0f;

        Fp8Args args;
        memset(&args, 0, sizeof(args));
        args.M = M;
        args.N = N;
        args.K = K;
        args.A = dA;
        args.B = dB;
        args.C = dC;
        args.alpha = &alpha;
        args.beta = &beta;

        // Warmup
        int ret = 0;
        for (int i = 0; i < warmup; i++)
        {
            ret = cutlass_fp8_gemm(&args);
            if (ret != 0)
                break;
        }
        cuda_device_sync();

        if (ret != 0)
        {
            printf("%-8d %-10s %-10s err=%d\n", M, "-", "-", ret);
            cudaFree(dA);
            cudaFree(dB);
            cudaFree(dC);
            continue;
        }

        // Benchmark
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);

        cudaEventRecord(t0);
        for (int i = 0; i < iters; i++)
        {
            cutlass_fp8_gemm(&args);
        }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);

        float ms;
        cudaEventElapsedTime(&ms, t0, t1);
        float avg_ms = ms / iters;

        double flops = 2.0 * (double)M * (double)N * (double)K;
        double tflops = (flops / (avg_ms / 1000.0)) / 1e12;

        printf("%-8d %-10.3f %-10.1f OK\n", M, avg_ms, tflops);

        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }

    printf("\nReference: cuBLASLt FP8 = 330 TFLOPS, FP16 = 165 TFLOPS\n");
    return 0;
}