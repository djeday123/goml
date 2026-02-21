#include <cublasLt.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct
{
    cublasLtHandle_t handle;
    int32_t M, N, K, _pad;
    const void *A, *B;
    void *C;
    const float *alpha, *beta;
    const float *scaleA, *scaleB;
    float *scaleD, *amaxD;
} Fp8MatmulArgsEx;

extern int fp8_matmul_fp8out(Fp8MatmulArgsEx *a);
extern int fp8_matmul_wrapper(void *a); // existing FP16 output
extern int cuda_device_sync(void);

int main()
{
    cublasLtHandle_t handle;
    cublasLtCreate(&handle);

    int sizes[] = {1024, 2048, 4096, 8192};
    int nsizes = 4;
    int iters = 50, warmup = 10;

    printf("=== FP8 Output Mode Benchmark ===\n");
    printf("%-8s %-14s %-14s %-10s\n", "Size", "FP16out(ms)", "FP8out(ms)", "Savings");
    printf("-----------------------------------------------\n");

    for (int si = 0; si < nsizes; si++)
    {
        int M = sizes[si], N = sizes[si], K = sizes[si];
        size_t sA = (size_t)M * K;
        size_t sB = (size_t)K * N;
        size_t sC16 = (size_t)M * N * 2; // FP16
        size_t sC8 = (size_t)M * N;      // FP8

        void *dA, *dB, *dC16, *dC8;
        float *dScaleA, *dScaleB, *dScaleD, *dAmaxD;

        cudaMalloc(&dA, sA);
        cudaMalloc(&dB, sB);
        cudaMalloc(&dC16, sC16);
        cudaMalloc(&dC8, sC8);
        cudaMalloc((void**)&dScaleA, sizeof(float));
        cudaMalloc((void**)&dScaleB, sizeof(float));
        cudaMalloc((void**)&dScaleD, sizeof(float));
        cudaMalloc((void**)&dAmaxD, sizeof(float));

        // Fill with random bytes
        uint8_t *tmp = (uint8_t *)malloc(sA > sB ? sA : sB);
        for (size_t i = 0; i < (sA > sB ? sA : sB); i++)
            tmp[i] = rand() & 0x3F;
        cudaMemcpy(dA, tmp, sA, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, tmp, sB, cudaMemcpyHostToDevice);
        free(tmp);

        float one = 1.0f;
        cudaMemcpy(dScaleA, &one, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dScaleB, &one, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dScaleD, &one, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(dAmaxD, 0, sizeof(float));

        float alpha = 1.0f, beta = 0.0f;

        // --- FP16 output timing ---
        struct
        {
            cublasLtHandle_t h;
            int32_t M, N, K, p;
            const void *A, *B;
            void *C;
            const float *a, *b;
        } args16;
        args16.h = handle;
        args16.M = M;
        args16.N = N;
        args16.K = K;
        args16.p = 0;
        args16.A = dA;
        args16.B = dB;
        args16.C = dC16;
        args16.a = &alpha;
        args16.b = &beta;

        for (int i = 0; i < warmup; i++)
            fp8_matmul_wrapper(&args16);
        cuda_device_sync();

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        cudaEventRecord(t0, 0);
        for (int i = 0; i < iters; i++)
            fp8_matmul_wrapper(&args16);
        cudaEventRecord(t1, 0);
        cudaEventSynchronize(t1);
        float ms16;
        cudaEventElapsedTime(&ms16, t0, t1);
        float avg16 = ms16 / iters;

        // --- FP8 output timing ---
        Fp8MatmulArgsEx args8;
        args8.handle = handle;
        args8.M = M;
        args8.N = N;
        args8.K = K;
        args8._pad = 0;
        args8.A = dA;
        args8.B = dB;
        args8.C = dC8;
        args8.alpha = &alpha;
        args8.beta = &beta;
        args8.scaleA = dScaleA;
        args8.scaleB = dScaleB;
        args8.scaleD = dScaleD;
        args8.amaxD = dAmaxD;

        int ret = fp8_matmul_fp8out(&args8);
        cuda_device_sync();

        float avg8 = -1;
        if (ret == 0)
        {
            for (int i = 0; i < warmup; i++)
                fp8_matmul_fp8out(&args8);
            cuda_device_sync();

            cudaEventRecord(t0, 0);
            for (int i = 0; i < iters; i++)
                fp8_matmul_fp8out(&args8);
            cudaEventRecord(t1, 0);
            cudaEventSynchronize(t1);
            float ms8;
            cudaEventElapsedTime(&ms8, t0, t1);
            avg8 = ms8 / iters;
        }

        if (avg8 > 0)
        {
            float savings = (1.0f - avg8 / avg16) * 100.0f;
            printf("%-8d %-14.3f %-14.3f %.1f%%\n", M, avg16, avg8, savings);
        }
        else
        {
            printf("%-8d %-14.3f %-14s NOT SUPPORTED (ret=%d)\n", M, avg16, "-", ret);
        }

        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC16);
        cudaFree(dC8);
        cudaFree(dScaleA);
        cudaFree(dScaleB);
        cudaFree(dScaleD);
        cudaFree(dAmaxD);
    }

    cublasLtDestroy(handle);
    return 0;
}