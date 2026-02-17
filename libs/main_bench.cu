#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Импортируем нашу функцию
extern "C" int run_cute_fp8_sm89(int M, int N, int K, const void *A, const void *B, void *C);

int main()
{
    // Размеры для теста (кратны 128 для оптимального тайлинга)
    int M = 8192;
    int N = 8192;
    int K = 8192;

    std::cout << "Starting SM89 FP8 -> FP16-Acc Benchmark..." << std::endl;

    size_t sizeA = (size_t)M * K * 1; // 1 byte for FP8
    size_t sizeB = (size_t)N * K * 1;
    size_t sizeC = (size_t)M * N * 2; // 2 bytes for FP16 (half)

    void *d_A, *d_B, *d_C;
    if (cudaMalloc(&d_A, sizeA) != cudaSuccess)
        return -1;
    if (cudaMalloc(&d_B, sizeB) != cudaSuccess)
        return -1;
    if (cudaMalloc(&d_C, sizeC) != cudaSuccess)
        return -1;

    // Инициализация (заполняем константой 1.0 в представлении FP8 e4m3 для простоты)
    cudaMemset(d_A, 0x3C, sizeA);
    cudaMemset(d_B, 0x3C, sizeB);
    cudaMemset(d_C, 0, sizeC);

    // 1. Разогрев (Warmup)
    for (int i = 0; i < 10; ++i)
    {
        run_cute_fp8_sm89(M, N, K, d_A, d_B, d_C);
    }
    cudaDeviceSynchronize();

    // 2. Замер времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i)
    {
        run_cute_fp8_sm89(M, N, K, d_A, d_B, d_C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);

    float msecPerIteration = msecTotal / iterations;
    double flopsPerIteration = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (flopsPerIteration / (msecPerIteration / 1000.0)) / 1e12;

    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Matrix Shape (M x N x K): " << M << " x " << N << " x " << K << std::endl;
    std::cout << "Avg Execution Time: " << msecPerIteration << " ms" << std::endl;
    std::cout << "Computed Performance: " << tflops << " TFLOPS" << std::endl;
    std::cout << "--------------------------------------" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}