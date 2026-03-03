// Throughput: FP32 __expf vs FP16 ex2.approx.f16x2 on SM89
// Build: nvcc -O3 -arch=sm_89 -std=c++17 bench_hexp2.cu -o bench_hexp2 -lcudart

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define THREADS 128
#define ITERS 10000

// Kernel 1: FP32 __expf — current v20 approach
// 32 exp per thread per iteration (matches softmax: 8 tiles × 4 values)
__global__ void bench_fp32_expf(float *out)
{
    float v0 = 0.1f, v1 = 0.2f, v2 = 0.3f, v3 = 0.4f;
    float v4 = 0.5f, v5 = 0.6f, v6 = 0.7f, v7 = 0.8f;
#pragma unroll 1
    for (int i = 0; i < ITERS; i++)
    {
        v0 = __expf(v0 - 1.0f);
        v1 = __expf(v1 - 1.0f);
        v2 = __expf(v2 - 1.0f);
        v3 = __expf(v3 - 1.0f);
        v4 = __expf(v4 - 1.0f);
        v5 = __expf(v5 - 1.0f);
        v6 = __expf(v6 - 1.0f);
        v7 = __expf(v7 - 1.0f);
        v0 = __expf(v0 - 1.0f);
        v1 = __expf(v1 - 1.0f);
        v2 = __expf(v2 - 1.0f);
        v3 = __expf(v3 - 1.0f);
        v4 = __expf(v4 - 1.0f);
        v5 = __expf(v5 - 1.0f);
        v6 = __expf(v6 - 1.0f);
        v7 = __expf(v7 - 1.0f);
        v0 = __expf(v0 - 1.0f);
        v1 = __expf(v1 - 1.0f);
        v2 = __expf(v2 - 1.0f);
        v3 = __expf(v3 - 1.0f);
        v4 = __expf(v4 - 1.0f);
        v5 = __expf(v5 - 1.0f);
        v6 = __expf(v6 - 1.0f);
        v7 = __expf(v7 - 1.0f);
        v0 = __expf(v0 - 1.0f);
        v1 = __expf(v1 - 1.0f);
        v2 = __expf(v2 - 1.0f);
        v3 = __expf(v3 - 1.0f);
        v4 = __expf(v4 - 1.0f);
        v5 = __expf(v5 - 1.0f);
        v6 = __expf(v6 - 1.0f);
        v7 = __expf(v7 - 1.0f);
        // 32 __expf per iteration
    }
    if (threadIdx.x == 0)
        out[0] = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
}

// Kernel 2: FP16 ex2.approx.f16x2 — proposed approach
// Same 32 exp per thread but as 16 × f16x2
__device__ __forceinline__ uint32_t hexp2x2(uint32_t in)
{
    uint32_t out;
    asm("ex2.approx.f16x2 %0, %1;" : "=r"(out) : "r"(in));
    return out;
}

__global__ void bench_fp16_ex2(float *out)
{
    // Pack initial values as half2
    __half2 h0 = __float2half2_rn(0.1f);
    __half2 h1 = __float2half2_rn(0.2f);
    __half2 h2 = __float2half2_rn(0.3f);
    __half2 h3 = __float2half2_rn(0.4f);
    __half2 h4 = __float2half2_rn(0.5f);
    __half2 h5 = __float2half2_rn(0.6f);
    __half2 h6 = __float2half2_rn(0.7f);
    __half2 h7 = __float2half2_rn(0.8f);

    __half2 one = __float2half2_rn(1.0f);
    __half2 log2e = __float2half2_rn(1.4426950408889634f);

    uint32_t *p0 = (uint32_t *)&h0, *p1 = (uint32_t *)&h1;
    uint32_t *p2 = (uint32_t *)&h2, *p3 = (uint32_t *)&h3;
    uint32_t *p4 = (uint32_t *)&h4, *p5 = (uint32_t *)&h5;
    uint32_t *p6 = (uint32_t *)&h6, *p7 = (uint32_t *)&h7;

#pragma unroll 1
    for (int i = 0; i < ITERS; i++)
    {
        // exp(x) = exp2(x * log2e), subtract 1.0 first
        // Each f16x2 does 2 exp → 16 calls = 32 exp
        h0 = __hsub2(h0, one);
        *p0 = hexp2x2(*reinterpret_cast<uint32_t *>(&(__half2 &)(h0 = __hmul2(h0, log2e))));
        h1 = __hsub2(h1, one);
        *p1 = hexp2x2(*reinterpret_cast<uint32_t *>(&(__half2 &)(h1 = __hmul2(h1, log2e))));
        h2 = __hsub2(h2, one);
        *p2 = hexp2x2(*reinterpret_cast<uint32_t *>(&(__half2 &)(h2 = __hmul2(h2, log2e))));
        h3 = __hsub2(h3, one);
        *p3 = hexp2x2(*reinterpret_cast<uint32_t *>(&(__half2 &)(h3 = __hmul2(h3, log2e))));
        h4 = __hsub2(h4, one);
        *p4 = hexp2x2(*reinterpret_cast<uint32_t *>(&(__half2 &)(h4 = __hmul2(h4, log2e))));
        h5 = __hsub2(h5, one);
        *p5 = hexp2x2(*reinterpret_cast<uint32_t *>(&(__half2 &)(h5 = __hmul2(h5, log2e))));
        h6 = __hsub2(h6, one);
        *p6 = hexp2x2(*reinterpret_cast<uint32_t *>(&(__half2 &)(h6 = __hmul2(h6, log2e))));
        h7 = __hsub2(h7, one);
        *p7 = hexp2x2(*reinterpret_cast<uint32_t *>(&(__half2 &)(h7 = __hmul2(h7, log2e))));
        h0 = *reinterpret_cast<__half2 *>(p0);
        h1 = *reinterpret_cast<__half2 *>(p1);
        h2 = *reinterpret_cast<__half2 *>(p2);
        h3 = *reinterpret_cast<__half2 *>(p3);
        h4 = *reinterpret_cast<__half2 *>(p4);
        h5 = *reinterpret_cast<__half2 *>(p5);
        h6 = *reinterpret_cast<__half2 *>(p6);
        h7 = *reinterpret_cast<__half2 *>(p7);

        h0 = __hsub2(h0, one);
        *p0 = hexp2x2(*reinterpret_cast<uint32_t *>(&(__half2 &)(h0 = __hmul2(h0, log2e))));
        h1 = __hsub2(h1, one);
        *p1 = hexp2x2(*reinterpret_cast<uint32_t *>(&(__half2 &)(h1 = __hmul2(h1, log2e))));
        h2 = __hsub2(h2, one);
        *p2 = hexp2x2(*reinterpret_cast<uint32_t *>(&(__half2 &)(h2 = __hmul2(h2, log2e))));
        h3 = __hsub2(h3, one);
        *p3 = hexp2x2(*reinterpret_cast<uint32_t *>(&(__half2 &)(h3 = __hmul2(h3, log2e))));
        h4 = __hsub2(h4, one);
        *p4 = hexp2x2(*reinterpret_cast<uint32_t *>(&(__half2 &)(h4 = __hmul2(h4, log2e))));
        h5 = __hsub2(h5, one);
        *p5 = hexp2x2(*reinterpret_cast<uint32_t *>(&(__half2 &)(h5 = __hmul2(h5, log2e))));
        h6 = __hsub2(h6, one);
        *p6 = hexp2x2(*reinterpret_cast<uint32_t *>(&(__half2 &)(h6 = __hmul2(h6, log2e))));
        h7 = __hsub2(h7, one);
        *p7 = hexp2x2(*reinterpret_cast<uint32_t *>(&(__half2 &)(h7 = __hmul2(h7, log2e))));
        h0 = *reinterpret_cast<__half2 *>(p0);
        h1 = *reinterpret_cast<__half2 *>(p1);
        h2 = *reinterpret_cast<__half2 *>(p2);
        h3 = *reinterpret_cast<__half2 *>(p3);
        h4 = *reinterpret_cast<__half2 *>(p4);
        h5 = *reinterpret_cast<__half2 *>(p5);
        h6 = *reinterpret_cast<__half2 *>(p6);
        h7 = *reinterpret_cast<__half2 *>(p7);
        // 32 exp per iteration (16 × f16x2 × 2)
    }
    if (threadIdx.x == 0)
        out[0] = __half2float(__low2half(h0));
}

// Kernel 3: FP32 exp with F2F convert overhead (what __expf actually does)
// This simulates the full v20 path: Sr is float, exp in float, then convert to half for P
__global__ void bench_fp32_full_path(float *out)
{
    float v0 = 0.1f, v1 = 0.2f, v2 = 0.3f, v3 = 0.4f;
    float v4 = 0.5f, v5 = 0.6f, v6 = 0.7f, v7 = 0.8f;
    __half2 dummy = __float2half2_rn(0.0f);

#pragma unroll 1
    for (int i = 0; i < ITERS; i++)
    {
        // 32 × (expf + float2half pack) — full softmax path
        v0 = __expf(v0 - 1.0f);
        v1 = __expf(v1 - 1.0f);
        v2 = __expf(v2 - 1.0f);
        v3 = __expf(v3 - 1.0f);
        v4 = __expf(v4 - 1.0f);
        v5 = __expf(v5 - 1.0f);
        v6 = __expf(v6 - 1.0f);
        v7 = __expf(v7 - 1.0f);
        // Pack to P (half2) — this is what v20 does
        dummy = __halves2half2(__float2half(v0), __float2half(v1));
        dummy = __halves2half2(__float2half(v2), __float2half(v3));
        dummy = __halves2half2(__float2half(v4), __float2half(v5));
        dummy = __halves2half2(__float2half(v6), __float2half(v7));

        v0 = __expf(v0 - 1.0f);
        v1 = __expf(v1 - 1.0f);
        v2 = __expf(v2 - 1.0f);
        v3 = __expf(v3 - 1.0f);
        v4 = __expf(v4 - 1.0f);
        v5 = __expf(v5 - 1.0f);
        v6 = __expf(v6 - 1.0f);
        v7 = __expf(v7 - 1.0f);
        dummy = __halves2half2(__float2half(v0), __float2half(v1));
        dummy = __halves2half2(__float2half(v2), __float2half(v3));
        dummy = __halves2half2(__float2half(v4), __float2half(v5));
        dummy = __halves2half2(__float2half(v6), __float2half(v7));

        v0 = __expf(v0 - 1.0f);
        v1 = __expf(v1 - 1.0f);
        v2 = __expf(v2 - 1.0f);
        v3 = __expf(v3 - 1.0f);
        v4 = __expf(v4 - 1.0f);
        v5 = __expf(v5 - 1.0f);
        v6 = __expf(v6 - 1.0f);
        v7 = __expf(v7 - 1.0f);
        dummy = __halves2half2(__float2half(v0), __float2half(v1));
        dummy = __halves2half2(__float2half(v2), __float2half(v3));
        dummy = __halves2half2(__float2half(v4), __float2half(v5));
        dummy = __halves2half2(__float2half(v6), __float2half(v7));

        v0 = __expf(v0 - 1.0f);
        v1 = __expf(v1 - 1.0f);
        v2 = __expf(v2 - 1.0f);
        v3 = __expf(v3 - 1.0f);
        v4 = __expf(v4 - 1.0f);
        v5 = __expf(v5 - 1.0f);
        v6 = __expf(v6 - 1.0f);
        v7 = __expf(v7 - 1.0f);
        dummy = __halves2half2(__float2half(v0), __float2half(v1));
        dummy = __halves2half2(__float2half(v2), __float2half(v3));
        dummy = __halves2half2(__float2half(v4), __float2half(v5));
        dummy = __halves2half2(__float2half(v6), __float2half(v7));
    }
    if (threadIdx.x == 0)
    {
        out[0] = v0 + __half2float(__low2half(dummy));
    }
}

int main()
{
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("=== MUFU.EX2 Throughput: FP32 vs FP16 ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n", p.name, p.multiProcessorCount, p.clockRate / 1000);
    printf("%d threads, %d iters, 32 exp/iter\n\n", THREADS, ITERS);

    float *d_out;
    cudaMalloc(&d_out, 64);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // Warmup
    bench_fp32_expf<<<1, THREADS>>>(d_out);
    bench_fp16_ex2<<<1, THREADS>>>(d_out);
    bench_fp32_full_path<<<1, THREADS>>>(d_out);
    cudaDeviceSynchronize();

    // FP32 __expf only
    cudaEventRecord(t0);
    bench_fp32_expf<<<1, THREADS>>>(d_out);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_fp32;
    cudaEventElapsedTime(&ms_fp32, t0, t1);

    // FP16 ex2.approx.f16x2
    cudaEventRecord(t0);
    bench_fp16_ex2<<<1, THREADS>>>(d_out);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_fp16;
    cudaEventElapsedTime(&ms_fp16, t0, t1);

    // FP32 full path (expf + float2half)
    cudaEventRecord(t0);
    bench_fp32_full_path<<<1, THREADS>>>(d_out);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_full;
    cudaEventElapsedTime(&ms_full, t0, t1);

    double ops = (double)THREADS * ITERS * 32;
    printf("FP32 __expf only:    %8.3f ms  (%7.1f Gops/s)\n", ms_fp32, ops / ms_fp32 / 1e6);
    printf("FP16 ex2.f16x2:      %8.3f ms  (%7.1f Gops/s)\n", ms_fp16, ops / ms_fp16 / 1e6);
    printf("FP32 full (exp+cvt): %8.3f ms  (%7.1f Gops/s)\n", ms_full, ops / ms_full / 1e6);
    printf("\nFP16/FP32 speedup:   %.2fx\n", ms_fp32 / ms_fp16);
    printf("FP16/Full speedup:   %.2fx\n", ms_full / ms_fp16);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_out);
    return 0;
}
