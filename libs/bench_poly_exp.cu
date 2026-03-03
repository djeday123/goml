// Throughput: MUFU.EX2.F16 vs polynomial exp2 on SM89
// Build: nvcc -O3 -arch=sm_89 -std=c++17 bench_poly_exp.cu -o bench_poly_exp -lcudart

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define THREADS 128
#define ITERS 10000

// ---- FP16 MUFU.EX2.F16 (current v54) ----
__device__ __forceinline__ uint32_t hexp2x2(uint32_t in)
{
    uint32_t out;
    asm("ex2.approx.f16x2 %0, %1;" : "=r"(out) : "r"(in));
    return out;
}

__global__ void bench_mufu_f16(float *out)
{
    __half2 h0 = __float2half2_rn(0.1f);
    __half2 h1 = __float2half2_rn(0.2f);
    __half2 h2 = __float2half2_rn(0.3f);
    __half2 h3 = __float2half2_rn(0.4f);
    __half2 one = __float2half2_rn(1.0f);
    __half2 log2e = __float2half2_rn(1.4426950408889634f);

#pragma unroll 1
    for (int i = 0; i < ITERS; i++)
    {
        // 16 ex2.f16x2 = 32 exp per iter (match v54 softmax: 34 exp but close)
        h0 = __hsub2(h0, one);
        h1 = __hsub2(h1, one);
        h2 = __hsub2(h2, one);
        h3 = __hsub2(h3, one);
        __half2 t0 = __hmul2(h0, log2e);
        __half2 t1 = __hmul2(h1, log2e);
        __half2 t2 = __hmul2(h2, log2e);
        __half2 t3 = __hmul2(h3, log2e);
        uint32_t *p0 = (uint32_t *)&t0, *p1 = (uint32_t *)&t1, *p2 = (uint32_t *)&t2, *p3 = (uint32_t *)&t3;
        *p0 = hexp2x2(*p0);
        *p1 = hexp2x2(*p1);
        *p2 = hexp2x2(*p2);
        *p3 = hexp2x2(*p3);
        h0 = t0;
        h1 = t1;
        h2 = t2;
        h3 = t3;

        h0 = __hsub2(h0, one);
        h1 = __hsub2(h1, one);
        h2 = __hsub2(h2, one);
        h3 = __hsub2(h3, one);
        t0 = __hmul2(h0, log2e);
        t1 = __hmul2(h1, log2e);
        t2 = __hmul2(h2, log2e);
        t3 = __hmul2(h3, log2e);
        p0 = (uint32_t *)&t0;
        p1 = (uint32_t *)&t1;
        p2 = (uint32_t *)&t2;
        p3 = (uint32_t *)&t3;
        *p0 = hexp2x2(*p0);
        *p1 = hexp2x2(*p1);
        *p2 = hexp2x2(*p2);
        *p3 = hexp2x2(*p3);
        h0 = t0;
        h1 = t1;
        h2 = t2;
        h3 = t3;
        // 32 exp total
    }
    if (threadIdx.x == 0)
        out[0] = __half2float(__low2half(h0));
}

// ---- FP32 polynomial exp2(x) on CUDA cores ----
// exp2(x) ≈ 1 + x*(0.6931472 + x*(0.2402265 + x*0.0558011))
// 3 FMA per exp, runs on CUDA cores (128/SM) not SFU (4/SM)
__device__ __forceinline__ float poly_exp2(float x)
{
    // Clamp to avoid overflow
    x = fmaxf(fminf(x, 10.0f), -15.0f);
    float fi = floorf(x);
    float f = x - fi;
    int i = (int)fi;
    // Horner: 2^f ≈ 1 + f*(c1 + f*(c2 + f*c3))
    float p = 1.0f + f * (0.6931472f + f * (0.2402265f + f * 0.0558011f));
    // 2^i via integer trick
    union
    {
        float fv;
        int iv;
    } u;
    u.iv = (i + 127) << 23;
    return u.fv * p;
}

__global__ void bench_poly_fp32(float *out)
{
    float v0 = 0.1f, v1 = 0.2f, v2 = 0.3f, v3 = 0.4f;
    float v4 = 0.5f, v5 = 0.6f, v6 = 0.7f, v7 = 0.8f;

#pragma unroll 1
    for (int i = 0; i < ITERS; i++)
    {
        // 32 poly_exp2 per iter
        float l = 1.4426950408889634f;
        v0 = poly_exp2((v0 - 1.0f) * l);
        v1 = poly_exp2((v1 - 1.0f) * l);
        v2 = poly_exp2((v2 - 1.0f) * l);
        v3 = poly_exp2((v3 - 1.0f) * l);
        v4 = poly_exp2((v4 - 1.0f) * l);
        v5 = poly_exp2((v5 - 1.0f) * l);
        v6 = poly_exp2((v6 - 1.0f) * l);
        v7 = poly_exp2((v7 - 1.0f) * l);
        v0 = poly_exp2((v0 - 1.0f) * l);
        v1 = poly_exp2((v1 - 1.0f) * l);
        v2 = poly_exp2((v2 - 1.0f) * l);
        v3 = poly_exp2((v3 - 1.0f) * l);
        v4 = poly_exp2((v4 - 1.0f) * l);
        v5 = poly_exp2((v5 - 1.0f) * l);
        v6 = poly_exp2((v6 - 1.0f) * l);
        v7 = poly_exp2((v7 - 1.0f) * l);
        v0 = poly_exp2((v0 - 1.0f) * l);
        v1 = poly_exp2((v1 - 1.0f) * l);
        v2 = poly_exp2((v2 - 1.0f) * l);
        v3 = poly_exp2((v3 - 1.0f) * l);
        v4 = poly_exp2((v4 - 1.0f) * l);
        v5 = poly_exp2((v5 - 1.0f) * l);
        v6 = poly_exp2((v6 - 1.0f) * l);
        v7 = poly_exp2((v7 - 1.0f) * l);
        v0 = poly_exp2((v0 - 1.0f) * l);
        v1 = poly_exp2((v1 - 1.0f) * l);
        v2 = poly_exp2((v2 - 1.0f) * l);
        v3 = poly_exp2((v3 - 1.0f) * l);
        v4 = poly_exp2((v4 - 1.0f) * l);
        v5 = poly_exp2((v5 - 1.0f) * l);
        v6 = poly_exp2((v6 - 1.0f) * l);
        v7 = poly_exp2((v7 - 1.0f) * l);
    }
    if (threadIdx.x == 0)
        out[0] = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
}

// ---- FP16 polynomial exp2 on CUDA cores (half2 HFMA2) ----
__device__ __forceinline__ __half2 poly_exp2_h2(__half2 x)
{
    const __half2 c3 = __float2half2_rn(0.0558011f);
    const __half2 c2 = __float2half2_rn(0.2402265f);
    const __half2 c1 = __float2half2_rn(0.6931472f);
    const __half2 one = __float2half2_rn(1.0f);
    // Horner: 1 + x*(c1 + x*(c2 + x*c3))
    __half2 p = __hfma2(x, c3, c2);
    p = __hfma2(x, p, c1);
    p = __hfma2(x, p, one);
    // Skip 2^i integer trick — just measure polynomial throughput
    return p;
}

__global__ void bench_poly_fp16(float *out)
{
    __half2 h0 = __float2half2_rn(0.1f);
    __half2 h1 = __float2half2_rn(0.2f);
    __half2 h2 = __float2half2_rn(0.3f);
    __half2 h3 = __float2half2_rn(0.4f);
    __half2 one = __float2half2_rn(1.0f);
    __half2 log2e = __float2half2_rn(1.4426950408889634f);

#pragma unroll 1
    for (int i = 0; i < ITERS; i++)
    {
        // 16 poly_exp2_h2 = 32 exp per group × 2 groups = 32 per iter
        __half2 t0 = __hmul2(__hsub2(h0, one), log2e);
        __half2 t1 = __hmul2(__hsub2(h1, one), log2e);
        __half2 t2 = __hmul2(__hsub2(h2, one), log2e);
        __half2 t3 = __hmul2(__hsub2(h3, one), log2e);
        h0 = poly_exp2_h2(t0);
        h1 = poly_exp2_h2(t1);
        h2 = poly_exp2_h2(t2);
        h3 = poly_exp2_h2(t3);

        t0 = __hmul2(__hsub2(h0, one), log2e);
        t1 = __hmul2(__hsub2(h1, one), log2e);
        t2 = __hmul2(__hsub2(h2, one), log2e);
        t3 = __hmul2(__hsub2(h3, one), log2e);
        h0 = poly_exp2_h2(t0);
        h1 = poly_exp2_h2(t1);
        h2 = poly_exp2_h2(t2);
        h3 = poly_exp2_h2(t3);
    }
    if (threadIdx.x == 0)
        out[0] = __half2float(__low2half(h0));
}

// ---- Baseline: FP32 MUFU.EX2 (original v20 path) ----
__global__ void bench_mufu_fp32(float *out)
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
    }
    if (threadIdx.x == 0)
        out[0] = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
}

int main()
{
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("=== Exp Throughput: MUFU vs Polynomial ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n", p.name, p.multiProcessorCount, p.clockRate / 1000);
    printf("%d threads, %d iters, 32 exp/iter\n\n", THREADS, ITERS);

    float *d_out;
    cudaMalloc(&d_out, 64);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // Warmup all
    bench_mufu_fp32<<<1, THREADS>>>(d_out);
    bench_mufu_f16<<<1, THREADS>>>(d_out);
    bench_poly_fp32<<<1, THREADS>>>(d_out);
    bench_poly_fp16<<<1, THREADS>>>(d_out);
    cudaDeviceSynchronize();

    double ops = (double)THREADS * ITERS * 32;

    // 1) FP32 MUFU.EX2 (v20 baseline)
    cudaEventRecord(t0);
    bench_mufu_fp32<<<1, THREADS>>>(d_out);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms1;
    cudaEventElapsedTime(&ms1, t0, t1);

    // 2) FP16 MUFU.EX2.F16 (v54 current)
    cudaEventRecord(t0);
    bench_mufu_f16<<<1, THREADS>>>(d_out);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms2;
    cudaEventElapsedTime(&ms2, t0, t1);

    // 3) FP32 polynomial
    cudaEventRecord(t0);
    bench_poly_fp32<<<1, THREADS>>>(d_out);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms3;
    cudaEventElapsedTime(&ms3, t0, t1);

    // 4) FP16 polynomial (HFMA2)
    cudaEventRecord(t0);
    bench_poly_fp16<<<1, THREADS>>>(d_out);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms4;
    cudaEventElapsedTime(&ms4, t0, t1);

    printf("FP32 MUFU.EX2 (v20):    %8.3f ms  (%7.1f Gops/s)\n", ms1, ops / ms1 / 1e6);
    printf("FP16 MUFU.EX2.F16 (v54):%8.3f ms  (%7.1f Gops/s)\n", ms2, ops / ms2 / 1e6);
    printf("FP32 polynomial:        %8.3f ms  (%7.1f Gops/s)\n", ms3, ops / ms3 / 1e6);
    printf("FP16 polynomial HFMA2:  %8.3f ms  (%7.1f Gops/s)\n", ms4, ops / ms4 / 1e6);
    printf("\nRelative to FP16 MUFU (v54 current):\n");
    printf("  FP32 MUFU:  %.2fx slower\n", ms1 / ms2);
    printf("  FP32 poly:  %.2fx %s\n", ms2 < ms3 ? ms3 / ms2 : ms2 / ms3, ms3 < ms2 ? "faster" : "slower");
    printf("  FP16 poly:  %.2fx %s\n", ms2 < ms4 ? ms4 / ms2 : ms2 / ms4, ms4 < ms2 ? "faster" : "slower");

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_out);
    return 0;
}
