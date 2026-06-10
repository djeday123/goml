// sparse_mma_peak.cu — peak microbench for sparse mma variants
// Hypothesis: NVIDIA's "4000 AI TOPS" marketing might be sparse FP8 (= ~2200T)
// or sparse FP4 (= ~4400T). Probe to find out what really runs at top speed.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdint.h>

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); }} while(0)

#define ITERS 1000000

// Sparse FP8 m16n8k64 = 16384 FLOPs/MMA (2× of dense m16n8k32 8192)
__global__ void sparse_fp8(int *bo)
{
    uint32_t a0=0x3C3C3C3Cu, a1=0x3C3C3C3Cu, a2=0x3C3C3C3Cu, a3=0x3C3C3C3Cu;
    uint32_t b0=0x3C3C3C3Cu, b1=0x3C3C3C3Cu, b2=0x3C3C3C3Cu, b3=0x3C3C3C3Cu;
    uint32_t meta=0xE4u;  // 2:4 pattern
    float c0=0, c1=0, c2=0, c3=0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        asm volatile(
            "mma.sp.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%0,%1,%2,%3},%12, 0x0;\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(meta));
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) bo[0] = (int)(c0+c1+c2+c3);
}

// Sparse INT8 same shape
__global__ void sparse_int8(int *bo)
{
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,b2=0,b3=0;
    uint32_t meta=0xE4u;
    int c0=0, c1=0, c2=0, c3=0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        asm volatile(
            "mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%0,%1,%2,%3},%12, 0x0;\n"
            : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(meta));
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) bo[0] = c0+c1+c2+c3;
}

double bench(void (*k)(int*), int blocks, int threads, double flops)
{
    int *bo; CK(cudaMalloc(&bo, sizeof(int)));
    CK(cudaMemset(bo, 0, sizeof(int)));
    k<<<blocks, threads>>>(bo);
    CK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 3; i++) k<<<blocks, threads>>>(bo);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    double tf = 3.0 * blocks * threads / 32.0 * (double)ITERS * flops / (ms / 1000.0) / 1e12;
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(bo);
    return tf;
}

int main()
{
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    int sm = p.multiProcessorCount;
    int clk = 0; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    printf("=== Sparse MMA peak microbench ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n\n", p.name, sm, clk / 1000);

    int blocks = sm * 4;
    int threads = 128;
    printf("Launch: %d blocks × %d threads\n\n", blocks, threads);

    double tf1 = bench(sparse_fp8, blocks, threads, 16384.0);
    printf("  Sparse FP8 m16n8k64 (.f32 accum) → %.1f TFLOPS\n", tf1);

    double tf2 = bench(sparse_int8, blocks, threads, 16384.0);
    printf("  Sparse INT8 m16n8k64 (.s32 accum) → %.1f TOPS\n", tf2);

    printf("\n=== Comparison ===\n");
    printf("Dense FP8  = 1099 T\n");
    printf("Sparse FP8 = %.0f T  (ratio %.2f×)\n", tf1, tf1 / 1099.0);
    printf("Dense FP4  = 2198 T\n");
    printf("Sparse FP4 = (not supported on sm_120a)\n");
    printf("Sparse INT8 = %.0f TOPS\n", tf2);

    if (tf1 > 1500) {
        printf("\n*** 4000 TOPS marketing number ≈ sparse FP8 (≈ %.0fT) or sparse FP4 (~4400T)\n", tf1);
    }

    return 0;
}
