// fp8_mma_peak.cu — Raw FP8 MMA throughput microbench
//
// Probes the true tensor core peak by running m16n8k32 FP8 MMA in a tight
// loop with NO memory traffic. This tells us the hardware ceiling.
//
// Each MMA = 16*8*32*2 = 8192 FLOPs.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdint.h>

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); }} while(0)

#define ITERS 1000000

// FP16 accumulator path (the path v23 uses) — equivalent to QMMA.F16 SASS
__global__ void fp8_mma_peak_f16acc(int *barrier_out)
{
    // Input fragments (arbitrary, kept in registers)
    uint32_t a0 = 0x3C3C3C3Cu, a1 = 0x3C3C3C3Cu, a2 = 0x3C3C3C3Cu, a3 = 0x3C3C3C3Cu;
    uint32_t b0 = 0x3C3C3C3Cu, b1 = 0x3C3C3C3Cu;
    uint32_t c0 = 0u, c1 = 0u;
    uint32_t d0, d1;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++)
    {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
            "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
            : "=r"(d0), "=r"(d1)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1), "r"(c0), "r"(c1));
        c0 = d0; c1 = d1;
    }
    // Prevent dead-code elimination
    if (threadIdx.x == 0 && blockIdx.x == 0) barrier_out[0] = c0 ^ c1;
}

// FP32 accumulator path (what kind::f8f6f4 with f16 input apparently produces in SASS)
__global__ void fp8_mma_peak_f32acc(int *barrier_out)
{
    uint32_t a0 = 0x3C3C3C3Cu, a1 = 0x3C3C3C3Cu, a2 = 0x3C3C3C3Cu, a3 = 0x3C3C3C3Cu;
    uint32_t b0 = 0x3C3C3C3Cu, b1 = 0x3C3C3C3Cu;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++)
    {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1));
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int dummy = (int)(c0 + c1 + c2 + c3);
        barrier_out[0] = dummy;
    }
}

// kind::f8f6f4 variant — what our v24 actually compiled into
__global__ void fp8_mma_peak_kindf8f6f4(int *barrier_out)
{
    uint32_t a0 = 0x3C3C3C3Cu, a1 = 0x3C3C3C3Cu, a2 = 0x3C3C3C3Cu, a3 = 0x3C3C3C3Cu;
    uint32_t b0 = 0x3C3C3C3Cu, b1 = 0x3C3C3C3Cu;
    uint32_t c0 = 0u, c1 = 0u;
    uint32_t d0, d1;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++)
    {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e4m3.e4m3.f16 "
            "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
            : "=r"(d0), "=r"(d1)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1), "r"(c0), "r"(c1));
        c0 = d0; c1 = d1;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) barrier_out[0] = c0 ^ c1;
}

// FP16 m16n8k16 (для сравнения с FP8 — должен быть в 2× медленнее)
__global__ void fp16_mma_peak_f16acc(int *barrier_out)
{
    uint32_t a0 = 0u, a1 = 0u, a2 = 0u, a3 = 0u;
    uint32_t b0 = 0u, b1 = 0u;
    uint32_t c0 = 0u, c1 = 0u;
    uint32_t d0, d1;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++)
    {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
            "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
            : "=r"(d0), "=r"(d1)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1), "r"(c0), "r"(c1));
        c0 = d0; c1 = d1;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) barrier_out[0] = c0 ^ c1;
}

double bench(void (*k)(int*), int blocks, int threads, double flops_per_mma)
{
    int *bo;
    CK(cudaMalloc(&bo, sizeof(int)));
    CK(cudaMemset(bo, 0, sizeof(int)));

    // Warmup
    k<<<blocks, threads>>>(bo);
    CK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    int rounds = 3;
    for (int i = 0; i < rounds; i++) k<<<blocks, threads>>>(bo);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    double total_flops = (double)rounds * (double)blocks * (double)threads / 32.0 * (double)ITERS * flops_per_mma;
    double tflops = total_flops / (ms / 1000.0) / 1e12;
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(bo);
    return tflops;
}

int main()
{
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    int clk = 0; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    int sm = p.multiProcessorCount;
    printf("=== Raw MMA peak microbench ===\n");
    printf("GPU: %s (%d SMs, %d MHz boost)\n\n", p.name, sm, clk / 1000);

    // FP8 m16n8k32 = 8192 FLOPs per MMA
    double fp8_flops = 8192.0;
    // FP16 m16n8k16 = 4096 FLOPs per MMA
    double fp16_flops = 4096.0;

    // Target: saturate every warp scheduler.
    // 4 warp schedulers per SM × 188 SMs = 752 schedulers
    // 1 warp = 32 threads. Use 4 warps per block = 128 threads.
    // Blocks = 4 × SMs to ensure full occupancy.
    int blocks = sm * 4;
    int threads = 128;

    printf("Launch: %d blocks × %d threads = %d warps active\n\n",
           blocks, threads, blocks * threads / 32);

    printf("--- FP8 m16n8k32 ---\n");
    double tf1 = bench(fp8_mma_peak_f16acc, blocks, threads, fp8_flops);
    printf("  .f16.e4m3.e4m3.f16        → %.1f TFLOPS\n", tf1);

    double tf2 = bench(fp8_mma_peak_f32acc, blocks, threads, fp8_flops);
    printf("  .f32.e4m3.e4m3.f32        → %.1f TFLOPS\n", tf2);

    double tf3 = bench(fp8_mma_peak_kindf8f6f4, blocks, threads, fp8_flops);
    printf("  kind::f8f6f4.f16.e4m3.e4m3.f16 → %.1f TFLOPS\n", tf3);

    printf("\n--- FP16 m16n8k16 ---\n");
    double tf4 = bench(fp16_mma_peak_f16acc, blocks, threads, fp16_flops);
    printf("  .f16.f16.f16.f16          → %.1f TFLOPS\n", tf4);

    printf("\n=== Interpretation ===\n");
    printf("FP8 hardware peak ≈ max(%.0f, %.0f, %.0f) = %.0f TFLOPS\n",
           tf1, tf2, tf3, fmax(fmax(tf1, tf2), tf3));
    printf("FP16 hardware peak ≈ %.0f TFLOPS\n", tf4);
    printf("Ratio FP8/FP16 = %.2f×\n", fmax(fmax(tf1, tf2), tf3) / tf4);

    return 0;
}
