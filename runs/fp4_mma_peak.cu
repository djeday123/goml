// fp4_mma_peak.cu — Raw FP4 (e2m1) MMA throughput on Blackwell sm_120a
//
// FP4 uses mma m16n8k64 (k twice as wide as FP8 m16n8k32) →
// 16*8*64*2 = 16384 FLOPs per MMA (2× FP8's 8192).
//
// Expected: ~2× FP8 raw peak (~2200 T) if TC issue rate is constant.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdint.h>

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); }} while(0)

#define ITERS 1000000

// FP4 via kind::mxf4 with block scale.
// Each thread holds 4 uint32 for A (16 fp8 = 32 fp4 → m16k64) and 2 uint32 for B
// (8 fp8 = 16 fp4 → n8k64). FP16 accumulator output as 2 uint32 (4 packed half).
// Block-scale operand is 1 uint32 per matrix (scale vector index + indices).
__global__ void fp4_mma_peak(int *barrier_out)
{
    uint32_t a0 = 0x11111111u, a1 = 0x11111111u, a2 = 0x11111111u, a3 = 0x11111111u;
    uint32_t b0 = 0x11111111u, b1 = 0x11111111u;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    uint32_t scale_a = 0u, scale_b = 0u;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++)
    {
        // mxf4: e2m1 inputs with ue8m0 (FP8 e8m0) scales. scale_vec::2X means
        // one scale value covers 2 mma's worth of K.
        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1), "r"(scale_a), "r"(scale_b));
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int dummy = (int)(c0 + c1 + c2 + c3);
        barrier_out[0] = dummy;
    }
}

// FP4 mxf4nvf4 (NVIDIA's FP4 format with FP8 e4m3 scales) — scale_vec::4X
__global__ void fp4_nvf4_mma_peak(int *barrier_out)
{
    uint32_t a0 = 0x11111111u, a1 = 0x11111111u, a2 = 0x11111111u, a3 = 0x11111111u;
    uint32_t b0 = 0x11111111u, b1 = 0x11111111u;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    uint32_t scale_a = 0u, scale_b = 0u;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++)
    {
        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1), "r"(scale_a), "r"(scale_b));
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int dummy = (int)(c0 + c1 + c2 + c3);
        barrier_out[0] = dummy;
    }
}

double bench(void (*k)(int*), int blocks, int threads, double flops_per_mma)
{
    int *bo;
    CK(cudaMalloc(&bo, sizeof(int)));
    CK(cudaMemset(bo, 0, sizeof(int)));

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
    printf("=== Raw FP4 MMA peak microbench ===\n");
    printf("GPU: %s (%d SMs, %d MHz boost)\n\n", p.name, sm, clk / 1000);

    double fp4_flops = 16384.0;  // m16n8k64 = 16*8*64*2

    int blocks = sm * 4;
    int threads = 128;
    printf("Launch: %d blocks × %d threads = %d warps active\n\n",
           blocks, threads, blocks * threads / 32);

    printf("--- FP4 m16n8k64 ---\n");
    double tf1 = bench(fp4_mma_peak, blocks, threads, fp4_flops);
    printf("  mxf4 (ue8m0 scale)       → %.1f TFLOPS\n", tf1);

    double tf2 = bench(fp4_nvf4_mma_peak, blocks, threads, fp4_flops);
    printf("  mxf4nvf4 (ue4m3 scale)   → %.1f TFLOPS\n", tf2);

    printf("\n=== Comparison with prior measurements ===\n");
    printf("FP4 peak  ≈ %.0f TFLOPS\n", fmax(tf1, tf2));
    printf("FP8 peak  ≈ 1099 TFLOPS  (from fp8_mma_peak)\n");
    printf("FP16 peak ≈ 402 TFLOPS  (from fp8_mma_peak)\n");
    printf("Ratio FP4/FP8  = %.2f×\n", fmax(tf1, tf2) / 1099.0);
    printf("Ratio FP4/FP16 = %.2f×\n", fmax(tf1, tf2) / 402.0);
    return 0;
}
