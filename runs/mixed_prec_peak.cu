#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdint.h>

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) { exit(1); }} while(0)
#define ITERS 1000000

// Mixed precision: A=e2m1 (FP4), B=e4m3 (FP8), accum f32
__global__ void mixed_fp4_fp8(int *bo) {
    uint32_t a0=0x11111111u, a1=0x11111111u, a2=0x11111111u, a3=0x11111111u;
    uint32_t b0=0x11111111u, b1=0x11111111u;
    float c0=0, c1=0, c2=0, c3=0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e4m3.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) bo[0] = (int)(c0+c1+c2+c3);
}

// Mixed: A=e4m3, B=e2m1
__global__ void mixed_fp8_fp4(int *bo) {
    uint32_t a0=0x11111111u, a1=0x11111111u, a2=0x11111111u, a3=0x11111111u;
    uint32_t b0=0x11111111u, b1=0x11111111u;
    float c0=0, c1=0, c2=0, c3=0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e2m1.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) bo[0] = (int)(c0+c1+c2+c3);
}

// Mixed FP6 × FP8
__global__ void mixed_fp6_fp8(int *bo) {
    uint32_t a0=0x11111111u, a1=0x11111111u, a2=0x11111111u, a3=0x11111111u;
    uint32_t b0=0x11111111u, b1=0x11111111u;
    float c0=0, c1=0, c2=0, c3=0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e3m2.e4m3.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) bo[0] = (int)(c0+c1+c2+c3);
}

double bench(void (*k)(int*), int blocks, int threads, double flops) {
    int *bo; CK(cudaMalloc(&bo, sizeof(int)));
    CK(cudaMemset(bo, 0, sizeof(int)));
    k<<<blocks, threads>>>(bo);
    cudaDeviceSynchronize();
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 3; i++) k<<<blocks, threads>>>(bo);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    double tf = 3.0 * blocks * threads / 32.0 * (double)ITERS * flops / (ms / 1000.0) / 1e12;
    cudaFree(bo);
    return tf;
}

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    int sm = p.multiProcessorCount;
    printf("GPU: %s (%d SMs)\n\n", p.name, sm);
    int blocks = sm * 4, threads = 128;

    printf("Mixed precision benchmarks (kind::f8f6f4 m16n8k32):\n");
    printf("  e2m1 × e4m3 (FP4 × FP8):  %.1f TFLOPS\n", bench(mixed_fp4_fp8, blocks, threads, 8192.0));
    printf("  e4m3 × e2m1 (FP8 × FP4):  %.1f TFLOPS\n", bench(mixed_fp8_fp4, blocks, threads, 8192.0));
    printf("  e3m2 × e4m3 (FP6 × FP8):  %.1f TFLOPS\n", bench(mixed_fp6_fp8, blocks, threads, 8192.0));
    printf("\nReference dense at same m16n8k32 shape:\n");
    printf("  e4m3 × e4m3 (FP8):  1099 TFLOPS\n");
    printf("  e2m1 × e2m1 (FP4 in f8f6f4): 1098 TFLOPS\n");
    return 0;
}
