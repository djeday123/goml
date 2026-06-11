// QMMA microbenchmark for sm_120a (Blackwell consumer)
// Measures L (latency from issue→D ready) and T (issue interval between
// independent MMAs) for:
//   - mma.m16n8k32 .row.col .f16 .e4m3 .e4m3 .f16    (our FP8 path)
//   - mma.m16n8k64 .row.col .f16 .e2m1 .e2m1 .f16    (FP4 candidate)
//
// Output: L, T, required_ILP = L/T
// Required for v118+ design decisions on accumulator depth.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);}}while(0)

#define CHAIN_N      512
#define TPUT_ITERS   256

// =========================================================================
// LATENCY kernels: dependent chain of CHAIN_N MMAs
// Each MMA writes to the same (c0,c1) accumulator → next MMA waits for D
// =========================================================================
__global__ void qmma_latency_fp8(uint32_t *out)
{
    uint32_t a0 = threadIdx.x + 1, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3;
    uint32_t b0 = a0 + 5, b1 = a0 + 7;
    uint32_t c0 = 0, c1 = 0;

    long long t0 = clock64();
#pragma unroll
    for (int i = 0; i < CHAIN_N; i++) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e4m3.e4m3.f16 "
            "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n"
            : "+r"(c0), "+r"(c1)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) {
        out[0] = (uint32_t)(t1 - t0);
        out[1] = c0 ^ c1;
    }
}

__global__ void qmma_latency_fp4(uint32_t *out)
{
    // m16n8k64 e2m1: requires kind::mxf4nvf4.block_scale (sm_120a only supports
    // block-scaled FP4 variants). Accumulator = f32 (4 floats per thread).
    uint32_t a0 = threadIdx.x + 1, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3;
    uint32_t b0 = a0 + 5, b1 = a0 + 7;
    uint32_t sa = 0x3c3c3c3c, sb = 0x3c3c3c3c;  // ue4m3 packed scale=1.0
    float c0 = 0, c1 = 0, c2 = 0, c3 = 0;

    long long t0 = clock64();
#pragma unroll
    for (int i = 0; i < CHAIN_N; i++) {
        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) {
        out[0] = (uint32_t)(t1 - t0);
        out[1] = (uint32_t)(c0 + c1 + c2 + c3);
    }
}

// =========================================================================
// THROUGHPUT kernels: K independent chains
// Templated over K_CHAINS for full unroll → no LDL on dynamic index
// =========================================================================
template<int K_CHAINS>
__global__ void qmma_throughput_fp8(uint32_t *out)
{
    uint32_t a0 = threadIdx.x + 1, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3;
    uint32_t b0 = a0 + 5, b1 = a0 + 7;
    uint32_t c0[K_CHAINS], c1[K_CHAINS];
#pragma unroll
    for (int k = 0; k < K_CHAINS; k++) { c0[k] = k; c1[k] = k + 1; }

    long long t0 = clock64();
#pragma unroll 1
    for (int i = 0; i < TPUT_ITERS; i++) {
#pragma unroll
        for (int k = 0; k < K_CHAINS; k++) {
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e4m3.e4m3.f16 "
                "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n"
                : "+r"(c0[k]), "+r"(c1[k])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    long long t1 = clock64();

    uint32_t x = 0;
#pragma unroll
    for (int k = 0; k < K_CHAINS; k++) x ^= c0[k] ^ c1[k];
    if (threadIdx.x == 0) {
        out[0] = (uint32_t)(t1 - t0);
        out[1] = x;
    }
}

template<int K_CHAINS>
__global__ void qmma_throughput_fp4(uint32_t *out)
{
    uint32_t a0 = threadIdx.x + 1, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3;
    uint32_t b0 = a0 + 5, b1 = a0 + 7;
    uint32_t sa = 0x3c3c3c3c, sb = 0x3c3c3c3c;
    float c0[K_CHAINS], c1[K_CHAINS], c2[K_CHAINS], c3[K_CHAINS];
#pragma unroll
    for (int k = 0; k < K_CHAINS; k++) {
        c0[k] = (float)k; c1[k] = (float)(k+1);
        c2[k] = (float)(k+2); c3[k] = (float)(k+3);
    }

    long long t0 = clock64();
#pragma unroll 1
    for (int i = 0; i < TPUT_ITERS; i++) {
#pragma unroll
        for (int k = 0; k < K_CHAINS; k++) {
            asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\n"
                : "+f"(c0[k]), "+f"(c1[k]), "+f"(c2[k]), "+f"(c3[k])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
        }
    }
    long long t1 = clock64();

    float x = 0;
#pragma unroll
    for (int k = 0; k < K_CHAINS; k++) x += c0[k] + c1[k] + c2[k] + c3[k];
    if (threadIdx.x == 0) {
        out[0] = (uint32_t)(t1 - t0);
        out[1] = (uint32_t)x;
    }
}

// =========================================================================
// Host driver
// =========================================================================
template<typename KernelFn>
double run_min_cycles(KernelFn kernel, uint32_t *d_out, int reps, int N_div)
{
    uint32_t best = UINT32_MAX;
    for (int r = 0; r < reps; r++) {
        kernel<<<1, 32>>>(d_out);
        CK(cudaDeviceSynchronize());
        uint32_t cycles;
        CK(cudaMemcpy(&cycles, d_out, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (cycles < best) best = cycles;
    }
    return (double)best / (double)N_div;
}

int main(int argc, char **argv)
{
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    int clk;
    cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    printf("GPU: %s (%d SMs, %d MHz)\n", p.name, p.multiProcessorCount, clk / 1000);
    printf("CC: %d.%d\n\n", p.major, p.minor);

    uint32_t *d_out;
    CK(cudaMalloc(&d_out, 8));

    const int REPS = 5;

    // =========================================================================
    printf("================================================================\n");
    printf("  FP8 e4m3 m16n8k32  (kind::f8f6f4)\n");
    printf("================================================================\n");

    double L_fp8 = run_min_cycles(qmma_latency_fp8, d_out, REPS, CHAIN_N);
    printf("  Latency  L_fp8 = %6.2f cycles/MMA  (chain of %d, min of %d)\n",
           L_fp8, CHAIN_N, REPS);

    printf("\n  Throughput sweep (K independent chains):\n");
    printf("  %4s %12s %12s\n", "K", "cycles/MMA", "TFLOPS@K");

    double T_fp8 = 0;
    auto bench_tput_fp8 = [&](auto kernel, int K) {
        uint32_t best = UINT32_MAX;
        for (int r = 0; r < REPS; r++) {
            kernel<<<1, 32>>>(d_out);
            CK(cudaDeviceSynchronize());
            uint32_t cy;
            CK(cudaMemcpy(&cy, d_out, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            if (cy < best) best = cy;
        }
        double cyc_per_mma = (double)best / (double)(TPUT_ITERS * K);
        // FP8 m16n8k32: 16*8*32*2 = 8192 FLOPs per MMA
        double tflops_at_clk = (8192.0 / cyc_per_mma) * ((double)clk * 1000.0) * (double)p.multiProcessorCount / 1e12;
        printf("  %4d %12.3f %12.2f\n", K, cyc_per_mma, tflops_at_clk);
        if (cyc_per_mma < T_fp8 || T_fp8 == 0) T_fp8 = cyc_per_mma;
        return cyc_per_mma;
    };

    bench_tput_fp8(qmma_throughput_fp8<1>,  1);
    bench_tput_fp8(qmma_throughput_fp8<2>,  2);
    bench_tput_fp8(qmma_throughput_fp8<3>,  3);
    bench_tput_fp8(qmma_throughput_fp8<4>,  4);
    bench_tput_fp8(qmma_throughput_fp8<6>,  6);
    bench_tput_fp8(qmma_throughput_fp8<8>,  8);
    bench_tput_fp8(qmma_throughput_fp8<12>, 12);
    bench_tput_fp8(qmma_throughput_fp8<16>, 16);

    printf("\n  L_fp8 = %.2f  T_fp8 = %.2f  required_ILP = %.2f\n",
           L_fp8, T_fp8, L_fp8 / T_fp8);

    // =========================================================================
    printf("\n================================================================\n");
    printf("  FP4 e2m1 m16n8k64  (kind::f8f6f4)\n");
    printf("================================================================\n");

    double L_fp4 = run_min_cycles(qmma_latency_fp4, d_out, REPS, CHAIN_N);
    printf("  Latency  L_fp4 = %6.2f cycles/MMA  (chain of %d, min of %d)\n",
           L_fp4, CHAIN_N, REPS);

    printf("\n  Throughput sweep (K independent chains):\n");
    printf("  %4s %12s %12s\n", "K", "cycles/MMA", "TFLOPS@K");

    double T_fp4 = 0;
    auto bench_tput_fp4 = [&](auto kernel, int K) {
        uint32_t best = UINT32_MAX;
        for (int r = 0; r < REPS; r++) {
            kernel<<<1, 32>>>(d_out);
            CK(cudaDeviceSynchronize());
            uint32_t cy;
            CK(cudaMemcpy(&cy, d_out, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            if (cy < best) best = cy;
        }
        double cyc_per_mma = (double)best / (double)(TPUT_ITERS * K);
        // FP4 m16n8k64: 16*8*64*2 = 16384 FLOPs per MMA (2× FP8 per MMA)
        double tflops_at_clk = (16384.0 / cyc_per_mma) * ((double)clk * 1000.0) * (double)p.multiProcessorCount / 1e12;
        printf("  %4d %12.3f %12.2f\n", K, cyc_per_mma, tflops_at_clk);
        if (cyc_per_mma < T_fp4 || T_fp4 == 0) T_fp4 = cyc_per_mma;
        return cyc_per_mma;
    };

    bench_tput_fp4(qmma_throughput_fp4<1>,  1);
    bench_tput_fp4(qmma_throughput_fp4<2>,  2);
    bench_tput_fp4(qmma_throughput_fp4<3>,  3);
    bench_tput_fp4(qmma_throughput_fp4<4>,  4);
    bench_tput_fp4(qmma_throughput_fp4<6>,  6);
    bench_tput_fp4(qmma_throughput_fp4<8>,  8);
    bench_tput_fp4(qmma_throughput_fp4<12>, 12);
    bench_tput_fp4(qmma_throughput_fp4<16>, 16);

    printf("\n  L_fp4 = %.2f  T_fp4 = %.2f  required_ILP = %.2f\n",
           L_fp4, T_fp4, L_fp4 / T_fp4);

    // =========================================================================
    printf("\n================================================================\n");
    printf("  Verdict\n");
    printf("================================================================\n");
    printf("  T_fp4 / T_fp8 = %.3f\n", T_fp4 / T_fp8);
    printf("  If T_fp4 ≈ T_fp8:  FP4 doubles FLOPs at same issue rate → +2× FP4 path opens\n");
    printf("  If T_fp4 ≈ 2×T_fp8: FP4 is consumer-throttled → close direction\n");
    printf("  Required ILP for v118+ accumulator design:\n");
    printf("    FP8: %.1f independent accumulators per warp\n", L_fp8 / T_fp8);
    printf("    FP4: %.1f independent accumulators per warp\n", L_fp4 / T_fp4);

    cudaFree(d_out);
    return 0;
}
