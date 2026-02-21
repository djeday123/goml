// FP8 GEMM v18 - BK=64, XOR swizzle, 4 blocks/SM (32 warps)
// Strategy: sweet spot between occupancy and register budget
//
// v10b: BK=128, 3 blk/SM, 24 warps → 546 TFLOPS
// v17:  BK=64,  5 blk/SM, 40 warps → risky (51 regs max)
// v18:  BK=64,  4 blk/SM, 32 warps → safe  (64 regs max)
//
// SMEM: 128*64*2 = 16384 bytes/block
// 4 blocks: 65536 <= 102400 ✓
// Registers: 65536/(4*256) = 64 max per thread
// Acc=32, A_frag=4, B_frag=2, overhead~12 = 50 → fits with margin
//
// Tradeoff: 2x barriers vs v10b, but 33% more warps (32 vs 24)
// If barrier overhead < occupancy gain → win

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BM 128
#define BN 128
#define BK 64
#define SMEM_STRIDE 64

#define MMA_M 16
#define MMA_N 8
#define MMA_K 32

#define WARPS_M 2
#define WARPS_N 4
#define WM (BM / WARPS_M)
#define WN (BN / WARPS_N)
#define M_TILES (WM / MMA_M)
#define N_TILES (WN / MMA_N)
#define K_STEPS (BK / MMA_K)

#define SMEM_A_SIZE (BM * SMEM_STRIDE)
#define SMEM_B_SIZE (BN * SMEM_STRIDE)
#define SMEM_TOTAL (SMEM_A_SIZE + SMEM_B_SIZE)

// XOR swizzle: col ^ ((row & 7) << 4)
// Maps 8 consecutive rows to different 16-byte banks
__device__ __forceinline__ uint32_t swiz(uint32_t row, uint32_t col)
{
    return row * SMEM_STRIDE + (col ^ ((row & 7) << 4));
}

__global__ void __launch_bounds__(256, 4)
    fp8_gemm_v18_kernel(const uint8_t *__restrict__ A,
                        const uint8_t *__restrict__ B,
                        half *__restrict__ C,
                        int M, int N, int K)
{
    __shared__ uint8_t smem_A[SMEM_A_SIZE];
    __shared__ uint8_t smem_B[SMEM_B_SIZE];

    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    const int wm = warp_m * WM;
    const int wn = warp_n * WN;

    const int group_id = lane_id / 4;
    const int tid = lane_id % 4;

    // Precompute swizzled base addresses for fragments
    // A: rows = wm + mi*16 + {group_id, group_id+8}
    // B: rows = wn + ni*8 + group_id
    uint32_t a_bases[M_TILES][2]; // [mi][top/bot], offset by tid*4 and +16 at load time
    uint32_t b_bases[N_TILES];

#pragma unroll
    for (int mi = 0; mi < M_TILES; mi++)
    {
        int r0 = wm + mi * MMA_M + group_id;
        int r1 = r0 + 8;
        // Store row*STRIDE (before XOR) - we'll add col at load time
        a_bases[mi][0] = r0;
        a_bases[mi][1] = r1;
    }
#pragma unroll
    for (int ni = 0; ni < N_TILES; ni++)
    {
        b_bases[ni] = wn + ni * MMA_N + group_id;
    }

    // Accumulators
    uint32_t acc[M_TILES][N_TILES][2] = {};

    // Load helpers
    const int loads_per_pass = BK / 16; // 4 uint4's per row

    for (int k = 0; k < K; k += BK)
    {
// Load A [BM, BK] → smem_A with XOR swizzle
// 256 threads, 16B each = 4KB/pass, need 8KB → 2 passes
#pragma unroll
        for (int pass = 0; pass < 2; pass++)
        {
            int idx = pass * 256 + threadIdx.x;
            int row = idx / loads_per_pass;
            int col = (idx % loads_per_pass) * 16;
            uint4 val = __ldg(reinterpret_cast<const uint4 *>(&A[(bm + row) * K + k + col]));
            *reinterpret_cast<uint4 *>(&smem_A[swiz(row, col)]) = val;
        }

// Load B [BN, BK]
#pragma unroll
        for (int pass = 0; pass < 2; pass++)
        {
            int idx = pass * 256 + threadIdx.x;
            int row = idx / loads_per_pass;
            int col = (idx % loads_per_pass) * 16;
            uint4 val = __ldg(reinterpret_cast<const uint4 *>(&B[(bn + row) * K + k + col]));
            *reinterpret_cast<uint4 *>(&smem_B[swiz(row, col)]) = val;
        }

        __syncthreads();

// Compute
#pragma unroll
        for (int ki = 0; ki < K_STEPS; ki++)
        {
            uint32_t koff = ki * MMA_K;

#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                uint32_t a_frag[4];
                // SM89 layout: {row_g/k_lo, row_g+8/k_lo, row_g/k_hi, row_g+8/k_hi}
                a_frag[0] = *reinterpret_cast<uint32_t *>(&smem_A[swiz(a_bases[mi][0], koff + tid * 4)]);
                a_frag[1] = *reinterpret_cast<uint32_t *>(&smem_A[swiz(a_bases[mi][1], koff + tid * 4)]);
                a_frag[2] = *reinterpret_cast<uint32_t *>(&smem_A[swiz(a_bases[mi][0], koff + tid * 4 + 16)]);
                a_frag[3] = *reinterpret_cast<uint32_t *>(&smem_A[swiz(a_bases[mi][1], koff + tid * 4 + 16)]);

#pragma unroll
                for (int ni = 0; ni < N_TILES; ni++)
                {
                    uint32_t b_frag[2];
                    b_frag[0] = *reinterpret_cast<uint32_t *>(&smem_B[swiz(b_bases[ni], koff + tid * 4)]);
                    b_frag[1] = *reinterpret_cast<uint32_t *>(&smem_B[swiz(b_bases[ni], koff + tid * 4 + 16)]);

                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
                        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
                        : "+r"(acc[mi][ni][0]), "+r"(acc[mi][ni][1])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                          "r"(b_frag[0]), "r"(b_frag[1]));
                }
            }
        }

        __syncthreads();
    }

// Store C
#pragma unroll
    for (int mi = 0; mi < M_TILES; mi++)
    {
        int row0 = bm + wm + mi * MMA_M + group_id;
        int row1 = row0 + 8;
#pragma unroll
        for (int ni = 0; ni < N_TILES; ni++)
        {
            int col = bn + wn + ni * MMA_N + tid * 2;
            if (row0 < M && col + 1 < N)
                *reinterpret_cast<uint32_t *>(&C[row0 * N + col]) = acc[mi][ni][0];
            if (row1 < M && col + 1 < N)
                *reinterpret_cast<uint32_t *>(&C[row1 * N + col]) = acc[mi][ni][1];
        }
    }
}

// ===================== FP8 helpers =====================
uint8_t f2e4m3(float v)
{
    if (v == 0.0f)
        return 0;
    uint8_t sign = (v < 0) ? 0x80 : 0;
    v = fabsf(v);
    if (v >= 448.0f)
        return sign | 0x7E;
    if (v < 0.001953125f)
        return sign;
    int e;
    float m = frexpf(v, &e);
    e += 6;
    if (e < 1)
    {
        m = v / 0.001953125f;
        return sign | (uint8_t)(m + 0.5f);
    }
    if (e > 15)
        e = 15;
    int mant = (int)((m * 2.0f - 1.0f) * 8.0f + 0.5f);
    if (mant > 7)
    {
        mant = 0;
        e++;
    }
    if (e > 15)
        return sign | 0x7E;
    return sign | (uint8_t)((e << 3) | mant);
}

float e4m3_to_f(uint8_t v)
{
    uint8_t sign = v >> 7, exp = (v >> 3) & 0xF, mant = v & 0x7;
    float r;
    if (exp == 0)
        r = mant * 0.001953125f * 0.125f;
    else if (exp == 15 && mant == 7)
        return sign ? -INFINITY : INFINITY;
    else
        r = (1.0f + mant / 8.0f) * powf(2.0f, (float)exp - 7.0f);
    return sign ? -r : r;
}

// ===================== Benchmark =====================
void run_benchmark(int M, int N, int K, float *best)
{
    size_t sA = (size_t)M * K, sB = (size_t)N * K, sC = (size_t)M * N * sizeof(half);
    uint8_t *dA, *dB;
    half *dC;
    cudaMalloc(&dA, sA);
    cudaMalloc(&dB, sB);
    cudaMalloc(&dC, sC);
    cudaMemset(dA, 0x3C, sA);
    cudaMemset(dB, 0x3C, sB);

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM), block(256);

    for (int i = 0; i < 5; i++)
        fp8_gemm_v18_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    const int iters = 20;
    cudaEventRecord(t0);
    for (int i = 0; i < iters; i++)
        fp8_gemm_v18_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= iters;
    double tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
    printf("  %5d x %5d x %5d  %7.3f ms  %7.1f TFLOPS%s\n",
           M, N, K, ms, tflops, tflops > *best ? "  *best*" : "");
    if (tflops > *best)
        *best = tflops;

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

void run_correctness()
{
    printf("--- Correctness ---\n");
    int sizes[] = {128, 256, 512};
    for (int si = 0; si < 3; si++)
    {
        int N = sizes[si];
        uint8_t *hA = (uint8_t *)calloc(N * N, 1), *hB = (uint8_t *)calloc(N * N, 1);
        for (int i = 0; i < N; i++)
            hA[i * N + i] = f2e4m3(1.0f);
        for (int i = 0; i < N * N; i++)
            hB[i] = f2e4m3((float)((i * 7 + 3) % 15) - 7.0f);

        uint8_t *dA, *dB;
        half *dC;
        cudaMalloc(&dA, N * N);
        cudaMalloc(&dB, N * N);
        cudaMalloc(&dC, N * N * sizeof(half));
        cudaMemcpy(dA, hA, N * N, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, N * N, cudaMemcpyHostToDevice);
        dim3 grid((N + BN - 1) / BN, (N + BM - 1) / BM);
        fp8_gemm_v18_kernel<<<grid, 256>>>(dA, dB, dC, N, N, N);
        half *hC = (half *)malloc(N * N * sizeof(half));
        cudaMemcpy(hC, dC, N * N * sizeof(half), cudaMemcpyDeviceToHost);

        int errs = 0;
        for (int i = 0; i < N && errs < 5; i++)
            for (int j = 0; j < N; j++)
            {
                float g = __half2float(hC[i * N + j]), r = e4m3_to_f(hB[j * N + i]);
                if (fabsf(g - r) > 0.5f)
                    errs++;
            }
        printf("  Identity %d: %s\n", N, errs == 0 ? "PASS" : "FAIL");

        for (int i = 0; i < N * N; i++)
        {
            hA[i] = f2e4m3(((float)((i * 13 + 5) % 15) - 7.0f) * 0.5f);
            hB[i] = f2e4m3(((float)((i * 11 + 7) % 15) - 7.0f) * 0.5f);
        }
        cudaMemcpy(dA, hA, N * N, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, N * N, cudaMemcpyHostToDevice);
        fp8_gemm_v18_kernel<<<grid, 256>>>(dA, dB, dC, N, N, N);
        cudaMemcpy(hC, dC, N * N * sizeof(half), cudaMemcpyDeviceToHost);

        errs = 0;
        float maxe = 0;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                float g = __half2float(hC[i * N + j]), r = 0;
                for (int kk = 0; kk < N; kk++)
                    r += e4m3_to_f(hA[i * N + kk]) * e4m3_to_f(hB[j * N + kk]);
                float e = fabsf(g - (float)__float2half(r));
                if (e > maxe)
                    maxe = e;
                if (e > 1.0f)
                    errs++;
            }
        printf("  Random  %d: max_err=%.2f %s\n", N, maxe, errs == 0 ? "PASS" : "FAIL");

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        free(hA);
        free(hB);
        free(hC);
    }
}

int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int numBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, fp8_gemm_v18_kernel, 256, SMEM_TOTAL);

    printf("=== FP8 GEMM v18 (BK=64, XOR swizzle, 4 blocks/SM) ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n", prop.name, prop.multiProcessorCount, prop.clockRate / 1000);
    printf("SMEM: %d B/block, achieved blocks/SM: %d (target: 4)\n", SMEM_TOTAL, numBlocks);

    run_correctness();

    printf("--- Performance ---\n");
    float best = 0;
    run_benchmark(1024, 1024, 1024, &best);
    run_benchmark(2048, 2048, 2048, &best);
    run_benchmark(4096, 4096, 4096, &best);
    run_benchmark(8192, 8192, 8192, &best);

    printf("--- LLM shapes ---\n");
    run_benchmark(2048, 4096, 4096, &best);
    run_benchmark(2048, 11008, 4096, &best);
    run_benchmark(2048, 4096, 11008, &best);
    run_benchmark(4096, 4096, 4096, &best);
    run_benchmark(4096, 11008, 4096, &best);
    run_benchmark(8192, 4096, 4096, &best);
    run_benchmark(8192, 11008, 4096, &best);

    printf("\nPeak:660 | cuBLAS:~330 | v10b:546 | v18: %.0f TFLOPS\n", best);
    return 0;
}
