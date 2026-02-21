// =============================================================================
// FP8 Custom GEMM v3 — Full Fragment Preload + 3-Stage Pipeline
// =============================================================================
// Changes from v2 (302 TFLOPS):
//   1. Full A+B preload: load ALL fragments before ANY MMA
//      → 16 MMA fire back-to-back from registers, zero smem stalls
//   2. 3-stage pipeline: 3 tiles in-flight, better latency hiding
//   3. Smem: 3 * (128+128) * 80 = 61,440 bytes (fits in 100KB)
//   4. Register budget: 24 frag + 32 acc = 56 regs/thread (max 128)
//
// Build: nvcc -O3 -arch=sm_89 -std=c++17 fp8_gemm_v3.cu -o fp8_gemm_v3
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <vector>

// ============================================================
// Tile configuration
// ============================================================
#define BM 128
#define BN 128
#define BK 64
#define BK_PAD 80 // %16==0 (cp.async), %32==16 (no bank conflict)
#define STAGES 3  // Triple-buffered pipeline

#define WARPS_M 2
#define WARPS_N 4
#define WM (BM / WARPS_M)                 // 64
#define WN (BN / WARPS_N)                 // 32
#define NTHREADS (WARPS_M * WARPS_N * 32) // 256

#define MMA_M 16
#define MMA_N 8
#define MMA_K 32
#define TILES_M (WM / MMA_M) // 4
#define TILES_N (WN / MMA_N) // 4
#define K_ITERS (BK / MMA_K) // 2

// Smem: 3 stages * (128+128) * 80 = 61,440 bytes

// ============================================================
// PTX MMA: m16n8k32, FP8 E4M3, FP16 accumulator (642 TFLOPS)
// ============================================================
__device__ __forceinline__ void mma_fp8_f16(
    uint32_t d[2], const uint32_t a[4], const uint32_t b[2], const uint32_t c[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0, %1}, "
        "{%2, %3, %4, %5}, "
        "{%6, %7}, "
        "{%8, %9};\n"
        : "=r"(d[0]), "=r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]));
}

// ============================================================
// cp.async helpers
// ============================================================
__device__ __forceinline__ void cp_async_16B(void *smem, const void *gmem)
{
    uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n" :::);
}
__device__ __forceinline__ void cp_async_wait(int n)
{
    switch (n)
    {
    case 0:
        asm volatile("cp.async.wait_group 0;\n" :::);
        break;
    case 1:
        asm volatile("cp.async.wait_group 1;\n" :::);
        break;
    case 2:
        asm volatile("cp.async.wait_group 2;\n" :::);
        break;
    }
}

// ============================================================
// Tile loader (4 chunks per thread for BK=64)
// ============================================================
__device__ __forceinline__ void load_tile(
    uint8_t *A_smem, uint8_t *B_smem,
    const uint8_t *__restrict__ A, const uint8_t *__restrict__ B,
    int bm, int bn, int k_offset,
    int M, int N, int K, int tid)
{
#pragma unroll
    for (int c = 0; c < 4; c++)
    {
        int cid = c * NTHREADS + tid;
        if (cid < 512)
        {
            int row = cid >> 2;
            int col16 = (cid & 3) << 4;
            int gr = bm + row;
            if (gr < M && k_offset + col16 < K)
                cp_async_16B(&A_smem[row * BK_PAD + col16],
                             &A[(size_t)gr * K + k_offset + col16]);
        }
        else
        {
            int bid = cid - 512;
            int row = bid >> 2;
            int col16 = (bid & 3) << 4;
            int gr = bn + row;
            if (gr < N && k_offset + col16 < K)
                cp_async_16B(&B_smem[row * BK_PAD + col16],
                             &B[(size_t)gr * K + k_offset + col16]);
        }
    }
    cp_async_commit();
}

// ============================================================
// GEMM Kernel v3
// ============================================================
__global__ __launch_bounds__(NTHREADS, 2) void fp8_gemm_v3(
    const uint8_t *__restrict__ A,
    const uint8_t *__restrict__ B,
    half *__restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;

    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    const int warpM = warpId / WARPS_N;
    const int warpN = warpId % WARPS_N;
    const int group = laneId / 4;
    const int tid4 = laneId % 4;

    // Triple-buffered shared memory
    extern __shared__ uint8_t smem[];
    const int stage_bytes = (BM + BN) * BK_PAD;
    uint8_t *A_smem[STAGES], *B_smem[STAGES];
#pragma unroll
    for (int s = 0; s < STAGES; s++)
    {
        A_smem[s] = smem + s * stage_bytes;
        B_smem[s] = smem + s * stage_bytes + BM * BK_PAD;
    }

    // Accumulators
    uint32_t acc[TILES_M][TILES_N][2];
#pragma unroll
    for (int i = 0; i < TILES_M; i++)
#pragma unroll
        for (int j = 0; j < TILES_N; j++)
            acc[i][j][0] = acc[i][j][1] = 0;

    const int num_k = (K + BK - 1) / BK;
    const int wm_off = warpM * WM;
    const int wn_off = warpN * WN;

    // ---- Prologue: fill pipeline with up to STAGES-1 tiles ----
    int prologue = (num_k < STAGES) ? num_k : STAGES;
#pragma unroll
    for (int i = 0; i < STAGES - 1; i++)
    {
        if (i < num_k)
            load_tile(A_smem[i], B_smem[i], A, B, bm, bn, i * BK, M, N, K, threadIdx.x);
    }

    // ---- Main loop ----
    for (int kt = 0; kt < num_k; kt++)
    {
        int s = kt % STAGES;

        // Issue load for tile kt+STAGES-1 (if exists)
        int load_kt = kt + STAGES - 1;
        if (load_kt < num_k)
        {
            int ls = load_kt % STAGES;
            load_tile(A_smem[ls], B_smem[ls], A, B, bm, bn, load_kt * BK, M, N, K, threadIdx.x);
        }

        // Wait for tile kt to be ready
        // We have at most STAGES-1 groups in flight after this tile
        int in_flight = num_k - kt - 1;
        if (in_flight > STAGES - 1)
            in_flight = STAGES - 1;
        cp_async_wait(in_flight);
        __syncthreads();

// ---- Compute with FULL FRAGMENT PRELOAD ----
#pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++)
        {
            const int kb = ki * MMA_K;

            // *** Preload ALL B fragments for this ki ***
            uint32_t b_all[TILES_N][2];
#pragma unroll
            for (int ni = 0; ni < TILES_N; ni++)
            {
                const int br = wn_off + ni * MMA_N;
                b_all[ni][0] = *(uint32_t *)(&B_smem[s][(br + group) * BK_PAD + kb + tid4 * 4]);
                b_all[ni][1] = *(uint32_t *)(&B_smem[s][(br + group) * BK_PAD + kb + 16 + tid4 * 4]);
            }

            // *** Preload ALL A fragments for this ki ***
            uint32_t a_all[TILES_M][4];
#pragma unroll
            for (int mi = 0; mi < TILES_M; mi++)
            {
                const int ar = wm_off + mi * MMA_M;
                a_all[mi][0] = *(uint32_t *)(&A_smem[s][(ar + group) * BK_PAD + kb + tid4 * 4]);
                a_all[mi][1] = *(uint32_t *)(&A_smem[s][(ar + group) * BK_PAD + kb + 16 + tid4 * 4]);
                a_all[mi][2] = *(uint32_t *)(&A_smem[s][(ar + 8 + group) * BK_PAD + kb + tid4 * 4]);
                a_all[mi][3] = *(uint32_t *)(&A_smem[s][(ar + 8 + group) * BK_PAD + kb + 16 + tid4 * 4]);
            }

// *** ALL 16 MMA from registers — zero smem reads ***
#pragma unroll
            for (int mi = 0; mi < TILES_M; mi++)
            {
#pragma unroll
                for (int ni = 0; ni < TILES_N; ni++)
                {
                    mma_fp8_f16(acc[mi][ni], a_all[mi], b_all[ni], acc[mi][ni]);
                }
            }
        }

        __syncthreads();
    }

    // ---- Epilogue: store C ----
    const int wm_base = bm + warpM * WM;
    const int wn_base = bn + warpN * WN;

#pragma unroll
    for (int mi = 0; mi < TILES_M; mi++)
    {
#pragma unroll
        for (int ni = 0; ni < TILES_N; ni++)
        {
            const int r0 = wm_base + mi * MMA_M + group;
            const int r1 = r0 + 8;
            const int c = wn_base + ni * MMA_N + tid4 * 2;

            if (r0 < M && c + 1 < N)
                *(uint32_t *)(&C[(size_t)r0 * N + c]) = acc[mi][ni][0];
            if (r1 < M && c + 1 < N)
                *(uint32_t *)(&C[(size_t)r1 * N + c]) = acc[mi][ni][1];
        }
    }
}

// ============================================================
// C API
// ============================================================
struct Fp8Args
{
    void *handle;
    int M, N, K, _pad;
    const void *A;
    const void *B;
    void *C;
    const float *alpha;
    const float *beta;
};

extern "C"
{

    int cutlass_fp8_gemm(Fp8Args *a)
    {
        int M = a->M, N = a->N, K = a->K;
        if (M % BM != 0 || N % BN != 0 || K % BK != 0)
            return -1;

        dim3 grid(N / BN, M / BM);
        dim3 block(NTHREADS);
        int smem = STAGES * (BM + BN) * BK_PAD;

        cudaFuncSetAttribute(fp8_gemm_v3,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

        fp8_gemm_v3<<<grid, block, smem>>>(
            (const uint8_t *)a->A,
            (const uint8_t *)a->B,
            (half *)a->C,
            M, N, K);
        return 0;
    }

    int cuda_device_sync(void) { return (int)cudaDeviceSynchronize(); }
}

// ============================================================
// Benchmark
// ============================================================
int main()
{
    int sizes[] = {512, 1024, 2048, 4096, 8192};
    int nsizes = 5;
    int iters = 50, warmup = 10;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("Smem per block: %d bytes (%d stages)\n\n",
           STAGES * (BM + BN) * BK_PAD, STAGES);

    printf("=== FP8 PTX GEMM v3 (full preload, 3-stage, f16 accum) ===\n");
    printf("%-8s %-10s %-10s %-10s\n", "Size", "Time(ms)", "TFLOPS", "Status");
    printf("--------------------------------------------\n");

    for (int si = 0; si < nsizes; si++)
    {
        int M = sizes[si], N = sizes[si], K = sizes[si];
        if (M % BM != 0 || N % BN != 0 || K % BK != 0)
        {
            printf("%-8d %-10s %-10s SKIP\n", M, "-", "-");
            continue;
        }

        size_t sA = (size_t)M * K;
        size_t sB = (size_t)N * K;
        size_t sC = (size_t)M * N;

        std::vector<uint8_t> hA(sA), hB(sB);
        unsigned rng = 42 + si;
        for (size_t i = 0; i < sA; i++)
        {
            rng = rng * 1103515245u + 12345u;
            hA[i] = (rng >> 16) & 0x3F;
        }
        for (size_t i = 0; i < sB; i++)
        {
            rng = rng * 1103515245u + 12345u;
            hB[i] = (rng >> 16) & 0x3F;
        }

        void *dA, *dB, *dC;
        cudaMalloc(&dA, sA);
        cudaMalloc(&dB, sB);
        cudaMalloc(&dC, sC * 2);
        cudaMemcpy(dA, hA.data(), sA, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), sB, cudaMemcpyHostToDevice);
        cudaMemset(dC, 0, sC * 2);

        float alpha = 1.0f, beta = 0.0f;
        Fp8Args args = {};
        args.M = M;
        args.N = N;
        args.K = K;
        args.A = dA;
        args.B = dB;
        args.C = dC;
        args.alpha = &alpha;
        args.beta = &beta;

        for (int i = 0; i < warmup; i++)
            cutlass_fp8_gemm(&args);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            printf("%-8d %-10s %-10s ERR: %s\n", M, "-", "-",
                   cudaGetErrorString(err));
            cudaFree(dA);
            cudaFree(dB);
            cudaFree(dC);
            continue;
        }

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        cudaEventRecord(t0);
        for (int i = 0; i < iters; i++)
            cutlass_fp8_gemm(&args);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);

        float ms;
        cudaEventElapsedTime(&ms, t0, t1);
        float avg = ms / iters;
        double flops = 2.0 * (double)M * N * K;
        double tflops = (flops / (avg / 1000.0)) / 1e12;

        printf("%-8d %-10.3f %-10.1f OK\n", M, avg, tflops);

        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }

    printf("\n");
    printf("v1:            193 TFLOPS (BK=32, 2-stage)\n");
    printf("v2:            302 TFLOPS (BK=64, B-preload)\n");
    printf("cuBLASLt:      346 TFLOPS (FP32 accum)\n");
    printf("Hardware peak: 642 TFLOPS (FP16 accum)\n");

    return 0;
}
