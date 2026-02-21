// =============================================================================
// FP8 Custom GEMM v4 — More MMA per tile, fewer registers
// =============================================================================
// Diagnostic from v2/v3:
//   116 registers → likely spills, bad scheduling
//   MMA/smem_load = 0.67 → smem loads bottleneck compute
//   3-stage killed occupancy (1 block/SM vs 2)
//
// V4 changes:
//   1. BK=128: K_ITERS=4, so 64 MMA per tile (vs 32). More compute per sync.
//   2. -maxrregcount=80: force compiler to avoid spills, better ILP
//   3. 2-stage pipeline (confirmed better occupancy: 2 blocks/SM)
//   4. A-interleaved: load A[mi] + fire 4 MMA immediately (saves 12 regs)
//   5. BK_PAD=144: 128+16, %16==0 (cp.async), %32==16 (no bank conflict)
//
// Build: nvcc -O3 -arch=sm_89 -std=c++17 -maxrregcount=80 fp8_gemm_v4.cu -o fp8_gemm_v4
// Also try without maxrregcount to compare:
//        nvcc -O3 -arch=sm_89 -std=c++17 fp8_gemm_v4.cu -o fp8_gemm_v4b
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
#define BK 128     // v2 was 64
#define BK_PAD 144 // 128+16: %16==0, %32==16

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
#define K_ITERS (BK / MMA_K) // 4 (was 2 in v2)

// Smem: 2 * (128+128) * 144 = 73,728 bytes (fits in 100KB)
// With maxrregcount=80: 256 * 80 = 20,480 regs/block → 3 blocks possible by regs
// But smem: 73728 * 2 = 147,456 > 101,376 → only 1 block/SM. Hmm.
// Let's check: 73,728 ≤ 101,376 → 1 block fits. 73,728 * 2 > 101,376 → only 1 block.
// PROBLEM: BK=128 with BK_PAD=144 needs too much smem for 2 blocks.
//
// SOLUTION: Use BK_PAD=128+16=144 but BK=128 needs (128+128)*144*2 = 73728.
// 101376 / 73728 = 1.37 → only 1 block/SM. BAD.
//
// Alternative: Keep BK=64, use BK_PAD=80, 2 blocks/SM, but try maxrregcount.
// Let's do BOTH configurations and compare.

// ============================================================
// Config A: BK=64, BK_PAD=80, 2 blocks/SM (known good occupancy)
// Config B: BK=128, BK_PAD=144, 1 block/SM (more MMA per tile)
// ============================================================

// We'll use BK=64 as primary (better occupancy), but with register limit

#undef BK
#undef BK_PAD
#undef K_ITERS
#define BK 64
#define BK_PAD 80
#define K_ITERS (BK / MMA_K) // 2

// ============================================================
// PTX MMA
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
// cp.async
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
__device__ __forceinline__ void cp_async_wait0()
{
    asm volatile("cp.async.wait_group 0;\n" :::);
}
__device__ __forceinline__ void cp_async_wait1()
{
    asm volatile("cp.async.wait_group 1;\n" :::);
}

// ============================================================
// GEMM Kernel v4
//   Key: A-interleaved compute — load A[mi], fire 4 MMA, load next A[mi+1]
//   This uses only 4 A regs at a time (vs 16 with full preload)
//   Total fragment regs: 4(A) + 8(B) + 32(acc) = 44 (vs 56 in v3)
// ============================================================
__global__ __launch_bounds__(NTHREADS, 2) void fp8_gemm_v4(
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

    extern __shared__ uint8_t smem[];
    const int stage_bytes = (BM + BN) * BK_PAD;
    uint8_t *A_smem[2], *B_smem[2];
    A_smem[0] = smem;
    B_smem[0] = smem + BM * BK_PAD;
    A_smem[1] = smem + stage_bytes;
    B_smem[1] = smem + stage_bytes + BM * BK_PAD;

    uint32_t acc[TILES_M][TILES_N][2];
#pragma unroll
    for (int i = 0; i < TILES_M; i++)
#pragma unroll
        for (int j = 0; j < TILES_N; j++)
            acc[i][j][0] = acc[i][j][1] = 0;

    const int num_k = (K + BK - 1) / BK;
    const int wm_off = warpM * WM;
    const int wn_off = warpN * WN;

    // ---- Prologue: load tile 0 ----
    {
#pragma unroll
        for (int c = 0; c < 4; c++)
        {
            int cid = c * NTHREADS + threadIdx.x;
            if (cid < 512)
            {
                int row = cid >> 2;
                int col16 = (cid & 3) << 4;
                int gr = bm + row;
                if (gr < M && col16 < K)
                    cp_async_16B(&A_smem[0][row * BK_PAD + col16],
                                 &A[(size_t)gr * K + col16]);
            }
            else
            {
                int bid = cid - 512;
                int row = bid >> 2;
                int col16 = (bid & 3) << 4;
                int gr = bn + row;
                if (gr < N && col16 < K)
                    cp_async_16B(&B_smem[0][row * BK_PAD + col16],
                                 &B[(size_t)gr * K + col16]);
            }
        }
        cp_async_commit();
    }

    // ---- Main loop ----
    for (int kt = 0; kt < num_k; kt++)
    {
        // Load next tile (overlaps with compute)
        if (kt + 1 < num_k)
        {
            int k_off = (kt + 1) * BK;
            int ns = (kt + 1) & 1;
#pragma unroll
            for (int c = 0; c < 4; c++)
            {
                int cid = c * NTHREADS + threadIdx.x;
                if (cid < 512)
                {
                    int row = cid >> 2;
                    int col16 = (cid & 3) << 4;
                    int gr = bm + row;
                    if (gr < M && k_off + col16 < K)
                        cp_async_16B(&A_smem[ns][row * BK_PAD + col16],
                                     &A[(size_t)gr * K + k_off + col16]);
                }
                else
                {
                    int bid = cid - 512;
                    int row = bid >> 2;
                    int col16 = (bid & 3) << 4;
                    int gr = bn + row;
                    if (gr < N && k_off + col16 < K)
                        cp_async_16B(&B_smem[ns][row * BK_PAD + col16],
                                     &B[(size_t)gr * K + k_off + col16]);
                }
            }
            cp_async_commit();
        }

        if (kt + 1 < num_k)
            cp_async_wait1();
        else
            cp_async_wait0();
        __syncthreads();

        const int s = kt & 1;

// ---- Compute: A-interleaved, B-preloaded ----
#pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++)
        {
            const int kb = ki * MMA_K;

            // Preload ALL B fragments for this ki (8 regs, reused 4x)
            uint32_t b_frag[TILES_N][2];
#pragma unroll
            for (int ni = 0; ni < TILES_N; ni++)
            {
                const int br = wn_off + ni * MMA_N;
                b_frag[ni][0] = *(uint32_t *)(&B_smem[s][(br + group) * BK_PAD + kb + tid4 * 4]);
                b_frag[ni][1] = *(uint32_t *)(&B_smem[s][(br + group) * BK_PAD + kb + 16 + tid4 * 4]);
            }

// For each mi: load A (4 regs) + fire 4 MMA immediately
// Compiler can overlap A[mi+1] load with MMA[mi]
#pragma unroll
            for (int mi = 0; mi < TILES_M; mi++)
            {
                const int ar = wm_off + mi * MMA_M;
                uint32_t a_frag[4];

                a_frag[0] = *(uint32_t *)(&A_smem[s][(ar + group) * BK_PAD + kb + tid4 * 4]);
                a_frag[1] = *(uint32_t *)(&A_smem[s][(ar + group) * BK_PAD + kb + 16 + tid4 * 4]);
                a_frag[2] = *(uint32_t *)(&A_smem[s][(ar + 8 + group) * BK_PAD + kb + tid4 * 4]);
                a_frag[3] = *(uint32_t *)(&A_smem[s][(ar + 8 + group) * BK_PAD + kb + 16 + tid4 * 4]);

// 4 MMA with preloaded B — compiler can interleave with next A load
#pragma unroll
                for (int ni = 0; ni < TILES_N; ni++)
                {
                    mma_fp8_f16(acc[mi][ni], a_frag, b_frag[ni], acc[mi][ni]);
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
// Also BK=128 variant for comparison
// ============================================================
#define BK2 128
#define BK_PAD2 144            // 128+16
#define K_ITERS2 (BK2 / MMA_K) // 4

__global__ __launch_bounds__(NTHREADS, 1) // only 1 block/SM (smem limit)
    void fp8_gemm_v4_bk128(
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

    extern __shared__ uint8_t smem[];
    const int stage_bytes = (BM + BN) * BK_PAD2;
    uint8_t *A_smem[2], *B_smem[2];
    A_smem[0] = smem;
    B_smem[0] = smem + BM * BK_PAD2;
    A_smem[1] = smem + stage_bytes;
    B_smem[1] = smem + stage_bytes + BM * BK_PAD2;

    uint32_t acc[TILES_M][TILES_N][2];
#pragma unroll
    for (int i = 0; i < TILES_M; i++)
#pragma unroll
        for (int j = 0; j < TILES_N; j++)
            acc[i][j][0] = acc[i][j][1] = 0;

    const int num_k = (K + BK2 - 1) / BK2;
    const int wm_off = warpM * WM;
    const int wn_off = warpN * WN;

    // Loading BK=128: 8 chunks of 16B per row (128/16=8)
    // A: 128 rows × 8 chunks = 1024 chunks
    // B: 128 rows × 8 chunks = 1024 chunks
    // Total: 2048 chunks / 256 threads = 8 chunks/thread
    const int chunks_A = BM * (BK2 / 16);            // 1024
    const int chunks_total = (BM + BN) * (BK2 / 16); // 2048
    const int cpt = chunks_total / NTHREADS;         // 8

    auto do_load = [&](int stage, int k_off)
    {
#pragma unroll
        for (int c = 0; c < cpt; c++)
        {
            int cid = c * NTHREADS + threadIdx.x;
            if (cid < chunks_A)
            {
                int row = cid / (BK2 / 16);
                int col16 = (cid % (BK2 / 16)) * 16;
                int gr = bm + row;
                if (gr < M && k_off + col16 < K)
                    cp_async_16B(&A_smem[stage][row * BK_PAD2 + col16],
                                 &A[(size_t)gr * K + k_off + col16]);
            }
            else
            {
                int bid = cid - chunks_A;
                int row = bid / (BK2 / 16);
                int col16 = (bid % (BK2 / 16)) * 16;
                int gr = bn + row;
                if (gr < N && k_off + col16 < K)
                    cp_async_16B(&B_smem[stage][row * BK_PAD2 + col16],
                                 &B[(size_t)gr * K + k_off + col16]);
            }
        }
        cp_async_commit();
    };

    do_load(0, 0);

    for (int kt = 0; kt < num_k; kt++)
    {
        if (kt + 1 < num_k)
            do_load((kt + 1) & 1, (kt + 1) * BK2);

        if (kt + 1 < num_k)
            cp_async_wait1();
        else
            cp_async_wait0();
        __syncthreads();

        const int s = kt & 1;

#pragma unroll
        for (int ki = 0; ki < K_ITERS2; ki++)
        {
            const int kb = ki * MMA_K;

            uint32_t b_frag[TILES_N][2];
#pragma unroll
            for (int ni = 0; ni < TILES_N; ni++)
            {
                const int br = wn_off + ni * MMA_N;
                b_frag[ni][0] = *(uint32_t *)(&B_smem[s][(br + group) * BK_PAD2 + kb + tid4 * 4]);
                b_frag[ni][1] = *(uint32_t *)(&B_smem[s][(br + group) * BK_PAD2 + kb + 16 + tid4 * 4]);
            }

#pragma unroll
            for (int mi = 0; mi < TILES_M; mi++)
            {
                const int ar = wm_off + mi * MMA_M;
                uint32_t a_frag[4];
                a_frag[0] = *(uint32_t *)(&A_smem[s][(ar + group) * BK_PAD2 + kb + tid4 * 4]);
                a_frag[1] = *(uint32_t *)(&A_smem[s][(ar + group) * BK_PAD2 + kb + 16 + tid4 * 4]);
                a_frag[2] = *(uint32_t *)(&A_smem[s][(ar + 8 + group) * BK_PAD2 + kb + tid4 * 4]);
                a_frag[3] = *(uint32_t *)(&A_smem[s][(ar + 8 + group) * BK_PAD2 + kb + 16 + tid4 * 4]);

#pragma unroll
                for (int ni = 0; ni < TILES_N; ni++)
                {
                    mma_fp8_f16(acc[mi][ni], a_frag, b_frag[ni], acc[mi][ni]);
                }
            }
        }

        __syncthreads();
    }

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
struct Fp8Args
{
    void *handle;
    int M, N, K, _pad;
    const void *A, *B;
    void *C;
    const float *alpha, *beta;
};

extern "C"
{
    int cutlass_fp8_gemm(Fp8Args *a)
    {
        int M = a->M, N = a->N, K = a->K;
        if (M % BM != 0 || N % BN != 0 || K % BK != 0)
            return -1;
        dim3 grid(N / BN, M / BM), block(NTHREADS);
        int smem = 2 * (BM + BN) * BK_PAD;
        cudaFuncSetAttribute(fp8_gemm_v4, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        fp8_gemm_v4<<<grid, block, smem>>>((const uint8_t *)a->A, (const uint8_t *)a->B, (half *)a->C, M, N, K);
        return 0;
    }
    int cuda_device_sync(void) { return (int)cudaDeviceSynchronize(); }
}

// ============================================================
// Benchmark: compare BK=64 vs BK=128, with and without regcount limit
// ============================================================
static double bench_kernel(auto kernel_fn, int M, int N, int K,
                           void *dA, void *dB, void *dC, int smem_bytes)
{
    cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    dim3 grid(N / BN, M / BM), block(NTHREADS);

    // Warmup
    for (int i = 0; i < 10; i++)
        kernel_fn<<<grid, block, smem_bytes>>>((const uint8_t *)dA, (const uint8_t *)dB, (half *)dC, M, N, K);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        return -1.0;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 50; i++)
        kernel_fn<<<grid, block, smem_bytes>>>((const uint8_t *)dA, (const uint8_t *)dB, (half *)dC, M, N, K);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    float avg = ms / 50;
    double flops = 2.0 * (double)M * N * K;
    return (flops / (avg / 1000.0)) / 1e12;
}

int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d, %d SMs)\n\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    // Occupancy check
    {
        int smem64 = 2 * (BM + BN) * BK_PAD;
        int smem128 = 2 * (BM + BN) * BK_PAD2;
        int b64, b128;
        cudaFuncSetAttribute(fp8_gemm_v4, cudaFuncAttributeMaxDynamicSharedMemorySize, smem64);
        cudaFuncSetAttribute(fp8_gemm_v4_bk128, cudaFuncAttributeMaxDynamicSharedMemorySize, smem128);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b64, fp8_gemm_v4, NTHREADS, smem64);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b128, fp8_gemm_v4_bk128, NTHREADS, smem128);
        printf("Occupancy:\n");
        printf("  BK=64  smem=%5d → %d blocks/SM (%d warps)\n", smem64, b64, b64 * 8);
        printf("  BK=128 smem=%5d → %d blocks/SM (%d warps)\n", smem128, b128, b128 * 8);
        printf("\n");
    }

    int sizes[] = {1024, 2048, 4096, 8192};
    int nsizes = 4;

    printf("=== FP8 PTX GEMM v4 Comparison ===\n");
    printf("%-8s %-14s %-14s\n", "Size", "BK=64 (TFLOPS)", "BK=128 (TFLOPS)");
    printf("------------------------------------------\n");

    for (int si = 0; si < nsizes; si++)
    {
        int M = sizes[si], N = sizes[si], K = sizes[si];
        // BK=128 needs K%128==0
        if (M % BM != 0 || N % BN != 0)
            continue;

        size_t sA = (size_t)M * K;
        size_t sB = (size_t)N * K;
        size_t sC = (size_t)M * N;

        void *dA, *dB, *dC;
        cudaMalloc(&dA, sA);
        cudaMalloc(&dB, sB);
        cudaMalloc(&dC, sC * 2);

        std::vector<uint8_t> h(sA > sB ? sA : sB);
        unsigned rng = 42 + si;
        for (size_t i = 0; i < h.size(); i++)
        {
            rng = rng * 1103515245u + 12345u;
            h[i] = (rng >> 16) & 0x3F;
        }
        cudaMemcpy(dA, h.data(), sA, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, h.data(), sB, cudaMemcpyHostToDevice);

        int smem64 = 2 * (BM + BN) * BK_PAD;
        double tf64 = bench_kernel(fp8_gemm_v4, M, N, K, dA, dB, dC, smem64);

        double tf128 = -1;
        if (K % BK2 == 0)
        {
            int smem128 = 2 * (BM + BN) * BK_PAD2;
            tf128 = bench_kernel(fp8_gemm_v4_bk128, M, N, K, dA, dB, dC, smem128);
        }

        printf("%-8d %-14.1f ", M, tf64);
        if (tf128 > 0)
            printf("%-14.1f\n", tf128);
        else
            printf("SKIP\n");

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }

    printf("\n");
    printf("v1:            193 TFLOPS\n");
    printf("v2:            302 TFLOPS\n");
    printf("cuBLASLt:      346 TFLOPS (FP32 accum)\n");
    printf("Hardware peak: 642 TFLOPS (FP16 accum)\n");

    return 0;
}
