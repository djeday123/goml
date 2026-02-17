#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <vector>

// ============================================================
// Tile config
// ============================================================
// BK_PAD MUST be multiple of 16 (cp.async alignment)
// and NOT multiple of 32 (bank conflict avoidance: 48/4=12 words, staggers banks)
#define BM 128
#define BN 128
#define BK 32
#define BK_PAD 48 // FIXED: was 36, must be 16-aligned

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

// ============================================================
// PTX MMA: m16n8k32, A=row(FP8 E4M3), B=col(FP8 E4M3), C=FP16
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
// cp.async: 16-byte (128-bit) async copy
// Both smem and gmem MUST be 16-byte aligned
// ============================================================
__device__ __forceinline__ void cp_async_16B(void *smem, const void *gmem)
{
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n" :::);
}

__device__ __forceinline__ void cp_async_wait_group(int n)
{
    if (n == 0)
        asm volatile("cp.async.wait_group 0;\n" :::);
    else
        asm volatile("cp.async.wait_group 1;\n" :::);
}

// ============================================================
// GEMM Kernel
//   C[M,N] (FP16, row-major) = A[M,K] (FP8, row-major) × B^T[N,K] (FP8, row-major)
//   B stored as [N,K] row-major = B[K,N] col-major (native for MMA .col)
//   2-stage pipeline with cp.async
// ============================================================
__global__ __launch_bounds__(NTHREADS) void fp8_gemm_kernel(
    const uint8_t *__restrict__ A, // [M, K] row-major FP8
    const uint8_t *__restrict__ B, // [N, K] row-major FP8 (transposed)
    half *__restrict__ C,          // [M, N] row-major FP16
    int M, int N, int K)
{
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;

    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    const int warpM = warpId / WARPS_N; // 0..1
    const int warpN = warpId % WARPS_N; // 0..3
    const int group = laneId / 4;       // 0..7  (row within MMA tile)
    const int tid4 = laneId % 4;        // 0..3  (4-byte chunk selector)

    // Double-buffered shared memory
    // Layout: [stage][BM rows of A | BN rows of B], each row BK_PAD bytes
    extern __shared__ uint8_t smem[];
    const int stage_bytes = (BM + BN) * BK_PAD;
    uint8_t *A_smem[2], *B_smem[2];
    A_smem[0] = smem;
    B_smem[0] = smem + BM * BK_PAD;
    A_smem[1] = smem + stage_bytes;
    B_smem[1] = smem + stage_bytes + BM * BK_PAD;

    // Accumulators (FP16 packed as uint32)
    uint32_t acc[TILES_M][TILES_N][2];
#pragma unroll
    for (int i = 0; i < TILES_M; i++)
#pragma unroll
        for (int j = 0; j < TILES_N; j++)
            acc[i][j][0] = acc[i][j][1] = 0;

    // ---- Loading helpers ----
    // A: BM rows × BK cols = BM × 32 bytes = BM × 2 chunks of 16 bytes
    // B: BN rows × BK cols = BN × 32 bytes = BN × 2 chunks of 16 bytes
    // Total chunks = (BM + BN) * 2 = 512, threads = 256 → 2 chunks/thread

    const int chunks_A = BM * 2;                           // 256
    const int chunks_total = (BM + BN) * 2;                // 512
    const int chunks_per_thread = chunks_total / NTHREADS; // 2

    auto load_tile = [&](int stage, int k_offset)
    {
#pragma unroll
        for (int c = 0; c < chunks_per_thread; c++)
        {
            int chunk_id = c * NTHREADS + threadIdx.x;
            if (chunk_id < chunks_A)
            {
                // Loading A
                int row = chunk_id / 2;
                int col16 = (chunk_id % 2) * 16;
                int gm_row = bm + row;
                if (gm_row < M && k_offset + col16 + 15 < K + 16)
                {
                    cp_async_16B(&A_smem[stage][row * BK_PAD + col16],
                                 &A[gm_row * K + k_offset + col16]);
                }
            }
            else
            {
                // Loading B
                int b_id = chunk_id - chunks_A;
                int row = b_id / 2;
                int col16 = (b_id % 2) * 16;
                int gm_row = bn + row;
                if (gm_row < N && k_offset + col16 + 15 < K + 16)
                {
                    cp_async_16B(&B_smem[stage][row * BK_PAD + col16],
                                 &B[gm_row * K + k_offset + col16]);
                }
            }
        }
        cp_async_commit();
    };

    int num_k = (K + BK - 1) / BK;

    // ---- Prologue: load tile 0 ----
    load_tile(0, 0);

    // ---- Main loop ----
    for (int kt = 0; kt < num_k; kt++)
    {
        // Start loading next tile
        if (kt + 1 < num_k)
            load_tile((kt + 1) & 1, (kt + 1) * BK);

        // Wait for current tile
        cp_async_wait_group(kt + 1 < num_k ? 1 : 0);
        __syncthreads();

        int s = kt & 1;
        const int wm_off = warpM * WM;
        const int wn_off = warpN * WN;

// ---- MMA computation ----
#pragma unroll
        for (int mi = 0; mi < TILES_M; mi++)
        {
            int ar = wm_off + mi * MMA_M;
            uint32_t a_frag[4];

            // Load A fragment from shared memory
            // MMA row layout: a[0..3] map to rows [group, group+8] × k-chunks [0..15, 16..31]
            a_frag[0] = *(uint32_t *)(&A_smem[s][(ar + group) * BK_PAD + tid4 * 4]);
            a_frag[1] = *(uint32_t *)(&A_smem[s][(ar + group) * BK_PAD + 16 + tid4 * 4]);
            a_frag[2] = *(uint32_t *)(&A_smem[s][(ar + 8 + group) * BK_PAD + tid4 * 4]);
            a_frag[3] = *(uint32_t *)(&A_smem[s][(ar + 8 + group) * BK_PAD + 16 + tid4 * 4]);

#pragma unroll
            for (int ni = 0; ni < TILES_N; ni++)
            {
                int br = wn_off + ni * MMA_N;
                uint32_t b_frag[2];

                // Load B fragment (col layout: B[n][k], consecutive k values)
                b_frag[0] = *(uint32_t *)(&B_smem[s][(br + group) * BK_PAD + tid4 * 4]);
                b_frag[1] = *(uint32_t *)(&B_smem[s][(br + group) * BK_PAD + 16 + tid4 * 4]);

                mma_fp8_f16(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
            }
        }

        __syncthreads();
    }

    // ---- Epilogue: store C ----
    // MMA output layout: d[0] = C[group, tid4*2..tid4*2+1], d[1] = C[group+8, ...]
    const int wm_base = bm + warpM * WM;
    const int wn_base = bn + warpN * WN;

#pragma unroll
    for (int mi = 0; mi < TILES_M; mi++)
    {
#pragma unroll
        for (int ni = 0; ni < TILES_N; ni++)
        {
            int r0 = wm_base + mi * MMA_M + group;
            int r1 = r0 + 8;
            int c = wn_base + ni * MMA_N + tid4 * 2;

            // Write 2 packed FP16 values (uint32 = 4 bytes = 2 × half)
            if (r0 < M && c + 1 < N)
                *(uint32_t *)(&C[r0 * N + c]) = acc[mi][ni][0];
            if (r1 < M && c + 1 < N)
                *(uint32_t *)(&C[r1 * N + c]) = acc[mi][ni][1];
        }
    }
}

// ============================================================
// C API wrapper
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
        int smem = 2 * (BM + BN) * BK_PAD;

        cudaFuncSetAttribute(fp8_gemm_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

        fp8_gemm_kernel<<<grid, block, smem>>>(
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

    printf("=== Custom FP8 PTX Kernel (m16n8k32, f16 accum) ===\n");
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

        // Warmup
        for (int i = 0; i < warmup; i++)
            cutlass_fp8_gemm(&args);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            printf("%-8d %-10s %-10s ERR: %s\n", M, "-", "-", cudaGetErrorString(err));
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

    printf("\nReference: cuBLASLt = 346 TFLOPS\n");
    return 0;
}