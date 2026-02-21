// =============================================================================
// FP8 GEMM Diagnostic — Find exactly where time is spent
// =============================================================================
// Instruments the kernel with clock64() to measure:
//   1. Load time (cp.async + wait)
//   2. Compute time (fragment load + MMA)
//   3. Store time (epilogue)
// Also reports: registers, smem, occupancy, bank conflicts
//
// Build: nvcc -O3 -arch=sm_89 -std=c++17 -Xptxas -v fp8_diag.cu -o fp8_diag
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <vector>

// ============================================================
// We test BOTH v2 config (2-stage) and v3 config (3-stage)
// to see which has better occupancy
// ============================================================

#define BM 128
#define BN 128
#define BK 64
#define BK_PAD 80

#define WARPS_M 2
#define WARPS_N 4
#define WM (BM / WARPS_M)
#define WN (BN / WARPS_N)
#define NTHREADS (WARPS_M * WARPS_N * 32)

#define MMA_M 16
#define MMA_N 8
#define MMA_K 32
#define TILES_M (WM / MMA_M)
#define TILES_N (WN / MMA_N)
#define K_ITERS (BK / MMA_K)

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
// Diagnostic counters (per-block, written by thread 0)
// ============================================================
struct DiagCounters
{
    long long load_cycles;    // total cycles in load + wait
    long long compute_cycles; // total cycles in fragment load + MMA
    long long store_cycles;   // total cycles in epilogue
    long long total_cycles;   // total kernel cycles
    int num_tiles;            // number of K tiles processed
};

// ============================================================
// Instrumented v2 kernel (2-stage, B-preload)
// ============================================================
__global__ __launch_bounds__(NTHREADS, 2) void fp8_gemm_diag(
    const uint8_t *__restrict__ A,
    const uint8_t *__restrict__ B,
    half *__restrict__ C,
    DiagCounters *__restrict__ diag,
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

    // 2-stage double buffer
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

    // Timing
    long long t_load = 0, t_compute = 0, t_store = 0;
    long long t0, t1;
    bool is_timer = (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0);

    long long t_total_start = clock64();

    // ---- Prologue ----
    if (is_timer)
        t0 = clock64();
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
    if (is_timer)
    {
        t1 = clock64();
        t_load += (t1 - t0);
    }

    // ---- Main loop ----
    for (int kt = 0; kt < num_k; kt++)
    {
        // Load next tile
        if (is_timer)
            t0 = clock64();
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

        // Wait for current tile
        if (kt + 1 < num_k)
            cp_async_wait1();
        else
            cp_async_wait0();
        __syncthreads();
        if (is_timer)
        {
            t1 = clock64();
            t_load += (t1 - t0);
        }

        // ---- Compute ----
        if (is_timer)
            t0 = clock64();
        {
            const int s = kt & 1;

#pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
            {
                const int kb = ki * MMA_K;

                // Preload ALL B fragments
                uint32_t b_all[TILES_N][2];
#pragma unroll
                for (int ni = 0; ni < TILES_N; ni++)
                {
                    const int br = wn_off + ni * MMA_N;
                    b_all[ni][0] = *(uint32_t *)(&B_smem[s][(br + group) * BK_PAD + kb + tid4 * 4]);
                    b_all[ni][1] = *(uint32_t *)(&B_smem[s][(br + group) * BK_PAD + kb + 16 + tid4 * 4]);
                }

                // Preload ALL A fragments
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

// ALL MMA from registers
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
        }
        if (is_timer)
        {
            t1 = clock64();
            t_compute += (t1 - t0);
        }

        __syncthreads();
    }

    // ---- Epilogue ----
    if (is_timer)
        t0 = clock64();
    {
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
    if (is_timer)
    {
        t1 = clock64();
        t_store = t1 - t0;
        long long t_total = t1 - t_total_start;

        diag->load_cycles = t_load;
        diag->compute_cycles = t_compute;
        diag->store_cycles = t_store;
        diag->total_cycles = t_total;
        diag->num_tiles = num_k;
    }
}

// ============================================================
// C API (same interface)
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

        cudaFuncSetAttribute(fp8_gemm_diag,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

        // Dummy diag buffer (not used in perf path)
        DiagCounters *d_diag;
        cudaMalloc(&d_diag, sizeof(DiagCounters));

        fp8_gemm_diag<<<grid, block, smem>>>(
            (const uint8_t *)a->A,
            (const uint8_t *)a->B,
            (half *)a->C,
            d_diag,
            M, N, K);

        cudaFree(d_diag);
        return 0;
    }
    int cuda_device_sync(void) { return (int)cudaDeviceSynchronize(); }
}

// ============================================================
// Main: Diagnostics + Benchmark
// ============================================================
int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float gpu_mhz = prop.clockRate / 1000.0f;

    printf("GPU: %s (SM %d.%d, %d SMs, %.0f MHz boost)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, gpu_mhz);
    printf("Max smem/block: %zu bytes\n", prop.sharedMemPerBlockOptin);
    printf("Max regs/thread: %d\n", prop.regsPerBlock / (NTHREADS));
    printf("\n");

    // ---- Occupancy check ----
    printf("=== Occupancy Analysis ===\n");
    {
        int smem_2stage = 2 * (BM + BN) * BK_PAD;
        int smem_3stage = 3 * (BM + BN) * BK_PAD;

        int blocks_2s = 0, blocks_3s = 0;
        cudaFuncSetAttribute(fp8_gemm_diag,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_3stage);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_2s, fp8_gemm_diag, NTHREADS, smem_2stage);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_3s, fp8_gemm_diag, NTHREADS, smem_3stage);

        printf("  2-stage smem: %6d bytes → %d blocks/SM → %d warps/SM\n",
               smem_2stage, blocks_2s, blocks_2s * (NTHREADS / 32));
        printf("  3-stage smem: %6d bytes → %d blocks/SM → %d warps/SM\n",
               smem_3stage, blocks_3s, blocks_3s * (NTHREADS / 32));
        printf("  Max warps/SM: %d\n", prop.maxThreadsPerMultiProcessor / 32);
        printf("\n");
    }

    // ---- Diagnostic run (single size, detailed timing) ----
    printf("=== Cycle-Level Diagnostics (4096×4096×4096) ===\n");
    {
        int M = 4096, N = 4096, K = 4096;
        size_t sA = (size_t)M * K;
        size_t sB = (size_t)N * K;
        size_t sC = (size_t)M * N;

        void *dA, *dB, *dC;
        cudaMalloc(&dA, sA);
        cudaMalloc(&dB, sB);
        cudaMalloc(&dC, sC * 2);

        std::vector<uint8_t> hA(sA), hB(sB);
        unsigned rng = 42;
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
        cudaMemcpy(dA, hA.data(), sA, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), sB, cudaMemcpyHostToDevice);

        DiagCounters *d_diag;
        cudaMalloc(&d_diag, sizeof(DiagCounters));
        cudaMemset(d_diag, 0, sizeof(DiagCounters));

        dim3 grid(N / BN, M / BM);
        dim3 block(NTHREADS);
        int smem = 2 * (BM + BN) * BK_PAD;
        cudaFuncSetAttribute(fp8_gemm_diag,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

        // Warmup
        fp8_gemm_diag<<<grid, block, smem>>>(
            (const uint8_t *)dA, (const uint8_t *)dB, (half *)dC, d_diag, M, N, K);
        cudaDeviceSynchronize();

        // Diagnostic run
        cudaMemset(d_diag, 0, sizeof(DiagCounters));
        fp8_gemm_diag<<<grid, block, smem>>>(
            (const uint8_t *)dA, (const uint8_t *)dB, (half *)dC, d_diag, M, N, K);
        cudaDeviceSynchronize();

        DiagCounters h_diag;
        cudaMemcpy(&h_diag, d_diag, sizeof(DiagCounters), cudaMemcpyDeviceToHost);

        long long total = h_diag.total_cycles;
        if (total > 0)
        {
            printf("  K tiles:        %d\n", h_diag.num_tiles);
            printf("  Load cycles:    %lld (%.1f%%)\n", h_diag.load_cycles,
                   100.0 * h_diag.load_cycles / total);
            printf("  Compute cycles: %lld (%.1f%%)\n", h_diag.compute_cycles,
                   100.0 * h_diag.compute_cycles / total);
            printf("  Store cycles:   %lld (%.1f%%)\n", h_diag.store_cycles,
                   100.0 * h_diag.store_cycles / total);
            printf("  Total cycles:   %lld\n", total);
            printf("  Unaccounted:    %.1f%%\n",
                   100.0 * (total - h_diag.load_cycles - h_diag.compute_cycles - h_diag.store_cycles) / total);
            printf("\n");

            // Per-tile breakdown
            long long load_per_tile = h_diag.load_cycles / h_diag.num_tiles;
            long long compute_per_tile = h_diag.compute_cycles / h_diag.num_tiles;
            printf("  Per tile:\n");
            printf("    Load:    %lld cycles (%.1f µs @ %.0f MHz)\n",
                   load_per_tile, load_per_tile / (gpu_mhz * 1000.0), gpu_mhz);
            printf("    Compute: %lld cycles (%.1f µs @ %.0f MHz)\n",
                   compute_per_tile, compute_per_tile / (gpu_mhz * 1000.0), gpu_mhz);
            printf("    Ratio:   compute/load = %.2fx\n",
                   (double)compute_per_tile / load_per_tile);
            printf("\n");

            // Theoretical analysis
            int mma_per_tile = TILES_M * TILES_N * K_ITERS;                  // 4*4*2 = 32
            int smem_loads_per_tile = (TILES_M * 4 + TILES_N * 2) * K_ITERS; // (16+8)*2 = 48
            printf("  MMA per tile:        %d\n", mma_per_tile);
            printf("  Smem loads per tile: %d (uint32)\n", smem_loads_per_tile);
            printf("  MMA/smem_load ratio: %.2f\n", (double)mma_per_tile / smem_loads_per_tile);
        }

        // ---- Timed benchmark ----
        printf("\n=== Performance Benchmark ===\n");
        printf("%-8s %-10s %-10s\n", "Size", "Time(ms)", "TFLOPS");
        printf("--------------------------------------------\n");

        int sizes[] = {1024, 2048, 4096, 8192};
        for (int si = 0; si < 4; si++)
        {
            M = sizes[si];
            N = sizes[si];
            K = sizes[si];
            if (M % BM != 0 || N % BN != 0 || K % BK != 0)
            {
                printf("%-8d SKIP\n", M);
                continue;
            }

            sA = (size_t)M * K;
            sB = (size_t)N * K;
            sC = (size_t)M * N;
            cudaFree(dA);
            cudaFree(dB);
            cudaFree(dC);
            cudaMalloc(&dA, sA);
            cudaMalloc(&dB, sB);
            cudaMalloc(&dC, sC * 2);

            hA.resize(sA);
            hB.resize(sB);
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
            cudaMemcpy(dA, hA.data(), sA, cudaMemcpyHostToDevice);
            cudaMemcpy(dB, hB.data(), sB, cudaMemcpyHostToDevice);

            grid = dim3(N / BN, M / BM);
            for (int i = 0; i < 10; i++)
                fp8_gemm_diag<<<grid, block, smem>>>(
                    (const uint8_t *)dA, (const uint8_t *)dB, (half *)dC, d_diag, M, N, K);
            cudaDeviceSynchronize();

            cudaEvent_t t0, t1;
            cudaEventCreate(&t0);
            cudaEventCreate(&t1);
            cudaEventRecord(t0);
            for (int i = 0; i < 50; i++)
                fp8_gemm_diag<<<grid, block, smem>>>(
                    (const uint8_t *)dA, (const uint8_t *)dB, (half *)dC, d_diag, M, N, K);
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            float ms;
            cudaEventElapsedTime(&ms, t0, t1);
            float avg = ms / 50;
            double flops = 2.0 * (double)M * N * K;
            double tflops = (flops / (avg / 1000.0)) / 1e12;
            printf("%-8d %-10.3f %-10.1f\n", M, avg, tflops);

            cudaEventDestroy(t0);
            cudaEventDestroy(t1);
        }

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        cudaFree(d_diag);
    }

    return 0;
}
