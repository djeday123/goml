// FP8 GEMM v17 - BK=64, XOR swizzle, 5 blocks/SM (40 warps)
// Strategy: trade compute/barrier ratio for massive occupancy increase
// v10b: BK=128, 3 blk/SM, 24 warps, 64 QMMA/barrier → 546 TFLOPS
// v17:  BK=64,  5 blk/SM, 40 warps, 32 QMMA/barrier → target 570+
//
// SMEM per block: 128*64 + 128*64 = 16384 bytes
// 5 blocks: 16384*5 = 81920 <= 102400 ✓
// Registers: 65536/(5*256) = 51 max → need tight register budget

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Tile dimensions
#define BM 128
#define BN 128
#define BK 64
#define SMEM_STRIDE 64 // No padding, XOR swizzle handles banks

// MMA dimensions (m16n8k32)
#define MMA_M 16
#define MMA_N 8
#define MMA_K 32

// Warp layout: 2x4
#define WARPS_M 2
#define WARPS_N 4

// Tiles per warp
#define WM (BM / WARPS_M)    // 64
#define WN (BN / WARPS_N)    // 32
#define M_TILES (WM / MMA_M) // 4
#define N_TILES (WN / MMA_N) // 4
#define K_STEPS (BK / MMA_K) // 2

// SMEM sizes
#define SMEM_A_SIZE (BM * SMEM_STRIDE)         // 8192
#define SMEM_B_SIZE (BN * SMEM_STRIDE)         // 8192
#define SMEM_TOTAL (SMEM_A_SIZE + SMEM_B_SIZE) // 16384

// FP8 e4m3 helpers
__device__ __forceinline__ uint32_t swizzle_addr(uint32_t row, uint32_t col)
{
    // XOR swizzle: rotate 16-byte chunk by (row & 7)
    uint32_t addr = row * SMEM_STRIDE + col;
    uint32_t row_swizzle = row & 7;
    uint32_t col_swizzle = row_swizzle << 4; // shift by 16 bytes
    addr ^= col_swizzle;
    return addr;
}

__global__ void __launch_bounds__(256, 5)
    fp8_gemm_v17_kernel(const uint8_t *__restrict__ A,
                        const uint8_t *__restrict__ B,
                        half *__restrict__ C,
                        int M, int N, int K)
{
    // Shared memory
    __shared__ uint8_t smem_A[SMEM_A_SIZE];
    __shared__ uint8_t smem_B[SMEM_B_SIZE];

    // Block tile position
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;

    // Warp and lane IDs
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_m = warp_id / WARPS_N; // 0-1
    const int warp_n = warp_id % WARPS_N; // 0-3

    // Warp tile base
    const int wm = warp_m * WM; // 0 or 64
    const int wn = warp_n * WN; // 0, 32, 64, 96

    // MMA thread decomposition
    const int group_id = lane_id / 4; // 0-7
    const int tid = lane_id % 4;      // 0-3

    // Accumulators (4x4 tiles, each uint32 = 2 x fp16)
    uint32_t acc[M_TILES][N_TILES][2] = {};

    // Main K loop
    for (int k = 0; k < K; k += BK)
    {
// ===== Load A tile [BM, BK] from global to shared =====
// 256 threads, 16 bytes each = 4096 bytes/pass
// Need BM * BK = 8192 bytes → 2 passes
#pragma unroll
        for (int pass = 0; pass < 2; pass++)
        {
            int elem_idx = pass * 256 + threadIdx.x;
            int row = elem_idx / (BK / 16); // 16 bytes per load
            int col16 = elem_idx % (BK / 16);
            int col = col16 * 16;

            if (bm + row < M && k + col < K)
            {
                uint4 val = *reinterpret_cast<const uint4 *>(&A[(bm + row) * K + k + col]);
                uint32_t sw_addr = swizzle_addr(row, col);
                *reinterpret_cast<uint4 *>(&smem_A[sw_addr]) = val;
            }
        }

// ===== Load B tile [BN, BK] (B stored as [N, K]) =====
#pragma unroll
        for (int pass = 0; pass < 2; pass++)
        {
            int elem_idx = pass * 256 + threadIdx.x;
            int row = elem_idx / (BK / 16);
            int col16 = elem_idx % (BK / 16);
            int col = col16 * 16;

            if (bn + row < N && k + col < K)
            {
                uint4 val = *reinterpret_cast<const uint4 *>(&B[(bn + row) * K + k + col]);
                uint32_t sw_addr = swizzle_addr(row, col);
                *reinterpret_cast<uint4 *>(&smem_B[sw_addr]) = val;
            }
        }

        __syncthreads();

// ===== Compute: iterate over K_STEPS =====
#pragma unroll
        for (int ki = 0; ki < K_STEPS; ki++)
        {
            uint32_t k_off = ki * MMA_K; // 0 or 32

// Load A fragments and compute MMA for each tile
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                // A fragment: 4 uint32 per mi
                // SM89 layout: a0=row_g/k_low, a1=row_g+8/k_low, a2=row_g/k_high, a3=row_g+8/k_high
                uint32_t a_frag[4];
                int a_row0 = wm + mi * MMA_M + group_id;
                int a_row1 = a_row0 + 8;
                uint32_t a_base0 = swizzle_addr(a_row0, k_off + tid * 4);
                uint32_t a_base0h = swizzle_addr(a_row0, k_off + tid * 4 + 16);
                uint32_t a_base1 = swizzle_addr(a_row1, k_off + tid * 4);
                uint32_t a_base1h = swizzle_addr(a_row1, k_off + tid * 4 + 16);

                a_frag[0] = *reinterpret_cast<uint32_t *>(&smem_A[a_base0]);  // row_g, k_low
                a_frag[1] = *reinterpret_cast<uint32_t *>(&smem_A[a_base1]);  // row_g+8, k_low  (SM89 swap!)
                a_frag[2] = *reinterpret_cast<uint32_t *>(&smem_A[a_base0h]); // row_g, k_high   (SM89 swap!)
                a_frag[3] = *reinterpret_cast<uint32_t *>(&smem_A[a_base1h]); // row_g+8, k_high

#pragma unroll
                for (int ni = 0; ni < N_TILES; ni++)
                {
                    // B fragment: 2 uint32 per ni
                    uint32_t b_frag[2];
                    int b_row = wn + ni * MMA_N + group_id;
                    uint32_t b_base0 = swizzle_addr(b_row, k_off + tid * 4);
                    uint32_t b_base1 = swizzle_addr(b_row, k_off + tid * 4 + 16);
                    b_frag[0] = *reinterpret_cast<uint32_t *>(&smem_B[b_base0]);
                    b_frag[1] = *reinterpret_cast<uint32_t *>(&smem_B[b_base1]);

                    // MMA: m16n8k32 FP8 with F16 accumulator
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
                        "{%0, %1}, "
                        "{%2, %3, %4, %5}, "
                        "{%6, %7}, "
                        "{%0, %1};"
                        : "+r"(acc[mi][ni][0]), "+r"(acc[mi][ni][1])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                          "r"(b_frag[0]), "r"(b_frag[1]));
                }
            }
        }

        __syncthreads();
    }

// ===== Store C tile =====
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
            {
                *reinterpret_cast<uint32_t *>(&C[row0 * N + col]) = acc[mi][ni][0];
            }
            if (row1 < M && col + 1 < N)
            {
                *reinterpret_cast<uint32_t *>(&C[row1 * N + col]) = acc[mi][ni][1];
            }
        }
    }
}

// ===================== FP8 conversion helpers =====================
static const uint8_t fp8_e4m3_table[16] = {
    0x00, 0x3C, 0x40, 0x42, 0x44, 0x45, 0x46, 0x47,
    0x48, 0x49, 0x49, 0x4A, 0x4A, 0x4B, 0x4B, 0x4C};

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
    uint8_t sign = v >> 7;
    uint8_t exp = (v >> 3) & 0xF;
    uint8_t mant = v & 0x7;
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
void run_benchmark(int M, int N, int K, float *best_tflops)
{
    size_t sizeA = (size_t)M * K;
    size_t sizeB = (size_t)N * K;
    size_t sizeC = (size_t)M * N * sizeof(half);

    uint8_t *d_A, *d_B;
    half *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // Init with FP8 1.5 (0x3C)
    cudaMemset(d_A, 0x3C, sizeA);
    cudaMemset(d_B, 0x3C, sizeB);

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(256);

    // Warmup
    for (int i = 0; i < 5; i++)
        fp8_gemm_v17_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iters = 20;
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++)
        fp8_gemm_v17_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= iters;

    double flops = 2.0 * M * N * K;
    double tflops = flops / (ms * 1e-3) / 1e12;

    const char *mark = (tflops > 500) ? "  * >500 *" : (tflops > *best_tflops) ? "  * best *"
                                                                               : "";
    printf("  %5d x %5d x %5d  %7.3f ms  %7.1f TFLOPS%s\n", M, N, K, ms, tflops, mark);
    if (tflops > *best_tflops)
        *best_tflops = tflops;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void run_correctness()
{
    printf("--- Correctness ---\n");

    int sizes[] = {128, 256, 512};
    for (int si = 0; si < 3; si++)
    {
        int N = sizes[si];

        // Identity test
        uint8_t *h_A = (uint8_t *)calloc(N * N, 1);
        uint8_t *h_B = (uint8_t *)calloc(N * N, 1);

        // A = identity in FP8
        for (int i = 0; i < N; i++)
            h_A[i * N + 0] = 0; // zero matrix first
        for (int i = 0; i < N; i++)
            h_A[i * N + i] = f2e4m3(1.0f);

        // B = random-ish
        for (int i = 0; i < N * N; i++)
            h_B[i] = f2e4m3((float)((i * 7 + 3) % 15) - 7.0f);

        uint8_t *d_A, *d_B;
        half *d_C;
        cudaMalloc(&d_A, N * N);
        cudaMalloc(&d_B, N * N);
        cudaMalloc(&d_C, N * N * sizeof(half));
        cudaMemcpy(d_A, h_A, N * N, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, N * N, cudaMemcpyHostToDevice);

        dim3 grid((N + BN - 1) / BN, (N + BM - 1) / BM);
        fp8_gemm_v17_kernel<<<grid, 256>>>(d_A, d_B, d_C, N, N, N);

        half *h_C = (half *)malloc(N * N * sizeof(half));
        cudaMemcpy(h_C, d_C, N * N * sizeof(half), cudaMemcpyDeviceToHost);

        // Check: C should equal B^T for identity A
        int errors = 0;
        float max_err = 0;
        for (int i = 0; i < N && errors < 10; i++)
        {
            for (int j = 0; j < N; j++)
            {
                float gpu = __half2float(h_C[i * N + j]);
                float ref = e4m3_to_f(h_B[j * N + i]);
                float err = fabsf(gpu - ref);
                if (err > max_err)
                    max_err = err;
                if (err > 0.5f)
                    errors++;
            }
        }
        printf("  Identity %d: %d/%d → %s\n", N, errors, N * N, errors == 0 ? "PASS" : "FAIL");

        // Random test
        for (int i = 0; i < N * N; i++)
        {
            h_A[i] = f2e4m3(((float)((i * 13 + 5) % 15) - 7.0f) * 0.5f);
            h_B[i] = f2e4m3(((float)((i * 11 + 7) % 15) - 7.0f) * 0.5f);
        }
        cudaMemcpy(d_A, h_A, N * N, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, N * N, cudaMemcpyHostToDevice);
        fp8_gemm_v17_kernel<<<grid, 256>>>(d_A, d_B, d_C, N, N, N);
        cudaMemcpy(h_C, d_C, N * N * sizeof(half), cudaMemcpyDeviceToHost);

        errors = 0;
        max_err = 0;
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                float gpu = __half2float(h_C[i * N + j]);
                float ref = 0;
                for (int kk = 0; kk < N; kk++)
                    ref += e4m3_to_f(h_A[i * N + kk]) * e4m3_to_f(h_B[j * N + kk]);
                float err = fabsf(gpu - (float)__float2half(ref));
                if (err > max_err)
                    max_err = err;
                if (err > 1.0f)
                    errors++;
            }
        }
        printf("  Random %dx%dx%d: max_err=%.4f errors=%d/%d → %s\n",
               N, N, N, max_err, errors, N * N, errors == 0 ? "PASS" : "FAIL");

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);
    }
}

int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Check occupancy
    int numBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, fp8_gemm_v17_kernel, 256, SMEM_TOTAL);

    printf("=== FP8 GEMM v17 (BK=64, XOR swizzle, target 5 blocks/SM) ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n", prop.name, prop.multiProcessorCount, prop.clockRate / 1000);
    printf("SMEM: %d B/block → %d blocks = %d ≤ %d\n",
           SMEM_TOTAL, numBlocks, SMEM_TOTAL * numBlocks, (int)prop.sharedMemPerMultiprocessor);
    printf("Achieved blocks/SM: %d (target: 5)\n", numBlocks);

    run_correctness();

    printf("--- Performance ---\n");
    float best = 0;
    run_benchmark(1024, 1024, 1024, &best);
    run_benchmark(2048, 2048, 2048, &best);
    run_benchmark(4096, 4096, 4096, &best);
    run_benchmark(8192, 8192, 8192, &best);

    printf("--- LLM ---\n");
    run_benchmark(2048, 4096, 4096, &best);
    run_benchmark(2048, 11008, 4096, &best);
    run_benchmark(2048, 4096, 11008, &best);
    run_benchmark(4096, 4096, 4096, &best);
    run_benchmark(4096, 11008, 4096, &best);
    run_benchmark(4096, 4096, 11008, &best);
    run_benchmark(8192, 4096, 4096, &best);
    run_benchmark(8192, 11008, 4096, &best);

    printf("Peak:660 | cuBLAS:~330 | v10b:546 | v17: %.0f\n", best);
    return 0;
}
