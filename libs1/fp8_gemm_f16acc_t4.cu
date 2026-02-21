// =============================================================================
// FP8 GEMM with FP16 Accumulator — v6
// =============================================================================
// Changes from v4 (473 TFLOPS):
//   1. BK 64→128: K_STEPS=4, halves barrier count, doubles compute per tile
//   2. SMEM_STRIDE=144 (128+16): bank-conflict-free for 8-group access
//   3. Software-pipelined fragment loads: prefetch next ki while MMA executes
//   4. No cp.async (regressed in v5 due to overhead)
//   5. 8 load passes per matrix (was 2), but 2x fewer tiles overall
//
// SMEM: 128*144 + 128*144 = 36864 bytes < 48KB default carveout
//
// Build:
//   nvcc -O3 -arch=sm_89 -std=c++17 fp8_gemm_f16acc.cu -o fp8_gemm -lcudart
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define BM 128
#define BN 128
#define BK 128
#define SMEM_STRIDE 144 // 128 + 16 padding for bank-conflict-free access
#define WARPS_M 2
#define WARPS_N 4
#define BLOCK_THREADS (WARPS_M * WARPS_N * 32) // 256
#define WM (BM / WARPS_M)                      // 64
#define WN (BN / WARPS_N)                      // 32
#define MMA_M 16
#define MMA_N 8
#define MMA_K 32
#define M_TILES (WM / MMA_M) // 4
#define N_TILES (WN / MMA_N) // 4
#define K_STEPS (BK / MMA_K) // 4

__device__ __forceinline__ void mma_fp8(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t &c0, uint32_t &c1)
{
    uint32_t d0, d1;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1));
    c0 = d0;
    c1 = d1;
}

__global__ void __launch_bounds__(BLOCK_THREADS)
    fp8_gemm_f16acc_kernel(
        const uint8_t *__restrict__ A,
        const uint8_t *__restrict__ B,
        uint16_t *__restrict__ C,
        int M, int N, int K)
{
    __shared__ uint8_t smem_A[BM * SMEM_STRIDE]; // 128 * 144 = 18432
    __shared__ uint8_t smem_B[BN * SMEM_STRIDE]; // 128 * 144 = 18432

    const int bm = blockIdx.x * BM;
    const int bn = blockIdx.y * BN;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    const int wm = warp_m * WM;
    const int wn = warp_n * WN;
    const int group_id = lane_id / 4;
    const int tid = lane_id % 4;

    // Load geometry: 256 threads × 16 bytes = 4096 bytes/pass
    // A: 128 rows × 128 cols = 16384 bytes → 4 passes  (32 rows/pass)
    // B: 128 rows × 128 cols = 16384 bytes → 4 passes  (32 rows/pass)
    const int thr_per_row = BK / 16;                        // 128/16 = 8 threads per row
    const int rows_per_pass = BLOCK_THREADS / thr_per_row;  // 256/8 = 32 rows/pass
    const int load_row_in_pass = threadIdx.x / thr_per_row; // 0..31
    const int load_col = (threadIdx.x % thr_per_row) * 16;  // 0,16,32,...,112

    // Accumulators: 4×4×2 = 32 registers
    uint32_t acc[M_TILES][N_TILES][2];
#pragma unroll
    for (int mi = 0; mi < M_TILES; mi++)
#pragma unroll
        for (int ni = 0; ni < N_TILES; ni++)
        {
            acc[mi][ni][0] = 0u;
            acc[mi][ni][1] = 0u;
        }

    for (int bk = 0; bk < K; bk += BK)
    {
// === Load A tile [BM×BK] → smem ===
#pragma unroll
        for (int pass = 0; pass < (BM / rows_per_pass); pass++)
        {
            int row = pass * rows_per_pass + load_row_in_pass;
            int gm = bm + row;
            int gk = bk + load_col;

            uint4 val = make_uint4(0u, 0u, 0u, 0u);
            if (gm < M && gk + 16 <= K)
                val = __ldg((const uint4 *)&A[gm * K + gk]);
            *(uint4 *)&smem_A[row * SMEM_STRIDE + load_col] = val;
        }

// === Load B tile [BN×BK] → smem ===
#pragma unroll
        for (int pass = 0; pass < (BN / rows_per_pass); pass++)
        {
            int row = pass * rows_per_pass + load_row_in_pass;
            int gn = bn + row;
            int gk = bk + load_col;

            uint4 val = make_uint4(0u, 0u, 0u, 0u);
            if (gn < N && gk + 16 <= K)
                val = __ldg((const uint4 *)&B[gn * K + gk]);
            *(uint4 *)&smem_B[row * SMEM_STRIDE + load_col] = val;
        }

        __syncthreads();

        // === Compute: software-pipelined fragment loads ===
        // Prefetch fragments for ki=0
        uint32_t a_curr[M_TILES][4], b_curr[N_TILES][2];

#pragma unroll
        for (int mi = 0; mi < M_TILES; mi++)
        {
            int a_row = wm + mi * MMA_M;
            a_curr[mi][0] = *(uint32_t *)&smem_A[(a_row + group_id) * SMEM_STRIDE + tid * 4];
            a_curr[mi][1] = *(uint32_t *)&smem_A[(a_row + group_id + 8) * SMEM_STRIDE + tid * 4];
            a_curr[mi][2] = *(uint32_t *)&smem_A[(a_row + group_id) * SMEM_STRIDE + tid * 4 + 16];
            a_curr[mi][3] = *(uint32_t *)&smem_A[(a_row + group_id + 8) * SMEM_STRIDE + tid * 4 + 16];
        }
#pragma unroll
        for (int ni = 0; ni < N_TILES; ni++)
        {
            int b_row = wn + ni * MMA_N;
            b_curr[ni][0] = *(uint32_t *)&smem_B[(b_row + group_id) * SMEM_STRIDE + tid * 4];
            b_curr[ni][1] = *(uint32_t *)&smem_B[(b_row + group_id) * SMEM_STRIDE + tid * 4 + 16];
        }

#pragma unroll
        for (int ki = 0; ki < K_STEPS; ki++)
        {
            // Prefetch NEXT k-step fragments (overlaps with MMA)
            uint32_t a_next[M_TILES][4], b_next[N_TILES][2];
            if (ki + 1 < K_STEPS)
            {
                int k_next = (ki + 1) * MMA_K;
#pragma unroll
                for (int mi = 0; mi < M_TILES; mi++)
                {
                    int a_row = wm + mi * MMA_M;
                    a_next[mi][0] = *(uint32_t *)&smem_A[(a_row + group_id) * SMEM_STRIDE + k_next + tid * 4];
                    a_next[mi][1] = *(uint32_t *)&smem_A[(a_row + group_id + 8) * SMEM_STRIDE + k_next + tid * 4];
                    a_next[mi][2] = *(uint32_t *)&smem_A[(a_row + group_id) * SMEM_STRIDE + k_next + tid * 4 + 16];
                    a_next[mi][3] = *(uint32_t *)&smem_A[(a_row + group_id + 8) * SMEM_STRIDE + k_next + tid * 4 + 16];
                }
#pragma unroll
                for (int ni = 0; ni < N_TILES; ni++)
                {
                    int b_row = wn + ni * MMA_N;
                    b_next[ni][0] = *(uint32_t *)&smem_B[(b_row + group_id) * SMEM_STRIDE + k_next + tid * 4];
                    b_next[ni][1] = *(uint32_t *)&smem_B[(b_row + group_id) * SMEM_STRIDE + k_next + tid * 4 + 16];
                }
            }

// MMA on current fragments
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
#pragma unroll
                for (int ni = 0; ni < N_TILES; ni++)
                    mma_fp8(
                        a_curr[mi][0], a_curr[mi][1],
                        a_curr[mi][2], a_curr[mi][3],
                        b_curr[ni][0], b_curr[ni][1],
                        acc[mi][ni][0], acc[mi][ni][1]);

            // Swap: next → current
            if (ki + 1 < K_STEPS)
            {
#pragma unroll
                for (int mi = 0; mi < M_TILES; mi++)
                {
                    a_curr[mi][0] = a_next[mi][0];
                    a_curr[mi][1] = a_next[mi][1];
                    a_curr[mi][2] = a_next[mi][2];
                    a_curr[mi][3] = a_next[mi][3];
                }
#pragma unroll
                for (int ni = 0; ni < N_TILES; ni++)
                {
                    b_curr[ni][0] = b_next[ni][0];
                    b_curr[ni][1] = b_next[ni][1];
                }
            }
        }

        __syncthreads();
    }

// === Store C ===
#pragma unroll
    for (int mi = 0; mi < M_TILES; mi++)
    {
#pragma unroll
        for (int ni = 0; ni < N_TILES; ni++)
        {
            int row0 = bm + wm + mi * MMA_M + group_id;
            int row1 = row0 + 8;
            int col = bn + wn + ni * MMA_N + tid * 2;

            if (row0 < M && col + 1 < N)
                *(uint32_t *)&C[row0 * N + col] = acc[mi][ni][0];
            if (row1 < M && col + 1 < N)
                *(uint32_t *)&C[row1 * N + col] = acc[mi][ni][1];
        }
    }
}

extern "C" int fp8_gemm_f16acc(
    int M, int N, int K,
    const void *A, const void *B, void *C)
{
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(BLOCK_THREADS);
    fp8_gemm_f16acc_kernel<<<grid, block>>>(
        (const uint8_t *)A, (const uint8_t *)B, (uint16_t *)C, M, N, K);
    return (int)cudaGetLastError();
}

// =============================================================================
// Test / Bench infrastructure
// =============================================================================
static inline uint8_t float_to_e4m3(float f)
{
    if (f != f)
        return 0x7Fu;
    int sign = (f < 0.0f) ? 1 : 0;
    float af = fabsf(f);
    if (af > 448.0f)
        return sign ? 0xFEu : 0x7Eu;
    if (af < 1.953125e-3f)
        return sign ? 0x80u : 0x00u;
    int exp_unbiased = (int)floorf(log2f(af));
    float mant_f = af / ldexpf(1.0f, exp_unbiased) - 1.0f;
    int mant3 = (int)(mant_f * 8.0f + 0.5f);
    if (mant3 >= 8)
    {
        mant3 = 0;
        exp_unbiased++;
    }
    int exp_biased = exp_unbiased + 7;
    if (exp_biased < 1)
    {
        int mant_sub = (int)(af / ldexpf(1.0f, -9) + 0.5f);
        if (mant_sub > 7)
            mant_sub = 7;
        return (uint8_t)((sign << 7) | (mant_sub & 0x7));
    }
    if (exp_biased > 15)
        exp_biased = 15;
    return (uint8_t)((sign << 7) | (exp_biased << 3) | (mant3 & 0x7));
}

static inline float e4m3_to_float(uint8_t v)
{
    int sign = (v >> 7) & 1, exp = (v >> 3) & 0xF, mant = v & 0x7;
    if (exp == 0xF && mant == 0x7)
        return nanf("");
    float r = (exp == 0) ? ldexpf((float)mant, -9) : ldexpf(1.0f + mant / 8.0f, exp - 7);
    return sign ? -r : r;
}

static inline float fp16_to_float(uint16_t h)
{
    __half hv;
    memcpy(&hv, &h, 2);
    return __half2float(hv);
}

#define CHECK_CUDA(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                              \
        }                                                         \
    } while (0)

void test_identity(int N)
{
    printf("  Identity %d: ", N);
    int M = N, K = N;
    size_t sA = (size_t)M * K, sB = (size_t)N * K, sC = (size_t)M * N * 2;
    uint8_t *hA = (uint8_t *)calloc(sA, 1);
    uint8_t *hB = (uint8_t *)calloc(sB, 1);
    uint16_t *hC = (uint16_t *)malloc(sC);
    uint8_t one = float_to_e4m3(1.0f);
    for (int i = 0; i < N; i++)
        hA[i * K + i] = one;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < K; j++)
            hB[i * K + j] = float_to_e4m3((float)((i + j) % 7) - 3.0f);
    void *dA, *dB;
    uint16_t *dC;
    CHECK_CUDA(cudaMalloc(&dA, sA));
    CHECK_CUDA(cudaMalloc(&dB, sB));
    CHECK_CUDA(cudaMalloc(&dC, sC));
    CHECK_CUDA(cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sC));
    fp8_gemm_f16acc(M, N, K, dA, dB, dC);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hC, dC, sC, cudaMemcpyDeviceToHost));
    int errors = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            if (fabsf(fp16_to_float(hC[m * N + n]) - e4m3_to_float(hB[n * K + m])) > 0.125f)
                errors++;
    printf("%d/%d → %s\n", errors, M * N, errors == 0 ? "PASS" : "FAIL");
    if (errors > 0 && errors < 10)
    {
        for (int m = 0; m < 2; m++)
            for (int n = 0; n < 16; n++)
                printf("    [%d][%d] got=%.2f exp=%.2f\n", m, n,
                       fp16_to_float(hC[m * N + n]), e4m3_to_float(hB[n * K + m]));
    }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);
    free(hC);
}

void test_random(int M, int N, int K)
{
    printf("  Random %dx%dx%d: ", M, N, K);
    size_t sA = (size_t)M * K, sB = (size_t)N * K, sC = (size_t)M * N * 2;
    uint8_t *hA = (uint8_t *)malloc(sA);
    uint8_t *hB = (uint8_t *)malloc(sB);
    uint16_t *hC = (uint16_t *)malloc(sC);
    float *ref = (float *)malloc((size_t)M * N * 4);
    srand(42);
    for (size_t i = 0; i < sA; i++)
        hA[i] = float_to_e4m3(((float)(rand() % 16) - 8.0f) * 0.25f);
    for (size_t i = 0; i < sB; i++)
        hB[i] = float_to_e4m3(((float)(rand() % 16) - 8.0f) * 0.25f);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
        {
            float sum = 0;
            for (int k = 0; k < K; k++)
                sum += e4m3_to_float(hA[m * K + k]) * e4m3_to_float(hB[n * K + k]);
            ref[m * N + n] = sum;
        }
    void *dA, *dB;
    uint16_t *dC;
    CHECK_CUDA(cudaMalloc(&dA, sA));
    CHECK_CUDA(cudaMalloc(&dB, sB));
    CHECK_CUDA(cudaMalloc(&dC, sC));
    CHECK_CUDA(cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sC));
    fp8_gemm_f16acc(M, N, K, dA, dB, dC);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hC, dC, sC, cudaMemcpyDeviceToHost));
    int errors = 0;
    float max_abs = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
        {
            float gpu = fp16_to_float(hC[m * N + n]);
            float ae = fabsf(gpu - ref[m * N + n]);
            if (ae > max_abs)
                max_abs = ae;
            if (ae > fmaxf(1.0f, fabsf(ref[m * N + n]) * 0.05f))
                errors++;
        }
    printf("max_err=%.4f errors=%d/%d → %s\n", max_abs, errors, M * N, errors == 0 ? "PASS" : "FAIL");
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);
    free(hC);
    free(ref);
}

void bench(int M, int N, int K)
{
    void *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, (size_t)M * K));
    CHECK_CUDA(cudaMalloc(&dB, (size_t)N * K));
    CHECK_CUDA(cudaMalloc(&dC, (size_t)M * N * 2));
    CHECK_CUDA(cudaMemset(dA, 0x38, (size_t)M * K));
    CHECK_CUDA(cudaMemset(dB, 0x38, (size_t)N * K));
    for (int i = 0; i < 10; i++)
        fp8_gemm_f16acc(M, N, K, dA, dB, dC);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    int iters = 200;
    CHECK_CUDA(cudaEventRecord(t0));
    for (int i = 0; i < iters; i++)
        fp8_gemm_f16acc(M, N, K, dA, dB, dC);
    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
    double tflops = 2.0 * (double)M * (double)N * (double)K / (ms / iters / 1000.0) / 1e12;
    printf("  %5d x %5d x %5d  %7.3f ms  %7.1f TFLOPS", M, N, K, ms / iters, tflops);
    if (tflops > 600)
        printf("  *** >600! ***");
    else if (tflops > 500)
        printf("  ** >500 **");
    printf("\n");
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

int main()
{
    cudaDeviceProp p;
    CHECK_CUDA(cudaGetDeviceProperties(&p, 0));
    printf("=== FP8 GEMM F16-Acc v6 (BK=128, pipelined frags) ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n", p.name, p.multiProcessorCount, p.clockRate / 1000);
    printf("SMEM: %d bytes (A=%d + B=%d) < 48KB\n\n",
           BM * SMEM_STRIDE + BN * SMEM_STRIDE, BM * SMEM_STRIDE, BN * SMEM_STRIDE);

    printf("--- Correctness ---\n");
    test_identity(128);
    test_identity(256);
    test_identity(512);
    test_random(128, 128, 128);
    test_random(256, 256, 256);
    test_random(512, 512, 512);

    printf("\n--- Performance ---\n");
    for (int s : {1024, 2048, 4096, 8192})
        bench(s, s, s);
    printf("\n--- LLM shapes ---\n");
    bench(2048, 4096, 4096);
    bench(2048, 11008, 4096);
    bench(2048, 4096, 11008);
    bench(4096, 4096, 4096);
    bench(4096, 11008, 4096);
    bench(4096, 4096, 11008);

    printf("\nPeak: 660 | cuBLASLt: ~330 | v4: 473\n");
    return 0;
}
