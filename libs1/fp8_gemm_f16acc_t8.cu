// =============================================================================
// FP8 GEMM with FP16 Accumulator — v18
// =============================================================================
// BK=96, stride=96, 128×128 output tile, targeting 4 blocks/SM
//
// Trade-offs vs v10b (BK=128, 3 blocks):
//   + 4 blocks/SM (32 warps, 66.7% occupancy vs 50%)
//   - K/96 vs K/128 tiles → 33% more barriers
//   - 2-way bank conflicts (stride 96: bank shift 24/row → groups 0,4 conflict)
//   - K_STEPS=3 vs 4 (but 33% more tiles compensates)
//
// SMEM: 128×96×2 = 24576/block → 4×24576 = 98304 ≤ 102400 ✓
// Regs: launch_bounds(256,4) → ≤64 regs/thread
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
#define BK 96
#define SMEM_STRIDE 96
#define BLOCK_THREADS 256
#define WARPS_M 2
#define WARPS_N 4
#define WM 64
#define WN 32
#define MMA_M 16
#define MMA_N 8
#define MMA_K 32
#define M_TILES 4
#define N_TILES 4
#define K_STEPS 3 // 96/32

#define SMEM_PER_MAT (BM * SMEM_STRIDE)   // 12288
#define SMEM_PER_BLOCK (2 * SMEM_PER_MAT) // 24576

// No swizzle — stride=96 not power-of-2, accept 2-way bank conflicts
// Bank analysis: stride=96/4=24 words. Groups 0-7 access banks:
//   0, 24, 16, 8, 0, 24, 16, 8 mod 32 → 2-way conflicts (pairs 0-4, 1-5, 2-6, 3-7)

__device__ __forceinline__ void mma_fp8(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, uint32_t &c0, uint32_t &c1)
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

extern __shared__ uint8_t dyn_smem[];

__global__ void __launch_bounds__(BLOCK_THREADS, 4)
    fp8_gemm_f16acc_kernel(
        const uint8_t *__restrict__ A,
        const uint8_t *__restrict__ B,
        uint16_t *__restrict__ C,
        int M, int N, int K)
{
    uint8_t *smem_A = dyn_smem;
    uint8_t *smem_B = dyn_smem + SMEM_PER_MAT;

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

    // Load geometry: BK=96, 256 threads × 16B = 4096B/pass
    // 128×96 = 12288B → 3 passes per matrix
    const int thr_per_row = BK / 16;                       // 6
    const int rows_per_pass = BLOCK_THREADS / thr_per_row; // 42 (with 4 leftover threads)
    // Handle non-even division: 256/6 = 42 rem 4
    // Use: rows_per_pass = 42, threads 0..251 load, threads 252..255 idle per pass
    const int my_load_idx = threadIdx.x;
    const int load_row_in_pass = my_load_idx / thr_per_row;
    const int load_col = (my_load_idx % thr_per_row) * 16;
    const bool load_active = (my_load_idx < rows_per_pass * thr_per_row); // 252

    // For 128 rows: need ceil(128/42) = 4 passes (42+42+42+2)
    // Actually let's use simpler approach: each thread loads one 16B chunk
    // Total chunks = 128 * 6 = 768. 256 threads → 3 passes of 256 chunks = 768. EXACT!
    // chunk_id (0..767) → row = chunk_id / 6, col = (chunk_id % 6) * 16

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
// Load A: 768 chunks / 256 threads = 3 passes
#pragma unroll
        for (int pass = 0; pass < 3; pass++)
        {
            int chunk_id = pass * BLOCK_THREADS + threadIdx.x;
            int row = chunk_id / 6; // thr_per_row=6
            int col = (chunk_id % 6) * 16;
            int gm = bm + row;
            int gk = bk + col;
            uint4 val = make_uint4(0u, 0u, 0u, 0u);
            if (row < BM && gm < M && gk + 16 <= K)
                val = __ldg((const uint4 *)&A[gm * K + gk]);
            *(uint4 *)&smem_A[row * SMEM_STRIDE + col] = val;
        }
// Load B: same pattern
#pragma unroll
        for (int pass = 0; pass < 3; pass++)
        {
            int chunk_id = pass * BLOCK_THREADS + threadIdx.x;
            int row = chunk_id / 6;
            int col = (chunk_id % 6) * 16;
            int gn = bn + row;
            int gk = bk + col;
            uint4 val = make_uint4(0u, 0u, 0u, 0u);
            if (row < BN && gn < N && gk + 16 <= K)
                val = __ldg((const uint4 *)&B[gn * K + gk]);
            *(uint4 *)&smem_B[row * SMEM_STRIDE + col] = val;
        }

        __syncthreads();

// Compute: K_STEPS=3 (BK=96, MMA_K=32)
#pragma unroll
        for (int ki = 0; ki < K_STEPS; ki++)
        {
            int k_off = ki * MMA_K;

            uint32_t a_frag[M_TILES][4];
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                int a_row = wm + mi * MMA_M;
                int col_lo = k_off + tid * 4;
                int col_hi = col_lo + 16;
                a_frag[mi][0] = *(uint32_t *)&smem_A[(a_row + group_id) * SMEM_STRIDE + col_lo];
                a_frag[mi][1] = *(uint32_t *)&smem_A[(a_row + group_id + 8) * SMEM_STRIDE + col_lo];
                a_frag[mi][2] = *(uint32_t *)&smem_A[(a_row + group_id) * SMEM_STRIDE + col_hi];
                a_frag[mi][3] = *(uint32_t *)&smem_A[(a_row + group_id + 8) * SMEM_STRIDE + col_hi];
            }

            uint32_t b_frag[N_TILES][2];
#pragma unroll
            for (int ni = 0; ni < N_TILES; ni++)
            {
                int b_row = wn + ni * MMA_N;
                int col_lo = k_off + tid * 4;
                int col_hi = col_lo + 16;
                b_frag[ni][0] = *(uint32_t *)&smem_B[(b_row + group_id) * SMEM_STRIDE + col_lo];
                b_frag[ni][1] = *(uint32_t *)&smem_B[(b_row + group_id) * SMEM_STRIDE + col_hi];
            }

#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
#pragma unroll
                for (int ni = 0; ni < N_TILES; ni++)
                    mma_fp8(a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                            b_frag[ni][0], b_frag[ni][1], acc[mi][ni][0], acc[mi][ni][1]);
        }

        __syncthreads();
    }

// Store C
#pragma unroll
    for (int mi = 0; mi < M_TILES; mi++)
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

static bool smem_configured = false;

extern "C" int fp8_gemm_f16acc(
    int M, int N, int K,
    const void *A, const void *B, void *C)
{
    if (!smem_configured)
    {
        cudaFuncSetAttribute(fp8_gemm_f16acc_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_PER_BLOCK);
        smem_configured = true;
    }
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(BLOCK_THREADS);
    fp8_gemm_f16acc_kernel<<<grid, block, SMEM_PER_BLOCK>>>(
        (const uint8_t *)A, (const uint8_t *)B, (uint16_t *)C, M, N, K);
    return (int)cudaGetLastError();
}

// =============================================================================
// Tests
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
    int eu = (int)floorf(log2f(af));
    float mf = af / ldexpf(1.0f, eu) - 1.0f;
    int m3 = (int)(mf * 8.0f + 0.5f);
    if (m3 >= 8)
    {
        m3 = 0;
        eu++;
    }
    int eb = eu + 7;
    if (eb < 1)
    {
        int ms = (int)(af / ldexpf(1.0f, -9) + 0.5f);
        if (ms > 7)
            ms = 7;
        return (uint8_t)((sign << 7) | (ms & 7));
    }
    if (eb > 15)
        eb = 15;
    return (uint8_t)((sign << 7) | (eb << 3) | (m3 & 7));
}
static inline float e4m3_to_float(uint8_t v)
{
    int s = (v >> 7) & 1, e = (v >> 3) & 0xF, m = v & 7;
    if (e == 0xF && m == 7)
        return nanf("");
    float r = (e == 0) ? ldexpf((float)m, -9) : ldexpf(1.0f + m / 8.0f, e - 7);
    return s ? -r : r;
}
static inline float fp16f(uint16_t h)
{
    __half hv;
    memcpy(&hv, &h, 2);
    return __half2float(hv);
}
#define CK(c)                                                                               \
    do                                                                                      \
    {                                                                                       \
        cudaError_t e = (c);                                                                \
        if (e != cudaSuccess)                                                               \
        {                                                                                   \
            fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(1);                                                                        \
        }                                                                                   \
    } while (0)

void test_identity(int N)
{
    printf("  Identity %d: ", N);
    int M = N, K = N;
    size_t sA = (size_t)M * K, sB = (size_t)N * K, sC = (size_t)M * N * 2;
    uint8_t *hA = (uint8_t *)calloc(sA, 1), *hB = (uint8_t *)calloc(sB, 1);
    uint16_t *hC = (uint16_t *)malloc(sC);
    uint8_t one = float_to_e4m3(1.0f);
    for (int i = 0; i < N; i++)
        hA[i * K + i] = one;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < K; j++)
            hB[i * K + j] = float_to_e4m3((float)((i + j) % 7) - 3.0f);
    void *dA, *dB;
    uint16_t *dC;
    CK(cudaMalloc(&dA, sA));
    CK(cudaMalloc(&dB, sB));
    CK(cudaMalloc(&dC, sC));
    CK(cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice));
    CK(cudaMemset(dC, 0, sC));
    fp8_gemm_f16acc(M, N, K, dA, dB, dC);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hC, dC, sC, cudaMemcpyDeviceToHost));
    int err = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            if (fabsf(fp16f(hC[m * N + n]) - e4m3_to_float(hB[n * K + m])) > 0.125f)
                err++;
    printf("%d/%d → %s\n", err, M * N, err == 0 ? "PASS" : "FAIL");
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
    uint8_t *hA = (uint8_t *)malloc(sA), *hB = (uint8_t *)malloc(sB);
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
            float s = 0;
            for (int k = 0; k < K; k++)
                s += e4m3_to_float(hA[m * K + k]) * e4m3_to_float(hB[n * K + k]);
            ref[m * N + n] = s;
        }
    void *dA, *dB;
    uint16_t *dC;
    CK(cudaMalloc(&dA, sA));
    CK(cudaMalloc(&dB, sB));
    CK(cudaMalloc(&dC, sC));
    CK(cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice));
    CK(cudaMemset(dC, 0, sC));
    fp8_gemm_f16acc(M, N, K, dA, dB, dC);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hC, dC, sC, cudaMemcpyDeviceToHost));
    int err = 0;
    float mx = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
        {
            float ae = fabsf(fp16f(hC[m * N + n]) - ref[m * N + n]);
            if (ae > mx)
                mx = ae;
            if (ae > fmaxf(1.0f, fabsf(ref[m * N + n]) * 0.05f))
                err++;
        }
    printf("max_err=%.4f errors=%d/%d → %s\n", mx, err, M * N, err == 0 ? "PASS" : "FAIL");
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
    CK(cudaMalloc(&dA, (size_t)M * K));
    CK(cudaMalloc(&dB, (size_t)N * K));
    CK(cudaMalloc(&dC, (size_t)M * N * 2));
    CK(cudaMemset(dA, 0x38, (size_t)M * K));
    CK(cudaMemset(dB, 0x38, (size_t)N * K));
    for (int i = 0; i < 10; i++)
        fp8_gemm_f16acc(M, N, K, dA, dB, dC);
    CK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    int it = 200;
    CK(cudaEventRecord(t0));
    for (int i = 0; i < it; i++)
        fp8_gemm_f16acc(M, N, K, dA, dB, dC);
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));
    float ms;
    CK(cudaEventElapsedTime(&ms, t0, t1));
    double tf = 2.0 * (double)M * (double)N * (double)K / (ms / it / 1000.0) / 1e12;
    printf("  %5d x %5d x %5d  %7.3f ms  %7.1f TFLOPS", M, N, K, ms / it, tf);
    if (tf > 600)
        printf("  *** >600! ***");
    else if (tf > 550)
        printf("  ** >550 **");
    else if (tf > 500)
        printf("  * >500 *");
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
    CK(cudaGetDeviceProperties(&p, 0));
    printf("=== FP8 GEMM F16-Acc v18 (BK=96, 4 blocks/SM) ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n", p.name, p.multiProcessorCount, p.clockRate / 1000);
    printf("SMEM: %d B/block\n", SMEM_PER_BLOCK);
    int occ;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,
                                                  fp8_gemm_f16acc_kernel, BLOCK_THREADS, SMEM_PER_BLOCK);
    printf("Blocks/SM: %d (target: 4)\n\n", occ);

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
    printf("\n--- LLM ---\n");
    bench(2048, 4096, 4096);
    bench(2048, 11008, 4096);
    bench(2048, 4096, 11008);
    bench(4096, 4096, 4096);
    bench(4096, 11008, 4096);
    bench(4096, 4096, 11008);
    bench(8192, 4096, 4096);
    bench(8192, 11008, 4096);

    printf("\nPeak:723 (@2760MHz) | cuBLAS:~330 | v10b:546\n");
    return 0;
}
