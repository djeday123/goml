// =============================================================================
// FP8 GEMM with FP16 Accumulator — v2 fixed (SM89 A-fragment order)
// =============================================================================
// Fix: SM89 QMMA expects A registers as {row_lo/k_lo, row_hi/k_lo, row_lo/k_hi, row_hi/k_hi}
//      NOT PTX-documented {row_lo/k_lo, row_lo/k_hi, row_hi/k_lo, row_hi/k_hi}
// Output D mapping: standard tid*2, tid*2+1 (confirmed by probe)
//
// Build:
//   nvcc -O3 -arch=sm_89 -std=c++17 fp8_gemm_f16acc_v2.cu -o fp8_gemm -lcudart
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
#define BK 64
#define SMEM_STRIDE 80
#define WARPS_M 2
#define WARPS_N 4
#define BLOCK_THREADS (WARPS_M * WARPS_N * 32)
#define WM (BM / WARPS_M)
#define WN (BN / WARPS_N)
#define MMA_M 16
#define MMA_N 8
#define MMA_K 32
#define M_TILES (WM / MMA_M)
#define N_TILES (WN / MMA_N)
#define K_STEPS (BK / MMA_K)

// =============================================================================
// MMA wrapper — in-place accumulation
// =============================================================================
__device__ __forceinline__ void mma_fp8_f16acc(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t &c0, uint32_t &c1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0, %1}, "
        "{%2, %3, %4, %5}, "
        "{%6, %7}, "
        "{%0, %1};\n"
        : "+r"(c0), "+r"(c1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1));
}

// =============================================================================
// GEMM Kernel
// =============================================================================
__global__ void __launch_bounds__(BLOCK_THREADS)
    fp8_gemm_f16acc_kernel(
        const uint8_t *__restrict__ A,
        const uint8_t *__restrict__ B,
        uint16_t *__restrict__ C,
        int M, int N, int K)
{
    __shared__ uint8_t smem_A[BM * SMEM_STRIDE];
    __shared__ uint8_t smem_B[BN * SMEM_STRIDE];

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

    const int load_row = threadIdx.x / 4;
    const int load_col = (threadIdx.x % 4) * 16;

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
// Load A
#pragma unroll
        for (int chunk = 0; chunk < 2; chunk++)
        {
            int row = chunk * 64 + load_row;
            int gm = bm + row;
            int gk = bk + load_col;
            uint4 val = make_uint4(0u, 0u, 0u, 0u);
            if (gm < M && gk + 16 <= K)
                val = __ldg((const uint4 *)&A[gm * K + gk]);
            else if (gm < M)
            {
                uint8_t tmp[16] = {};
                for (int b = 0; b < 16 && gk + b < K; b++)
                    tmp[b] = A[gm * K + gk + b];
                val = *(uint4 *)tmp;
            }
            *(uint4 *)&smem_A[row * SMEM_STRIDE + load_col] = val;
        }

// Load B
#pragma unroll
        for (int chunk = 0; chunk < 2; chunk++)
        {
            int row = chunk * 64 + load_row;
            int gn = bn + row;
            int gk = bk + load_col;
            uint4 val = make_uint4(0u, 0u, 0u, 0u);
            if (gn < N && gk + 16 <= K)
                val = __ldg((const uint4 *)&B[gn * K + gk]);
            else if (gn < N)
            {
                uint8_t tmp[16] = {};
                for (int b = 0; b < 16 && gk + b < K; b++)
                    tmp[b] = B[gn * K + gk + b];
                val = *(uint4 *)tmp;
            }
            *(uint4 *)&smem_B[row * SMEM_STRIDE + load_col] = val;
        }

        __syncthreads();

// Compute
#pragma unroll
        for (int ki = 0; ki < K_STEPS; ki++)
        {
            int k_off = ki * MMA_K;

            uint32_t a_frag[M_TILES][4];
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                int a_row = wm + mi * MMA_M;
                // SM89 order: {row_lo/k_lo, row_hi/k_lo, row_lo/k_hi, row_hi/k_hi}
                a_frag[mi][0] = *(uint32_t *)&smem_A[(a_row + group_id) * SMEM_STRIDE + k_off + tid * 4];
                a_frag[mi][1] = *(uint32_t *)&smem_A[(a_row + group_id + 8) * SMEM_STRIDE + k_off + tid * 4];
                a_frag[mi][2] = *(uint32_t *)&smem_A[(a_row + group_id) * SMEM_STRIDE + k_off + tid * 4 + 16];
                a_frag[mi][3] = *(uint32_t *)&smem_A[(a_row + group_id + 8) * SMEM_STRIDE + k_off + tid * 4 + 16];
            }

            uint32_t b_frag[N_TILES][2];
#pragma unroll
            for (int ni = 0; ni < N_TILES; ni++)
            {
                int b_row = wn + ni * MMA_N;
                b_frag[ni][0] = *(uint32_t *)&smem_B[(b_row + group_id) * SMEM_STRIDE + k_off + tid * 4];
                b_frag[ni][1] = *(uint32_t *)&smem_B[(b_row + group_id) * SMEM_STRIDE + k_off + tid * 4 + 16];
            }

#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
#pragma unroll
                for (int ni = 0; ni < N_TILES; ni++)
                    mma_fp8_f16acc(
                        a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                        b_frag[ni][0], b_frag[ni][1],
                        acc[mi][ni][0], acc[mi][ni][1]);
        }

        __syncthreads();
    }

    // Store C — standard tid*2 mapping (probe confirmed)
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
// FP8 helpers
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
    int sign = (v >> 7) & 1;
    int exp = (v >> 3) & 0xF;
    int mant = v & 0x7;
    if (exp == 0xF && mant == 0x7)
        return nanf("");
    float result;
    if (exp == 0)
        result = ldexpf((float)mant, -9);
    else
        result = ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
    return sign ? -result : result;
}

static inline float fp16_to_float(uint16_t h)
{
    __half hv;
    memcpy(&hv, &h, 2);
    return __half2float(hv);
}

#define CHECK_CUDA(call)                                                                      \
    do                                                                                        \
    {                                                                                         \
        cudaError_t err = (call);                                                             \
        if (err)                                                                              \
        {                                                                                     \
            fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                                                          \
        }                                                                                     \
    } while (0)

// =============================================================================
// Tests
// =============================================================================
void test_identity(int N)
{
    printf("\n--- Identity Test (N=%d) ---\n", N);
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
        {
            float expected = e4m3_to_float(hB[n * K + m]);
            float got = fp16_to_float(hC[m * N + n]);
            if (fabsf(got - expected) > 0.125f)
            {
                if (errors < 10)
                    printf("  ERR [%d][%d]: got=%.4f exp=%.4f\n", m, n, got, expected);
                errors++;
            }
        }
    printf("  Mismatches: %d / %d → %s\n", errors, M * N, errors == 0 ? "PASS" : "FAIL");

    if (N >= 8)
    {
        printf("  C[0][0:8]: ");
        for (int j = 0; j < 8; j++)
            printf("%.1f ", fp16_to_float(hC[j]));
        printf("\n  Expected:  ");
        for (int j = 0; j < 8; j++)
            printf("%.1f ", e4m3_to_float(hB[j * K + 0]));
        printf("\n");
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);
    free(hC);
}

void test_ones(int M, int N, int K)
{
    printf("\n--- Ones Test (%dx%dx%d) ---\n", M, N, K);
    size_t sA = (size_t)M * K, sB = (size_t)N * K, sC = (size_t)M * N;
    void *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, sA));
    CHECK_CUDA(cudaMalloc(&dB, sB));
    CHECK_CUDA(cudaMalloc(&dC, sC * 2));
    CHECK_CUDA(cudaMemset(dA, 0x38, sA));
    CHECK_CUDA(cudaMemset(dB, 0x38, sB));
    CHECK_CUDA(cudaMemset(dC, 0, sC * 2));

    fp8_gemm_f16acc(M, N, K, dA, dB, dC);
    CHECK_CUDA(cudaDeviceSynchronize());

    uint16_t *hC = (uint16_t *)malloc(sC * 2);
    CHECK_CUDA(cudaMemcpy(hC, dC, sC * 2, cudaMemcpyDeviceToHost));

    int errors = 0;
    float expected = (float)K;
    for (int i = 0; i < M * N; i++)
    {
        float got = fp16_to_float(hC[i]);
        if (fabsf(got - expected) > 1.0f)
        {
            if (errors < 5)
                printf("  ERR [%d][%d]: got=%.1f exp=%.1f\n", i / N, i % N, got, expected);
            errors++;
        }
    }
    printf("  Errors: %d / %d → %s\n", errors, M * N, errors == 0 ? "PASS" : "FAIL");

    free(hC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

void test_random(int M, int N, int K)
{
    printf("\n--- Random Test (%dx%dx%d) ---\n", M, N, K);
    size_t sA = (size_t)M * K, sB = (size_t)N * K, sC = (size_t)M * N;

    uint8_t *hA = (uint8_t *)malloc(sA);
    uint8_t *hB = (uint8_t *)malloc(sB);
    uint16_t *hC = (uint16_t *)malloc(sC * 2);
    float *ref = (float *)malloc(sC * 4);

    srand(42);
    for (size_t i = 0; i < sA; i++)
        hA[i] = float_to_e4m3(((float)(rand() % 16) - 8) * 0.25f);
    for (size_t i = 0; i < sB; i++)
        hB[i] = float_to_e4m3(((float)(rand() % 16) - 8) * 0.25f);

    printf("  CPU reference...\n");
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
    CHECK_CUDA(cudaMalloc(&dC, sC * 2));
    CHECK_CUDA(cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sC * 2));

    fp8_gemm_f16acc(M, N, K, dA, dB, dC);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hC, dC, sC * 2, cudaMemcpyDeviceToHost));

    int errors = 0;
    float max_abs = 0;
    for (int i = 0; i < M * N; i++)
    {
        float gpu = fp16_to_float(hC[i]);
        float ae = fabsf(gpu - ref[i]);
        if (ae > max_abs)
            max_abs = ae;
        float tol = fmaxf(2.0f, fabsf(ref[i]) * 0.05f);
        if (ae > tol)
        {
            if (errors < 10)
                printf("  ERR [%d][%d]: gpu=%.2f ref=%.2f err=%.2f\n", i / N, i % N, gpu, ref[i], ae);
            errors++;
        }
    }
    printf("  Max abs err: %.4f\n", max_abs);
    printf("  Errors: %d / %d → %s\n", errors, M * N, errors == 0 ? "PASS" : "FAIL");

    if (errors > 0 && M >= 4 && N >= 4)
    {
        printf("  GPU [0:4][0:8]:\n");
        for (int m = 0; m < 4; m++)
        {
            printf("   ");
            for (int n = 0; n < (N < 8 ? N : 8); n++)
                printf(" %7.2f", fp16_to_float(hC[m * N + n]));
            printf("\n");
        }
        printf("  REF [0:4][0:8]:\n");
        for (int m = 0; m < 4; m++)
        {
            printf("   ");
            for (int n = 0; n < (N < 8 ? N : 8); n++)
                printf(" %7.2f", ref[m * N + n]);
            printf("\n");
        }
    }

    free(hA);
    free(hB);
    free(hC);
    free(ref);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
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
    CHECK_CUDA(cudaEventRecord(t0));
    for (int i = 0; i < 100; i++)
        fp8_gemm_f16acc(M, N, K, dA, dB, dC);
    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
    printf("  %5d x %5d x %5d  %7.3f ms  %7.1f TFLOPS\n",
           M, N, K, ms / 100, 2.0 * (double)M * N * K / (ms / 100 / 1000.0) / 1e12);

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
    printf("=== FP8 GEMM F16-Acc v2 fixed (SM89 A-frag order) ===\n");
    printf("GPU: %s (%d SMs)\n", p.name, p.multiProcessorCount);

    test_ones(128, 128, 64);
    test_ones(128, 128, 128);
    test_identity(128);
    test_identity(256);
    test_random(128, 128, 64);
    test_random(128, 128, 128);
    test_random(256, 256, 256);

    printf("\n--- Perf ---\n");
    for (int s : {1024, 2048, 4096, 8192})
        bench(s, s, s);
    printf("\n--- LLM ---\n");
    bench(2048, 4096, 4096);
    bench(2048, 11008, 4096);
    bench(2048, 4096, 11008);

    return 0;
}
