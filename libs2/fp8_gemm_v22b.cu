// =============================================================================
// FP8 GEMM v22b — Forced 4 blocks/SM via maxrregcount=64
// =============================================================================
// Compile with: nvcc -O3 -arch=sm_89 --maxrregcount=64 -Xptxas -v ...
// This forces ALL kernels to use ≤64 regs → 65536/(64*256) = 4 blocks/SM
//
// Also includes singlesync (best from v22) as comparison baseline.
// singlesync will also be limited to 64 regs — interesting to see if
// 4 blocks helps it too.
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
#define SMEM_STRIDE 128
#define BLOCK_THREADS 256
#define MMA_M 16
#define MMA_N 8
#define MMA_K 32
#define K_STEPS 4
#define SMEM_PER_MAT (BM * SMEM_STRIDE)
#define SMEM_PER_BLOCK (2 * SMEM_PER_MAT)

__device__ __forceinline__ int swizzle16(int row, int col16)
{
    int chunk = col16 >> 4;
    int phys_chunk = chunk ^ (row & 7);
    return row * SMEM_STRIDE + (phys_chunk << 4);
}

__device__ __forceinline__ int swizzle4(int row, int col)
{
    int chunk = col >> 4;
    int within = col & 15;
    int phys_chunk = chunk ^ (row & 7);
    return row * SMEM_STRIDE + (phys_chunk << 4) + within;
}

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
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(c0), "r"(c1));
    c0 = d0;
    c1 = d1;
}

#define LOAD_TILE_TO_SMEM(smem_ptr, gmem_base, g_base, stride_g, g_limit, bk_offset) \
    do                                                                               \
    {                                                                                \
        const int thr_per_row = BK / 16;                                             \
        const int rows_per_pass = BLOCK_THREADS / thr_per_row;                       \
        const int load_row_in_pass = threadIdx.x / thr_per_row;                      \
        const int load_col = (threadIdx.x % thr_per_row) * 16;                       \
        _Pragma("unroll") for (int pass = 0; pass < 4; pass++)                       \
        {                                                                            \
            int row = pass * rows_per_pass + load_row_in_pass;                       \
            int gm = g_base + row;                                                   \
            int gk = bk_offset + load_col;                                           \
            uint4 val = make_uint4(0u, 0u, 0u, 0u);                                  \
            if (gm < g_limit && gk + 16 <= stride_g)                                 \
                val = __ldg((const uint4 *)&gmem_base[gm * stride_g + gk]);          \
            *(uint4 *)&smem_ptr[swizzle16(row, load_col)] = val;                     \
        }                                                                            \
    } while (0)

// =============================================================================
// Kernel 1: reg64 — v10b with forced 4 blocks/SM
// =============================================================================
extern __shared__ uint8_t dyn_smem_r64[];

__global__ void __launch_bounds__(BLOCK_THREADS, 4)
    kernel_reg64(
        const uint8_t *__restrict__ A,
        const uint8_t *__restrict__ B,
        uint16_t *__restrict__ C,
        int M, int N, int K)
{
    uint8_t *smem_A = dyn_smem_r64;
    uint8_t *smem_B = dyn_smem_r64 + SMEM_PER_MAT;

    const int bm = blockIdx.x * BM;
    const int bn = blockIdx.y * BN;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_m = warp_id / 4;
    const int warp_n = warp_id % 4;
    const int wm = warp_m * 64;
    const int wn = warp_n * 32;
    const int group_id = lane_id / 4;
    const int tid = lane_id % 4;

    uint32_t acc[4][4][2];
#pragma unroll
    for (int mi = 0; mi < 4; mi++)
#pragma unroll
        for (int ni = 0; ni < 4; ni++)
        {
            acc[mi][ni][0] = 0u;
            acc[mi][ni][1] = 0u;
        }

    for (int bk = 0; bk < K; bk += BK)
    {
        LOAD_TILE_TO_SMEM(smem_A, A, bm, K, M, bk);
        LOAD_TILE_TO_SMEM(smem_B, B, bn, K, N, bk);
        __syncthreads();

#pragma unroll
        for (int ki = 0; ki < K_STEPS; ki++)
        {
            int k_off = ki * MMA_K;
            uint32_t a_frag[4][4];
#pragma unroll
            for (int mi = 0; mi < 4; mi++)
            {
                int a_row = wm + mi * MMA_M;
                int col_lo = k_off + tid * 4;
                int col_hi = k_off + tid * 4 + 16;
                a_frag[mi][0] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id, col_lo)];
                a_frag[mi][1] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id + 8, col_lo)];
                a_frag[mi][2] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id, col_hi)];
                a_frag[mi][3] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id + 8, col_hi)];
            }
            uint32_t b_frag[4][2];
#pragma unroll
            for (int ni = 0; ni < 4; ni++)
            {
                int b_row = wn + ni * MMA_N;
                int col_lo = k_off + tid * 4;
                int col_hi = k_off + tid * 4 + 16;
                b_frag[ni][0] = *(uint32_t *)&smem_B[swizzle4(b_row + group_id, col_lo)];
                b_frag[ni][1] = *(uint32_t *)&smem_B[swizzle4(b_row + group_id, col_hi)];
            }
#pragma unroll
            for (int mi = 0; mi < 4; mi++)
#pragma unroll
                for (int ni = 0; ni < 4; ni++)
                    mma_fp8(a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                            b_frag[ni][0], b_frag[ni][1], acc[mi][ni][0], acc[mi][ni][1]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int mi = 0; mi < 4; mi++)
#pragma unroll
        for (int ni = 0; ni < 4; ni++)
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

// =============================================================================
// Kernel 2: singlesync with reg64 — paired K-steps + forced 4 blocks
// =============================================================================
extern __shared__ uint8_t dyn_smem_ss64[];

__global__ void __launch_bounds__(BLOCK_THREADS, 4)
    kernel_singlesync_reg64(
        const uint8_t *__restrict__ A,
        const uint8_t *__restrict__ B,
        uint16_t *__restrict__ C,
        int M, int N, int K)
{
    uint8_t *smem_A = dyn_smem_ss64;
    uint8_t *smem_B = dyn_smem_ss64 + SMEM_PER_MAT;

    const int bm = blockIdx.x * BM;
    const int bn = blockIdx.y * BN;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_m = warp_id / 4;
    const int warp_n = warp_id % 4;
    const int wm = warp_m * 64;
    const int wn = warp_n * 32;
    const int group_id = lane_id / 4;
    const int tid = lane_id % 4;

    uint32_t acc[4][4][2];
#pragma unroll
    for (int mi = 0; mi < 4; mi++)
#pragma unroll
        for (int ni = 0; ni < 4; ni++)
        {
            acc[mi][ni][0] = 0u;
            acc[mi][ni][1] = 0u;
        }

    for (int bk = 0; bk < K; bk += BK)
    {
        LOAD_TILE_TO_SMEM(smem_A, A, bm, K, M, bk);
        LOAD_TILE_TO_SMEM(smem_B, B, bn, K, N, bk);
        __syncthreads();

#pragma unroll
        for (int kp = 0; kp < K_STEPS; kp += 2)
        {
            uint32_t a_frag0[4][4], a_frag1[4][4];
            uint32_t b_frag0[4][2], b_frag1[4][2];

            int k_off0 = kp * MMA_K;
            int k_off1 = (kp + 1) * MMA_K;

#pragma unroll
            for (int mi = 0; mi < 4; mi++)
            {
                int a_row = wm + mi * MMA_M;
                a_frag0[mi][0] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id, k_off0 + tid * 4)];
                a_frag0[mi][1] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id + 8, k_off0 + tid * 4)];
                a_frag0[mi][2] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id, k_off0 + tid * 4 + 16)];
                a_frag0[mi][3] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id + 8, k_off0 + tid * 4 + 16)];
                a_frag1[mi][0] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id, k_off1 + tid * 4)];
                a_frag1[mi][1] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id + 8, k_off1 + tid * 4)];
                a_frag1[mi][2] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id, k_off1 + tid * 4 + 16)];
                a_frag1[mi][3] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id + 8, k_off1 + tid * 4 + 16)];
            }

#pragma unroll
            for (int ni = 0; ni < 4; ni++)
            {
                int b_row = wn + ni * MMA_N;
                b_frag0[ni][0] = *(uint32_t *)&smem_B[swizzle4(b_row + group_id, k_off0 + tid * 4)];
                b_frag0[ni][1] = *(uint32_t *)&smem_B[swizzle4(b_row + group_id, k_off0 + tid * 4 + 16)];
                b_frag1[ni][0] = *(uint32_t *)&smem_B[swizzle4(b_row + group_id, k_off1 + tid * 4)];
                b_frag1[ni][1] = *(uint32_t *)&smem_B[swizzle4(b_row + group_id, k_off1 + tid * 4 + 16)];
            }

#pragma unroll
            for (int mi = 0; mi < 4; mi++)
#pragma unroll
                for (int ni = 0; ni < 4; ni++)
                    mma_fp8(a_frag0[mi][0], a_frag0[mi][1], a_frag0[mi][2], a_frag0[mi][3],
                            b_frag0[ni][0], b_frag0[ni][1], acc[mi][ni][0], acc[mi][ni][1]);

#pragma unroll
            for (int mi = 0; mi < 4; mi++)
#pragma unroll
                for (int ni = 0; ni < 4; ni++)
                    mma_fp8(a_frag1[mi][0], a_frag1[mi][1], a_frag1[mi][2], a_frag1[mi][3],
                            b_frag1[ni][0], b_frag1[ni][1], acc[mi][ni][0], acc[mi][ni][1]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int mi = 0; mi < 4; mi++)
#pragma unroll
        for (int ni = 0; ni < 4; ni++)
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

// =============================================================================
// Test infrastructure
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

typedef void (*kernel_fn)(const uint8_t *, const uint8_t *, uint16_t *, int, int, int);
struct KernelInfo
{
    const char *name;
    kernel_fn fn;
    int smem_bytes;
    int target_blocks;
};

void launch_kernel(const KernelInfo &ki, int M, int N, int K,
                   const void *dA, const void *dB, void *dC)
{
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(BLOCK_THREADS);
    ki.fn<<<grid, block, ki.smem_bytes>>>((const uint8_t *)dA, (const uint8_t *)dB, (uint16_t *)dC, M, N, K);
}

bool test_correctness(const KernelInfo &ki)
{
    int M = 512, N = 512, K = 512;
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
    uint16_t *dC_d;
    CK(cudaMalloc(&dA, sA));
    CK(cudaMalloc(&dB, sB));
    CK(cudaMalloc(&dC_d, sC));
    CK(cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice));
    CK(cudaMemset(dC_d, 0, sC));
    cudaFuncSetAttribute(ki.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, ki.smem_bytes);
    launch_kernel(ki, M, N, K, dA, dB, dC_d);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hC, dC_d, sC, cudaMemcpyDeviceToHost));
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
    printf("  %-16s 512³: max_err=%.4f err=%d → %s\n", ki.name, mx, err, err == 0 ? "PASS" : "FAIL");
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC_d);
    free(hA);
    free(hB);
    free(hC);
    free(ref);
    return err == 0;
}

double bench_kernel(const KernelInfo &ki, int M, int N, int K)
{
    void *dA, *dB, *dC;
    CK(cudaMalloc(&dA, (size_t)M * K));
    CK(cudaMalloc(&dB, (size_t)N * K));
    CK(cudaMalloc(&dC, (size_t)M * N * 2));
    CK(cudaMemset(dA, 0x38, (size_t)M * K));
    CK(cudaMemset(dB, 0x38, (size_t)N * K));
    cudaFuncSetAttribute(ki.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, ki.smem_bytes);
    for (int i = 0; i < 10; i++)
        launch_kernel(ki, M, N, K, dA, dB, dC);
    CK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    int it = 200;
    CK(cudaEventRecord(t0));
    for (int i = 0; i < it; i++)
        launch_kernel(ki, M, N, K, dA, dB, dC);
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));
    float ms;
    CK(cudaEventElapsedTime(&ms, t0, t1));
    double tf = 2.0 * (double)M * (double)N * (double)K / (ms / it / 1000.0) / 1e12;
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return tf;
}

int main()
{
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("=== FP8 GEMM v22b: Forced maxrregcount=64 → 4 blocks/SM ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n", p.name, p.multiProcessorCount, p.clockRate / 1000);
    printf("Compiled with --maxrregcount=64\n\n");

    KernelInfo kernels[] = {
        {"reg64", kernel_reg64, SMEM_PER_BLOCK, 4},
        {"ss+reg64", kernel_singlesync_reg64, SMEM_PER_BLOCK, 4},
    };
    const int nk = 2;

    printf("--- Occupancy ---\n");
    for (int i = 0; i < nk; i++)
    {
        cudaFuncSetAttribute(kernels[i].fn,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kernels[i].smem_bytes);
        int occ;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,
                                                      kernels[i].fn, BLOCK_THREADS, kernels[i].smem_bytes);
        printf("  %-16s: %d blocks/SM (target: %d) %s\n",
               kernels[i].name, occ, kernels[i].target_blocks,
               occ >= kernels[i].target_blocks ? "✓" : "✗ MISSED");
    }

    printf("--- Correctness ---\n");
    for (int i = 0; i < nk; i++)
        test_correctness(kernels[i]);

    printf("--- Performance (TFLOPS) ---\n");
    struct
    {
        int M, N, K;
        const char *label;
    } sizes[] = {
        {1024, 1024, 1024, "1K³"},
        {2048, 2048, 2048, "2K³"},
        {4096, 4096, 4096, "4K³"},
        {8192, 8192, 8192, "8K³"},
        {4096, 11008, 4096, "4Kx11Kx4K"},
        {4096, 4096, 11008, "4Kx4Kx11K"},
        {8192, 4096, 4096, "8Kx4Kx4K"},
        {8192, 11008, 4096, "8Kx11Kx4K"},
    };

    printf("%-14s %10s %10s   v10b@OC\n", "Size", "reg64", "ss+reg64");
    printf("--------------------------------------------------\n");
    for (auto &s : sizes)
    {
        printf("%-14s", s.label);
        for (int i = 0; i < nk; i++)
        {
            double tf = bench_kernel(kernels[i], s.M, s.N, s.K);
            printf(" %10.1f", tf);
        }
        printf("   (~588)\n");
    }

    printf("\nBaseline: v10b@OC=588, singlesync@3blk=590\n");
    printf("If 4 blocks > 590: occupancy wins over spill cost\n");
    printf("If 4 blocks < 590: spill cost too high\n");
    return 0;
}
