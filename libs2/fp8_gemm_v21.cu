// =============================================================================
// FP8 GEMM v21 — ldmatrix vs scalar LDS comparison
// =============================================================================
//
// Hypothesis: replacing 96 scalar LDS with 8 ldmatrix calls per K-step
// reduces instruction count 3× and improves tensor core utilization.
//
// Key insight: FP8 m16n8k32 fragment layout is byte-identical to
// FP16 m16n8k16 fragment layout. So ldmatrix.m8n8.x4 (designed for FP16)
// produces the correct register distribution for FP8 MMA directly.
//
// Per K-step instruction count:
//   Original:  16 LDS (A) + 8 LDS (B) = 24 scalar loads
//   ldmatrix:   4 ldmatrix.x4 (A) + 4 ldmatrix.x2 (B) = 8 instructions
//   Both:      16 QMMA (unchanged)
//
// Per K-iteration (4 K-steps):
//   Original:  96 LDS + 64 QMMA
//   ldmatrix:  32 ldmatrix + 64 QMMA
//
// Build:
//   nvcc -O3 -arch=sm_89 -std=c++17 fp8_gemm_v21.cu -o v21 -lcudart
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
#define WARPS_M 2
#define WARPS_N 4
#define WM 64
#define WN 32
#define MMA_M 16
#define MMA_N 8
#define MMA_K 32
#define M_TILES 4
#define N_TILES 4
#define K_STEPS 4

#define SMEM_PER_MAT (BM * SMEM_STRIDE)
#define SMEM_PER_BLOCK (2 * SMEM_PER_MAT)

// ---- Shared memory swizzle ----
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

// ---- MMA instruction ----
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
          "r"(b0), "r"(b1),
          "r"(c0), "r"(c1));
    c0 = d0;
    c1 = d1;
}

// ---- ldmatrix intrinsics ----
// Convert generic pointer to shared memory address for PTX
__device__ __forceinline__ uint32_t to_smem_addr(const void *ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Load 4 8×8 matrices (m16×k16 FP16 = m16×k32 FP8 fragment)
// Returns 4 registers matching MMA A-operand layout
__device__ __forceinline__ void ldmatrix_x4(
    uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3,
    const void *smem_ptr)
{
    uint32_t addr = to_smem_addr(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}

// Load 2 8×8 matrices (n8×k16 FP16 = n8×k32 FP8 fragment)
// Returns 2 registers matching MMA B-operand layout
__device__ __forceinline__ void ldmatrix_x2(
    uint32_t &r0, uint32_t &r1,
    const void *smem_ptr)
{
    uint32_t addr = to_smem_addr(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(addr));
}

// =============================================================================
// Kernel A: Original v10b (scalar LDS fragment loads)
// =============================================================================
extern __shared__ uint8_t dyn_smem_A[];

__global__ void __launch_bounds__(BLOCK_THREADS, 3)
    kernel_original(
        const uint8_t *__restrict__ A,
        const uint8_t *__restrict__ B,
        uint16_t *__restrict__ C,
        int M, int N, int K)
{
    uint8_t *smem_A = dyn_smem_A;
    uint8_t *smem_B = dyn_smem_A + SMEM_PER_MAT;

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

    const int thr_per_row = BK / 16;
    const int rows_per_pass = BLOCK_THREADS / thr_per_row;
    const int load_row_in_pass = threadIdx.x / thr_per_row;
    const int load_col = (threadIdx.x % thr_per_row) * 16;

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
#pragma unroll
        for (int pass = 0; pass < 4; pass++)
        {
            int row = pass * rows_per_pass + load_row_in_pass;
            int gm = bm + row;
            int gk = bk + load_col;
            uint4 val = make_uint4(0u, 0u, 0u, 0u);
            if (gm < M && gk + 16 <= K)
                val = __ldg((const uint4 *)&A[gm * K + gk]);
            *(uint4 *)&smem_A[swizzle16(row, load_col)] = val;
        }
#pragma unroll
        for (int pass = 0; pass < 4; pass++)
        {
            int row = pass * rows_per_pass + load_row_in_pass;
            int gn = bn + row;
            int gk = bk + load_col;
            uint4 val = make_uint4(0u, 0u, 0u, 0u);
            if (gn < N && gk + 16 <= K)
                val = __ldg((const uint4 *)&B[gn * K + gk]);
            *(uint4 *)&smem_B[swizzle16(row, load_col)] = val;
        }
        __syncthreads();

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
                int col_hi = k_off + tid * 4 + 16;
                a_frag[mi][0] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id, col_lo)];
                a_frag[mi][1] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id + 8, col_lo)];
                a_frag[mi][2] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id, col_hi)];
                a_frag[mi][3] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id + 8, col_hi)];
            }
            uint32_t b_frag[N_TILES][2];
#pragma unroll
            for (int ni = 0; ni < N_TILES; ni++)
            {
                int b_row = wn + ni * MMA_N;
                int col_lo = k_off + tid * 4;
                int col_hi = k_off + tid * 4 + 16;
                b_frag[ni][0] = *(uint32_t *)&smem_B[swizzle4(b_row + group_id, col_lo)];
                b_frag[ni][1] = *(uint32_t *)&smem_B[swizzle4(b_row + group_id, col_hi)];
            }
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
#pragma unroll
                for (int ni = 0; ni < N_TILES; ni++)
                    mma_fp8(a_frag[mi][0], a_frag[mi][1],
                            a_frag[mi][2], a_frag[mi][3],
                            b_frag[ni][0], b_frag[ni][1],
                            acc[mi][ni][0], acc[mi][ni][1]);
        }
        __syncthreads();
    }

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

// =============================================================================
// Kernel B: ldmatrix fragment loads
// =============================================================================
//
// ldmatrix.x4 thread→address mapping for A (m16×k32 FP8 tile):
//   Each thread provides address of one 16-byte row.
//   The 4 matrices form a 2×2 grid covering the 16×32 byte tile:
//
//     [M0: rows 0-7, k 0-15 ] [M2: rows 0-7, k 16-31 ]
//     [M1: rows 8-15, k 0-15] [M3: rows 8-15, k 16-31]
//
//   lane_group = lane_id / 8  (0,1,2,3)
//   Group 0 (lanes 0-7):   M0 — row = base + lane%8,     col = base + 0
//   Group 1 (lanes 8-15):  M1 — row = base + lane%8 + 8, col = base + 0
//   Group 2 (lanes 16-23): M2 — row = base + lane%8,     col = base + 16
//   Group 3 (lanes 24-31): M3 — row = base + lane%8 + 8, col = base + 16
//
//   Output: {r0,r1,r2,r3} maps to {a0,a1,a2,a3} of MMA fragment
//
// ldmatrix.x2 thread→address mapping for B (n8×k32 FP8 tile):
//   2 matrices covering 8×32 bytes:
//
//     [M0: rows 0-7, k 0-15] [M1: rows 0-7, k 16-31]
//
//   Group 0 (lanes 0-7):  M0 — row = base + lane%8, col = base + 0
//   Group 1 (lanes 8-15): M1 — row = base + lane%8, col = base + 16
//   Lanes 16-31: addresses ignored (provide valid dummy)
//
//   Output: {r0,r1} maps to {b0,b1} of MMA fragment
// =============================================================================
extern __shared__ uint8_t dyn_smem_B[];

__global__ void __launch_bounds__(BLOCK_THREADS, 3)
    kernel_ldmatrix(
        const uint8_t *__restrict__ A,
        const uint8_t *__restrict__ B,
        uint16_t *__restrict__ C,
        int M, int N, int K)
{
    uint8_t *smem_A = dyn_smem_B;
    uint8_t *smem_B = dyn_smem_B + SMEM_PER_MAT;

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

    // Global memory load setup (identical to original)
    const int thr_per_row = BK / 16;
    const int rows_per_pass = BLOCK_THREADS / thr_per_row;
    const int load_row_in_pass = threadIdx.x / thr_per_row;
    const int load_col = (threadIdx.x % thr_per_row) * 16;

    // Precompute lane-dependent ldmatrix address components
    const int lane_sub = lane_id & 7;    // 0..7 within group
    const int lane_group = lane_id >> 3; // 0..3 for x4, 0..1 useful for x2

    // A ldmatrix: row offset and col offset within 16×32 tile
    const int a_ldm_row_off = lane_sub + (lane_group & 1) * 8; // 0-7 or 8-15
    const int a_ldm_col_off = (lane_group >> 1) * 16;          // 0 or 16

    // B ldmatrix: row offset and col offset within 8×32 tile
    const int b_ldm_row_off = lane_sub;              // 0-7
    const int b_ldm_col_off = (lane_group & 1) * 16; // 0 or 16

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
        // ---- Global → Shared memory loads (identical to original) ----
#pragma unroll
        for (int pass = 0; pass < 4; pass++)
        {
            int row = pass * rows_per_pass + load_row_in_pass;
            int gm = bm + row;
            int gk = bk + load_col;
            uint4 val = make_uint4(0u, 0u, 0u, 0u);
            if (gm < M && gk + 16 <= K)
                val = __ldg((const uint4 *)&A[gm * K + gk]);
            *(uint4 *)&smem_A[swizzle16(row, load_col)] = val;
        }
#pragma unroll
        for (int pass = 0; pass < 4; pass++)
        {
            int row = pass * rows_per_pass + load_row_in_pass;
            int gn = bn + row;
            int gk = bk + load_col;
            uint4 val = make_uint4(0u, 0u, 0u, 0u);
            if (gn < N && gk + 16 <= K)
                val = __ldg((const uint4 *)&B[gn * K + gk]);
            *(uint4 *)&smem_B[swizzle16(row, load_col)] = val;
        }
        __syncthreads();

        // ---- Compute with ldmatrix fragment loads ----
#pragma unroll
        for (int ki = 0; ki < K_STEPS; ki++)
        {
            int k_base = ki * MMA_K; // byte offset within BK tile

            // Load A fragments via ldmatrix.x4
            uint32_t a_frag[M_TILES][4];
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                int a_row = wm + mi * MMA_M + a_ldm_row_off;
                int a_col = k_base + a_ldm_col_off;
                ldmatrix_x4(
                    a_frag[mi][0], a_frag[mi][1],
                    a_frag[mi][2], a_frag[mi][3],
                    &smem_A[swizzle16(a_row, a_col)]);
            }

            // Load B fragments via ldmatrix.x2
            uint32_t b_frag[N_TILES][2];
#pragma unroll
            for (int ni = 0; ni < N_TILES; ni++)
            {
                int b_row = wn + ni * MMA_N + b_ldm_row_off;
                int b_col = k_base + b_ldm_col_off;
                ldmatrix_x2(
                    b_frag[ni][0], b_frag[ni][1],
                    &smem_B[swizzle16(b_row, b_col)]);
            }

            // MMA (identical to original)
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
#pragma unroll
                for (int ni = 0; ni < N_TILES; ni++)
                    mma_fp8(a_frag[mi][0], a_frag[mi][1],
                            a_frag[mi][2], a_frag[mi][3],
                            b_frag[ni][0], b_frag[ni][1],
                            acc[mi][ni][0], acc[mi][ni][1]);
        }
        __syncthreads();
    }

    // ---- Store C (identical to original) ----
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

// =============================================================================
// Launch wrappers
// =============================================================================
static bool cfg_orig = false, cfg_ldm = false;

extern "C" void launch_original(int M, int N, int K,
                                const void *A, const void *B, void *C)
{
    if (!cfg_orig)
    {
        cudaFuncSetAttribute(kernel_original,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_PER_BLOCK);
        cfg_orig = true;
    }
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    kernel_original<<<grid, BLOCK_THREADS, SMEM_PER_BLOCK>>>(
        (const uint8_t *)A, (const uint8_t *)B, (uint16_t *)C, M, N, K);
}

extern "C" void launch_ldmatrix(int M, int N, int K,
                                const void *A, const void *B, void *C)
{
    if (!cfg_ldm)
    {
        cudaFuncSetAttribute(kernel_ldmatrix,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_PER_BLOCK);
        cfg_ldm = true;
    }
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    kernel_ldmatrix<<<grid, BLOCK_THREADS, SMEM_PER_BLOCK>>>(
        (const uint8_t *)A, (const uint8_t *)B, (uint16_t *)C, M, N, K);
}

typedef void (*launch_fn)(int, int, int, const void *, const void *, void *);

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

// ---- Correctness ----
void test_correctness(const char *name, launch_fn fn)
{
    // Test 1: Identity matrix
    {
        int N = 256, M = N, K = N;
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
        fn(M, N, K, dA, dB, dC);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hC, dC, sC, cudaMemcpyDeviceToHost));
        int err = 0;
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
                if (fabsf(fp16f(hC[m * N + n]) - e4m3_to_float(hB[n * K + m])) > 0.125f)
                    err++;
        printf("  %-12s identity 256:  %s (%d err)\n", name, err == 0 ? "PASS" : "FAIL", err);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        free(hA);
        free(hB);
        free(hC);
    }

    // Test 2: Random
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
        uint16_t *dC;
        CK(cudaMalloc(&dA, sA));
        CK(cudaMalloc(&dB, sB));
        CK(cudaMalloc(&dC, sC));
        CK(cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice));
        CK(cudaMemset(dC, 0, sC));
        fn(M, N, K, dA, dB, dC);
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
        printf("  %-12s random 512³:  %s (max_err=%.4f, %d err)\n",
               name, err == 0 ? "PASS" : "FAIL", mx, err);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        free(hA);
        free(hB);
        free(hC);
        free(ref);
    }
}

// ---- Benchmark ----
double bench_one(launch_fn fn, int M, int N, int K, int warmup = 10, int iters = 200)
{
    void *dA, *dB, *dC;
    CK(cudaMalloc(&dA, (size_t)M * K));
    CK(cudaMalloc(&dB, (size_t)N * K));
    CK(cudaMalloc(&dC, (size_t)M * N * 2));
    CK(cudaMemset(dA, 0x38, (size_t)M * K));
    CK(cudaMemset(dB, 0x38, (size_t)N * K));
    for (int i = 0; i < warmup; i++)
        fn(M, N, K, dA, dB, dC);
    CK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0));
    for (int i = 0; i < iters; i++)
        fn(M, N, K, dA, dB, dC);
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));
    float ms;
    CK(cudaEventElapsedTime(&ms, t0, t1));
    double tflops = 2.0 * (double)M * (double)N * (double)K / (ms / iters / 1000.0) / 1e12;
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return tflops;
}

int main()
{
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("=== FP8 GEMM v21: Original vs ldmatrix ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n", p.name, p.multiProcessorCount, p.clockRate / 1000);

    int occ_orig, occ_ldm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_orig,
                                                  kernel_original, BLOCK_THREADS, SMEM_PER_BLOCK);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_ldm,
                                                  kernel_ldmatrix, BLOCK_THREADS, SMEM_PER_BLOCK);
    printf("Occupancy: original=%d blocks/SM, ldmatrix=%d blocks/SM\n\n", occ_orig, occ_ldm);

    // ---- Correctness ----
    printf("--- Correctness ---\n");
    test_correctness("original", launch_original);
    test_correctness("ldmatrix", launch_ldmatrix);

    // Check if ldmatrix failed
    printf("\n");

    // ---- Performance ----
    struct
    {
        int M, N, K;
        const char *label;
    } sizes[] = {
        {1024, 1024, 1024, "1K³"},
        {2048, 2048, 2048, "2K³"},
        {4096, 4096, 4096, "4K³"},
        {8192, 8192, 8192, "8K³"},
        {2048, 4096, 4096, "2Kx4Kx4K"},
        {2048, 11008, 4096, "2Kx11Kx4K"},
        {4096, 4096, 4096, "4K³ llm"},
        {4096, 11008, 4096, "4Kx11Kx4K"},
        {8192, 4096, 4096, "8Kx4Kx4K"},
        {8192, 11008, 4096, "8Kx11Kx4K"},
    };
    int ns = sizeof(sizes) / sizeof(sizes[0]);

    printf("--- Performance (TFLOPS) ---\n");
    printf("%-14s %10s %10s %10s\n", "Size", "original", "ldmatrix", "delta");
    for (int j = 0; j < 48; j++)
        printf("-");
    printf("\n");

    double sum_orig = 0, sum_ldm = 0;
    for (int s = 0; s < ns; s++)
    {
        int M = sizes[s].M, N = sizes[s].N, K = sizes[s].K;
        double orig = bench_one(launch_original, M, N, K);
        double ldm = bench_one(launch_ldmatrix, M, N, K);
        double delta = ldm - orig;
        sum_orig += orig;
        sum_ldm += ldm;

        printf("%-14s %10.1f %10.1f %+10.1f", sizes[s].label, orig, ldm, delta);
        if (delta > 10.0)
            printf("  ** WIN **");
        else if (delta < -10.0)
            printf("  !! LOSS !!");
        printf("\n");
    }

    printf("%-14s %10.1f %10.1f %+10.1f\n", "AVERAGE",
           sum_orig / ns, sum_ldm / ns, (sum_ldm - sum_orig) / ns);

    printf("\n--- Instruction count analysis ---\n");
    printf("Per K-step (BK=128, 4 K-steps per tile):\n");
    printf("  original: 4×4=16 LDS(A) + 4×2=8 LDS(B) = 24 scalar loads\n");
    printf("  ldmatrix: 4×ldmatrix.x4(A) + 4×ldmatrix.x2(B) = 8 loads\n");
    printf("  Both: 4×4=16 MMA instructions\n");
    printf("  Ratio: 96 LDS → 32 ldmatrix per K-tile = 3× fewer load instructions\n");

    printf("\nPeak:660 | cuBLAS:~330 | v10b:546\n");
    return 0;
}
