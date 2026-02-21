// =============================================================================
// FP8 GEMM v20 — v10b vs v10b+Tile Swizzle (A/B comparison)
// =============================================================================
// Two kernels, identical inner loop. Only difference: tile→block mapping.
//
// Tile swizzle idea: remap blockIdx so that concurrent blocks access
// nearby rows/columns → better L2 cache reuse.
//
// Pattern: divide grid into groups of SWIZZLE_WIDTH columns.
// Within each group, blocks are ordered column-first instead of row-first.
// This means blocks running on the same SM wave hit adjacent L2 lines.
//
// Build:
//   nvcc -O3 -arch=sm_89 -std=c++17 fp8_gemm_v20.cu -o v20 -lcudart
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ---- Tile parameters (identical for both kernels) ----
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

#define SMEM_PER_MAT (BM * SMEM_STRIDE)   // 16384
#define SMEM_PER_BLOCK (2 * SMEM_PER_MAT) // 32768

// ---- Swizzle helpers (shared memory) ----
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
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1));
    c0 = d0;
    c1 = d1;
}

// =============================================================================
// MACRO: shared compute body (identical for both kernels)
// =============================================================================
#define GEMM_BODY(BM_IDX, BN_IDX)                                                             \
    extern __shared__ uint8_t dyn_smem[];                                                     \
    uint8_t *smem_A = dyn_smem;                                                               \
    uint8_t *smem_B = dyn_smem + SMEM_PER_MAT;                                                \
                                                                                              \
    const int bm = (BM_IDX) * BM;                                                             \
    const int bn = (BN_IDX) * BN;                                                             \
                                                                                              \
    const int warp_id = threadIdx.x / 32;                                                     \
    const int lane_id = threadIdx.x % 32;                                                     \
    const int warp_m = warp_id / WARPS_N;                                                     \
    const int warp_n = warp_id % WARPS_N;                                                     \
    const int wm = warp_m * WM;                                                               \
    const int wn = warp_n * WN;                                                               \
    const int group_id = lane_id / 4;                                                         \
    const int tid = lane_id % 4;                                                              \
                                                                                              \
    const int thr_per_row = BK / 16;                                                          \
    const int rows_per_pass = BLOCK_THREADS / thr_per_row;                                    \
    const int load_row_in_pass = threadIdx.x / thr_per_row;                                   \
    const int load_col = (threadIdx.x % thr_per_row) * 16;                                    \
                                                                                              \
    uint32_t acc[M_TILES][N_TILES][2];                                                        \
    _Pragma("unroll") for (int mi = 0; mi < M_TILES; mi++)                                    \
        _Pragma("unroll") for (int ni = 0; ni < N_TILES; ni++)                                \
    {                                                                                         \
        acc[mi][ni][0] = 0u;                                                                  \
        acc[mi][ni][1] = 0u;                                                                  \
    }                                                                                         \
                                                                                              \
    for (int bk = 0; bk < K; bk += BK)                                                        \
    {                                                                                         \
        _Pragma("unroll") for (int pass = 0; pass < 4; pass++)                                \
        {                                                                                     \
            int row = pass * rows_per_pass + load_row_in_pass;                                \
            int gm = bm + row;                                                                \
            int gk = bk + load_col;                                                           \
            uint4 val = make_uint4(0u, 0u, 0u, 0u);                                           \
            if (gm < M && gk + 16 <= K)                                                       \
                val = __ldg((const uint4 *)&A[gm * K + gk]);                                  \
            *(uint4 *)&smem_A[swizzle16(row, load_col)] = val;                                \
        }                                                                                     \
        _Pragma("unroll") for (int pass = 0; pass < 4; pass++)                                \
        {                                                                                     \
            int row = pass * rows_per_pass + load_row_in_pass;                                \
            int gn = bn + row;                                                                \
            int gk = bk + load_col;                                                           \
            uint4 val = make_uint4(0u, 0u, 0u, 0u);                                           \
            if (gn < N && gk + 16 <= K)                                                       \
                val = __ldg((const uint4 *)&B[gn * K + gk]);                                  \
            *(uint4 *)&smem_B[swizzle16(row, load_col)] = val;                                \
        }                                                                                     \
        __syncthreads();                                                                      \
        _Pragma("unroll") for (int ki = 0; ki < K_STEPS; ki++)                                \
        {                                                                                     \
            int k_off = ki * MMA_K;                                                           \
            uint32_t a_frag[M_TILES][4];                                                      \
            _Pragma("unroll") for (int mi = 0; mi < M_TILES; mi++)                            \
            {                                                                                 \
                int a_row = wm + mi * MMA_M;                                                  \
                int col_lo = k_off + tid * 4;                                                 \
                int col_hi = k_off + tid * 4 + 16;                                            \
                a_frag[mi][0] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id, col_lo)];     \
                a_frag[mi][1] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id + 8, col_lo)]; \
                a_frag[mi][2] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id, col_hi)];     \
                a_frag[mi][3] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id + 8, col_hi)]; \
            }                                                                                 \
            uint32_t b_frag[N_TILES][2];                                                      \
            _Pragma("unroll") for (int ni = 0; ni < N_TILES; ni++)                            \
            {                                                                                 \
                int b_row = wn + ni * MMA_N;                                                  \
                int col_lo = k_off + tid * 4;                                                 \
                int col_hi = k_off + tid * 4 + 16;                                            \
                b_frag[ni][0] = *(uint32_t *)&smem_B[swizzle4(b_row + group_id, col_lo)];     \
                b_frag[ni][1] = *(uint32_t *)&smem_B[swizzle4(b_row + group_id, col_hi)];     \
            }                                                                                 \
            _Pragma("unroll") for (int mi = 0; mi < M_TILES; mi++)                            \
                _Pragma("unroll") for (int ni = 0; ni < N_TILES; ni++)                        \
                    mma_fp8(a_frag[mi][0], a_frag[mi][1],                                     \
                            a_frag[mi][2], a_frag[mi][3],                                     \
                            b_frag[ni][0], b_frag[ni][1],                                     \
                            acc[mi][ni][0], acc[mi][ni][1]);                                  \
        }                                                                                     \
        __syncthreads();                                                                      \
    }                                                                                         \
    _Pragma("unroll") for (int mi = 0; mi < M_TILES; mi++)                                    \
        _Pragma("unroll") for (int ni = 0; ni < N_TILES; ni++)                                \
    {                                                                                         \
        int row0 = bm + wm + mi * MMA_M + group_id;                                           \
        int row1 = row0 + 8;                                                                  \
        int col = bn + wn + ni * MMA_N + tid * 2;                                             \
        if (row0 < M && col + 1 < N)                                                          \
            *(uint32_t *)&C[row0 * N + col] = acc[mi][ni][0];                                 \
        if (row1 < M && col + 1 < N)                                                          \
            *(uint32_t *)&C[row1 * N + col] = acc[mi][ni][1];                                 \
    }

// =============================================================================
// Kernel A: Original v10b (row-major block order)
// =============================================================================
__global__ void __launch_bounds__(BLOCK_THREADS, 3)
    kernel_original(
        const uint8_t *__restrict__ A,
        const uint8_t *__restrict__ B,
        uint16_t *__restrict__ C,
        int M, int N, int K)
{
    GEMM_BODY(blockIdx.x, blockIdx.y);
}

// =============================================================================
// Kernel B: v10b + Tile Swizzle (better L2 locality)
// =============================================================================
// Swizzle width = number of column-tiles grouped together.
// Blocks within a group are reordered: instead of sweeping row-by-row,
// we sweep in SWIZZLE_W-wide vertical strips.
//
//  Original (row-major):         Swizzled (SW=4):
//   0  1  2  3  4  5  6  7       0  4  8 12 16 20 24 28
//   8  9 10 11 12 13 14 15       1  5  9 13 17 21 25 29
//  16 17 18 19 20 21 22 23       2  6 10 14 18 22 26 30
//  24 25 26 27 28 29 30 31       3  7 11 15 19 23 27 31
//
// Adjacent block IDs now map to same column group → same B data in L2.
// =============================================================================

// Try multiple swizzle widths
template <int SWIZZLE_W>
__global__ void __launch_bounds__(BLOCK_THREADS, 3)
    kernel_swizzled(
        const uint8_t *__restrict__ A,
        const uint8_t *__restrict__ B,
        uint16_t *__restrict__ C,
        int M, int N, int K)
{
    // Remap blockIdx for better L2 locality
    const int grid_m = gridDim.x; // number of M-tiles
    const int grid_n = gridDim.y; // number of N-tiles

    // Linear block ID
    int bid = blockIdx.x * grid_n + blockIdx.y;

    // Number of blocks in one swizzle group = SWIZZLE_W * grid_m
    int group_size = SWIZZLE_W * grid_m;
    int group_id_val = bid / group_size;
    int within_group = bid % group_size;

    // Within group: column-first ordering
    int local_m = within_group / SWIZZLE_W;
    int local_n = within_group % SWIZZLE_W;

    int new_bx = local_m;
    int new_by = group_id_val * SWIZZLE_W + local_n;

    // Clamp for edge cases (grid_n not divisible by SWIZZLE_W)
    if (new_by >= grid_n)
    {
        new_bx = blockIdx.x;
        new_by = blockIdx.y;
    }

    GEMM_BODY(new_bx, new_by);
}

// =============================================================================
// Launch wrappers
// =============================================================================
static bool smem_configured_orig = false;
static bool smem_configured_sw[5] = {false};

extern "C" void launch_original(
    int M, int N, int K,
    const void *A, const void *B, void *C)
{
    if (!smem_configured_orig)
    {
        cudaFuncSetAttribute(kernel_original,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_PER_BLOCK);
        smem_configured_orig = true;
    }
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(BLOCK_THREADS);
    kernel_original<<<grid, block, SMEM_PER_BLOCK>>>(
        (const uint8_t *)A, (const uint8_t *)B, (uint16_t *)C, M, N, K);
}

template <int SW>
void launch_swizzled_t(int M, int N, int K,
                       const void *A, const void *B, void *C, int sw_idx)
{
    if (!smem_configured_sw[sw_idx])
    {
        cudaFuncSetAttribute(kernel_swizzled<SW>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_PER_BLOCK);
        smem_configured_sw[sw_idx] = true;
    }
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(BLOCK_THREADS);
    kernel_swizzled<SW><<<grid, block, SMEM_PER_BLOCK>>>(
        (const uint8_t *)A, (const uint8_t *)B, (uint16_t *)C, M, N, K);
}

typedef void (*launch_fn)(int, int, int, const void *, const void *, void *);

void launch_sw2(int M, int N, int K, const void *A, const void *B, void *C) { launch_swizzled_t<2>(M, N, K, A, B, C, 0); }
void launch_sw4(int M, int N, int K, const void *A, const void *B, void *C) { launch_swizzled_t<4>(M, N, K, A, B, C, 1); }
void launch_sw8(int M, int N, int K, const void *A, const void *B, void *C) { launch_swizzled_t<8>(M, N, K, A, B, C, 2); }
void launch_sw16(int M, int N, int K, const void *A, const void *B, void *C) { launch_swizzled_t<16>(M, N, K, A, B, C, 3); }

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

// Correctness check for a launch function
void test_correctness(const char *name, launch_fn fn)
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
    printf("  %-12s 512x512x512: max_err=%.4f errors=%d → %s\n",
           name, mx, err, err == 0 ? "PASS" : "FAIL");
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);
    free(hC);
    free(ref);
}

// Benchmark one launch function at one size
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
    printf("=== FP8 GEMM v20: v10b vs Tile Swizzle Comparison ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n", p.name, p.multiProcessorCount, p.clockRate / 1000);
    printf("L2 Cache: %d KB\n\n", p.l2CacheSize / 1024);

    // Check occupancy
    int occ;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel_original, BLOCK_THREADS, SMEM_PER_BLOCK);
    printf("Occupancy: %d blocks/SM\n\n", occ);

    // ---- Correctness ----
    printf("--- Correctness ---\n");
    test_correctness("original", launch_original);
    test_correctness("sw2", launch_sw2);
    test_correctness("sw4", launch_sw4);
    test_correctness("sw8", launch_sw8);
    test_correctness("sw16", launch_sw16);

    // ---- Performance comparison ----
    struct
    {
        const char *name;
        launch_fn fn;
    } variants[] = {
        {"original", launch_original},
        {"sw2", launch_sw2},
        {"sw4", launch_sw4},
        {"sw8", launch_sw8},
        {"sw16", launch_sw16},
    };
    int nv = sizeof(variants) / sizeof(variants[0]);

    struct
    {
        int M, N, K;
        const char *label;
    } sizes[] = {
        {1024, 1024, 1024, "1K³"},
        {2048, 2048, 2048, "2K³"},
        {4096, 4096, 4096, "4K³"},
        {8192, 8192, 8192, "8K³"},
        // LLM shapes
        {2048, 4096, 4096, "2Kx4Kx4K"},
        {2048, 11008, 4096, "2Kx11Kx4K"},
        {4096, 4096, 4096, "4K³ llm"},
        {4096, 11008, 4096, "4Kx11Kx4K"},
        {8192, 4096, 4096, "8Kx4Kx4K"},
        {8192, 11008, 4096, "8Kx11Kx4K"},
    };
    int ns = sizeof(sizes) / sizeof(sizes[0]);

    printf("\n--- Performance (TFLOPS) ---\n");
    printf("%-14s", "Size");
    for (int v = 0; v < nv; v++)
        printf(" %8s", variants[v].name);
    printf("  best_sw  delta\n");

    for (int j = 0; j < 14 + 9 * nv + 16; j++)
        printf("-");
    printf("\n");

    for (int s = 0; s < ns; s++)
    {
        int M = sizes[s].M, N = sizes[s].N, K = sizes[s].K;
        printf("%-14s", sizes[s].label);

        double results[5];
        double orig = 0, best_sw = 0;
        const char *best_name = "";

        for (int v = 0; v < nv; v++)
        {
            results[v] = bench_one(variants[v].fn, M, N, K);
            printf(" %8.1f", results[v]);
            if (v == 0)
                orig = results[v];
            else if (results[v] > best_sw)
            {
                best_sw = results[v];
                best_name = variants[v].name;
            }
        }

        double delta = best_sw - orig;
        printf("  %s", best_name);
        if (delta > 0)
            printf("  +%.1f", delta);
        else
            printf("  %.1f", delta);
        printf("\n");
    }

    printf("\n--- Summary ---\n");
    printf("If all deltas are ≤0: tile swizzle doesn't help (L2 not the bottleneck)\n");
    printf("If some deltas >0: found an optimal swizzle width for those shapes\n");
    printf("If delta >10 TFLOPS: significant win, worth keeping\n");

    printf("\nPeak:660 | cuBLAS:~330 | v10b:546\n");
    return 0;
}
