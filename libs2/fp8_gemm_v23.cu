// =============================================================================
// FP8 GEMM v23 — mbarrier vs __syncthreads()
// =============================================================================
// SM89 supports: mbarrier.init, mbarrier.arrive, mbarrier.test_wait
// SM89 missing:  mbarrier.try_wait.parity, mbarrier.expect_tx
//
// Hypothesis: __syncthreads() = BAR.SYNC = hard warp stall.
//   mbarrier.arrive + test_wait polling = warp stays "active" → scheduler
//   might overlap polling with other warps' compute.
//
// Three variants:
//   syncthreads — singlesync with __syncthreads() (baseline from v22)
//   mbar_poll   — mbarrier with busy-wait polling
//   mbar_yield  — mbarrier with __nanosleep() in polling loop
//
// Build:
//   nvcc -O3 -arch=sm_89 -std=c++17 fp8_gemm_v23.cu -o v23 -lcudart
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
// Extra 8 bytes for mbarrier storage, aligned to 8
#define MBAR_SIZE 8
#define SMEM_PER_BLOCK_BASE (2 * SMEM_PER_MAT)
// mbarrier at end of dynamic smem, 8-byte aligned
#define SMEM_PER_BLOCK (SMEM_PER_BLOCK_BASE + MBAR_SIZE)

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

// ---- mbarrier helpers via inline PTX ----
__device__ __forceinline__ void mbar_init(uint32_t smem_addr, int count)
{
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(smem_addr), "r"(count));
}

__device__ __forceinline__ uint64_t mbar_arrive(uint32_t smem_addr)
{
    uint64_t token;
    asm volatile("mbarrier.arrive.shared.b64 %0, [%1];\n"
                 : "=l"(token) : "r"(smem_addr));
    return token;
}

__device__ __forceinline__ bool mbar_test_wait(uint32_t smem_addr, uint64_t token)
{
    uint32_t done;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "mbarrier.test_wait.shared.b64 p, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t"
        "}\n"
        : "=r"(done) : "r"(smem_addr), "l"(token));
    return done != 0;
}

// Composite: arrive + busy-poll until complete
__device__ __forceinline__ void mbar_sync_poll(uint32_t smem_addr)
{
    uint64_t token = mbar_arrive(smem_addr);
    while (!mbar_test_wait(smem_addr, token))
    {
        // Busy wait — warp stays active, scheduler can interleave
    }
}

// Composite: arrive + poll with nanosleep yield
__device__ __forceinline__ void mbar_sync_yield(uint32_t smem_addr)
{
    uint64_t token = mbar_arrive(smem_addr);
    while (!mbar_test_wait(smem_addr, token))
    {
        __nanosleep(32); // yield ~32ns, let scheduler switch warps
    }
}

// Get shared memory address from pointer for PTX instructions
__device__ __forceinline__ uint32_t to_smem_addr(void *ptr)
{
    return (uint32_t)(uint64_t)__cvta_generic_to_shared(ptr);
}

// ---- GMEM→SMEM load macro ----
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

// ---- Singlesync compute macro (paired K-steps) ----
#define SINGLESYNC_COMPUTE(smem_A, smem_B, wm, wn, group_id, tid, acc)                                        \
    do                                                                                                        \
    {                                                                                                         \
        _Pragma("unroll") for (int kp = 0; kp < K_STEPS; kp += 2)                                             \
        {                                                                                                     \
            uint32_t a_frag0[4][4], a_frag1[4][4];                                                            \
            uint32_t b_frag0[4][2], b_frag1[4][2];                                                            \
            int k_off0 = kp * MMA_K;                                                                          \
            int k_off1 = (kp + 1) * MMA_K;                                                                    \
            _Pragma("unroll") for (int mi = 0; mi < 4; mi++)                                                  \
            {                                                                                                 \
                int a_row = wm + mi * MMA_M;                                                                  \
                a_frag0[mi][0] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id, k_off0 + tid * 4)];          \
                a_frag0[mi][1] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id + 8, k_off0 + tid * 4)];      \
                a_frag0[mi][2] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id, k_off0 + tid * 4 + 16)];     \
                a_frag0[mi][3] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id + 8, k_off0 + tid * 4 + 16)]; \
                a_frag1[mi][0] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id, k_off1 + tid * 4)];          \
                a_frag1[mi][1] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id + 8, k_off1 + tid * 4)];      \
                a_frag1[mi][2] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id, k_off1 + tid * 4 + 16)];     \
                a_frag1[mi][3] = *(uint32_t *)&smem_A[swizzle4(a_row + group_id + 8, k_off1 + tid * 4 + 16)]; \
            }                                                                                                 \
            _Pragma("unroll") for (int ni = 0; ni < 4; ni++)                                                  \
            {                                                                                                 \
                int b_row = wn + ni * MMA_N;                                                                  \
                b_frag0[ni][0] = *(uint32_t *)&smem_B[swizzle4(b_row + group_id, k_off0 + tid * 4)];          \
                b_frag0[ni][1] = *(uint32_t *)&smem_B[swizzle4(b_row + group_id, k_off0 + tid * 4 + 16)];     \
                b_frag1[ni][0] = *(uint32_t *)&smem_B[swizzle4(b_row + group_id, k_off1 + tid * 4)];          \
                b_frag1[ni][1] = *(uint32_t *)&smem_B[swizzle4(b_row + group_id, k_off1 + tid * 4 + 16)];     \
            }                                                                                                 \
            _Pragma("unroll") for (int mi = 0; mi < 4; mi++)                                                  \
                _Pragma("unroll") for (int ni = 0; ni < 4; ni++)                                              \
                    mma_fp8(a_frag0[mi][0], a_frag0[mi][1], a_frag0[mi][2], a_frag0[mi][3],                   \
                            b_frag0[ni][0], b_frag0[ni][1], acc[mi][ni][0], acc[mi][ni][1]);                  \
            _Pragma("unroll") for (int mi = 0; mi < 4; mi++)                                                  \
                _Pragma("unroll") for (int ni = 0; ni < 4; ni++)                                              \
                    mma_fp8(a_frag1[mi][0], a_frag1[mi][1], a_frag1[mi][2], a_frag1[mi][3],                   \
                            b_frag1[ni][0], b_frag1[ni][1], acc[mi][ni][0], acc[mi][ni][1]);                  \
        }                                                                                                     \
    } while (0)

#define STORE_C(acc, bm, wm, bn, wn, group_id, tid, C, M, N)      \
    do                                                            \
    {                                                             \
        _Pragma("unroll") for (int mi = 0; mi < 4; mi++)          \
            _Pragma("unroll") for (int ni = 0; ni < 4; ni++)      \
        {                                                         \
            int row0 = bm + wm + mi * MMA_M + group_id;           \
            int row1 = row0 + 8;                                  \
            int col = bn + wn + ni * MMA_N + tid * 2;             \
            if (row0 < M && col + 1 < N)                          \
                *(uint32_t *)&C[row0 * N + col] = acc[mi][ni][0]; \
            if (row1 < M && col + 1 < N)                          \
                *(uint32_t *)&C[row1 * N + col] = acc[mi][ni][1]; \
        }                                                         \
    } while (0)

// =============================================================================
// Variant 0: __syncthreads() baseline (singlesync from v22)
// =============================================================================
extern __shared__ uint8_t dyn_smem_st[];

__global__ void __launch_bounds__(BLOCK_THREADS, 3)
    kernel_syncthreads(
        const uint8_t *__restrict__ A,
        const uint8_t *__restrict__ B,
        uint16_t *__restrict__ C,
        int M, int N, int K)
{
    uint8_t *smem_A = dyn_smem_st;
    uint8_t *smem_B = dyn_smem_st + SMEM_PER_MAT;

    const int bm = blockIdx.x * BM, bn = blockIdx.y * BN;
    const int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32;
    const int warp_m = warp_id / 4, warp_n = warp_id % 4;
    const int wm = warp_m * 64, wn = warp_n * 32;
    const int group_id = lane_id / 4, tid = lane_id % 4;

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
        SINGLESYNC_COMPUTE(smem_A, smem_B, wm, wn, group_id, tid, acc);
        __syncthreads();
    }
    STORE_C(acc, bm, wm, bn, wn, group_id, tid, C, M, N);
}

// =============================================================================
// Variant 1: mbarrier with busy-wait polling
// =============================================================================
extern __shared__ uint8_t dyn_smem_mp[];

__global__ void __launch_bounds__(BLOCK_THREADS, 3)
    kernel_mbar_poll(
        const uint8_t *__restrict__ A,
        const uint8_t *__restrict__ B,
        uint16_t *__restrict__ C,
        int M, int N, int K)
{
    uint8_t *smem_A = dyn_smem_mp;
    uint8_t *smem_B = dyn_smem_mp + SMEM_PER_MAT;
    // mbarrier at end of smem, 8-byte aligned
    uint64_t *mbar_ptr = (uint64_t *)(dyn_smem_mp + SMEM_PER_BLOCK_BASE);
    uint32_t mbar_addr = to_smem_addr(mbar_ptr);

    const int bm = blockIdx.x * BM, bn = blockIdx.y * BN;
    const int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32;
    const int warp_m = warp_id / 4, warp_n = warp_id % 4;
    const int wm = warp_m * 64, wn = warp_n * 32;
    const int group_id = lane_id / 4, tid = lane_id % 4;

    // Init mbarrier (thread 0 only)
    if (threadIdx.x == 0)
        mbar_init(mbar_addr, BLOCK_THREADS);
    __syncthreads(); // one-time init sync

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
        mbar_sync_poll(mbar_addr); // replace __syncthreads()
        SINGLESYNC_COMPUTE(smem_A, smem_B, wm, wn, group_id, tid, acc);
        mbar_sync_poll(mbar_addr); // replace __syncthreads()
    }
    STORE_C(acc, bm, wm, bn, wn, group_id, tid, C, M, N);
}

// =============================================================================
// Variant 2: mbarrier with nanosleep yield in polling
// =============================================================================
extern __shared__ uint8_t dyn_smem_my[];

__global__ void __launch_bounds__(BLOCK_THREADS, 3)
    kernel_mbar_yield(
        const uint8_t *__restrict__ A,
        const uint8_t *__restrict__ B,
        uint16_t *__restrict__ C,
        int M, int N, int K)
{
    uint8_t *smem_A = dyn_smem_my;
    uint8_t *smem_B = dyn_smem_my + SMEM_PER_MAT;
    uint64_t *mbar_ptr = (uint64_t *)(dyn_smem_my + SMEM_PER_BLOCK_BASE);
    uint32_t mbar_addr = to_smem_addr(mbar_ptr);

    const int bm = blockIdx.x * BM, bn = blockIdx.y * BN;
    const int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32;
    const int warp_m = warp_id / 4, warp_n = warp_id % 4;
    const int wm = warp_m * 64, wn = warp_n * 32;
    const int group_id = lane_id / 4, tid = lane_id % 4;

    if (threadIdx.x == 0)
        mbar_init(mbar_addr, BLOCK_THREADS);
    __syncthreads();

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
        mbar_sync_yield(mbar_addr); // arrive + poll with yield
        SINGLESYNC_COMPUTE(smem_A, smem_B, wm, wn, group_id, tid, acc);
        mbar_sync_yield(mbar_addr);
    }
    STORE_C(acc, bm, wm, bn, wn, group_id, tid, C, M, N);
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
};

void launch_kernel(const KernelInfo &ki, int M, int N, int K,
                   const void *dA, const void *dB, void *dC)
{
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    ki.fn<<<grid, BLOCK_THREADS, ki.smem_bytes>>>((const uint8_t *)dA, (const uint8_t *)dB, (uint16_t *)dC, M, N, K);
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
    printf("  %-14s 512³: max_err=%.4f err=%d → %s\n", ki.name, mx, err, err == 0 ? "PASS" : "FAIL");
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
    printf("=== FP8 GEMM v23: mbarrier vs __syncthreads() ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n", p.name, p.multiProcessorCount, p.clockRate / 1000);

    KernelInfo kernels[] = {
        {"syncthreads", kernel_syncthreads, SMEM_PER_BLOCK_BASE},
        {"mbar_poll", kernel_mbar_poll, SMEM_PER_BLOCK},
        {"mbar_yield", kernel_mbar_yield, SMEM_PER_BLOCK},
    };
    const int nk = 3;

    printf("--- Occupancy ---\n");
    for (int i = 0; i < nk; i++)
    {
        cudaFuncSetAttribute(kernels[i].fn,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kernels[i].smem_bytes);
        int occ;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,
                                                      kernels[i].fn, BLOCK_THREADS, kernels[i].smem_bytes);
        printf("  %-14s: %d blocks/SM\n", kernels[i].name, occ);
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

    printf("%-14s", "Size");
    for (int i = 0; i < nk; i++)
        printf(" %12s", kernels[i].name);
    printf("    best  delta_vs_sync\n");
    printf("----------------------------------------------------------------------\n");

    for (auto &s : sizes)
    {
        printf("%-14s", s.label);
        double results[3];
        for (int i = 0; i < nk; i++)
        {
            results[i] = bench_kernel(kernels[i], s.M, s.N, s.K);
            printf(" %12.1f", results[i]);
        }
        double base = results[0];
        double best = results[0];
        int bi = 0;
        for (int i = 1; i < nk; i++)
            if (results[i] > best)
            {
                best = results[i];
                bi = i;
            }
        printf("  %10s %+.1f", kernels[bi].name, best - base);
        if (best - base > 5)
            printf(" !!");
        printf("\n");
    }

    printf("\nBaseline: singlesync@OC = ~590 TFLOPS\n");
    printf("If mbar > syncthreads: barrier overhead was a bottleneck\n");
    printf("If mbar ≈ syncthreads: BAR.SYNC already optimal\n");
    printf("If mbar < syncthreads: polling overhead > barrier cost\n");
    return 0;
}
