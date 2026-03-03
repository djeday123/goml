// =============================================================================
// FlashAttention v21 — exp2f optimization over v20 (reg-based S/P)
// =============================================================================
// Single change: __expf(x) → exp2f(x * LOG2E)
//
// __expf is already a fast approximation (~2 instructions on SM89)
// exp2f(x * 1.4427) = 1 FMA + 1 ex2.approx.f32 = potentially 1 fewer instr
//
// 6 replacement sites in the kernel:
//   2× rescale O:  __expf(rmax[i] - nm[i])
//   4× softmax:    __expf(Sr[nt][i] - rmax[i])
//
// Everything else identical to v20.
//
// Build:
//   nvcc -O3 -arch=sm_89 -std=c++17 flash_attention_v21.cu -o fa_v21 -lcudart
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define FA20_BR 64
#define FA20_BC 64
#define FA20_THREADS 128
#define FA20_STRIDE 128

#define LOG2E 1.4426950408889634f

// =============================================================================
// Shared helpers (identical to v20)
// =============================================================================

__device__ __forceinline__ int swz(int row, int col)
{
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}
__device__ __forceinline__ void cpa16(void *s, const void *g, int n)
{
    uint32_t sa = __cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" ::"r"(sa), "l"(g), "r"(n));
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cpa_wait()
{
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}
__device__ __forceinline__ void ldm4(uint32_t &r0, uint32_t &r1,
                                     uint32_t &r2, uint32_t &r3, const void *p)
{
    uint32_t a = __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldm2(uint32_t &r0, uint32_t &r1, const void *p)
{
    uint32_t a = __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1},[%2];"
                 : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void ldm2t(uint32_t &r0, uint32_t &r1, const void *p)
{
    uint32_t a = __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16.trans {%0,%1},[%2];"
                 : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void mma16816(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__ void ld_a_sw(
    uint32_t &a0, uint32_t &a1, uint32_t &a2, uint32_t &a3,
    const __half *sm, int stride, int rb, int kb, int lane)
{
    int sub = lane / 8, sr = lane % 8;
    int r = rb + (sub & 1) * 8 + sr, lc = kb + (sub >> 1) * 8;
    ldm4(a0, a1, a2, a3, &sm[r * stride + swz(r, lc)]);
}
__device__ __forceinline__ void ld_b_sw(
    uint32_t &b0, uint32_t &b1,
    const __half *sm, int stride, int nb, int kb, int lane)
{
    int sub = (lane / 8) % 2, sr = lane % 8;
    int r = nb + sr, lc = kb + sub * 8;
    ldm2(b0, b1, &sm[r * stride + swz(r, lc)]);
}
__device__ __forceinline__ void ld_b_vt(
    uint32_t &b0, uint32_t &b1,
    const __half *sm, int stride, int nb, int kb, int lane)
{
    int sub = (lane / 8) % 2, sr = lane % 8;
    int k = kb + sub * 8 + sr;
    ldm2t(b0, b1, &sm[k * stride + swz(k, nb)]);
}

__device__ __forceinline__ void load_tile(
    __half *dst, const __half *src, int start, int rows,
    int seq_len, int head_dim)
{
    constexpr int CPR = 16;
    int total = rows * CPR;
#pragma unroll 4
    for (int c = threadIdx.x; c < total; c += FA20_THREADS)
    {
        int row = c / CPR, lch = c % CPR, gr = start + row;
        int pch = lch ^ (row & 7);
        cpa16(&dst[row * FA20_STRIDE + pch * 8],
              &src[gr * head_dim + lch * 8], (gr < seq_len) ? 16 : 0);
    }
}

// =============================================================================
// Macro: kernel body parameterized by EXP function
// =============================================================================

#define FA20_KERNEL_BODY(EXP_FN)                                                                  \
    int nqt = (seq_len + FA20_BR - 1) / FA20_BR;                                                  \
    int bh = blockIdx.x / nqt;                                                                    \
    int qt = blockIdx.x % nqt;                                                                    \
    int qs = qt * FA20_BR;                                                                        \
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;                                          \
    int gid = lane >> 2, tid = lane & 3;                                                          \
    extern __shared__ char raw[];                                                                 \
    __half *buf0 = (__half *)raw;                                                                 \
    __half *buf1 = (__half *)(raw + FA20_BC * FA20_STRIDE * sizeof(__half));                      \
    int hs = seq_len * head_dim;                                                                  \
    const __half *Qh = Q + bh * hs, *Kh = K + bh * hs, *Vh = V + bh * hs;                         \
    __half *Oh = O + bh * hs;                                                                     \
    load_tile(buf0, Qh, qs, FA20_BR, seq_len, head_dim);                                          \
    cpa_commit();                                                                                 \
    cpa_wait<0>();                                                                                \
    __syncthreads();                                                                              \
    int mrb = wid * 16;                                                                           \
    uint32_t Qr[8][4];                                                                            \
    _Pragma("unroll") for (int ks = 0; ks < 8; ks++)                                              \
        ld_a_sw(Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3],                                       \
                buf0, FA20_STRIDE, mrb, ks * 16, lane);                                           \
    __syncthreads();                                                                              \
    float Or[16][4];                                                                              \
    _Pragma("unroll") for (int t = 0; t < 16; t++) Or[t][0] = Or[t][1] = Or[t][2] = Or[t][3] = 0; \
    float rmax[2] = {-1e30f, -1e30f};                                                             \
    float rsexp[2] = {0.0f, 0.0f};                                                                \
    int nkv = (seq_len + FA20_BC - 1) / FA20_BC;                                                  \
    load_tile(buf0, Kh, 0, FA20_BC, seq_len, head_dim);                                           \
    cpa_commit();                                                                                 \
    for (int kv = 0; kv < nkv; kv++)                                                              \
    {                                                                                             \
        int kvs = kv * FA20_BC;                                                                   \
        if (causal && kvs > qs + FA20_BR - 1)                                                     \
            break;                                                                                \
        __half *cur = (kv & 1) ? buf1 : buf0;                                                     \
        __half *nxt = (kv & 1) ? buf0 : buf1;                                                     \
        cpa_wait<0>();                                                                            \
        __syncthreads();                                                                          \
        float Sr[8][4];                                                                           \
        _Pragma("unroll") for (int nt = 0; nt < 8; nt++)                                          \
            Sr[nt][0] = Sr[nt][1] = Sr[nt][2] = Sr[nt][3] = 0;                                    \
        _Pragma("unroll") for (int ks = 0; ks < 8; ks++)                                          \
        {                                                                                         \
            _Pragma("unroll") for (int nt = 0; nt < 8; nt++)                                      \
            {                                                                                     \
                uint32_t b0, b1;                                                                  \
                ld_b_sw(b0, b1, cur, FA20_STRIDE, nt * 8, ks * 16, lane);                         \
                mma16816(Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3],                              \
                         Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3],                              \
                         b0, b1, Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3]);                     \
            }                                                                                     \
        }                                                                                         \
        __syncthreads();                                                                          \
        load_tile(cur, Vh, kvs, FA20_BC, seq_len, head_dim);                                      \
        cpa_commit();                                                                             \
        int nkvs = (kv + 1) * FA20_BC;                                                            \
        bool has_nxt = (kv + 1 < nkv) && (!causal || nkvs <= qs + FA20_BR - 1);                   \
        if (has_nxt)                                                                              \
            load_tile(nxt, Kh, nkvs, FA20_BC, seq_len, head_dim);                                 \
        cpa_commit();                                                                             \
        _Pragma("unroll") for (int nt = 0; nt < 8; nt++)                                          \
        {                                                                                         \
            Sr[nt][0] *= scale;                                                                   \
            Sr[nt][1] *= scale;                                                                   \
            Sr[nt][2] *= scale;                                                                   \
            Sr[nt][3] *= scale;                                                                   \
            if (causal)                                                                           \
            {                                                                                     \
                int gq0 = qs + mrb + gid, gq8 = gq0 + 8;                                          \
                int gk0 = kvs + nt * 8 + tid * 2, gk1 = gk0 + 1;                                  \
                if (gk0 > gq0)                                                                    \
                    Sr[nt][0] = -1e30f;                                                           \
                if (gk1 > gq0)                                                                    \
                    Sr[nt][1] = -1e30f;                                                           \
                if (gk0 > gq8)                                                                    \
                    Sr[nt][2] = -1e30f;                                                           \
                if (gk1 > gq8)                                                                    \
                    Sr[nt][3] = -1e30f;                                                           \
                if (gq0 >= seq_len)                                                               \
                {                                                                                 \
                    Sr[nt][0] = -1e30f;                                                           \
                    Sr[nt][1] = -1e30f;                                                           \
                }                                                                                 \
                if (gq8 >= seq_len)                                                               \
                {                                                                                 \
                    Sr[nt][2] = -1e30f;                                                           \
                    Sr[nt][3] = -1e30f;                                                           \
                }                                                                                 \
                if (gk0 >= seq_len)                                                               \
                {                                                                                 \
                    Sr[nt][0] = -1e30f;                                                           \
                    Sr[nt][2] = -1e30f;                                                           \
                }                                                                                 \
                if (gk1 >= seq_len)                                                               \
                {                                                                                 \
                    Sr[nt][1] = -1e30f;                                                           \
                    Sr[nt][3] = -1e30f;                                                           \
                }                                                                                 \
            }                                                                                     \
        }                                                                                         \
        float nm[2] = {-1e30f, -1e30f};                                                           \
        _Pragma("unroll") for (int nt = 0; nt < 8; nt++)                                          \
        {                                                                                         \
            nm[0] = fmaxf(nm[0], fmaxf(Sr[nt][0], Sr[nt][1]));                                    \
            nm[1] = fmaxf(nm[1], fmaxf(Sr[nt][2], Sr[nt][3]));                                    \
        }                                                                                         \
        nm[0] = fmaxf(nm[0], __shfl_xor_sync(0xffffffff, nm[0], 1));                              \
        nm[0] = fmaxf(nm[0], __shfl_xor_sync(0xffffffff, nm[0], 2));                              \
        nm[1] = fmaxf(nm[1], __shfl_xor_sync(0xffffffff, nm[1], 1));                              \
        nm[1] = fmaxf(nm[1], __shfl_xor_sync(0xffffffff, nm[1], 2));                              \
        nm[0] = fmaxf(nm[0], rmax[0]);                                                            \
        nm[1] = fmaxf(nm[1], rmax[1]);                                                            \
        float rsc0 = EXP_FN(rmax[0] - nm[0]);                                                     \
        float rsc1 = EXP_FN(rmax[1] - nm[1]);                                                     \
        _Pragma("unroll") for (int t = 0; t < 16; t++)                                            \
        {                                                                                         \
            Or[t][0] *= rsc0;                                                                     \
            Or[t][1] *= rsc0;                                                                     \
            Or[t][2] *= rsc1;                                                                     \
            Or[t][3] *= rsc1;                                                                     \
        }                                                                                         \
        rmax[0] = nm[0];                                                                          \
        rmax[1] = nm[1];                                                                          \
        float ns[2] = {0.0f, 0.0f};                                                               \
        uint32_t Pr[4][4];                                                                        \
        _Pragma("unroll") for (int nt = 0; nt < 8; nt++)                                          \
        {                                                                                         \
            Sr[nt][0] = EXP_FN(Sr[nt][0] - rmax[0]);                                              \
            Sr[nt][1] = EXP_FN(Sr[nt][1] - rmax[0]);                                              \
            Sr[nt][2] = EXP_FN(Sr[nt][2] - rmax[1]);                                              \
            Sr[nt][3] = EXP_FN(Sr[nt][3] - rmax[1]);                                              \
            ns[0] += Sr[nt][0] + Sr[nt][1];                                                       \
            ns[1] += Sr[nt][2] + Sr[nt][3];                                                       \
            int pi = nt / 2, half = nt % 2;                                                       \
            __half2 *p = (__half2 *)Pr[pi];                                                       \
            p[half * 2] = __halves2half2(__float2half(Sr[nt][0]), __float2half(Sr[nt][1]));       \
            p[half * 2 + 1] = __halves2half2(__float2half(Sr[nt][2]), __float2half(Sr[nt][3]));   \
        }                                                                                         \
        ns[0] += __shfl_xor_sync(0xffffffff, ns[0], 1);                                           \
        ns[0] += __shfl_xor_sync(0xffffffff, ns[0], 2);                                           \
        ns[1] += __shfl_xor_sync(0xffffffff, ns[1], 1);                                           \
        ns[1] += __shfl_xor_sync(0xffffffff, ns[1], 2);                                           \
        rsexp[0] = rsexp[0] * rsc0 + ns[0];                                                       \
        rsexp[1] = rsexp[1] * rsc1 + ns[1];                                                       \
        cpa_wait<1>();                                                                            \
        __syncthreads();                                                                          \
        _Pragma("unroll") for (int ks = 0; ks < 4; ks++)                                          \
        {                                                                                         \
            _Pragma("unroll") for (int nt = 0; nt < 16; nt++)                                     \
            {                                                                                     \
                uint32_t b0, b1;                                                                  \
                ld_b_vt(b0, b1, cur, FA20_STRIDE, nt * 8, ks * 16, lane);                         \
                mma16816(Or[nt][0], Or[nt][1], Or[nt][2], Or[nt][3],                              \
                         Pr[ks][0], Pr[ks][1], Pr[ks][2], Pr[ks][3],                              \
                         b0, b1, Or[nt][0], Or[nt][1], Or[nt][2], Or[nt][3]);                     \
            }                                                                                     \
        }                                                                                         \
    }                                                                                             \
    float li0 = (rsexp[0] > 0) ? 1.0f / rsexp[0] : 0.0f;                                          \
    float li1 = (rsexp[1] > 0) ? 1.0f / rsexp[1] : 0.0f;                                          \
    int gr0 = qs + mrb + gid, gr8 = gr0 + 8;                                                      \
    _Pragma("unroll") for (int nt = 0; nt < 16; nt++)                                             \
    {                                                                                             \
        int c0 = nt * 8 + tid * 2, c1 = c0 + 1;                                                   \
        if (gr0 < seq_len && c0 < head_dim)                                                       \
            Oh[gr0 * head_dim + c0] = __float2half(Or[nt][0] * li0);                              \
        if (gr0 < seq_len && c1 < head_dim)                                                       \
            Oh[gr0 * head_dim + c1] = __float2half(Or[nt][1] * li0);                              \
        if (gr8 < seq_len && c0 < head_dim)                                                       \
            Oh[gr8 * head_dim + c0] = __float2half(Or[nt][2] * li1);                              \
        if (gr8 < seq_len && c1 < head_dim)                                                       \
            Oh[gr8 * head_dim + c1] = __float2half(Or[nt][3] * li1);                              \
    }

// =============================================================================
// v20: __expf (original)
// =============================================================================
__global__ void __launch_bounds__(FA20_THREADS, 2)
    fa20_kernel(const __half *__restrict__ Q, const __half *__restrict__ K,
                const __half *__restrict__ V, __half *__restrict__ O,
                int seq_len, int head_dim, int causal, float scale){
        FA20_KERNEL_BODY(__expf)}

// =============================================================================
// v21: exp2f (only change)
// =============================================================================
__device__ __forceinline__ float exp2f_scaled(float x)
{
    return exp2f(x * LOG2E);
}

__global__ void __launch_bounds__(FA20_THREADS, 2)
    fa21_kernel(const __half *__restrict__ Q, const __half *__restrict__ K,
                const __half *__restrict__ V, __half *__restrict__ O,
                int seq_len, int head_dim, int causal, float scale){
        FA20_KERNEL_BODY(exp2f_scaled)}

// =============================================================================
// Benchmark harness
// =============================================================================

#define CK(c)                                                       \
    do                                                              \
    {                                                               \
        cudaError_t e = (c);                                        \
        if (e != cudaSuccess)                                       \
        {                                                           \
            fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(e));                         \
            exit(1);                                                \
        }                                                           \
    } while (0)

static inline float h2f(uint16_t h)
{
    __half hv;
    memcpy(&hv, &h, 2);
    return __half2float(hv);
}
static inline uint16_t f2h(float f)
{
    __half hv = __float2half(f);
    uint16_t r;
    memcpy(&r, &hv, 2);
    return r;
}

void fill_random_fp16(uint16_t *d_ptr, int n)
{
    uint16_t *h = (uint16_t *)malloc(n * 2);
    for (int i = 0; i < n; i++)
        h[i] = f2h(((float)(rand() % 2001) - 1000.0f) / 1000.0f);
    CK(cudaMemcpy(d_ptr, h, n * 2, cudaMemcpyHostToDevice));
    free(h);
}

struct Timer
{
    cudaEvent_t t0, t1;
    Timer()
    {
        CK(cudaEventCreate(&t0));
        CK(cudaEventCreate(&t1));
    }
    ~Timer()
    {
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }
    void start() { CK(cudaEventRecord(t0)); }
    float stop()
    {
        CK(cudaEventRecord(t1));
        CK(cudaEventSynchronize(t1));
        float ms;
        CK(cudaEventElapsedTime(&ms, t0, t1));
        return ms;
    }
};

static int g_smem20 = 0, g_smem21 = 0;

void launch_v20(const __half *Q, const __half *K, const __half *V, __half *O,
                int total_heads, int seq_len, int head_dim, int causal)
{
    int smem = 2 * FA20_BC * FA20_STRIDE * (int)sizeof(__half);
    if (smem > g_smem20)
    {
        CK(cudaFuncSetAttribute(fa20_kernel,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        g_smem20 = smem;
    }
    float sc = 1.0f / sqrtf((float)head_dim);
    int nqt = (seq_len + FA20_BR - 1) / FA20_BR;
    fa20_kernel<<<total_heads * nqt, FA20_THREADS, smem>>>(
        Q, K, V, O, seq_len, head_dim, causal, sc);
}

void launch_v21(const __half *Q, const __half *K, const __half *V, __half *O,
                int total_heads, int seq_len, int head_dim, int causal)
{
    int smem = 2 * FA20_BC * FA20_STRIDE * (int)sizeof(__half);
    if (smem > g_smem21)
    {
        CK(cudaFuncSetAttribute(fa21_kernel,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        g_smem21 = smem;
    }
    float sc = 1.0f / sqrtf((float)head_dim);
    int nqt = (seq_len + FA20_BR - 1) / FA20_BR;
    fa21_kernel<<<total_heads * nqt, FA20_THREADS, smem>>>(
        Q, K, V, O, seq_len, head_dim, causal, sc);
}

void test_correctness()
{
    printf("--- Correctness: v20 (__expf) vs v21 (exp2f) vs CPU ---\n");
    int configs[][3] = {
        {1, 32, 128}, {1, 64, 128}, {2, 128, 128}, {1, 256, 128}, {1, 512, 128}, {32, 1024, 128}};
    for (auto &c : configs)
    {
        int heads = c[0], seq = c[1], dim = c[2], n = heads * seq * dim;
        size_t sz = (size_t)n * 2;
        uint16_t *hQ = (uint16_t *)malloc(sz), *hK = (uint16_t *)malloc(sz), *hV = (uint16_t *)malloc(sz);
        float *ref = (float *)calloc(n, 4);
        srand(42);
        for (int i = 0; i < n; i++)
            hQ[i] = f2h(((float)(rand() % 2001) - 1000.0f) / 1000.0f);
        for (int i = 0; i < n; i++)
            hK[i] = f2h(((float)(rand() % 2001) - 1000.0f) / 1000.0f);
        for (int i = 0; i < n; i++)
            hV[i] = f2h(((float)(rand() % 2001) - 1000.0f) / 1000.0f);
        float scale = 1.0f / sqrtf((float)dim);
        for (int h = 0; h < heads; h++)
        {
            int off = h * seq * dim;
            for (int q = 0; q < seq; q++)
            {
                float rm = -1e30f;
                float *sc = (float *)calloc(seq, 4);
                for (int k = 0; k <= q; k++)
                {
                    float d = 0;
                    for (int d2 = 0; d2 < dim; d2++)
                        d += h2f(hQ[off + q * dim + d2]) * h2f(hK[off + k * dim + d2]);
                    sc[k] = d * scale;
                    rm = fmaxf(rm, sc[k]);
                }
                float sm = 0;
                for (int k = 0; k <= q; k++)
                {
                    sc[k] = expf(sc[k] - rm);
                    sm += sc[k];
                }
                for (int d2 = 0; d2 < dim; d2++)
                {
                    float a = 0;
                    for (int k = 0; k <= q; k++)
                        a += (sc[k] / sm) * h2f(hV[off + k * dim + d2]);
                    ref[off + q * dim + d2] = a;
                }
                free(sc);
            }
        }
        void *dQ, *dK, *dV;
        __half *dO20, *dO21;
        CK(cudaMalloc(&dQ, sz));
        CK(cudaMalloc(&dK, sz));
        CK(cudaMalloc(&dV, sz));
        CK(cudaMalloc(&dO20, sz));
        CK(cudaMalloc(&dO21, sz));
        CK(cudaMemcpy(dQ, hQ, sz, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dK, hK, sz, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dV, hV, sz, cudaMemcpyHostToDevice));
        CK(cudaMemset(dO20, 0, sz));
        CK(cudaMemset(dO21, 0, sz));
        launch_v20((const __half *)dQ, (const __half *)dK, (const __half *)dV, dO20, heads, seq, dim, 1);
        launch_v21((const __half *)dQ, (const __half *)dK, (const __half *)dV, dO21, heads, seq, dim, 1);
        CK(cudaDeviceSynchronize());
        uint16_t *hO20 = (uint16_t *)malloc(sz), *hO21 = (uint16_t *)malloc(sz);
        CK(cudaMemcpy(hO20, dO20, sz, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hO21, dO21, sz, cudaMemcpyDeviceToHost));
        float mx20 = 0, mx21 = 0;
        int e20 = 0, e21 = 0;
        for (int i = 0; i < n; i++)
        {
            float a20 = fabsf(h2f(hO20[i]) - ref[i]), a21 = fabsf(h2f(hO21[i]) - ref[i]);
            if (a20 > mx20)
                mx20 = a20;
            if (a21 > mx21)
                mx21 = a21;
            float thr = fmaxf(0.002f, fabsf(ref[i]) * 0.05f);
            if (a20 > thr)
                e20++;
            if (a21 > thr)
                e21++;
        }
        printf("  h=%d s=%4d  v20:max=%.4f err=%d %s  v21:max=%.4f err=%d %s\n",
               heads, seq, mx20, e20, e20 == 0 ? "PASS" : "FAIL", mx21, e21, e21 == 0 ? "PASS" : "FAIL");
        cudaFree(dQ);
        cudaFree(dK);
        cudaFree(dV);
        cudaFree(dO20);
        cudaFree(dO21);
        free(hQ);
        free(hK);
        free(hV);
        free(ref);
        free(hO20);
        free(hO21);
    }
}

void bench()
{
    printf("\n--- Performance: v20 (__expf) vs v21 (exp2f) ---\n");
    printf("%-12s %10s %10s %8s %7s\n", "Config", "v20 (T)", "v21 (T)", "delta", "peak%");
    printf("------------------------------------------------------\n");
    struct
    {
        int b, h, s, d;
        const char *l;
    } configs[] = {
        {1, 32, 256, 128, "7B-256"},
        {1, 32, 512, 128, "7B-512"},
        {1, 32, 1024, 128, "7B-1K"},
        {1, 32, 2048, 128, "7B-2K"},
        {1, 32, 4096, 128, "7B-4K"},
        {1, 32, 8192, 128, "7B-8K"},
        {1, 40, 512, 128, "13B-512"},
        {1, 64, 512, 128, "70B-512"},
        {1, 64, 2048, 128, "70B-2K"},
        {1, 64, 4096, 128, "70B-4K"},
    };
    Timer t;
    for (auto &c : configs)
    {
        int n = c.b * c.h * c.s * c.d, th = c.b * c.h;
        double flops = th * (4.0 * c.s * (double)c.s * c.d) / 2.0;
        void *dQ, *dK, *dV;
        __half *dO;
        CK(cudaMalloc(&dQ, (size_t)n * 2));
        CK(cudaMalloc(&dK, (size_t)n * 2));
        CK(cudaMalloc(&dV, (size_t)n * 2));
        CK(cudaMalloc(&dO, (size_t)n * 2));
        fill_random_fp16((uint16_t *)dQ, n);
        fill_random_fp16((uint16_t *)dK, n);
        fill_random_fp16((uint16_t *)dV, n);
        int it = (c.s <= 1024) ? 100 : (c.s <= 4096 ? 20 : 10);
        for (int i = 0; i < 3; i++)
            launch_v20((const __half *)dQ, (const __half *)dK, (const __half *)dV, dO, th, c.s, c.d, 1);
        CK(cudaDeviceSynchronize());
        t.start();
        for (int i = 0; i < it; i++)
            launch_v20((const __half *)dQ, (const __half *)dK, (const __half *)dV, dO, th, c.s, c.d, 1);
        float ms20 = t.stop();
        double tf20 = flops / (ms20 / it / 1000.0) / 1e12;
        CK(cudaMemset(dO, 0, (size_t)n * 2));
        for (int i = 0; i < 3; i++)
            launch_v21((const __half *)dQ, (const __half *)dK, (const __half *)dV, dO, th, c.s, c.d, 1);
        CK(cudaDeviceSynchronize());
        t.start();
        for (int i = 0; i < it; i++)
            launch_v21((const __half *)dQ, (const __half *)dK, (const __half *)dV, dO, th, c.s, c.d, 1);
        float ms21 = t.stop();
        double tf21 = flops / (ms21 / it / 1000.0) / 1e12;
        printf("%-12s %8.2f T %8.2f T %+7.2f %6.1f%%", c.l, tf20, tf21, tf21 - tf20, tf21 / 165.2 * 100);
        if (tf21 > tf20 * 1.005)
            printf("  *");
        printf("\n");
        cudaFree(dQ);
        cudaFree(dK);
        cudaFree(dV);
        cudaFree(dO);
    }
    printf("\nPeak = 165.2 T | v20 record: 151T @ 7B-8K stock\n");
}

int main()
{
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("=== FlashAttention v21 — exp2f over v20 ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n\n", p.name, p.multiProcessorCount, p.clockRate / 1000);
    test_correctness();
    bench();
    return 0;
}
