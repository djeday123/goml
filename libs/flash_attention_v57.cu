// =============================================================================
// FlashAttention v57 — Unroll factor experiments on v54 base
// =============================================================================
// Test: different #pragma unroll values for QK^T inner loop
//
// v54 (baseline): both loops fully unrolled (#pragma unroll = 8)
//   for (ks=0; ks<8; ks++)      // #pragma unroll → 8
//     for (nt=0; nt<8; nt++)    // #pragma unroll → 8
//       ldm2 + mma
//
// v57a: outer ks unroll 4
// v57b: inner nt unroll 4
// v57c: both unroll 4
//
// Hypothesis: full unroll = 64 LDSM + 64 HMMA = 128 instructions unrolled.
// Instruction cache on SM89 = 128KB. If kernel exceeds icache, thrashing.
// Partial unroll reduces code size, may improve icache hit rate.
//
// Build: nvcc -O3 -arch=sm_89 -std=c++17 flash_attention_v57.cu -o fa_v57 -lcudart
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define FA_BR 64
#define FA_BC 64
#define FA_THREADS 128
#define FA_STRIDE 128

// Helpers
__device__ __forceinline__ int swz(int r, int c) { return (((c >> 3) ^ (r & 7)) << 3) | (c & 7); }
__device__ __forceinline__ void cpa16(void *s, const void *g, int n)
{
    uint32_t sa = __cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" ::"r"(sa), "l"(g), "r"(n));
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }
__device__ __forceinline__ void ldm4(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, const void *p)
{
    uint32_t a = __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldm2(uint32_t &r0, uint32_t &r1, const void *p)
{
    uint32_t a = __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1},[%2];" : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void ldm2t(uint32_t &r0, uint32_t &r1, const void *p)
{
    uint32_t a = __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16.trans {%0,%1},[%2];" : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void mma16816(float &d0, float &d1, float &d2, float &d3, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1, float c0, float c1, float c2, float c3)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}
__device__ __forceinline__ void ld_a_sw(uint32_t &a0, uint32_t &a1, uint32_t &a2, uint32_t &a3, const __half *sm, int stride, int rb, int kb, int lane)
{
    int sub = lane / 8, sr = lane % 8;
    int r = rb + (sub & 1) * 8 + sr, lc = kb + (sub >> 1) * 8;
    ldm4(a0, a1, a2, a3, &sm[r * stride + swz(r, lc)]);
}
__device__ __forceinline__ void ld_b_sw(uint32_t &b0, uint32_t &b1, const __half *sm, int stride, int nb, int kb, int lane)
{
    int sub = (lane / 8) % 2, sr = lane % 8;
    int r = nb + sr, lc = kb + sub * 8;
    ldm2(b0, b1, &sm[r * stride + swz(r, lc)]);
}
__device__ __forceinline__ void ld_b_vt(uint32_t &b0, uint32_t &b1, const __half *sm, int stride, int nb, int kb, int lane)
{
    int sub = (lane / 8) % 2, sr = lane % 8;
    int k = kb + sub * 8 + sr;
    ldm2t(b0, b1, &sm[k * stride + swz(k, nb)]);
}
__device__ __forceinline__ void load_tile(__half *dst, const __half *src, int start, int rows, int seq_len, int hd)
{
    constexpr int CPR = 16;
    int total = rows * CPR;
#pragma unroll 4
    for (int c = threadIdx.x; c < total; c += FA_THREADS)
    {
        int row = c / CPR, lch = c % CPR, gr = start + row;
        int pch = lch ^ (row & 7);
        cpa16(&dst[row * FA_STRIDE + pch * 8], &src[gr * hd + lch * 8], (gr < seq_len) ? 16 : 0);
    }
}
__device__ __forceinline__ uint32_t hexp2x2(uint32_t in)
{
    uint32_t out;
    asm("ex2.approx.f16x2 %0,%1;" : "=r"(out) : "r"(in));
    return out;
}

// Macro for softmax + PV (identical across all variants)
#define SOFTMAX_AND_PV()                                                                                                                                          \
    for (int nt = 0; nt < 8; nt++)                                                                                                                                \
    {                                                                                                                                                             \
        Sr[nt][0] *= scale;                                                                                                                                       \
        Sr[nt][1] *= scale;                                                                                                                                       \
        Sr[nt][2] *= scale;                                                                                                                                       \
        Sr[nt][3] *= scale;                                                                                                                                       \
        if (causal)                                                                                                                                               \
        {                                                                                                                                                         \
            int gq0 = qs + mrb + gid, gq8 = gq0 + 8;                                                                                                              \
            int gk0 = kvs + nt * 8 + tid * 2, gk1 = gk0 + 1;                                                                                                      \
            if (gk0 > gq0)                                                                                                                                        \
                Sr[nt][0] = -1e30f;                                                                                                                               \
            if (gk1 > gq0)                                                                                                                                        \
                Sr[nt][1] = -1e30f;                                                                                                                               \
            if (gk0 > gq8)                                                                                                                                        \
                Sr[nt][2] = -1e30f;                                                                                                                               \
            if (gk1 > gq8)                                                                                                                                        \
                Sr[nt][3] = -1e30f;                                                                                                                               \
            if (gq0 >= seq_len)                                                                                                                                   \
            {                                                                                                                                                     \
                Sr[nt][0] = -1e30f;                                                                                                                               \
                Sr[nt][1] = -1e30f;                                                                                                                               \
            }                                                                                                                                                     \
            if (gq8 >= seq_len)                                                                                                                                   \
            {                                                                                                                                                     \
                Sr[nt][2] = -1e30f;                                                                                                                               \
                Sr[nt][3] = -1e30f;                                                                                                                               \
            }                                                                                                                                                     \
            if (gk0 >= seq_len)                                                                                                                                   \
            {                                                                                                                                                     \
                Sr[nt][0] = -1e30f;                                                                                                                               \
                Sr[nt][2] = -1e30f;                                                                                                                               \
            }                                                                                                                                                     \
            if (gk1 >= seq_len)                                                                                                                                   \
            {                                                                                                                                                     \
                Sr[nt][1] = -1e30f;                                                                                                                               \
                Sr[nt][3] = -1e30f;                                                                                                                               \
            }                                                                                                                                                     \
        }                                                                                                                                                         \
    }                                                                                                                                                             \
    float nm[2] = {-1e30f, -1e30f};                                                                                                                               \
    for (int nt = 0; nt < 8; nt++)                                                                                                                                \
    {                                                                                                                                                             \
        nm[0] = fmaxf(nm[0], fmaxf(Sr[nt][0], Sr[nt][1]));                                                                                                        \
        nm[1] = fmaxf(nm[1], fmaxf(Sr[nt][2], Sr[nt][3]));                                                                                                        \
    }                                                                                                                                                             \
    nm[0] = fmaxf(nm[0], __shfl_xor_sync(0xffffffff, nm[0], 1));                                                                                                  \
    nm[0] = fmaxf(nm[0], __shfl_xor_sync(0xffffffff, nm[0], 2));                                                                                                  \
    nm[1] = fmaxf(nm[1], __shfl_xor_sync(0xffffffff, nm[1], 1));                                                                                                  \
    nm[1] = fmaxf(nm[1], __shfl_xor_sync(0xffffffff, nm[1], 2));                                                                                                  \
    nm[0] = fmaxf(nm[0], rmax[0]);                                                                                                                                \
    nm[1] = fmaxf(nm[1], rmax[1]);                                                                                                                                \
    __half h_d0 = __float2half(rmax[0] - nm[0]), h_d1 = __float2half(rmax[1] - nm[1]);                                                                            \
    __half2 h2d = __halves2half2(h_d0, h_d1);                                                                                                                     \
    h2d = __hmul2(h2d, h2_log2e);                                                                                                                                 \
    uint32_t rsc_p = hexp2x2(*reinterpret_cast<uint32_t *>(&h2d));                                                                                                \
    __half2 h2r = *reinterpret_cast<__half2 *>(&rsc_p);                                                                                                           \
    float rsc0 = __half2float(__low2half(h2r)), rsc1 = __half2float(__high2half(h2r));                                                                            \
    for (int t = 0; t < 16; t++)                                                                                                                                  \
    {                                                                                                                                                             \
        Or[t][0] *= rsc0;                                                                                                                                         \
        Or[t][1] *= rsc0;                                                                                                                                         \
        Or[t][2] *= rsc1;                                                                                                                                         \
        Or[t][3] *= rsc1;                                                                                                                                         \
    }                                                                                                                                                             \
    rmax[0] = nm[0];                                                                                                                                              \
    rmax[1] = nm[1];                                                                                                                                              \
    float ns[2] = {0.0f, 0.0f};                                                                                                                                   \
    uint32_t Pr[4][4];                                                                                                                                            \
    for (int nt = 0; nt < 8; nt++)                                                                                                                                \
    {                                                                                                                                                             \
        __half2 h01 = __halves2half2(__float2half(Sr[nt][0] - rmax[0]), __float2half(Sr[nt][1] - rmax[0]));                                                       \
        h01 = __hmul2(h01, h2_log2e);                                                                                                                             \
        uint32_t e01 = hexp2x2(*reinterpret_cast<uint32_t *>(&h01));                                                                                              \
        __half2 r01 = *reinterpret_cast<__half2 *>(&e01);                                                                                                         \
        __half2 h23 = __halves2half2(__float2half(Sr[nt][2] - rmax[1]), __float2half(Sr[nt][3] - rmax[1]));                                                       \
        h23 = __hmul2(h23, h2_log2e);                                                                                                                             \
        uint32_t e23 = hexp2x2(*reinterpret_cast<uint32_t *>(&h23));                                                                                              \
        __half2 r23 = *reinterpret_cast<__half2 *>(&e23);                                                                                                         \
        ns[0] += __half2float(__low2half(r01)) + __half2float(__high2half(r01));                                                                                  \
        ns[1] += __half2float(__low2half(r23)) + __half2float(__high2half(r23));                                                                                  \
        int pi = nt / 2, half = nt % 2;                                                                                                                           \
        __half2 *p = (__half2 *)Pr[pi];                                                                                                                           \
        p[half * 2] = r01;                                                                                                                                        \
        p[half * 2 + 1] = r23;                                                                                                                                    \
    }                                                                                                                                                             \
    ns[0] += __shfl_xor_sync(0xffffffff, ns[0], 1);                                                                                                               \
    ns[0] += __shfl_xor_sync(0xffffffff, ns[0], 2);                                                                                                               \
    ns[1] += __shfl_xor_sync(0xffffffff, ns[1], 1);                                                                                                               \
    ns[1] += __shfl_xor_sync(0xffffffff, ns[1], 2);                                                                                                               \
    rsexp[0] = rsexp[0] * rsc0 + ns[0];                                                                                                                           \
    rsexp[1] = rsexp[1] * rsc1 + ns[1];                                                                                                                           \
    cpa_wait<1>();                                                                                                                                                \
    __syncthreads();                                                                                                                                              \
    _Pragma("unroll") for (int ks = 0; ks < 4; ks++)                                                                                                              \
    {                                                                                                                                                             \
        _Pragma("unroll") for (int nt = 0; nt < 16; nt++)                                                                                                         \
        {                                                                                                                                                         \
            uint32_t b0, b1;                                                                                                                                      \
            ld_b_vt(b0, b1, cur, FA_STRIDE, nt * 8, ks * 16, lane);                                                                                               \
            mma16816(Or[nt][0], Or[nt][1], Or[nt][2], Or[nt][3], Pr[ks][0], Pr[ks][1], Pr[ks][2], Pr[ks][3], b0, b1, Or[nt][0], Or[nt][1], Or[nt][2], Or[nt][3]); \
        }                                                                                                                                                         \
    }

// Kernel preamble macro
#define KERNEL_PREAMBLE()                                                                                                                      \
    int nqt = (seq_len + FA_BR - 1) / FA_BR;                                                                                                   \
    int bh = blockIdx.x / nqt, qt = blockIdx.x % nqt, qs = qt * FA_BR;                                                                         \
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;                                                                                       \
    int gid = lane >> 2, tid = lane & 3;                                                                                                       \
    extern __shared__ char raw[];                                                                                                              \
    __half *buf0 = (__half *)raw;                                                                                                              \
    __half *buf1 = (__half *)(raw + FA_BC * FA_STRIDE * sizeof(__half));                                                                       \
    int hs = seq_len * head_dim;                                                                                                               \
    const __half *Qh = Q + bh * hs, *Kh = K + bh * hs, *Vh = V + bh * hs;                                                                      \
    __half *Oh = O + bh * hs;                                                                                                                  \
    load_tile(buf0, Qh, qs, FA_BR, seq_len, head_dim);                                                                                         \
    cpa_commit();                                                                                                                              \
    cpa_wait<0>();                                                                                                                             \
    __syncthreads();                                                                                                                           \
    int mrb = wid * 16;                                                                                                                        \
    uint32_t Qr[8][4];                                                                                                                         \
    _Pragma("unroll") for (int ks = 0; ks < 8; ks++) ld_a_sw(Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3], buf0, FA_STRIDE, mrb, ks * 16, lane); \
    __syncthreads();                                                                                                                           \
    float Or[16][4];                                                                                                                           \
    _Pragma("unroll") for (int t = 0; t < 16; t++) Or[t][0] = Or[t][1] = Or[t][2] = Or[t][3] = 0;                                              \
    float rmax[2] = {-1e30f, -1e30f}, rsexp[2] = {0.0f, 0.0f};                                                                                 \
    int nkv = (seq_len + FA_BC - 1) / FA_BC;                                                                                                   \
    load_tile(buf0, Kh, 0, FA_BC, seq_len, head_dim);                                                                                          \
    cpa_commit();                                                                                                                              \
    const __half2 h2_log2e = __float2half2_rn(1.4426950408889634f);

// Kernel epilogue macro
#define KERNEL_EPILOGUE()                                                                               \
    float li0 = (rsexp[0] > 0) ? 1.0f / rsexp[0] : 0.0f, li1 = (rsexp[1] > 0) ? 1.0f / rsexp[1] : 0.0f; \
    int gr0 = qs + mrb + gid, gr8 = gr0 + 8;                                                            \
    _Pragma("unroll") for (int nt = 0; nt < 16; nt++)                                                   \
    {                                                                                                   \
        int c0 = nt * 8 + tid * 2, c1 = c0 + 1;                                                         \
        if (gr0 < seq_len && c0 < head_dim)                                                             \
            Oh[gr0 * head_dim + c0] = __float2half(Or[nt][0] * li0);                                    \
        if (gr0 < seq_len && c1 < head_dim)                                                             \
            Oh[gr0 * head_dim + c1] = __float2half(Or[nt][1] * li0);                                    \
        if (gr8 < seq_len && c0 < head_dim)                                                             \
            Oh[gr8 * head_dim + c0] = __float2half(Or[nt][2] * li1);                                    \
        if (gr8 < seq_len && c1 < head_dim)                                                             \
            Oh[gr8 * head_dim + c1] = __float2half(Or[nt][3] * li1);                                    \
    }

// Loop body start macro
#define LOOP_START()                                                         \
    for (int kv = 0; kv < nkv; kv++)                                         \
    {                                                                        \
        int kvs = kv * FA_BC;                                                \
        if (causal && kvs > qs + FA_BR - 1)                                  \
            break;                                                           \
        __half *cur = (kv & 1) ? buf1 : buf0, *nxt = (kv & 1) ? buf0 : buf1; \
        cpa_wait<0>();                                                       \
        __syncthreads();                                                     \
        float Sr[8][4];                                                      \
        _Pragma("unroll") for (int nt = 0; nt < 8; nt++) Sr[nt][0] = Sr[nt][1] = Sr[nt][2] = Sr[nt][3] = 0;

// After QK^T, before softmax
#define AFTER_QKT()                                                       \
    __syncthreads();                                                      \
    load_tile(cur, Vh, kvs, FA_BC, seq_len, head_dim);                    \
    cpa_commit();                                                         \
    int nkvs = (kv + 1) * FA_BC;                                          \
    bool has_nxt = (kv + 1 < nkv) && (!causal || nkvs <= qs + FA_BR - 1); \
    if (has_nxt)                                                          \
        load_tile(nxt, Kh, nkvs, FA_BC, seq_len, head_dim);               \
    cpa_commit();

// =============================================================================
// v54: full unroll (baseline)
// =============================================================================
__global__ void __launch_bounds__(FA_THREADS, 2)
    fa54_kernel(const __half *__restrict__ Q, const __half *__restrict__ K, const __half *__restrict__ V, __half *__restrict__ O, int seq_len, int head_dim, int causal, float scale)
{
    KERNEL_PREAMBLE()
    LOOP_START()
#pragma unroll
    for (int ks = 0; ks < 8; ks++)
    {
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            uint32_t b0, b1;
            ld_b_sw(b0, b1, cur, FA_STRIDE, nt * 8, ks * 16, lane);
            mma16816(Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3], Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3], b0, b1, Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3]);
        }
    }
    AFTER_QKT()
    SOFTMAX_AND_PV()
}
KERNEL_EPILOGUE()
}

// =============================================================================
// v57a: outer ks unroll 4, inner nt full
// =============================================================================
__global__ void __launch_bounds__(FA_THREADS, 2)
    fa57a_kernel(const __half *__restrict__ Q, const __half *__restrict__ K, const __half *__restrict__ V, __half *__restrict__ O, int seq_len, int head_dim, int causal, float scale)
{
    KERNEL_PREAMBLE()
    LOOP_START()
#pragma unroll 4
    for (int ks = 0; ks < 8; ks++)
    {
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            uint32_t b0, b1;
            ld_b_sw(b0, b1, cur, FA_STRIDE, nt * 8, ks * 16, lane);
            mma16816(Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3], Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3], b0, b1, Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3]);
        }
    }
    AFTER_QKT()
    SOFTMAX_AND_PV()
}
KERNEL_EPILOGUE()
}

// =============================================================================
// v57b: outer ks full, inner nt unroll 4
// =============================================================================
__global__ void __launch_bounds__(FA_THREADS, 2)
    fa57b_kernel(const __half *__restrict__ Q, const __half *__restrict__ K, const __half *__restrict__ V, __half *__restrict__ O, int seq_len, int head_dim, int causal, float scale)
{
    KERNEL_PREAMBLE()
    LOOP_START()
#pragma unroll
    for (int ks = 0; ks < 8; ks++)
    {
#pragma unroll 4
        for (int nt = 0; nt < 8; nt++)
        {
            uint32_t b0, b1;
            ld_b_sw(b0, b1, cur, FA_STRIDE, nt * 8, ks * 16, lane);
            mma16816(Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3], Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3], b0, b1, Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3]);
        }
    }
    AFTER_QKT()
    SOFTMAX_AND_PV()
}
KERNEL_EPILOGUE()
}

// =============================================================================
// v57c: both unroll 4
// =============================================================================
__global__ void __launch_bounds__(FA_THREADS, 2)
    fa57c_kernel(const __half *__restrict__ Q, const __half *__restrict__ K, const __half *__restrict__ V, __half *__restrict__ O, int seq_len, int head_dim, int causal, float scale)
{
    KERNEL_PREAMBLE()
    LOOP_START()
#pragma unroll 4
    for (int ks = 0; ks < 8; ks++)
    {
#pragma unroll 4
        for (int nt = 0; nt < 8; nt++)
        {
            uint32_t b0, b1;
            ld_b_sw(b0, b1, cur, FA_STRIDE, nt * 8, ks * 16, lane);
            mma16816(Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3], Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3], b0, b1, Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3]);
        }
    }
    AFTER_QKT()
    SOFTMAX_AND_PV()
}
KERNEL_EPILOGUE()
}

// =============================================================================
// v57d: outer unroll 2, inner full
// =============================================================================
__global__ void __launch_bounds__(FA_THREADS, 2)
    fa57d_kernel(const __half *__restrict__ Q, const __half *__restrict__ K, const __half *__restrict__ V, __half *__restrict__ O, int seq_len, int head_dim, int causal, float scale)
{
    KERNEL_PREAMBLE()
    LOOP_START()
#pragma unroll 2
    for (int ks = 0; ks < 8; ks++)
    {
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            uint32_t b0, b1;
            ld_b_sw(b0, b1, cur, FA_STRIDE, nt * 8, ks * 16, lane);
            mma16816(Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3], Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3], b0, b1, Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3]);
        }
    }
    AFTER_QKT()
    SOFTMAX_AND_PV()
}
KERNEL_EPILOGUE()
}

// =============================================================================
// Benchmark
// =============================================================================
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
void fill_random_fp16(uint16_t *d, int n)
{
    uint16_t *h = (uint16_t *)malloc(n * 2);
    for (int i = 0; i < n; i++)
        h[i] = f2h(((float)(rand() % 2001) - 1000.0f) / 1000.0f);
    CK(cudaMemcpy(d, h, n * 2, cudaMemcpyHostToDevice));
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

typedef void (*kernel_t)(const __half *, const __half *, const __half *, __half *, int, int, int, float);
static int g_smem[8] = {};

void launch(kernel_t kern, int idx, const __half *Q, const __half *K, const __half *V, __half *O, int th, int sl, int hd, int ca)
{
    int sm = 2 * FA_BC * FA_STRIDE * (int)sizeof(__half);
    if (sm > g_smem[idx])
    {
        CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
        g_smem[idx] = sm;
    }
    int nqt = (sl + FA_BR - 1) / FA_BR;
    kern<<<th * nqt, FA_THREADS, sm>>>(Q, K, V, O, sl, hd, ca, 1.0f / sqrtf((float)hd));
}

double bench_kernel(kernel_t kern, int idx, int th, int sl, int hd, int iters)
{
    int n = th * sl * hd;
    double flops = th * (4.0 * sl * (double)sl * hd) / 2.0;
    void *dQ, *dK, *dV;
    __half *dO;
    CK(cudaMalloc(&dQ, (size_t)n * 2));
    CK(cudaMalloc(&dK, (size_t)n * 2));
    CK(cudaMalloc(&dV, (size_t)n * 2));
    CK(cudaMalloc(&dO, (size_t)n * 2));
    fill_random_fp16((uint16_t *)dQ, n);
    fill_random_fp16((uint16_t *)dK, n);
    fill_random_fp16((uint16_t *)dV, n);
    for (int i = 0; i < 3; i++)
        launch(kern, idx, (const __half *)dQ, (const __half *)dK, (const __half *)dV, dO, th, sl, hd, 1);
    CK(cudaDeviceSynchronize());
    Timer t;
    t.start();
    for (int i = 0; i < iters; i++)
        launch(kern, idx, (const __half *)dQ, (const __half *)dK, (const __half *)dV, dO, th, sl, hd, 1);
    float ms = t.stop();
    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
    return flops / (ms / iters / 1000.0) / 1e12;
}

int main()
{
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("=== FlashAttention v57 — Unroll factor experiments ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n\n", p.name, p.multiProcessorCount, p.clockRate / 1000);

    struct
    {
        int h, s, d;
        int it;
        const char *l;
    } configs[] = {
        {32, 512, 128, 100, "7B-512"},
        {32, 2048, 128, 20, "7B-2K"},
        {32, 4096, 128, 20, "7B-4K"},
        {32, 8192, 128, 10, "7B-8K"},
        {64, 2048, 128, 20, "70B-2K"},
    };

    kernel_t kerns[] = {fa54_kernel, fa57a_kernel, fa57b_kernel, fa57c_kernel, fa57d_kernel};
    const char *names[] = {"v54(8,8)", "v57a(4,8)", "v57b(8,4)", "v57c(4,4)", "v57d(2,8)"};
    int nk = 5;

    printf("%-10s", "Config");
    for (int k = 0; k < nk; k++)
        printf(" %11s", names[k]);
    printf("\n");
    for (int i = 0; i < 10 + nk * 12; i++)
        printf("-");
    printf("\n");

    for (auto &c : configs)
    {
        printf("%-10s", c.l);
        double best = 0;
        double results[5];
        for (int k = 0; k < nk; k++)
        {
            results[k] = bench_kernel(kerns[k], k, c.h, c.s, c.d, c.it);
            if (results[k] > best)
                best = results[k];
        }
        for (int k = 0; k < nk; k++)
        {
            printf(" %7.2f T", results[k]);
            if (results[k] >= best - 0.3)
                printf("*");
            else
                printf(" ");
            printf(" ");
        }
        printf("\n");
    }
    printf("\n* = best (within 0.3T)\n");
    printf("Unroll format: (outer_ks, inner_nt)\n");
}