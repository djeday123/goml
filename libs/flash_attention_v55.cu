// =============================================================================
// FlashAttention v55 — FP16 polynomial exp2 on CUDA cores
// =============================================================================
//
// v54 used MUFU.EX2.F16 (SFU): 136 Gops/s
// v55 uses HFMA2 polynomial:   236 Gops/s (1.74x faster in isolation)
//
// exp(x) = exp2(x * log2e)
// For softmax: x = score - max, always <= 0
// So x * log2e is in [-inf, 0]
//
// Split: exp2(x) = 2^floor(x) * exp2(frac(x))
// exp2(frac) ≈ polynomial on [0, 1)
// 2^floor = bit manipulation on FP16 exponent
//
// FP16 format: 1 sign | 5 exponent | 10 mantissa, bias=15
// 2^n for n in [-14, 15]: set exponent = n + 15
// For n < -14: subnormal or zero (which is fine for softmax — means ~0)
//
// Build:
//   nvcc -O3 -arch=sm_89 -std=c++17 flash_attention_v55.cu -o fa_v55 -lcudart
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

// =============================================================================
// Helpers (same as v54/v20)
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
__device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }
__device__ __forceinline__ void ldm4(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, const void *p)
{
    uint32_t a = __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
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
__device__ __forceinline__ void ld_a_sw(uint32_t &a0, uint32_t &a1, uint32_t &a2, uint32_t &a3,
                                        const __half *sm, int stride, int rb, int kb, int lane)
{
    int sub = lane / 8, sr = lane % 8;
    int r = rb + (sub & 1) * 8 + sr, lc = kb + (sub >> 1) * 8;
    ldm4(a0, a1, a2, a3, &sm[r * stride + swz(r, lc)]);
}
__device__ __forceinline__ void ld_b_sw(uint32_t &b0, uint32_t &b1,
                                        const __half *sm, int stride, int nb, int kb, int lane)
{
    int sub = (lane / 8) % 2, sr = lane % 8;
    int r = nb + sr, lc = kb + sub * 8;
    ldm2(b0, b1, &sm[r * stride + swz(r, lc)]);
}
__device__ __forceinline__ void ld_b_vt(uint32_t &b0, uint32_t &b1,
                                        const __half *sm, int stride, int nb, int kb, int lane)
{
    int sub = (lane / 8) % 2, sr = lane % 8;
    int k = kb + sub * 8 + sr;
    ldm2t(b0, b1, &sm[k * stride + swz(k, nb)]);
}
__device__ __forceinline__ void load_tile(
    __half *dst, const __half *src, int start, int rows, int seq_len, int head_dim)
{
    constexpr int CPR = 16;
    int total = rows * CPR;
#pragma unroll 4
    for (int c = threadIdx.x; c < total; c += FA_THREADS)
    {
        int row = c / CPR, lch = c % CPR, gr = start + row;
        int pch = lch ^ (row & 7);
        cpa16(&dst[row * FA_STRIDE + pch * 8], &src[gr * head_dim + lch * 8], (gr < seq_len) ? 16 : 0);
    }
}

// =============================================================================
// FP16 polynomial exp2 via HFMA2 on CUDA cores
// =============================================================================
// exp2(x) for x in (-inf, 0]:
//   floor_x = hfloor(x)       — integer part
//   frac_x  = x - floor_x     — fractional part in [0, 1)
//   poly    = 1 + f*(c1 + f*(c2 + f*c3))  — approx 2^frac
//   2^floor = ldexp via FP16 exponent manipulation
//   result  = 2^floor * poly
//
// For softmax, values that would underflow to 0 are fine (exp(-big) ≈ 0)

__device__ __forceinline__ __half poly_exp2_h(__half x)
{
    // For very negative x, just return 0
    if (__hlt(x, __float2half(-14.0f)))
        return __float2half(0.0f);

    // Split into integer and fractional parts
    __half fl = hfloor(x);
    __half fr = __hsub(x, fl);

    // Polynomial: 2^fr ≈ 1 + fr*(c1 + fr*(c2 + fr*c3))
    // Coefficients for cubic Horner on [0,1)
    const __half c3 = __float2half(0.0558011f);
    const __half c2 = __float2half(0.2402265f);
    const __half c1 = __float2half(0.6931472f);
    const __half one = __float2half(1.0f);

    __half p = __hfma(fr, c3, c2);
    p = __hfma(fr, p, c1);
    p = __hfma(fr, p, one);

    // 2^floor via exponent manipulation
    // FP16: sign(1) | exp(5) | mant(10), bias=15
    // 2^n = exponent field = (n+15), mantissa = 0
    int n = __half2int_rd(fl);
    if (n < -14)
        return __float2half(0.0f);
    unsigned short pow2 = (unsigned short)((n + 15) << 10);
    __half h_pow2 = __ushort_as_half(pow2);

    return __hmul(h_pow2, p);
}

// Vectorized half2 version
__device__ __forceinline__ __half2 poly_exp2_h2(__half2 x)
{
    __half lo = poly_exp2_h(__low2half(x));
    __half hi = poly_exp2_h(__high2half(x));
    return __halves2half2(lo, hi);
}

// MUFU version for v54 comparison
__device__ __forceinline__ uint32_t hexp2x2(uint32_t in)
{
    uint32_t out;
    asm("ex2.approx.f16x2 %0, %1;" : "=r"(out) : "r"(in));
    return out;
}

// =============================================================================
// v54 kernel (FP16 MUFU.EX2.F16 — baseline)
// =============================================================================
__global__ void __launch_bounds__(FA_THREADS, 2)
    fa54_kernel(const __half *__restrict__ Q, const __half *__restrict__ K,
                const __half *__restrict__ V, __half *__restrict__ O,
                int seq_len, int head_dim, int causal, float scale)
{
    int nqt = (seq_len + FA_BR - 1) / FA_BR;
    int bh = blockIdx.x / nqt, qt = blockIdx.x % nqt, qs = qt * FA_BR;
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    int gid = lane >> 2, tid = lane & 3;
    extern __shared__ char raw[];
    __half *buf0 = (__half *)raw;
    __half *buf1 = (__half *)(raw + FA_BC * FA_STRIDE * sizeof(__half));
    int hs = seq_len * head_dim;
    const __half *Qh = Q + bh * hs, *Kh = K + bh * hs, *Vh = V + bh * hs;
    __half *Oh = O + bh * hs;

    load_tile(buf0, Qh, qs, FA_BR, seq_len, head_dim);
    cpa_commit();
    cpa_wait<0>();
    __syncthreads();
    int mrb = wid * 16;
    uint32_t Qr[8][4];
#pragma unroll
    for (int ks = 0; ks < 8; ks++)
        ld_a_sw(Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3], buf0, FA_STRIDE, mrb, ks * 16, lane);
    __syncthreads();

    float Or[16][4];
#pragma unroll
    for (int t = 0; t < 16; t++)
        Or[t][0] = Or[t][1] = Or[t][2] = Or[t][3] = 0;
    float rmax[2] = {-1e30f, -1e30f}, rsexp[2] = {0.0f, 0.0f};
    int nkv = (seq_len + FA_BC - 1) / FA_BC;
    load_tile(buf0, Kh, 0, FA_BC, seq_len, head_dim);
    cpa_commit();
    const __half2 h2_log2e = __float2half2_rn(1.4426950408889634f);

    for (int kv = 0; kv < nkv; kv++)
    {
        int kvs = kv * FA_BC;
        if (causal && kvs > qs + FA_BR - 1)
            break;
        __half *cur = (kv & 1) ? buf1 : buf0, *nxt = (kv & 1) ? buf0 : buf1;
        cpa_wait<0>();
        __syncthreads();

        float Sr[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
            Sr[nt][0] = Sr[nt][1] = Sr[nt][2] = Sr[nt][3] = 0;
#pragma unroll
        for (int ks = 0; ks < 8; ks++)
        {
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
            {
                uint32_t b0, b1;
                ld_b_sw(b0, b1, cur, FA_STRIDE, nt * 8, ks * 16, lane);
                mma16816(Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3],
                         Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3],
                         b0, b1, Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3]);
            }
        }
        __syncthreads();
        load_tile(cur, Vh, kvs, FA_BC, seq_len, head_dim);
        cpa_commit();
        int nkvs = (kv + 1) * FA_BC;
        bool has_nxt = (kv + 1 < nkv) && (!causal || nkvs <= qs + FA_BR - 1);
        if (has_nxt)
            load_tile(nxt, Kh, nkvs, FA_BC, seq_len, head_dim);
        cpa_commit();

#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            Sr[nt][0] *= scale;
            Sr[nt][1] *= scale;
            Sr[nt][2] *= scale;
            Sr[nt][3] *= scale;
            if (causal)
            {
                int gq0 = qs + mrb + gid, gq8 = gq0 + 8;
                int gk0 = kvs + nt * 8 + tid * 2, gk1 = gk0 + 1;
                if (gk0 > gq0)
                    Sr[nt][0] = -1e30f;
                if (gk1 > gq0)
                    Sr[nt][1] = -1e30f;
                if (gk0 > gq8)
                    Sr[nt][2] = -1e30f;
                if (gk1 > gq8)
                    Sr[nt][3] = -1e30f;
                if (gq0 >= seq_len)
                {
                    Sr[nt][0] = -1e30f;
                    Sr[nt][1] = -1e30f;
                }
                if (gq8 >= seq_len)
                {
                    Sr[nt][2] = -1e30f;
                    Sr[nt][3] = -1e30f;
                }
                if (gk0 >= seq_len)
                {
                    Sr[nt][0] = -1e30f;
                    Sr[nt][2] = -1e30f;
                }
                if (gk1 >= seq_len)
                {
                    Sr[nt][1] = -1e30f;
                    Sr[nt][3] = -1e30f;
                }
            }
        }

        float nm[2] = {-1e30f, -1e30f};
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            nm[0] = fmaxf(nm[0], fmaxf(Sr[nt][0], Sr[nt][1]));
            nm[1] = fmaxf(nm[1], fmaxf(Sr[nt][2], Sr[nt][3]));
        }
        nm[0] = fmaxf(nm[0], __shfl_xor_sync(0xffffffff, nm[0], 1));
        nm[0] = fmaxf(nm[0], __shfl_xor_sync(0xffffffff, nm[0], 2));
        nm[1] = fmaxf(nm[1], __shfl_xor_sync(0xffffffff, nm[1], 1));
        nm[1] = fmaxf(nm[1], __shfl_xor_sync(0xffffffff, nm[1], 2));
        nm[0] = fmaxf(nm[0], rmax[0]);
        nm[1] = fmaxf(nm[1], rmax[1]);

        __half h_d0 = __float2half(rmax[0] - nm[0]), h_d1 = __float2half(rmax[1] - nm[1]);
        __half2 h2d = __halves2half2(h_d0, h_d1);
        h2d = __hmul2(h2d, h2_log2e);
        uint32_t rsc_p = hexp2x2(*reinterpret_cast<uint32_t *>(&h2d));
        __half2 h2r = *reinterpret_cast<__half2 *>(&rsc_p);
        float rsc0 = __half2float(__low2half(h2r)), rsc1 = __half2float(__high2half(h2r));

#pragma unroll
        for (int t = 0; t < 16; t++)
        {
            Or[t][0] *= rsc0;
            Or[t][1] *= rsc0;
            Or[t][2] *= rsc1;
            Or[t][3] *= rsc1;
        }
        rmax[0] = nm[0];
        rmax[1] = nm[1];

        float ns[2] = {0.0f, 0.0f};
        uint32_t Pr[4][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            __half2 h01 = __halves2half2(__float2half(Sr[nt][0] - rmax[0]), __float2half(Sr[nt][1] - rmax[0]));
            h01 = __hmul2(h01, h2_log2e);
            uint32_t e01 = hexp2x2(*reinterpret_cast<uint32_t *>(&h01));
            __half2 r01 = *reinterpret_cast<__half2 *>(&e01);
            __half2 h23 = __halves2half2(__float2half(Sr[nt][2] - rmax[1]), __float2half(Sr[nt][3] - rmax[1]));
            h23 = __hmul2(h23, h2_log2e);
            uint32_t e23 = hexp2x2(*reinterpret_cast<uint32_t *>(&h23));
            __half2 r23 = *reinterpret_cast<__half2 *>(&e23);
            ns[0] += __half2float(__low2half(r01)) + __half2float(__high2half(r01));
            ns[1] += __half2float(__low2half(r23)) + __half2float(__high2half(r23));
            int pi = nt / 2, half = nt % 2;
            __half2 *p = (__half2 *)Pr[pi];
            p[half * 2] = r01;
            p[half * 2 + 1] = r23;
        }
        ns[0] += __shfl_xor_sync(0xffffffff, ns[0], 1);
        ns[0] += __shfl_xor_sync(0xffffffff, ns[0], 2);
        ns[1] += __shfl_xor_sync(0xffffffff, ns[1], 1);
        ns[1] += __shfl_xor_sync(0xffffffff, ns[1], 2);
        rsexp[0] = rsexp[0] * rsc0 + ns[0];
        rsexp[1] = rsexp[1] * rsc1 + ns[1];

        cpa_wait<1>();
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < 4; ks++)
        {
#pragma unroll
            for (int nt = 0; nt < 16; nt++)
            {
                uint32_t b0, b1;
                ld_b_vt(b0, b1, cur, FA_STRIDE, nt * 8, ks * 16, lane);
                mma16816(Or[nt][0], Or[nt][1], Or[nt][2], Or[nt][3],
                         Pr[ks][0], Pr[ks][1], Pr[ks][2], Pr[ks][3],
                         b0, b1, Or[nt][0], Or[nt][1], Or[nt][2], Or[nt][3]);
            }
        }
    }
    float li0 = (rsexp[0] > 0) ? 1.0f / rsexp[0] : 0.0f, li1 = (rsexp[1] > 0) ? 1.0f / rsexp[1] : 0.0f;
    int gr0 = qs + mrb + gid, gr8 = gr0 + 8;
#pragma unroll
    for (int nt = 0; nt < 16; nt++)
    {
        int c0 = nt * 8 + tid * 2, c1 = c0 + 1;
        if (gr0 < seq_len && c0 < head_dim)
            Oh[gr0 * head_dim + c0] = __float2half(Or[nt][0] * li0);
        if (gr0 < seq_len && c1 < head_dim)
            Oh[gr0 * head_dim + c1] = __float2half(Or[nt][1] * li0);
        if (gr8 < seq_len && c0 < head_dim)
            Oh[gr8 * head_dim + c0] = __float2half(Or[nt][2] * li1);
        if (gr8 < seq_len && c1 < head_dim)
            Oh[gr8 * head_dim + c1] = __float2half(Or[nt][3] * li1);
    }
}

// =============================================================================
// v55 kernel (FP16 polynomial exp2 on CUDA cores)
// =============================================================================
__global__ void __launch_bounds__(FA_THREADS, 2)
    fa55_kernel(const __half *__restrict__ Q, const __half *__restrict__ K,
                const __half *__restrict__ V, __half *__restrict__ O,
                int seq_len, int head_dim, int causal, float scale)
{
    int nqt = (seq_len + FA_BR - 1) / FA_BR;
    int bh = blockIdx.x / nqt, qt = blockIdx.x % nqt, qs = qt * FA_BR;
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    int gid = lane >> 2, tid = lane & 3;
    extern __shared__ char raw[];
    __half *buf0 = (__half *)raw;
    __half *buf1 = (__half *)(raw + FA_BC * FA_STRIDE * sizeof(__half));
    int hs = seq_len * head_dim;
    const __half *Qh = Q + bh * hs, *Kh = K + bh * hs, *Vh = V + bh * hs;
    __half *Oh = O + bh * hs;

    load_tile(buf0, Qh, qs, FA_BR, seq_len, head_dim);
    cpa_commit();
    cpa_wait<0>();
    __syncthreads();
    int mrb = wid * 16;
    uint32_t Qr[8][4];
#pragma unroll
    for (int ks = 0; ks < 8; ks++)
        ld_a_sw(Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3], buf0, FA_STRIDE, mrb, ks * 16, lane);
    __syncthreads();

    float Or[16][4];
#pragma unroll
    for (int t = 0; t < 16; t++)
        Or[t][0] = Or[t][1] = Or[t][2] = Or[t][3] = 0;
    float rmax[2] = {-1e30f, -1e30f}, rsexp[2] = {0.0f, 0.0f};
    int nkv = (seq_len + FA_BC - 1) / FA_BC;
    load_tile(buf0, Kh, 0, FA_BC, seq_len, head_dim);
    cpa_commit();

    const __half2 h2_log2e = __float2half2_rn(1.4426950408889634f);

    for (int kv = 0; kv < nkv; kv++)
    {
        int kvs = kv * FA_BC;
        if (causal && kvs > qs + FA_BR - 1)
            break;
        __half *cur = (kv & 1) ? buf1 : buf0, *nxt = (kv & 1) ? buf0 : buf1;
        cpa_wait<0>();
        __syncthreads();

        float Sr[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
            Sr[nt][0] = Sr[nt][1] = Sr[nt][2] = Sr[nt][3] = 0;
#pragma unroll
        for (int ks = 0; ks < 8; ks++)
        {
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
            {
                uint32_t b0, b1;
                ld_b_sw(b0, b1, cur, FA_STRIDE, nt * 8, ks * 16, lane);
                mma16816(Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3],
                         Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3],
                         b0, b1, Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3]);
            }
        }
        __syncthreads();
        load_tile(cur, Vh, kvs, FA_BC, seq_len, head_dim);
        cpa_commit();
        int nkvs = (kv + 1) * FA_BC;
        bool has_nxt = (kv + 1 < nkv) && (!causal || nkvs <= qs + FA_BR - 1);
        if (has_nxt)
            load_tile(nxt, Kh, nkvs, FA_BC, seq_len, head_dim);
        cpa_commit();

#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            Sr[nt][0] *= scale;
            Sr[nt][1] *= scale;
            Sr[nt][2] *= scale;
            Sr[nt][3] *= scale;
            if (causal)
            {
                int gq0 = qs + mrb + gid, gq8 = gq0 + 8;
                int gk0 = kvs + nt * 8 + tid * 2, gk1 = gk0 + 1;
                if (gk0 > gq0)
                    Sr[nt][0] = -1e30f;
                if (gk1 > gq0)
                    Sr[nt][1] = -1e30f;
                if (gk0 > gq8)
                    Sr[nt][2] = -1e30f;
                if (gk1 > gq8)
                    Sr[nt][3] = -1e30f;
                if (gq0 >= seq_len)
                {
                    Sr[nt][0] = -1e30f;
                    Sr[nt][1] = -1e30f;
                }
                if (gq8 >= seq_len)
                {
                    Sr[nt][2] = -1e30f;
                    Sr[nt][3] = -1e30f;
                }
                if (gk0 >= seq_len)
                {
                    Sr[nt][0] = -1e30f;
                    Sr[nt][2] = -1e30f;
                }
                if (gk1 >= seq_len)
                {
                    Sr[nt][1] = -1e30f;
                    Sr[nt][3] = -1e30f;
                }
            }
        }

        float nm[2] = {-1e30f, -1e30f};
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            nm[0] = fmaxf(nm[0], fmaxf(Sr[nt][0], Sr[nt][1]));
            nm[1] = fmaxf(nm[1], fmaxf(Sr[nt][2], Sr[nt][3]));
        }
        nm[0] = fmaxf(nm[0], __shfl_xor_sync(0xffffffff, nm[0], 1));
        nm[0] = fmaxf(nm[0], __shfl_xor_sync(0xffffffff, nm[0], 2));
        nm[1] = fmaxf(nm[1], __shfl_xor_sync(0xffffffff, nm[1], 1));
        nm[1] = fmaxf(nm[1], __shfl_xor_sync(0xffffffff, nm[1], 2));
        nm[0] = fmaxf(nm[0], rmax[0]);
        nm[1] = fmaxf(nm[1], rmax[1]);

        // Rescale O: use polynomial exp2 for rescale factors
        __half2 h2d = __halves2half2(__float2half(rmax[0] - nm[0]), __float2half(rmax[1] - nm[1]));
        h2d = __hmul2(h2d, h2_log2e);
        __half2 h2r = poly_exp2_h2(h2d);
        float rsc0 = __half2float(__low2half(h2r)), rsc1 = __half2float(__high2half(h2r));

#pragma unroll
        for (int t = 0; t < 16; t++)
        {
            Or[t][0] *= rsc0;
            Or[t][1] *= rsc0;
            Or[t][2] *= rsc1;
            Or[t][3] *= rsc1;
        }
        rmax[0] = nm[0];
        rmax[1] = nm[1];

        // ---- FP16 polynomial exp2 for softmax ----
        float ns[2] = {0.0f, 0.0f};
        uint32_t Pr[4][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            __half2 h01 = __halves2half2(__float2half(Sr[nt][0] - rmax[0]), __float2half(Sr[nt][1] - rmax[0]));
            h01 = __hmul2(h01, h2_log2e);
            __half2 r01 = poly_exp2_h2(h01);

            __half2 h23 = __halves2half2(__float2half(Sr[nt][2] - rmax[1]), __float2half(Sr[nt][3] - rmax[1]));
            h23 = __hmul2(h23, h2_log2e);
            __half2 r23 = poly_exp2_h2(h23);

            ns[0] += __half2float(__low2half(r01)) + __half2float(__high2half(r01));
            ns[1] += __half2float(__low2half(r23)) + __half2float(__high2half(r23));
            int pi = nt / 2, half = nt % 2;
            __half2 *p = (__half2 *)Pr[pi];
            p[half * 2] = r01;
            p[half * 2 + 1] = r23;
        }
        ns[0] += __shfl_xor_sync(0xffffffff, ns[0], 1);
        ns[0] += __shfl_xor_sync(0xffffffff, ns[0], 2);
        ns[1] += __shfl_xor_sync(0xffffffff, ns[1], 1);
        ns[1] += __shfl_xor_sync(0xffffffff, ns[1], 2);
        rsexp[0] = rsexp[0] * rsc0 + ns[0];
        rsexp[1] = rsexp[1] * rsc1 + ns[1];

        cpa_wait<1>();
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < 4; ks++)
        {
#pragma unroll
            for (int nt = 0; nt < 16; nt++)
            {
                uint32_t b0, b1;
                ld_b_vt(b0, b1, cur, FA_STRIDE, nt * 8, ks * 16, lane);
                mma16816(Or[nt][0], Or[nt][1], Or[nt][2], Or[nt][3],
                         Pr[ks][0], Pr[ks][1], Pr[ks][2], Pr[ks][3],
                         b0, b1, Or[nt][0], Or[nt][1], Or[nt][2], Or[nt][3]);
            }
        }
    }
    float li0 = (rsexp[0] > 0) ? 1.0f / rsexp[0] : 0.0f, li1 = (rsexp[1] > 0) ? 1.0f / rsexp[1] : 0.0f;
    int gr0 = qs + mrb + gid, gr8 = gr0 + 8;
#pragma unroll
    for (int nt = 0; nt < 16; nt++)
    {
        int c0 = nt * 8 + tid * 2, c1 = c0 + 1;
        if (gr0 < seq_len && c0 < head_dim)
            Oh[gr0 * head_dim + c0] = __float2half(Or[nt][0] * li0);
        if (gr0 < seq_len && c1 < head_dim)
            Oh[gr0 * head_dim + c1] = __float2half(Or[nt][1] * li0);
        if (gr8 < seq_len && c0 < head_dim)
            Oh[gr8 * head_dim + c0] = __float2half(Or[nt][2] * li1);
        if (gr8 < seq_len && c1 < head_dim)
            Oh[gr8 * head_dim + c1] = __float2half(Or[nt][3] * li1);
    }
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
static int g54 = 0, g55 = 0;
void launch_v54(const __half *Q, const __half *K, const __half *V, __half *O, int th, int sl, int hd, int ca)
{
    int sm = 2 * FA_BC * FA_STRIDE * (int)sizeof(__half);
    if (sm > g54)
    {
        CK(cudaFuncSetAttribute(fa54_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
        g54 = sm;
    }
    int nqt = (sl + FA_BR - 1) / FA_BR;
    fa54_kernel<<<th * nqt, FA_THREADS, sm>>>(Q, K, V, O, sl, hd, ca, 1.0f / sqrtf((float)hd));
}
void launch_v55(const __half *Q, const __half *K, const __half *V, __half *O, int th, int sl, int hd, int ca)
{
    int sm = 2 * FA_BC * FA_STRIDE * (int)sizeof(__half);
    if (sm > g55)
    {
        CK(cudaFuncSetAttribute(fa55_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
        g55 = sm;
    }
    int nqt = (sl + FA_BR - 1) / FA_BR;
    fa55_kernel<<<th * nqt, FA_THREADS, sm>>>(Q, K, V, O, sl, hd, ca, 1.0f / sqrtf((float)hd));
}

void test_correctness()
{
    printf("--- Correctness: v54 (MUFU.EX2.F16) vs v55 (polynomial) vs CPU ---\n");
    int configs[][3] = {{1, 32, 128}, {1, 64, 128}, {2, 128, 128}, {1, 256, 128}, {1, 512, 128}, {32, 1024, 128}};
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
        float sc = 1.0f / sqrtf((float)dim);
        for (int h = 0; h < heads; h++)
        {
            int off = h * seq * dim;
            for (int q = 0; q < seq; q++)
            {
                float rm = -1e30f;
                float *scores = (float *)calloc(seq, 4);
                for (int k = 0; k <= q; k++)
                {
                    float d = 0;
                    for (int d2 = 0; d2 < dim; d2++)
                        d += h2f(hQ[off + q * dim + d2]) * h2f(hK[off + k * dim + d2]);
                    scores[k] = d * sc;
                    rm = fmaxf(rm, scores[k]);
                }
                float sm = 0;
                for (int k = 0; k <= q; k++)
                {
                    scores[k] = expf(scores[k] - rm);
                    sm += scores[k];
                }
                for (int d2 = 0; d2 < dim; d2++)
                {
                    float a = 0;
                    for (int k = 0; k <= q; k++)
                        a += (scores[k] / sm) * h2f(hV[off + k * dim + d2]);
                    ref[off + q * dim + d2] = a;
                }
                free(scores);
            }
        }
        void *dQ, *dK, *dV;
        __half *dO54, *dO55;
        CK(cudaMalloc(&dQ, sz));
        CK(cudaMalloc(&dK, sz));
        CK(cudaMalloc(&dV, sz));
        CK(cudaMalloc(&dO54, sz));
        CK(cudaMalloc(&dO55, sz));
        CK(cudaMemcpy(dQ, hQ, sz, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dK, hK, sz, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dV, hV, sz, cudaMemcpyHostToDevice));
        CK(cudaMemset(dO54, 0, sz));
        CK(cudaMemset(dO55, 0, sz));
        launch_v54((const __half *)dQ, (const __half *)dK, (const __half *)dV, dO54, heads, seq, dim, 1);
        launch_v55((const __half *)dQ, (const __half *)dK, (const __half *)dV, dO55, heads, seq, dim, 1);
        CK(cudaDeviceSynchronize());
        uint16_t *hO54 = (uint16_t *)malloc(sz), *hO55 = (uint16_t *)malloc(sz);
        CK(cudaMemcpy(hO54, dO54, sz, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hO55, dO55, sz, cudaMemcpyDeviceToHost));
        float mx54 = 0, mx55 = 0;
        int e54 = 0, e55 = 0;
        for (int i = 0; i < n; i++)
        {
            float a54 = fabsf(h2f(hO54[i]) - ref[i]), a55 = fabsf(h2f(hO55[i]) - ref[i]);
            if (a54 > mx54)
                mx54 = a54;
            if (a55 > mx55)
                mx55 = a55;
            float t54 = fmaxf(0.003f, fabsf(ref[i]) * 0.08f);
            float t55 = fmaxf(0.005f, fabsf(ref[i]) * 0.10f); // relaxed for polynomial
            if (a54 > t54)
                e54++;
            if (a55 > t55)
                e55++;
        }
        printf("  h=%d s=%4d  v54:max=%.4f err=%d %s  v55:max=%.4f err=%d %s\n",
               heads, seq, mx54, e54, e54 == 0 ? "PASS" : "FAIL", mx55, e55, e55 == 0 ? "PASS" : "FAIL");
        cudaFree(dQ);
        cudaFree(dK);
        cudaFree(dV);
        cudaFree(dO54);
        cudaFree(dO55);
        free(hQ);
        free(hK);
        free(hV);
        free(ref);
        free(hO54);
        free(hO55);
    }
}

void bench()
{
    printf("\n--- Performance: v54 (MUFU.EX2.F16) vs v55 (polynomial HFMA2) ---\n");
    printf("%-12s %10s %10s %8s %7s\n", "Config", "v54 (T)", "v55 (T)", "delta", "peak%");
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
            launch_v54((const __half *)dQ, (const __half *)dK, (const __half *)dV, dO, th, c.s, c.d, 1);
        CK(cudaDeviceSynchronize());
        t.start();
        for (int i = 0; i < it; i++)
            launch_v54((const __half *)dQ, (const __half *)dK, (const __half *)dV, dO, th, c.s, c.d, 1);
        float ms54 = t.stop();
        double tf54 = flops / (ms54 / it / 1000.0) / 1e12;
        CK(cudaMemset(dO, 0, (size_t)n * 2));
        for (int i = 0; i < 3; i++)
            launch_v55((const __half *)dQ, (const __half *)dK, (const __half *)dV, dO, th, c.s, c.d, 1);
        CK(cudaDeviceSynchronize());
        t.start();
        for (int i = 0; i < it; i++)
            launch_v55((const __half *)dQ, (const __half *)dK, (const __half *)dV, dO, th, c.s, c.d, 1);
        float ms55 = t.stop();
        double tf55 = flops / (ms55 / it / 1000.0) / 1e12;
        printf("%-12s %8.2f T %8.2f T %+7.2f %6.1f%%", c.l, tf54, tf55, tf55 - tf54, tf55 / 165.2 * 100);
        if (tf55 > tf54 * 1.02)
            printf("  ** WIN **");
        else if (tf55 < tf54 * 0.98)
            printf("  !! LOSS !!");
        printf("\n");
        cudaFree(dQ);
        cudaFree(dK);
        cudaFree(dV);
        cudaFree(dO);
    }
    printf("\nPeak = 165.2 T | v54: 153T | FA2: 159T @ 7B-8K\n");
}

int main()
{
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("=== FlashAttention v55 — Polynomial exp2 (HFMA2) ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n\n", p.name, p.multiProcessorCount, p.clockRate / 1000);
    test_correctness();
    bench();
    return 0;
}
