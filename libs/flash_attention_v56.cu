// =============================================================================
// FlashAttention v56 — v54 (FP16 ex2) + 3-buffer V overlap
// =============================================================================
//
// Base: v54 (MUFU.EX2.F16 softmax, 153T)
// Change: 3 SMEM buffers instead of 2
//   kbuf0 (16KB): K double-buf ping
//   kbuf1 (16KB): K double-buf pong
//   vbuf  (16KB): V dedicated buffer
//   Total: 48KB (vs 32KB in v54)
//   Occupancy: floor(100KB/48KB) = 2 blocks/SM — same as v54
//
// Pipeline:
//   v54:  wait K → QK^T → sync → load V to cur → softmax → wait V → PV
//   v56:  wait K → QK^T → load V to vbuf (no sync needed!) → softmax → wait V → PV
//
// Key insight: V loads to SEPARATE buffer, doesn't overwrite K.
// So V prefetch can start BEFORE syncthreads after QK^T.
// But actually we still need sync because all warps must finish reading K
// before we start loading K[next] which overwrites cur.
// The real win: V is always in vbuf, K double-buffers in kbuf0/kbuf1.
// No conflict between V load and K data.
//
// Build:
//   nvcc -O3 -arch=sm_89 -std=c++17 flash_attention_v56.cu -o fa_v56 -lcudart
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

// Helpers (same as v54)
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

__device__ __forceinline__ uint32_t hexp2x2(uint32_t in)
{
    uint32_t out;
    asm("ex2.approx.f16x2 %0, %1;" : "=r"(out) : "r"(in));
    return out;
}

// =============================================================================
// v54 kernel (2-buffer baseline)
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
// v56 kernel (3-buffer: kbuf0, kbuf1 for K; vbuf for V)
// =============================================================================
__global__ void __launch_bounds__(FA_THREADS, 2)
    fa56_kernel(const __half *__restrict__ Q, const __half *__restrict__ K,
                const __half *__restrict__ V, __half *__restrict__ O,
                int seq_len, int head_dim, int causal, float scale)
{
    int nqt = (seq_len + FA_BR - 1) / FA_BR;
    int bh = blockIdx.x / nqt, qt = blockIdx.x % nqt, qs = qt * FA_BR;
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    int gid = lane >> 2, tid = lane & 3;

    extern __shared__ char raw[];
    // 3 buffers: kbuf0, kbuf1 (K double-buf), vbuf (V dedicated)
    constexpr int BUF_SIZE = FA_BC * FA_STRIDE * sizeof(__half); // 16KB each
    __half *kbuf0 = (__half *)raw;
    __half *kbuf1 = (__half *)(raw + BUF_SIZE);
    __half *vbuf = (__half *)(raw + 2 * BUF_SIZE);

    int hs = seq_len * head_dim;
    const __half *Qh = Q + bh * hs, *Kh = K + bh * hs, *Vh = V + bh * hs;
    __half *Oh = O + bh * hs;

    // Load Q → kbuf0 (temporary, will be reused for K)
    load_tile(kbuf0, Qh, qs, FA_BR, seq_len, head_dim);
    cpa_commit();
    cpa_wait<0>();
    __syncthreads();
    int mrb = wid * 16;
    uint32_t Qr[8][4];
#pragma unroll
    for (int ks = 0; ks < 8; ks++)
        ld_a_sw(Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3], kbuf0, FA_STRIDE, mrb, ks * 16, lane);
    __syncthreads();

    float Or[16][4];
#pragma unroll
    for (int t = 0; t < 16; t++)
        Or[t][0] = Or[t][1] = Or[t][2] = Or[t][3] = 0;
    float rmax[2] = {-1e30f, -1e30f}, rsexp[2] = {0.0f, 0.0f};
    int nkv = (seq_len + FA_BC - 1) / FA_BC;

    // Prefetch K[0] → kbuf0
    load_tile(kbuf0, Kh, 0, FA_BC, seq_len, head_dim);
    cpa_commit();

    const __half2 h2_log2e = __float2half2_rn(1.4426950408889634f);

    for (int kv = 0; kv < nkv; kv++)
    {
        int kvs = kv * FA_BC;
        if (causal && kvs > qs + FA_BR - 1)
            break;

        __half *kcur = (kv & 1) ? kbuf1 : kbuf0;
        __half *knxt = (kv & 1) ? kbuf0 : kbuf1;

        // Wait for K[kv]
        cpa_wait<0>();
        __syncthreads();

        // ---- QK^T ----
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
                ld_b_sw(b0, b1, kcur, FA_STRIDE, nt * 8, ks * 16, lane);
                mma16816(Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3],
                         Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3],
                         b0, b1, Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3]);
            }
        }

        // K consumed. Start V[kv] → vbuf AND K[kv+1] → knxt
        // V goes to dedicated vbuf — no conflict with K!
        __syncthreads();
        load_tile(vbuf, Vh, kvs, FA_BC, seq_len, head_dim);
        cpa_commit(); // group for V

        int nkvs = (kv + 1) * FA_BC;
        bool has_nxt = (kv + 1 < nkv) && (!causal || nkvs <= qs + FA_BR - 1);
        if (has_nxt)
            load_tile(knxt, Kh, nkvs, FA_BC, seq_len, head_dim);
        cpa_commit(); // group for K[next]

        // ---- Scale + causal ----
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

        // ---- Softmax (FP16 ex2 — same as v54) ----
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

        // ---- P@V from vbuf (dedicated V buffer) ----
        cpa_wait<1>();
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < 4; ks++)
        {
#pragma unroll
            for (int nt = 0; nt < 16; nt++)
            {
                uint32_t b0, b1;
                ld_b_vt(b0, b1, vbuf, FA_STRIDE, nt * 8, ks * 16, lane);
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
static int g54 = 0, g56 = 0;
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
void launch_v56(const __half *Q, const __half *K, const __half *V, __half *O, int th, int sl, int hd, int ca)
{
    int sm = 3 * FA_BC * FA_STRIDE * (int)sizeof(__half); // 48KB!
    if (sm > g56)
    {
        CK(cudaFuncSetAttribute(fa56_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
        g56 = sm;
    }
    int nqt = (sl + FA_BR - 1) / FA_BR;
    fa56_kernel<<<th * nqt, FA_THREADS, sm>>>(Q, K, V, O, sl, hd, ca, 1.0f / sqrtf((float)hd));
}

void test_correctness()
{
    printf("--- Correctness: v54 (2-buf) vs v56 (3-buf) vs CPU ---\n");
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
        __half *dO54, *dO56;
        CK(cudaMalloc(&dQ, sz));
        CK(cudaMalloc(&dK, sz));
        CK(cudaMalloc(&dV, sz));
        CK(cudaMalloc(&dO54, sz));
        CK(cudaMalloc(&dO56, sz));
        CK(cudaMemcpy(dQ, hQ, sz, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dK, hK, sz, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dV, hV, sz, cudaMemcpyHostToDevice));
        CK(cudaMemset(dO54, 0, sz));
        CK(cudaMemset(dO56, 0, sz));
        launch_v54((const __half *)dQ, (const __half *)dK, (const __half *)dV, dO54, heads, seq, dim, 1);
        launch_v56((const __half *)dQ, (const __half *)dK, (const __half *)dV, dO56, heads, seq, dim, 1);
        CK(cudaDeviceSynchronize());
        uint16_t *hO54 = (uint16_t *)malloc(sz), *hO56 = (uint16_t *)malloc(sz);
        CK(cudaMemcpy(hO54, dO54, sz, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hO56, dO56, sz, cudaMemcpyDeviceToHost));
        float mx54 = 0, mx56 = 0;
        int e54 = 0, e56 = 0;
        for (int i = 0; i < n; i++)
        {
            float a54 = fabsf(h2f(hO54[i]) - ref[i]), a56 = fabsf(h2f(hO56[i]) - ref[i]);
            if (a54 > mx54)
                mx54 = a54;
            if (a56 > mx56)
                mx56 = a56;
            float thr = fmaxf(0.003f, fabsf(ref[i]) * 0.08f);
            if (a54 > thr)
                e54++;
            if (a56 > thr)
                e56++;
        }
        printf("  h=%d s=%4d  v54:max=%.4f err=%d %s  v56:max=%.4f err=%d %s\n",
               heads, seq, mx54, e54, e54 == 0 ? "PASS" : "FAIL", mx56, e56, e56 == 0 ? "PASS" : "FAIL");
        cudaFree(dQ);
        cudaFree(dK);
        cudaFree(dV);
        cudaFree(dO54);
        cudaFree(dO56);
        free(hQ);
        free(hK);
        free(hV);
        free(ref);
        free(hO54);
        free(hO56);
    }
}

void bench()
{
    printf("\n--- Performance: v54 (2-buf 32KB) vs v56 (3-buf 48KB) ---\n");
    printf("%-12s %10s %10s %8s %7s\n", "Config", "v54 (T)", "v56 (T)", "delta", "peak%");
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
            launch_v56((const __half *)dQ, (const __half *)dK, (const __half *)dV, dO, th, c.s, c.d, 1);
        CK(cudaDeviceSynchronize());
        t.start();
        for (int i = 0; i < it; i++)
            launch_v56((const __half *)dQ, (const __half *)dK, (const __half *)dV, dO, th, c.s, c.d, 1);
        float ms56 = t.stop();
        double tf56 = flops / (ms56 / it / 1000.0) / 1e12;
        printf("%-12s %8.2f T %8.2f T %+7.2f %6.1f%%", c.l, tf54, tf56, tf56 - tf54, tf56 / 165.2 * 100);
        if (tf56 > tf54 * 1.02)
            printf("  ** WIN **");
        else if (tf56 < tf54 * 0.98)
            printf("  !! LOSS !!");
        printf("\n");
        cudaFree(dQ);
        cudaFree(dK);
        cudaFree(dV);
        cudaFree(dO);
    }
    printf("\nPeak = 165.2 T | v54: 153T | FA2: 159T @ 7B-8K\n");
    printf("v56: 48KB SMEM (3×16KB) vs v54: 32KB (2×16KB)\n");
}

int main()
{
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("=== FlashAttention v56 — 3-buffer V overlap ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n\n", p.name, p.multiProcessorCount, p.clockRate / 1000);
    test_correctness();
    bench();
    return 0;
}
