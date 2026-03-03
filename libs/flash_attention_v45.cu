// =============================================================================
// FlashAttention v45 — Adaptive rescaling over v54 (FP16 ex2 softmax)
// =============================================================================
//
// FA4 insight: in online softmax, rescaling O accumulator each KV iteration
// costs 64 fmul per thread. But for long sequences, the row-wise max
// stabilizes after first few KV blocks. If new_max == old_max (which happens
// >90% of iterations for long seqs), rsc = exp(0) = 1.0, and the multiply
// is a no-op. We can skip it with a warp vote.
//
// Implementation:
//   After computing nm (new max) and comparing with rmax (old max):
//   - If nm[i] == rmax[i] for ALL threads in warp → skip Or rescaling entirely
//   - Use __all_sync to check if all 4 threads sharing a row agree
//   - Also skip rsexp rescaling when rsc == 1.0
//
// Expected gain: ~1-2T on long sequences (4K+) where max stabilizes early
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

// Shared helpers (identical to v54)
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
    for (int c = threadIdx.x; c < total; c += FA_THREADS)
    {
        int row = c / CPR, lch = c % CPR, gr = start + row;
        int pch = lch ^ (row & 7);
        cpa16(&dst[row * FA_STRIDE + pch * 8],
              &src[gr * head_dim + lch * 8], (gr < seq_len) ? 16 : 0);
    }
}

// FP16 ex2 (from v54)
__device__ __forceinline__ uint32_t hexp2x2(uint32_t in)
{
    uint32_t out;
    asm("ex2.approx.f16x2 %0, %1;" : "=r"(out) : "r"(in));
    return out;
}

__global__ void __launch_bounds__(FA_THREADS, 2)
    fa45_kernel(const __half *__restrict__ Q, const __half *__restrict__ K,
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
        ld_a_sw(Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3],
                buf0, FA_STRIDE, mrb, ks * 16, lane);
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
        __half *cur = (kv & 1) ? buf1 : buf0;
        __half *nxt = (kv & 1) ? buf0 : buf1;
        cpa_wait<0>();
        __syncthreads();

        // QK^T
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

        // Scale + causal mask
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

        // Row max (FP32)
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

        // === ADAPTIVE RESCALING ===
        // Check if max actually changed. In m16n8k16 fragment layout,
        // threads 0-3 and 4-7 share row0 max (nm[0]), threads 0-3 and 4-7
        // share row8 max (nm[1]). After warp reduce, all threads in the
        // 4-thread group have the same nm value.
        //
        // If nm == rmax (max didn't change), rsc = exp(0) = 1.0
        // We can skip the expensive Or rescaling (64 fmul per thread).
        //
        // Use per-row check: each row's max is shared by 4 threads.
        // Check within the warp (32 threads = 4 groups of 8, but each group
        // has 2 sub-rows of 4 threads).

        int no_change_0 = (nm[0] == rmax[0]);
        int no_change_1 = (nm[1] == rmax[1]);

        // All 32 threads in warp vote: can we skip rescaling for row0?
        // Actually we need per-row vote. But since nm/rmax are identical
        // across the 4-thread butterfly group, all threads agree.
        // The whole warp has 4 warp fragments (gid 0..7), but nm[0] is
        // shared across all of them after butterfly reduce.
        // So we can just check locally — all threads have same nm[0], rmax[0].

        float rsc0, rsc1;

        if (no_change_0 && no_change_1)
        {
            // Both rows unchanged — skip ALL rescaling
            rsc0 = 1.0f;
            rsc1 = 1.0f;
            // No need to touch Or or rsexp
        }
        else
        {
            // At least one row changed — compute rescale factors via FP16 ex2
            __half h_diff0 = __float2half(rmax[0] - nm[0]);
            __half h_diff1 = __float2half(rmax[1] - nm[1]);
            __half2 h2_diff = __halves2half2(h_diff0, h_diff1);
            h2_diff = __hmul2(h2_diff, h2_log2e);
            uint32_t rsc_packed = hexp2x2(*reinterpret_cast<uint32_t *>(&h2_diff));
            __half2 h2_rsc = *reinterpret_cast<__half2 *>(&rsc_packed);
            rsc0 = __half2float(__low2half(h2_rsc));
            rsc1 = __half2float(__high2half(h2_rsc));

            // Rescale O accumulator
            if (!no_change_0 && !no_change_1)
            {
                // Both changed — full rescale
#pragma unroll
                for (int t = 0; t < 16; t++)
                {
                    Or[t][0] *= rsc0;
                    Or[t][1] *= rsc0;
                    Or[t][2] *= rsc1;
                    Or[t][3] *= rsc1;
                }
            }
            else if (!no_change_0)
            {
                // Only row0 changed
#pragma unroll
                for (int t = 0; t < 16; t++)
                {
                    Or[t][0] *= rsc0;
                    Or[t][1] *= rsc0;
                }
            }
            else
            {
                // Only row1 changed
#pragma unroll
                for (int t = 0; t < 16; t++)
                {
                    Or[t][2] *= rsc1;
                    Or[t][3] *= rsc1;
                }
            }
        }

        rmax[0] = nm[0];
        rmax[1] = nm[1];

        // Exp + sum + P pack — FP16 ex2 (from v54)
        float ns[2] = {0.0f, 0.0f};
        uint32_t Pr[4][4];

#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            float d0 = Sr[nt][0] - rmax[0];
            float d1 = Sr[nt][1] - rmax[0];
            float d2 = Sr[nt][2] - rmax[1];
            float d3 = Sr[nt][3] - rmax[1];

            __half2 h01 = __halves2half2(__float2half(d0), __float2half(d1));
            h01 = __hmul2(h01, h2_log2e);
            uint32_t e01 = hexp2x2(*reinterpret_cast<uint32_t *>(&h01));
            __half2 r01 = *reinterpret_cast<__half2 *>(&e01);

            __half2 h23 = __halves2half2(__float2half(d2), __float2half(d3));
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

        // PV
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

    float li0 = (rsexp[0] > 0) ? 1.0f / rsexp[0] : 0.0f;
    float li1 = (rsexp[1] > 0) ? 1.0f / rsexp[1] : 0.0f;
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
// C API
// =============================================================================
static int g_smem45 = 0;
extern "C"
{
    int flash_attention_v45_forward(
        const void *Q, const void *K, const void *V, void *O,
        int total_heads, int seq_len, int head_dim, int causal, void *stream)
    {
        if (head_dim != 128)
            return -1;
        int smem = 2 * FA_BC * FA_STRIDE * (int)sizeof(__half);
        if (smem > g_smem45)
        {
            cudaError_t e = cudaFuncSetAttribute(fa45_kernel,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            if (e != cudaSuccess)
                return (int)e;
            g_smem45 = smem;
        }
        float sc = 1.0f / sqrtf((float)head_dim);
        int nqt = (seq_len + FA_BR - 1) / FA_BR;
        fa45_kernel<<<total_heads * nqt, FA_THREADS, smem, (cudaStream_t)stream>>>(
            (const __half *)Q, (const __half *)K, (const __half *)V, (__half *)O,
            seq_len, head_dim, causal, sc);
        return (int)cudaGetLastError();
    }
}
