// =============================================================================
// FlashAttention v10 — Br=128, Bc=64, 512 threads (16 warps), 8M×2N
// =============================================================================
// 16 warps: warp_m = warp_id / 2 (0-7), warp_n = warp_id % 2 (0-1)
// Each warp: 16 Q-rows, exactly like v7
//   QK^T: 16 rows × 32 K-cols = 1 m-tile × 4 n-tiles
//   PV:   16 rows × 64 D-cols = 1 m-tile × 8 n-tiles
//   Softmax: 8 rows per warp (srow = warp_m*16 + warp_n*8)
//
// Register budget per warp (same as v7):
//   o_acc[8][4]  = 32 regs
//   s_acc[4][4]  = 16 regs
//   m_reg[8]     =  8 regs
//   l_reg[8]     =  8 regs
//   fragments    = ~40 regs
//   Total        = ~104 regs (vs v7=126, v10-fail=233)
//
// SMEM (unchanged from Br=128):
//   Q_s:  128 × 128 × 2 = 32,768 B
//   KV_s:  64 × 128 × 2 = 16,384 B
//   S_s:  128 ×  72 × 2 = 18,432 B
//   m+l:  128 ×  2  × 4 =  1,024 B
//   Total: 68,608 B → 1 block/SM (100KB avail)
//
// cp.async: 512 threads → Q: 2048 chunks / 512 = 4 per thread
//                          KV: 1024 chunks / 512 = 2 per thread (fast!)
//
// Latency hiding: 16 warps/block (same as v7's 2 blocks × 8 warps)
// Win: 2× arithmetic intensity — 128 Q-rows per KV tile load
// =============================================================================

#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define FA10_BR 128
#define FA10_BC 64
#define FA10_THREADS 512

#define FA10_Q_STRIDE 128
#define FA10_KV_STRIDE 128
#define FA10_S_STRIDE 72

// --- Swizzle ---
__device__ __forceinline__ int swz(int row, int col)
{
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

// --- cp.async ---
__device__ __forceinline__ void cpa16(void *s, const void *g, int n)
{
    uint32_t sa = __cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" ::"r"(sa), "l"(g), "r"(n));
}
__device__ __forceinline__ void cpa_commit()
{
    asm volatile("cp.async.commit_group;");
}
template <int N>
__device__ __forceinline__ void cpa_wait()
{
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

// --- ldmatrix ---
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

// --- Warp reduce ---
__device__ __forceinline__ float wrmax(float v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
__device__ __forceinline__ float wrsum(float v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// --- MMA ---
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
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// --- Fragment loaders ---
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

__device__ __forceinline__ void ld_a_pl(
    uint32_t &a0, uint32_t &a1, uint32_t &a2, uint32_t &a3,
    const __half *sm, int stride, int rb, int kb, int lane)
{
    int sub = lane / 8, sr = lane % 8;
    int r = rb + (sub & 1) * 8 + sr, c = kb + (sub >> 1) * 8;
    ldm4(a0, a1, a2, a3, &sm[r * stride + c]);
}

__device__ __forceinline__ void st_d(__half *sm, int stride, int rb, int cb,
                                     float d0, float d1, float d2, float d3, int lane)
{
    int g = lane >> 2, t = lane & 3;
    int r0 = rb + g, r8 = r0 + 8, c0 = cb + t * 2, c1 = c0 + 1;
    sm[r0 * stride + c0] = __float2half(d0);
    sm[r0 * stride + c1] = __float2half(d1);
    sm[r8 * stride + c0] = __float2half(d2);
    sm[r8 * stride + c1] = __float2half(d3);
}

// --- Async tile loads ---
__device__ __forceinline__ void load_q_sw(
    __half *Q_s, const __half *Qh, int qs, int sl, int hd)
{
    // 128 rows × 16 chunks = 2048 total, 512 threads → 4 per thread
    constexpr int CPR = 16, TOT = FA10_BR * CPR;
#pragma unroll 4
    for (int c = threadIdx.x; c < TOT; c += FA10_THREADS)
    {
        int row = c / CPR, lch = c % CPR, gr = qs + row;
        int pch = lch ^ (row & 7);
        cpa16(&Q_s[row * FA10_Q_STRIDE + pch * 8], &Qh[gr * hd + lch * 8], (gr < sl) ? 16 : 0);
    }
}

__device__ __forceinline__ void load_kv_sw(
    __half *KV_s, const __half *src, int ts, int sl, int hd)
{
    // 64 rows × 16 chunks = 1024 total, 512 threads → 2 per thread
    constexpr int CPR = 16, TOT = FA10_BC * CPR;
#pragma unroll 2
    for (int c = threadIdx.x; c < TOT; c += FA10_THREADS)
    {
        int row = c / CPR, lch = c % CPR, gr = ts + row;
        int pch = lch ^ (row & 7);
        cpa16(&KV_s[row * FA10_KV_STRIDE + pch * 8], &src[gr * hd + lch * 8], (gr < sl) ? 16 : 0);
    }
}

// =============================================================================
// Kernel
// =============================================================================

__global__ void __launch_bounds__(FA10_THREADS, 1)
    flash_attention_v10_kernel(
        const __half *__restrict__ Q, const __half *__restrict__ K,
        const __half *__restrict__ V, __half *__restrict__ O,
        int seq_len, int head_dim, int causal, float scale)
{
    int nqt = (seq_len + FA10_BR - 1) / FA10_BR;
    int bh = blockIdx.x / nqt;
    int qt = blockIdx.x % nqt;
    int qs = qt * FA10_BR;

    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    int wm = wid / 2; // 0-7: which 16-row chunk
    int wn = wid % 2; // 0-1: which N-half
    int gid = lane >> 2, tid = lane & 3;

    extern __shared__ char raw[];
    __half *Q_s = (__half *)raw;
    __half *KV_s = (__half *)(raw + FA10_BR * FA10_Q_STRIDE * 2);
    __half *S_s = (__half *)(raw + FA10_BR * FA10_Q_STRIDE * 2 + FA10_BC * FA10_KV_STRIDE * 2);
    float *m_sm = (float *)(raw + FA10_BR * FA10_Q_STRIDE * 2 + FA10_BC * FA10_KV_STRIDE * 2 + FA10_BR * FA10_S_STRIDE * 2);
    float *l_sm = m_sm + FA10_BR;

    int hs = seq_len * head_dim;
    const __half *Qh = Q + bh * hs, *Kh = K + bh * hs, *Vh = V + bh * hs;
    __half *Oh = O + bh * hs;

    // Load Q (128 rows)
    load_q_sw(Q_s, Qh, qs, seq_len, head_dim);
    cpa_commit();

    // Init m/l
    for (int i = threadIdx.x; i < FA10_BR; i += FA10_THREADS)
    {
        m_sm[i] = -1e30f;
        l_sm[i] = 0.0f;
    }

    // O accumulators: 1 m-tile × 8 n-tiles = 32 regs (same as v7)
    float o_acc[8][4];
#pragma unroll
    for (int t = 0; t < 8; t++)
        o_acc[t][0] = o_acc[t][1] = o_acc[t][2] = o_acc[t][3] = 0;

    // Warp's 16-row base
    int mrb = wm * 16;
    int my_r0 = mrb + gid, my_r8 = my_r0 + 8;

    // Softmax: 8 rows per warp
    int srow_base = wm * 16 + wn * 8;
    float m_reg[8], l_reg[8];
#pragma unroll
    for (int r = 0; r < 8; r++)
    {
        m_reg[r] = -1e30f;
        l_reg[r] = 0.0f;
    }

    int nkv = (seq_len + FA10_BC - 1) / FA10_BC;
    int kst = head_dim / 16;

    // Prologue: K[0]
    load_kv_sw(KV_s, Kh, 0, seq_len, head_dim);
    cpa_commit();

    for (int kv = 0; kv < nkv; kv++)
    {
        int kvs = kv * FA10_BC;
        if (causal && kvs > qs + FA10_BR - 1)
            break;

        cpa_wait<0>();
        __syncthreads();

        // ---- S = Q @ K^T ----
        // 16 rows × 32 K-cols = 4 n-tiles
        float s_acc[4][4];
#pragma unroll
        for (int t = 0; t < 4; t++)
            s_acc[t][0] = s_acc[t][1] = s_acc[t][2] = s_acc[t][3] = 0;

        for (int ks = 0; ks < kst; ks++)
        {
            uint32_t a0, a1, a2, a3;
            ld_a_sw(a0, a1, a2, a3, Q_s, FA10_Q_STRIDE, mrb, ks * 16, lane);

#pragma unroll
            for (int nt = 0; nt < 4; nt++)
            {
                uint32_t b0, b1;
                ld_b_sw(b0, b1, KV_s, FA10_KV_STRIDE, wn * 32 + nt * 8, ks * 16, lane);
                mma16816(s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3],
                         a0, a1, a2, a3, b0, b1,
                         s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3]);
            }
        }

// Scale + causal + store → S_s
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
        {
            int cb = wn * 32 + nt * 8;
            s_acc[nt][0] *= scale;
            s_acc[nt][1] *= scale;
            s_acc[nt][2] *= scale;
            s_acc[nt][3] *= scale;

            if (causal)
            {
                int r0 = mrb + gid, r8 = r0 + 8;
                int c0 = cb + tid * 2, c1 = c0 + 1;
                int gq0 = qs + r0, gq8 = qs + r8, gk0 = kvs + c0, gk1 = kvs + c1;
                if (gk0 > gq0)
                    s_acc[nt][0] = -1e30f;
                if (gk1 > gq0)
                    s_acc[nt][1] = -1e30f;
                if (gk0 > gq8)
                    s_acc[nt][2] = -1e30f;
                if (gk1 > gq8)
                    s_acc[nt][3] = -1e30f;
                if (gq0 >= seq_len)
                {
                    s_acc[nt][0] = -1e30f;
                    s_acc[nt][1] = -1e30f;
                }
                if (gq8 >= seq_len)
                {
                    s_acc[nt][2] = -1e30f;
                    s_acc[nt][3] = -1e30f;
                }
                if (gk0 >= seq_len)
                {
                    s_acc[nt][0] = -1e30f;
                    s_acc[nt][2] = -1e30f;
                }
                if (gk1 >= seq_len)
                {
                    s_acc[nt][1] = -1e30f;
                    s_acc[nt][3] = -1e30f;
                }
            }

            st_d(S_s, FA10_S_STRIDE, mrb, cb,
                 s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3], lane);
        }

        __syncthreads();

        // V load overlapped with softmax
        load_kv_sw(KV_s, Vh, kvs, seq_len, head_dim);
        cpa_commit();

        // m_old for O rescale
        float m_old_r0 = m_sm[my_r0], m_old_r8 = m_sm[my_r8];

// ---- Softmax: 8 rows per warp (identical to v7) ----
#pragma unroll
        for (int rr = 0; rr < 8; rr += 2)
        {
            int sa = srow_base + rr, sb = sa + 1;

            float va0 = __half2float(S_s[sa * FA10_S_STRIDE + lane * 2]);
            float vb0 = __half2float(S_s[sb * FA10_S_STRIDE + lane * 2]);
            float va1 = __half2float(S_s[sa * FA10_S_STRIDE + lane * 2 + 1]);
            float vb1 = __half2float(S_s[sb * FA10_S_STRIDE + lane * 2 + 1]);

            float rma = wrmax(fmaxf(va0, va1));
            float rmb = wrmax(fmaxf(vb0, vb1));

            float moa = m_reg[rr], mob = m_reg[rr + 1];
            float mna = fmaxf(moa, rma), mnb = fmaxf(mob, rmb);

            float ea0 = __expf(va0 - mna), ea1 = __expf(va1 - mna);
            float eb0 = __expf(vb0 - mnb), eb1 = __expf(vb1 - mnb);

            float rsa = wrsum(ea0 + ea1), rsb = wrsum(eb0 + eb1);
            float rca = __expf(moa - mna), rcb = __expf(mob - mnb);

            m_reg[rr] = mna;
            m_reg[rr + 1] = mnb;
            if (lane == 0)
            {
                l_reg[rr] = l_reg[rr] * rca + rsa;
                l_reg[rr + 1] = l_reg[rr + 1] * rcb + rsb;
            }

            S_s[sa * FA10_S_STRIDE + lane * 2] = __float2half(ea0);
            S_s[sb * FA10_S_STRIDE + lane * 2] = __float2half(eb0);
            S_s[sa * FA10_S_STRIDE + lane * 2 + 1] = __float2half(ea1);
            S_s[sb * FA10_S_STRIDE + lane * 2 + 1] = __float2half(eb1);
        }

// Flush m/l
#pragma unroll
        for (int r = 0; r < 8; r++)
            m_sm[srow_base + r] = m_reg[r];
        if (lane == 0)
        {
#pragma unroll
            for (int r = 0; r < 8; r++)
                l_sm[srow_base + r] = l_reg[r];
        }

        cpa_wait<0>();
        __syncthreads();

        // ---- Rescale O ----
        float mn_r0 = m_sm[my_r0], mn_r8 = m_sm[my_r8];
        float rc0 = __expf(m_old_r0 - mn_r0), rc8 = __expf(m_old_r8 - mn_r8);

#pragma unroll
        for (int t = 0; t < 8; t++)
        {
            o_acc[t][0] *= rc0;
            o_acc[t][1] *= rc0;
            o_acc[t][2] *= rc8;
            o_acc[t][3] *= rc8;
        }

        // ---- O += P @ V ----
        // 16 rows × 64 D-cols = 8 n-tiles, k_steps=4
        int o_n_base = wn * 64;

#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            int dc = o_n_base + nt * 8;
#pragma unroll
            for (int ks = 0; ks < 4; ks++)
            {
                uint32_t a0, a1, a2, a3;
                ld_a_pl(a0, a1, a2, a3, S_s, FA10_S_STRIDE, mrb, ks * 16, lane);

                uint32_t b0, b1;
                ld_b_vt(b0, b1, KV_s, FA10_KV_STRIDE, dc, ks * 16, lane);

                mma16816(o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3],
                         a0, a1, a2, a3, b0, b1,
                         o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3]);
            }
        }

        // Prefetch K[next]
        __syncthreads();
        int nxt = kv + 1, nxts = nxt * FA10_BC;
        if (nxt < nkv && (!causal || nxts <= qs + FA10_BR - 1))
        {
            load_kv_sw(KV_s, Kh, nxts, seq_len, head_dim);
            cpa_commit();
        }
    }

    // ---- Final: O / l → global ----
    {
        float li0 = (l_sm[my_r0] > 0) ? 1.0f / l_sm[my_r0] : 0;
        float li8 = (l_sm[my_r8] > 0) ? 1.0f / l_sm[my_r8] : 0;
        int gr0 = qs + my_r0, gr8 = qs + my_r8;
        int o_n_base = wn * 64;

#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            int c0 = o_n_base + nt * 8 + tid * 2, c1 = c0 + 1;
            if (gr0 < seq_len && c0 < head_dim)
                Oh[gr0 * head_dim + c0] = __float2half(o_acc[nt][0] * li0);
            if (gr0 < seq_len && c1 < head_dim)
                Oh[gr0 * head_dim + c1] = __float2half(o_acc[nt][1] * li0);
            if (gr8 < seq_len && c0 < head_dim)
                Oh[gr8 * head_dim + c0] = __float2half(o_acc[nt][2] * li8);
            if (gr8 < seq_len && c1 < head_dim)
                Oh[gr8 * head_dim + c1] = __float2half(o_acc[nt][3] * li8);
        }
    }
}

// =============================================================================
// C API
// =============================================================================
static int g_fa10_smem_max = 0;

extern "C"
{

    int flash_attention_v10_forward(
        const void *Q, const void *K, const void *V, void *O,
        int total_heads, int seq_len, int head_dim,
        int causal, void *stream)
    {
        if (head_dim % 16 != 0 || head_dim > 128)
            return -1;

        int smem = FA10_BR * FA10_Q_STRIDE * 2 + FA10_BC * FA10_KV_STRIDE * 2 + FA10_BR * FA10_S_STRIDE * 2 + FA10_BR * 2 * 4;

        if (smem > g_fa10_smem_max)
        {
            cudaError_t e = cudaFuncSetAttribute(flash_attention_v10_kernel,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            if (e != cudaSuccess)
                return (int)e;
            g_fa10_smem_max = smem;
        }

        float sc = 1.0f / sqrtf((float)head_dim);
        int nqt = (seq_len + FA10_BR - 1) / FA10_BR;

        flash_attention_v10_kernel<<<total_heads * nqt, FA10_THREADS, smem, (cudaStream_t)stream>>>(
            (const __half *)Q, (const __half *)K, (const __half *)V, (__half *)O,
            seq_len, head_dim, causal, sc);

        return (int)cudaGetLastError();
    }

} // extern "C"
