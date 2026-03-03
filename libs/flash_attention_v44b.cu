// =============================================================================
// FlashAttention v44b — Minimal exp2f change from v20
// =============================================================================
// ONLY change: __expf(x) -> exp2f(x * 1.4426950408889634f)
// Everything else 100% identical to v20.
// If this fails, exp2f precision is different from __expf.
// If this passes, the v44 refactoring introduced a bug.
// =============================================================================

#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define FA44B_BR 64
#define FA44B_BC 64
#define FA44B_THREADS 128
#define FA44B_STRIDE 128

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
__device__ __forceinline__ void ldm4(uint32_t &r0, uint32_t &r1,
                                     uint32_t &r2, uint32_t &r3, const void *p)
{
    uint32_t a = __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
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
    for (int c = threadIdx.x; c < total; c += FA44B_THREADS)
    {
        int row = c / CPR, lch = c % CPR, gr = start + row;
        int pch = lch ^ (row & 7);
        cpa16(&dst[row * FA44B_STRIDE + pch * 8],
              &src[gr * head_dim + lch * 8], (gr < seq_len) ? 16 : 0);
    }
}

// Replace __expf with exp2f
__device__ __forceinline__ float fast_exp(float x)
{
    return exp2f(x * 1.4426950408889634f);
}

__global__ void __launch_bounds__(FA44B_THREADS, 2)
    flash_attention_v44b_kernel(
        const __half *__restrict__ Q, const __half *__restrict__ K,
        const __half *__restrict__ V, __half *__restrict__ O,
        int seq_len, int head_dim, int causal, float scale)
{
    int nqt = (seq_len + FA44B_BR - 1) / FA44B_BR;
    int bh = blockIdx.x / nqt, qt = blockIdx.x % nqt, qs = qt * FA44B_BR;
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    int gid = lane >> 2, tid = lane & 3;

    extern __shared__ char raw[];
    __half *buf0 = (__half *)raw;
    __half *buf1 = (__half *)(raw + FA44B_BC * FA44B_STRIDE * sizeof(__half));

    int hs = seq_len * head_dim;
    const __half *Qh = Q + bh * hs, *Kh = K + bh * hs, *Vh = V + bh * hs;
    __half *Oh = O + bh * hs;

    load_tile(buf0, Qh, qs, FA44B_BR, seq_len, head_dim);
    cpa_commit();
    cpa_wait<0>();
    __syncthreads();

    int mrb = wid * 16;
    uint32_t Qr[8][4];
#pragma unroll
    for (int ks = 0; ks < 8; ks++)
        ld_a_sw(Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3],
                buf0, FA44B_STRIDE, mrb, ks * 16, lane);
    __syncthreads();

    float Or[16][4];
#pragma unroll
    for (int t = 0; t < 16; t++)
        Or[t][0] = Or[t][1] = Or[t][2] = Or[t][3] = 0;

    float rmax[2] = {-1e30f, -1e30f}, rsexp[2] = {0, 0};
    int nkv = (seq_len + FA44B_BC - 1) / FA44B_BC;

    load_tile(buf0, Kh, 0, FA44B_BC, seq_len, head_dim);
    cpa_commit();

    for (int kv = 0; kv < nkv; kv++)
    {
        int kvs = kv * FA44B_BC;
        if (causal && kvs > qs + FA44B_BR - 1)
            break;

        __half *cur = (kv & 1) ? buf1 : buf0;
        __half *nxt = (kv & 1) ? buf0 : buf1;

        cpa_wait<0>();
        __syncthreads();

        // ==== QK^T ==== (identical to v20)
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
                ld_b_sw(b0, b1, cur, FA44B_STRIDE, nt * 8, ks * 16, lane);
                mma16816(Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3],
                         Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3],
                         b0, b1,
                         Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3]);
            }
        }

        __syncthreads();
        load_tile(cur, Vh, kvs, FA44B_BC, seq_len, head_dim);
        cpa_commit();
        int nkvs = (kv + 1) * FA44B_BC;
        bool has_nxt = (kv + 1 < nkv) && (!causal || nkvs <= qs + FA44B_BR - 1);
        if (has_nxt)
            load_tile(nxt, Kh, nkvs, FA44B_BC, seq_len, head_dim);
        cpa_commit();

        // Scale + causal mask (identical to v20)
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

        // Online softmax — ONLY CHANGE: fast_exp instead of __expf
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

        float rsc0 = fast_exp(rmax[0] - nm[0]), rsc1 = fast_exp(rmax[1] - nm[1]);
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

        float ns[2] = {0, 0};
        uint32_t Pr[4][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            Sr[nt][0] = fast_exp(Sr[nt][0] - rmax[0]);
            Sr[nt][1] = fast_exp(Sr[nt][1] - rmax[0]);
            Sr[nt][2] = fast_exp(Sr[nt][2] - rmax[1]);
            Sr[nt][3] = fast_exp(Sr[nt][3] - rmax[1]);
            ns[0] += Sr[nt][0] + Sr[nt][1];
            ns[1] += Sr[nt][2] + Sr[nt][3];
            int pi = nt / 2, half = nt % 2;
            __half2 *p = (__half2 *)Pr[pi];
            p[half * 2] = __halves2half2(__float2half(Sr[nt][0]), __float2half(Sr[nt][1]));
            p[half * 2 + 1] = __halves2half2(__float2half(Sr[nt][2]), __float2half(Sr[nt][3]));
        }
        ns[0] += __shfl_xor_sync(0xffffffff, ns[0], 1);
        ns[0] += __shfl_xor_sync(0xffffffff, ns[0], 2);
        ns[1] += __shfl_xor_sync(0xffffffff, ns[1], 1);
        ns[1] += __shfl_xor_sync(0xffffffff, ns[1], 2);
        rsexp[0] = rsexp[0] * rsc0 + ns[0];
        rsexp[1] = rsexp[1] * rsc1 + ns[1];

        // ==== PV ==== (identical to v20)
        cpa_wait<1>();
        __syncthreads();

#pragma unroll
        for (int ks = 0; ks < 4; ks++)
        {
#pragma unroll
            for (int nt = 0; nt < 16; nt++)
            {
                uint32_t b0, b1;
                ld_b_vt(b0, b1, cur, FA44B_STRIDE, nt * 8, ks * 16, lane);
                mma16816(Or[nt][0], Or[nt][1], Or[nt][2], Or[nt][3],
                         Pr[ks][0], Pr[ks][1], Pr[ks][2], Pr[ks][3],
                         b0, b1,
                         Or[nt][0], Or[nt][1], Or[nt][2], Or[nt][3]);
            }
        }
    }

    float li0 = (rsexp[0] > 0) ? 1.f / rsexp[0] : 0, li1 = (rsexp[1] > 0) ? 1.f / rsexp[1] : 0;
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

static int g_FA44B_smem = 0;
extern "C"
{
    int flash_attention_v44b_forward(
        const void *Q, const void *K, const void *V, void *O,
        int total_heads, int seq_len, int head_dim, int causal, void *stream)
    {
        if (head_dim != 128)
            return -1;
        int smem = 2 * FA44B_BC * FA44B_STRIDE * (int)sizeof(__half);
        if (smem > g_FA44B_smem)
        {
            cudaError_t e = cudaFuncSetAttribute(flash_attention_v44b_kernel,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            if (e != cudaSuccess)
                return (int)e;
            g_FA44B_smem = smem;
        }
        float sc = 1.0f / sqrtf((float)head_dim);
        int nqt = (seq_len + FA44B_BR - 1) / FA44B_BR;
        flash_attention_v44b_kernel<<<total_heads * nqt, FA44B_THREADS, smem, (cudaStream_t)stream>>>(
            (const __half *)Q, (const __half *)K, (const __half *)V, (__half *)O,
            seq_len, head_dim, causal, sc);
        return (int)cudaGetLastError();
    }
}
