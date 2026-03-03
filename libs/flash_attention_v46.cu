// =============================================================================
// FlashAttention v46 — Variable Bc Tile + FP16 ex2 Softmax (SM89)
// =============================================================================
//
// Based on v54 (FP16 ex2 softmax, 153T production kernel).
// Key change: template Bc tile size for KV dimension.
//
// FA2 insight: optimal Bc depends on sequence length.
//   - Bc=32:  fewer K columns per tile → more KV iterations but less smem
//             Better for very short seq where occupancy matters
//   - Bc=64:  default (our v20/v54 baseline)
//   - Bc=128: fewer KV iterations, better amortization of softmax/rescale overhead
//             Needs 2x smem but halves iteration count for long sequences
//
// For s=8192, d=128:
//   Bc=64:  128 KV iterations, smem = 2*64*128*2 = 32KB, occupancy=2
//   Bc=128: 64 KV iterations,  smem = 2*128*128*2 = 64KB, occupancy=1
//
// Trade-off: Bc=128 halves loop overhead but may hurt occupancy.
// On SM89 with 100KB smem, Bc=128 fits with occupancy=1.
//
// Dispatch: seq<=512 → Bc=32, seq<=2048 → Bc=64, seq>2048 → Bc=128
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define FA46_BR 64
#define FA46_THREADS 128
#define FA46_STRIDE 128

// --- Shared helpers (identical to v54) ---
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

// Templated load_tile — Bc is template param but rows is runtime
template <int BC>
__device__ __forceinline__ void load_tile_t(
    __half *dst, const __half *src, int start, int rows,
    int seq_len, int head_dim)
{
    constexpr int CPR = 16;
    int total = rows * CPR;
#pragma unroll 4
    for (int c = threadIdx.x; c < total; c += FA46_THREADS)
    {
        int row = c / CPR, lch = c % CPR, gr = start + row;
        int pch = lch ^ (row & 7);
        cpa16(&dst[row * FA46_STRIDE + pch * 8],
              &src[gr * head_dim + lch * 8], (gr < seq_len) ? 16 : 0);
    }
}

// FP16 ex2
__device__ __forceinline__ uint32_t hexp2x2(uint32_t in)
{
    uint32_t out;
    asm("ex2.approx.f16x2 %0, %1;" : "=r"(out) : "r"(in));
    return out;
}

// =============================================================================
// Templated FA kernel — BC is compile-time constant
// =============================================================================
// NT_K = BC/8 = number of 8-row K tiles in QK^T
// NT_PV = BC/16 = number of 16-col V tiles in PV (since mma k-dim=16 for PV)
// Actually for PV: P is [BR x BC], V is [BC x d].
// P tiles: BC/8 groups of 8 rows for B-side load
// PV MMA: K-dim = BC, so ks iterations = BC/16

template <int BC>
__global__ void __launch_bounds__(FA46_THREADS, 2)
    fa46_kernel(const __half *__restrict__ Q, const __half *__restrict__ K,
                const __half *__restrict__ V, __half *__restrict__ O,
                int seq_len, int head_dim, int causal, float scale)
{
    constexpr int NTK = BC / 8;   // QK^T B-side tiles: 4 for Bc=32, 8 for Bc=64, 16 for Bc=128
    constexpr int NTPV = BC / 16; // PV K-dim iterations: 2 for Bc=32, 4 for Bc=64, 8 for Bc=128

    int nqt = (seq_len + FA46_BR - 1) / FA46_BR;
    int bh = blockIdx.x / nqt, qt = blockIdx.x % nqt, qs = qt * FA46_BR;
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    int gid = lane >> 2, tid = lane & 3;
    extern __shared__ char raw[];
    __half *buf0 = (__half *)raw;
    __half *buf1 = (__half *)(raw + BC * FA46_STRIDE * sizeof(__half));
    int hs = seq_len * head_dim;
    const __half *Qh = Q + bh * hs, *Kh = K + bh * hs, *Vh = V + bh * hs;
    __half *Oh = O + bh * hs;

    // Load Q tile (always BR=64 rows)
    load_tile_t<FA46_BR>(buf0, Qh, qs, FA46_BR, seq_len, head_dim);
    cpa_commit();
    cpa_wait<0>();
    __syncthreads();
    int mrb = wid * 16;
    uint32_t Qr[8][4];
#pragma unroll
    for (int ks = 0; ks < 8; ks++)
        ld_a_sw(Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3],
                buf0, FA46_STRIDE, mrb, ks * 16, lane);
    __syncthreads();

    float Or[16][4];
#pragma unroll
    for (int t = 0; t < 16; t++)
        Or[t][0] = Or[t][1] = Or[t][2] = Or[t][3] = 0;
    float rmax[2] = {-1e30f, -1e30f}, rsexp[2] = {0.0f, 0.0f};
    int nkv = (seq_len + BC - 1) / BC;
    load_tile_t<BC>(buf0, Kh, 0, BC, seq_len, head_dim);
    cpa_commit();

    const __half2 h2_log2e = __float2half2_rn(1.4426950408889634f);

    for (int kv = 0; kv < nkv; kv++)
    {
        int kvs = kv * BC;
        if (causal && kvs > qs + FA46_BR - 1)
            break;
        __half *cur = (kv & 1) ? buf1 : buf0;
        __half *nxt = (kv & 1) ? buf0 : buf1;
        cpa_wait<0>();
        __syncthreads();

        // ==== QK^T ====
        float Sr[NTK][4];
#pragma unroll
        for (int nt = 0; nt < NTK; nt++)
            Sr[nt][0] = Sr[nt][1] = Sr[nt][2] = Sr[nt][3] = 0;
#pragma unroll
        for (int ks = 0; ks < 8; ks++)
        {
#pragma unroll
            for (int nt = 0; nt < NTK; nt++)
            {
                uint32_t b0, b1;
                ld_b_sw(b0, b1, cur, FA46_STRIDE, nt * 8, ks * 16, lane);
                mma16816(Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3],
                         Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3],
                         b0, b1, Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3]);
            }
        }
        __syncthreads();
        load_tile_t<BC>(cur, Vh, kvs, BC, seq_len, head_dim);
        cpa_commit();
        int nkvs = (kv + 1) * BC;
        bool has_nxt = (kv + 1 < nkv) && (!causal || nkvs <= qs + FA46_BR - 1);
        if (has_nxt)
            load_tile_t<BC>(nxt, Kh, nkvs, BC, seq_len, head_dim);
        cpa_commit();

        // Scale + causal mask
#pragma unroll
        for (int nt = 0; nt < NTK; nt++)
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
        for (int nt = 0; nt < NTK; nt++)
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

        // Rescale O via FP16 ex2 (v54 style — no branching)
        __half h_diff0 = __float2half(rmax[0] - nm[0]);
        __half h_diff1 = __float2half(rmax[1] - nm[1]);
        __half2 h2_diff = __halves2half2(h_diff0, h_diff1);
        h2_diff = __hmul2(h2_diff, h2_log2e);
        uint32_t rsc_packed = hexp2x2(*reinterpret_cast<uint32_t *>(&h2_diff));
        __half2 h2_rsc = *reinterpret_cast<__half2 *>(&rsc_packed);
        float rsc0 = __half2float(__low2half(h2_rsc));
        float rsc1 = __half2float(__high2half(h2_rsc));

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

        // Exp + sum + P pack — FP16 ex2
        float ns[2] = {0.0f, 0.0f};
        // Pr tiles: for PV MMA, we need NTPV tiles along K-dim
        // Each Pr[ks] packs 16 rows of P (4 mma fragments of 8 rows × 2 halves)
        uint32_t Pr[NTPV][4];

#pragma unroll
        for (int nt = 0; nt < NTK; nt++)
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

            // Pack into Pr: every 2 nt tiles → 1 Pr tile (16 K-rows per mma K-step)
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

        // ==== PV ====
        cpa_wait<1>();
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < NTPV; ks++)
        {
#pragma unroll
            for (int nt = 0; nt < 16; nt++)
            {
                uint32_t b0, b1;
                ld_b_vt(b0, b1, cur, FA46_STRIDE, nt * 8, ks * 16, lane);
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
// C API — dispatches to best Bc for given seq_len
// =============================================================================
static int g_smem46_32 = 0, g_smem46_64 = 0, g_smem46_128 = 0;

static int setup_smem(int smem, void *kernel_ptr)
{
    cudaError_t e = cudaFuncSetAttribute(
        kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    return (e != cudaSuccess) ? (int)e : 0;
}

extern "C"
{
    int flash_attention_v46_forward(
        const void *Q, const void *K, const void *V, void *O,
        int total_heads, int seq_len, int head_dim, int causal, void *stream)
    {
        if (head_dim != 128)
            return -1;
        float sc = 1.0f / sqrtf((float)head_dim);
        int nqt = (seq_len + FA46_BR - 1) / FA46_BR;
        dim3 grid(total_heads * nqt);
        cudaStream_t st = (cudaStream_t)stream;

        // Dispatch based on seq_len
        if (seq_len > 2048)
        {
            // Bc=128: fewer iterations, more smem (64KB)
            constexpr int BC = 128;
            int smem = 2 * BC * FA46_STRIDE * (int)sizeof(__half);
            if (smem > g_smem46_128)
            {
                int r = setup_smem(smem, (void *)fa46_kernel<BC>);
                if (r)
                    return r;
                g_smem46_128 = smem;
            }
            fa46_kernel<BC><<<grid, FA46_THREADS, smem, st>>>(
                (const __half *)Q, (const __half *)K, (const __half *)V, (__half *)O,
                seq_len, head_dim, causal, sc);
        }
        else if (seq_len > 512)
        {
            // Bc=64: balanced (default, same as v54)
            constexpr int BC = 64;
            int smem = 2 * BC * FA46_STRIDE * (int)sizeof(__half);
            if (smem > g_smem46_64)
            {
                int r = setup_smem(smem, (void *)fa46_kernel<BC>);
                if (r)
                    return r;
                g_smem46_64 = smem;
            }
            fa46_kernel<BC><<<grid, FA46_THREADS, smem, st>>>(
                (const __half *)Q, (const __half *)K, (const __half *)V, (__half *)O,
                seq_len, head_dim, causal, sc);
        }
        else
        {
            // Bc=32: less smem, potentially higher occupancy
            constexpr int BC = 32;
            int smem = 2 * BC * FA46_STRIDE * (int)sizeof(__half);
            if (smem > g_smem46_32)
            {
                int r = setup_smem(smem, (void *)fa46_kernel<BC>);
                if (r)
                    return r;
                g_smem46_32 = smem;
            }
            fa46_kernel<BC><<<grid, FA46_THREADS, smem, st>>>(
                (const __half *)Q, (const __half *)K, (const __half *)V, (__half *)O,
                seq_len, head_dim, causal, sc);
        }
        return (int)cudaGetLastError();
    }
}
