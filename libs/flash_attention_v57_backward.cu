// =============================================================================
// FlashAttention v57 — Backward pass, 3-stage cp.async pipeline
//
// Change vs v56:
//   • 3-stage pipeline: prefetch two K/V (Pass 1) or Q/dO (Pass 2) tiles ahead
//     instead of one, hiding more global-memory latency behind compute.
//
// SMEM trick (sm_120 caps at 99 KB/block, can't naively add a 3rd buffer):
//   Pass 1:
//     Phase A — load Q, dO into SMEM at raw[0..16K) and raw[16..32K), frag-load
//               them to registers (Qr, dOr), __syncthreads().
//     Phase B — reinterpret raw[] as smK[3] + smV[3] (96 KB total). The smK[0]
//               and smK[1] regions overlap the discarded Q,dO staging. From
//               this point on, smQ/smdO are dead — only Qr/dOr registers remain.
//   Pass 2 (mirror) — same trick with K,V → smQ[3] + smdO[3].
//
// Pipeline structure:
//   • Pre-loop: issue prefetch for iter 0 (commit g0) and iter 1 (commit g1).
//   • Per iter i:
//       cpa_wait<1>()  ← drain the oldest group (iter i's data lands)
//       __syncthreads()
//       if (i + 2 < nblk) issue prefetch iter i+2
//       cpa_commit()
//       [compute MMAs using buffer (i%3)]
//   Steady-state: 2 prefetches in flight, overlapping with compute.
//
// Targets sm_80+ (works on Ampere/Ada/Hopper/Blackwell).
//
// Standalone test: nvcc -O3 -arch=sm_120 flash_attention_v57_backward.cu -o fa_v57_bw
// Shared library:   nvcc … -DBUILD_AS_LIB --shared --compiler-options -fPIC …
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
// Helpers — duplicated from v54 forward so this TU is self-contained.
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
    for (int c = threadIdx.x; c < total; c += FA_THREADS)
    {
        int row = c / CPR, lch = c % CPR, gr = start + row;
        int pch = lch ^ (row & 7);
        cpa16(&dst[row * FA_STRIDE + pch * 8],
              &src[gr * head_dim + lch * 8], (gr < seq_len) ? 16 : 0);
    }
}

__device__ __forceinline__ uint32_t hexp2x2(uint32_t in)
{
    uint32_t out;
    asm("ex2.approx.f16x2 %0, %1;" : "=r"(out) : "r"(in));
    return out;
}

// =============================================================================
// Pre-compute D_i = Σ_d dO_i[d] · O_i[d] per row.
// Stored as float[batch_head * seq_len].
// =============================================================================

__global__ void compute_D_kernel(
    const __half *__restrict__ dO, const __half *__restrict__ O,
    float *__restrict__ D, int seq_len, int head_dim)
{
    int bh = blockIdx.x;
    int q = blockIdx.y * blockDim.x + threadIdx.x;
    if (q >= seq_len)
        return;
    int hs = seq_len * head_dim;
    const __half *dOh = dO + bh * hs;
    const __half *Oh = O + bh * hs;
    float acc = 0.0f;
    for (int d = 0; d < head_dim; d++)
    {
        acc += __half2float(dOh[q * head_dim + d]) * __half2float(Oh[q * head_dim + d]);
    }
    D[bh * seq_len + q] = acc;
}

// =============================================================================
// Pass 1 — dQ kernel.
//
// Grid: (batch_head, n_q_blocks)
// Block: FA_THREADS = 128 (4 warps), each warp owns 16 Q-rows.
//
// Per block:
//   • Load Q-block [64, D] into shared memory (resident across K-iterations).
//   • Maintain dQ accumulator [16, D] per warp in registers (Or[16][4]).
//   • Iterate over K-blocks (Bc=64 each):
//       1. Load K, V tiles into SMEM (double-buffered).
//       2. S = Q · Kᵀ · scale  (m16n8k16 MMA, 8 n-tiles, 8 k-steps per warp).
//       3. Apply causal mask.
//       4. P = exp(S − LSE)  via FP16 ex2.approx.f16x2 (matches v54 forward).
//       5. dP = dO · Vᵀ  (m16n8k16 MMA).
//       6. dS = P · (dP − D)  (per-element FP32, then convert to half for MMA).
//       7. dQ += dS · K · scale  (m16n8k16 MMA).
//   • After all K-blocks: write dQ to global.
// =============================================================================

__global__ void __launch_bounds__(FA_THREADS, 1)
    fa57_backward_dq_kernel(
        const __half *__restrict__ Q, const __half *__restrict__ K,
        const __half *__restrict__ V, const __half *__restrict__ dO,
        const float *__restrict__ LSE, const float *__restrict__ D,
        __half *__restrict__ dQ_out,
        int seq_len, int head_dim, int causal, float scale)
{
    int bh = blockIdx.x;
    int qt = blockIdx.y;
    int qs = qt * FA_BR;
    if (qs >= seq_len)
        return;

    int wid = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int gid = lane >> 2;
    int tid = lane & 3;

    extern __shared__ char raw[];
    // SMEM layout (96 KB total = max under sm_120 99 KB cap):
    //   Phase A (initial):   smQ at raw[0..16K), smdO at raw[16..32K)
    //   Phase B (steady):    smK[3] at raw[0..48K), smV[3] at raw[48..96K)
    //   The smQ/smdO regions in Phase A overlap smK[0]/smK[1] in Phase B.
    //   After Q,dO are frag-loaded to registers we never touch smQ/smdO again,
    //   so the regions are safe to recycle as K-prefetch destinations.
    __half *base = (__half *)raw;
    __half *smQ = base;
    __half *smdO = base + FA_BR * FA_STRIDE;

    int hs = seq_len * head_dim;
    const __half *Qh = Q + bh * hs;
    const __half *Kh = K + bh * hs;
    const __half *Vh = V + bh * hs;
    const __half *dOh = dO + bh * hs;
    const float *LSEh = LSE + bh * seq_len;
    const float *Dh = D + bh * seq_len;
    __half *dQh = dQ_out + bh * hs;

    // Phase A — stage Q, dO into SMEM, frag-load to registers.
    load_tile(smQ, Qh, qs, FA_BR, seq_len, head_dim);
    load_tile(smdO, dOh, qs, FA_BR, seq_len, head_dim);
    cpa_commit();
    cpa_wait<0>();
    __syncthreads();

    int mrb = wid * 16;
    uint32_t Qr[8][4], dOr[8][4];
#pragma unroll
    for (int ks = 0; ks < 8; ks++)
    {
        ld_a_sw(Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3],
                smQ, FA_STRIDE, mrb, ks * 16, lane);
        ld_a_sw(dOr[ks][0], dOr[ks][1], dOr[ks][2], dOr[ks][3],
                smdO, FA_STRIDE, mrb, ks * 16, lane);
    }
    __syncthreads();  // ensure all threads finished reading Q/dO before recycle

    // Phase B — reinterpret raw[] as smK[3] + smV[3] (overlaps with discarded Q/dO).
    __half *smK[3] = {
        base,
        base + 1 * FA_BC * FA_STRIDE,
        base + 2 * FA_BC * FA_STRIDE,
    };
    __half *smV[3] = {
        base + 3 * FA_BC * FA_STRIDE,
        base + 4 * FA_BC * FA_STRIDE,
        base + 5 * FA_BC * FA_STRIDE,
    };

    // dQ accumulator [16 rows × 128 cols = 16 n-tiles × 4 elements per thread per tile]
    float dQr[16][4];
#pragma unroll
    for (int t = 0; t < 16; t++)
        dQr[t][0] = dQr[t][1] = dQr[t][2] = dQr[t][3] = 0.0f;

    // Cache per-row LSE and D for the 16 rows this warp owns (2 rows per thread).
    int gr0 = qs + mrb + gid;
    int gr8 = gr0 + 8;
    float lse0 = (gr0 < seq_len) ? LSEh[gr0] : 0.0f;
    float lse8 = (gr8 < seq_len) ? LSEh[gr8] : 0.0f;
    float D0 = (gr0 < seq_len) ? Dh[gr0] : 0.0f;
    float D8 = (gr8 < seq_len) ? Dh[gr8] : 0.0f;

    int nkv = (seq_len + FA_BC - 1) / FA_BC;
    int kv_max_blocks = causal ? ((qs + FA_BR - 1) / FA_BC + 1) : nkv;
    if (kv_max_blocks > nkv)
        kv_max_blocks = nkv;

    // 3-stage prefetch: issue first TWO K/V blocks before entering the loop.
    load_tile(smK[0], Kh, 0, FA_BC, seq_len, head_dim);
    load_tile(smV[0], Vh, 0, FA_BC, seq_len, head_dim);
    cpa_commit();

    if (kv_max_blocks >= 2)
    {
        load_tile(smK[1], Kh, 1 * FA_BC, FA_BC, seq_len, head_dim);
        load_tile(smV[1], Vh, 1 * FA_BC, FA_BC, seq_len, head_dim);
    }
    cpa_commit();

    const __half2 h2_log2e = __float2half2_rn(1.4426950408889634f);

    for (int kv = 0; kv < kv_max_blocks; kv++)
    {
        int kvs = kv * FA_BC;
        int buf = kv % 3;
        __half *smKcur = smK[buf];
        __half *smVcur = smV[buf];

        // Drain the oldest pending group: this releases buffer `buf` filled with K[kv], V[kv].
        cpa_wait<1>();
        __syncthreads();

        // Schedule the prefetch for kv+2 NOW so it overlaps with the compute below.
        if (kv + 2 < kv_max_blocks)
        {
            int nbuf = (kv + 2) % 3;
            load_tile(smK[nbuf], Kh, (kv + 2) * FA_BC, FA_BC, seq_len, head_dim);
            load_tile(smV[nbuf], Vh, (kv + 2) * FA_BC, FA_BC, seq_len, head_dim);
        }
        cpa_commit();

        // ── S = Q · Kᵀ · scale  (per warp: 16 rows × Bc=64 cols) ──
        float Sr[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
            Sr[nt][0] = Sr[nt][1] = Sr[nt][2] = Sr[nt][3] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < 8; ks++)
        {
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
            {
                uint32_t b0, b1;
                ld_b_sw(b0, b1, smKcur, FA_STRIDE, nt * 8, ks * 16, lane);
                mma16816(Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3],
                         Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3],
                         b0, b1, Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3]);
            }
        }

        // Apply scale + causal mask + subtract LSE → exponent input.
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            // Output layout owned by this thread: 4 elements split across two
            // rows (gr0 and gr8) and two cols within a 16×8 tile.
            int gk0 = kvs + nt * 8 + tid * 2;
            int gk1 = gk0 + 1;

            Sr[nt][0] = Sr[nt][0] * scale - lse0;
            Sr[nt][1] = Sr[nt][1] * scale - lse0;
            Sr[nt][2] = Sr[nt][2] * scale - lse8;
            Sr[nt][3] = Sr[nt][3] * scale - lse8;

            if (causal)
            {
                if (gk0 > gr0)
                    Sr[nt][0] = -1e30f;
                if (gk1 > gr0)
                    Sr[nt][1] = -1e30f;
                if (gk0 > gr8)
                    Sr[nt][2] = -1e30f;
                if (gk1 > gr8)
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

        // ── P = exp(Sr) via FP16 ex2.approx.f16x2 ──
        // Pr layout matches what's expected by next MMA as A operand.
        uint32_t Pr[4][4];
        float Pf[8][4]; // also keep FP32 copies for dS calculation later
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            __half2 h01 = __halves2half2(__float2half(Sr[nt][0]), __float2half(Sr[nt][1]));
            __half2 h23 = __halves2half2(__float2half(Sr[nt][2]), __float2half(Sr[nt][3]));
            h01 = __hmul2(h01, h2_log2e);
            h23 = __hmul2(h23, h2_log2e);
            uint32_t e01 = hexp2x2(*reinterpret_cast<uint32_t *>(&h01));
            uint32_t e23 = hexp2x2(*reinterpret_cast<uint32_t *>(&h23));
            __half2 r01 = *reinterpret_cast<__half2 *>(&e01);
            __half2 r23 = *reinterpret_cast<__half2 *>(&e23);

            Pf[nt][0] = __half2float(__low2half(r01));
            Pf[nt][1] = __half2float(__high2half(r01));
            Pf[nt][2] = __half2float(__low2half(r23));
            Pf[nt][3] = __half2float(__high2half(r23));

            int pi = nt / 2, half = nt % 2;
            __half2 *p = (__half2 *)Pr[pi];
            p[half * 2] = r01;
            p[half * 2 + 1] = r23;
        }

        // ── dP = dO · Vᵀ  (m16n8k16, Vᵀ via ld_b_sw — same transpose semantics
        //    as QKᵀ forward; ld_b_vt would walk rows past smV's 64-row tile.) ──
        float dPr[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
            dPr[nt][0] = dPr[nt][1] = dPr[nt][2] = dPr[nt][3] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < 8; ks++)
        {
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
            {
                uint32_t b0, b1;
                ld_b_sw(b0, b1, smVcur, FA_STRIDE, nt * 8, ks * 16, lane);
                mma16816(dPr[nt][0], dPr[nt][1], dPr[nt][2], dPr[nt][3],
                         dOr[ks][0], dOr[ks][1], dOr[ks][2], dOr[ks][3],
                         b0, b1, dPr[nt][0], dPr[nt][1], dPr[nt][2], dPr[nt][3]);
            }
        }

        // ── dS = P · (dP − D)  → packed FP16 fragments for next MMA ──
        uint32_t dSr[4][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            float dS0 = Pf[nt][0] * (dPr[nt][0] - D0);
            float dS1 = Pf[nt][1] * (dPr[nt][1] - D0);
            float dS2 = Pf[nt][2] * (dPr[nt][2] - D8);
            float dS3 = Pf[nt][3] * (dPr[nt][3] - D8);

            // Multiply by scale once now so the dQ MMA doesn't need to scale.
            dS0 *= scale;
            dS1 *= scale;
            dS2 *= scale;
            dS3 *= scale;

            __half2 h01 = __halves2half2(__float2half(dS0), __float2half(dS1));
            __half2 h23 = __halves2half2(__float2half(dS2), __float2half(dS3));

            int pi = nt / 2, half = nt % 2;
            __half2 *p = (__half2 *)dSr[pi];
            p[half * 2] = h01;
            p[half * 2 + 1] = h23;
        }

        // ── dQ += dS · K   (Kr from smK loaded as B operand) ──
#pragma unroll
        for (int ks = 0; ks < 4; ks++)
        {
#pragma unroll
            for (int nt = 0; nt < 16; nt++)
            {
                uint32_t b0, b1;
                ld_b_vt(b0, b1, smKcur, FA_STRIDE, nt * 8, ks * 16, lane);
                mma16816(dQr[nt][0], dQr[nt][1], dQr[nt][2], dQr[nt][3],
                         dSr[ks][0], dSr[ks][1], dSr[ks][2], dSr[ks][3],
                         b0, b1, dQr[nt][0], dQr[nt][1], dQr[nt][2], dQr[nt][3]);
            }
        }
    }

    // ── Write dQ to global (vectorized __half2 stores) ──
#pragma unroll
    for (int nt = 0; nt < 16; nt++)
    {
        int c0 = nt * 8 + tid * 2, c1 = c0 + 1;
        bool pair_ok = (c1 < head_dim);
        if (gr0 < seq_len)
        {
            if (pair_ok)
            {
                __half2 v = __halves2half2(__float2half(dQr[nt][0]), __float2half(dQr[nt][1]));
                *reinterpret_cast<__half2 *>(&dQh[gr0 * head_dim + c0]) = v;
            }
            else if (c0 < head_dim)
                dQh[gr0 * head_dim + c0] = __float2half(dQr[nt][0]);
        }
        if (gr8 < seq_len)
        {
            if (pair_ok)
            {
                __half2 v = __halves2half2(__float2half(dQr[nt][2]), __float2half(dQr[nt][3]));
                *reinterpret_cast<__half2 *>(&dQh[gr8 * head_dim + c0]) = v;
            }
            else if (c0 < head_dim)
                dQh[gr8 * head_dim + c0] = __float2half(dQr[nt][2]);
        }
    }
}

// =============================================================================
// Pass 2 — dKdV kernel.
//
// Grid: (batch_head, n_kv_blocks)
// Block: FA_THREADS = 128 (4 warps), each warp owns 16 K-rows.
//
// Per block:
//   • Load K-block [Bc, D] and V-block [Bc, D] into SMEM (resident).
//   • Maintain dK, dV accumulators [16 rows × D cols] per warp in registers.
//   • Iterate over Q-blocks (Br=64 each):
//       1. Load Q, dO tiles into SMEM (double-buffered).
//       2. S = K · Qᵀ · scale  → produces S in [K-row, Q-col] layout, which
//          equals P_transposed conceptually.
//       3. Apply causal mask + subtract LSE_q per Q-col.
//       4. P = exp(S − LSE) via FP16 ex2.
//       5. dV += P · dO  (Pr already in [K-row, Q-row] = A-operand layout).
//       6. dP = V · dOᵀ.
//       7. dS = P · (dP − D_q) · scale.
//       8. dK += dS · Q.
//   • After all Q-blocks: write dK, dV to global.
//
// Causal: only Q-blocks with qs >= kvs contribute (K[k]·Q[q] active iff k <= q).
// =============================================================================

__global__ void __launch_bounds__(FA_THREADS, 1)
    fa57_backward_dkdv_kernel(
        const __half *__restrict__ Q, const __half *__restrict__ K,
        const __half *__restrict__ V, const __half *__restrict__ dO,
        const float *__restrict__ LSE, const float *__restrict__ D,
        __half *__restrict__ dK_out, __half *__restrict__ dV_out,
        int seq_len, int head_dim, int causal, float scale)
{
    int bh = blockIdx.x;
    int kt = blockIdx.y;
    int kvs = kt * FA_BC;
    if (kvs >= seq_len)
        return;

    int wid = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int gid = lane >> 2;
    int tid = lane & 3;

    extern __shared__ char raw[];
    // SMEM layout (mirror of Pass 1):
    //   Phase A (initial):   smK at raw[0..16K), smV at raw[16..32K)
    //   Phase B (steady):    smQ[3] at raw[0..48K), smdO[3] at raw[48..96K)
    //   After K,V frag-loaded to registers we never touch smK/smV again.
    __half *base = (__half *)raw;
    __half *smK = base;
    __half *smV = base + FA_BC * FA_STRIDE;

    int hs = seq_len * head_dim;
    const __half *Qh = Q + bh * hs;
    const __half *Kh = K + bh * hs;
    const __half *Vh = V + bh * hs;
    const __half *dOh = dO + bh * hs;
    const float *LSEh = LSE + bh * seq_len;
    const float *Dh = D + bh * seq_len;
    __half *dKh = dK_out + bh * hs;
    __half *dVh = dV_out + bh * hs;

    // Phase A — stage K, V into SMEM, frag-load to registers.
    load_tile(smK, Kh, kvs, FA_BC, seq_len, head_dim);
    load_tile(smV, Vh, kvs, FA_BC, seq_len, head_dim);
    cpa_commit();
    cpa_wait<0>();
    __syncthreads();

    int mrb = wid * 16;
    uint32_t Kr[8][4], Vr[8][4];
#pragma unroll
    for (int ks = 0; ks < 8; ks++)
    {
        ld_a_sw(Kr[ks][0], Kr[ks][1], Kr[ks][2], Kr[ks][3],
                smK, FA_STRIDE, mrb, ks * 16, lane);
        ld_a_sw(Vr[ks][0], Vr[ks][1], Vr[ks][2], Vr[ks][3],
                smV, FA_STRIDE, mrb, ks * 16, lane);
    }
    __syncthreads();  // ensure all threads finished reading K/V before recycle

    // Phase B — reinterpret raw[] as smQ[3] + smdO[3] (overlaps with discarded K/V).
    __half *smQ[3] = {
        base,
        base + 1 * FA_BR * FA_STRIDE,
        base + 2 * FA_BR * FA_STRIDE,
    };
    __half *smdO[3] = {
        base + 3 * FA_BR * FA_STRIDE,
        base + 4 * FA_BR * FA_STRIDE,
        base + 5 * FA_BR * FA_STRIDE,
    };

    float dKr[16][4], dVr[16][4];
#pragma unroll
    for (int t = 0; t < 16; t++)
    {
        dKr[t][0] = dKr[t][1] = dKr[t][2] = dKr[t][3] = 0.0f;
        dVr[t][0] = dVr[t][1] = dVr[t][2] = dVr[t][3] = 0.0f;
    }

    int nq = (seq_len + FA_BR - 1) / FA_BR;
    int q_start = causal ? (kvs / FA_BR) : 0;
    if (q_start >= nq)
        return;
    int q_count = nq - q_start;

    // 3-stage prefetch: issue first TWO Q/dO blocks before entering the loop.
    load_tile(smQ[0], Qh, q_start * FA_BR, FA_BR, seq_len, head_dim);
    load_tile(smdO[0], dOh, q_start * FA_BR, FA_BR, seq_len, head_dim);
    cpa_commit();

    if (q_count >= 2)
    {
        load_tile(smQ[1], Qh, (q_start + 1) * FA_BR, FA_BR, seq_len, head_dim);
        load_tile(smdO[1], dOh, (q_start + 1) * FA_BR, FA_BR, seq_len, head_dim);
    }
    cpa_commit();

    const __half2 h2_log2e = __float2half2_rn(1.4426950408889634f);

    for (int qb = q_start; qb < nq; qb++)
    {
        int qs = qb * FA_BR;
        int loopi = qb - q_start;
        int buf = loopi % 3;
        __half *smQcur = smQ[buf];
        __half *smdOcur = smdO[buf];

        // Drain the oldest pending group: this releases buffer `buf` for iter loopi.
        cpa_wait<1>();
        __syncthreads();

        // Schedule prefetch for loopi+2 NOW so it overlaps with the compute below.
        if (loopi + 2 < q_count)
        {
            int nbuf = (loopi + 2) % 3;
            int next_qb = q_start + loopi + 2;
            load_tile(smQ[nbuf], Qh, next_qb * FA_BR, FA_BR, seq_len, head_dim);
            load_tile(smdO[nbuf], dOh, next_qb * FA_BR, FA_BR, seq_len, head_dim);
        }
        cpa_commit();

        // ── S = K · Qᵀ · scale  (per warp: 16 K-rows × Br=64 Q-cols) ──
        // Result layout: S[K-row = mrb+gid/gid+8, Q-col = nt*8+tid*2/+1].
        float Sr[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
            Sr[nt][0] = Sr[nt][1] = Sr[nt][2] = Sr[nt][3] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < 8; ks++)
        {
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
            {
                uint32_t b0, b1;
                ld_b_sw(b0, b1, smQcur, FA_STRIDE, nt * 8, ks * 16, lane);
                mma16816(Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3],
                         Kr[ks][0], Kr[ks][1], Kr[ks][2], Kr[ks][3],
                         b0, b1, Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3]);
            }
        }

        // Apply scale, subtract LSE_q, apply causal mask. K-row = gk, Q-col = gq;
        // attention is valid iff gk <= gq.
        int gk0 = kvs + mrb + gid;
        int gk8 = gk0 + 8;
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            int gq0 = qs + nt * 8 + tid * 2;
            int gq1 = gq0 + 1;
            float lseA = (gq0 < seq_len) ? LSEh[gq0] : 0.0f;
            float lseB = (gq1 < seq_len) ? LSEh[gq1] : 0.0f;

            Sr[nt][0] = Sr[nt][0] * scale - lseA;
            Sr[nt][1] = Sr[nt][1] * scale - lseB;
            Sr[nt][2] = Sr[nt][2] * scale - lseA;
            Sr[nt][3] = Sr[nt][3] * scale - lseB;

            if (causal)
            {
                if (gk0 > gq0)
                    Sr[nt][0] = -1e30f;
                if (gk0 > gq1)
                    Sr[nt][1] = -1e30f;
                if (gk8 > gq0)
                    Sr[nt][2] = -1e30f;
                if (gk8 > gq1)
                    Sr[nt][3] = -1e30f;
            }
            if (gk0 >= seq_len)
            {
                Sr[nt][0] = -1e30f;
                Sr[nt][1] = -1e30f;
            }
            if (gk8 >= seq_len)
            {
                Sr[nt][2] = -1e30f;
                Sr[nt][3] = -1e30f;
            }
            if (gq0 >= seq_len)
            {
                Sr[nt][0] = -1e30f;
                Sr[nt][2] = -1e30f;
            }
            if (gq1 >= seq_len)
            {
                Sr[nt][1] = -1e30f;
                Sr[nt][3] = -1e30f;
            }
        }

        // ── P = exp(Sr) via FP16 ex2 ──
        uint32_t Pr[4][4];
        float Pf[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            __half2 h01 = __halves2half2(__float2half(Sr[nt][0]), __float2half(Sr[nt][1]));
            __half2 h23 = __halves2half2(__float2half(Sr[nt][2]), __float2half(Sr[nt][3]));
            h01 = __hmul2(h01, h2_log2e);
            h23 = __hmul2(h23, h2_log2e);
            uint32_t e01 = hexp2x2(*reinterpret_cast<uint32_t *>(&h01));
            uint32_t e23 = hexp2x2(*reinterpret_cast<uint32_t *>(&h23));
            __half2 r01 = *reinterpret_cast<__half2 *>(&e01);
            __half2 r23 = *reinterpret_cast<__half2 *>(&e23);

            Pf[nt][0] = __half2float(__low2half(r01));
            Pf[nt][1] = __half2float(__high2half(r01));
            Pf[nt][2] = __half2float(__low2half(r23));
            Pf[nt][3] = __half2float(__high2half(r23));

            int pi = nt / 2, half = nt % 2;
            __half2 *p = (__half2 *)Pr[pi];
            p[half * 2] = r01;
            p[half * 2 + 1] = r23;
        }

        // ── dV += P · dO   (P already in [K-row, Q-row] = A-operand layout) ──
        //   m=16 K-rows, n=8 head_dim chunk, k=16 Q-rows.
#pragma unroll
        for (int ks = 0; ks < 4; ks++)
        {
#pragma unroll
            for (int nt = 0; nt < 16; nt++)
            {
                uint32_t b0, b1;
                ld_b_vt(b0, b1, smdOcur, FA_STRIDE, nt * 8, ks * 16, lane);
                mma16816(dVr[nt][0], dVr[nt][1], dVr[nt][2], dVr[nt][3],
                         Pr[ks][0], Pr[ks][1], Pr[ks][2], Pr[ks][3],
                         b0, b1, dVr[nt][0], dVr[nt][1], dVr[nt][2], dVr[nt][3]);
            }
        }

        // ── dP = V · dOᵀ   (V resident in Vr; dO transposed via ld_b_sw) ──
        //   m=16 K-rows, n=8 Q-cols, k=16 head_dim.
        float dPr[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
            dPr[nt][0] = dPr[nt][1] = dPr[nt][2] = dPr[nt][3] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < 8; ks++)
        {
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
            {
                uint32_t b0, b1;
                ld_b_sw(b0, b1, smdOcur, FA_STRIDE, nt * 8, ks * 16, lane);
                mma16816(dPr[nt][0], dPr[nt][1], dPr[nt][2], dPr[nt][3],
                         Vr[ks][0], Vr[ks][1], Vr[ks][2], Vr[ks][3],
                         b0, b1, dPr[nt][0], dPr[nt][1], dPr[nt][2], dPr[nt][3]);
            }
        }

        // ── dS = P · (dP − D_q) · scale  → packed FP16 fragments ──
        uint32_t dSr[4][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            int gq0 = qs + nt * 8 + tid * 2;
            int gq1 = gq0 + 1;
            float Dq0 = (gq0 < seq_len) ? Dh[gq0] : 0.0f;
            float Dq1 = (gq1 < seq_len) ? Dh[gq1] : 0.0f;

            float dS0 = Pf[nt][0] * (dPr[nt][0] - Dq0);
            float dS1 = Pf[nt][1] * (dPr[nt][1] - Dq1);
            float dS2 = Pf[nt][2] * (dPr[nt][2] - Dq0);
            float dS3 = Pf[nt][3] * (dPr[nt][3] - Dq1);
            dS0 *= scale;
            dS1 *= scale;
            dS2 *= scale;
            dS3 *= scale;

            __half2 h01 = __halves2half2(__float2half(dS0), __float2half(dS1));
            __half2 h23 = __halves2half2(__float2half(dS2), __float2half(dS3));

            int pi = nt / 2, half = nt % 2;
            __half2 *p = (__half2 *)dSr[pi];
            p[half * 2] = h01;
            p[half * 2 + 1] = h23;
        }

        // ── dK += dS · Q   (dS in [K-row, Q-row] = A-operand layout; Q via ld_b_vt) ──
        //   m=16 K-rows, n=8 head_dim chunk, k=16 Q-rows.
#pragma unroll
        for (int ks = 0; ks < 4; ks++)
        {
#pragma unroll
            for (int nt = 0; nt < 16; nt++)
            {
                uint32_t b0, b1;
                ld_b_vt(b0, b1, smQcur, FA_STRIDE, nt * 8, ks * 16, lane);
                mma16816(dKr[nt][0], dKr[nt][1], dKr[nt][2], dKr[nt][3],
                         dSr[ks][0], dSr[ks][1], dSr[ks][2], dSr[ks][3],
                         b0, b1, dKr[nt][0], dKr[nt][1], dKr[nt][2], dKr[nt][3]);
            }
        }
    }

    // ── Write dK, dV to global (vectorized __half2 stores) ──
    int gr0 = kvs + mrb + gid;
    int gr8 = gr0 + 8;
#pragma unroll
    for (int nt = 0; nt < 16; nt++)
    {
        int c0 = nt * 8 + tid * 2, c1 = c0 + 1;
        bool pair_ok = (c1 < head_dim);
        if (gr0 < seq_len)
        {
            if (pair_ok)
            {
                __half2 vK = __halves2half2(__float2half(dKr[nt][0]), __float2half(dKr[nt][1]));
                __half2 vV = __halves2half2(__float2half(dVr[nt][0]), __float2half(dVr[nt][1]));
                *reinterpret_cast<__half2 *>(&dKh[gr0 * head_dim + c0]) = vK;
                *reinterpret_cast<__half2 *>(&dVh[gr0 * head_dim + c0]) = vV;
            }
            else if (c0 < head_dim)
            {
                dKh[gr0 * head_dim + c0] = __float2half(dKr[nt][0]);
                dVh[gr0 * head_dim + c0] = __float2half(dVr[nt][0]);
            }
        }
        if (gr8 < seq_len)
        {
            if (pair_ok)
            {
                __half2 vK = __halves2half2(__float2half(dKr[nt][2]), __float2half(dKr[nt][3]));
                __half2 vV = __halves2half2(__float2half(dVr[nt][2]), __float2half(dVr[nt][3]));
                *reinterpret_cast<__half2 *>(&dKh[gr8 * head_dim + c0]) = vK;
                *reinterpret_cast<__half2 *>(&dVh[gr8 * head_dim + c0]) = vV;
            }
            else if (c0 < head_dim)
            {
                dKh[gr8 * head_dim + c0] = __float2half(dKr[nt][2]);
                dVh[gr8 * head_dim + c0] = __float2half(dVr[nt][2]);
            }
        }
    }
}

// =============================================================================
// Public launchers
// =============================================================================

static int g_smem_dq = 0;
static int g_smem_dkdv = 0;

extern "C" void launch_v57_backward_dq(
    const __half *Q, const __half *K, const __half *V,
    const __half *dO, const float *LSE, const float *D,
    __half *dQ_out,
    int th, int sl, int hd, int ca)
{
    float scale = 1.0f / sqrtf((float)hd);
    int nqt = (sl + FA_BR - 1) / FA_BR;

    int smem = (2 * FA_BR + 4 * FA_BC) * FA_STRIDE * (int)sizeof(__half);
    if (smem > g_smem_dq)
    {
        cudaFuncSetAttribute(fa57_backward_dq_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        g_smem_dq = smem;
    }

    dim3 grid(th, nqt);
    fa57_backward_dq_kernel<<<grid, FA_THREADS, smem>>>(
        Q, K, V, dO, LSE, D, dQ_out, sl, hd, ca, scale);
}

extern "C" void launch_compute_D(
    const __half *dO, const __half *O, float *D,
    int th, int sl, int hd)
{
    const int THREADS = 128;
    dim3 grid(th, (sl + THREADS - 1) / THREADS);
    compute_D_kernel<<<grid, THREADS>>>(dO, O, D, sl, hd);
}

extern "C" void launch_v57_backward_dkdv(
    const __half *Q, const __half *K, const __half *V,
    const __half *dO, const float *LSE, const float *D,
    __half *dK_out, __half *dV_out,
    int th, int sl, int hd, int ca)
{
    float scale = 1.0f / sqrtf((float)hd);
    int nkv = (sl + FA_BC - 1) / FA_BC;

    int smem = (2 * FA_BC + 4 * FA_BR) * FA_STRIDE * (int)sizeof(__half);
    if (smem > g_smem_dkdv)
    {
        cudaFuncSetAttribute(fa57_backward_dkdv_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        g_smem_dkdv = smem;
    }

    dim3 grid(th, nkv);
    fa57_backward_dkdv_kernel<<<grid, FA_THREADS, smem>>>(
        Q, K, V, dO, LSE, D, dK_out, dV_out, sl, hd, ca, scale);
}

// Combined entry-point: runs compute_D → Pass 1 (dQ) → Pass 2 (dKdV).
// dQ_out, dK_out, dV_out must be zeroed by the caller.
extern "C" void launch_v57_backward(
    const __half *Q, const __half *K, const __half *V,
    const __half *dO, const __half *O, const float *LSE,
    __half *dQ_out, __half *dK_out, __half *dV_out,
    float *D_scratch,  // [th * sl]
    int th, int sl, int hd, int ca)
{
    launch_compute_D(dO, O, D_scratch, th, sl, hd);
    launch_v57_backward_dq(Q, K, V, dO, LSE, D_scratch, dQ_out, th, sl, hd, ca);
    launch_v57_backward_dkdv(Q, K, V, dO, LSE, D_scratch, dK_out, dV_out, th, sl, hd, ca);
}

// =============================================================================
// CPU reference + standalone test harness
// (Pass 2 dKdV kernel will be added in a follow-up commit. For now we test
//  dQ only, comparing against a CPU reference that computes all three.)
// =============================================================================
#ifndef BUILD_AS_LIB

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

// CPU reference: same as v54_backward standalone. Returns O, LSE, D, dQ, dK, dV.
void cpu_attention_forward_backward(
    const float *Q, const float *K, const float *V, const float *dO,
    float *O_out, float *LSE_out, float *D_out,
    float *dQ_out, float *dK_out, float *dV_out,
    int th, int sl, int hd, int causal)
{
    float scale = 1.0f / sqrtf((float)hd);
    int hs = sl * hd;
    memset(O_out, 0, sizeof(float) * th * hs);
    memset(dQ_out, 0, sizeof(float) * th * hs);
    memset(dK_out, 0, sizeof(float) * th * hs);
    memset(dV_out, 0, sizeof(float) * th * hs);
    memset(D_out, 0, sizeof(float) * th * sl);

    for (int bh = 0; bh < th; bh++)
    {
        const float *Qh = Q + bh * hs;
        const float *Kh = K + bh * hs;
        const float *Vh = V + bh * hs;
        const float *dOh = dO + bh * hs;
        float *Oh = O_out + bh * hs;
        float *LSEh = LSE_out + bh * sl;
        float *Dh = D_out + bh * sl;
        float *dQh = dQ_out + bh * hs;
        float *dKh = dK_out + bh * hs;
        float *dVh = dV_out + bh * hs;

        float *P = (float *)malloc(sizeof(float) * sl * sl);
        for (int q = 0; q < sl; q++)
        {
            int kv_max = causal ? (q + 1) : sl;
            float rmax = -1e30f;
            for (int k = 0; k < kv_max; k++)
            {
                float s = 0.0f;
                for (int d = 0; d < hd; d++)
                    s += Qh[q * hd + d] * Kh[k * hd + d];
                s *= scale;
                P[q * sl + k] = s;
                if (s > rmax)
                    rmax = s;
            }
            for (int k = kv_max; k < sl; k++)
                P[q * sl + k] = -1e30f;
            float rsum = 0.0f;
            for (int k = 0; k < kv_max; k++)
            {
                P[q * sl + k] = expf(P[q * sl + k] - rmax);
                rsum += P[q * sl + k];
            }
            float inv = (rsum > 0.0f) ? 1.0f / rsum : 0.0f;
            for (int k = 0; k < kv_max; k++)
                P[q * sl + k] *= inv;
            LSEh[q] = logf(fmaxf(rsum, 1e-30f)) + rmax;
            for (int d = 0; d < hd; d++)
            {
                float o = 0.0f;
                for (int k = 0; k < kv_max; k++)
                    o += P[q * sl + k] * Vh[k * hd + d];
                Oh[q * hd + d] = o;
            }
        }
        for (int q = 0; q < sl; q++)
        {
            int kv_max = causal ? (q + 1) : sl;
            float Dv = 0.0f;
            for (int d = 0; d < hd; d++)
                Dv += dOh[q * hd + d] * Oh[q * hd + d];
            Dh[q] = Dv;
            for (int k = 0; k < kv_max; k++)
            {
                float Pqk = P[q * sl + k];
                for (int d = 0; d < hd; d++)
                    dVh[k * hd + d] += Pqk * dOh[q * hd + d];
                float dP = 0.0f;
                for (int d = 0; d < hd; d++)
                    dP += dOh[q * hd + d] * Vh[k * hd + d];
                float dS = Pqk * (dP - Dv);
                float dS_scaled = dS * scale;
                for (int d = 0; d < hd; d++)
                {
                    dQh[q * hd + d] += dS_scaled * Kh[k * hd + d];
                    dKh[k * hd + d] += dS_scaled * Qh[q * hd + d];
                }
            }
        }
        free(P);
    }
}

float maxabs_diff(const uint16_t *a_h, const float *b, int n)
{
    float m = 0.0f;
    for (int i = 0; i < n; i++)
    {
        float d = fabsf(h2f(a_h[i]) - b[i]);
        if (d > m)
            m = d;
    }
    return m;
}

void test_correctness()
{
    printf("--- Correctness (GPU dQ/dK/dV via tensor cores vs CPU reference) ---\n");
    int configs[][4] = {
        {1, 64, 128, 1},
        {1, 128, 128, 1},
        {2, 128, 128, 1},
        {1, 256, 128, 1},
        {2, 512, 128, 1},
        {1, 1024, 128, 1},
    };
    for (auto &c : configs)
    {
        int th = c[0], sl = c[1], hd = c[2], ca = c[3];
        size_t n_elems = (size_t)th * sl * hd;
        size_t lse_elems = (size_t)th * sl;

        float *Qf = (float *)malloc(sizeof(float) * n_elems);
        float *Kf = (float *)malloc(sizeof(float) * n_elems);
        float *Vf = (float *)malloc(sizeof(float) * n_elems);
        float *dOf = (float *)malloc(sizeof(float) * n_elems);
        float *Of_ref = (float *)malloc(sizeof(float) * n_elems);
        float *LSEf_ref = (float *)malloc(sizeof(float) * lse_elems);
        float *Df_ref = (float *)malloc(sizeof(float) * lse_elems);
        float *dQf_ref = (float *)malloc(sizeof(float) * n_elems);
        float *dKf_ref = (float *)malloc(sizeof(float) * n_elems);
        float *dVf_ref = (float *)malloc(sizeof(float) * n_elems);

        for (size_t i = 0; i < n_elems; i++)
        {
            Qf[i] = ((float)(rand() % 2001) - 1000.0f) / 1000.0f;
            Kf[i] = ((float)(rand() % 2001) - 1000.0f) / 1000.0f;
            Vf[i] = ((float)(rand() % 2001) - 1000.0f) / 1000.0f;
            dOf[i] = ((float)(rand() % 2001) - 1000.0f) / 1000.0f;
        }

        cpu_attention_forward_backward(Qf, Kf, Vf, dOf,
                                       Of_ref, LSEf_ref, Df_ref,
                                       dQf_ref, dKf_ref, dVf_ref,
                                       th, sl, hd, ca);

        uint16_t *Qh_cpu = (uint16_t *)malloc(n_elems * 2);
        uint16_t *Kh_cpu = (uint16_t *)malloc(n_elems * 2);
        uint16_t *Vh_cpu = (uint16_t *)malloc(n_elems * 2);
        uint16_t *dOh_cpu = (uint16_t *)malloc(n_elems * 2);
        for (size_t i = 0; i < n_elems; i++)
        {
            Qh_cpu[i] = f2h(Qf[i]);
            Kh_cpu[i] = f2h(Kf[i]);
            Vh_cpu[i] = f2h(Vf[i]);
            dOh_cpu[i] = f2h(dOf[i]);
        }

        __half *Q_d, *K_d, *V_d, *dO_d, *dQ_d, *dK_d, *dV_d;
        float *LSE_d, *D_d;
        CK(cudaMalloc(&Q_d, n_elems * 2));
        CK(cudaMalloc(&K_d, n_elems * 2));
        CK(cudaMalloc(&V_d, n_elems * 2));
        CK(cudaMalloc(&dO_d, n_elems * 2));
        CK(cudaMalloc(&dQ_d, n_elems * 2));
        CK(cudaMalloc(&dK_d, n_elems * 2));
        CK(cudaMalloc(&dV_d, n_elems * 2));
        CK(cudaMalloc(&LSE_d, lse_elems * 4));
        CK(cudaMalloc(&D_d, lse_elems * 4));
        CK(cudaMemset(dQ_d, 0, n_elems * 2));
        CK(cudaMemset(dK_d, 0, n_elems * 2));
        CK(cudaMemset(dV_d, 0, n_elems * 2));
        CK(cudaMemcpy(Q_d, Qh_cpu, n_elems * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(K_d, Kh_cpu, n_elems * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(V_d, Vh_cpu, n_elems * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dO_d, dOh_cpu, n_elems * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(LSE_d, LSEf_ref, lse_elems * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(D_d, Df_ref, lse_elems * 4, cudaMemcpyHostToDevice));

        launch_v57_backward_dq(Q_d, K_d, V_d, dO_d, LSE_d, D_d, dQ_d, th, sl, hd, ca);
        launch_v57_backward_dkdv(Q_d, K_d, V_d, dO_d, LSE_d, D_d, dK_d, dV_d, th, sl, hd, ca);
        CK(cudaDeviceSynchronize());

        uint16_t *dQ_cpu = (uint16_t *)malloc(n_elems * 2);
        uint16_t *dK_cpu = (uint16_t *)malloc(n_elems * 2);
        uint16_t *dV_cpu = (uint16_t *)malloc(n_elems * 2);
        CK(cudaMemcpy(dQ_cpu, dQ_d, n_elems * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(dK_cpu, dK_d, n_elems * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(dV_cpu, dV_d, n_elems * 2, cudaMemcpyDeviceToHost));

        float mQ = maxabs_diff(dQ_cpu, dQf_ref, (int)n_elems);
        float mK = maxabs_diff(dK_cpu, dKf_ref, (int)n_elems);
        float mV = maxabs_diff(dV_cpu, dVf_ref, (int)n_elems);
        const float tol = 0.05f;
        bool ok = (mQ < tol) && (mK < tol) && (mV < tol);
        const char *st = ok ? "PASS" : "FAIL";
        printf("  th=%d sl=%d hd=%d ca=%d  dQ=%.4f dK=%.4f dV=%.4f  %s\n",
               th, sl, hd, ca, mQ, mK, mV, st);

        free(Qf); free(Kf); free(Vf); free(dOf);
        free(Of_ref); free(LSEf_ref); free(Df_ref);
        free(dQf_ref); free(dKf_ref); free(dVf_ref);
        free(Qh_cpu); free(Kh_cpu); free(Vh_cpu); free(dOh_cpu);
        free(dQ_cpu); free(dK_cpu); free(dV_cpu);
        CK(cudaFree(Q_d)); CK(cudaFree(K_d)); CK(cudaFree(V_d));
        CK(cudaFree(dO_d)); CK(cudaFree(dQ_d)); CK(cudaFree(dK_d));
        CK(cudaFree(dV_d)); CK(cudaFree(LSE_d)); CK(cudaFree(D_d));
    }
}

void bench_perf()
{
    printf("\n--- Performance (combined dQ + dKdV) ---\n");
    int configs[][4] = {
        {4, 1024, 128, 1},
        {4, 2048, 128, 1},
        {8, 2048, 128, 1},
        {4, 4096, 128, 1},
    };
    for (auto &c : configs)
    {
        int th = c[0], sl = c[1], hd = c[2], ca = c[3];
        size_t n_elems = (size_t)th * sl * hd;
        size_t lse_elems = (size_t)th * sl;

        __half *Q_d, *K_d, *V_d, *dO_d, *dQ_d, *dK_d, *dV_d;
        float *LSE_d, *D_d;
        CK(cudaMalloc(&Q_d, n_elems * 2));
        CK(cudaMalloc(&K_d, n_elems * 2));
        CK(cudaMalloc(&V_d, n_elems * 2));
        CK(cudaMalloc(&dO_d, n_elems * 2));
        CK(cudaMalloc(&dQ_d, n_elems * 2));
        CK(cudaMalloc(&dK_d, n_elems * 2));
        CK(cudaMalloc(&dV_d, n_elems * 2));
        CK(cudaMalloc(&LSE_d, lse_elems * 4));
        CK(cudaMalloc(&D_d, lse_elems * 4));
        CK(cudaMemset(Q_d, 0, n_elems * 2));
        CK(cudaMemset(K_d, 0, n_elems * 2));
        CK(cudaMemset(V_d, 0, n_elems * 2));
        CK(cudaMemset(dO_d, 0, n_elems * 2));
        CK(cudaMemset(LSE_d, 0, lse_elems * 4));
        CK(cudaMemset(D_d, 0, lse_elems * 4));

        // Warmup
        for (int i = 0; i < 3; i++)
        {
            CK(cudaMemset(dQ_d, 0, n_elems * 2));
            CK(cudaMemset(dK_d, 0, n_elems * 2));
            CK(cudaMemset(dV_d, 0, n_elems * 2));
            launch_v57_backward_dq(Q_d, K_d, V_d, dO_d, LSE_d, D_d, dQ_d, th, sl, hd, ca);
            launch_v57_backward_dkdv(Q_d, K_d, V_d, dO_d, LSE_d, D_d, dK_d, dV_d, th, sl, hd, ca);
        }
        CK(cudaDeviceSynchronize());

        cudaEvent_t s, e;
        cudaEventCreate(&s);
        cudaEventCreate(&e);
        int iters = 20;
        cudaEventRecord(s);
        for (int i = 0; i < iters; i++)
        {
            launch_v57_backward_dq(Q_d, K_d, V_d, dO_d, LSE_d, D_d, dQ_d, th, sl, hd, ca);
            launch_v57_backward_dkdv(Q_d, K_d, V_d, dO_d, LSE_d, D_d, dK_d, dV_d, th, sl, hd, ca);
        }
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, s, e);
        ms /= iters;

        // FA2 backward FLOPs: ~10 * th * sl^2 * hd (5 matmuls per pair, each 2 * sl * sl * hd halved by causal).
        // Standard convention: 4 * 2 * th * sl^2 * hd for non-causal, 5 * 2 * th * sl^2 * hd for full (forward).
        // For backward only (5 matmuls × causal): 5 * (2 * th * sl^2 * hd) * 0.5 = 5 * th * sl^2 * hd
        double flops = 5.0 * 2.0 * (double)th * (double)sl * (double)sl * (double)hd;
        if (ca) flops *= 0.5;  // causal halves work
        double tflops = flops / (ms / 1000.0) / 1e12;

        printf("  th=%d sl=%d hd=%d ca=%d  time=%.3f ms  perf=%.1f TFLOPS\n",
               th, sl, hd, ca, ms, tflops);

        cudaEventDestroy(s); cudaEventDestroy(e);
        CK(cudaFree(Q_d)); CK(cudaFree(K_d)); CK(cudaFree(V_d));
        CK(cudaFree(dO_d)); CK(cudaFree(dQ_d)); CK(cudaFree(dK_d));
        CK(cudaFree(dV_d)); CK(cudaFree(LSE_d)); CK(cudaFree(D_d));
    }
}

int main()
{
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    printf("=== FlashAttention v57 backward — 3-stage pipeline + SMEM recycle ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n\n", p.name, p.multiProcessorCount, clock_khz / 1000);
    srand(42);
    test_correctness();
    bench_perf();
    return 0;
}

#endif // BUILD_AS_LIB
