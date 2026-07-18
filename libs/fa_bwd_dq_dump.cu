// =====================================================================
//  fa_bwd_dq.cu — B4.2 baseline dQ kernel (Pass 1 of backward).
//
//  Design (per Vugar's lock):
//   - Variant A: grid = bh * n_qt. Block owns Q-tile [Br, hd], accumulates
//     dQ_acc across all K via inner kt-loop. No atomicAdd across blocks.
//   - Three MMA chain per kt:
//       MMA-A: Q·K^T → S      (FP8 × FP8 → FP16-acc, same as dK MMA #1)
//       MMA-B: dP = dO · V^T  (FP16 × FP16 → FP32-acc, V cast e4m3→fp16)
//       MMA-C: dQ_acc += dS·K (FP8 × FP8 → FP32-acc, dS quantize e4m3)
//     dS = P · (dP - D) computed in regs between MMA-B and smdS scatter.
//   - dS lays out NATURAL [i][j] in smdS (NOT transposed — упрощение vs dK
//     where smdST was [j][i]). dS as A operand for MMA-C, K=j fits directly.
//   - K needs transpose [j][d] → [d][j] for MMA-C B operand (col-major K=j).
//     smK ↔ smK_T aliased (smK free after MMA-A; transpose overwrites with
//     barrier before write — same pattern as dK smQ ↔ smQ_T).
//   - stride-68 padding on BOTH smdS and smK_T (baseline f-lever applied;
//     stride-65 = followup probe candidate, NOT now).
//
//  SMEM layout (~45.25 KB, target 2 blocks/SM):
//    smK_aliased   8704 B  (max of K natural 8K and K_T padded 128*68)
//    smV           8192 B
//    smQ           8192 B  (Q-tile resident, loaded once at warmup)
//    smdO         16384 B  (dO-tile resident FP16, loaded once at warmup)
//    smdS          4352 B  (Br * SMDS_STRIDE = 64 * 68, NATURAL [i][j])
//    smL+smD        512 B
//  Total ≈ 46336 B = 45.25 KB.
//
//  Barriers per kt: 4
//    #1 after cp.async K/V → before MMA-A
//    #2 after smdS scatter + before smK→smK_T transpose write
//        (cross-warp sync: all warps done with smK reads from MMA-A)
//    #3 after transpose write → before MMA-C reads smK_T
//    #4 end of kt (smK_area free for next kt cp.async)
//
//  Correctness gate BEFORE wall: 11/11 forms + canary vs FP64-golden
//  (harness in fa_bwd_dq_test.cu, B4.1).
// =====================================================================

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "fa_bwd_common.cuh"

#define FA_DQ_BC         64
#define FA_DQ_BR         64
#define FA_DQ_HD         128
#define FA_DQ_THREADS    128
#define FA_DQ_KT_STRIDE  68    // mirror dK QT_STRIDE; stride-76 0-way write tested, no wall conversion (C-probe pattern)
#define FA_DQ_SMDS_STRIDE 68   // padding for smdS scatter writes (mirror dK SMDST_STRIDE/lever f)

namespace fa_bwd_dq_dump {

// =====================================================================
// MMA wrappers — mma_fp8_f16, mma_m16n8k16_f32, mma_m16n8k32_e4m3_f32.
// mma_fp8_f16 + fp16x2_to_e4m3x2 already in fa_bwd_common.cuh (Кирпич 5,6).
// Local definitions for the f32-acc variants (not in common.cuh).
// =====================================================================
__device__ __forceinline__ void mma_m16n8k16_f32(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__ void mma_m16n8k32_e4m3_f32(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// AA1: FP16-accumulator variant for MMA-C. C-fragment is 2 uint32 (4x f16 packed).
__device__ __forceinline__ void mma_m16n8k32_e4m3_f16(
    uint32_t &d0, uint32_t &d1,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t c0, uint32_t c1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(c0), "r"(c1));
}

__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint16_t fp8x2) {
    uint32_t r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(fp8x2));
    return r;
}

// =====================================================================
// dQ baseline kernel.
//   Inputs:  Q, K, V (FP8 e4m3), dO (FP16), L, D (FP32 vectors), scale.
//   Output:  dQ (FP32) [bh, sl, hd]
// =====================================================================
__global__ void kernel_dq_dump(
    const uint8_t * __restrict__ Q,
    const uint8_t * __restrict__ K,
    const uint8_t * __restrict__ V,
    const __half  * __restrict__ dO_g,
    const float   * __restrict__ L,
    const float   * __restrict__ D,
    float         * __restrict__ dQ,
    uint8_t       * __restrict__ dS_scratch,   // R1a-instrument: mirror smdS bytes to global [bh,sl,sl]
    int bh, int sl, int hd,
    int causal, int window,
    float scale)
{
    constexpr int Bc          = FA_DQ_BC;
    constexpr int Br          = FA_DQ_BR;
    constexpr int Hd          = FA_DQ_HD;
    constexpr int KT_STRIDE   = FA_DQ_KT_STRIDE;
    constexpr int SMDS_STRIDE = FA_DQ_SMDS_STRIDE;
    constexpr int NI_QK       = Bc / 8;            // 8 (MMA-A)
    constexpr int KS_QK       = Hd / 32;           // 4 (MMA-A K-batches m16n8k32)
    constexpr int NI_DP       = Bc / 8;            // 8 (MMA-B)
    constexpr int KS_DP       = Hd / 16;           // 8 (MMA-B K-batches m16n8k16)
    constexpr int NI_DQ       = Hd / 8;            // 16 (MMA-C N-tiles over Hd)
    constexpr int KB_DQ       = Bc / 32;           // 2 (MMA-C K-batches m16n8k32)

    const int tid    = threadIdx.x;
    const int wid    = tid >> 5;
    const int lane   = tid & 31;
    const int l_div4 = lane >> 2;
    const int l_mod4 = lane & 3;

    // Swizzle XOR masks (lane-constant across entire kernel)
    const int k_xor     = l_div4 << 4;    // byte-space XOR for smK/smV/smQ reads
    const int dO_xor_el = l_div4 << 3;    // element-space XOR for smdO reads (FP16 half of byte XOR)

    const int n_qt = (sl + Br - 1) / Br;
    const int b    = blockIdx.x / n_qt;
    const int qt   = blockIdx.x % n_qt;
    if (b >= bh) return;
    const int qt_base = qt * Br;

    // ---- SMEM layout P1a (~37.25 KB): smQ→Qr regs, smV↔smdS aliased (max=8192).
    //      smL/smD KEPT (no diet). KT_STRIDE=68. NO launch_bounds. ----
    constexpr int SMK_AREA_BYTES = (Bc * Hd > Hd * KT_STRIDE) ? Bc * Hd : Hd * KT_STRIDE;  // 8704
    constexpr int SMV_SMDS_BYTES = (Bc * Hd > Br * SMDS_STRIDE) ? Bc * Hd : Br * SMDS_STRIDE;  // 8192
    extern __shared__ uint8_t smem_raw[];
    uint8_t *smK_area = smem_raw;
    uint8_t *smV      = smK_area + SMK_AREA_BYTES;
    uint8_t *smdS     = smV;                                                 // ALIASED w/ smV
    __half  *smdO     = reinterpret_cast<__half*>(smV + SMV_SMDS_BYTES);
    float   *smL      = reinterpret_cast<float*>(smdO + Br * Hd);
    float   *smD      = smL + Br;

    // ============================================================
    // Warmup P1a: Q → Qr[KS_QK][4] direct-LDG regs (smQ REMOVED); dO/L/D staged.
    // ============================================================
    uint32_t Qr[KS_QK][4];
    {
        const uint8_t *Qb = Q + b * sl * Hd;
        int m_lo_ = wid * 16 + l_div4 + 0;
        int m_hi_ = wid * 16 + l_div4 + 8;
        int i_g_lo_ = qt_base + m_lo_;
        int i_g_hi_ = qt_base + m_hi_;
        bool lo_v = (i_g_lo_ < sl);
        bool hi_v = (i_g_hi_ < sl);
        #pragma unroll
        for (int ks = 0; ks < KS_QK; ++ks) {
            int k_lo = ks * 32 + l_mod4 * 4 + 0;
            int k_hi = ks * 32 + l_mod4 * 4 + 16;
            Qr[ks][0] = lo_v ? *reinterpret_cast<const uint32_t*>(&Qb[i_g_lo_ * Hd + k_lo]) : 0u;
            Qr[ks][1] = hi_v ? *reinterpret_cast<const uint32_t*>(&Qb[i_g_hi_ * Hd + k_lo]) : 0u;
            Qr[ks][2] = lo_v ? *reinterpret_cast<const uint32_t*>(&Qb[i_g_lo_ * Hd + k_hi]) : 0u;
            Qr[ks][3] = hi_v ? *reinterpret_cast<const uint32_t*>(&Qb[i_g_hi_ * Hd + k_hi]) : 0u;
        }
    }
    {
        constexpr int CHUNK = 16;
        const __half *dOb = dO_g + b * sl * Hd;
        constexpr int dO_bpr   = Hd * 2;
        constexpr int dO_cpr   = dO_bpr / CHUNK;
        constexpr int dO_total = Br * dO_cpr;
        uint8_t *smdO_b = reinterpret_cast<uint8_t*>(smdO);
        const uint8_t *dOb_b = reinterpret_cast<const uint8_t*>(dOb);
        for (int c = tid; c < dO_total; c += FA_DQ_THREADS) {
            int i_local = c / dO_cpr;
            int col_byte = (c % dO_cpr) * CHUNK;
            int i_g = qt_base + i_local;
            int dO_xor = (i_local & 7) << 4;    // smdO-swizzle (byte space, stride 256 → chunk 0..15, XOR keeps in-range)
            cpa16(smdO_b + i_local * dO_bpr + (col_byte ^ dO_xor),
                  dOb_b   + i_g * dO_bpr + col_byte,
                  (i_g < sl) ? CHUNK : 0);
        }
        if (tid < Br) {
            int i_g = qt_base + tid;
            smL[tid] = (i_g < sl) ? L[b * sl + i_g] : 0.0f;
            smD[tid] = (i_g < sl) ? D[b * sl + i_g] : 0.0f;
        }
        cpa_commit();
        cpa_wait<0>();
    }
    __syncthreads();   // BARRIER WARMUP

    // ---- AA1: dQ_acc as FP16 packed (2 u32 per ni tile = 4 f16). Saves 32 regs.
    uint32_t dQ_acc[NI_DQ][2];
    #pragma unroll
    for (int ni = 0; ni < NI_DQ; ++ni) { dQ_acc[ni][0] = 0u; dQ_acc[ni][1] = 0u; }

    // Per-Q-row L/D held in regs (reused across kt-loop)
    const float L_lo = smL[wid * 16 + l_div4 + 0];
    const float L_hi = smL[wid * 16 + l_div4 + 8];
    const float D_lo = smD[wid * 16 + l_div4 + 0];
    const float D_hi = smD[wid * 16 + l_div4 + 8];
    const int   i_g_lo = qt_base + wid * 16 + l_div4 + 0;
    const int   i_g_hi = qt_base + wid * 16 + l_div4 + 8;
    const bool  i_lo_oob = (i_g_lo >= sl);
    const bool  i_hi_oob = (i_g_hi >= sl);

    const int n_kt = (sl + Bc - 1) / Bc;
    // Causal-aware KV-skip (Pass 1 MIRROR of Pass 2): skip kt > qt (tile fully masked).
    // Tile fully masked iff min(j_tile)=kt*Bc > max(i_tile)=qt*Br + Br - 1 → kt > qt for Br=Bc.
    // Loop iterates kt = 0..qt (inclusive of diagonal). Non-causal: kt_end=n_kt, no change.
    const int kt_end = causal ? (qt + 1) : n_kt;
    for (int kt = 0; kt < kt_end; ++kt) {
        const int kt_base = kt * Bc;

        // ===== step A: cp.async K (XOR-swizzled dest) and V =====
        // smK-swizzle probe: cp.async writes K with dest = j_local*Hd + (col_byte XOR ((j_local & 7) << 4))
        // XOR mask (j_local & 7) is lane-constant (j_local = c/8, c iterates by 128, so (j_local & 7)=(tid>>3)&7).
        // XOR-16 preserves 16-byte alignment ((row&7)<<4 ∈ {0,16,...,112} — multiples of 16). smV unchanged in this step.
        {
            const uint8_t *Kb = K + b * sl * Hd;
            const uint8_t *Vb = V + b * sl * Hd;
            constexpr int CHUNK = 16;
            constexpr int chunks_per_row = Hd / CHUNK;       // 8
            constexpr int total = Bc * chunks_per_row;       // 512
            for (int c = tid; c < total; c += FA_DQ_THREADS) {
                int j_local = c / chunks_per_row;
                int col_byte = (c % chunks_per_row) * CHUNK;
                int j_g = kt_base + j_local;
                int k_xor = (j_local & 7) << 4;              // smK+smV-swizzle: XOR chunk index with (row & 7)
                cpa16(&smK_area[j_local * Hd + (col_byte ^ k_xor)],
                      &Kb[j_g * Hd + col_byte],
                      (j_g < sl) ? CHUNK : 0);
                cpa16(&smV[j_local * Hd + (col_byte ^ k_xor)],
                      &Vb[j_g * Hd + col_byte],
                      (j_g < sl) ? CHUNK : 0);
            }
            cpa_commit();
            cpa_wait<0>();
        }
        __syncthreads();    // BARRIER #1

        // ===== step B: MMA-A Q·K^T → Sr (FP8 × FP8 → FP16 acc) =====
        // K2-probe: (b)-revert — Kr_cache REMOVED, phase 1.5 re-reads swizzled smK all-warps
        // smK/smV/smQ swizzle: XOR mask k_xor = l_div4 << 4 (byte-space, hoisted to kt-loop scope)
        uint32_t Sr[NI_QK][2];
        #pragma unroll
        for (int ni = 0; ni < NI_QK; ++ni) { Sr[ni][0] = 0u; Sr[ni][1] = 0u; }

        #pragma unroll
        for (int ks = 0; ks < KS_QK; ++ks) {
            int m_lo = wid * 16 + l_div4 + 0;
            int m_hi = wid * 16 + l_div4 + 8;
            int k_lo = ks * 32 + l_mod4 * 4 + 0;
            int k_hi = ks * 32 + l_mod4 * 4 + 16;

            // P1a: Q from persistent Qr registers (smQ REMOVED)
            uint32_t Qr0 = Qr[ks][0];
            uint32_t Qr1 = Qr[ks][1];
            uint32_t Qr2 = Qr[ks][2];
            uint32_t Qr3 = Qr[ks][3];

            #pragma unroll
            for (int ni = 0; ni < NI_QK; ++ni) {
                int n_K = ni * 8 + l_div4;
                uint32_t Kr_lo = *reinterpret_cast<uint32_t*>(&smK_area[n_K * Hd + (k_lo ^ k_xor)]);
                uint32_t Kr_hi = *reinterpret_cast<uint32_t*>(&smK_area[n_K * Hd + (k_hi ^ k_xor)]);
                mma_fp8_f16(Sr[ni][0], Sr[ni][1],
                            Qr0, Qr1, Qr2, Qr3,
                            Kr_lo, Kr_hi,
                            Sr[ni][0], Sr[ni][1]);
            }
        }

        // ===== step C: softmax Sr → Pr (FP16 packed) =====
        uint32_t Pr[NI_QK][2];
        // L3-probe: unroll-and-jam x2 — process ni_a and ni_b as pair, interleave issue.
        // Arithmetic per-tile UNCHANGED (bit-exact preserved). Only compiler issue-schedule differs.
        auto mask = [&](int i_g, bool i_o, int j_g, bool j_o) -> bool {
            if (i_o || j_o)                           return true;
            if (causal && j_g > i_g)                  return true;
            if (window > 0 && j_g < i_g + 1 - window) return true;
            return false;
        };
        #pragma unroll
        for (int np = 0; np < NI_QK / 2; ++np) {
            const int ni_a = 2 * np;
            const int ni_b = 2 * np + 1;

            // (1) Load Sr for both tiles
            __half2 sa_lo_h2 = *reinterpret_cast<__half2*>(&Sr[ni_a][0]);
            __half2 sa_hi_h2 = *reinterpret_cast<__half2*>(&Sr[ni_a][1]);
            __half2 sb_lo_h2 = *reinterpret_cast<__half2*>(&Sr[ni_b][0]);
            __half2 sb_hi_h2 = *reinterpret_cast<__half2*>(&Sr[ni_b][1]);

            // (2) F2FP interleaved
            float a_mlo_nlo = (float)__low2half (sa_lo_h2);
            float b_mlo_nlo = (float)__low2half (sb_lo_h2);
            float a_mlo_nhi = (float)__high2half(sa_lo_h2);
            float b_mlo_nhi = (float)__high2half(sb_lo_h2);
            float a_mhi_nlo = (float)__low2half (sa_hi_h2);
            float b_mhi_nlo = (float)__low2half (sb_hi_h2);
            float a_mhi_nhi = (float)__high2half(sa_hi_h2);
            float b_mhi_nhi = (float)__high2half(sb_hi_h2);

            // (3) j_local ranges per tile
            int ja_lo   = ni_a * 8 + l_mod4 * 2 + 0;
            int ja_hi   = ni_a * 8 + l_mod4 * 2 + 1;
            int jb_lo   = ni_b * 8 + l_mod4 * 2 + 0;
            int jb_hi   = ni_b * 8 + l_mod4 * 2 + 1;
            int ja_g_lo = kt_base + ja_lo;
            int ja_g_hi = kt_base + ja_hi;
            int jb_g_lo = kt_base + jb_lo;
            int jb_g_hi = kt_base + jb_hi;
            bool ja_lo_o = (ja_g_lo >= sl), ja_hi_o = (ja_g_hi >= sl);
            bool jb_lo_o = (jb_g_lo >= sl), jb_hi_o = (jb_g_hi >= sl);

            // (4) expf both tiles interleaved (schedule-only, per-tile arithmetic unchanged)
            float pa00 = mask(i_g_lo, i_lo_oob, ja_g_lo, ja_lo_o) ? 0.0f : __expf(scale * a_mlo_nlo - L_lo);
            float pb00 = mask(i_g_lo, i_lo_oob, jb_g_lo, jb_lo_o) ? 0.0f : __expf(scale * b_mlo_nlo - L_lo);
            float pa01 = mask(i_g_lo, i_lo_oob, ja_g_hi, ja_hi_o) ? 0.0f : __expf(scale * a_mlo_nhi - L_lo);
            float pb01 = mask(i_g_lo, i_lo_oob, jb_g_hi, jb_hi_o) ? 0.0f : __expf(scale * b_mlo_nhi - L_lo);
            float pa10 = mask(i_g_hi, i_hi_oob, ja_g_lo, ja_lo_o) ? 0.0f : __expf(scale * a_mhi_nlo - L_hi);
            float pb10 = mask(i_g_hi, i_hi_oob, jb_g_lo, jb_lo_o) ? 0.0f : __expf(scale * b_mhi_nlo - L_hi);
            float pa11 = mask(i_g_hi, i_hi_oob, ja_g_hi, ja_hi_o) ? 0.0f : __expf(scale * a_mhi_nhi - L_hi);
            float pb11 = mask(i_g_hi, i_hi_oob, jb_g_hi, jb_hi_o) ? 0.0f : __expf(scale * b_mhi_nhi - L_hi);

            // (5) Pack both tiles interleaved
            __half2 pa_lo = __halves2half2(__float2half(pa00), __float2half(pa01));
            __half2 pb_lo = __halves2half2(__float2half(pb00), __float2half(pb01));
            __half2 pa_hi = __halves2half2(__float2half(pa10), __float2half(pa11));
            __half2 pb_hi = __halves2half2(__float2half(pb10), __float2half(pb11));
            Pr[ni_a][0] = *reinterpret_cast<uint32_t*>(&pa_lo);
            Pr[ni_b][0] = *reinterpret_cast<uint32_t*>(&pb_lo);
            Pr[ni_a][1] = *reinterpret_cast<uint32_t*>(&pa_hi);
            Pr[ni_b][1] = *reinterpret_cast<uint32_t*>(&pb_hi);
        }

        // ===== step D: MMA-B dP = dO·V^T → dPr (FP16 × FP16 → FP32 acc) =====
        float dPr[NI_DP][4];
        #pragma unroll
        for (int ni = 0; ni < NI_DP; ++ni)
            #pragma unroll
            for (int s = 0; s < 4; ++s) dPr[ni][s] = 0.0f;

        // smdO-swizzle XOR (element-space) hoisted from kernel-scope (same as k_xor >> 1)
        #pragma unroll
        for (int ks = 0; ks < KS_DP; ++ks) {
            int m_lo = wid * 16 + l_div4 + 0;
            int m_hi = wid * 16 + l_div4 + 8;
            int k_lo = ks * 16 + l_mod4 * 2 + 0;
            int k_hi = ks * 16 + l_mod4 * 2 + 8;

            uint32_t A0 = *reinterpret_cast<uint32_t*>(&smdO[m_lo * Hd + (k_lo ^ dO_xor_el)]);
            uint32_t A1 = *reinterpret_cast<uint32_t*>(&smdO[m_hi * Hd + (k_lo ^ dO_xor_el)]);
            uint32_t A2 = *reinterpret_cast<uint32_t*>(&smdO[m_lo * Hd + (k_hi ^ dO_xor_el)]);
            uint32_t A3 = *reinterpret_cast<uint32_t*>(&smdO[m_hi * Hd + (k_hi ^ dO_xor_el)]);

            #pragma unroll
            for (int ni = 0; ni < NI_DP; ++ni) {
                int n = ni * 8 + l_div4;
                // smV-swizzle: (n & 7) = l_div4 → XOR = k_xor (already hoisted for smK MMA-A read)
                uint16_t v0_u16 = *reinterpret_cast<uint16_t*>(&smV[n * Hd + (k_lo ^ k_xor)]);
                uint16_t v1_u16 = *reinterpret_cast<uint16_t*>(&smV[n * Hd + (k_hi ^ k_xor)]);
                uint32_t B0 = e4m3x2_to_f16x2(v0_u16);
                uint32_t B1 = e4m3x2_to_f16x2(v1_u16);

                mma_m16n8k16_f32(
                    dPr[ni][0], dPr[ni][1], dPr[ni][2], dPr[ni][3],
                    A0, A1, A2, A3, B0, B1,
                    dPr[ni][0], dPr[ni][1], dPr[ni][2], dPr[ni][3]);
            }
        }

        __syncthreads();    // P1a: BARRIER smV↔smdS aliased — MMA-B V-reads must finish before smdS writes

        // ===== step E+F: dS = P*(dP-D) quantize → smdS NATURAL [i][j] =====
        // L3-probe: unroll-and-jam x2 (schedule-only, per-tile arithmetic unchanged)
        #pragma unroll
        for (int np = 0; np < NI_DP / 2; ++np) {
            const int ni_a = 2 * np;
            const int ni_b = 2 * np + 1;

            __half2 pa_lo_h2 = *reinterpret_cast<__half2*>(&Pr[ni_a][0]);
            __half2 pa_hi_h2 = *reinterpret_cast<__half2*>(&Pr[ni_a][1]);
            __half2 pb_lo_h2 = *reinterpret_cast<__half2*>(&Pr[ni_b][0]);
            __half2 pb_hi_h2 = *reinterpret_cast<__half2*>(&Pr[ni_b][1]);

            float pa_mlo_nlo = (float)__low2half (pa_lo_h2);
            float pb_mlo_nlo = (float)__low2half (pb_lo_h2);
            float pa_mlo_nhi = (float)__high2half(pa_lo_h2);
            float pb_mlo_nhi = (float)__high2half(pb_lo_h2);
            float pa_mhi_nlo = (float)__low2half (pa_hi_h2);
            float pb_mhi_nlo = (float)__low2half (pb_hi_h2);
            float pa_mhi_nhi = (float)__high2half(pa_hi_h2);
            float pb_mhi_nhi = (float)__high2half(pb_hi_h2);

            float dSa_mlo_nlo = pa_mlo_nlo * (dPr[ni_a][0] - D_lo);
            float dSb_mlo_nlo = pb_mlo_nlo * (dPr[ni_b][0] - D_lo);
            float dSa_mlo_nhi = pa_mlo_nhi * (dPr[ni_a][1] - D_lo);
            float dSb_mlo_nhi = pb_mlo_nhi * (dPr[ni_b][1] - D_lo);
            float dSa_mhi_nlo = pa_mhi_nlo * (dPr[ni_a][2] - D_hi);
            float dSb_mhi_nlo = pb_mhi_nlo * (dPr[ni_b][2] - D_hi);
            float dSa_mhi_nhi = pa_mhi_nhi * (dPr[ni_a][3] - D_hi);
            float dSb_mhi_nhi = pb_mhi_nhi * (dPr[ni_b][3] - D_hi);

            __half2 dsa_lo = __halves2half2(__float2half(dSa_mlo_nlo), __float2half(dSa_mlo_nhi));
            __half2 dsb_lo = __halves2half2(__float2half(dSb_mlo_nlo), __float2half(dSb_mlo_nhi));
            __half2 dsa_hi = __halves2half2(__float2half(dSa_mhi_nlo), __float2half(dSa_mhi_nhi));
            __half2 dsb_hi = __halves2half2(__float2half(dSb_mhi_nlo), __float2half(dSb_mhi_nhi));

            uint32_t dsa_lo_u32 = *reinterpret_cast<uint32_t*>(&dsa_lo);
            uint32_t dsb_lo_u32 = *reinterpret_cast<uint32_t*>(&dsb_lo);
            uint32_t dsa_hi_u32 = *reinterpret_cast<uint32_t*>(&dsa_hi);
            uint32_t dsb_hi_u32 = *reinterpret_cast<uint32_t*>(&dsb_hi);

            uint16_t dsa_lo_fp8 = fp16x2_to_e4m3x2(dsa_lo_u32);
            uint16_t dsb_lo_fp8 = fp16x2_to_e4m3x2(dsb_lo_u32);
            uint16_t dsa_hi_fp8 = fp16x2_to_e4m3x2(dsa_hi_u32);
            uint16_t dsb_hi_fp8 = fp16x2_to_e4m3x2(dsb_hi_u32);

            int i_local_lo = wid * 16 + l_div4 + 0;
            int i_local_hi = wid * 16 + l_div4 + 8;
            int ja_lo = ni_a * 8 + l_mod4 * 2 + 0;
            int ja_hi = ni_a * 8 + l_mod4 * 2 + 1;
            int jb_lo = ni_b * 8 + l_mod4 * 2 + 0;
            int jb_hi = ni_b * 8 + l_mod4 * 2 + 1;

            // Interleaved STS for both tiles
            uint8_t b_lo_ja = dsa_lo_fp8 & 0xFF;
            uint8_t b_lo_jb = dsb_lo_fp8 & 0xFF;
            uint8_t b_lo_jah = (dsa_lo_fp8 >> 8) & 0xFF;
            uint8_t b_lo_jbh = (dsb_lo_fp8 >> 8) & 0xFF;
            uint8_t b_hi_ja = dsa_hi_fp8 & 0xFF;
            uint8_t b_hi_jb = dsb_hi_fp8 & 0xFF;
            uint8_t b_hi_jah = (dsa_hi_fp8 >> 8) & 0xFF;
            uint8_t b_hi_jbh = (dsb_hi_fp8 >> 8) & 0xFF;

            smdS[i_local_lo * SMDS_STRIDE + ja_lo] = b_lo_ja;
            smdS[i_local_lo * SMDS_STRIDE + jb_lo] = b_lo_jb;
            smdS[i_local_lo * SMDS_STRIDE + ja_hi] = b_lo_jah;
            smdS[i_local_lo * SMDS_STRIDE + jb_hi] = b_lo_jbh;
            smdS[i_local_hi * SMDS_STRIDE + ja_lo] = b_hi_ja;
            smdS[i_local_hi * SMDS_STRIDE + jb_lo] = b_hi_jb;
            smdS[i_local_hi * SMDS_STRIDE + ja_hi] = b_hi_jah;
            smdS[i_local_hi * SMDS_STRIDE + jb_hi] = b_hi_jbh;

            // R1a-instrument: mirror to dS_scratch [b, i, j] natural, with OOB guards
            uint8_t *dS_base = dS_scratch + (size_t)b * sl * sl;
            int i_g_lo_ = qt_base + i_local_lo;
            int i_g_hi_ = qt_base + i_local_hi;
            int ja_g_lo_ = kt_base + ja_lo;
            int ja_g_hi_ = kt_base + ja_hi;
            int jb_g_lo_ = kt_base + jb_lo;
            int jb_g_hi_ = kt_base + jb_hi;
            if (i_g_lo_ < sl) {
                if (ja_g_lo_ < sl) dS_base[(size_t)i_g_lo_ * sl + ja_g_lo_] = b_lo_ja;
                if (jb_g_lo_ < sl) dS_base[(size_t)i_g_lo_ * sl + jb_g_lo_] = b_lo_jb;
                if (ja_g_hi_ < sl) dS_base[(size_t)i_g_lo_ * sl + ja_g_hi_] = b_lo_jah;
                if (jb_g_hi_ < sl) dS_base[(size_t)i_g_lo_ * sl + jb_g_hi_] = b_lo_jbh;
            }
            if (i_g_hi_ < sl) {
                if (ja_g_lo_ < sl) dS_base[(size_t)i_g_hi_ * sl + ja_g_lo_] = b_hi_ja;
                if (jb_g_lo_ < sl) dS_base[(size_t)i_g_hi_ * sl + jb_g_lo_] = b_hi_jb;
                if (ja_g_hi_ < sl) dS_base[(size_t)i_g_hi_ * sl + ja_g_hi_] = b_hi_jah;
                if (jb_g_hi_ < sl) dS_base[(size_t)i_g_hi_ * sl + jb_g_hi_] = b_hi_jbh;
            }
        }
        // L3-probe: legacy per-ni STS block removed (replaced by paired STS above)

        // ===== Phase 1.5 K2-probe: all-warps transpose, per-warp ks=wid distribution =====
        // Each warp (wid ∈ [0..3]) handles ks=wid. Reads swizzled smK inline (XOR k_xor).
        // Two-phase to avoid RAW hazard on aliased smK_area:
        //   phase 1: all-warps read smK into local temps (kr_lo[NI_QK], kr_hi[NI_QK])
        //   BARRIER: everyone read before anyone writes
        //   phase 2: all-warps write smK_T from local temps
        __syncthreads();    // BARRIER #2: all warps past MMA-A K-reads + smdS visibility
        {
            const int ks = wid;                                // per-warp ks
            const int k_lo   = ks * 32 + l_mod4 * 4 + 0;
            const int k_hi   = ks * 32 + l_mod4 * 4 + 16;
            const int k_lo_b = k_lo;
            const int k_hi_b = k_hi;
            uint32_t kr_lo[NI_QK], kr_hi[NI_QK];
            // Read phase (swizzle-aware LDS on swizzled smK natural)
            #pragma unroll
            for (int ni = 0; ni < NI_QK; ++ni) {
                int n_K = ni * 8 + l_div4;
                kr_lo[ni] = *reinterpret_cast<uint32_t*>(&smK_area[n_K * Hd + (k_lo ^ k_xor)]);
                kr_hi[ni] = *reinterpret_cast<uint32_t*>(&smK_area[n_K * Hd + (k_hi ^ k_xor)]);
            }
            __syncthreads();    // BARRIER #2.5: RAW hazard — all reads done before any writes (aliased smK↔smK_T)
            // Write phase (natural STS to smK_T at KT_STRIDE=68)
            #pragma unroll
            for (int ni = 0; ni < NI_QK; ++ni) {
                int n_K = ni * 8 + l_div4;
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    smK_area[(k_lo_b + b) * KT_STRIDE + n_K] = (kr_lo[ni] >> (b * 8)) & 0xFF;
                    smK_area[(k_hi_b + b) * KT_STRIDE + n_K] = (kr_hi[ni] >> (b * 8)) & 0xFF;
                }
            }
        }
        __syncthreads();    // BARRIER #3: smK_T ready for MMA-C (all warps see writes)

        // ===== step F: MMA-C dS·K_T → dQ_acc (FP8 × FP8 → FP32 acc) =====
        // A = dS NATURAL row-major [M=i=Br][K=j=Bc] from smdS (stride SMDS_STRIDE)
        // B = K_T col-major K=j, N=d stored [N=d][K=j] = transposed K (stride KT_STRIDE)
        #pragma unroll
        for (int kb = 0; kb < KB_DQ; ++kb) {
            int m_lo = wid * 16 + l_div4 + 0;
            int m_hi = wid * 16 + l_div4 + 8;
            int k_j_lo = kb * 32 + l_mod4 * 4 + 0;
            int k_j_hi = kb * 32 + l_mod4 * 4 + 16;

            uint32_t A0 = *reinterpret_cast<uint32_t*>(&smdS[m_lo * SMDS_STRIDE + k_j_lo]);
            uint32_t A1 = *reinterpret_cast<uint32_t*>(&smdS[m_hi * SMDS_STRIDE + k_j_lo]);
            uint32_t A2 = *reinterpret_cast<uint32_t*>(&smdS[m_lo * SMDS_STRIDE + k_j_hi]);
            uint32_t A3 = *reinterpret_cast<uint32_t*>(&smdS[m_hi * SMDS_STRIDE + k_j_hi]);

            #pragma unroll
            for (int ni = 0; ni < NI_DQ; ++ni) {
                int n_d = ni * 8 + l_div4;
                uint32_t B0 = *reinterpret_cast<uint32_t*>(
                    &smK_area[n_d * KT_STRIDE + k_j_lo]);
                uint32_t B1 = *reinterpret_cast<uint32_t*>(
                    &smK_area[n_d * KT_STRIDE + k_j_hi]);

                mma_m16n8k32_e4m3_f16(
                    dQ_acc[ni][0], dQ_acc[ni][1],
                    A0, A1, A2, A3, B0, B1,
                    dQ_acc[ni][0], dQ_acc[ni][1]);
            }
        }

        __syncthreads();    // BARRIER #4: end of kt (smK_area free for next cp.async)
    }

    // ---- Final write dQ_acc * scale → gmem ----
    {
        int i_local_lo = wid * 16 + l_div4 + 0;
        int i_local_hi = wid * 16 + l_div4 + 8;
        int i_g_lo_out = qt_base + i_local_lo;
        int i_g_hi_out = qt_base + i_local_hi;
        bool i_lo_ok = (i_g_lo_out < sl);
        bool i_hi_ok = (i_g_hi_out < sl);

        float *dQb = dQ + b * sl * Hd;
        // AA1: unpack packed f16 dQ_acc → fp32 → scale → store.
        //   Layout mirrors f32-acc: dQ_acc[ni][0]=lo lane {d_lo@m_lo, d_hi@m_lo},
        //                           dQ_acc[ni][1]=hi lane {d_lo@m_hi, d_hi@m_hi}.
        #pragma unroll
        for (int ni = 0; ni < NI_DQ; ++ni) {
            int d_lo = ni * 8 + l_mod4 * 2 + 0;
            int d_hi = ni * 8 + l_mod4 * 2 + 1;
            __half2 lo_h2 = *reinterpret_cast<__half2*>(&dQ_acc[ni][0]);
            __half2 hi_h2 = *reinterpret_cast<__half2*>(&dQ_acc[ni][1]);
            float lo_d_lo = (float)__low2half (lo_h2);
            float lo_d_hi = (float)__high2half(lo_h2);
            float hi_d_lo = (float)__low2half (hi_h2);
            float hi_d_hi = (float)__high2half(hi_h2);
            if (i_lo_ok) {
                dQb[i_g_lo_out * Hd + d_lo] = lo_d_lo * scale;
                dQb[i_g_lo_out * Hd + d_hi] = lo_d_hi * scale;
            }
            if (i_hi_ok) {
                dQb[i_g_hi_out * Hd + d_lo] = hi_d_lo * scale;
                dQb[i_g_hi_out * Hd + d_hi] = hi_d_hi * scale;
            }
        }
    }
}

// =====================================================================
// Host launcher.
// =====================================================================
void launch_dq_dump(
    const uint8_t *Q, const uint8_t *K, const uint8_t *V,
    const __half *dO_g, const float *L, const float *D,
    float *dQ,
    uint8_t *dS_scratch,
    int bh, int sl, int hd,
    int causal, int window,
    float scale, cudaStream_t stream)
{
    if (hd != FA_DQ_HD) {
        fprintf(stderr, "fa_bwd_dq_dump: hd=%d, expected %d\n", hd, FA_DQ_HD);
        exit(1);
    }
    const int Bc   = FA_DQ_BC;
    const int Br   = FA_DQ_BR;
    const int n_qt = (sl + Br - 1) / Br;
    const int grid = bh * n_qt;
    constexpr int SMK_AREA = (FA_DQ_BC * FA_DQ_HD > FA_DQ_HD * FA_DQ_KT_STRIDE)
                             ? FA_DQ_BC * FA_DQ_HD
                             : FA_DQ_HD * FA_DQ_KT_STRIDE;
    constexpr int SMV_SMDS = (FA_DQ_BC * FA_DQ_HD > FA_DQ_BR * FA_DQ_SMDS_STRIDE)
                             ? FA_DQ_BC * FA_DQ_HD
                             : FA_DQ_BR * FA_DQ_SMDS_STRIDE;
    const int smem_bytes =
        SMK_AREA
      + SMV_SMDS
      + Br * hd * sizeof(__half)
      + 2 * Br * sizeof(float);

    cudaFuncSetAttribute(kernel_dq_dump,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    kernel_dq_dump<<<grid, FA_DQ_THREADS, smem_bytes, stream>>>(
        Q, K, V, dO_g, L, D, dQ, dS_scratch, bh, sl, hd, causal, window, scale);
}

} // namespace fa_bwd_dq_dump
