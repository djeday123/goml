// =====================================================================
//  fa_bwd_dk_baseline.cu — B3.2 minimal dK baseline FOR ptxas N МЕРЕ.
//
//  GOAL: получить честный ptxas N regs / spill / smem для dK MMA-цепочки
//        БЕЗ дополнительных рычагов (D = 0 placeholder, sync loads без
//        cp.async). От этого N считаем D-overhead (+6) и cp.async-Δ.
//
//  Q-LAYOUT TODO: draft использует option (b) — 4×LDS.U8+pack для dK MMA #2.
//        Bank analysis показал (b) structurally diseased (16-way conflict).
//        Финальный выбор (a) с smQ_T копией решается ПОСЛЕ ptxas замера
//        этого draft (Q-layout не меняет N сильно — только +pack/copy regs).
//
//  Корректность НЕ проверяется на этом шаге. Validation — на B3.2 final
//  body с готовой Q-layout decision.
//
//  Geometry: Bc=64, Br=64, Hd=128, 128 threads, 4 warps.
//  SMEM ~44.5 KB (smK 8 + smV 8 + smQ 8 + smdO 16 + smdST 4 + smL 0.25).
// =====================================================================

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "fa_bwd_common.cuh"

#define FA_DK_BC      64
#define FA_DK_BR      64
#define FA_DK_HD      128
#define FA_DK_THREADS 128

namespace fa_bwd_dk_baseline {

// PTX MMA m16n8k16 row.col.f32.f16.f16.f32 (FP16×FP16 → F32 acc).
// Used for MMA #1 (dP = dO · V^T).
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

// PTX MMA m16n8k32 row.col.f32.e4m3.e4m3.f32 (FP8×FP8 → F32 acc).
// Used for MMA #2 (dK_contrib = dS^T · Q).
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

// PTX cvt.rn.f16x2.e4m3x2 — uint16 (2 fp8) → uint32 (2 fp16).
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint16_t fp8x2) {
    uint32_t r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(fp8x2));
    return r;
}

// =====================================================================
// Minimal dK kernel — placeholder D=0, sync loads, draft Q reads via option (b).
// =====================================================================
__global__ void kernel_dk_baseline(
    const uint8_t * __restrict__ Q,     // FP8 e4m3 [bh, sl, hd]
    const uint8_t * __restrict__ K,     // FP8 e4m3
    const uint8_t * __restrict__ V,     // FP8 e4m3 NEW
    const __half  * __restrict__ dO_g,  // FP16
    const float   * __restrict__ L,     // FP32 [bh, sl]
    float         * __restrict__ dK,    // FP32 out [bh, sl, hd]
    int bh, int sl, int hd,
    int causal, int window,
    float scale)
{
    constexpr int Bc       = FA_DK_BC;
    constexpr int Br       = FA_DK_BR;
    constexpr int Hd       = FA_DK_HD;
    constexpr int NI_QK    = Bc / 8;      // 8 N-tiles for Q·K^T
    constexpr int NI_DP    = Bc / 8;      // 8 N-tiles for dP MMA #1 (same N=Bc)
    constexpr int NI_DK    = Hd / 8;      // 16 N-tiles for dK MMA #2 (N=hd)
    constexpr int KS_QK    = Hd / 32;     // 4 ks-batches FP8 m16n8k32
    constexpr int KS_DP    = Hd / 16;     // 8 ks-batches FP16 m16n8k16
    constexpr int KB_DK    = Br / 32;     // 2 k-batches FP8 m16n8k32 (Br reduction)

    const int tid    = threadIdx.x;
    const int wid    = tid >> 5;
    const int lane   = tid & 31;
    const int l_div4 = lane >> 2;
    const int l_mod4 = lane & 3;

    const int n_kt = (sl + Bc - 1) / Bc;
    const int b    = blockIdx.x / n_kt;
    const int kt   = blockIdx.x % n_kt;
    if (b >= bh) return;

    // ---- SMEM layout (~44.5 KB) ----
    extern __shared__ uint8_t smem_raw[];
    uint8_t *smK   = smem_raw;                                       // 8 KB
    uint8_t *smV   = smK   + Bc * Hd;                                // 8 KB NEW
    uint8_t *smQ   = smV   + Bc * Hd;                                // 8 KB
    __half  *smdO  = reinterpret_cast<__half*>(smQ + Br * Hd);       // 16 KB
    uint8_t *smdST = reinterpret_cast<uint8_t*>(smdO + Br * Hd);     // 4 KB
    float   *smL   = reinterpret_cast<float*>(smdST + Bc * Br);      // 256 B

    // ---- Warmup K + V (sync LDG+STS) ----
    {
        const uint8_t *Kb = K + b * sl * Hd;
        for (int e = tid; e < Bc * Hd; e += FA_DK_THREADS) {
            int j_local = e / Hd, d = e % Hd;
            int j_g = kt * Bc + j_local;
            smK[swz_byte(j_local, d)] = (j_g < sl) ? Kb[j_g * Hd + d] : (uint8_t)0;
        }
    }
    {
        const uint8_t *Vb = V + b * sl * Hd;
        for (int e = tid; e < Bc * Hd; e += FA_DK_THREADS) {
            int j_local = e / Hd, d = e % Hd;
            int j_g = kt * Bc + j_local;
            // smV row-major (NO swizzle) — natural col-major B reads for MMA #1.
            smV[j_local * Hd + d] = (j_g < sl) ? Vb[j_g * Hd + d] : (uint8_t)0;
        }
    }
    __syncthreads();

    // ---- dK accumulator (FP32, in registers) ----
    float dK_acc[NI_DK][4];
    #pragma unroll
    for (int ni = 0; ni < NI_DK; ++ni)
        #pragma unroll
        for (int s = 0; s < 4; ++s) dK_acc[ni][s] = 0.0f;

    const int n_qt = (sl + Br - 1) / Br;
    for (int qt = 0; qt < n_qt; ++qt) {
        const int qt_base = qt * Br;

        // ===== step A: sync loads Q, dO, L (D=0 placeholder) =====
        {
            const uint8_t *Qb = Q    + b * sl * Hd;
            const __half  *dB = dO_g + b * sl * Hd;
            for (int e = tid; e < Br * Hd; e += FA_DK_THREADS) {
                int i_local = e / Hd, d = e % Hd;
                int i_g = qt_base + i_local;
                // smQ row-major (no swizzle for simplicity).
                smQ[i_local * Hd + d] = (i_g < sl) ? Qb[i_g * Hd + d] : (uint8_t)0;
            }
            for (int e = tid; e < Br * Hd; e += FA_DK_THREADS) {
                int i_local = e / Hd, d = e % Hd;
                int i_g = qt_base + i_local;
                smdO[i_local * Hd + d] = (i_g < sl)
                    ? dB[i_g * Hd + d] : __float2half(0.0f);
            }
            if (tid < Br) {
                int i_g = qt_base + tid;
                smL[tid] = (i_g < sl) ? L[b * sl + i_g] : 0.0f;
            }
        }
        __syncthreads();

        // ===== step B: Q·K^T MMA → Sr (FP8 e4m3 m16n8k32 → f16 acc) =====
        // Same pattern as dV P1 (warp owns M-tile, full N=Bc/8, 4 ks-batches).
        uint32_t Qr[KS_QK][4];
        {
            int m_lo = wid * 16 + l_div4 + 0;
            int m_hi = wid * 16 + l_div4 + 8;
            int k0   = l_mod4 * 4;
            #pragma unroll
            for (int ks = 0; ks < KS_QK; ++ks) {
                int k_lo = ks * 32 + k0 + 0;
                int k_hi = ks * 32 + k0 + 16;
                Qr[ks][0] = *reinterpret_cast<uint32_t*>(&smQ[m_lo * Hd + k_lo]);
                Qr[ks][1] = *reinterpret_cast<uint32_t*>(&smQ[m_hi * Hd + k_lo]);
                Qr[ks][2] = *reinterpret_cast<uint32_t*>(&smQ[m_lo * Hd + k_hi]);
                Qr[ks][3] = *reinterpret_cast<uint32_t*>(&smQ[m_hi * Hd + k_hi]);
            }
        }

        uint32_t Sr[NI_QK][2];
        #pragma unroll
        for (int ni = 0; ni < NI_QK; ++ni) { Sr[ni][0] = 0u; Sr[ni][1] = 0u; }

        #pragma unroll
        for (int ks = 0; ks < KS_QK; ++ks) {
            int k_lo = ks * 32 + l_mod4 * 4 + 0;
            int k_hi = ks * 32 + l_mod4 * 4 + 16;
            #pragma unroll
            for (int ni = 0; ni < NI_QK; ++ni) {
                int n_K = ni * 8 + l_div4;
                uint32_t Kr0 = *reinterpret_cast<uint32_t*>(&smK[swz_byte(n_K, k_lo)]);
                uint32_t Kr1 = *reinterpret_cast<uint32_t*>(&smK[swz_byte(n_K, k_hi)]);
                mma_fp8_f16(Sr[ni][0], Sr[ni][1],
                            Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3],
                            Kr0, Kr1,
                            Sr[ni][0], Sr[ni][1]);
            }
        }

        // ===== step C: softmax → Pr (FP16 packed in regs, no smP write) =====
        const float L_lo = smL[wid * 16 + l_div4 + 0];
        const float L_hi = smL[wid * 16 + l_div4 + 8];
        const int i_g_lo = qt_base + wid * 16 + l_div4 + 0;
        const int i_g_hi = qt_base + wid * 16 + l_div4 + 8;
        const bool i_lo_oob = (i_g_lo >= sl);
        const bool i_hi_oob = (i_g_hi >= sl);

        uint32_t Pr[NI_QK][2];

        #pragma unroll
        for (int ni = 0; ni < NI_QK; ++ni) {
            __half2 s_lo_h2 = *reinterpret_cast<__half2*>(&Sr[ni][0]);
            __half2 s_hi_h2 = *reinterpret_cast<__half2*>(&Sr[ni][1]);
            float s_mlo_nlo = (float)__low2half (s_lo_h2);
            float s_mlo_nhi = (float)__high2half(s_lo_h2);
            float s_mhi_nlo = (float)__low2half (s_hi_h2);
            float s_mhi_nhi = (float)__high2half(s_hi_h2);

            int j_local_lo = ni * 8 + l_mod4 * 2 + 0;
            int j_local_hi = ni * 8 + l_mod4 * 2 + 1;
            int j_g_lo = kt * Bc + j_local_lo;
            int j_g_hi = kt * Bc + j_local_hi;
            bool j_lo_oob = (j_g_lo >= sl);
            bool j_hi_oob = (j_g_hi >= sl);

            auto mask = [&](int i_g, bool i_o, int j_g, bool j_o) -> bool {
                if (i_o || j_o)                           return true;
                if (causal && j_g > i_g)                  return true;
                if (window > 0 && j_g < i_g + 1 - window) return true;
                return false;
            };

            float p00 = mask(i_g_lo, i_lo_oob, j_g_lo, j_lo_oob) ? 0.0f
                       : __expf(scale * s_mlo_nlo - L_lo);
            float p01 = mask(i_g_lo, i_lo_oob, j_g_hi, j_hi_oob) ? 0.0f
                       : __expf(scale * s_mlo_nhi - L_lo);
            float p10 = mask(i_g_hi, i_hi_oob, j_g_lo, j_lo_oob) ? 0.0f
                       : __expf(scale * s_mhi_nlo - L_hi);
            float p11 = mask(i_g_hi, i_hi_oob, j_g_hi, j_hi_oob) ? 0.0f
                       : __expf(scale * s_mhi_nhi - L_hi);

            __half2 p_lo = __halves2half2(__float2half(p00), __float2half(p01));
            __half2 p_hi = __halves2half2(__float2half(p10), __float2half(p11));
            Pr[ni][0] = *reinterpret_cast<uint32_t*>(&p_lo);
            Pr[ni][1] = *reinterpret_cast<uint32_t*>(&p_hi);
        }

        // ===== step D: dP MMA #1 (FP16 m16n8k16 → F32 acc) =====
        //   A = dO row-major from smdO (single LDS.U32 per uint32 fragment)
        //   B = V col-major from smV row-major (FP8) → cast to FP16 via cvt
        //   Each B uint32 = pack of 2 fp16 cast from 2 fp8 = LDS.U16 + cvt.
        float dPr[NI_DP][4];
        #pragma unroll
        for (int ni = 0; ni < NI_DP; ++ni)
            #pragma unroll
            for (int s = 0; s < 4; ++s) dPr[ni][s] = 0.0f;

        #pragma unroll
        for (int ks = 0; ks < KS_DP; ++ks) {
            int m_lo = wid * 16 + l_div4 + 0;
            int m_hi = wid * 16 + l_div4 + 8;
            int k_lo = ks * 16 + l_mod4 * 2 + 0;
            int k_hi = ks * 16 + l_mod4 * 2 + 8;

            // A=dO: 4 uint32 per lane, FP16 row-major from smdO
            uint32_t A0 = *reinterpret_cast<uint32_t*>(&smdO[m_lo * Hd + k_lo]);
            uint32_t A1 = *reinterpret_cast<uint32_t*>(&smdO[m_hi * Hd + k_lo]);
            uint32_t A2 = *reinterpret_cast<uint32_t*>(&smdO[m_lo * Hd + k_hi]);
            uint32_t A3 = *reinterpret_cast<uint32_t*>(&smdO[m_hi * Hd + k_hi]);

            #pragma unroll
            for (int ni = 0; ni < NI_DP; ++ni) {
                int n = ni * 8 + l_div4;
                // B=V FP8 → FP16 cast inline
                uint16_t v0_u16 = *reinterpret_cast<uint16_t*>(&smV[n * Hd + k_lo]);
                uint16_t v1_u16 = *reinterpret_cast<uint16_t*>(&smV[n * Hd + k_hi]);
                uint32_t B0 = e4m3x2_to_f16x2(v0_u16);
                uint32_t B1 = e4m3x2_to_f16x2(v1_u16);

                mma_m16n8k16_f32(
                    dPr[ni][0], dPr[ni][1], dPr[ni][2], dPr[ni][3],
                    A0, A1, A2, A3, B0, B1,
                    dPr[ni][0], dPr[ni][1], dPr[ni][2], dPr[ni][3]);
            }
        }

        // ===== step E: dS = P · (dP - D), D=0 placeholder, quantize → smdST =====
        // Lane writes 4 cells per ni: smdST transposed [j_local, i_local].
        #pragma unroll
        for (int ni = 0; ni < NI_DP; ++ni) {
            __half2 p_lo_h2 = *reinterpret_cast<__half2*>(&Pr[ni][0]);
            __half2 p_hi_h2 = *reinterpret_cast<__half2*>(&Pr[ni][1]);
            float p_mlo_nlo = (float)__low2half (p_lo_h2);
            float p_mlo_nhi = (float)__high2half(p_lo_h2);
            float p_mhi_nlo = (float)__low2half (p_hi_h2);
            float p_mhi_nhi = (float)__high2half(p_hi_h2);

            // D=0 placeholder
            float dS_mlo_nlo = p_mlo_nlo * (dPr[ni][0] - 0.0f);
            float dS_mlo_nhi = p_mlo_nhi * (dPr[ni][1] - 0.0f);
            float dS_mhi_nlo = p_mhi_nlo * (dPr[ni][2] - 0.0f);
            float dS_mhi_nhi = p_mhi_nhi * (dPr[ni][3] - 0.0f);

            // Quantize FP32 → FP16 → e4m3 (two-step; PTX has direct f32→e4m3 too)
            __half2 ds_lo = __halves2half2(
                __float2half(dS_mlo_nlo), __float2half(dS_mlo_nhi));
            __half2 ds_hi = __halves2half2(
                __float2half(dS_mhi_nlo), __float2half(dS_mhi_nhi));
            uint32_t ds_lo_u32 = *reinterpret_cast<uint32_t*>(&ds_lo);
            uint32_t ds_hi_u32 = *reinterpret_cast<uint32_t*>(&ds_hi);
            uint16_t ds_lo_fp8 = fp16x2_to_e4m3x2(ds_lo_u32);
            uint16_t ds_hi_fp8 = fp16x2_to_e4m3x2(ds_hi_u32);

            uint8_t b00 = ds_lo_fp8 & 0xFF;
            uint8_t b01 = (ds_lo_fp8 >> 8) & 0xFF;
            uint8_t b10 = ds_hi_fp8 & 0xFF;
            uint8_t b11 = (ds_hi_fp8 >> 8) & 0xFF;

            int i_local_lo = wid * 16 + l_div4 + 0;
            int i_local_hi = wid * 16 + l_div4 + 8;
            int j_local_lo = ni * 8 + l_mod4 * 2 + 0;
            int j_local_hi = ni * 8 + l_mod4 * 2 + 1;

            // smdST[j_local, i_local] = dS[i, j]  (transposed)
            smdST[j_local_lo * Br + i_local_lo] = b00;
            smdST[j_local_hi * Br + i_local_lo] = b01;
            smdST[j_local_lo * Br + i_local_hi] = b10;
            smdST[j_local_hi * Br + i_local_hi] = b11;
        }

        __syncthreads();

        // ===== step G: dK MMA #2 (FP8 e4m3 m16n8k32 → F32 acc) =====
        //   A = dS^T row-major from smdST (single LDS.U32 per uint32 ✓)
        //   B = Q col-major. DRAFT Q-LAYOUT: option (b) — 4×LDS.U8+pack
        //   (TODO: switch to option (a) smQ_T copy after ptxas measurement.)
        #pragma unroll
        for (int kb = 0; kb < KB_DK; ++kb) {
            int m_lo = wid * 16 + l_div4 + 0;
            int m_hi = wid * 16 + l_div4 + 8;
            int k_q_lo = kb * 32 + l_mod4 * 4 + 0;
            int k_q_hi = kb * 32 + l_mod4 * 4 + 16;

            // A=dS^T: 4 uint32 per lane, FP8 row-major from smdST
            uint32_t A0 = *reinterpret_cast<uint32_t*>(&smdST[m_lo * Br + k_q_lo]);
            uint32_t A1 = *reinterpret_cast<uint32_t*>(&smdST[m_hi * Br + k_q_lo]);
            uint32_t A2 = *reinterpret_cast<uint32_t*>(&smdST[m_lo * Br + k_q_hi]);
            uint32_t A3 = *reinterpret_cast<uint32_t*>(&smdST[m_hi * Br + k_q_hi]);

            #pragma unroll
            for (int ni = 0; ni < NI_DK; ++ni) {
                int n_d = ni * 8 + l_div4;

                // B=Q col-major from smQ row-major: 4×LDS.U8 + pack per uint32
                uint8_t q0 = smQ[(k_q_lo + 0) * Hd + n_d];
                uint8_t q1 = smQ[(k_q_lo + 1) * Hd + n_d];
                uint8_t q2 = smQ[(k_q_lo + 2) * Hd + n_d];
                uint8_t q3 = smQ[(k_q_lo + 3) * Hd + n_d];
                uint32_t B0 = (uint32_t)q0 | ((uint32_t)q1 << 8)
                            | ((uint32_t)q2 << 16) | ((uint32_t)q3 << 24);

                uint8_t q4 = smQ[(k_q_hi + 0) * Hd + n_d];
                uint8_t q5 = smQ[(k_q_hi + 1) * Hd + n_d];
                uint8_t q6 = smQ[(k_q_hi + 2) * Hd + n_d];
                uint8_t q7 = smQ[(k_q_hi + 3) * Hd + n_d];
                uint32_t B1 = (uint32_t)q4 | ((uint32_t)q5 << 8)
                            | ((uint32_t)q6 << 16) | ((uint32_t)q7 << 24);

                mma_m16n8k32_e4m3_f32(
                    dK_acc[ni][0], dK_acc[ni][1], dK_acc[ni][2], dK_acc[ni][3],
                    A0, A1, A2, A3, B0, B1,
                    dK_acc[ni][0], dK_acc[ni][1], dK_acc[ni][2], dK_acc[ni][3]);
            }
        }

        __syncthreads();
    }

    // ---- Final dK write — scale * dK_acc → gmem ----
    {
        int j_local_lo = wid * 16 + l_div4 + 0;
        int j_local_hi = wid * 16 + l_div4 + 8;
        int j_g_lo = kt * Bc + j_local_lo;
        int j_g_hi = kt * Bc + j_local_hi;
        bool j_lo_ok = (j_g_lo < sl);
        bool j_hi_ok = (j_g_hi < sl);

        float *dKb = dK + b * sl * Hd;

        #pragma unroll
        for (int ni = 0; ni < NI_DK; ++ni) {
            int d_lo = ni * 8 + l_mod4 * 2 + 0;
            int d_hi = ni * 8 + l_mod4 * 2 + 1;
            if (j_lo_ok) {
                dKb[j_g_lo * Hd + d_lo] = dK_acc[ni][0] * scale;
                dKb[j_g_lo * Hd + d_hi] = dK_acc[ni][1] * scale;
            }
            if (j_hi_ok) {
                dKb[j_g_hi * Hd + d_lo] = dK_acc[ni][2] * scale;
                dKb[j_g_hi * Hd + d_hi] = dK_acc[ni][3] * scale;
            }
        }
    }
}

// Host launcher.
void launch(
    const uint8_t *Q, const uint8_t *K, const uint8_t *V,
    const __half *dO_g, const float *L, float *dK,
    int bh, int sl, int hd, int causal, int window,
    float scale, cudaStream_t stream)
{
    if (hd != FA_DK_HD) {
        fprintf(stderr, "fa_bwd_dk_baseline: hd=%d, expected %d\n", hd, FA_DK_HD);
        exit(1);
    }
    const int Bc   = FA_DK_BC;
    const int Br   = FA_DK_BR;
    const int n_kt = (sl + Bc - 1) / Bc;
    const int grid = bh * n_kt;
    const int smem_bytes =
        Bc * hd * sizeof(uint8_t)            // smK     8K
      + Bc * hd * sizeof(uint8_t)            // smV     8K
      + Br * hd * sizeof(uint8_t)            // smQ     8K
      + Br * hd * sizeof(__half)             // smdO   16K
      + Bc * Br * sizeof(uint8_t)            // smdST   4K
      + Br      * sizeof(float);             // smL  256B

    kernel_dk_baseline<<<grid, FA_DK_THREADS, smem_bytes, stream>>>(
        Q, K, V, dO_g, L, dK, bh, sl, hd, causal, window, scale);
}

} // namespace fa_bwd_dk_baseline
