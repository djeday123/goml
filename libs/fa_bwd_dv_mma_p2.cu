// =====================================================================
//  fa_bwd_dv_mma_p2.cu — B2.2 P2 axis: same code as P1, force 1 block/SM.
//
//  Goal: isolate OCCUPANCY cost. Kernel body IDENTICAL to P1 (cp.async loads
//  same; no double buffer; ptxas regs must match P1 = 166 / 0 spill / 0 stack
//  / 1 barrier — verified before measurement).
//
//  Force 1 block/SM via dummy dynamic SMEM padding in launcher (~12 KB
//  reserve raises per-block SMEM above 50 KB → SM config 100 KB fits only
//  one block). NOT via __launch_bounds__ — that would re-spill registers
//  and mix occupancy cost with reg-budget shift.
//
//  Expected: P2 − P1 likely NEGATIVE (1 block, eligible warps drop, mio
//  saturation rises). It measures the COST of occupancy reduction.
//
//  Correctness: 11 forms + canary, FP8 floor unchanged (shift = race).
// =====================================================================

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "fa_bwd_common.cuh"   // cpa16, cpa_commit, cpa_wait, mma_fp8_f16, swz_byte

#ifndef FA_DV_MMA_BC
#define FA_DV_MMA_BC 64
#endif
#ifndef FA_DV_MMA_BR
#define FA_DV_MMA_BR 64
#endif
#ifndef FA_DV_MMA_HD
#define FA_DV_MMA_HD 128
#endif
#ifndef FA_DV_MMA_THREADS
#define FA_DV_MMA_THREADS 128
#endif

namespace fa_bwd_dv_mma_p2 {

// =====================================================================
// PTX m16n8k16.row.col.f32.f16.f16.f32 (FP16 MMA, FP32 accumulator).
// Identical to B2.1; copied verbatim to keep kernel self-contained.
// =====================================================================
__device__ __forceinline__ void mma_m16n8k16_f16_f32(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// =====================================================================
// Kernel body — P1 variant (cp.async loads, single buffer).
// =====================================================================
__global__ void kernel_dv_mma_p2(
    const uint8_t * __restrict__ Q,
    const uint8_t * __restrict__ K,
    const __half  * __restrict__ dO_g,
    const float   * __restrict__ L,
    float         * __restrict__ dV,
    int bh, int sl, int hd,
    int causal, int window,
    float scale)
{
    constexpr int Bc      = FA_DV_MMA_BC;
    constexpr int Br      = FA_DV_MMA_BR;
    constexpr int Hd      = FA_DV_MMA_HD;
    constexpr int NI_QK   = Bc / 8;
    constexpr int NI_DV   = Hd / 8;
    constexpr int KS_QK   = Hd / 32;
    constexpr int KB_DV   = Br / 16;

    const int tid    = threadIdx.x;
    const int wid    = tid >> 5;
    const int lane   = tid & 31;
    const int l_div4 = lane >> 2;
    const int l_mod4 = lane & 3;

    const int n_kt = (sl + Bc - 1) / Bc;
    const int b    = blockIdx.x / n_kt;
    const int kt   = blockIdx.x % n_kt;
    if (b >= bh) return;

    extern __shared__ uint8_t smem_raw[];
    uint8_t *smK  = smem_raw;
    uint8_t *smQ  = smK  + Bc * Hd;
    __half  *smdO = reinterpret_cast<__half*>(smQ + Bc * Hd);
    __half  *smPT = reinterpret_cast<__half*>(
        reinterpret_cast<uint8_t*>(smdO) + Br * Hd * 2);
    float   *smL  = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(smPT) + Bc * Br * 2);

    // ---- Warmup K-tile load — cp.async ----
    //   K is FP8 [Bc=64, Hd=128] = 8192 bytes, 16-byte chunks → 512 chunks.
    //   With 128 threads = 4 chunks/thread.
    //   Row stride = 128 bytes; chunk stride 16 bytes; 8 chunks/row.
    {
        const uint8_t *Kb = K + b * sl * Hd;
        constexpr int CHUNK = 16;
        constexpr int chunks_per_row = Hd / CHUNK;       // 8
        constexpr int total = Bc * chunks_per_row;       // 512
        for (int c = tid; c < total; c += FA_DV_MMA_THREADS) {
            int j_local  = c / chunks_per_row;
            int col_byte = (c % chunks_per_row) * CHUNK;
            int j_g      = kt * Bc + j_local;
            cpa16(&smK[swz_byte(j_local, col_byte)],
                  &Kb[j_g * Hd + col_byte],
                  (j_g < sl) ? CHUNK : 0);  // size=0 → smem zeroed
        }
        cpa_commit();
        cpa_wait<0>();
    }
    __syncthreads();

    // ---- dV accumulator ----
    float dV_acc[NI_DV][4];
    #pragma unroll
    for (int ni = 0; ni < NI_DV; ++ni) {
        #pragma unroll
        for (int s = 0; s < 4; ++s) dV_acc[ni][s] = 0.0f;
    }

    const int n_qt = (sl + Br - 1) / Br;
    for (int qt = 0; qt < n_qt; ++qt) {
        const int qt_base = qt * Br;

        // ===== step A: cp.async Q + dO; sync L =====
        {
            const uint8_t *Qb = Q    + b * sl * Hd;
            const __half  *dB = dO_g + b * sl * Hd;

            // Q FP8 [Br=64, Hd=128] = 8192 bytes, 512 chunks, 4/thread.
            constexpr int CHUNK = 16;
            constexpr int Q_cpr = Hd / CHUNK;                  // 8
            constexpr int Q_total = Br * Q_cpr;                // 512
            for (int c = tid; c < Q_total; c += FA_DV_MMA_THREADS) {
                int i_local  = c / Q_cpr;
                int col_byte = (c % Q_cpr) * CHUNK;
                int i_g      = qt_base + i_local;
                cpa16(&smQ[swz_byte(i_local, col_byte)],
                      &Qb[i_g * Hd + col_byte],
                      (i_g < sl) ? CHUNK : 0);
            }

            // dO FP16 [Br=64, Hd=128] = 16384 bytes, 1024 chunks, 8/thread.
            constexpr int dO_bpr = Hd * 2;                      // 256 bytes/row
            constexpr int dO_cpr = dO_bpr / CHUNK;              // 16
            constexpr int dO_total = Br * dO_cpr;               // 1024
            uint8_t *smdO_b = reinterpret_cast<uint8_t*>(smdO);
            const uint8_t *dB_b = reinterpret_cast<const uint8_t*>(dB);
            for (int c = tid; c < dO_total; c += FA_DV_MMA_THREADS) {
                int i_local  = c / dO_cpr;
                int col_byte = (c % dO_cpr) * CHUNK;
                int i_g      = qt_base + i_local;
                cpa16(smdO_b + i_local * dO_bpr + col_byte,
                      dB_b   + i_g * dO_bpr + col_byte,
                      (i_g < sl) ? CHUNK : 0);
            }

            // L FP32 [Br=64] — keep sync (256 B, OOB cleaner per-element).
            if (tid < Br) {
                int i_g  = qt_base + tid;
                smL[tid] = (i_g < sl) ? L[b * sl + i_g] : 0.0f;
            }

            cpa_commit();
            cpa_wait<0>();
        }
        __syncthreads();

        // ===== step B: Q·K^T MMA (unchanged from B2.1) =====
        uint32_t Qr[KS_QK][4];
        {
            int m_lo = wid * 16 + l_div4 + 0;
            int m_hi = wid * 16 + l_div4 + 8;
            int k0   = l_mod4 * 4;
            #pragma unroll
            for (int ks = 0; ks < KS_QK; ++ks) {
                int k_lo = ks * 32 + k0 + 0;
                int k_hi = ks * 32 + k0 + 16;
                Qr[ks][0] = *reinterpret_cast<uint32_t*>(&smQ[swz_byte(m_lo, k_lo)]);
                Qr[ks][1] = *reinterpret_cast<uint32_t*>(&smQ[swz_byte(m_hi, k_lo)]);
                Qr[ks][2] = *reinterpret_cast<uint32_t*>(&smQ[swz_byte(m_lo, k_hi)]);
                Qr[ks][3] = *reinterpret_cast<uint32_t*>(&smQ[swz_byte(m_hi, k_hi)]);
            }
        }

        uint32_t Sr[NI_QK][2];
        #pragma unroll
        for (int ni = 0; ni < NI_QK; ++ni) { Sr[ni][0] = 0u; Sr[ni][1] = 0u; }

        #pragma unroll
        for (int ks = 0; ks < KS_QK; ++ks) {
            int k0_b = ks * 32 + l_mod4 * 4;
            int k_lo = k0_b + 0;
            int k_hi = k0_b + 16;
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

        // ===== step C: softmax + smP_T write (unchanged from B2.1) =====
        const float L_lo = smL[wid * 16 + l_div4 + 0];
        const float L_hi = smL[wid * 16 + l_div4 + 8];
        const int   i_g_lo = qt_base + wid * 16 + l_div4 + 0;
        const int   i_g_hi = qt_base + wid * 16 + l_div4 + 8;
        const bool  i_lo_oob = (i_g_lo >= sl);
        const bool  i_hi_oob = (i_g_hi >= sl);

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

            auto mask_chk = [&](int i_g, bool i_oob, int j_g, bool j_oob) -> bool {
                if (i_oob || j_oob)                       return true;
                if (causal && j_g > i_g)                  return true;
                if (window > 0 && j_g < i_g + 1 - window) return true;
                return false;
            };

            float p00 = mask_chk(i_g_lo, i_lo_oob, j_g_lo, j_lo_oob) ? 0.0f
                       : __expf(scale * s_mlo_nlo - L_lo);
            float p01 = mask_chk(i_g_lo, i_lo_oob, j_g_hi, j_hi_oob) ? 0.0f
                       : __expf(scale * s_mlo_nhi - L_lo);
            float p10 = mask_chk(i_g_hi, i_hi_oob, j_g_lo, j_lo_oob) ? 0.0f
                       : __expf(scale * s_mhi_nlo - L_hi);
            float p11 = mask_chk(i_g_hi, i_hi_oob, j_g_hi, j_hi_oob) ? 0.0f
                       : __expf(scale * s_mhi_nhi - L_hi);

            __half h_p00 = __float2half(p00);
            __half h_p01 = __float2half(p01);
            __half h_p10 = __float2half(p10);
            __half h_p11 = __float2half(p11);

            int i_local_lo = wid * 16 + l_div4 + 0;
            int i_local_hi = wid * 16 + l_div4 + 8;
            smPT[j_local_lo * Br + i_local_lo] = h_p00;
            smPT[j_local_hi * Br + i_local_lo] = h_p01;
            smPT[j_local_lo * Br + i_local_hi] = h_p10;
            smPT[j_local_hi * Br + i_local_hi] = h_p11;
        }

        __syncthreads();

        // ===== step D: P^T·dO MMA (unchanged from B2.1, path β) =====
        #pragma unroll
        for (int kb = 0; kb < KB_DV; ++kb) {
            int m_lo = wid * 16 + l_div4 + 0;
            int m_hi = wid * 16 + l_div4 + 8;
            int k_lo = kb * 16 + l_mod4 * 2 + 0;
            int k_hi = kb * 16 + l_mod4 * 2 + 8;
            uint32_t Ar0 = *reinterpret_cast<uint32_t*>(&smPT[m_lo * Br + k_lo]);
            uint32_t Ar1 = *reinterpret_cast<uint32_t*>(&smPT[m_hi * Br + k_lo]);
            uint32_t Ar2 = *reinterpret_cast<uint32_t*>(&smPT[m_lo * Br + k_hi]);
            uint32_t Ar3 = *reinterpret_cast<uint32_t*>(&smPT[m_hi * Br + k_hi]);

            #pragma unroll
            for (int ni = 0; ni < NI_DV; ++ni) {
                int n   = ni * 8 + l_div4;
                int kA0 = kb * 16 + l_mod4 * 2 + 0;
                int kA1 = kb * 16 + l_mod4 * 2 + 1;
                int kB0 = kb * 16 + l_mod4 * 2 + 8;
                int kB1 = kb * 16 + l_mod4 * 2 + 9;

                uint16_t lo0 = *reinterpret_cast<uint16_t*>(&smdO[kA0 * Hd + n]);
                uint16_t hi0 = *reinterpret_cast<uint16_t*>(&smdO[kA1 * Hd + n]);
                uint16_t lo1 = *reinterpret_cast<uint16_t*>(&smdO[kB0 * Hd + n]);
                uint16_t hi1 = *reinterpret_cast<uint16_t*>(&smdO[kB1 * Hd + n]);
                uint32_t Br0 = ((uint32_t)hi0 << 16) | (uint32_t)lo0;
                uint32_t Br1 = ((uint32_t)hi1 << 16) | (uint32_t)lo1;

                mma_m16n8k16_f16_f32(
                    dV_acc[ni][0], dV_acc[ni][1], dV_acc[ni][2], dV_acc[ni][3],
                    Ar0, Ar1, Ar2, Ar3,
                    Br0, Br1,
                    dV_acc[ni][0], dV_acc[ni][1], dV_acc[ni][2], dV_acc[ni][3]);
            }
        }

        __syncthreads();
    }

    // ---- Final write dV ----
    {
        int j_local_lo = wid * 16 + l_div4 + 0;
        int j_local_hi = wid * 16 + l_div4 + 8;
        int j_g_lo = kt * Bc + j_local_lo;
        int j_g_hi = kt * Bc + j_local_hi;
        bool j_lo_ok = (j_g_lo < sl);
        bool j_hi_ok = (j_g_hi < sl);

        float *dVb = dV + b * sl * Hd;

        #pragma unroll
        for (int ni = 0; ni < NI_DV; ++ni) {
            int d_lo = ni * 8 + l_mod4 * 2 + 0;
            int d_hi = ni * 8 + l_mod4 * 2 + 1;
            if (j_lo_ok) {
                dVb[j_g_lo * Hd + d_lo] = dV_acc[ni][0];
                dVb[j_g_lo * Hd + d_hi] = dV_acc[ni][1];
            }
            if (j_hi_ok) {
                dVb[j_g_hi * Hd + d_lo] = dV_acc[ni][2];
                dVb[j_g_hi * Hd + d_hi] = dV_acc[ni][3];
            }
        }
    }
}

// =====================================================================
// Host launcher.
// =====================================================================
void launch(
    const uint8_t *Q, const uint8_t *K, const __half *dO_g,
    const float *L, float *dV,
    int bh, int sl, int hd,
    int causal, int window,
    float scale, cudaStream_t stream)
{
    if (hd != FA_DV_MMA_HD) {
        fprintf(stderr, "fa_bwd_dv_mma_p2: hd=%d, expected %d\n", hd, FA_DV_MMA_HD);
        exit(1);
    }
    const int Bc   = FA_DV_MMA_BC;
    const int Br   = FA_DV_MMA_BR;
    const int n_kt = (sl + Bc - 1) / Bc;
    const int grid = bh * n_kt;
    const int smem_bytes_base =
        Bc * hd * sizeof(uint8_t)
      + Br * hd * sizeof(uint8_t)
      + Br * hd * sizeof(__half)
      + Bc * Br * sizeof(__half)
      + Br      * sizeof(float);
    // Force 1 block/SM: reserve enough dummy dynamic SMEM so that per-block
    // SMEM exceeds 50 KB. With sm_120a's 100 KB SMEM configuration only one
    // such block fits per SM. P1 used 41 KB → 2 blocks. P2 = same kernel +
    // 12 KB padding → 53 KB → 1 block. Kernel never touches the padding.
    // This forces occupancy change WITHOUT touching register layout.
    constexpr int FORCE_1BLOCK_PAD = 12 * 1024;
    const int smem_bytes = smem_bytes_base + FORCE_1BLOCK_PAD;

    cudaFuncSetAttribute(kernel_dv_mma_p2,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);
    kernel_dv_mma_p2<<<grid, FA_DV_MMA_THREADS, smem_bytes, stream>>>(
        Q, K, dO_g, L, dV, bh, sl, hd, causal, window, scale);
}

} // namespace fa_bwd_dv_mma_p2
