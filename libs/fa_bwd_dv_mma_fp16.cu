// =====================================================================
//  fa_bwd_dv_mma_fp16.cu — B2.1 DIAGNOSTIC: FP16 recompute Q·K^T.
//
//  ЦЕЛЬ ДИАГНОСТИКИ: исключить FP8 quantize Q/K как источник наблюдаемого
//  10× разброса floor (causal F2/F4/F6/F8 ~5e-2 vs non-causal F9 ~1e-3).
//
//  Q,K теперь FP16 (host upload без e4m3 quantize), Q·K^T через m16n8k16
//  FP16 → FP32 acc. Всё остальное идентично fa_bwd_dv_mma.cu:
//    - Тот же block-per-K-tile, Bc=64, Br=64
//    - Тот же threading 128/4 warps, M_TILES_per_warp=1
//    - Тот же softmax (FP16), smP_T write transposed
//    - Тот же P^T·dO MMA m16n8k16 FP16 → FP32 acc (path β)
//
//  Если floor выравнивается ~1e-3 на ВСЕХ формах → FP8 был источником
//    разброса, который зависел от mask geometry (weak averaging при
//    causal small i усиливал FP8 noise). FP8 floor подтверждён.
//
//  Если causal формы остаются 5e-2 → НЕ FP8 floor, а структурный баг
//    в causal-ветке (маска, или N_eff boundary, или partial tile).
//    Локализовать до B2.2.
//
//  SMEM растёт до ~56 KB (Q,K FP16 удваивает smQ+smK). Используем
//  opt-in dynamic SMEM до 100 KB на sm_120a. 1 block/SM — для
//  диагностики OK.
// =====================================================================

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define FA_DV_MMA_FP16_BC 64
#define FA_DV_MMA_FP16_BR 64
#define FA_DV_MMA_FP16_HD 128
#define FA_DV_MMA_FP16_THREADS 128

namespace fa_bwd_dv_mma_fp16 {

// =====================================================================
// PTX m16n8k16.row.col.f32.f16.f16.f32 — same as in fa_bwd_dv_mma.cu.
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
__global__ void kernel_dv_mma_fp16(
    const __half * __restrict__ Q,     // fp16 [bh, sl, hd]
    const __half * __restrict__ K,     // fp16 [bh, sl, hd]
    const __half * __restrict__ dO_g,  // fp16 [bh, sl, hd]
    const float  * __restrict__ L,     // fp32 [bh, sl]
    float        * __restrict__ dV,    // fp32 [bh, sl, hd]
    int bh, int sl, int hd,
    int causal, int window,
    float scale)
{
    constexpr int Bc      = FA_DV_MMA_FP16_BC;   // 64
    constexpr int Br      = FA_DV_MMA_FP16_BR;   // 64
    constexpr int Hd      = FA_DV_MMA_FP16_HD;   // 128
    constexpr int NI_QK   = Bc / 8;              // 8
    constexpr int NI_DV   = Hd / 8;              // 16
    constexpr int KS_QK   = Hd / 16;             // 8  (FP16 m16n8k16 vs 4 FP8 m16n8k32)
    constexpr int KB_DV   = Br / 16;             // 4

    const int tid    = threadIdx.x;
    const int wid    = tid >> 5;
    const int lane   = tid & 31;
    const int l_div4 = lane >> 2;
    const int l_mod4 = lane & 3;

    const int n_kt = (sl + Bc - 1) / Bc;
    const int b    = blockIdx.x / n_kt;
    const int kt   = blockIdx.x % n_kt;
    if (b >= bh) return;

    // SMEM ~56 KB total (opt-in via cudaFuncSetAttribute in launcher).
    extern __shared__ uint8_t smem_raw[];
    __half *smK  = reinterpret_cast<__half*>(smem_raw);                  // 16 KB
    __half *smQ  = smK  + Bc * Hd;                                       // 16 KB
    __half *smdO = smQ  + Br * Hd;                                       // 16 KB
    __half *smPT = smdO + Br * Hd;                                       //  8 KB
    float  *smL  = reinterpret_cast<float*>(smPT + Bc * Br);             //  256 B

    // Warmup: K-tile FP16 load (no swizzle for diagnostic simplicity).
    {
        const __half *Kb = K + b * sl * Hd;
        constexpr int total = Bc * Hd;
        for (int e = tid; e < total; e += FA_DV_MMA_FP16_THREADS) {
            int j_local = e / Hd;
            int d       = e % Hd;
            int j_g     = kt * Bc + j_local;
            __half v    = (j_g < sl) ? Kb[j_g * Hd + d] : __float2half(0.0f);
            smK[j_local * Hd + d] = v;
        }
    }
    __syncthreads();

    // dV accumulator
    float dV_acc[NI_DV][4];
    #pragma unroll
    for (int ni = 0; ni < NI_DV; ++ni) {
        #pragma unroll
        for (int s = 0; s < 4; ++s) dV_acc[ni][s] = 0.0f;
    }

    const int n_qt = (sl + Br - 1) / Br;
    for (int qt = 0; qt < n_qt; ++qt) {
        const int qt_base = qt * Br;

        // Step A: load Q, dO (both FP16), L (FP32).
        {
            const __half *Qb = Q    + b * sl * Hd;
            const __half *dB = dO_g + b * sl * Hd;
            constexpr int total = Br * Hd;
            for (int e = tid; e < total; e += FA_DV_MMA_FP16_THREADS) {
                int i_local = e / Hd;
                int d       = e % Hd;
                int i_g     = qt_base + i_local;
                __half vQ = (i_g < sl) ? Qb[i_g * Hd + d] : __float2half(0.0f);
                __half vO = (i_g < sl) ? dB[i_g * Hd + d] : __float2half(0.0f);
                smQ [i_local * Hd + d] = vQ;
                smdO[i_local * Hd + d] = vO;
            }
            if (tid < Br) {
                int i_g  = qt_base + tid;
                smL[tid] = (i_g < sl) ? L[b * sl + i_g] : 0.0f;
            }
        }
        __syncthreads();

        // ===== Step B: Q·K^T MMA (FP16 m16n8k16 → f32 acc) =====
        // KS_QK = 8 (Hd/16) vs 4 (Hd/32) for FP8.
        // Pre-load Qr fragments — row-major in smQ FP16.
        uint32_t Qr[KS_QK][4];
        {
            int m_lo = wid * 16 + l_div4 + 0;
            int m_hi = wid * 16 + l_div4 + 8;
            #pragma unroll
            for (int ks = 0; ks < KS_QK; ++ks) {
                int k_lo = ks * 16 + l_mod4 * 2 + 0;
                int k_hi = ks * 16 + l_mod4 * 2 + 8;
                Qr[ks][0] = *reinterpret_cast<uint32_t*>(&smQ[m_lo * Hd + k_lo]);
                Qr[ks][1] = *reinterpret_cast<uint32_t*>(&smQ[m_hi * Hd + k_lo]);
                Qr[ks][2] = *reinterpret_cast<uint32_t*>(&smQ[m_lo * Hd + k_hi]);
                Qr[ks][3] = *reinterpret_cast<uint32_t*>(&smQ[m_hi * Hd + k_hi]);
            }
        }

        // Sr accumulator FP32 — 8 ni × 4 fp32 per lane.
        float Sr[NI_QK][4];
        #pragma unroll
        for (int ni = 0; ni < NI_QK; ++ni) {
            Sr[ni][0] = 0.0f; Sr[ni][1] = 0.0f;
            Sr[ni][2] = 0.0f; Sr[ni][3] = 0.0f;
        }

        #pragma unroll
        for (int ks = 0; ks < KS_QK; ++ks) {
            int k_lo = ks * 16 + l_mod4 * 2 + 0;
            int k_hi = ks * 16 + l_mod4 * 2 + 8;
            #pragma unroll
            for (int ni = 0; ni < NI_QK; ++ni) {
                int n_K = ni * 8 + l_div4;
                uint32_t Kr0 = *reinterpret_cast<uint32_t*>(&smK[n_K * Hd + k_lo]);
                uint32_t Kr1 = *reinterpret_cast<uint32_t*>(&smK[n_K * Hd + k_hi]);
                mma_m16n8k16_f16_f32(
                    Sr[ni][0], Sr[ni][1], Sr[ni][2], Sr[ni][3],
                    Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3],
                    Kr0, Kr1,
                    Sr[ni][0], Sr[ni][1], Sr[ni][2], Sr[ni][3]);
            }
        }

        // ===== Step C: Softmax + smP_T write (identical to FP8 version, except
        //               Sr is FP32 acc not half2 packed) =====
        const float L_lo = smL[wid * 16 + l_div4 + 0];
        const float L_hi = smL[wid * 16 + l_div4 + 8];
        const int   i_g_lo = qt_base + wid * 16 + l_div4 + 0;
        const int   i_g_hi = qt_base + wid * 16 + l_div4 + 8;
        const bool  i_lo_oob = (i_g_lo >= sl);
        const bool  i_hi_oob = (i_g_hi >= sl);

        #pragma unroll
        for (int ni = 0; ni < NI_QK; ++ni) {
            // FP32 direct unpacking (per probe-verified m16n8k16 D layout).
            float s_mlo_nlo = Sr[ni][0];
            float s_mlo_nhi = Sr[ni][1];
            float s_mhi_nlo = Sr[ni][2];
            float s_mhi_nhi = Sr[ni][3];

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

            int i_local_lo = wid * 16 + l_div4 + 0;
            int i_local_hi = wid * 16 + l_div4 + 8;
            smPT[j_local_lo * Br + i_local_lo] = __float2half(p00);
            smPT[j_local_hi * Br + i_local_lo] = __float2half(p01);
            smPT[j_local_lo * Br + i_local_hi] = __float2half(p10);
            smPT[j_local_hi * Br + i_local_hi] = __float2half(p11);
        }

        __syncthreads();

        // ===== Step D: P^T·dO MMA (path β, IDENTICAL to FP8 version) =====
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

    // ===== Final write dV[b, j_g, d_g] = dV_acc[ni][slot] FP32 =====
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

// Host launcher with opt-in dynamic SMEM.
void launch(
    const __half *Q, const __half *K, const __half *dO_g,
    const float *L, float *dV,
    int bh, int sl, int hd,
    int causal, int window,
    float scale, cudaStream_t stream)
{
    if (hd != FA_DV_MMA_FP16_HD) {
        fprintf(stderr, "fa_bwd_dv_mma_fp16: hd=%d, expected %d\n",
                hd, FA_DV_MMA_FP16_HD);
        exit(1);
    }
    const int Bc   = FA_DV_MMA_FP16_BC;
    const int Br   = FA_DV_MMA_FP16_BR;
    const int n_kt = (sl + Bc - 1) / Bc;
    const int grid = bh * n_kt;
    const int smem_bytes =
        Bc * hd * sizeof(__half)   // smK
      + Br * hd * sizeof(__half)   // smQ
      + Br * hd * sizeof(__half)   // smdO
      + Bc * Br * sizeof(__half)   // smPT
      + Br      * sizeof(float);   // smL

    cudaFuncSetAttribute(kernel_dv_mma_fp16,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    kernel_dv_mma_fp16<<<grid, FA_DV_MMA_FP16_THREADS, smem_bytes, stream>>>(
        Q, K, dO_g, L, dV, bh, sl, hd, causal, window, scale);
}

} // namespace fa_bwd_dv_mma_fp16
