// =====================================================================
//  fa_bwd_dv_mma_p3.cu — B2.2 P3 axis: full double-buffer overlap.
//
//  Vs P1: smQ/smdO/smL all double-buffered (buf[2]); cp.async fires next
//  qt's tile while computing on current. wait_group(1) keeps one prefetch
//  in flight. Last iteration drops to wait_group(0).
//
//  Forced 1 block/SM (SMEM ~65 KB > 50 → only one block fits at 100 KB
//  config). Same occupancy as P2 → P3−P2 measures overlap benefit in
//  isolation from occupancy cost.
//
//  SMEM layout (~65 KB):
//      smK     [Bc=64 × hd=128]    e4m3   =  8 192 B  single
//      smQ[2]  [Br=64 × hd=128]    e4m3   = 16 384 B  double
//      smdO[2] [Br=64 × hd=128]    f16    = 32 768 B  double
//      smL[2]  [Br=64]             f32    =    512 B  double
//      smPT    [Bc=64 × Br=64]     f16    =  8 192 B  single (compute out)
//      ----
//      Σ ≈ 66 048 B (≈ 64.5 KB)
//
//  Pipeline:
//      Pre-loop: issue qt=0 → buf[0]; commit_group
//      for qt in 0..n_qt:
//          buf_cur = qt & 1
//          if qt+1 < n_qt: issue qt+1 → buf[1-buf_cur]; commit_group
//          wait_group(qt+1 < n_qt ? 1 : 0)
//          __syncthreads
//          compute on buf[buf_cur]
//          __syncthreads (before next iter overwrites)
//
//  Correctness: 11 forms + canary, FP8 floor unchanged (shift = race).
// =====================================================================

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "fa_bwd_common.cuh"

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

namespace fa_bwd_dv_mma_p3 {

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

// ----- Helpers: cooperative cp.async issue for one qt-tile, into specified buf. -----
//
// Issue Q, dO, L for given qt_base into smQ_buf / smdO_buf / smL_buf.
// Caller batches these into a single cp.async.commit_group.
__device__ __forceinline__ void issue_qt_loads(
    const uint8_t *Qb, const __half *dB, const float *Lb_base,
    uint8_t *smQ_buf, __half *smdO_buf, float *smL_buf,
    int qt_base, int sl, int tid)
{
    constexpr int Bc      = FA_DV_MMA_BC;
    constexpr int Br      = FA_DV_MMA_BR;
    constexpr int Hd      = FA_DV_MMA_HD;
    constexpr int CHUNK   = 16;
    constexpr int Q_cpr   = Hd / CHUNK;                // 8
    constexpr int Q_total = Br * Q_cpr;                // 512
    constexpr int dO_bpr  = Hd * 2;                    // 256
    constexpr int dO_cpr  = dO_bpr / CHUNK;            // 16
    constexpr int dO_total = Br * dO_cpr;              // 1024

    // Q FP8
    for (int c = tid; c < Q_total; c += FA_DV_MMA_THREADS) {
        int i_local  = c / Q_cpr;
        int col_byte = (c % Q_cpr) * CHUNK;
        int i_g      = qt_base + i_local;
        cpa16(&smQ_buf[swz_byte(i_local, col_byte)],
              &Qb[i_g * Hd + col_byte],
              (i_g < sl) ? CHUNK : 0);
    }

    // dO FP16
    uint8_t *smdO_b = reinterpret_cast<uint8_t*>(smdO_buf);
    const uint8_t *dB_b = reinterpret_cast<const uint8_t*>(dB);
    for (int c = tid; c < dO_total; c += FA_DV_MMA_THREADS) {
        int i_local  = c / dO_cpr;
        int col_byte = (c % dO_cpr) * CHUNK;
        int i_g      = qt_base + i_local;
        cpa16(smdO_b + i_local * dO_bpr + col_byte,
              dB_b   + i_g * dO_bpr + col_byte,
              (i_g < sl) ? CHUNK : 0);
    }

    // L FP32 — keep sync (256 B per buf, OOB-cleaner). Use single LDG+STS.
    if (tid < Br) {
        int i_g = qt_base + tid;
        smL_buf[tid] = (i_g < sl) ? Lb_base[i_g] : 0.0f;
    }
}

__global__ void kernel_dv_mma_p3(
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

    // ---- SMEM layout (double-buffer for Q/dO/L) ----
    extern __shared__ uint8_t smem_raw[];
    uint8_t *smK    = smem_raw;                                    // 8 KB
    uint8_t *smQ_0  = smK    + Bc * Hd;                            // 8 KB
    uint8_t *smQ_1  = smQ_0  + Br * Hd;                            // 8 KB
    __half  *smdO_0 = reinterpret_cast<__half*>(smQ_1 + Br * Hd);  // 16 KB
    __half  *smdO_1 = reinterpret_cast<__half*>(
        reinterpret_cast<uint8_t*>(smdO_0) + Br * Hd * 2);         // 16 KB
    __half  *smPT   = reinterpret_cast<__half*>(
        reinterpret_cast<uint8_t*>(smdO_1) + Br * Hd * 2);         // 8 KB
    float   *smL_0  = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(smPT) + Bc * Br * 2);           // 256 B
    float   *smL_1  = smL_0 + Br;                                  // 256 B

    uint8_t *smQ_buf[2]  = { smQ_0,  smQ_1  };
    __half  *smdO_buf[2] = { smdO_0, smdO_1 };
    float   *smL_buf[2]  = { smL_0,  smL_1  };

    // ---- Warmup K-tile cp.async ----
    {
        const uint8_t *Kb = K + b * sl * Hd;
        constexpr int CHUNK = 16;
        constexpr int chunks_per_row = Hd / CHUNK;
        constexpr int total = Bc * chunks_per_row;
        for (int c = tid; c < total; c += FA_DV_MMA_THREADS) {
            int j_local  = c / chunks_per_row;
            int col_byte = (c % chunks_per_row) * CHUNK;
            int j_g      = kt * Bc + j_local;
            cpa16(&smK[swz_byte(j_local, col_byte)],
                  &Kb[j_g * Hd + col_byte],
                  (j_g < sl) ? CHUNK : 0);
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
    const uint8_t *Qb = Q    + b * sl * Hd;
    const __half  *dB = dO_g + b * sl * Hd;
    const float   *Lb = L    + b * sl;

    // ---- Pre-loop: issue qt=0 into buf[0] ----
    issue_qt_loads(Qb, dB, Lb, smQ_buf[0], smdO_buf[0], smL_buf[0],
                   0, sl, tid);
    cpa_commit();

    for (int qt = 0; qt < n_qt; ++qt) {
        const int buf_cur = qt & 1;
        const int qt_base = qt * Br;

        // ---- Issue qt+1's loads into next buf (if any) ----
        const bool has_next = (qt + 1 < n_qt);
        if (has_next) {
            const int buf_nxt = buf_cur ^ 1;
            const int qt_nxt_base = (qt + 1) * Br;
            issue_qt_loads(Qb, dB, Lb,
                           smQ_buf[buf_nxt], smdO_buf[buf_nxt], smL_buf[buf_nxt],
                           qt_nxt_base, sl, tid);
            cpa_commit();
        }

        // ---- Wait for current buf data; let next buf stay in flight ----
        if (has_next) cpa_wait<1>();
        else          cpa_wait<0>();
        __syncthreads();

        uint8_t *smQ_cur  = smQ_buf[buf_cur];
        __half  *smdO_cur = smdO_buf[buf_cur];
        float   *smL_cur  = smL_buf[buf_cur];

        // ===== step B: Q·K^T MMA =====
        uint32_t Qr[KS_QK][4];
        {
            int m_lo = wid * 16 + l_div4 + 0;
            int m_hi = wid * 16 + l_div4 + 8;
            int k0   = l_mod4 * 4;
            #pragma unroll
            for (int ks = 0; ks < KS_QK; ++ks) {
                int k_lo = ks * 32 + k0 + 0;
                int k_hi = ks * 32 + k0 + 16;
                Qr[ks][0] = *reinterpret_cast<uint32_t*>(&smQ_cur[swz_byte(m_lo, k_lo)]);
                Qr[ks][1] = *reinterpret_cast<uint32_t*>(&smQ_cur[swz_byte(m_hi, k_lo)]);
                Qr[ks][2] = *reinterpret_cast<uint32_t*>(&smQ_cur[swz_byte(m_lo, k_hi)]);
                Qr[ks][3] = *reinterpret_cast<uint32_t*>(&smQ_cur[swz_byte(m_hi, k_hi)]);
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

        // ===== step C: softmax + smP_T write =====
        const float L_lo = smL_cur[wid * 16 + l_div4 + 0];
        const float L_hi = smL_cur[wid * 16 + l_div4 + 8];
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

        // ===== step D: P^T·dO MMA =====
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

                uint16_t lo0 = *reinterpret_cast<uint16_t*>(&smdO_cur[kA0 * Hd + n]);
                uint16_t hi0 = *reinterpret_cast<uint16_t*>(&smdO_cur[kA1 * Hd + n]);
                uint16_t lo1 = *reinterpret_cast<uint16_t*>(&smdO_cur[kB0 * Hd + n]);
                uint16_t hi1 = *reinterpret_cast<uint16_t*>(&smdO_cur[kB1 * Hd + n]);
                uint32_t Br0 = ((uint32_t)hi0 << 16) | (uint32_t)lo0;
                uint32_t Br1 = ((uint32_t)hi1 << 16) | (uint32_t)lo1;

                mma_m16n8k16_f16_f32(
                    dV_acc[ni][0], dV_acc[ni][1], dV_acc[ni][2], dV_acc[ni][3],
                    Ar0, Ar1, Ar2, Ar3,
                    Br0, Br1,
                    dV_acc[ni][0], dV_acc[ni][1], dV_acc[ni][2], dV_acc[ni][3]);
            }
        }

        __syncthreads();  // before next iter overwrites smPT (current buf still being used for cp.async-next? no, current buf data already consumed)
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

void launch(
    const uint8_t *Q, const uint8_t *K, const __half *dO_g,
    const float *L, float *dV,
    int bh, int sl, int hd,
    int causal, int window,
    float scale, cudaStream_t stream)
{
    if (hd != FA_DV_MMA_HD) {
        fprintf(stderr, "fa_bwd_dv_mma_p3: hd=%d, expected %d\n", hd, FA_DV_MMA_HD);
        exit(1);
    }
    const int Bc   = FA_DV_MMA_BC;
    const int Br   = FA_DV_MMA_BR;
    const int n_kt = (sl + Bc - 1) / Bc;
    const int grid = bh * n_kt;
    const int smem_bytes =
        Bc * hd * sizeof(uint8_t)            // smK   (single)  8K
      + 2 * Br * hd * sizeof(uint8_t)        // smQ[2]          16K
      + 2 * Br * hd * sizeof(__half)         // smdO[2]         32K
      + Bc * Br * sizeof(__half)             // smPT  (single)   8K
      + 2 * Br * sizeof(float);              // smL[2]         512B

    cudaFuncSetAttribute(kernel_dv_mma_p3,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);
    kernel_dv_mma_p3<<<grid, FA_DV_MMA_THREADS, smem_bytes, stream>>>(
        Q, K, dO_g, L, dV, bh, sl, hd, causal, window, scale);
}

} // namespace fa_bwd_dv_mma_p3
