// R1b-fix: kernel_dk_new v2 with Q → Q_T transpose phase.
//   dK[j][d] = sum_i dS^T[j,i] * Q[i,d]
//   MMA m16n8k32 row.col:
//     A [M=j][K=i] row-major = dS_T   (Bc rows × Br cols, stride Br)
//     B [K=i][N=d] col-major = Q_T[d][i] row-major с stride QT_STRIDE (mirror sealed dK)
//   → Q loaded row-major → transposed to smQ_T (mirror sealed dK phase 1.5).
//
//   Grid: bh × n_kt (block owns K-tile).
//   Loop qt: load Q + dS_T tile → transpose Q → smQ_T → MMA.
//   Epilogue: dK_acc * scale → global.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "fa_bwd_common.cuh"

#define FA_DKN_BC        64
#define FA_DKN_BR        64
#define FA_DKN_HD        128
#define FA_DKN_THREADS   128
#define FA_DKN_QT_STRIDE 68     // stride for smQ_T bank-conflict-free (mirror sealed dK)

namespace fa_bwd_dk_new {

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

__global__ void kernel_dk_new(
    const uint8_t * __restrict__ Q,       // [bh, sl, hd] FP8
    const uint8_t * __restrict__ dS_T,    // [bh, sl_j, sl_i] FP8
    float         * __restrict__ dK,      // [bh, sl, hd] FP32
    int bh, int sl, int hd,
    int causal, int window,
    float scale)
{
    constexpr int Bc         = FA_DKN_BC;
    constexpr int Br         = FA_DKN_BR;
    constexpr int Hd         = FA_DKN_HD;
    constexpr int QT_STRIDE  = FA_DKN_QT_STRIDE;
    constexpr int NI_DK      = Hd / 8;         // 16
    constexpr int KB_DK      = Br / 32;        // 2
    constexpr int KS_QK      = Hd / 32;        // 4  (used for Q-fragment load pattern)

    const int tid    = threadIdx.x;
    const int wid    = tid >> 5;
    const int lane   = tid & 31;
    const int l_div4 = lane >> 2;
    const int l_mod4 = lane & 3;

    const int n_kt = (sl + Bc - 1) / Bc;
    const int b    = blockIdx.x / n_kt;
    const int kt   = blockIdx.x % n_kt;
    if (b >= bh) return;
    const int j_base = kt * Bc;

    // SMEM: smQ (Br*Hd=8192) + smQ_T (Hd*QT_STRIDE=8704) + smdS_T (Bc*Br=4096) = 20992 B.
    //   4 blocks/SM (SMEM-limited, floor(102400/(20992+1024))=4).
    extern __shared__ uint8_t smem_raw[];
    uint8_t *smQ    = smem_raw;                             // 8192
    uint8_t *smQ_T  = smem_raw + Br * Hd;                   // 8704 (stride QT_STRIDE)
    uint8_t *smdS_T = smem_raw + Br * Hd + Hd * QT_STRIDE;  // 4096

    // dK_acc FP32 [16][4] = 64 regs
    float dK_acc[NI_DK][4];
    #pragma unroll
    for (int ni = 0; ni < NI_DK; ++ni)
        #pragma unroll
        for (int s = 0; s < 4; ++s) dK_acc[ni][s] = 0.0f;

    const int n_qt = (sl + Br - 1) / Br;
    const int qt_start = causal ? kt : 0;

    for (int qt = qt_start; qt < n_qt; ++qt) {
        const int i_base = qt * Br;

        // cp.async Q rows → smQ (Br*Hd = 8KB, 16-byte chunks × 512 total)
        {
            const uint8_t *Qb = Q + b * sl * Hd;
            constexpr int CHUNK = 16;
            constexpr int cpr   = Hd / CHUNK;    // 8
            constexpr int total = Br * cpr;      // 512
            for (int c = tid; c < total; c += FA_DKN_THREADS) {
                int i_local = c / cpr;
                int col_byte = (c % cpr) * CHUNK;
                int i_g = i_base + i_local;
                bool in_bounds = (i_g < sl);
                if (in_bounds) {
                    cpa16(&smQ[i_local * Hd + col_byte],
                          &Qb[i_g * Hd + col_byte], CHUNK);
                } else {
                    uint32_t *sp = reinterpret_cast<uint32_t*>(&smQ[i_local * Hd + col_byte]);
                    sp[0] = 0u; sp[1] = 0u; sp[2] = 0u; sp[3] = 0u;
                }
            }
        }
        // cp.async dS_T tile → smdS_T (Bc rows × Br cols).
        //   ABI: dS_T stored with padded row stride = (sl + 15) & ~15 for 16-byte alignment.
        {
            const int stride_ds = (sl + 15) & ~15;    // padded stride for 16B alignment
            const uint8_t *dS_Tb = dS_T + (size_t)b * sl * stride_ds;
            constexpr int CHUNK = 16;
            constexpr int cpr   = Br / CHUNK;    // 4
            constexpr int total = Bc * cpr;      // 256
            for (int c = tid; c < total; c += FA_DKN_THREADS) {
                int j_local = c / cpr;
                int col_byte = (c % cpr) * CHUNK;
                int j_g = j_base + j_local;
                int i_g = i_base + col_byte;
                // In-bounds: coalesced cp.async (stride guarantees 16B alignment for any j_g).
                // OOB: STS-zero at aligned SMEM position.
                bool in_bounds = (j_g < sl && i_g < sl);
                if (in_bounds) {
                    cpa16(&smdS_T[j_local * Br + col_byte],
                          &dS_Tb[(size_t)j_g * stride_ds + i_g], CHUNK);
                } else {
                    uint32_t *sp = reinterpret_cast<uint32_t*>(&smdS_T[j_local * Br + col_byte]);
                    sp[0] = 0u; sp[1] = 0u; sp[2] = 0u; sp[3] = 0u;
                }
            }
        }
        cpa_commit();
        cpa_wait<0>();
        __syncthreads();

        // Load Q fragments (matches sealed dK step B pattern) → Qr[KS_QK][4].
        //   Per lane: 4 ks × 4 u32 = 16 LDS.U32.
        uint32_t Qr[KS_QK][4];
        #pragma unroll
        for (int ks = 0; ks < KS_QK; ++ks) {
            int m_lo = wid * 16 + l_div4 + 0;
            int m_hi = wid * 16 + l_div4 + 8;
            int k_lo = ks * 32 + l_mod4 * 4 + 0;
            int k_hi = ks * 32 + l_mod4 * 4 + 16;
            Qr[ks][0] = *reinterpret_cast<uint32_t*>(&smQ[m_lo * Hd + k_lo]);
            Qr[ks][1] = *reinterpret_cast<uint32_t*>(&smQ[m_hi * Hd + k_lo]);
            Qr[ks][2] = *reinterpret_cast<uint32_t*>(&smQ[m_lo * Hd + k_hi]);
            Qr[ks][3] = *reinterpret_cast<uint32_t*>(&smQ[m_hi * Hd + k_hi]);
        }

        // Transpose Qr → smQ_T via pack (Vugar spec verbatim, unit-tested 8192/8192).
        //   Фазы A/B/C/D: gather-PRMT / SHFL exchange / receive-PRMT / STS.32
        //   Счёт per qt/lane: 12 SHFL + 16 STS.32 + 64 PRMT + 24 SEL, 0 STS.U8, 0 LDL/STL.
        //   Coords: c = l_mod4, p = l_div4 & 3, h = l_div4 >> 2 (quad structure).
        {
            const int c = l_mod4;
            const int p = l_div4 & 3;
            const int h = l_div4 >> 2;
            #pragma unroll
            for (int s = 0; s < 4; ++s) {
                // Фаза A — gather вдоль ks (8 PRMT, fixed selectors)
                uint32_t t01_lo, t01_hi, t23_lo, t23_hi;
                asm volatile("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(t01_lo) : "r"(Qr[0][s]), "r"(Qr[1][s]));
                asm volatile("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(t01_hi) : "r"(Qr[0][s]), "r"(Qr[1][s]));
                asm volatile("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(t23_lo) : "r"(Qr[2][s]), "r"(Qr[3][s]));
                asm volatile("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(t23_hi) : "r"(Qr[2][s]), "r"(Qr[3][s]));
                uint32_t G0, G1, G2, G3;
                asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(G0) : "r"(t01_lo), "r"(t23_lo));
                asm volatile("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(G1) : "r"(t01_lo), "r"(t23_lo));
                asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(G2) : "r"(t01_hi), "r"(t23_hi));
                asm volatile("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(G3) : "r"(t01_hi), "r"(t23_hi));

                // Фаза B — обмен (3 SHFL, rounds r=1..3). V0..V3 в регистрах (без local memory).
                //   src(r) = c + 4*((p−r)&3) + 16h, expose(r) = G[(p+r)&3]
                uint32_t V0 = G0, V1 = G1, V2 = G2, V3 = G3;   // init with own G[i]
                #pragma unroll
                for (int r = 1; r <= 3; ++r) {
                    int src_p = (p - r) & 3;
                    int src_lane = c + 4 * src_p + 16 * h;
                    int idx = (p + r) & 3;
                    // Two-level SEL: expose_val = G[idx]
                    uint32_t lo = (idx & 1) ? G1 : G0;
                    uint32_t hi = (idx & 1) ? G3 : G2;
                    uint32_t expose_val = (idx & 2) ? hi : lo;
                    uint32_t val = __shfl_sync(0xFFFFFFFF, expose_val, src_lane);
                    // Conditional SEL-based write to V0..V3 (compile-time indexed vars)
                    V0 = (src_p == 0) ? val : V0;
                    V1 = (src_p == 1) ? val : V1;
                    V2 = (src_p == 2) ? val : V2;
                    V3 = (src_p == 3) ? val : V3;
                }
                // Own p stays as G[p] via init (no override needed since src_p != p in Phase B)

                // Фаза C — приёмное транспонирование (8 PRMT, тот же tree)
                uint32_t u01_lo, u01_hi, u23_lo, u23_hi;
                asm volatile("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(u01_lo) : "r"(V0), "r"(V1));
                asm volatile("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(u01_hi) : "r"(V0), "r"(V1));
                asm volatile("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(u23_lo) : "r"(V2), "r"(V3));
                asm volatile("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(u23_hi) : "r"(V2), "r"(V3));
                uint32_t OUT0, OUT1, OUT2, OUT3;
                asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(OUT0) : "r"(u01_lo), "r"(u23_lo));
                asm volatile("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(OUT1) : "r"(u01_lo), "r"(u23_lo));
                asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(OUT2) : "r"(u01_hi), "r"(u23_hi));
                asm volatile("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(OUT3) : "r"(u01_hi), "r"(u23_hi));

                // Фаза D — стор (4 STS.32 per slot)
                int colbase = wid * 16 + 8 * (s & 1) + 4 * h;
                int row_base_ks = 16 * (s >> 1) + 4 * c + p;
                *reinterpret_cast<uint32_t*>(&smQ_T[(0 * 32 + row_base_ks) * QT_STRIDE + colbase]) = OUT0;
                *reinterpret_cast<uint32_t*>(&smQ_T[(1 * 32 + row_base_ks) * QT_STRIDE + colbase]) = OUT1;
                *reinterpret_cast<uint32_t*>(&smQ_T[(2 * 32 + row_base_ks) * QT_STRIDE + colbase]) = OUT2;
                *reinterpret_cast<uint32_t*>(&smQ_T[(3 * 32 + row_base_ks) * QT_STRIDE + colbase]) = OUT3;
            }
        }
        __syncthreads();

        // MMA dS_T · Q_T → dK_acc.
        //   A = smdS_T [M=j][K=i] row-major stride Br
        //   B = smQ_T  [N=d][K=i] row-major stride QT_STRIDE (== col-major B[K=i][N=d])
        #pragma unroll
        for (int kb = 0; kb < KB_DK; ++kb) {
            int m_lo = wid * 16 + l_div4 + 0;
            int m_hi = wid * 16 + l_div4 + 8;
            int k_i_lo = kb * 32 + l_mod4 * 4 + 0;
            int k_i_hi = kb * 32 + l_mod4 * 4 + 16;

            uint32_t A0 = *reinterpret_cast<uint32_t*>(&smdS_T[m_lo * Br + k_i_lo]);
            uint32_t A1 = *reinterpret_cast<uint32_t*>(&smdS_T[m_hi * Br + k_i_lo]);
            uint32_t A2 = *reinterpret_cast<uint32_t*>(&smdS_T[m_lo * Br + k_i_hi]);
            uint32_t A3 = *reinterpret_cast<uint32_t*>(&smdS_T[m_hi * Br + k_i_hi]);

            #pragma unroll
            for (int ni = 0; ni < NI_DK; ++ni) {
                int n_d = ni * 8 + l_div4;
                uint32_t B0 = *reinterpret_cast<uint32_t*>(&smQ_T[n_d * QT_STRIDE + k_i_lo]);
                uint32_t B1 = *reinterpret_cast<uint32_t*>(&smQ_T[n_d * QT_STRIDE + k_i_hi]);

                mma_m16n8k32_e4m3_f32(
                    dK_acc[ni][0], dK_acc[ni][1], dK_acc[ni][2], dK_acc[ni][3],
                    A0, A1, A2, A3, B0, B1,
                    dK_acc[ni][0], dK_acc[ni][1], dK_acc[ni][2], dK_acc[ni][3]);
            }
        }
        __syncthreads();
    }

    // Epilogue: dK_acc * scale → global dK
    {
        int j_local_lo = wid * 16 + l_div4 + 0;
        int j_local_hi = wid * 16 + l_div4 + 8;
        int j_g_lo = j_base + j_local_lo;
        int j_g_hi = j_base + j_local_hi;
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

void launch_dk_new(
    const uint8_t *Q, const uint8_t *dS_T,
    float *dK,
    int bh, int sl, int hd,
    int causal, int window,
    float scale, cudaStream_t stream)
{
    if (hd != FA_DKN_HD) {
        fprintf(stderr, "fa_bwd_dk_new: hd=%d, expected %d\n", hd, FA_DKN_HD);
        exit(1);
    }
    const int Bc = FA_DKN_BC;
    const int Br = FA_DKN_BR;
    const int n_kt = (sl + Bc - 1) / Bc;
    const int grid = bh * n_kt;
    const int smem_bytes = Br * hd + hd * FA_DKN_QT_STRIDE + Bc * Br;  // 8192 + 8704 + 4096
    cudaFuncSetAttribute(kernel_dk_new,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    kernel_dk_new<<<grid, FA_DKN_THREADS, smem_bytes, stream>>>(
        Q, dS_T, dK, bh, sl, hd, causal, window, scale);
}

} // namespace fa_bwd_dk_new
