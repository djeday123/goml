// =====================================================================
//  fa_bwd_dv_baseline.cu — B2.0 dV-only baseline (FP32 CUDA-cores).
//
//  ЦЕЛЬ ЭТАПА: первое работающее backward-вычисление на GPU. Только dV.
//  Корректность, НЕ скорость. Опорная точка «математика верна, mapping
//  на GPU-модель исполнения работает» — отдельная от будущей «MMA-раскладка
//  верна» (B2.1).
//
//  ЗАПРЕЩЕНО на этом этапе: MMA, dK, dQ, swizzle, hoisting, half2,
//  любая оптимизация скорости, бенчи.
//  Только: правильный медленный dV + железная валидация.
//
//  10-30 TFLOPS ожидаемо и нормально.
//
// ---------------------------------------------------------------------
//  МАТЕМАТИКА (Tri Dao Variant 3, выжимка для dV):
//
//      P[b, i, j] = exp(scale * (Q[b,i] · K[b,j]) - L[b,i])
//      dV[b, j, d] = Σ_i P[b, i, j] * dO[b, i, d]
//      scale = 1 / sqrt(hd),   L = log-sum-exp (natural log)
//
//  Маски (causal / window) применяются к P (skip → P=0):
//      causal: skip if j > i
//      window: skip if j < i + 1 - window  (sliding на правую границу)
//
//  P recompute, НЕ хранение: P[b,i,j] восстанавливается из Q,K,L
//  (sl=8192 хранение P = 256 MB/head — недопустимо).
// ---------------------------------------------------------------------
//  ГЕОМЕТРИЯ (block-residency per K-tile, атомиков нет):
//
//      Bc = 64        — K/V-строк на блок (= forward Bc)
//      Br = 1         — итерируем по одной Q-строке за раз (baseline,
//                       не Q-тайлами; экономит SMEM и предельно прост)
//      hd = 128       — фиксируем хардкодом
//      threads = 128  — 4 warps на блок
//
//      Grid: (bh × n_kt),  n_kt = ceil(sl / Bc)
//      blockIdx.x = b * n_kt + kt
//
//      Блок владеет K/V-тайлом [kt*Bc .. kt*Bc + Bc), пробегает все i ∈ [0, sl),
//      аккумулирует dV[j_inner, 0..hd-1] в РЕГИСТРАХ. Один write в gmem в конце.
// ---------------------------------------------------------------------
//  THREADING / РАСКЛАДКА ЭЛЕМЕНТОВ:
//
//      tid = 0..127
//      j_inner = tid / 2          ∈ [0, 64)   — какую j-строку держит thread
//      d_pack  = tid & 1          ∈ {0, 1}    — какую половину hd держит
//
//      Per-thread accumulator:
//          float dV_acc[64];                  — dV[j_inner, d_pack*64 + 0..63]
//      Итого 64 FP32 регистра на acc + scratch — комфортно (~80 regs total).
// ---------------------------------------------------------------------
//  SMEM LAYOUT (~33 KB, без opt-in max smem):
//
//      smK[Bc * hd] FP32   = 64*128*4 = 32 KB   — K-tile, один раз на блок
//      smQ[hd]      FP32   = 128*4    = 512 B   — текущая Q[i]
//      smdO[hd]     FP32   = 128*4    = 512 B   — текущая dO[i]
//      smL          FP32   = 4 B                — текущая L[i]
//      ---
//      Σ ≈ 33 KB. Дефолтная 48 KB limit per block — влезает.
//
//  V в dV-pass НЕ нужен (dV = P^T · dO; V не входит в формулу). Подтверждено
//  в fa_bwd_cpu_reference_fp64_golden.cu:131 (dV update не читает V).
// ---------------------------------------------------------------------
//  HOT LOOP (per block, после warmup-загрузки smK):
//
//      for (i = 0; i < sl; ++i) {
//          // step A: cooperative load Q[b,i,:], dO[b,i,:], L[b,i] → SMEM
//          if (tid < hd) smQ[tid]  = Q [b*sl*hd + i*hd + tid];   // 128 elems / 128 threads
//          if (tid < hd) smdO[tid] = dO[b*sl*hd + i*hd + tid];
//          if (tid == 0) smL       = L [b*sl    + i];
//          __syncthreads();
//
//          // step B: маска (causal / window)
//          int j_global = kt*Bc + j_inner;
//          bool masked = (causal && j_global > i) ||
//                        (window > 0 && j_global < i + 1 - window) ||
//                        (j_global >= sl);
//
//          if (!masked) {
//              // step C: score = scale * dot(smQ, smK[j_inner])
//              float score = 0.0f;
//              #pragma unroll
//              for (int d = 0; d < hd; ++d)
//                  score += smQ[d] * smK[j_inner * hd + d];
//              score *= scale;
//
//              // step D: P = exp(score - L[i])
//              float P = expf(score - smL);
//
//              // step E: dV_acc[d_local] += P * smdO[d_pack*64 + d_local]
//              int d_base = d_pack * 64;
//              #pragma unroll
//              for (int d_local = 0; d_local < 64; ++d_local)
//                  dV_acc[d_local] += P * smdO[d_base + d_local];
//          }
//          __syncthreads();  // перед reload smQ/smdO следующей итерации
//      }
//
//      // финальный write: dV_global[b, j_global, d_pack*64 + 0..63]
//      if (j_global < sl) {
//          #pragma unroll
//          for (int d_local = 0; d_local < 64; ++d_local)
//              dV[b*sl*hd + j_global*hd + d_base + d_local] = dV_acc[d_local];
//      }
//
// ---------------------------------------------------------------------
//  ВАЛИДАЦИЯ (двухуровневая из B1-FIX-EXTRA §5.6):
//
//      Уровень 1 — CI:    GPU dV (FP32) ↔ CPU FP32 dV-only reference
//                         hybrid tol: abs 1e-4 + rel 1e-3
//      Уровень 2 — Debug: GPU dV (FP32, upcast to FP64) ↔ FP64-golden
//                         реалистично ожидать FP32 floor ~1e-4 abs / ~1e-3 rel
//                         (FP64 golden rel 1e-5..1e-6 в handoff — это для FP64
//                         backward vs FP64 finite-diff, не FP32-GPU vs FP64-ref)
//
//      Прогон: 8 форм + канарейка (sl=300 wnd=96), причинная и оконная маски.
//
//  Приёмка B2.0: dV сходится на всех 8 формах + канарейке. ptxas зафиксирован
//  как точка отсчёта. Скорость НЕ мерить, НЕ оптимизировать.
// =====================================================================

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#ifndef FA_DV_BASELINE_BC
#define FA_DV_BASELINE_BC 64
#endif
#ifndef FA_DV_BASELINE_HD
#define FA_DV_BASELINE_HD 128
#endif
#ifndef FA_DV_BASELINE_THREADS
#define FA_DV_BASELINE_THREADS 128
#endif
#ifndef FA_DV_BASELINE_DPACK
#define FA_DV_BASELINE_DPACK (FA_DV_BASELINE_HD / 2)   // 64 acc per thread
#endif

#define DV_CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) {            \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                   \
            cudaGetErrorString(e)); exit(1); }} while (0)

namespace fa_bwd_dv_baseline {

// ---------------------------------------------------------------------
// dV-only baseline ядро. FP32 in/out. Без MMA.
//
// Параметры:
//   Q       [bh, sl, hd]  FP32 — input (host gradient src)
//   K       [bh, sl, hd]  FP32
//   dO      [bh, sl, hd]  FP32
//   L       [bh, sl]      FP32 — L_i = m_i + log(l_i), natural log
//   dV      [bh, sl, hd]  FP32 — output, перезаписывается полностью
//   bh, sl, hd            — формы
//   causal                — 0 / 1
//   window                — 0 (off) или >0 (sliding)
//   scale                 — 1 / sqrt(hd)
// ---------------------------------------------------------------------
__global__ void dv_baseline_kernel(
    const float * __restrict__ Q,
    const float * __restrict__ K,
    const float * __restrict__ dO_g,
    const float * __restrict__ L,
    float       * __restrict__ dV,
    int bh, int sl, int hd,
    int causal, int window,
    float scale)
{
    constexpr int Bc    = FA_DV_BASELINE_BC;
    constexpr int Hd    = FA_DV_BASELINE_HD;
    constexpr int Dpack = FA_DV_BASELINE_DPACK;  // 64

    const int tid     = threadIdx.x;
    const int j_inner = tid >> 1;        // 0..63
    const int d_pack  = tid & 1;         // 0 or 1
    const int d_base  = d_pack * Dpack;  // 0 or 64

    const int n_kt = (sl + Bc - 1) / Bc;
    const int b    = blockIdx.x / n_kt;
    const int kt   = blockIdx.x % n_kt;
    if (b >= bh) return;

    const int j_global = kt * Bc + j_inner;

    // Per-thread FP32 dV accumulator (64 elements).
    float dV_acc[Dpack];
    #pragma unroll
    for (int d = 0; d < Dpack; ++d) dV_acc[d] = 0.0f;

    // SMEM layout.
    extern __shared__ float smem[];
    float *smK  = smem;                       // [Bc * Hd]
    float *smQ  = smK  + Bc * Hd;             // [Hd]
    float *smdO = smQ  + Hd;                  // [Hd]
    float *smL  = smdO + Hd;                  // [1]

    // ---- Warmup: full K-tile into SMEM (one-shot, FP32 gmem reads) ----
    const float *Kb = K + b * sl * Hd;
    {
        // Bc * Hd = 64 * 128 = 8192 elements, 128 threads → 64 per thread
        constexpr int total = Bc * Hd;
        #pragma unroll
        for (int e = tid; e < total; e += FA_DV_BASELINE_THREADS) {
            int j_local = e / Hd;
            int d_idx   = e % Hd;
            int j_g     = kt * Bc + j_local;
            float v     = 0.0f;
            if (j_g < sl) v = Kb[j_g * Hd + d_idx];
            smK[e] = v;
        }
    }
    __syncthreads();

    // ---- Hot loop over Q-rows ----
    const float *Qb  = Q    + b * sl * Hd;
    const float *dOb = dO_g + b * sl * Hd;
    const float *Lb  = L    + b * sl;

    for (int i = 0; i < sl; ++i) {
        // step A: cooperative load Q[i], dO[i], L[i] into SMEM (128 thr × 128 hd → 1 per thread)
        if (tid < Hd) {
            smQ [tid] = Qb [i * Hd + tid];
            smdO[tid] = dOb[i * Hd + tid];
        }
        if (tid == 0) smL[0] = Lb[i];
        __syncthreads();

        // step B: mask check
        bool masked = (j_global >= sl);
        if (causal && j_global > i)                         masked = true;
        if (window > 0 && j_global < i + 1 - window)        masked = true;

        if (!masked) {
            // step C: score = scale * dot(Q[i], K[j])
            float score = 0.0f;
            #pragma unroll
            for (int d = 0; d < Hd; ++d)
                score += smQ[d] * smK[j_inner * Hd + d];
            score *= scale;

            // step D: P = exp(score - L[i])
            float P = __expf(score - smL[0]);

            // step E: dV_acc[d_local] += P * dO[i, d_base + d_local]
            #pragma unroll
            for (int d_local = 0; d_local < Dpack; ++d_local)
                dV_acc[d_local] += P * smdO[d_base + d_local];
        }
        __syncthreads();  // before next iteration overwrites smQ/smdO
    }

    // ---- Final write ----
    if (j_global < sl) {
        float *dVb = dV + b * sl * Hd + j_global * Hd + d_base;
        #pragma unroll
        for (int d_local = 0; d_local < Dpack; ++d_local)
            dVb[d_local] = dV_acc[d_local];
    }
}

// ---------------------------------------------------------------------
// Host launcher.
// ---------------------------------------------------------------------
void launch(
    const float *Q, const float *K, const float *dO_g,
    const float *L, float *dV,
    int bh, int sl, int hd,
    int causal, int window,
    float scale, cudaStream_t stream = 0)
{
    if (hd != FA_DV_BASELINE_HD) {
        fprintf(stderr, "fa_bwd_dv_baseline: hd=%d, expected %d\n",
                hd, FA_DV_BASELINE_HD);
        exit(1);
    }
    const int Bc   = FA_DV_BASELINE_BC;
    const int n_kt = (sl + Bc - 1) / Bc;
    const int grid = bh * n_kt;
    const int smem_bytes =
        Bc * hd * sizeof(float)   // smK
        + hd     * sizeof(float)  // smQ
        + hd     * sizeof(float)  // smdO
        + 1      * sizeof(float); // smL

    dv_baseline_kernel<<<grid, FA_DV_BASELINE_THREADS, smem_bytes, stream>>>(
        Q, K, dO_g, L, dV, bh, sl, hd, causal, window, scale);
}

} // namespace fa_bwd_dv_baseline
