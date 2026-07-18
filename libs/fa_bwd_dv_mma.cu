// =====================================================================
//  fa_bwd_dv_mma.cu — B2.1 dV-only with MMA (FP8 Q·K^T + FP16 P^T·dO).
//
//  STATUS: design comment only — KERNEL BODY NOT YET WRITTEN.
//          Design reviewed + probe-confirmed. Ready to write body.
//
//  Цель этапа B2.1: первая корректная MMA-версия backward dV. НЕ скорость.
//  Baseline B2.0 (fa_bwd_dv_baseline.cu, FP32 CUDA-cores, 11/11 × 4 seeds
//  PASS, max_rel_af ≤ 2.88e-5) — вторая опорная точка. Любое расхождение
//  MMA-версии за пределами FP8-quantize floor → bug в MMA-раскладке
//  (математика доказана B2.0).
//
//  РЕШЕНИЯ (Vugar, 2026-06-14) + ревью:
//    [Q1] ОТМЕНЁН ldmatrix.trans. Закрытое направление (forward E1 Probe C:
//         −55% wall на sm_120a). Никакого ldmatrix ни как основного, ни как
//         fallback — не плодим долг на B2.2.
//    [Q1] P записывается в smP сразу в транспонированной раскладке (smP_T)
//         тем же приёмом, которым forward пишет smV_T (стиль transpose_v).
//         P производится нами в регистрах после recompute — раскладку
//         выбираем мы, transpose свободен.
//    [Q1] dO precision: FP16 (m16n8k16). НЕ квантуем в FP8 — нулевой
//         quantize-шум на A-стороне второй MMA. FP8 dO → B2.2.
//    [Q2] M_TILES_per_warp = 1 (ТЕМПОРАЛЬНО). По калибровке tensor-util
//         M_TILES / (M_TILES + 0.6): util = 1/(1+0.6) = 0.625 = util-killer
//         ×0.625 vs M_TILES=2 (forward production). На B2.2 вернём M_TILES≥2.
//         Сейчас выбран ради простоты per-warp ownership на этапе корректности.
//    [Q3] scale умножение — explicit __expf(scale * S − L), byte-for-byte
//         как в B2.0 baseline. Никакого фолдинга в Q pre-multiply.
//    [Q4] Маска применяется ДО smP_T write: P_elem = mask ? 0.0h : __expf(...).
//         Гарантирует ровно 0 (не -inf, не NaN, не мусор) в фрагментах MMA.
//    [Q5] dO upload через __float2half_rn per element. Cast FP32 → FP16 на хосте.
//    [Q6] Tolerance тугой, не мягкий: abs 1e-3 + rel 5e-3. Floor по теории:
//         FP8-quantize Q/K в recompute P ≈ 5e-3; dO FP16 добавляет ~1e-3.
//         Эмпирическая проверка:
//           проходит на 5e-3 → floor подтверждён.
//           не проходит у 5e-3 → фиксируй реальный FP8-floor с объяснением.
//           не проходит у 1e-1 → баг MMA-раскладки.
//         НЕ ставить добрый допуск «чтобы прошло» (93.3% lesson).
//    [PATH β SELECTED via probe_m16n8k16_fp16_layout.cu]:
//         m16n8k16.row.col.f32.f16.f16.f32 на sm_120a сходится bit-exact с CPU
//         для ОБОИХ раскладок B-операнда (smdO row-major + 2×LDS.U16/pack;
//         smdO_T row-major + 1×LDS.U32). Docs не врали. Это perf-tradeoff,
//         не correctness:
//           β платит ~4× LDS-bw на b-фрагмент в hot-loop.
//           α платит один full transpose pass на Q-tile.
//         β выбран на B2.1 ПО ПРОСТОТЕ РЕАЛИЗАЦИИ (один transpose, forward-стиль,
//         быстрее до корректного ядра), НЕ как доказанный оптимум.
//         β-vs-α по перфу — ОТКРЫТЫЙ ВОПРОС B2.2, решается NCu-замером
//         (LDS-bandwidth β vs transpose-cost α). НЕ считать β оптимальным
//         без замера.
//    [SWIZZLE smP_T — B2.2, НЕ СЕЙЧАС]:
//         Прямой урок forward-кладбища: bank-conflict-фикс дал −1.2% wall —
//         не на критическом пути. На B2.1 — простейшая smP_T раскладка
//         (row-major linear, stride 128B = 32 banks × 4B → структурный
//         32-bank conflict на write), без swizzle. Корректность сначала.
//         На B2.2 NCu покажет, на критическом ли пути конфликты, и нужен
//         ли swizzle вообще.
//
//  ЗАПРЕЩЕНО на B2.1: FP8 dO, dK, dQ, оптимизация скорости, бенчи.
//
// ---------------------------------------------------------------------
//  МАТЕМАТИКА (та же что в B2.0):
//      S[i, j]    = scale * Q[i] · K[j]
//      P[i, j]    = exp(S[i, j] - L[i])              ; mask 0 if causal/window
//      dV[j, d]   = Σ_i P[i, j] * dO[i, d]
//      scale      = 1 / sqrt(hd),   L = m + log(l)   (natural log)
// ---------------------------------------------------------------------
//
//  ГЕОМЕТРИЯ (block-per-K-tile, инвертирована относительно forward):
//
//      Bc = 64          — K/V-строк на блок (= forward Bc)
//      Br = 64          — Q-строк на Q-tile (итерируем по qt)
//      hd = 128         — фиксируем
//      threads = 128    — 4 warps × 32 lanes
//
//      Grid: (bh × n_kt),  n_kt = ceil(sl / Bc)
//      blockIdx.x = b * n_kt + kt
//
//      Блок владеет K-tile [kt*Bc .. kt*Bc + Bc), пробегает все qt ∈ [0, n_qt),
//      n_qt = ceil(sl / Br). dV[kt-slice, hd] аккумулируется в РЕГИСТРАХ
//      (FP32, как и baseline). Один write в gmem в конце.
//
//      Инверсия vs forward: forward iterates K (Q фиксирован per block),
//      backward dV iterates Q (K фиксирован per block).
//
// ---------------------------------------------------------------------
//
//  THREADING / WARP OWNERSHIP:
//
//      wid  = threadIdx.x / 32        — warp index ∈ [0, 4)
//      lane = threadIdx.x % 32
//      gid  = lane / 4                — sub-warp group ∈ [0, 8)
//      tid4 = lane % 4                — lane within group ∈ [0, 4)
//
//      Warp wid owns M-tile mi=wid of dV output (M_TILES_per_warp = 1 TEMP):
//        dV_acc[ni=0..15][4 FP32 per lane] in registers
//      Total dV_acc footprint per lane: 16 ni × 4 FP32 = 64 FP32 ≈ 64 regs.
//
//      Warp wid owns M-tile mi=wid of Q·K^T tile (Q-rows [wid*16 ..]):
//        Sr[ni=0..7][2 uint32 f16] per lane = 16 uint32 ≈ 16 regs.
//        M_TILES_QK_per_warp = 1, N_TILES_QK = Bc/8 = 8.
//
// ---------------------------------------------------------------------
//
//  SMEM LAYOUT (~40 KB total, под default 48 KB limit, без opt-in):
//
//      smK    [Bc=64 × hd=128] e4m3   =  8 192 B  — K-tile, once per block
//      smQ    [Br=64 × hd=128] e4m3   =  8 192 B  — текущий Q-tile
//      smdO   [Br=64 × hd=128] f16    = 16 384 B  — dO row-major (Q-tile)
//      smP_T  [Bc=64 × Br=64]  f16    =  8 192 B  — P stored transposed
//                                                   smP_T[j, i] = P[i, j]
//      smL    [Br=64]          f32    =    256 B  — L_i per Q-row
//      ----
//      Σ ≈ 41 168 B (≈ 40.2 KB). 48 KB default OK.
//
//      Swizzles переиспользуем из fa_bwd_common.cuh:
//        smK, smQ — swz_byte (hd=128 stride, FP8, как forward smK).
//        smdO — row-major linear, stride hd*2=256B (для baseline OK; на B2.2
//               добавим swizzle если bank conflicts заметны в NCu).
//        smP_T — row-major linear, stride Br*2=128B. Stride 128B = 32 banks
//               × 4B → структурный 32-bank conflict на write. Для baseline
//               OK; B2.2 swz_byte_bc-pattern уберёт.
//        smL — linear.
//
// ---------------------------------------------------------------------
//
//  HOT LOOP (per block, после single-shot загрузки smK):
//
//      // ---- WARMUP: K-tile cooperative load (FP8) ----
//      for e = tid; e < Bc*hd; e += 128:
//          int j_local = e / hd; int d = e % hd;
//          int j_g = kt*Bc + j_local;
//          smK[swz_byte(j_local, d)] = (j_g < sl) ? K_e4m3[b, j_g, d] : 0
//      __syncthreads()
//
//      // ---- INIT dV accumulators ----
//      float dV_acc[16][4] = {0}   // 16 ni × 4 FP32 per lane
//
//      // ---- OUTER LOOP over Q-tiles ----
//      for qt in 0..n_qt:
//          int qt_base = qt * Br
//
//          // step A: cooperative load Q (FP8), dO (FP16), L (FP32) → SMEM
//          //   smQ:  Br*hd = 8192 B / 128 thr = 64 B per thread
//          //   smdO: Br*hd*2 = 16384 B / 128 = 128 B per thread
//          //   smL:  Br*4 = 256 B (only first 64 lanes write 1 each)
//          //   All with bounds check (i_g >= sl → 0).
//          __syncthreads()
//
//          // step B: Q·K^T MMA (FP8 e4m3, m16n8k32 → f16 acc)
//          //   Warp wid owns M-tile mi=wid (Q-rows [qt_base + wid*16 .. +15]).
//          //   N-tiles ni ∈ [0, 8) (full Bc), ks ∈ [0, 4) (hd=128/32).
//          //   Sr[ni][2 uint32 f16] = Σ_ks MMA(Qr[ks], Kr[ni, ks], Sr)
//          //
//          //   Qr fragments: pre-loop manual uint32 loads из smQ (как forward Qr).
//          //   Lane (m_owned = wid*16 + lane/4 + 0/8, k_owned = (lane%4)*4 + ks*32).
//          //   Kr fragments: per-(ni, ks) manual uint32 loads из smK с hoisted swz.
//          //   Lane (n_owned = ni*8 + lane/4, k_owned = (lane%4)*4 + ks*32) — col view.
//          //   K row-major в smK с k-dim вдоль строки → 4 fp8 в k-pack adjacent в памяти.
//          //   mma_fp8_f16(Sr_p[0,1], Qr[0..3], Kr_p[0,1], Sr_p[0,1])
//
//          // step C: SOFTMAX per warp (in registers, FP16)
//          //   For each owned (i_local=0..15, j_local=0..63):
//          //     S_elem_f16 = unpack Sr[ni][n_subidx]  // FP16
//          //     i_g = qt_base + wid*16 + i_local
//          //     j_g = kt*Bc + j_local
//          //     mask = (i_g >= sl) || (j_g >= sl)
//          //         || (causal && j_g > i_g)
//          //         || (window > 0 && j_g < i_g + 1 - window)
//          //     // Q4: explicit 0 on mask BEFORE smP_T write
//          //     P_elem_h = mask ? __float2half(0.0f)
//          //                     : __float2half(__expf(scale * (float)S_elem_f16
//          //                                           - smL[i_local]))
//          //
//          //   // smP_T write — TRANSPOSED address: P^T[j_local, i_local]
//          //   smP_T[j_local * Br + i_local] = P_elem_h
//          //
//          //   Each lane in warp wid owns specific (i_local, j_local) cells per
//          //   Sr layout (m16n8k32: m=(lane/4)+0/+8, n=(lane/4)+0/+8 from k×ks idx,
//          //   detailed lane→cell mapping fixed by mma docs). Write addresses are
//          //   precomputable per-lane.
//
//          __syncthreads()  // smP_T fully written before P^T·dO reads
//
//          // step D: P^T·dO MMA (FP16, m16n8k16 → f32 acc)
//          //   Warp wid owns M-tile mi=wid of dV (j-rows [wid*16 .. wid*16+15]).
//          //   N-tiles ni ∈ [0, 16) (full hd=128).
//          //   K reduction: 4 k-batches (Br=64/16 = 4).
//          //
//          //   For each kb ∈ [0, 4):
//          //     // A-operand (P^T row-major из smP_T)
//          //     // Lane l holds (m=(l/4)+0/+8, k=(l%4)*2+0/+1 / +8/+9)
//          //     uint32_t Ar[4];
//          //     int m_lo = wid*16 + lane/4 + 0;
//          //     int m_hi = wid*16 + lane/4 + 8;
//          //     int k_lo = kb*16 + (lane%4)*2 + 0;
//          //     int k_hi = kb*16 + (lane%4)*2 + 8;
//          //     Ar[0] = LDS.U32 &smP_T[m_lo * Br + k_lo]   // (k, k+1) adjacent ✓
//          //     Ar[1] = LDS.U32 &smP_T[m_hi * Br + k_lo]
//          //     Ar[2] = LDS.U32 &smP_T[m_lo * Br + k_hi]
//          //     Ar[3] = LDS.U32 &smP_T[m_hi * Br + k_hi]
//          //
//          //     For ni in 0..15:
//          //       // B-operand (dO row-major в smdO, path β → 2× LDS.U16 + pack)
//          //       uint32_t Br_b[2];
//          //       int n = ni*8 + lane/4;
//          //       int k0 = kb*16 + (lane%4)*2 + 0;
//          //       int k1 = kb*16 + (lane%4)*2 + 8;
//          //       // dO[i=k, d=n] row-major: stride hd=128
//          //       uint16_t lo0 = LDS.U16 &smdO[k0 * hd + n];
//          //       uint16_t hi0 = LDS.U16 &smdO[(k0+1) * hd + n];
//          //       uint16_t lo1 = LDS.U16 &smdO[k1 * hd + n];
//          //       uint16_t hi1 = LDS.U16 &smdO[(k1+1) * hd + n];
//          //       Br_b[0] = (hi0 << 16) | lo0
//          //       Br_b[1] = (hi1 << 16) | lo1
//          //
//          //       mma_m16n8k16_f16_f32(dV_acc[ni][0..3], Ar, Br_b, dV_acc[ni][0..3])
//
//          __syncthreads()  // before next qt overwrites smQ/smdO/smP_T
//      end for
//
//      // ---- Final write: dV_acc → dV_global ----
//      //   Per-lane: dV[b, j_global, d_global] = dV_acc[ni][slot]
//      //     j_global = kt*Bc + wid*16 + (lane/4) + 0/+8   (slots 0,1 / 2,3)
//      //     d_global = ni*8 + (lane%4)*2 + 0/+1           (slots 0,2 / 1,3)
//      //   Boundary: skip if j_global >= sl.
//
// ---------------------------------------------------------------------
//
//  РЕГИСТРОВЫЙ БЮДЖЕТ (estimate):
//
//      dV_acc             : 16 ni × 4 FP32 = 64 regs
//      Sr (QK accum)      : 8 ni × 2 uint32 = 16 regs
//      Qr (pre-loaded)    : 4 ks × 4 uint32 = 16 regs
//      Kr/Pr/Br_b transient: ~8 per ks/kb
//      L cache (smL read) : ~2 regs
//      scratch (mask, exp, addr arith): ~25 regs
//      ----
//      Σ estimate: ~130 regs. ПОД 255 budget с большим запасом.
//      Spill: ожидаю 0. Если нет — фиксируем в ptxas + разбираем.
//
// ---------------------------------------------------------------------
//
//  TOLERANCE (per Q6, тугой):
//
//      abs 1e-3 + rel 5e-3 (sig 1e-2).
//
//      Floor по теории:
//        FP8-quantize Q/K в recompute P ≈ 5e-3
//        dO FP16 + FP32 acc → нigible ~1e-3 add
//        Сумма ~5e-3 worst rel above-floor.
//
//      Эмпирически:
//        PASS на 5e-3   → FP8-floor подтверждён.
//        FAIL на 5e-3   → фиксируем реальный floor с разбором.
//        FAIL на 1e-1   → баг MMA-раскладки.
//
// ---------------------------------------------------------------------
//
//  ВАЛИДАЦИЯ (тот же harness fa_bwd_dv_baseline_test.cu, обновлённый):
//
//      - 11 форм + canary
//      - GPU dV (MMA, FP32 out) vs:
//          - CPU FP32 baseline B2.0 (вторая опорная точка для localizing
//            MMA-bugs — она доказала math)
//          - FP64 golden (sanity)
//      - Tolerance abs 1e-3 + rel 5e-3
//      - multi-seed НЕ обязателен на этом шаге (multi-seed вернётся при
//        объединении с dK в B2.3+)
//
//  Приёмка B2.1: MMA-dV сходится с baseline и FP64-golden на 11 формах
//  в пределах FP8-quantize floor. ptxas чистый (spill — фиксируем если есть).
//  Скорость не мерить.
//
// ---------------------------------------------------------------------
//
//  ЭМПИРИЧЕСКИЙ FP8 FLOOR (зафиксирован после L-fix + FP16 diag):
//
//  L-fix bug: smL индекс не учитывал wid*16 offset → wid=1,2,3 использовали
//             L строк wid=0 → wrong P → contaminated smP_T → wrong dV.
//             Fixed: smL[wid*16 + l_div4 + 0/+8].
//
//  После L-fix:
//    ptxas: 128 regs, 0 spill, 0 stack, 1 barrier.
//    Floor (seed=42, FP8 recompute):
//      F1 non-causal sl=128       max_abs 3.6e-3
//      F2 causal sl=128           max_abs 2.96e-2  ← worst clust @i=0..2
//      F3 non-causal sl=256       max_abs 2.6e-3
//      F4 causal sl=256           max_abs 4.6e-2   ← worst clust @i=0..2
//      F5 non-causal sl=384       max_abs 2.2e-3
//      F6 causal sl=384           max_abs 3.9e-2
//      F7 non-causal sl=512 wnd   max_abs 3.4e-3
//      F8 causal sl=512 wnd       max_abs 2.96e-2
//      F9 non-causal sl=2048      max_abs 9.7e-4
//      F10 causal sl=2048         max_abs 2.9e-2
//      canary sl=300 wnd=96       max_abs 3.5e-3
//
//    Diagnostic FP16 recompute (fa_bwd_dv_mma_fp16.cu): same kernel logic,
//    Q,K uploaded as FP16 (no e4m3 quantize), Q·K^T via FP16 m16n8k16:
//      All forms PASS at tight tol abs 1e-3 + rel 5e-3.
//      Floor falls ~50× across all forms (causal F2: 6.7e-4, F9: 2.0e-5).
//      Worst-case clustering @i=0..2 SURVIVES in FP16 at 50× smaller scale —
//      это N_eff geometry effect (mask-induced weak averaging при causal
//      малых i), не структурный баг causal-ветки.
//
//  ИТОГ: FP8 e4m3 quantize gives ~6% per-element ε на Q,K. Q·K accumulation
//        +scale + exp pass-through → ~1-3% rel на P[i, j]. dV сумма усреднения
//        suppress √N_eff — но N_eff зависит от mask geometry:
//          strong (non-causal long sl): √N suppression → ~1e-3 abs floor
//          weak (causal small i):       few terms, ~5e-2 abs floor
//        Sign flips на |ref|~1e-2 дают max_rel_af до 100%+ — standard FP8
//        backward profile без post-quantize compensation.
//
//  Принято как production B2.1 с зафиксированным mask-geometry-dependent
//  floor. FP16 fallback для high-precision use cases — future B2.2 option.
// =====================================================================

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "fa_bwd_common.cuh"   // mma_fp8_f16, swz_byte, FA_BWD_STRIDE etc.

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

namespace fa_bwd_dv_mma {

// =====================================================================
// PTX m16n8k16.row.col.f32.f16.f16.f32  (FP16 MMA, FP32 accumulator).
// Lane layout (per PTX ISA, verified by probe_m16n8k16_fp16_layout.cu):
//   A (16×16 fp16, row-major): 4 uint32 per lane (4 half2)
//   B (16×8  fp16, col-major): 2 uint32 per lane (2 half2)
//   D (16×8  fp32):            4 fp32 per lane
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
// Kernel body.
// =====================================================================
__global__ void kernel_dv_mma(
    const uint8_t * __restrict__ Q,    // e4m3 [bh, sl, hd]
    const uint8_t * __restrict__ K,    // e4m3 [bh, sl, hd]
    const __half  * __restrict__ dO_g, // fp16 [bh, sl, hd]
    const float   * __restrict__ L,    // fp32 [bh, sl]
    float         * __restrict__ dV,   // fp32 [bh, sl, hd]
    int bh, int sl, int hd,
    int causal, int window,
    float scale)
{
    constexpr int Bc      = FA_DV_MMA_BC;       // 64
    constexpr int Br      = FA_DV_MMA_BR;       // 64
    constexpr int Hd      = FA_DV_MMA_HD;       // 128
    constexpr int NI_QK   = Bc / 8;             // 8 N-tiles for Q·K^T (Bc/8)
    constexpr int NI_DV   = Hd / 8;             // 16 N-tiles for P^T·dO (hd/8)
    constexpr int KS_QK   = Hd / 32;            // 4 ks-batches for FP8 m16n8k32
    constexpr int KB_DV   = Br / 16;            // 4 k-batches for FP16 m16n8k16

    const int tid    = threadIdx.x;
    const int wid    = tid >> 5;                // 0..3
    const int lane   = tid & 31;
    const int l_div4 = lane >> 2;               // 0..7
    const int l_mod4 = lane & 3;                // 0..3

    // Block ownership: (b, kt).
    const int n_kt = (sl + Bc - 1) / Bc;
    const int b    = blockIdx.x / n_kt;
    const int kt   = blockIdx.x % n_kt;
    if (b >= bh) return;

    // ---- SMEM layout (~40 KB) ----
    extern __shared__ uint8_t smem_raw[];
    uint8_t *smK  = smem_raw;                                  // Bc * Hd = 8 KB
    uint8_t *smQ  = smK  + Bc * Hd;                            //          8 KB
    __half  *smdO = reinterpret_cast<__half*>(smQ + Bc * Hd);  // Br * Hd * 2 = 16 KB
    __half  *smPT = reinterpret_cast<__half*>(
        reinterpret_cast<uint8_t*>(smdO) + Br * Hd * 2);       // Bc * Br * 2 = 8 KB
    float   *smL  = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(smPT) + Bc * Br * 2);       // Br * 4 = 256 B

    // ---- Warmup: K-tile load (FP8 e4m3, swz_byte addressing) ----
    {
        const uint8_t *Kb = K + b * sl * Hd;
        constexpr int total = Bc * Hd;
        for (int e = tid; e < total; e += FA_DV_MMA_THREADS) {
            int j_local = e / Hd;
            int d       = e % Hd;
            int j_g     = kt * Bc + j_local;
            uint8_t v   = (j_g < sl) ? Kb[j_g * Hd + d] : (uint8_t)0;
            smK[swz_byte(j_local, d)] = v;
        }
    }
    __syncthreads();

    // ---- dV accumulator (FP32, in registers) ----
    // Per-lane: NI_DV ni × 4 fp32 slots = 64 fp32.
    float dV_acc[NI_DV][4];
    #pragma unroll
    for (int ni = 0; ni < NI_DV; ++ni) {
        #pragma unroll
        for (int s = 0; s < 4; ++s) dV_acc[ni][s] = 0.0f;
    }

    // ---- OUTER LOOP over Q-tiles ----
    const int n_qt = (sl + Br - 1) / Br;
    for (int qt = 0; qt < n_qt; ++qt) {
        const int qt_base = qt * Br;

        // ===== step A: cooperative load Q (FP8), dO (FP16), L (FP32) → SMEM =====
        {
            const uint8_t *Qb = Q    + b * sl * Hd;
            const __half  *dB = dO_g + b * sl * Hd;
            // smQ: Br*Hd = 8192 bytes / 128 thr = 64 bytes per thread
            constexpr int total_Q = Br * Hd;
            for (int e = tid; e < total_Q; e += FA_DV_MMA_THREADS) {
                int i_local = e / Hd;
                int d       = e % Hd;
                int i_g     = qt_base + i_local;
                uint8_t v   = (i_g < sl) ? Qb[i_g * Hd + d] : (uint8_t)0;
                smQ[swz_byte(i_local, d)] = v;
            }
            // smdO: row-major linear (path β), stride Hd
            constexpr int total_dO = Br * Hd;
            for (int e = tid; e < total_dO; e += FA_DV_MMA_THREADS) {
                int i_local = e / Hd;
                int d       = e % Hd;
                int i_g     = qt_base + i_local;
                __half v    = (i_g < sl) ? dB[i_g * Hd + d] : __float2half(0.0f);
                smdO[i_local * Hd + d] = v;
            }
            // smL: Br fp32 entries
            if (tid < Br) {
                int i_g    = qt_base + tid;
                smL[tid]   = (i_g < sl) ? L[b * sl + i_g] : 0.0f;
            }
        }
        __syncthreads();

        // ===== step B: Q·K^T MMA (FP8 e4m3 m16n8k32 → f16 acc) =====
        //   Warp wid owns M-tile mi=wid (Q-rows wid*16 .. wid*16+15 of current Q-tile).
        //   N-tiles ni ∈ [0, NI_QK=8) cover Bc/8.
        //   ks ∈ [0, KS_QK=4) batches over hd/32.

        // Pre-load Qr fragments — A-operand (Q row-major in smQ, swz_byte addressed).
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

        // Sr accumulator (FP16 packed half2): NI_QK ni × 2 uint32 = 8 × 2 = 16 regs.
        uint32_t Sr[NI_QK][2];
        #pragma unroll
        for (int ni = 0; ni < NI_QK; ++ni) { Sr[ni][0] = 0u; Sr[ni][1] = 0u; }

        // Loop ks×ni: load Kr, accumulate Sr.
        #pragma unroll
        for (int ks = 0; ks < KS_QK; ++ks) {
            int k0_b = ks * 32 + l_mod4 * 4;
            int k_lo = k0_b + 0;
            int k_hi = k0_b + 16;
            #pragma unroll
            for (int ni = 0; ni < NI_QK; ++ni) {
                int n_K = ni * 8 + l_div4;  // K-row index within K-tile
                uint32_t Kr0 = *reinterpret_cast<uint32_t*>(&smK[swz_byte(n_K, k_lo)]);
                uint32_t Kr1 = *reinterpret_cast<uint32_t*>(&smK[swz_byte(n_K, k_hi)]);
                mma_fp8_f16(Sr[ni][0], Sr[ni][1],
                            Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3],
                            Kr0, Kr1,
                            Sr[ni][0], Sr[ni][1]);
            }
        }

        // ===== step C: SOFTMAX + smP_T write (per-lane owned cells) =====
        //   Lane l owns 4 cells per ni:
        //     (m_lo, n_lo), (m_lo, n_hi), (m_hi, n_lo), (m_hi, n_hi)
        //   where m_lo=l_div4+0, m_hi=l_div4+8 in warp's Q-rows
        //         n_lo=(l_mod4)*2+0, n_hi=(l_mod4)*2+1 within N-tile ni.
        //
        //   L cache: smL[i_local] for both m_lo and m_hi.
        //   i_local = wid*16 + l_div4 + 0/+8.
        const float L_lo = smL[wid * 16 + l_div4 + 0];
        const float L_hi = smL[wid * 16 + l_div4 + 8];
        const int   i_g_lo = qt_base + wid * 16 + l_div4 + 0;
        const int   i_g_hi = qt_base + wid * 16 + l_div4 + 8;
        const bool  i_lo_oob = (i_g_lo >= sl);
        const bool  i_hi_oob = (i_g_hi >= sl);

        #pragma unroll
        for (int ni = 0; ni < NI_QK; ++ni) {
            // Unpack Sr[ni] into 4 fp16 elements.
            __half2 s_lo_h2 = *reinterpret_cast<__half2*>(&Sr[ni][0]);  // (s[m_lo, n_lo], s[m_lo, n_hi])
            __half2 s_hi_h2 = *reinterpret_cast<__half2*>(&Sr[ni][1]);  // (s[m_hi, n_lo], s[m_hi, n_hi])

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

            // Mask helper:
            auto mask_chk = [&](int i_g, bool i_oob, int j_g, bool j_oob) -> bool {
                if (i_oob || j_oob)                       return true;
                if (causal && j_g > i_g)                  return true;
                if (window > 0 && j_g < i_g + 1 - window) return true;
                return false;
            };

            // Compute P for 4 cells.
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

            // smP_T transposed write: smP_T[j_local, i_local] = P[i_local, j_local]
            int i_local_lo = wid * 16 + l_div4 + 0;
            int i_local_hi = wid * 16 + l_div4 + 8;
            smPT[j_local_lo * Br + i_local_lo] = h_p00;
            smPT[j_local_hi * Br + i_local_lo] = h_p01;
            smPT[j_local_lo * Br + i_local_hi] = h_p10;
            smPT[j_local_hi * Br + i_local_hi] = h_p11;
        }

        __syncthreads();

        // ===== step D: P^T·dO MMA (FP16 m16n8k16 → f32 acc) =====
        //   Warp wid owns M-tile mi=wid of dV (j-rows in K-tile [wid*16 .. +15]).
        //   N-tiles ni ∈ [0, NI_DV=16) cover hd/8.
        //   K reduction: KB_DV=4 k-batches (Br/16 = 4).
        #pragma unroll
        for (int kb = 0; kb < KB_DV; ++kb) {
            // A-operand (P^T row-major from smP_T)
            // smP_T[j_local, i_local] addressed as smP_T[j_local * Br + i_local]
            // Lane holds:
            //   a0: m=(l/4)+0, k=(l%4)*2+0..1  (half2 adjacent)
            //   a1: m=(l/4)+8, k=(l%4)*2+0..1
            //   a2: m=(l/4)+0, k=(l%4)*2+8..9
            //   a3: m=(l/4)+8, k=(l%4)*2+8..9
            // m = j-position in warp's dV M-tile = wid*16 + (l/4) + 0/+8
            // k = i-position in current k-batch     = kb*16 + (l%4)*2 + 0/+8
            int m_lo = wid * 16 + l_div4 + 0;
            int m_hi = wid * 16 + l_div4 + 8;
            int k_lo = kb * 16 + l_mod4 * 2 + 0;
            int k_hi = kb * 16 + l_mod4 * 2 + 8;
#ifdef B21_NO_SMPT_READS
            (void)m_lo; (void)m_hi; (void)k_lo; (void)k_hi;
            uint32_t Ar0 = 0u, Ar1 = 0u, Ar2 = 0u, Ar3 = 0u;
#else
            uint32_t Ar0 = *reinterpret_cast<uint32_t*>(&smPT[m_lo * Br + k_lo]);
            uint32_t Ar1 = *reinterpret_cast<uint32_t*>(&smPT[m_hi * Br + k_lo]);
            uint32_t Ar2 = *reinterpret_cast<uint32_t*>(&smPT[m_lo * Br + k_hi]);
            uint32_t Ar3 = *reinterpret_cast<uint32_t*>(&smPT[m_hi * Br + k_hi]);
#endif

            #pragma unroll
            for (int ni = 0; ni < NI_DV; ++ni) {
                // B-operand (dO row-major in smdO, path β manual 2×LDS.U16 + pack)
                // Lane holds:
                //   b0: k=(l%4)*2+0..1, n=l/4  (half2 stride mismatch → 2 LDS + pack)
                //   b1: k=(l%4)*2+8..9, n=l/4
                // n = ni*8 + l/4 (d-position in current ni)
                int n   = ni * 8 + l_div4;
                int kA0 = kb * 16 + l_mod4 * 2 + 0;
                int kA1 = kb * 16 + l_mod4 * 2 + 1;
                int kB0 = kb * 16 + l_mod4 * 2 + 8;
                int kB1 = kb * 16 + l_mod4 * 2 + 9;

#ifdef B21_NO_SMDO_READS
                (void)kA0; (void)kA1; (void)kB0; (void)kB1; (void)n;
                uint32_t Br0 = 0u, Br1 = 0u;
#else
                uint16_t lo0 = *reinterpret_cast<uint16_t*>(&smdO[kA0 * Hd + n]);
                uint16_t hi0 = *reinterpret_cast<uint16_t*>(&smdO[kA1 * Hd + n]);
                uint16_t lo1 = *reinterpret_cast<uint16_t*>(&smdO[kB0 * Hd + n]);
                uint16_t hi1 = *reinterpret_cast<uint16_t*>(&smdO[kB1 * Hd + n]);
                uint32_t Br0 = ((uint32_t)hi0 << 16) | (uint32_t)lo0;
                uint32_t Br1 = ((uint32_t)hi1 << 16) | (uint32_t)lo1;
#endif

                mma_m16n8k16_f16_f32(
                    dV_acc[ni][0], dV_acc[ni][1], dV_acc[ni][2], dV_acc[ni][3],
                    Ar0, Ar1, Ar2, Ar3,
                    Br0, Br1,
                    dV_acc[ni][0], dV_acc[ni][1], dV_acc[ni][2], dV_acc[ni][3]);
            }
        }

        __syncthreads();  // before next qt overwrites smQ/smdO/smPT
    }

    // ---- Final write dV[b, j_g, d_g] = dV_acc[ni][slot], FP32 ----
    //   dV_acc[ni][slot] lane layout:
    //     slot 0: m=(l/4)+0, n=(l%4)*2+0  →  j_local_lo, d_lo
    //     slot 1: m=(l/4)+0, n=(l%4)*2+1  →  j_local_lo, d_hi
    //     slot 2: m=(l/4)+8, n=(l%4)*2+0  →  j_local_hi, d_lo
    //     slot 3: m=(l/4)+8, n=(l%4)*2+1  →  j_local_hi, d_hi
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
        fprintf(stderr, "fa_bwd_dv_mma: hd=%d, expected %d\n", hd, FA_DV_MMA_HD);
        exit(1);
    }
    const int Bc   = FA_DV_MMA_BC;
    const int Br   = FA_DV_MMA_BR;
    const int n_kt = (sl + Bc - 1) / Bc;
    const int grid = bh * n_kt;
    const int smem_bytes =
        Bc * hd * sizeof(uint8_t)     // smK   (FP8)
      + Br * hd * sizeof(uint8_t)     // smQ   (FP8)
      + Br * hd * sizeof(__half)      // smdO  (FP16)
      + Bc * Br * sizeof(__half)      // smPT  (FP16)
      + Br      * sizeof(float);      // smL   (FP32)

    kernel_dv_mma<<<grid, FA_DV_MMA_THREADS, smem_bytes, stream>>>(
        Q, K, dO_g, L, dV, bh, sl, hd, causal, window, scale);
}

} // namespace fa_bwd_dv_mma
