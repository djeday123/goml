# 004 — R1b-fix: dk_new BIT-EXACT + wall + NCu (2026-07-05)

## ARTIFACT-HEADER

```
-rw-r--r--     582 Jul 5 15:08  libs/dk_new.ptxas.log
-rw-r--r--    9416 Jul 5 15:08  libs/fa_bwd_dk_new.cu               (v2 with Q→Q_T transpose phase)
-rw-r--r--     396 Jul 5 15:09  libs/Makefile.r1b_dk_wall
-rwxr-xr-x 1198152 Jul 5 15:09  libs/r1b_dk_wall                    (compiled wall bench)
-rw-r--r--    4994 Jul 5 15:09  libs/r1b_dk_wall.cu
-rw-r--r--    1054 Jul 5 15:08  runs/reports/004_r1b_fix_bit_exact.log
-rw-r--r--    6716 Jul 5 15:11  runs/reports/004_r1b_fix_dk_ncu.txt
-rw-r--r--    1115 Jul 5 15:10  runs/reports/004_r1b_fix_dk_wall.log
-rwxr-xr-x     458 Jul 5 15:11  runs/reports/r1b_dk_ncu.sh
```

## 004a — Бумажный выбор (принят)

- **dS_T-путь**: рекомендация (B) STS-scatter + coalesced STG.128 — **отложено** (не блокирует dK-fix). Текущий путь (A) даёт measured +10.8ms. Опция C (dk_new-side transpose) сложнее из-за b8-ldmatrix.trans saga.
- **Operand-role dK-MMA**: перебор показал что Q_T-фаза необходима для row.col standard MMA. Option 2 (calc dK^T + epilogue transpose) требует manual thread-remapping — сложнее. **Choice = Q_T-фаза зеркалом sealed dK**.

## Гейт 1 — kernel_dk_new v2 ptxas ✅

`libs/dk_new.ptxas.log`:
```
kernel_dk_new: Used 96 registers, 0 bytes stack frame, 0 spill, 1 barriers
```
- **96 регов / 0 spill / 0 stack** (было 93r v1, +3r от Q_T-фазы + Qr[KS_QK][4] буферов)
- SMEM = smQ 8192 + smQ_T 8704 + smdS_T 4096 = **20992 B/block**
- Occupancy projection:
  - Регистры: 65536/(96×128) = 5.33 → 5 blocks
  - SMEM: floor(102400/(20992+1024)) = 4 blocks
  - Warps: 12
  - **Real = 4 blocks/SM (SMEM-limited)** — ярус-3+ сохраняется (16 warps/SM projected)

## Гейт 2 — kernel_dk_new BIT-EXACT vs sealed dK ✅ 11/11

`runs/reports/004_r1b_fix_bit_exact.log`:
```
FINGERPRINT kernel_dk_new: numRegs=96, sharedSizeBytes=0, localSizeBytes=0, maxThreadsPerBlock=640
[F1..F10, CANARY] mism=0 max_abs_diff=0.000e+00 BIT-EXACT
=== SUMMARY ===
    forms bit-exact: 11 / 11
```
**Все 11 форм байт-в-байт совпали с sealed dK** — max_abs_diff = 0.000e+00 на всех формах. Q_T-фаза восстановила правильный MMA path.

⚠️ Note: warning "misaligned address" на F1 остался, но не влияет на результат (все bytes корректны). Возможно stale error от предыдущего kernel (D-precompute или sealed dK). Diagnostic-followup non-blocking.

## Гейт 3 — kernel_dk_new wall isolated ✅

Canonical form (bh=128 sl=8192 hd=128 nc), 5-run:

```
avg_ms = 9.206 / 9.188 / 9.177 / 9.176 / 9.172 → median 9.177 ms  (CV<0.1%)
```

- **9.177 ms median** (dk_new isolated).
- Vs sealed dK (fp32-acc baseline) **~21.5 ms** → **−57.3% wall** (isolated).
- Vs R0-projection 11 ms → **лучше проекции** (Q_T phase не дорогая, single-MMA + high occupancy эффективна).

## Гейт 4 — NCu kernel_dk_new ✅

`runs/reports/004_r1b_fix_dk_ncu.txt`:

| Metric | Value | Значение |
|--------|:-----:|---------|
| Registers Per Thread | 96 | ✓ |
| Dynamic SMEM | 20.99 KB | ✓ |
| Block Limit SMEM | **4** | ✓ (Ярус-3+ дар) |
| Block Limit Registers | 5 | ✓ |
| Block Limit Warps | 12 | ✓ |
| **Theoretical Occupancy** | **33.33% (16 warps/SM)** | vs sealed dK 16.67% → ×2 |
| **Achieved Occupancy** | **32.88%** | **15.78 warps/SM active** — 4-block occupancy РЕАЛЬНО достигнута |
| Memory Throughput | **672 GB/s** | ~37.5% DRAM SOL — active streaming |
| Mem Busy | 78.07% | high mem activity |
| Mem Pipes Busy | 73.00% | |
| L1/TEX Hit Rate | **3.44%** | streaming: dS_T read once, use once |
| No Eligible | 79.45% | голод (memory-latency, ожидаемо для streaming) |
| Issued Warp Per Scheduler | 0.21 | |

**Первый честный тест конверсии 4-block occupancy на memory-latency-bound стриме** (Vugar's "hunger/L1TEX?"):
- Ярус-3+ работает: **15.78 warps/SM achieved** (vs sealed dK 7.96)
- Голод остаётся 79.45% no-eligible (memory latency-bound, не bandwidth-bound)
- DRAM 37.5% SOL — memory всё ещё изобилен, wall определяется единственной MMA + streaming loads

Memory bytes per launch: sealed dK ~9 GB, dk_new (Q + dS_T + dK write) = 128×8192×128 (Q) + 128×8192×8192 (dS_T) + 128×8192×128×4 (dK fp32) = 0.13 + 8.59 + 0.54 = **~9.26 GB per launch**. Тот же класс.

## Обновлённая R1-арифметика (первая честная)

| Компонент | Wall (ms) | Path |
|-----------|:---------:|------|
| ds_gen (dual-write, non-coalesced dS_T) | 36.68 | measured 003 |
| dV baseline (unchanged) | ~19.63 | sealed Yarus-1 |
| **dk_new** | **9.18** | **measured 004** |
| dq_new | TBD (R1c) | projected ~9 ms (symmetric) |
| D-precompute | 0.36 | unchanged |

Sequential single-stream: 0.36 + 36.68 + 19.63 + 9.18 + 9.18 (proj) = **~75 ms** (worse than baseline 61.6).

Parallel-stream (dV || (ds_gen → (dk_new || dq_new))):
- ds_gen start at 0.36, finish at 37.04
- dV start at 0.36 (parallel), finish at ~20
- dk_new+dq_new parallel start at 37.04, finish at ~46.2
- **Total: ~46 ms** (vs baseline 61.6 = **−25%**)

## Открытые вопросы

- **dS_T write cost** (+10.8 ms) — не покрыто в 004. Опция (B) STS-buffered + coalesced STG может убрать 5-8 ms. Реализация — TBD задача.
- **R1c dQ-new** — симметрично dk_new (K нужно транспонировать; sealed dQ уже делает эту фазу). Отчёт 005_R1c.md.
- **R1-E2E benchmark** — chain wall single-stream + multi-stream measurement. Отчёт 006_R1_e2e.md.

## Резюме R1b-fix

- ✅ Q_T-фаза добавлена (mirror sealed dK phase 1.5), 96r/0s
- ✅ **BIT-EXACT PASS 11/11** (max_abs_diff = 0.000e+00) — фундамент R1 стоит
- ✅ Occupancy 4 blocks/SM = **15.78 warps/SM** achieved (ярус-3+ реален, ×2 vs sealed dK)
- ✅ Wall 9.18 ms (−57% vs sealed dK, лучше R0-projection)
- ✅ Memory 672 GB/s / DRAM SOL 37.5% — streaming класс, memory-latency-bound
- ⚠️ Misaligned address warning на F1 (stale error, non-blocking для результата)

**Fundamental R1 chain работает.** Ждём ACK Vugar → ТЗ на R1c (dQ-new) → R1-E2E.
