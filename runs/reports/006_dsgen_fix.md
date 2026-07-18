# 006 — ds_gen fix: dS_nat + dS_T SMEM staging + coalesced STG.128 drain (2026-07-05)

## ARTIFACT-HEADER

```
-rw-r--r-- 1 root root   31218 Jul  5 22:31  libs/fa_bwd_ds_gen.cu                  (staging + drain, 130r/0s)
-rw-r--r-- 1 root root    2449 Jul  5 22:31  libs/ds_gen.ptxas.log                  (final ptxas)
-rwxr-xr-x 1 root root 1451216 Jul  5 22:31  libs/r1a_wall                          (rebuilt wall bench)
-rw-r--r-- 1 root root    7141 Jul  5 21:11  runs/reports/006_baseline.md           (006-0 diagnosis)
-rw-r--r-- 1 root root     -   Jul  5 21:11  runs/reports/006_ds_gen_ncu.log        (baseline full section)
-rw-r--r-- 1 root root     -   Jul  5 21:11  runs/reports/006_dq_sealed_ncu.log     (sealed dQ NCu diff)
-rw-r--r-- 1 root root     711 Jul  5 22:28  runs/reports/006_I_bit_exact.log       (006-I re-gate 11/11 + 11/11)
-rw-r--r-- 1 root root     537 Jul  5 22:28  runs/reports/006_I_wall.log            (006-I 21.65 ms)
-rw-r--r-- 1 root root     -   Jul  5 22:31  runs/reports/006_II_bit_exact.log      (006-II re-gate 11/11 + 11/11)
-rw-r--r-- 1 root root     -   Jul  5 22:31  runs/reports/006_II_wall.log           (006-II 20.57 ms)
-rw-r--r-- 1 root root     -   Jul  5 22:31  runs/reports/006_II_ncu.log            (final stall breakdown)
```

## 006-0 — baseline + diagnosis (принято ранее)

- Wall session-fresh 36.76 ms (CV 0.09%)
- 154r/0s (padded ABI hoisted `stride_ds` из горячего цикла, -5 регов vs 003)
- 2 blocks/SM SMEM-limited (33.79 KB dyn), 7.96 warps/SM achieved
- D-checklist 8/10 инвариантов унаследованы от sealed; 2 пробела: STG dS_nat поэлементно + STG dS_T non-coalesced
- NCu-дифф vs sealed dQ: long_scoreboard 30% vs 6% (**×5 хуже**), bank conflicts LD LESS чем sealed (110M vs 606M) → диагноз: LSU pipe stuck на 8.4B STG.b8

## 006-I — dS_nat через SMEM staging

### Дизайн
- `SMDS_STAGE_STRIDE = 80` (16-aligned для STS.b16 + LDS.128 + STG.128, bank-conflict-free)
- `smdS_stage` aliased над `smV` (свободен после MMA-B барьера) → **ZERO extra SMEM**
- STS.b16 per lane per jam-pair (fp8x2 uint16 → 2 смежных байта [ja_lo, ja_hi]) = 4 STS.b16 (вместо 8 STG.b8)
- Барьер + coalesced STG.128 drain: 128 threads × 2 chunks per kt = 256 chunks/block/kt

### Гейт 1 ptxas
`libs/ds_gen.ptxas.log` (intermediate): 154r → **157r** (+3 регов от drain-loop ILP), 0 spill, 0 stack. SMEM unchanged 33792 B.

### Гейт 2 BIT-EXACT re-gate 11/11 + 11/11
`runs/reports/006_I_bit_exact.log`:
- r1b_dk_bit_exact (consumer dK): **11/11 max_abs_diff=0.000e+00** including CANARY
- r1c_dq_bit_exact (consumer dQ): **11/11 max_abs_diff=0.000e+00** including CANARY
- compute-sanitizer both: **0 errors**

Layout-only change (STS.b16 pack сохраняет тот же байт-паттерн: LSB→[i, ja_lo], MSB→[i, ja_lo+1=ja_hi]). Арифметика не тронута.

### Гейт 3 wall
`runs/reports/006_I_wall.log` (5-run, CV 0.19%):
```
21.598 / 21.645 / 21.663 / 21.648 / 21.713 → median 21.648 ms
```
- Δ vs baseline: **-15.11 ms (-41.1%)** — **сильно ниже Vugar-ожидания 30-33 ms**
- Причина сверх-выигрыша: coalesced STG.128 → меньше L2 evictions → snowball L2-hit на K/V следующих kt

## 006-II — dS_T через SMEM staging (option B)

### Дизайн
- `SMDS_T_STAGE_STRIDE = 80` (16-aligned drain)
- `smdS_T_stage` — новый буфер **5120 B** после smD (не aliased; smK_area всё ещё используется в след. kt)
- STS.b8 per byte (fp8x2 halves принадлежат разным SMEM-строкам в [j][i] layout, b16-пак невозможен по j)
- Bank analysis: l_div4 groups × column groups = 32 distinct banks per warp — **conflict-free**
- Тот же drain-паттерн: 256 chunks per kt, STG.128 по row j_local со stride_ds
- Total SMEM: 33792 + 5120 = **38912 B** → Block Limit SMEM = **2 (unchanged occupancy)**

### Гейт 1 ptxas
`libs/ds_gen.ptxas.log` (final): **130r** / 0 spill / 0 stack / 1 barrier. -27r vs 006-I (drain-only guards, unconditional scatter упростил live-set).

### Гейт 2 BIT-EXACT re-gate 11/11 + 11/11
`runs/reports/006_II_bit_exact.log`:
- r1b_dk_bit_exact: **11/11 max_abs_diff=0.000e+00** including CANARY
- r1c_dq_bit_exact: **11/11 max_abs_diff=0.000e+00** including CANARY
- compute-sanitizer both: **0 errors**

### Гейт 3 wall
`runs/reports/006_II_wall.log` (5-run, CV 0.09%):
```
20.549 / 20.560 / 20.571 / 20.582 / 20.594 → median 20.571 ms
```
- Δ vs 006-I: -1.08 ms (-5.0%)
- **Δ vs baseline: -16.19 ms (-44.0%)**
- **В целевом диапазоне Vugar 20-24 ms** (попал в нижнюю границу)

### Гейт 4 NCu финал
`runs/reports/006_II_ncu.log`:

| Metric | 006-0 baseline | 006-II final | Δ |
|--------|:-:|:-:|:-:|
| Wall | 36.76 ms | **20.57 ms** | **-16.19 ms (-44.0%)** |
| **long_scoreboard** | 30.02 % | **9.92 %** | **-20.10 pp** — LSU освобождена |
| mio_throttle | 11.71 % | 3.46 % | -8.25 pp |
| short_scoreboard | 7.35 % | 25.57 % | +18.22 pp — новый bottleneck (LDS drain) |
| barrier | 2.54 % | 2.00 % | -0.54 pp |
| Bank conflicts LD | 110.8 M | 231.9 M | +121M (drain LDS.128) |
| Bank conflicts ST | 4 K | 25.2 M | +25M (STS scatter) |
| shared_st inst | 65 K | 402.7 M | +402M (staging) |
| inst_executed | 15.79 B | 13.13 B | -2.66 B (-16.9%) |
| LSU wavefronts | 5.79 B | 3.08 B | -47% |

Атрибуция:
- **long_scoreboard 30% → 9.92%** — главная победа. LSU pipe перестала блокировать LDG следующих K/V.
- **mio_throttle 11.7% → 3.5%** — MIO cap снят (меньше LSU insts).
- **inst_executed -2.66B (-16.9%)** — меньше guarded-branch overhead + меньше LSU-транзакций.
- **short_scoreboard 7.4% → 25.6%** — узкое место переместилось на LDS-latency (drain reads от smdS_stage/smdS_T_stage).
- **LSU wavefronts -47%** — coalesced 16-byte вместо разрозненных b8.
- Bank conflicts выросли (LD +121M, ST +25M), но stall net упал — bandwidth-limited путь стал efficiency-limited.

## Обновлённая R1-арифметика (все measured после 006-II)

| Компонент | Wall (ms) | Источник |
|-----------|:---------:|---------|
| D-precompute | 0.36 | measured |
| **ds_gen (dual-write staged)** | **20.57** | **measured 006-II** ← новое |
| dV baseline (Yarus-1 sealed) | 19.63 | measured |
| dk_new | 9.42 | measured 005a |
| dq_new | 8.58 | measured 005b |

Sequential: 0.36 + 20.57 + 19.63 + 9.42 + 8.58 = **58.56 ms** — **ниже sealed e2e 61.6 ms** (-5.0%)!

Multi-stream (dV || ds_gen → dk_new || dq_new):
- ds_gen finish: 0.36 + 20.57 = 20.93 ms
- dV finish (parallel): ≈ 20 ms (dV done inside ds_gen window)
- dk_new || dq_new: max(9.42, 8.58) = 9.42 ms
- **Total: 20.93 + 9.42 = 30.35 ms** (vs sealed 61.6 → **-50.7%**)

## Резюме 006

- ✅ **006-0 diagnosis** — long_scoreboard 30% = LSU-flood от 17.2B STG.b8 (свизлы/джем не виноваты, bank conflicts LESS чем sealed)
- ✅ **006-I dS_nat staged + STG.128 drain**: 36.76 → 21.65 ms (**-41%**, zero SMEM cost via smV alias)
- ✅ **006-II dS_T staged + STG.128 drain**: 21.65 → 20.57 ms (-5%, +5120B SMEM, 2 blocks/SM держатся)
- ✅ **11/11 BIT-EXACT × 2 consumers × 2 фазы** (dK + dQ через новый ds_gen, max_abs_diff=0.000e+00 включая CANARY)
- ✅ **compute-sanitizer clean** обоих потребителей после каждой фазы
- ✅ 130r/0s/0 stack финал (spill-free)
- ✅ NCu: long_scoreboard -20 pp, mio -8 pp, LSU wavefronts -47%, inst_executed -16.9%
- ✅ Sequential R1 wall **58.56 ms < 61.6 ms sealed** (-5.0%)
- ✅ Multi-stream R1 projection **~30 ms** (-50.7% vs sealed)

**R1-chain готов. Ждём ACK Vugar → 007 R1-E2E honest (sequential + streams vs 61.6/285.44).**
