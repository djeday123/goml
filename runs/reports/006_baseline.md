# 006-0 — ds_gen baseline + D-checklist + NCu diff (2026-07-05)

## ARTIFACT-HEADER

```
-rw-r--r-- 1 root root       -  Jul 5 18:15  runs/reports/006_baseline_wall.log       (5-run ds_gen)
-rw-r--r-- 1 root root       -  Jul 5 18:15  runs/reports/006_ds_gen_ncu.log          (full section)
-rw-r--r-- 1 root root       -  Jul 5 18:15  runs/reports/006_ds_gen_ncu_detail.log   (stall metrics)
-rw-r--r-- 1 root root       -  Jul 5 18:15  runs/reports/006_dq_sealed_ncu.log       (sealed dQ compare)
```

## 006-0.1 — Свежий wall baseline ds_gen dual-write (в этой сессии)

`bench_ds_gen bh=128 sl=8192 hd=128 causal=0, warmup=5 iters=20`:
```
run 1: 36.722 ms   run 2: 36.731   run 3: 36.760   run 4: 36.789   run 5: 36.781
median: 36.760 ms   TF_2mma: 119.64   CV: 0.09%
```

Δ vs 003 (36.68 ms) = **+0.08 ms (тот же класс, thermal noise)**. Padded-ABI не сдвинул wall на канонической форме — stride_ds==sl=8192 → ноль в арифметике.

**159r → 154r после ABI (-5 регов)**: замена повторной `sl` (double-multiply в address computation, использовалась одновременно как размер и как stride) на выведенное `stride_ds` дала ptxas'у одну независимую scalar-переменную для hoist-инга `i*stride_ds` из горячего цикла. Освободились ~5 регов инвариант-редукция.

## 006-0.2 — D-checklist по коду ds_gen

| Пункт | Место в sealed AA1 dQ | Место в ds_gen | Статус |
|-------|-----------------------|----------------|:-:|
| (а) smK cp.async swizzle `col_byte ^ ((j_local & 7) << 4)` | fa_bwd_dq.cu:256 | fa_bwd_ds_gen.cu:264 | ✅ УНАСЛЕДОВАНО |
| (а) smV cp.async swizzle | fa_bwd_dq.cu:260 | fa_bwd_ds_gen.cu:268 | ✅ УНАСЛЕДОВАНО |
| (а) smdO element-XOR `(i_local & 7) << 4` | fa_bwd_dq.cu:204 | fa_bwd_ds_gen.cu:209 | ✅ УНАСЛЕДОВАНО |
| (а) MMA-A smK reads `k_lo ^ k_xor`, `k_xor = l_div4 << 4` | fa_bwd_dq.cu:292 | fa_bwd_ds_gen.cu:300 | ✅ УНАСЛЕДОВАНО |
| (а) smV MMA-B reads swizzled | fa_bwd_dq.cu:389 | fa_bwd_ds_gen.cu:397 | ✅ УНАСЛЕДОВАНО |
| (а) smdO MMA-B reads swizzled | fa_bwd_dq.cu:380-383 | fa_bwd_ds_gen.cu:388-391 | ✅ УНАСЛЕДОВАНО |
| (б) softmax jam-x2 `for np < NI_QK/2` | fa_bwd_dq.cu:312 | fa_bwd_ds_gen.cu:320 | ✅ УНАСЛЕДОВАНО |
| (б) dS-quantize/scatter jam-x2 `for np < NI_DP/2` | fa_bwd_dq.cu:406 | fa_bwd_ds_gen.cu:414 | ✅ УНАСЛЕДОВАНО |
| **(в) STG dS_nat vectorization** | (n/a — sealed пишет dQ f32, coalesced) | ❌ **байт-по-байту** (16 × STG.b8 per lane per jam-pair, lines 483-514) | **ПРОБЕЛ #1** |
| **(в) STG dS_T** | (n/a) | ❌ **байт-по-байту non-coalesced** (idx `j*stride_ds+i`, `j` внешний → каждая write в РАЗНУЮ CL) | **ПРОБЕЛ #2** → **option B (006-II)** |
| (г) cp.async pipeline: warmup dO + per-kt K/V, cpa_commit/wait<0> | fa_bwd_dq.cu:214-215, 264-265 | fa_bwd_ds_gen.cu:219-220, 272-273 | ✅ УНАСЛЕДОВАНО (один паттерн) |

**Итог чеклиста**: 8/10 пунктов унаследованы. Два пробела в STG dS_nat/dS_T — оба относятся к output layer, не тронуты в sealed (у sealed нет materialization dS).

## 006-0.3 — NCu diff ds_gen vs sealed dQ

Условия: `bh=128 sl=8192 hd=128 causal=0`, single kernel isolated.

| Metric | ds_gen | sealed dQ | Δ / verdict |
|--------|:------:|:--------:|:-----------|
| Wall (bench, 20-iter avg) | **36.76 ms** | **19.44 ms** | +17.32 ms Δ — надо снять |
| Duration (NCu profile, cache warm) | 43.73 ms | 30.74 ms | +13.0 ms |
| Registers | 154 | 161 | ≈ |
| Achieved Occupancy | 16.59 % (7.96 warps/SM) | ~16.6 % проекция | ≈ |
| **long_scoreboard** | **30.02 %** | **5.96 %** | **×5.0 хуже** — LSU/LDG pipe stuck |
| mio_throttle | 11.71 % | 16.71 % | ds_gen лучше |
| short_scoreboard | 7.35 % | 8.33 % | ≈ |
| barrier | 2.54 % | 2.93 % | ≈ |
| lg_throttle | 0.13 % | 0.02 % | ≈ (low) |
| Bank conflicts LD | 110.8 M | 605.8 M | ds_gen **лучше** (свизлы работают одинаково) |
| Bank conflicts ST | 4 K | 6.1 M | ds_gen negligible (нет smdS scatter) |
| shared_ld inst | 1.98 B | 2.72 B | ds_gen меньше (нет MMA-C read) |
| shared_st inst | 65 K | 671 M | ds_gen ×10000 меньше (нет smdS scatter) |
| L1/TEX Hit Rate | 68.04 % | (n/a here) | |
| L2 Hit Rate | 95.69 % | (n/a here) | K/V reuse ok |
| Max Bandwidth | **93.96 %** | | mem-bound |
| L1/TEX Cache Throughput | **82.62 %** | | high LSU pipe pressure |
| DRAM Throughput | 24.40 % | | нижний уровень |

### Диагноз

**Bank conflicts НЕ виноваты** — их у ds_gen меньше чем у sealed dQ (110M vs 606M). Свизлы работают идентично.

**Драйвер +17 ms = long_scoreboard 30% vs 6% в sealed** = LSU pipe stuck на STG dS_nat + dS_T per-byte.

Механика:
- Per lane per jam-x2 pair: 8 STG.b8 (4 для dS_nat + 4 для dS_T) — line 483-514
- NI_DP/2 = 4 пар → 32 STG.b8 per lane per kt
- 128 threads × 128 n_kt × 16384 blocks = **8.4 B STG.b8 per launch** = ~262 M warp-level STG insts
- Non-vectorized STG.b8 требует 1 LSU transaction each → LSU насыщена → LDG следующих K/V/dO блокирует long_scoreboard

**Sealed dQ пишет dQ_acc через STG.b32 coalesced** (line 559-565: `dQb[i*Hd + d] = f32 * scale`), 4-байтные vectorized stores → LSU не bottleneck.

## Что делаем в 006-I

**Пробел #1 (в) — STG dS_nat vectorization**:
- Опция (b16): pack ja_lo+ja_hi bytes в u16, STG.b16 per row per lane per jam. 8 STG.b8 → 4 STG.b16 (×2 меньше LSU transactions). Простая, no SMEM.
- Опция (SMEM+STG.128): STS byte-scatter в smdS (Br*Bc=4096B, паттерн = sealed AA1) → barrier → STG.128 coalesced. 32 STG.b8 → 2 STG.128 per lane per kt (×16 меньше LSU transactions, но + STS + барьер). Sealed AA1 УЖЕ делает STS-scatter — паттерн проверен.

Выбор для 006-I: **STG.b32 через SMEM staging** (агрессивно, максимум эффекта LSU relief). SMEM 33792 + 4096 = 37888 B → 2 блока (unchanged occupancy).

Пробел #2 (в') — STG dS_T non-coalesced — отдельно в **006-II** (option B: STS-scatter в smdS_T → coalesced STG.128 построчно).

## Пороки (baseline перед фиксом)

- Wall ds_gen dual-write: **36.76 ms** (session-fresh, CV 0.09%)
- Регистры: 154r/0s
- Occupancy: 2 блока/SM = 7.96 warps/SM (SMEM-limited 33.79 KB)
- long_scoreboard: 30.02 % (главный target 006-I)
- ptxas 154r/0s/0 stack — spill-free база

Готов к 006-I (D-фикс STG dS_nat через SMEM staging). Wait for ACK перед правкой.
