# 002 — R1a reconcile (2026-07-05)

## ARTIFACT-HEADER

```
-rw-r--r--    1235 Jul 5 11:23  runs/reports/002_r1a_canonical_wall.log   (5-run canonical wall)
-rw-r--r--    4332 Jul 5 11:23  runs/reports/002_r1a_dram_ncu.csv         (explicit dram__bytes.sum)
-rwxr-xr-x     622 Jul 5 11:23  runs/reports/r1a_dram_ncu.sh              (NCu script)
```

## 1. Точная форма wall-прогона (canonical bench-form)

`bench_ds_gen` r1a_wall с явными args:
```
bh=128 sl=8192 hd=128 causal=0 warmup=5 iters=20
FINGERPRINT kernel_ds_gen: numRegs=159, sharedSizeBytes=0, localSizeBytes=0, maxThreadsPerBlock=384
```
Соответствует канонической форме bench_dq/bench_e2e. **F9 sl=2048 в предыдущем отчёте — bit-exact test scope, НЕ wall (wall был на bh=128 sl=8192 всё время)**.

5-run на канонической: 25.861 / 25.875 / 25.877 / 25.893 / 25.901 → **median 25.877 ms** (CV<0.05%).

## 2. Bytes / DRAM reconcile

### Expected bytes per launch:
| Buffer | Size | Notes |
|--------|:----:|-------|
| dS write | 128 × 8192 × 8192 × 1 = **8.590 GB** | e4m3 output |
| Q read | 128 × 8192 × 128 = 128 MB | FP8 |
| K read | 128 × 8192 × 128 = 128 MB | FP8 (cp.async per kt-tile) |
| V read | 128 × 8192 × 128 = 128 MB | FP8 |
| dO read | 128 × 8192 × 128 × 2 = 256 MB | FP16 |
| L read | 128 × 8192 × 4 = 4 MB | FP32 |
| D read | 128 × 8192 × 4 = 4 MB | FP32 |
| **Total expected** | **~9.24 GB** | |

### NCu measured (canonical form, 1 launch after skip 5):
```
dram__bytes.sum                         9,290,198,016 byte  =  9.29 GB
dram__throughput.avg.pct_of_peak        15.29 %
lts__t_bytes.sum                        91,228,330,144 byte = 91.23 GB
lts__t_sector_hit_rate.pct              96.84 %
l1tex__t_bytes.sum                     103,624,474,624 byte = 103.62 GB
l1tex__throughput.avg                   68.05 %
```

**Expected 9.24 GB vs NCu measured 9.29 GB — расхождение 0.5%** (в шуме измерения). Байты СОШЛИСЬ.

### Ratio wall vs DRAM:
- Wall 25.877 ms × 359 GB/s = 9.29 GB → эмпирический DRAM throughput = **359 GB/s per launch**
- NCu SOL 15.29% × peak 1.79 TB/s = 273 GB/s (SOL method includes idle wave alignment)
- Разница NCu SOL vs empirical: NCu профилирует одиночный launch с sync-latency, wall timing amortizes через 20 iters. Не противоречие.

### Ключевой вывод
- **dS materialization = 9.29 GB DRAM/launch на канонической форме**
- **DRAM SOL 15.29%** (baseline 1% → **×15.3**) — memory всё ещё изобилен для R1-chain
- L2 traffic 91.23 GB — dS temporally reused в L2 96 MB между dS-gen tiles (i-tile stays hot для нескольких j-tiles)
- L2 hit 96.84% подтверждает cache-friendly паттерн доступа

## 3. Wall canonical median для E2E-бюджета R1

**25.877 ms** — dS-gen компонент R1-цепочки. R1-полный wall:
```
R1 = dS-gen + dV_baseline + dK-new + dQ-new
   = 25.877 + 19.63 + [~11 dK-new] + [~11 dQ-new]
   = ~67 ms projection
```
(vs текущий baseline 61.6 ms — **немного хуже без merged fuse**)

R0-прогноз R1 wall = 35-45 ms. Реальный dS-gen 25.87 ms — выше R0-DRAM-only projection (14.4 ms). Compute-portion не учтена в R0-projection. R1-projection пересматривается вверх до ~55-65 ms.

## 4. SMEM ds_gen (одной строкой)

**SMEM ds_gen = 33792 B > 33024 B (Y0 3-block порог) → 2 blocks/SM**. Резерв 768 B до 3-block. **НЕ строить (полировочный резерв, зафиксировано)**.

## Итог reconcile

- ✅ Форма подтверждена: канонические bh=128 sl=8192 hd=128 nc
- ✅ Bytes СОШЛИСЬ: expected 9.24 GB vs measured 9.29 GB (Δ 0.5%)
- ✅ Canonical wall = 25.877 ms median (входит в E2E-бюджет R1)
- ✅ NCu на канонической: DRAM 15.29% SOL, L2 hit 96.84% — аксиома "запись бесплатна на sm_120a backward" держится
- ✅ SMEM 33792 B > 33024 → 2 blocks/SM, резерв 768 B зафиксирован

## Открытые вопросы для R1b

R1-projection **55-65 ms** > R0-projection 35-45 ms. Причины:
1. dS-gen 25.87 ms (compute-portion + STG > DRAM-only forecast)
2. dK-new и dQ-new проекции нужны из R1b/c реальных измерений

R1b начинается по готовности. Vugar's ТЗ R1b:
1. Dual-layout write в ds_gen (dS_nat + dS_T) — re-gate bit-exact ОБОИХ
2. Wall ds_gen с dual-write (+8.59 GB → ожидание +2-4 ms)
3. kernel_dk_new: cp.async dS_T-tile → MMA dS^T·Q → dK_acc FP32 (fp16 dead per AB2)
4. Гейт bit-exact vs sealed dK эталонов
5. ptxas ~90-110r/0s, wall isolated, NCu
6. Отчёт runs/reports/003_R1b.md
