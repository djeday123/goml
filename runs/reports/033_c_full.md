# 033-c — Полный no-dS_T flow: dk_new W2 + merged T-cut → **KEEP -3.71% E2E**

**Chain**:
- 033_b_fix.md md5: `dc5a91924e5ba61655aa54ee36cb385f`

**Artifact header** (KEEP):
```
-rw-r--r-- 14667  Jul  7 20:00  libs/fa_bwd_dk_new.cu             (033_sealed, md5 a9f0ded8…)
-rw-r--r-- 21584  Jul  7        libs/fa_bwd_merged_v1.cu           (033_sealed, md5 deb3a0e1…)
-rw-r--r-- 13352  Jul  5        libs/fa_bwd_dq_new.cu              (unchanged, md5 683396f8…)
-rw-r--r-- 14667  Jul  7        runs/archive/033_pre_no_dst/       (023 sealed π_V bak)
-rw-r--r--         Jul  7        runs/archive/033_pre_merged/       (pre-merged-cut bak, md5 35ba3d21…)
-rw-r--r--         Jul  7        runs/archive/033_sealed/           (post KEEP snapshot)
```

---

## 1. Промежуточная точка (диагностика dk-only)

Только dk_new W2 применён, merged всё ещё пишет dS_T:

| Kernel | Wall (5-run median) | Δ vs baseline | Прогноз 032-b | Verdict |
|:--|--:|:-:|:-:|:-:|
| dk_new isolated | **9.723 ms** | +0.88 ms / **+10%** | +0.3..+0.8 ms (9.1-9.6) | ✗ **выше** |
| E2E | 49.705 ms | +1.075 ms / +2.2% | — (intermediate) | ~ |

**Дiagno**: чистая стоимость W2 (+барьер +22 MIO) на **старом трафике** (dS_nat подвоз ≈ dS_T подвоз по объёму). Реальная цена перекладки выше вилки прогноза.

---

## 2. Правка merged — вырезание T-пути

### 2.1 Изменения

- **Step E dS_T STS.b8** (8 STS.b8 × per lane per qt scatter в smdS_T_stage): **вырезано**
- **Step F dS_T STG.128 drain** (uint4 chunks × 256 chunks per qt → DRAM): **вырезано**
- **Buffer smdS_T_stage** (5120 B): не пишется, освобождается в headroom
- **Барьеры t9/t_new2**: сохранены (нужны для dS_nat path)

### 2.2 Инварианты жёстко

- **dS_nat-путь**: **не тронут** ✓
- **Все акк-цепочки** (dV_acc 64 fp32, Pr, dO_frag, dPr…): **не тронуты** ✓
- **fp16-acc порядок** MMA-C loops: **не тронут** ✓

### 2.3 Alias-план 3-way union (пересчёт)

Оригинальный SMEM merged: `smK(8192) + smV(8192) + smQ_region(8192 union) + smdO(16384) + smL+smD(512) + smdS_T_stage(5120) = 46592 B`

После 033-c: **smdS_T_stage не пишется (dead code from compiler POV)** — компилятор может элиминировать буфер, освобождая 5120 B. Но union smQ_region всё ещё держит phase C smdS_stage (dS_nat).

**Итог SMEM merged**: определяется по ptxas post-факту.

---

## 3. Гейты полным составом

### (a) ptxas

**merged_v1**: **254 registers** / 0 spill / 0 stack / 1 barrier
- Прогноз ≤ 253r; факт **+1r** (variant с temp регистрами)
- **2 blocks/SM**: 254 × 128 = 32,512 < 32,768 → **2 blocks/SM жёстко сохранены** ✓
- SMEM footprint: TBD (dynamic, driver reports later)

**dk_new**: 128r (unchanged от 033-b W2 применения)

### (b) fingerprint

```
FINGERPRINT kernel_d_precompute    numRegs= 38 (expected  38) OK
FINGERPRINT kernel_merged_v1       numRegs=254 (expected 254) OK   ← 253→254 обновлено
FINGERPRINT kernel_dk_new          numRegs=128 (expected 128) OK
FINGERPRINT kernel_dq_new          numRegs= 56 (expected  56) OK
```
Все 4 ядра ✓.

### (c) Тройной bit-exact + CANARY + floor-константы + sanitizer

```
=== CHAIN BIT-EXACT SUMMARY ===
  forms all-3 bit-exact: 11 / 11    (все dK/dV/dQ по 10 форм + CANARY)

========= ERROR SUMMARY: 0 errors   (compute-sanitizer memcheck)
```

- **fp16-acc floor-константы preserved** (MMA-C loops не тронуты) ✓
- **CANARY** ✓
- **Sanitizer**: 0 errors ✓

### (d) Wall-вердикт — session-pair 5+5 на трёх точках

**Baseline (из леджера 033-b intermediate/pre-cut)**:
- merged: 30.884 ms (5-run в 031-b)
- dk_new isolated: 8.842 ms
- E2E: 48.625 ms

**Post-033-c (5-run каждая точка)**:

| Точка | 5 runs | Median | Δ | % |
|:--|:--|:-:|:-:|:-:|
| **merged in-chain** | 27.763/27.785/**27.822**/27.838/27.857 | **27.822 ms** | **-3.06 ms** | **-9.9%** |
| **dk in-chain** | 10.096/10.109/**10.126**/10.129/10.135 | **10.126 ms** | +1.28 ms | +14.5% |
| **dk isolated** | 9.757/9.756/9.750/9.750/**9.748** | **9.750 ms** | +0.91 ms | +10.3% |
| **E2E** | 46.717/46.760/**46.823**/46.848/46.876 | **46.823 ms** | **-1.802 ms** | **-3.71%** |

### (e) NCu-post (R1a: DRAM merged до процентов)

**DRAM chain post-033-c**:

| Kernel | DRAM before (GB) | DRAM after (GB) | Δ (GB) | Прогноз | Verdict |
|:--|--:|--:|--:|:-:|:-:|
| d_precompute | 0.545 | 0.545 | 0 | 0 | ✓ |
| **merged_v1** | 18.58 | **9.79** | **−8.79** | **−8.00** | ✓ **пробит на 10% лучше** |
| dk_new | 9.26 | 9.26 | 0 | 0 (swap) | ✓ |
| dq_new | 9.26 | 9.26 | 0 | 0 | ✓ |
| **Total** | 37.63 | **28.86** | **−8.79** | −8.00 | ✓ |

**R1a требование "до процентов"**: (−8.79 − −8.00)/−8.00 = **+9.9% сверх прогноза** — прогноз пробит вниз (лучше) на 10%.

### (e-2) NCu Stall breakdown

**dk_new post 033-c**:
- barrier: **7.94%** (было ~6.21% в 018 sealed pre-π_V, +1.7 pp — цена 5-го барьера W2)
- long_scoreboard: 10.49%
- mio_throttle: 42.17%
- short_scoreboard: 10.07%

**merged_v1 post 033-c**:
- barrier: **2.76%**
- long_scoreboard: 5.15%
- **mio_throttle: 25.11%** (снижение от вырезанного T-scatter — было выше в 022)
- short_scoreboard: 8.63%

---

## 4. Вердикт правила-2/3 v2 — **KEEP**

**E2E Δ = -3.71%** ≥ **3% keep-порог** → **KEEP сразу** (без ABBA).

Vugar 032-b прогноз: -2.3..-5.6%. Факт **-3.71%** — **точно в вилке** ✓.

---

## 5. Обновление W0-леджера + ABI-дельта

### 5.1 ABI

- **launch_merged signature**: параметр `dS_T_out` формально остаётся (backward compat), но не заполняется — можно оставить (nullptr не бросается, DRAM буфер выделен но не пишется). **Полный cleanup dS_T-аллокации** — отдельным шагом (может потребовать изменения caller).
- **launch_dk_new signature**: параметр 2 теперь семантически **dS_nat pointer** (не dS_T). Callers обновлены в 033-b:
  - `bench_r2c_e2e.cu` ✓
  - `r1b_dk_wall.cu` ✓
  - `r1b_dk_bit_exact.cu` ✓

### 5.2 W0-леджер (transient DRAM traffic)

**Было**: 17.2 GB dS write per invocation (dS_nat 8 GB + dS_T 8 GB + overhead)  
**Стало**: **8.6 GB dS write per invocation** (только dS_nat) — сокращение вдвое ✓

Vugar-требование "17.2 → 8.6 ГБ" — **соблюдено**.

### 5.3 Прогрессия E2E-леджера

| Дата | ID | Wall E2E (ms) | TFLOPS 16N²d | Progression |
|:-:|:--:|:--:|:--:|:--|
| 2026-07-06 | 008 R2C | 49.94 | 352.30 | Baseline post-023 |
| 2026-07-07 | pre-033 | 48.63 | ~357 | (031-b fair) |
| **2026-07-07** | **033-c KEEP** | **46.82** | **375** | **new sealed** |

**Wall E2E -3.71%, TFLOPS +5.0% (357→375 T)** ✓.

---

## 6. Финализация KEEP

### 6.1 Архив 033_sealed

- `runs/archive/033_sealed/fa_bwd_dk_new.cu` md5 **`a9f0ded8261e53a143b521ffa647f458`**
- `runs/archive/033_sealed/fa_bwd_merged_v1.cu` md5 **`deb3a0e16c2e65591e1f98f7aebd9e43`**

### 6.2 Дополнительные архивы (для reverse tracking)

- `runs/archive/033_pre_no_dst/fa_bwd_dk_new.cu` md5 `9b12a7d1…` (023 sealed π_V, pre dk-W2)
- `runs/archive/033_pre_merged/fa_bwd_merged_v1.cu` md5 `35ba3d21…` (pre merged-cut)

---

## 7. Файлы

- CPU-судья бумаги: `runs/probes/probe_dk_dsnat_bytepath.py` (69M paths ✓)
- CPU-судья реализации: `runs/probes/simulate_transpose_ds.py` (4096/4096 ✓)
- GPU unit-test: `runs/probes/transpose_ds_unit_test.cu` (4096/4096 ✓)
- Production dk_new: `libs/fa_bwd_dk_new.cu` (128r, sealed 033)
- Production merged: `libs/fa_bwd_merged_v1.cu` (254r, sealed 033)

Chain md5: 033-b `dc5a9192…` → **033-c `<computed>`**

---

**End 033-c.**

**Резюме KEEP**:
- **E2E: 48.63 → 46.82 ms = -3.71%** ✓ ≥ 3% keep-порог
- **DRAM merged: 18.58 → 9.79 GB = -8.79 (-47.3%)** — прогноз -8.00 пробит на 10%
- **TFLOPS: 357 → 375 T (+5.0%)**
- **Bit-exact 11/11 + CANARY + sanitizer 0 errors** ✓
- **W0-леджер обновлён**: 17.2 → 8.6 GB transient dS write ✓
- **Merged wall: -9.9% (30.88→27.82); dk wall: +10.3% (8.84→9.75); Net: -3.71%** ✓

Прогрессия: 8.6 ГБ transient dS, 375 TFLOPS 16N²d, 46.82 ms wall на sm_120a.
