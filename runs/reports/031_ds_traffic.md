# 031 — dS-байты по цепи R2C: баланс, полы, веер кандидатов (0 GPU-правок)

**Chain**:
- 030_dq_root.md md5: `33cd15205b25cb4d007a1d377de19298`

**Artifact header** (production не тронут):
```
-rw-r--r-- 13352  Jul  5 17:51  libs/fa_bwd_dq_new.cu    (sealed pre-pack, md5 683396f8…)
-rw-r--r-- 14667  Jul  7 10:03  libs/fa_bwd_dk_new.cu    (023 sealed π_V, md5 9b12a7d1…)
-rwxr-xr-x 2.26M  Jul  7        libs/bench_r2c_e2e       (используется для NCu-срезов)
```

Только measurement + paper.

---

## 1. Баланс-таблица dS-байтов по цепи

### 1.1 Форма dS

`dS[bh=128, sl_i=8192, sl_j=8192]` FP8 = **8 GB** per tensor (dS_nat и dS_T каждый).

ABI-padded stride_ds = (sl+15)&~15 = 8192 (уже aligned на sl=8192, padding=0).

### 1.2 DRAM per kernel (NCu freshsource, session 2026-07-07)

| Kernel | DRAM (GB) | wall NCu (ms) | throughput (GB/s) | BW-utilization |
|:--|--:|--:|--:|:-:|
| d_precompute | 0.545 | 0.335 | 1630 | **91%** (near-peak) |
| **merged_v1** | **18.58** | **45.49** | **408** | **22.8%** ← низкая |
| dk_new | 9.26 | 12.07 | 767 | 42.9% |
| dq_new | 9.26 | 10.96 | 845 | 47.2% |
| **Total** | **37.63** | 68.85 | avg 546 | — |

Peak DRAM bw Blackwell RTX PRO 6000 = **1.79 TB/s**.

### 1.3 DRAM-пол каждого ядра (BW-limited теоретический)

| Kernel | DRAM (GB) | **Floor** (ms) = DRAM/1.79TB/s | Actual (ms) | **Actual/Floor** ratio | Запас над полом |
|:--|--:|--:|--:|:-:|--:|
| d_precompute | 0.545 | 0.305 | 0.335 | 1.10× | 0.03 ms (10%) |
| **merged_v1** | 18.58 | 10.38 | 45.49 | **4.38×** | 35.1 ms (338%) |
| dk_new | 9.26 | 5.17 | 12.07 | 2.34× | 6.90 ms (134%) |
| dq_new | 9.26 | 5.17 | 10.96 | 2.12× | 5.79 ms (112%) |

**Вывод**:
- **merged_v1**: 4.38× выше пола = **338% headroom**, гигантский potentially reducible
- dk/dq: 2.1-2.3× floor, средний headroom
- d_precompute: near-peak (91%), уже почти BW-bound

### 1.4 Data footprint по ролям

**Merged writes** (dominant DRAM producer):
- dV: 512 MB (128×8192×128 × 4B FP32)
- **dS_nat**: **8 GB** (128×8192×8192 × 1B)
- **dS_T**: **8 GB** (128×8192×8192 × 1B — дубликат nat, transposed)
- Q, K, V read: 128 MB × 3 = 384 MB
- dO read: 256 MB (FP16)
- L, D read: 4 MB each

**dk_new reads** (dominant DRAM consumer):
- dS_T: 8 GB
- Q: 128 MB
- К (cached возможно): 128 MB
- **Total ~8.5 GB read**, writes dK 512 MB → ~9 GB match NCu 9.26

**dq_new reads**:
- dS_nat: 8 GB
- K: 128 MB
- **Total ~8.5 GB read**, writes dQ 512 MB → ~9 GB match NCu 9.26

### 1.5 dS-related DRAM traffic

- merged writes dS_nat + dS_T = **16 GB**
- dk_new reads dS_T = **8 GB**
- dq_new reads dS_nat = **8 GB**
- **Total dS DRAM traffic = 32 GB per E2E invocation** (85% of chain total 37.63 GB)

Floor 32 / 1.79 = **17.9 ms just for dS transfer** if всё было BW-bound.

---

## 2. Веер кандидатов (5-10 строк каждый)

### (a) Отказ от dS_T — dk читает dS_nat с транспонированием на лету

**Экономия ГБ по ядрам**:
- merged: не пишет dS_T → -8 GB DRAM write
- dk_new: reads dS_nat вместо dS_T → same 8 GB read
- dq_new: unchanged
- **Total dS DRAM: 32 → 24 GB (-25%)**

**Новые полы**:
- merged floor: 10.38 → 5.90 ms (-43%)
- dk_new floor: 5.17 (unchanged, тот же 8 GB read)
- dq_new floor: 5.17 (unchanged)

**Ожидаемое реальное сокращение** (при сохранении ratio 4.38×):
- merged wall: 45.49 → 26 ms (-19.5 ms) ← топ-1 выигрыш
- dk_new wall: +transposition overhead (0.5-2 ms cost), возможно net +0.5-1.5 ms
- **E2E: -18-22 ms = -35-45%** ← гигантский potential

**Риск**:
- **Bit-exact**: **сохраняется** — байты dS не меняются ни на бит, только кто транспонирует
- **ABI**: слом — dS_T buffer аллокация снимается, W0-ledger обновление; но R2C internal ABI, non-user-facing
- **Регистровая цена dk_new**: сейчас 124r/128 ceiling — тесно. Транспонирование = ~28 MIO-ops/qt/lane (pack-analog), возможно 130-140r → **выход за потолок 128** возможен
- **Время**: 3-5 дней (unit-test транспонирования + правка dk + правка merged + гейты + E2E)

**Кто выигрывает**: merged (-19 ms), почти чистый gain.

### (b) Tile-handoff через L2 (малые буферы без полной материализации)

**Экономия**:
- dS not materialized in DRAM at all — sits in L2 cache
- **Total dS DRAM: 32 → ~0 GB** if L2 residency maintained

**Новые полы**:
- Все ядра: DRAM полы упадут drastically (только Q/K/V/dQ/dK/dV/dO/D)

**Ожидаемое реальное сокращение**:
- merged wall: 45 → 20-30 ms (unpredictable, depends L2 residency)
- E2E: -15-25 ms (при удачной cache-стратегии)

**Риск**:
- L2 size ~140 MB (Blackwell). dS_nat 8 GB / 140 MB = **57 tile-passes** minimum → tile-batching overhead
- **Cache eviction unpredictable** — depends на активность других ядер и других kernels в GPU
- Требует **cross-kernel координации** (persistent kernel? cooperative groups? tensor memory accelerator? TMA?)
- **Bit-exact**: сохраняется (byte-identical), но timing-sensitive → возможно недетерминированные результаты
- **ABI**: полностью переработан — inter-kernel communication нужно перепроектировать
- **Время**: **2-3 недели** (сложная координация)

**Кто выигрывает**: все, но с большим риском.

### (c) FP4/E2M1-квант dS с per-tile scale

**Экономия**:
- FP8 → FP4: **0.5 byte per element**
- dS_nat: 8 → 4 GB, dS_T: 8 → 4 GB
- **Total dS DRAM: 32 → 16 GB (-50%)**

**Новые полы**:
- merged floor: 10.38 → 6.4 ms (-38%)
- dk_new floor: 5.17 → 3.2 ms (-38%)
- dq_new floor: 5.17 → 3.2 ms (-38%)

**Риск**:
- **Bit-exact НАРУШЕН** — FP4 quantization теряет 4 бита mantissa ✗
- Vugar-инвариант bit-exact **не соблюдён** → **невозможно применить**

**Кто выигрывает**: все, но **исключён** правилом bit-exact.

### (d) Собственный: dS_T inline compute — не хранить, вычислять в dk_new из dS_nat

Аналог (a), но без промежуточной материализации:
- merged пишет только dS_nat
- dk_new **вычисляет dS_T "на лету"** из dS_nat через smart LDS pattern (без явного транспонирования в отдельном SMEM буфере)

**Экономия**: аналогично (a): -8 GB DRAM.

**Отличие от (a)**: dk_new не строит smdS_T в SMEM — читает dS_nat напрямую в MMA-C с транспонированной раскладкой lane-per-thread. Экономит SMEM 5120 B в dk_new.

**Риск**:
- Более сложная реализация чем (a)
- SMEM саving позволит **потенциально снять регистровое давление** через дополнительные warps на SM
- Bit-exact preserved (bytes idennnтичны)
- **Время**: сопоставимо (a) + smem redesign

**Кто выигрывает**: merged + возможно dk_new (SMEM winner).

### (e) Fusion merged→dk_new/dq_new (single-kernel super-fused)

Полный fusion — one kernel does merged + dk + dq inline.
- **dS не материализуется** вообще
- Экономия: **32 GB dS DRAM → 0**
- Требует единый гигантский kernel с very high register pressure и SMEM
- **Bit-exact**: сохраняется
- **Time**: 3-4 недели, complexity очень высокая
- **Registers**: 350+r, spillage likely, occupancy drop
- **Riski**: очень высокий, unstable outcome

---

## 3. Рекомендация: **(a) отказ от dS_T**

**Обоснование**:
1. **Максимальный wall-выигрыш** при минимальном риске: **-35-45% E2E** (потенциал)
2. **Bit-exact сохраняется автоматически** (bytes идентичны, только transposition location меняется)
3. **ABI-слом ограниченный** — только W0-ledger dS_T аллокация снимается
4. **Единичное риск-ядро**: dk_new (тесное регистровое окно 124/128)
5. **Прогноз реален**: merged 22.8% BW-utilization → есть много запаса для DRAM-drop conversion в wall
6. **Vugar predavtorized** при MIO-add ≤ 30/qt/lane в dk_new

**Не рекомендую**:
- **(b)** — слишком рискованно (L2 residency)
- **(c)** — нарушает bit-exact
- **(d)** — эквивалентно (a) по DRAM, дополнительная сложность SMEM redesign без явного gain
- **(e)** — слишком масштабный refactor

---

## 4. Vugar-условие для (a) → фаза 032

**Автоматический переход к 032 если**:
- MIO-add в dk_new ≤ 30/qt/lane (SASS-бумага)
- Нет ABI-слома пользовательского уровня (только internal R2C chain — да, OK)

**Иначе стоп-доклад с вердиктом 031, выбор за Vugar** (может пойти на (b) или отложить).

Мой предварительный прогноз для (a):
- **MIO-add в dk_new**: 
  - Реализация: pack-analog для dS транспонирования = 12 SHFL + 16 STS.32 + 12 PRMT = **28 MIO ops/qt/lane** (аналог dk_new pack Q_T scatter)
  - **Под порогом 30 ≤ 30** ✓ 
  - Возможно также π-класс для bank optimization, +5-8 SEL — но всё еще под 30

Пред-verdict: **проходит порог**, переход к 032 auto-authorized.

Регистровая вилка dk_new при добавлении transpose staging:
- Base: 124r (sealed π_V)
- Add transpose registers: ~5-10r (kr-analog + G/V + OUT for dS)
- **Прогноз: 130-135r** — **выше 128 potollок occupancy** (blocks/SM 4 → 3)
- **Возможен регресс 4→3 blocks/SM** — требует явного решения Vugar

Это **новый риск** для (a). Регистры dk_new тесные. Возможно нужен smaller transpose (fewer SHFL + STS.32) или SMEM double-buffer trick.

---

## 5. Файлы

- NCu DRAM chain script: `runs/reports/031_dram_chain.sh`
- Существующие 3 бинаря dq для сравнения: `libs/r1c_dq_A/B/C_*`

Chain md5: 030 `33cd1520…` → **031 `<computed>`**

---

**End 031.**

**Рекомендация**: (a) отказ от dS_T. Vugar auto-authorize к 032 при MIO-check прошёл (мой прогноз 28 ≤ 30 ✓).  
**Новый риск для (a)**: dk_new 124r ceiling — транспонирование staging может выселить блоки (4→3). Обсуждение с Vugar до 032 желательно.
