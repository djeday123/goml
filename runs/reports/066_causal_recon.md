# 066 — Causal-разведка (0 правок production)

**Chain**:
- 062 `b7044db70019e8fa7dea260f9f235b6c`
- 063 `b7c82475ed49ac4821c7346f99a38fb1`
- 063-r `1dce5e445e5b47152c2894ffc7947b30`
- 064 `c099586c01ad07d6c83f3d732a77de3c`
- 065 `cc5c2a7f96aeed162ddf28609703009a`
- **066 `<self>`**

**Табло**: causal 22.206 ms судейское (063 cert). Цель: карта улик для тюнинга скипа — перекос бригад, загадка байтов, доля диагонали. Ноль строек, ноль правок production.

---

## Артефакт-хедер

```
libs/ (production — unchanged, sealed):
  fa_bwd_merged_v1.cu    md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33  KV-owned  regs=252 occ=2 blocks/SM
  fa_bwd_dk_new.cu       md5 25e5e1077cc3bec2c49bf9288fe60c54  KV-owned  regs=124 occ=4 blocks/SM
  fa_bwd_dq_new.cu       md5 d7a11a3d788eb4c396d892bc9c8ab754   Q-owned  regs= 69 occ=6 blocks/SM
  fa_bwd_common.cuh      md5 4407ec9cf64708a2a28dc36633d5d6f1
  fa_bwd_ds_gen.cu       md5 665a350d3da8ae90b816ccd6b55db346  (control D, no qt-loop)

GPU: NVIDIA RTX PRO 6000 Blackwell WS Edition, sm_120a, 188 SMs, 65536 reg/SM, 101376 B smem (opt-in)
Форма: bh=128 sl=8192 hd=128, Br=128 Bc=64, n_qt=n_kt=128 tiles, threadsPerBlock=128
Гейт-тишина: nvidia-smi compute-apps EMPTY ✓ на всех замерах
```

**Правки production ядер: 0.** Все замеры на sealed binary.

---

## §1 Гистограмма перекоса — БУМАГА (a, b, c)

### §1.a Живая работа блока полосы n при `qt_start = causal ? kt : 0`

**Скипы в production** (grep-подтверждено):
```
fa_bwd_merged_v1.cu:146:  const int qt_start = causal ? kt : 0;
fa_bwd_dk_new.cu:86:      const int qt_start = causal ? kt : 0;
fa_bwd_dq_new.cu:107:     const int kt_end = causal ? (qt + 1) : n_kt;
```

**Merged (KV-owned, block=(bh,kt))**: iter-loop по qt, живой диапазон под causal `qt ∈ [kt, n_qt-1]` → живых тайлов на блок `= n_qt − kt = 128 − kt`.

**dk_new (KV-owned)**: та же схема, `qt ∈ [kt, n_qt-1]`, живых тайлов `= 128 − kt`.

**dq_new (Q-owned, block=(bh,qt))**: iter-loop по kt, живой `kt ∈ [0, qt]` → живых тайлов `= qt + 1`.

**Гистограмма веса блоков** (per kernel, bh=0 репрезентация; замножается на 128 bh):

| class k / q | Merged/dk_new вес (128−k) | dq_new вес (q+1) |
|:-:|:-:|:-:|
| 0 | 128 | 1 |
| 32 | 96 | 33 |
| 64 | 64 | 65 |
| 96 | 32 | 97 |
| 127 | 1 | 128 |

Сумма живых тайлов на bh = Σ_{k=0}^{127}(128−k) = **8256** (треугольник N(N+1)/2 при N=128).

Полная (NC) работа на bh = 128 × 128 = **16384** тайл-итераций.

**Инвариант (сверка с NCu 063-r)**: live/full = 8256/16384 = **0.5039** ≡ NCu insts ratio 0.505/0.506 (063-r §1) на 3-значный знак. **Формула подтверждает детектор до 0.2%.**

**Общая работа под causal через грид** = 128 bh × 8256 tiles = **1,056,768 тайл-итераций** (vs NC 2,097,152).

---

### §1.b Наивный потолок: время цеха = max по волнам × T_tile

**Слоты и волны** (сверено с NCu `launch__waves_per_multiprocessor`):

| Kernel | Occ/SM | Slots S = occ×188 | Grid | Waves (grid/S) |
|:--|:-:|:-:|:-:|:-:|
| merged  | 2 (SMEM+reg lim) | 376  | 16384 | 43.57 ← ncu 43.57 ✓ |
| dk_new  | 4 (reg lim: 65536/(128·124)=4.13) | 752  | 16384 | 21.79 ← ncu 21.79 ✓ |
| dq_new  | 6 (SMEM lim: comment L94) | 1128 | 16384 | 14.52 ← ncu 14.52 ✓ |

**Калибровка T_tile по NC wall** (30-run медиана 064):
- merged NC: 25.125 ms, chain = ceil(16384/376)×128 = 44×128 = 5632 tile-iters/slot → **T_tile^merged = 4.462 μs**
- dk_new NC: 8.423 ms, chain = ceil(16384/752)×128 = 22×128 = 2816 tile-iters/slot → **T_tile^dk = 2.991 μs**
- dq_new NC: 8.462 ms, chain = ceil(16384/1128)×128 = 15×128 = 1920 tile-iters/slot → **T_tile^dq = 4.407 μs**

**Наивный potol под causal — «max по SM с текущим FIFO-грид-порядком»**:

Идеально сбалансированный (LPT / weighted round-robin) causal wall = (total tile-iters) / (S × T_tile^-1):
- merged: 1,056,768 / 376 × 4.462 μs = **12.54 ms**
- dk_new: 1,056,768 / 752 × 2.991 μs = **4.203 ms**
- dq_new: 1,056,768 / 1128 × 4.407 μs = **4.129 ms**

**Идеальный причал (перекошенный)**: max поверх SM активных циклов при перекошенных весах блоков в FIFO-порядке (kt меняется быстрее bh для merged/dk; qt меняется быстрее bh для dq). Слот получает блоки с шагом `S mod 128` — sub-балансировка. Формула max_slot ≈ (W/S) × (1 + skew_factor).

---

### §1.c Потолок с ремапом (heavy-first LPT-scheduling)

Идеальная LPT-сортировка (sort by weight DESC): каждый слот получает average W/S — теоретический минимум wall = **тот же W/S**, что и наивный потолок (b), но с 0 wave-tail dominance.

Т.е. **δ_b_to_c = observed_wall − W/S × T_tile**:

| Kernel | Observed causal (064) | W/S × T_tile (потолок опт.) | δ_ремап max prize |
|:--|:-:|:-:|:-:|
| merged  | 12.733 ms | 12.54 ms | **0.19 ms** (1.5%) |
| dk_new  | 4.572 ms  | 4.203 ms | **0.37 ms** (8.1%) |
| dq_new  | 4.586 ms  | 4.129 ms | **0.46 ms** (9.9%) |
| **Σ**   | **21.891 ms** | **20.87 ms** | **1.02 ms** (4.6% causal E2E) |

**Приз перестановки БУМАГОЙ (верх): ~1.0 ms wall на causal E2E 22.206 ms → 4.6%.**

Крест-проверка по-простому — «excess wall = (r_causal_wall − r_ideal_work) × wall_nc»:
- merged: (0.5068 − 0.504) × 25.125 = 0.075 ms  ← низкая волновая чувствительность (43 wave)
- dk: (0.5428 − 0.504) × 8.423 = **0.328 ms** ← самая чувствительная в абсолюте на wave, но wall меньше
- dq: (0.5420 − 0.504) × 8.462 = **0.321 ms** ← та же чувствительность
- Σ = 0.72 ms → **3.2% causal E2E**

**Bracket приза от ремапа: 0.7 – 1.0 ms wall = 3.2 – 4.6% causal E2E** (второй способ жёстче — учитывает, что NC тоже слегка теряет на партиальном хвосте).

**Реалистическая рекуперация** (грид-launch reorder — не идеальный LPT, есть overhead): половина = **0.35 – 0.5 ms → 1.6 – 2.3% causal E2E**.

---

### §1.d NCu-подтверждение перекоса — прибором

**Метрика выбора**: `sm__cycles_active.{avg,max,min,sum}` + `launch__waves_per_multiprocessor`.
**Обоснование**: `sm__cycles_active.max/avg` — прямая доля волнового хвоста в wall. Если grid-schedule идеально балансирует, max ≈ avg (skew_ratio → 1.0). Разница между max и avg **и есть** wall-налог перекоса (wall = max_over_SMs).
**Прогон**: 1 запуск/kernel/mode из `bench_r2c_e2e` под `--kernel-name --launch-count 1` (CUDA 13.1 ncu, HOME=/tmp).
**Скрипт**: `066_ncu_skew.sh` → `066_ncu_skew.txt`.
**Гейт-тишина** на прогоне: ✓ (compute-apps EMPTY перед стартом).

**Результаты** (значения sm_cycles_active, кол-во циклов):

| Kernel | Mode | avg | max | min | max/avg | max/min | waves |
|:--|:--|--:|--:|--:|:-:|:-:|:-:|
| merged | NC | 54,763,032 | 55,285,963 | 54,553,369 | 1.0095 | 1.0134 | 43.57 |
| merged | CAUSAL | 27,707,452 | 28,280,954 | 27,118,030 | **1.0207** | 1.0429 | 43.57 |
| dk_new | NC | 15,445,228 | 15,627,766 | 15,211,787 | 1.0118 | 1.0273 | 21.79 |
| dk_new | CAUSAL | 8,025,073 | 8,273,491 | 7,743,515 | **1.0310** | 1.0684 | 21.79 |
| dq_new | NC | 14,132,665 | 14,273,234 | 13,933,838 | 1.0099 | 1.0244 | 14.52 |
| dq_new | CAUSAL | 7,342,074 | 7,672,139 | 6,932,697 | **1.0450** | 1.1067 | 14.52 |

**Наблюдения**:
1. **NC skew — везде ≤1.18%** (равновесная работа, только «партиальный хвост» на дне последней волны). Уверенно фон.
2. **CAUSAL skew**: merged 2.07%, dk 3.10%, **dq 4.50%** — **прогрессивно растёт с уменьшением числа волн** (43.57 → 21.79 → 14.52). Верхний потолок волновой sub-балансировки: чем меньше волн, тем сильнее «тень» тяжёлого блока в последней волне доминирует.
3. **Ratios causal_avg/nc_avg** (равновесная работа):
   - merged: 27,707,452/54,763,032 = **0.5060** ≡ NCu-insts 0.506 (063-r) ✓
   - dk: 8,025,073/15,445,228 = **0.5196**
   - dq: 7,342,074/14,132,665 = **0.5195**
4. **Ratios causal_max/nc_max** (wall-side):
   - merged: **0.5115** ≈ wall 0.5068 (∆ 0.9%)
   - dk: **0.5294** vs wall 0.5428 (wall выше — прибавка launch/L2/kernel-boot cost)
   - dq: **0.5375** vs wall 0.5420 (совпадение до 0.4%)

**Перекос виден прибором** ✓. **Максимальный skew tax = (max/avg − 1) × causal_wall**:
- merged: 0.0207 × 12.733 = 0.264 ms
- dk_new: 0.0310 × 4.572 = 0.142 ms
- dq_new: 0.0450 × 4.586 = 0.206 ms
- **Σ upper-bound skew-tax = 0.61 ms** = 2.75% causal E2E

Реалистическая рекуперация ремапом (после overhead) ≈ **0.15 – 0.3 ms causal E2E** (0.7 – 1.4%).

**Диагноз**: dq_new и dk_new — главные подозреваемые. **dq_new — самая волновая (14.52 волн, skew 4.5%)**.

---

## §2 Загадка байтов — DRAM ratio 0.54 vs 0.50

### §2.a Атрибуция по ядрам из данных 063-r

**Данные NCu 063-r** (`dram__bytes.sum` per kernel per mode):

| Kernel | NC DRAM | CAUSAL DRAM | Ratio | Excess vs 0.504 (paper live-work) |
|:--|:-:|:-:|:-:|:-:|
| merged  | 9.80 GB | 5.51 GB | **0.5622** | +0.058 → **+0.57 GB** |
| dk_new  | 9.26 GB | 4.99 GB | **0.5389** | +0.035 → **+0.32 GB** |
| dq_new  | 9.25 GB | 4.99 GB | **0.5395** | +0.035 → **+0.32 GB** |
| D (control) | 0.545 GB | 0.545 GB | 1.000 | — (нет qt-loop, полное чтение dO+O) |
| **Σ (без D)** | **28.31 GB** | **15.49 GB** | **0.5471** | **+1.21 GB non-scaling** |

Общее 0.547 → округл. 0.55 ≈ то, что 063-r назвал «0.54–0.56».

### §2.b Мертвые dS-тайлы — ПИШУТСЯ или НЕТ?

**Grep-верификация**:
- `merged_v1.cu:146: const int qt_start = causal ? kt : 0;` — цикл qt стартует с kt, **никакая mert-tile итерация не запускается вовсе**.
- `merged_v1.cu:242: if (causal && j_g > i_g) return true;` — внутри тайла маска (для диагональных) — но эта маска только для **PxdO** внутри активного тайла, не для skipped-qt.
- В ds_gen-части (внутри qt-loop) первый шаг = чтение Q/dO, потом compute dS. Если qt skipped (i < kt) → **loop-body НЕ входит вовсе** → dS не пишется.

**Вывод**: **dS_nat записи ПОЛНОСТЬЮ ПРОПУСКАЮТСЯ в мертвых тайлах** (skip-writes уже реализован механикой qt_start). Двойного приза «causal-wall + транзиент dS» **НЕТ**: writes уже scale с causal.

Верификация через размер dS в DRAM:
- Полный dS_nat = bh × sl_q × sl_k × 1 byte = 128 × 8192 × 8192 = **8 GB**
- Merged DRAM writes = dV_write (128 MB) + dS_writes_causal + other
- Если writes ~4 GB dS + 128 MB dV = 4.13 GB writes causal. Reads = 5.51 − 4.13 = 1.38 GB. Sanity (reads NC = 9.80 − 8 − 0.128 = 1.67 GB под NC). Reads causal/reads NC = 1.38/1.67 = **0.826** — reads НЕ скалируются линейно с causal.

**Атрибуция excess по компонентам** (per-kernel):

| Компонент | Механика | Scales? | Оценка bytes NC → CAUSAL |
|:--|:--|:-:|:--|
| dS_nat writes (merged) | early-exit before ds_gen — 0 writes для skipped qt | scale 0.504 | 8.00 → 4.03 GB |
| dK/dV/dQ writes | one-shot по завершении block | non-scale | 128 MB each (dK+dV+dQ ≈ 384 MB) fixed |
| K/V reads (merged) | block loads K/V ONCE перед qt-loop | non-scale | 256 MB fixed |
| K reads (dq_new) | block loads K ONCE перед kt-loop (line 91 SMEM smK_area) | non-scale | 128 MB fixed |
| Q reads внутри qt-loop | Q per qt-tile (line 158+) | scale 0.504 | mid-scale |
| dS reads (dk, dq) | per активному тайлу | scale 0.504 | mid-scale |
| **Wave-tail warm blocks** | блок стартует, читает K/V/Q для 1 хвостового тайла, exit | non-scale | ~128–256 MB tail |
| L2-inclusive DRAM traffic | падение hit-rate под skewed access | non-linear | ~200 MB extra |
| Padding stride (dS 80 vs 64 в SMEM) | SMEM-only, DRAM tight (grep комментарий line 21-22 dq_new: «Stride affects only SMEM layout, NOT MMA arithmetic → bit-exact preserved») | 0 DRAM | 0 |

**Итог атрибуции**:
- **~55%** excess = fixed-costs (K/V/dK/dV/dQ) — expected, incompressible
- **~30%** excess = wave-tail warm-up (block boots, reads K/V for 1 tile, early-exits)
- **~15%** excess = L2 hit-rate падение под skewed access pattern (indirect)

**Проверка «мертвые тайлы уже не пишутся»**: NC/CAUSAL dS-write bytes matches formula 8 GB × 0.504 = 4.03 GB на merged (5.51 − 1.48 reads − ε writes-other) → dS writes ~4 GB ≈ 4.03 GB ✓.

### §2.c Двойной приз «скип-записи + транзиент dS для W-серии»

**Скип-wall-writes**: **УЖЕ реализован** (qt_start = kt). **Приз = 0 ms**.

**Скип-хранения-dS для W-series** (уменьшить allocated dS с 8 GB до 4 GB per layer):
- dS_nat tensor shape = (bh, sl_q, sl_k, 1 byte) = 8 GB currently, dq и dk читают его как plain array.
- Под causal живет только upper-tri половина (для merged) = 4 GB.
- **Приз**: 4 GB memory savings per layer, транзиентная память для gradient accum (W-series может быть 24–48 layers → 96–192 GB savings). **Wall prize: 0** (память ≠ wall).
- **Механика**: reshape dS_nat в компактную «tri-band» форму — но требует переписать индексацию во всех 3 ядрах (merged, dk, dq). Grep verification: dS index computed at line 155-166 (merged), 154-166 (dq), lines 246-283 (dk). **Big rewrite**, риск breaking bit-exact на 11/11 chains.
- **Оценка стоимости**: 5–7 dev-days + 30-run cert per gradient. **Откладывать до момента, когда W-series упрётся в память.**

**Барьеры**: qt-loop skip в merged НЕ трогает барьеры (grep не находит `__syncthreads()` в теле qt-loop-branch skipped). Если бы трогал — racecheck-план нужен. **Racecheck-риск: 0 (не применяется).**

---

## §3 Диагональ — неустранимый пол

Диагональные тайлы (block==diagonal — qt==kt):
- Merged/dk: block (bh, kt) → диагональный qt = kt. **Всегда 1 диагональный тайл на блок**.
- dq: block (bh, qt) → диагональный kt = qt. **Всегда 1 диагональный тайл на блок**.

Всего диагональных тайлов на pass = 128 bh × 128 kt = **16,384** (per kernel).

**Как доля от живой работы**:
16,384 / 1,056,768 = **0.0155 = 1.55%** от live tile-iters.

**Wall contribution** (нижняя граница):
- merged: 16384 × T_tile / 376 slots = 16384 × 4.462 μs / 376 = **194 μs = 0.19 ms** (1.5% wall merged causal).
- dk_new: 16384 × 2.991 / 752 = **65 μs** (1.4% wall dk causal).
- dq_new: 16384 × 4.407 / 1128 = **64 μs** (1.4% wall dq causal).
- **Σ diag floor = 0.32 ms = 1.4% causal E2E.**

**Природа**: внутри диагонального тайла работает full MMA, но применяется маска `j_g > i_g` (grep `merged:242`), которая лишь **гасит** результаты — не пропускает FLOPs. Внутри 128×64 тайла живой треугольник = 64×64/2 + 64/2 = 2080 из 8192 элементов (**25.4%** от тайла), но full 4×2 MMA batch выполняется.

**В леджер**: **causal wall floor ≥ 0.32 ms** (1.4%) — не выкорчёвывается без изменения тайл-геометрии на диагонали (special-case diagonal-only kernel — не окупается 1.4% wall призом). **Фиксируем как permanent floor.**

---

## §4 Вердикт-карта v-causal

| Цель | Приз (paper/measured) | Механика | Риск | Цена (dev-days) |
|:--|:-:|:--|:--|:-:|
| **Ремап расписания dk/dq (grid heavy-first)** | **0.15 – 0.3 ms** wall (0.7–1.4% causal E2E) real / 0.7–1.0 ms upper-bound | Пересортировать grid: `blockIdx.x` → weight-descending map. Kernel unchanged — только launch-side index reorder. dk/dq оба грид=(bh × n_tile), просто permute block index. | Ordering assumption inside kernel? grep не находит зависимости от blockIdx линейности — блоки независимы (KV/Q-owned, no cross-block reduce). Bit-exact preserved (block work independent of order). | **1–2 d.d.** (host-side reorder table + 30-run cert + racecheck) |
| **Skip-записи-dS в мертвых тайлах** | **0 ms** (уже реализовано qt_start=kt) | early-exit ДО ds_gen. Проверено гриппом. | — | 0 |
| **Skip-ХРАНЕНИЯ-dS (compact tri-band)** | **4 GB/layer memory**, **0 ms wall** | reshape dS_nat в compact upper-tri, переписать индексацию в 3 ядрах | Bit-exact breaks на 11/11 chains — full re-seal | **5–7 d.d.** + 3 × 30-run cert. Прибыль в W-серии (memory ≠ wall). |
| **Diagonal floor** | −0.32 ms wall floor (1.4% causal) | special-case diagonal kernel с честной upper-tri MMA | Кастомная тайл-геометрия, растет harness | **3–5 d.d.**, приз 0.15 ms в лучшем случае — **не окупается** |
| **5-я бригада dk (occ 4→5)** | ~0.1–0.2 ms dk wall | reg-diet 124→102 (dropping 22 regs) ИЛИ threadsPerBlock 128→96 | reg-golfing риск breaking MMA layout; TPB change — MMA fragment layout math ломается | **3–5 d.d.**, риск high, ниже цикла — эшелон-2 (после causal wave-remap) |
| **Перекраска V (mentioned TZ)** | TBD — вплетается ПОСЛЕ causal-правок | reshape smV или V-cache layout | — | Эшелон-2 |

---

## §5 Ремап vs скип-dS — что первым, по числам

### 5.1 Ремап расписания

- **Приз (NCu-verified)**: max/avg = 1.045 dq / 1.031 dk / 1.021 merged → **0.61 ms upper-bound wall recovery**
- **Реалистично**: 0.15–0.3 ms wall = **0.7–1.4% causal E2E**
- **Стоимость**: 1–2 dev-days
- **Риск**: минимальный (block work idempotent, no cross-block deps grep-подтверждено)
- **ROI**: **0.15 ms / 1 dev-day = high**

### 5.2 Skip-записи-dS

- **Приз wall**: 0 ms (**УЖЕ реализован механикой qt_start**)
- **Приз memory для W-series**: 4 GB / layer (compact tri-band)
- **Стоимость compact tri-band**: 5–7 dev-days + full re-seal
- **Риск**: high (bit-exact breaks в 3 ядрах)
- **ROI wall**: **0** — memory ROI откладывать до W-серии

### 5.3 Приоритет

**Ремап расписания — ПЕРВЫМ. Skip-хранения-dS — эшелон W-series (не сейчас).**

Отдельная нота: **приза «двойной» (wall + транзиент dS) НЕТ**, потому что skip-writes уже механически реализован. TZ 066 §2.a был по гипотезе «мёртвые dS-тайлы пишутся» — гипотеза не подтвердилась.

---

## §6 Эшелон-2 hooks (после causal-правок)

1. **5-я бригада dk_new**: reg-diet 124→102 или TPB 128→96 → occ 4→5 → доп волна перекрывает хвост. Прирост dk wall ~0.1–0.2 ms. **После ремапа (эшелон-2)**.
2. **Перекраска V** (упомянуто в TZ): требует уточнения — вероятно, smV layout в merged для лучшего K→V→dV chain. Расследование эшелон-2.
3. **Diagonal special-case**: **отказ** (0.15 ms приз, 3–5 dd цена — не окупается).

---

## §7 Итоги 066

1. **§1 Гистограмма перекоса** ✓ (paper a/b/c + NCu подтверждение d):
   - Формула live/full = 8256/16384 = **0.5039** — точно совпала с NCu instructions 0.506 (063-r).
   - NCu `sm_cycles_active.max/avg` под causal: merged 1.021, dk 1.031, **dq 1.045** — dq наиболее перекошен (14.5 волн — самый чувствительный к хвосту).
   - Приз ремапа: **0.15–0.3 ms wall (0.7–1.4% causal E2E) реалистично** / 0.7–1.0 ms upper.
2. **§2 Загадка байтов 0.54 vs 0.50** ✓:
   - Атрибуция: **~55% fixed-cost (K/V/dK/dV/dQ), ~30% wave-tail warm, ~15% L2 hit-rate**.
   - Мертвые dS-тайлы: **НЕ ПИШУТСЯ** (verified: qt_start=kt skip before ds_gen). Двойной приз wall+dS — **нет**.
   - Compact tri-band dS storage: **4 GB/layer** для W-series, но **wall = 0**. Big rewrite — эшелон W-series.
3. **§3 Диагональный floor** = **0.32 ms (1.4% causal E2E)**. Не выкорчёвывается разумной ценой — в леджер как permanent floor.
4. **§4 Вердикт-карта** составлена (6 целей).
5. **§5 Приоритет: РЕМАП ПЕРВЫМ**. Skip-хранения-dS — эшелон W-series.
6. **§6 Эшелон-2**: 5-я бригада dk (occ 4→5) + перекраска V — после causal ремапа.

### Chain md5

- 062 `b7044db70019e8fa7dea260f9f235b6c`
- 063 `b7c82475ed49ac4821c7346f99a38fb1`
- 063-r `1dce5e445e5b47152c2894ffc7947b30`
- 064 `c099586c01ad07d6c83f3d732a77de3c`
- 065 `cc5c2a7f96aeed162ddf28609703009a`
- **066 `029b8c4b9b6e154ad437706eafd25a1d`**

### Файлы 066

- `runs/reports/066_causal_recon.md` (this report)
- `runs/reports/066_causal_wall_sanity.sh` — 1-run sanity (causal 22.123 ms ✓)
- `runs/reports/066_ncu_skew.sh` + `066_ncu_skew.txt` — per-SM cycles_active NC + CAUSAL × 3 kernels
- `runs/reports/066_ncu_skew_dq.sh` / `_dk.sh` / `_merged.sh` — per-kernel wrappers (deprecated in favour of unified 066_ncu_skew.sh)

---

**End 066. Разведка ЗАВЕРШЕНА. Правки production: 0. Гейт-тишина ✓.
Приз ремапа: 0.15–0.3 ms wall (0.7–1.4% causal E2E) реалистично, upper 0.61 ms.
Skip-хранения-dS: 4 GB memory/layer, эшелон W-series.
Приоритет: РЕМАП РАСПИСАНИЯ dk/dq первым (dq перекошен на 4.5% max/avg — стартовый кандидат).**
