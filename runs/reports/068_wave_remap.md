# 068 — Ремап расписания полос: ROLLBACK КРАСНАЯ СТРОКА

**Chain**:
- 062 `b7044db70019e8fa7dea260f9f235b6c`
- 063 `b7c82475ed49ac4821c7346f99a38fb1`
- 063-r `1dce5e445e5b47152c2894ffc7947b30`
- 064 `c099586c01ad07d6c83f3d732a77de3c`
- 065 `cc5c2a7f96aeed162ddf28609703009a`
- 066 `029b8c4b9b6e154ad437706eafd25a1d`
- 067 `ecbdeff9a42be2cf20b5d4d2afc41de7`
- **068 `<self>`**

**Табло**: causal 22.206 ms судейское (063 cert).
**Гипотеза ТЗ 068**: heavy-first block remap срежет wave-tail (max/avg 1.045 dq / 1.031 dk из 066) → приз 0.15–0.3 ms causal E2E.
**Результат**: ремап дал **+2.61 ms wall REGRESS (+11.8%)** — правило-2/3 v2 → **ROLLBACK**.

**Правки production: 0 (после отката).**
**Гейт-тишина**: ✓ (nvidia-smi compute-apps EMPTY на всех прогонах).

---

## Артефакт-хедер

```
libs/ (post-rollback = pre-068 sealed):
  fa_bwd_merged_v1.cu  md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33  (unchanged 040 sealed) regs=252
  fa_bwd_dk_new.cu     md5 25e5e1077cc3bec2c49bf9288fe60c54  (unchanged 061 S2v4)   regs=124
  fa_bwd_dq_new.cu     md5 d7a11a3d788eb4c396d892bc9c8ab754  (unchanged 041)        regs= 69
  fa_bwd_common.cuh    md5 4407ec9cf64708a2a28dc36633d5d6f1  (unchanged)
  bench_r2c_e2e.cu     — fingerprint EXPECT rolled back to 252/124/69/38

archives (могут удаляться после подтверждения rollback):
  fa_bwd_merged_v1.cu.pre_068 = 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33  (same as sealed — safe)
  fa_bwd_dk_new.cu.pre_068    = 25e5e1077cc3bec2c49bf9288fe60c54
  fa_bwd_dq_new.cu.pre_068    = d7a11a3d788eb4c396d892bc9c8ab754

Форма: bh=128 sl=8192 hd=128, Br=64 (merged/dk) / 64 (dq) Bc=64
GPU: NVIDIA RTX PRO 6000 Blackwell WS Edition, sm_120a, 188 SMs
```

Baseline restored: post-rollback causal wall = **22.147 ms** (cert 22.206, drift +0.06% на первом шоте — в пределах шума ±1%).

---

## §1 БУМАГА — refresh histogram + mechanics + predictions

### §1.a Освежение гистограммы (post-067 landscape)

TZ 067 закрыт КРАСНОЙ строкой (0 правок production) — landscape НЕ изменился с 066. Гистограмма из 066 §1 остаётся actual:

| Kernel | Live/full = 8256/16384 = 0.5039 | Occ | Slots | Waves | max/avg (NCu 066) |
|:--|:-:|:-:|:-:|:-:|:-:|
| merged | 0.5039 | 2 | 376  | 43.57 | 1.021 |
| dk_new | 0.5039 | 4 | 752  | 21.79 | 1.031 |
| dq_new | 0.5039 | 6 | 1128 | 14.52 | **1.045** |

Приз-верх по 066: 0.61 ms wall bottleneck skew-tax. Реалистично 0.15–0.3 ms.

### §1.b Механика ремапа — LUT vs формула

**Выбор**: **формула-перестановка** без LUT.

**Обоснование**:
- LUT в constant memory (128 kt × 128 bh = 16384 × 2 bytes = 32 KB) — избыточно, лишний constant-memory сет-ап.
- Формула — 3–4 SASS инструкции, нулевая memory pressure.
- Инвариант «kt/bh полностью представлен» проверяется тривиально: div/mod на bh × n_kt.

**Формула** (для KV-owned merged/dk):
```
Under causal:
  kt = blockIdx.x / bh      // outer: kt varies SLOW (heavy kt=0 first)
  b  = blockIdx.x % bh      // inner: bh varies fast
Under nc: original linear (b = idx/n_kt, kt = idx%n_kt).
```

**Формула** (для Q-owned dq):
Heavy = qt=n_qt−1 (не qt=0!) — под causal `kt_end = qt+1`, живая работа растёт с qt.
```
Under causal:
  qt_raw = blockIdx.x / bh
  qt = (n_qt - 1) - qt_raw   // reverse: heavy qt=127 first
  b  = blockIdx.x % bh
Under nc: original.
```

### §1.c Инвариант независимости блоков — доказательство

grep-проверка **отсутствия межблочных зависимостей**:
```
grep atomic /data/lib/podman-data/projects/goml/libs/fa_bwd_merged_v1.cu ... → 0 matches
grep atomic /data/lib/podman-data/projects/goml/libs/fa_bwd_dk_new.cu     ... → 0 matches
grep atomic /data/lib/podman-data/projects/goml/libs/fa_bwd_dq_new.cu     ... → 0 matches
```

**Никаких atomicAdd / __reduce / cross-block reduce.**

Выходные регионы блоков:
- merged: `dS_nat[b, qt, kt_slice]` per qt-tile + `dV[b, kt_slice, :]` per block — **disjoint (bh, kt) регионы**.
- dk_new: `dK[b, kt_slice, :]` per block — **disjoint (bh, kt)**.
- dq_new: `dQ[b, qt_slice, :]` per block — **disjoint (bh, qt)**.

**Вывод бумагой**: **remap = чистая перестановка независимых заданий**. Bit-exact preserved by construction (order of independent commutative computations doesn't affect result). **Racecheck не требуется.**

### §1.d nc-нейтральность

Под nc все блоки равновесные (128 тайлов каждый). Перестановка равновесных блоков не меняет makespan (min = max = W/S для равных заданий). NC гейтится флагом `else` — остаётся оригинальный порядок. **Гарантированно нейтрально.**

### §1.e Именованные предсказания

| Метрика | Пре-068 | Пост-068 предсказание | Обоснование |
|:--|:-:|:-:|:--|
| merged max/avg (NCu) | 1.021 | **1.005** | LPT сбалансирует waves 43.6 |
| dk_new max/avg (NCu) | 1.031 | **1.010** | LPT сбалансирует waves 21.8 |
| dq_new max/avg (NCu) | 1.045 | **1.010** | LPT сбалансирует waves 14.5 |
| causal wall total | 22.206 ms | **21.90 ± 0.15 ms** | −0.3 ms реалистично |
| nc wall total | 42.346 ms | **42.35 ± 0.4 ms** | == (else-branch неизменна) |
| DRAM merged causal | 5.51 GB | == | writes/reads не тронуты |
| Occupancy | 2/4/6 | ==  | reg budget не тронут |
| Ptxas merged | 252r | **≥252** (может +1r на branch) | if-else держит `causal` в регистре |
| Ptxas dk/dq | 124/69 | == | те же формулы, меньше давление |

---

## §2 Патч (применён, гейт запущен, откачен)

### §2.a merged_v1.cu (line 93+)

```c
// 068 wave-remap: under causal, sort blocks by weight descending (heavy kt=0
// first) via kt-outer / bh-inner index mapping. Blocks are KV-owned and write
// to disjoint (bh, kt) regions of dV/dS_nat → permutation of independent
// tasks → bit-exact preserved; no cross-block hazard. nc branch unchanged
// (equal-weight blocks — order neutral).
int b, kt;
if (causal) {
    kt = blockIdx.x / bh;
    b  = blockIdx.x % bh;
} else {
    b  = blockIdx.x / n_kt;
    kt = blockIdx.x % n_kt;
}
if (b >= bh) return;
```

### §2.b dk_new.cu (line 66+)

Аналог merged (KV-owned).

### §2.c dq_new.cu (line 82+)

Reverse qt для правильного направления сортировки:
```c
int b, qt;
if (causal) {
    qt = (n_qt - 1) - (blockIdx.x / bh);
    b  = blockIdx.x % bh;
} else {
    b  = blockIdx.x / n_qt;
    qt = blockIdx.x % n_qt;
}
```

### §2.d bench EXPECT

Осознанное обновление: `kernel_merged_v1` 252→**253** (+1r для держания `causal` в регистре между branch'ами).
- 253 × 128 threads = 32384 регистров. **65536 / 32384 = 2.02 → occ 2 blocks/SM (неизменно)**.

---

## §3 Гейт-протокол (что сделано до отката)

### §3.a Build + ptxas + fingerprint ✓

```
FINGERPRINT kernel_d_precompute    numRegs= 38 (expected  38) OK
FINGERPRINT kernel_merged_v1       numRegs=253 (expected 253) OK  ← осознанно 252→253
FINGERPRINT kernel_dk_new          numRegs=124 (expected 124) OK
FINGERPRINT kernel_dq_new          numRegs= 69 (expected  69) OK
```

Все прошли. Merged +1 reg подтверждён (occ 2/SM неизменна).

### §3.b Sanity wall shots (2 causal + 1 nc)

```
POST-PATCH:
  causal shot 1:  D=0.369 merged=14.064 dk_new=5.308 dq_new=5.095 total=24.835 (+11.8%)
  causal shot 2:  D=0.369 merged=13.979 dk_new=5.294 dq_new=5.096 total=24.737 (+11.4%)
  nc      shot 1: D=0.342 merged=24.895 dk_new=8.317 dq_new=8.379 total=41.933 (-1.0%, else-branch OK)
```

**Причал двумя шотами согласен**: causal REGRESS +11.5% (среднее двух шотов).

Правило-2/3 v2: **KEEP требует ≥2% wall improvement** — у нас **−11.5% wall degradation** → правило автоматически даёт **ROLL**.

**Ни ABBA ≥8 пар, ни bit-exact 11/11 x3, ни NCu-post не запускаем** — правило 2/3 v2 срабатывает на sanity данным. **ROLLBACK немедленно.**

---

## §4 Причина регресса — атрибуция бумагой

Гипотеза ремапа: heavy-first сокращает wave-tail. Но **overhead L2/DRAM re-pattern превысил приз перекоса на порядок**.

### §4.a L2-cache pattern degradation

**Оригинальный порядок (bh × n_kt fast-kt)**:
- Первая волна = 376 слотов (merged) → блоки idx 0..375 = (bh=0..2, kt=0..127 partial). Часть на bh=0 полностью, часть bh=1..2.
- K/V-загрузка: consecutive blocks share bh → K[b][kt_slice] tiles от одного bh — **L2 hot** между блоками в одном bh.
- dS_nat writes: bh=0's 128 kt-columns записываются consecutively → одна bh-row в dS полностью первой.

**Ремап (kt fast-slow, bh fast)**:
- Первая волна = 376 слотов → блоки (kt=0, bh=0..127) + (kt=1, bh=0..127) + (kt=2, bh=0..119 partial).
- K/V: 128 разных bh простреливают ОДНОВРЕМЕННО, all K/V-tile регионы разные bh → **L2 thrash** — cache не удерживает 128 × 16 KB = 2 MB одновременно активных K/V-регионов.
- dS_nat writes: все bh пишут в одну kt-column одновременно → **write-hot spot на одном DRAM row** (memory controller contention).

### §4.b Оценка ущерба

Per-kernel wall delta (2 шота, средний):
- merged: 12.660→14.02 = **+1.36 ms (+10.7%)** — L2 K/V pattern loss dominant.
- dk_new: 4.566→5.30 = **+0.73 ms (+16.1%)** — Q reads spread, cache miss.
- dq_new: 4.581→5.10 = **+0.51 ms (+11.2%)** — K reads spread + write pattern.
- **Σ = +2.60 ms (+11.7% causal E2E).**

**Отношение убыток/приз-верх (066)**: 2.60 / 0.61 = **4.3× overhead vs skew-tax**. Даже если ремап полностью снял бы skew-tax, чистый убыток от L2 pattern составил бы +2.0 ms.

### §4.c 066 recon оценка была неполной

**066 § 5.1 говорил**: «Ordering assumption inside kernel? grep не находит зависимости от blockIdx линейности — блоки независимы».

**Действительно**: блоки независимы в смысле output correctness. Но **не независимы в смысле cache/DRAM pattern**. 066 упустил attribute **L2-locality от linear block iteration** — это скрытое свойство исходного FIFO-грид-порядка, которое ремап разрушает.

**Уроки 068 в леджер**:
1. Grid-order affects L2/DRAM pattern даже когда outputs disjoint. «Independent blocks» ≠ «independent memory access».
2. Wave-tail skew (4.5% max/avg на dq) **приз-верх, не приз реалистичный**. Реалистичный приз ремапа мог быть ниже L2-pattern убытка → ремап-класс убыточен на данной архитектуре без коррекции доступа.
3. Правильный подход к тесту: НЕ сразу heavy-first LPT — а **сохранить bh-major локальность** через intra-bh remap (e.g., переставить только kt внутри одной bh) или **Морок-hash** (сохранить псевдо-случайность которая приблизит max/avg к 1.0 без разрушения bh-locality).

---

## §5 Rollback + baseline restoration ✓

### §5.a Rollback выполнен

```
cp fa_bwd_merged_v1.cu.pre_068 → fa_bwd_merged_v1.cu   md5 2bf32ab7... ✓ (== sealed 040)
cp fa_bwd_dk_new.cu.pre_068    → fa_bwd_dk_new.cu      md5 25e5e107... ✓ (== sealed 061)
cp fa_bwd_dq_new.cu.pre_068    → fa_bwd_dq_new.cu      md5 d7a11a3d... ✓ (== sealed 041)
bench_r2c_e2e.cu EXPECT: kernel_merged_v1 253 → 252  ✓
```

### §5.b Post-rollback rebuild + sanity

```
FINGERPRINT kernel_merged_v1       numRegs=252 (expected 252) OK ✓
FINGERPRINT kernel_dk_new          numRegs=124 (expected 124) OK ✓
FINGERPRINT kernel_dq_new          numRegs= 69 (expected  69) OK ✓
FINGERPRINT kernel_d_precompute    numRegs= 38 (expected  38) OK ✓

CAUSAL shot post-rollback:
  D=0.340 merged=12.660 dk_new=4.566 dq_new=4.581 total=22.147 (drift +0.06% vs cert 22.206 ✓)
```

**Sealed baseline restored 100%.** Правки production в 068 (пост-откат): **0**.

---

## §6 Вердикт + сиквенс

### §6.a ТЗ 068 — КРАСНАЯ СТРОКА

Heavy-first LPT wave-remap класс **закрыт как убыточный** на данной архитектуре / данной форме:
- Приз-верх (перекос) 0.61 ms → **L2-pattern убыток 2.60 ms = ×4.3 overhead**
- Реалистичный приз (0.15–0.3 ms) ниже даже минимальной оценки L2 loss
- Гипотеза «блоки независимы = порядок нейтрален» опровергнута — grid-order determinирует K/V L2 locality
- **Ремап-класс НЕ реколоризовать без правки доступа к K/V** (например, kt-persist или tile-batching)

### §6.b Сиквенс — глава дожима ЗАКРЫВАЕТСЯ

С 062 (cert 400) → 063 (causal ledger) → 064 (v0.2.0 release) → 065 (GH draft) → 066 (recon) → 067 (dead-dS skip красный) → **068 (remap красный)** — глава дожима causal-дорожки **исчерпана без структурной правки доступов**.

**Sealed causal wall остаётся 22.206 ms** (cert 063 30-run) / **22.147 ms** (post-rollback sanity 068 first shot).

**Кампания 040-068**:
- Sealed KEEP-серии: 040 (LDSM.x4.trans #7 −12%) + 041 (dq разморозка −3.5%) + 061 (S2v4 dk −19.24%) = cumulative ~20% wall reduction E2E.
- Cert 400 TFLOPS взят с запасом (415 proj / 260 fused nc).
- Публичный релиз v0.2.0 запечатан (Apache-2.0 + SPDX + clean-clone verify).
- Causal-дорожка исчерпана.

### §6.c Эшелон-2 hooks (после 068 rollback)

Продолжение возможно только через СТРУКТУРНЫЕ правки, а не порядок:
1. **5-я бригада dk** (occ 4→5 через reg-diet 124→102): +0.1–0.2 ms dk wall (мешает reg-golfing риск). ЭМБ-2.
2. **Перекраска V**: требует уточнения TZ. ЭМБ-2.
3. **Compact tri-band dS** (W-эшелон): 4 GB/layer memory, wall=0. Big rewrite.
4. **Persist-архитектура (KV-persist)**: пересмотр всей K/V pipeline. Структурная глава — не оптимизация одного паттерна.

### §6.d W0-пакет frozen_v3 (для тренировочной ветки)

Пакет W0 для передачи Vugar **не форсируется** — 068 ремап не вошёл в KEEP-ленту → frozen_v2 остаётся actual. При успешной 5-й бригаде dk или compact tri-band **дельта уедет в frozen_v3 пакетом**, но не сегодня.

---

## §7 Итоги 068

1. **§1 Бумага**: histogram освежён (== 066), формула-перестановка обоснована (LUT избыточен), инвариант независимости блоков доказан grep-ом, nc-нейтральность гарантирована else-веткой.
2. **§2 Патч применён**: 3 kernel edits + bench EXPECT 252→253.
3. **§3 Гейт §a-b запущен**: ptxas + fingerprint ✓, 2 sanity causal shots — **обе +11.5% wall** → правило 2/3 v2 → **ROLL**.
4. **§4 Атрибуция**: L2-pattern degradation +2.60 ms >> skew-tax приз-верх 0.61 ms (×4.3 overhead). Оригинальный bh-major FIFO обеспечивал скрытую L2-locality на K/V-загрузках.
5. **§5 Rollback ✓**: sealed baselines восстановлены byte-identical + fingerprint ✓ + causal sanity 22.147 ms (drift +0.06% cert). Правки production **0**.
6. **§6 Вердикт**: heavy-first LPT wave-remap класс **ЗАКРЫТ УБЫТОЧНЫМ** без правки K/V-доступа. Глава causal-дожима исчерпана.
7. **§7 Ярлыки в леджер**:
   - «grid-order = скрытая L2-locality» (068 explicit lesson)
   - «wave-tail skew приз-верх, не реалистичный» (068)
   - «блочная независимость по output ≠ независимость по DRAM pattern» (068)

### Chain md5

- 062 `b7044db70019e8fa7dea260f9f235b6c`
- 063 `b7c82475ed49ac4821c7346f99a38fb1`
- 063-r `1dce5e445e5b47152c2894ffc7947b30`
- 064 `c099586c01ad07d6c83f3d732a77de3c`
- 065 `cc5c2a7f96aeed162ddf28609703009a`
- 066 `029b8c4b9b6e154ad437706eafd25a1d`
- 067 `ecbdeff9a42be2cf20b5d4d2afc41de7`
- **068 `0bba4f923390593e7b51b278c3891d56`**

### Файлы 068

- `runs/reports/068_wave_remap.md` (this report — ROLLBACK красная)
- `runs/reports/068_build.sh` + `068_build.txt` — build log post-patch
- `runs/reports/068_sanity_causal.sh` — causal shot runner
- `libs/fa_bwd_merged_v1.cu.pre_068` / `.dk_new.pre_068` / `.dq_new.pre_068` — pre-patch archives (== sealed, могут удаляться для чистоты; сохранены для аудит-trail)

---

**End 068. РЕМАП РАСПИСАНИЯ КЛАСС ЗАКРЫТ УБЫТОЧНЫМ. Правки production: 0 (после отката). Sealed causal wall 22.206 ms сохранён. Глава дожима causal-дорожки ИСЧЕРПАНА без структурной правки K/V-доступа. Сиквенс: эшелон-2 (5-я бригада dk / перекраска V) требует пере-осмысления или отдельные структурные ТЗ 069+.**
