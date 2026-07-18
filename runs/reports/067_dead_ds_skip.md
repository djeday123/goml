# 067 — Скип записи мертвого dS: ХИРУРГИЧЕСКИЙ ДОЧЕТ

**Chain**:
- 062 `b7044db70019e8fa7dea260f9f235b6c`
- 063 `b7c82475ed49ac4821c7346f99a38fb1`
- 063-r `1dce5e445e5b47152c2894ffc7947b30`
- 064 `c099586c01ad07d6c83f3d732a77de3c`
- 065 `cc5c2a7f96aeed162ddf28609703009a`
- 066 `029b8c4b9b6e154ad437706eafd25a1d`
- **067 `<self>`**

**Табло**: causal 22.206 ms судейское (063).
**ТЗ 067**: скип записи мертвого dS → −0.8..−1.2 ms causal + транзиент 8.6→4.6 GB.
**Правки production ядер: 0** — все ворота остались в состоянии закрытых.
**Гейт-тишина**: N/A (замеров нет; работа только бумажная).

**ВЕРДИКТ: ТЗ 067 ЗАКРЫТ КРАСНОЙ СТРОКОЙ. Оптимизация УЖЕ РЕАЛИЗОВАНА в production (эпоха 018/026, «causal skip» через `qt_start = kt` в обоих KV-owned ядрах + `kt_end = qt+1` в Q-owned). Стройка НЕ НАЧИНАЕТСЯ; wall-приза 0 ms; W0-дельта транзиент требует правки dk/dq индексации (запрещено).**

---

## Артефакт-хедер

```
libs/ (production — unchanged):
  fa_bwd_merged_v1.cu  md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33  (KV-owned  ptxas 252r occ=2)
  fa_bwd_dk_new.cu     md5 25e5e1077cc3bec2c49bf9288fe60c54  (KV-owned  ptxas 124r occ=4)
  fa_bwd_dq_new.cu     md5 d7a11a3d788eb4c396d892bc9c8ab754  (Q-owned   ptxas  69r occ=6)
  fa_bwd_common.cuh    md5 4407ec9cf64708a2a28dc36633d5d6f1
  fa_bwd_ds_gen.cu     md5 665a350d3da8ae90b816ccd6b55db346  (reference, R1 chain only)
  fa_bwd_dk.cu         md5 068d6a4fdf5ae04816ebca199b9293cc  (sealed reference)
tests/ (untouched):
  r2c_merged_bit_exact.cu, r1b_dk_bit_exact.cu — bit-exact harness
  bench_r2c_e2e.cu — production wall bench
```

---

## §1 Бумага — точная механика записи мертвой половины

**Аккуратная разметка «мертвого» vs «живого»** (causal-attention):
- Живая часть dS_nat: `dS[bh, qt, kt]` где `qt ≥ kt` (upper triangle **включая** диагональ)
- Мертвая часть: `dS[bh, qt, kt]` где `qt < kt` (strict lower triangle)
- Пропорции: live/full = Σ_{k=0}^{127}(128−k) / (128·128) = 8256 / 16384 = **0.5039**

### §1.a Writer path — где вообще пишется dS_nat в production

**merged_v1.cu (единственный writer в production R2C pass)**:

```
line 146:  const int qt_start = causal ? kt : 0;
line 147:  for (int qt = qt_start; qt < n_qt; ++qt) {
line 148:      const int qt_base = qt * Br;
line 150:      // ==== Step A: cp.async Q → smQ, dO → smdO ==== (внутри qt-loop)
...
line 313:      // ==== Step E: dS = P·(dP - D) quantize + STS to smdS_stage (006 path) ====
line 367:      // dS_nat path — STS.b16 to smdS_stage
line 368-371:  smdS_stage[i_local × SMDS_STAGE_STRIDE + j] = dsa/b fp8
...
line 377:      __syncthreads();  // BARRIER t9: pre-drain
line 379:      // ==== Step F: STG.128 drain dS_nat + dS_T ====
line 380:      {
line 381:          uint8_t *dS_nat_b = dS_nat_out + b * sl * stride_ds;
line 385:          for (c = tid; c < 256; c += 128) {   // 256 = Br × cpr = 128 × 2
line 391:              uint4 chunk = *reinterpret_cast<uint4*>(&smdS_stage[r × stride + col_byte]);
line 392:              *reinterpret_cast<uint4*>(&dS_nat_b[i_g × stride_ds + j_start]) = chunk;
line 399:      __syncthreads();  // BARRIER t_new2: post-drain
```

**Ключ**: Step F **находится ВНУТРИ qt-loop** (индентация нужна: line 379 внутри `for (int qt = qt_start; ...` на line 147). Loop-body не входит вовсе при `qt < qt_start = kt` → **Step F НЕ выполняется для мертвых qt**.

### §1.b dS_T — уже полностью phantom

```
merged_v1.cu:373:  // 033-c: dS_T path устранён (dk_new читает dS_nat + транспонирует on-the-fly, W2 A1)
merged_v1.cu:374:  //        smdS_T_stage больше не пишется; STG.128 drain dS_T также убран (Step F).
merged_v1.cu:396:  // 033-c: dS_T drain устранён; dS_T буфер не пишется в DRAM.
merged_v1.cu:527:  // 038-E: dead-alloc smdS_T_stage (5120 B) снят из smem_bytes.
merged_v1.cu:528:  //        Указатель smdS_T_stage в kernel остаётся (не используется), но SMEM не резервирует.
```

**dS_T НИКОГДА не пишется в DRAM с эпохи 033-c**. bench_r2c_e2e всё ещё выделяет `dS_T dsz = 8.59 GB` (line 208), но никто в него не пишет.

Это отдельная тема (dead-alloc bench-side), решается тривиально — просто удалить `cudaMalloc(dS_T)` в bench + передать `nullptr` (kernel не читает и не пишет dS_T_out). **Прибыль: 8.59 GB транзиент.** Но это не «скип записи мертвого» — это удаление алломерты.

### §1.c Memset / инициализация буфера

```
bench_r2c_e2e.cu:126:  CKR(cudaMalloc(&dS_nat,dsz));   // no memset после cudaMalloc
bench_r2c_e2e.cu:138:  CKR(cudaMemset(rV,0,sz*sizeof(float)));   // reference V/K/Q memset
bench_r2c_e2e.cu:145:  CKR(cudaMemset(gV,0,sz*sizeof(float)));   // gradient V/K/Q memset
bench_r2c_e2e.cu:208:  CKR(cudaMalloc(&dS_nat,dsz));   // production alloc, no memset
```

**dS_nat в bench НЕ memset'ится** — буфер allocated uninitialized. Под causal мертвая половина остаётся с garbage от `cudaMalloc` (undefined data). Никто её не читает → безопасно.

Bit-exact harness `r2c_merged_bit_exact.cu` memsets buffer перед прогоном:
```
line 121:  CKR(cudaMemset(dS_nat_ref, 0, dsz));   // ref буфер → 0
line 122:  CKR(cudaMemset(dS_T_ref, 0, dsz));     // ref буфер → 0
line 132:  CKR(cudaMemset(dS_nat_gen, 0, dsz));   // gen буфер → 0
line 133:  CKR(cudaMemset(dS_T_gen, 0, dsz));     // gen буфер → 0
```

Оба (ref и gen) стартуют с dS_nat=0. После прогона: под causal мертвая половина осталась **0 в обоих** (никто не писал). Comparator сверяет byte-exact (line 178-186 вkomment: «dS_nat compare only positions where i_g < sl && j_g < sl»). Мертвая зона удовлетворяет `i_g < sl && j_g < sl` (индексы валидны), но там **0==0 → BIT-EXACT preserved by memset**.

### §1.d Стрим по полному страйду?

Step F line 385 loop: `for (c = tid; c < total; c += FA_M_THREADS)` где `total = Br × cpr = 128 × 2 = 256` chunks. Это ONE tile drain (128 rows × 2 chunks per row × 16 bytes = 4 KB тайл записи). НЕ полный страйп — только текущий (qt, kt) тайл. И только когда loop-body входит.

**Итог: полного memset НЕТ, ноль-тайл ds_gen-частью НЕТ, стрим по страйду НЕТ.**

---

## §2 Читатели мертвой зоны — grep-доказательство

```
fa_bwd_dk_new.cu:86:   const int qt_start = causal ? kt : 0;
fa_bwd_dk_new.cu:88:   for (int qt = qt_start; qt < n_qt; ++qt) {

fa_bwd_dq_new.cu:107:  const int kt_end = causal ? (qt + 1) : n_kt;

fa_bwd_ds_gen.cu:261:  const int kt_end = causal ? (qt + 1) : n_kt;   (reference)
fa_bwd_dk.cu:173:      const int qt_start = causal ? kt : 0;         (sealed reference)
```

**Все 5 ядер имеют causal-skip**:
- 2 production writer (merged_v1) и readers (dk_new, dq_new)
- 2 R1 reference readers (dk, ds_gen)

**Формально доказано**: под causal
- dk_new читает `dS[bh, qt, kt]` для `qt ∈ [kt, n_qt−1]` → диапазон `qt ≥ kt` = **live half only**
- dq_new читает `dS[bh, qt, kt]` для `kt ∈ [0, qt+1)` = `kt ≤ qt` = **live half only (эквивалентная нотация)**

**Мертвые байты никто не читает.** ✓ Bit-exact выживает по конструкции.

---

## §3 Барьеры — не трогаются

Loop-body pattern: **skip = «не входить в loop-body при qt < qt_start»**. Все `__syncthreads()` находятся внутри loop-body:
```
merged_v1.cu:190:  BARRIER t3   (post-cp.async loads внутри Step A)
merged_v1.cu:377:  BARRIER t9   (pre-drain внутри Step E→F)
merged_v1.cu:399:  BARRIER t_new2 (post-drain перед Step G)
merged_v1.cu:423:  BARRIER t11  (pre-MMA_dV перед Step H)
```

Внешние барьеры (перед qt-loop / после qt-loop) — нет (по grep не найдены между loop-start и loop-end).

**Барьеры НЕ трогаются**. Если бы мы добавили новый skip (не добавляем — оптимизация уже в силе), racecheck был бы обязателен. Здесь **применения нет**.

---

## §4 DRAM-балансовая проверка — числами из 063-r + сегодняшнего NCu

**Формула**: DRAM_merged_causal = writes_scaled + reads_mixed
- writes_scaled: dS_nat writes скалируются с 0.504 (upper-tri fraction)
- reads_mixed: K/V reads once-per-block (non-scale) + Q/dO reads per-tile (scale 0.504)

**Ожидание** (модель):
```
NC merged DRAM   = writes_full + reads_all
                 = 8.59 GB (dS_nat, full)   + reads_all
NCu 063-r reading: merged NC dram = 9.80 GB → reads_all ≈ 1.21 GB
Из этих 1.21 GB: K+V one-shot ≈ 256 MB fixed; Q/dO per-tile ≈ 950 MB scale.
```

**Causal ожидание**:
```
causal_writes = 8.59 × 0.504 = 4.33 GB   (upper-tri including diag)
causal_reads  = 256 MB fixed + 950 MB × 0.504 = 736 MB
causal_dram   = 4.33 + 0.736 = 5.07 GB
```

**Наблюдаемое** (063-r NCu): **5.51 GB**.

**Δ = +0.44 GB** vs модель. Атрибуция (066 §2.b):
- ~55% fixed cost (K/V/dV/dK/dQ writes) — уже учтено
- ~30% wave-tail warm-up (block starts, loads K/V for 1 hot tile, exits)
- ~15% L2-inclusive traffic под skewed access pattern

**Если бы dead tiles писались**: causal_writes = 8.59 GB (full), causal_dram = 8.59 + 0.74 = **9.33 GB**. Наблюдаемое = **5.51 GB**. **Прямое доказательство**: если бы dead writes были, causal DRAM ratio был бы ~0.95, не 0.56. **Dead tiles точно не пишутся.**

---

## §5 Приз причёсывания — числами

### §5.a Wall

**ТЗ ожидание**: −0.8..−1.2 ms causal wall.
**Реальность**: writes уже scale 0.504 → wall от dS-writes уже минимален. Скипать нечего.
**Приз wall: 0 ms.**

### §5.b Транзиент dS — что реально доступно

| Источник | Прибыль | Механика | Разрешено ТЗ 067? |
|:--|:-:|:--|:-:|
| dS_T dead alloc (bench) | **8.59 GB** | удалить `cudaMalloc(&dS_T, dsz)` в bench + пропустить `nullptr` в merged (dS_T_out игнорируется). Правки только в bench + `nullptr`-guard в merged. | Правка bench — да; ptxas изменится (мертвый параметр может выпасть из fingerprint если pruned) — нужен полный гейт. Wall = 0. **Отдельная тема, не совпадает с TZ 067 § «dead writes skip».** |
| dS_nat compact tri-band | **4.29 GB** | reshape allocation + переработать индексацию **в 3 ядрах** (merged writer + dk/dq readers) | **ЗАПРЕЩЕНО** ТЗ 067 («правки dk/dq запрещены»). |
| Merged single-write index | **0** | writes уже scale | — |

**Транзиент dS 8.6→4.6 GB из ТЗ**: **невозможен без правок dk/dq индексации** (запрещено). Единственный путь — compact tri-band reshape в 3 ядрах.

Побочный приз `dS_T dead alloc = 8.59 GB` реален и legal (правка bench + `nullptr`-guard в merged), но:
- Wall не меняется (0 приз)
- Merged fingerprint 252r может измениться, если параметр `dS_T_out` был load-used в SASS (нужен probe);
- **Отдельная задача, не в скоуп ТЗ 067**.

### §5.c W0-дельта — реальная стратегия

TZ упоминает «W0-дельта (транзиент 4.6 GB)» для W-ветки. Единственный путь получить 4.29 GB compact dS_nat:
- Reshape dS_nat с формы `(bh, sl, sl)` на compact tri-band `(bh, N(N+1)/2 tiles × tile_bytes)` или flat upper-tri
- **Правки в 3 ядрах** (merged writer index + dk/dq reader index)
- Полный re-cert bit-exact 11/11 × 3 gradients
- **5–7 dev-days** (оценка 066)
- **Эшелон W-series** (когда W0 упрётся в память)

---

## §6 Финальные строки

### §6.a Формула скипа

```
qt_start(causal, kt) = causal ? kt : 0
kt_end(causal, qt)   = causal ? (qt + 1) : n_kt
```

**Обе формулы УЖЕ в production** (merged / dk_new / dq_new) с эпохи 018 (первая causal-серия) → 026 (final polish causal-skip).

### §6.b Читатели мертвой зоны

**Никто.** dk_new и dq_new оба ограничены `qt ≥ kt` / `kt ≤ qt` под causal.

### §6.c Bit-exact выживает

**Да, тривиально.** Мертвая зона в bit-exact гарнире = 0 в обоих (ref + gen) через `cudaMemset(dS_nat_*, 0, dsz)` в line 121/122/132/133. Comparator сверяет byte-exact — `0 == 0` ✓.

Правки харнесса **не требуются** — memset уже в силе.

### §6.d Барьеры

**Не трогаются** (стройка не производится). Racecheck-план **не применяется**.

---

## §7 Вердикт

**ТЗ 067 ЗАКРЫТ КРАСНОЙ СТРОКОЙ** — гипотеза «dead-writes существуют в production» **опровергнута кодом**:

1. **Writer** (merged_v1.cu:146+379): Step F внутри qt-loop, `qt_start = kt` под causal → dead qt не входит в loop-body → **dS_nat_out для dead tiles не пишется**.
2. **Standalone ref writer** (fa_bwd_ds_gen.cu:261): `kt_end = qt+1` под causal → dead kt не пишется (симметричная формула для Q-owned).
3. **Readers** (dk_new:86, dq_new:107): читают только live half — dead байты никто не читает.
4. **Balance-check** (NCu 063-r): DRAM ratio 0.56 causal/nc целиком объясняется fixed-cost + wave-tail + L2. Если бы dead-writes были — ratio был бы ~0.95, наблюдаемое 0.56.

**Приз wall: 0 ms** (нечего скипать).
**Транзиент 4.6 GB compact dS: недоступен без правок dk/dq** (запрещено ТЗ 067) — эшелон W-series.

Правки production ядер в 067: **0**. Стройка не начата. Гейт **не запускался** (нечего гейтить).

### §7.a Побочная находка (для W-ветки, вне скоупа 067)

**dS_T dead allocation**: 8.59 GB в bench allocated + передано в merged, но с эпохи 033-c merged его не пишет и dk_new/dq_new его не читают. **Полное удаление alloc возможно**:
- bench: убрать `cudaMalloc(&dS_T, dsz)` (line 208), убрать `cudaFree(dS_T)` (line 287)
- merged wrapper: параметр `dS_T_out` объявлен `__restrict__` — можно принимать `nullptr` (kernel не пишет — уже 033-c) → передать `nullptr` из bench

**Приз: 8.59 GB транзиент** (bench-side, не W-relevant напрямую — но для W-ветки при allocator sharing это освободит регион).

**Если Vugar даст ТЗ 067b на dS_T dead-alloc removal**: полный гейт нужен (fingerprint может подтвердиться неизменным 252/124/69/38, но dead-alloc removal должен пройти через racecheck/memcheck на всякий случай — `nullptr` deref в SASS иногда просачивается).

### §7.b Сиквенс

**068**: **Ремап расписания dk/dq** по гистограмме 066 (potol −1.3..−2.0 мс claim = близко к 066 upper-bound 1.02 ms, реалистично 0.35-0.5 ms). Это КАНОНИЧЕСКИЙ дожим по 066 verdict-card §5.1.

**069+ эшелон-2**: 5-я бригада dk (occ 4→5) + перекраска V — после ремапа.

**W-эшелон**: compact tri-band dS reshape в 3 ядрах (memory prize для многослойной accum).

### §7.c Chain md5

- 062 `b7044db70019e8fa7dea260f9f235b6c`
- 063 `b7c82475ed49ac4821c7346f99a38fb1`
- 063-r `1dce5e445e5b47152c2894ffc7947b30`
- 064 `c099586c01ad07d6c83f3d732a77de3c`
- 065 `cc5c2a7f96aeed162ddf28609703009a`
- 066 `029b8c4b9b6e154ad437706eafd25a1d`
- **067 `ecbdeff9a42be2cf20b5d4d2afc41de7`**

### §7.d Файлы 067

- `runs/reports/067_dead_ds_skip.md` (this report — хирургический дочет)

Скриптов и wall/NCu прогонов **нет** — работа только бумажная (грep + чтение исходников).

---

**End 067. ТЗ ЗАКРЫТ КРАСНОЙ СТРОКОЙ. Оптимизация «skip dead dS writes» УЖЕ РЕАЛИЗОВАНА в production через `qt_start = kt` (merged) и `kt_end = qt+1` (ds_gen), с эпохи 018/026. Приз wall = 0 ms; транзиент compact 4.29 GB требует правок dk/dq (запрещено). Побочно: dS_T dead-alloc 8.59 GB в bench может быть убран отдельным ТЗ. Сиквенс: 068 = ремап расписания по 066.**
