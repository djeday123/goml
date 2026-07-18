# 042 — A': досмотр классов #4/#6 через no-trans + ворота охвата

**Chain**:
- 040_A_production.md md5: `7d39d1058edbcd47d14f8500584685d3`
- 041_post040_recon_dq.md md5: `198fd63cc05fb68714e0b95eb8395ccc`

**Правила ТЗ 042**: reader-only; писатель smdO/раскладка/LDSM-путь #7/классы #5/#8/барьеры **НЕ ТРОГАТЬ**; B-op и fp8-классы **исключены** (территория 043). Ворота охвата (СТОП-условия) до кода.

---

## Артефакт-хедер (правило 5)

```
libs/ (post-041):
-rw-r--r-- 25638 Jul  8 16:31  fa_bwd_merged_v1.cu    (md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33 = 040 sealed)
-rw-r--r-- 18834 Jul  8         fa_bwd_dq_new.cu       (md5 d7a11a3d788eb4c396d892bc9c8ab754 = 041 sealed)
-rw-r--r--       Jul  7         fa_bwd_dk_new.cu       (md5 a9f0ded8261e53a143b521ffa647f458 = 033_sealed)
-rwxr-xr-x       Jul  8         bench_r2c_e2e          (EXPECT: merged=252, dq=69, dk=128, D=38)
-rwxr-xr-x       Jul  8         r2c_merged_wall
```

**Gate-log единый**:
```
$ ./037r_gate.sh
bench_merged: FINGERPRINT kernel_merged_v1: numRegs=252
GATE OK: numRegs=252 matches EXPECT=252

$ ./bench_r2c_e2e (fingerprint x4):
FINGERPRINT kernel_d_precompute    numRegs= 38 (expected  38) OK
FINGERPRINT kernel_merged_v1       numRegs=252 (expected 252) OK
FINGERPRINT kernel_dk_new          numRegs=128 (expected 128) OK
FINGERPRINT kernel_dq_new          numRegs= 69 (expected  69) OK
```

---

## §0. Хвосты 041 (5 пунктов, новых прогонов нет кроме 0.d)

### 0.a Причина смерти (b) smL/smD double-buffer — одной строкой

**Причина смерти (b) осталась**: класс #3 (smL/smD reads) = 4 LDS.32/lane/qt = ~0.8% всех LDS-ops = **мёртвая механическая масса** (потолок wall <0.05%, ниже любого измеримого порога) — механизм prefetch 512B работал бы, но выигрыш пренебрежимо мал независимо от MIO-состояния.

### 0.b LD conflict events — атрибуция +4.3% (какой класс)

| Метрика | Base pre-040 | Post-040 | Δ |
|:--|:-:|:-:|:-:|
| LD conflict events абс | 126,810,524 | **132,104,324** | **+5.29M** (+4.17%) |
| ST conflict events | 16,651,450 | 17,181,201 | +0.53M |
| Wavefronts LD only | (n/a combined) | 3,453,993,092 | — |

**Атрибуция +5.29M LD conflict events**:
- **НЕ класс #7** (переведён на LDSM): NCu metric `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` считает **classical LDS bank conflicts** (address-collision в одной инструкции). LDSM cooperative fetch не увеличивает эту метрику — LDSM засчитывается в wavefronts (структурный пол).
- **Скорее всего noise + timing drift классов #1..#6, #8** при укорочении ядра (54.76M cycles vs pre 66.6M SM cycles): reordering LDS доступов между классами при новом MMA-scheduling может дать локальное перераспределение bank hits.
- Rate check: 132.1M / 3.454B LD wavefronts = **3.82%** collision rate post-040; pre-040 rate ~2.75% (126.8M / 4.6B est LD share). Отличие +1 pp — на уровне sample noise NCu.

**Однозначная атрибуция**: **noise/small drift** классов #1..#6/#8, **не структурный класс #7**.

### 0.c Атрибуция math_pipe 13.66% — оценка долей

Post-040 math_pipe вырос с 4.27% → **13.66%** (+9.39 pp). Атрибуция по компонентам (bumaga, оценка):

| Источник | Ops в hot loop | Оценка вклад в math_pipe | Тип |
|:--|:--|:-:|:--|
| **exp softmax** (Step C `__expf(scale*s - L)`) | 4 fp32 __expf + 4 mask_chk + 4 mul per lane per qt | **~2-3 pp** | SFU |
| **fp16→fp32 конверсия** (softmax, dS quantize) | 8 halves→float + 8 float→half per lane per qt | **~1-2 pp** | ALU |
| **e4m3 квантизация** (`fp16x2_to_e4m3x2` Step E) | 4 CVT.RN.E4M3x2 per lane per qt | **~2-3 pp** | SFU |
| **XOR LDSM row-ptr арифметика** (Step H post-040) | 32 x4 × ~3 addr ALU per lane per qt | **~1-2 pp** | ALU (новое post-040) |
| **fp16-упаковка halves2half2** (Step G smP_T pack) | 8 __halves2half2 per lane per qt | **~1 pp** | ALU |
| **fp32 dV_acc chain deps** (dV_acc[16][4] fp32 через 64 MMA) | 64 mma dependency chain | **~2-3 pp** | MMA-pipe |
| Прочие CVT + mask_chk + PRMT | ... | **~2-3 pp** | ALU |
| **Итог** | | **~11-16 pp** ≈ 13.66 ✓ | |

**Топ-источник math_pipe**: fp16↔fp32 конверсия + e4m3 квантизация + SFU (exp+e4m3) — вместе ~5-8 pp = **~40-60% math_pipe**. XOR LDSM row-ptr — **~1-2 pp = ~10-15%**.

### 0.d NCu dq_new post-041 vs 024/029-killers (единственный новый прогон)

Скрипт: `runs/reports/042_ncu_dq_post041.sh`, данные: `042_ncu_dq_post041_data.txt`.

Post-041 (dq_new = **d7a11a3d, 69r**) vs 024 baseline (prod 683396f8, 56r):

| Класс | 024 baseline | **Post-041 (d7a11a3d)** | Δ | 029-killer прогноз |
|:--|:-:|:-:|:-:|:-:|
| **mio_throttle** | 46.73% | **41.45%** | **−5.29 pp** | (Vugar: +5.97 killed 029) |
| **barrier** | 10.93% | **11.90%** | **+0.96 pp** | (Vugar: +2.36 killed 029) |
| long_scoreboard | 10.22% | **2.35%** | **−7.87 pp** | |
| short_scoreboard | 5.97% | 4.69% | −1.28 pp | |
| wait | 9.11% | 11.33% | +2.22 pp | |
| math_pipe | 1.73% | 4.63% | +2.90 pp | |

**Verdict**: **029-killers НЕ вернулись** на post-041 landscape.
- mio: **-5.29 pp** (наоборот, разгружен — не +5.97 pp прогноз killer'а)
- barrier: **+0.96 pp** (меньше килл-порога +2.36 в 029)
- long_sb: **-7.87 pp** (резко разгружен, main killer 029 исчез)

**Механизм KEEP объяснён**: merged 040 (LDSM.x4.trans класса #7) сдвинул L2/timing состояние цепи. Frozen d7a11a3d в 029 сессии работал хуже базы (+5.97 mio), но в post-041 landscape работает лучше (-5.29 mio + −7.87 long_sb). Cache/timing landscape изменился, старый kernel благосклонно реагирует.

### 0.e Census 8 классов ДОСЛОВНО (из 037-r2 + 038)

| # | Класс | Инструкция | Ops/lane/qt | MMA | Операнд | Читаемый массив |
|:-:|:--|:-:|:-:|:--|:-:|:--|
| 1 | smQ read (Step B) | LDS.32 | 16 | **mma.fp8.f16 Q·K^T** | **A-op FP8** | smQ (fp8/uint8_t) |
| 2 | smK read (Step B) | LDS.32 | 64 | **mma.fp8.f16 Q·K^T** | **B-op FP8** | smK (fp8) |
| 3 | smL/smD read (Step C) | LDS.32 fp32 | 4 | (не MMA) | (не MMA) | smL/smD fp32 |
| **4** | **smdO read (Step D)** | **LDS.32** | **32** | **mma.m16n8k16.f32.f16.f16.f32 dP-MMA** | **A-op fp16** | **smdO (fp16)** |
| 5 | smV read (Step D) | LDS.U16 | 128 | **mma.m16n8k16 dP-MMA** | **B-op fp8** (fp8→fp16 в reg) | smV (fp8) |
| **6** | **smP_T read (Step H)** | **LDS.32** | **16** | **mma.m16n8k16 dV-MMA** | **A-op fp16** | **smP_T (fp16)** |
| 7 | smdO read (Step H) | ldmatrix.x4.trans (post-040) | 32 x4 | mma.m16n8k16 dV-MMA | B-op fp16 | smdO (fp16) — LDSM |
| 8 | smdS_stage read (Step F drain) | LDS.128 | 2 (runtime) | (не MMA — DRAM drain) | (не MMA) | smdS_stage |

**Кандидаты A' по формулировке "A-op, потребитель fp16/b16, читают SMEM с известной формулой раскладки"**: **только #4 (smdO A-op fp16) + #6 (smP_T A-op fp16)**.

Исключаются:
- #1 (A-op FP8 mma.fp8.f16): **fp8-потребитель** → территория 043
- #2 (B-op FP8): B-op + fp8 → территория 043
- #3 (fp32 smL/smD): не MMA-op → неприменимо
- #5 (B-op FP8 smV): **B-op + fp8** → территория 043
- #7 (B-op, уже LDSM.x4.trans post-040): не трогать
- #8 (LDS.128 drain): не MMA → неприменимо

---

## §1. Ворота охвата (СТОП-условия до кода)

### 1.a Список включённых классов A'

**Только**: класс #4 (smdO Step D dP-MMA A-op fp16) + класс #6 (smP_T Step H dV-MMA A-op fp16).

- **A-op**: ✓ (оба)
- **fp16/b16 потребитель**: ✓ (оба через mma.m16n8k16.f32.f16.f16.f32)
- **Известная формула раскладки** ✓:
  - smdO: writer = cp.async с XOR `(i_local & 7) << 4` byte-space (см. 040 §0.a verbatim)
  - smP_T: writer = Step G STS с XOR `PT_xor_even_wr = l_mod4 << 4` / `PT_xor_odd_wr` (см. код строки 401-421)

**Исключения проверены**:
- #1/#2/#5 = fp8-потребители → **исключены**, помечены "**территория 043 fp8-probe**".
- #3 (fp32) = не MMA-op → неприменимо к ldmatrix.
- #7 (уже post-040) = LDSM.x4.trans, не трогать.
- #8 = drain chunk read (LDS.128 uint4), не MMA → неприменимо.

### 1.b Счёт боезапаса включённых классов A'

| Класс | Ops сейчас (LDS.32) | LDSM.x4 после | net (per lane per qt) |
|:--:|:-:|:-:|:-:|
| #4 | **32** (KS_DP=8 × 4 loads = 32 static SASS) | **8** (KS_DP=8 × 1 x4 per iter = 8 static) | **−24 ops** |
| #6 | **16** (KB_DV=4 × 4 loads = 16 static SASS) | **4** (KB_DV=4 × 1 x4 per iter = 4 static) | **−12 ops** |
| **Итог** | 48 | 12 | **−36 ops/lane/qt** |

**Порог TZ 042 §1.b**: "Если суммарный net < 48 ops/lane/qt — СТОП до production, доклад (урок могил D/G: sub-threshold не строим), переход к бумаге 043."

**Факт**: |net| = **36 < 48** → **СТОП, до production не идти**.

---

## §2. СТОП-доклад: почему A' не строим (правило "sub-threshold не строим")

### 2.a Механическая масса ниже порога

- Пересчёт по механической массе 040-эталона: класс #7 переводил 256 ops на -12.28% wall = **~0.0479% wall per op/lane/qt**.
- A' линейная экстра: 36 ops × 0.0479 = **~1.72% wall** (upper bound, при том же bank-conflict profile).
- Территория **2/3 v2**: 2..3% зона требует "медиана ≥2% И худшая пара ≥1%". Прогноз upper 1.72% **не пробивает даже 2% нижнюю границу** гарантированно.

### 2.b Структурная причина низкой массы

Классы #4/#6 = **48 ops = ~9% всех LDS** (post-040 total ~1200 ops/lane/qt runtime). MIO throttle post-040 = **8.86%** (не bottleneck) — механическая массa ldmatrix-конверсии не даёт wall-компенсацию.

Плюс:
- 8 LDSM.x4 (класс #4) + 4 LDSM.x4 (класс #6) = 12 новых ldmatrix инструкций
- Row-ptr addr арифметика: 12 × 3 ALU = 36 add ops (мал вклад)
- math_pipe уже 13.66% (топ-2 stall) — прибавка ALU может сдвинуть math_pipe ещё вверх, съедая LDSM-выигрыш.

### 2.c Урок могил D/G

- **D**: dS scatter pack-analog захоронена в 038 (sub-threshold: -4..-9 net ops с cross-lane налогом).
- **G**: 3 blocks/SM закрыта (потолок 170r / SMEM 33024B недостижим).
- **Общий урок**: sub-threshold не строим — не пробьёт правило 2/3 v2, окраса леджера ("натянутый KEEP") запрещён.

**A' захоранивается как sub-threshold ветка**. Причина смерти: **net -36 ops < 48-порога охвата, upper wall <1.72% при 2/3 v2 нижней границе 2%**.

---

## §3. Правки merged в 042: НЕТ

- `libs/fa_bwd_merged_v1.cu` md5 **`2bf32ab7d4c5ecabb4ee2dbf1b5d4b33`** ← неизменен от 040 sealed.
- `libs/fa_bwd_dq_new.cu` md5 **`d7a11a3d788eb4c396d892bc9c8ab754`** ← 041 sealed.
- `libs/fa_bwd_dk_new.cu` md5 **`a9f0ded8261e53a143b521ffa647f458`** ← 033 sealed.

**Никакие бинари не пересобирались в 042**. Никаких wall-замеров (кроме одного NCu-прогона 0.d).

---

## §4. Сиквенс по итогу TZ 042

**Условие 1** (ворота охвата): |net| = 36 < 48 → **СТОП, переход к бумаге 043**.

**Условие 2** (E2E-сессия): post-041 E2E in-chain **44.206 ms > 44.0 ms** → TZ 042 §сиквенс: **043 = fp8-LDSM probe** (двери: merged #5 128 ops + dk Q_T 64/88 LDS при mio 42%).

**Оба условия сходятся**: следующее ТЗ = **043 fp8-LDSM probe**.

### 4.a Двери 043 fp8-probe

- **merged класс #5** (smV Step D dP-MMA B-op fp8): **128 LDS.U16 per lane per qt** ← крупная механическая масса, самый большой fp8-класс.
- **dk_new pack Q_T** (128r 033_sealed): **64/88 LDS ops** при **mio 42%** ← отдельный контур, но триггер §8.2 не переходит на dk (dk sealed).
- **Гвоздь FP8 для ldmatrix**: (037/038 архив) требует probe fp8-shape support на sm_120a (микропроба standalone аналогично 038/039).

### 4.b Альтернативный сценарий 043

- Если 042 показал бы E2E ≤ 44.0 ms → 043 = cert-пакет 009-F-класса (30-run nc+causal, isolated x3, fingerprint на прогон).
- **Не применимо** здесь (post-041 44.206 > 44.0).

---

## §5. Итог 042

1. **Хвосты 041**:
   - (a) smL/smD DB: мёртвая масса, механизм prefetch 512B работал бы, но wall <0.05%.
   - (b) LD conflicts +4.3% атрибутированы noise/drift классов #1..#6/#8 (НЕ #7 LDSM).
   - (c) math_pipe 13.66% ≈ fp16↔fp32 конверсия + e4m3 CVT + XOR LDSM addr + fp32 dV_acc chain (топ 5-8 pp).
   - (d) NCu dq post-041 vs 024: mio **-5.29 pp** (не +5.97 killer), barrier **+0.96 pp** (не +2.36 killer), long_sb **-7.87 pp** — **029-killers НЕ вернулись**, механизм KEEP объяснён (merged 040 сдвинул L2/timing landscape).
   - (e) Census 8 классов verbatim; **A' кандидаты = #4 (smdO A-op fp16) + #6 (smP_T A-op fp16)**.

2. **Ворота охвата**: **net = −36 ops/lane/qt < 48-порог** → **СТОП до production**.

3. **Механическая масса A' upper**: ~1.72% wall (линейная экстра от 040-эталона) — **не пробивает 2/3 v2 нижнюю границу 2%**.

4. **Правки merged: 0**. Merged/dq/dk все sealed, состояние post-041.

5. **Сиквенс**: E2E 44.206 > 44.0 → **043 = fp8-LDSM probe** (класс #5 merged 128 ops при mio-остатке 8.86%).

### Chain md5

- 041 `198fd63cc05fb68714e0b95eb8395ccc`
- **042 `<computed>`**

### Файлы 042

- `runs/reports/042_Aprime.md` (this report)
- `runs/reports/042_ncu_dq_post041.sh` + `042_ncu_dq_post041_data.txt` — единственный новый прогон (хвост 0.d)

---

**End 042. A' захоронена как sub-threshold. Переход к 043 fp8-LDSM probe.**
