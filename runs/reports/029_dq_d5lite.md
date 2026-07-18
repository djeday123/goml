# 029 — dq_new связка D5-lite + pack+π: long_sb побит на порядок, но wall стоит → **большой стоп-разбор** (правило-2/3 v2 <2%)

**Chain**:
- 028_dq_longsb.md md5: `e8e5b806943db6f422034f02c91a29f3`

**Artifact header** (после отката):
```
-rw-r--r-- 13352  Jul  5 17:51  libs/fa_bwd_dq_new.cu   (sealed pre-pack, md5 683396f8…)
-rwxr-xr-x         Jul  7        libs/r1c_dq_wall        (56r restored)
-rwxr-xr-x         Jul  7        libs/bench_r2c_e2e      (fingerprint 56 restored)
-rw-r--r--         Jul  7        runs/archive/029_d5lite_pack_pi/fa_bwd_dq_new.cu  (md5 d7a11a3d…)
-rw-r--r--         Jul  7        runs/archive/027_post_pack_pi_dq/                (md5 fe02e056…)
-rw-r--r--         Jul  7        runs/archive/025_post_pack_dq/                    (md5 8811e45d…)
-rw-r--r--         Jul  7        runs/archive/024_pre_pack_dq/                     (md5 683396f8…)
```

---

## 0. Reconcile occupancy 4.9 vs 23.45 (пункт 0 ТЗ — блокирующий)

**Из одного свежего прогона NCu на sealed pre-pack**:

| Метрика | Значение | Роль |
|:--|--:|:--|
| sm__warps_active.avg.per_cycle_active | **23.45 warps/SM** | ACTIVE (resident + running) |
| sm__warps_active.avg.pct_of_peak_sustained_active | 48.86% | vs peak 48 |
| smsp__warps_eligible.avg.per_cycle_active | **0.82 warps/scheduler** | ELIGIBLE ready-to-issue |
| smsp__warps_eligible.avg.pct_of_peak | 6.82% | 0.82/12 max per scheduler |
| **Eligible total per SM** | **3.28 warps** (0.82 × 4 schedulers) | Реально issue |
| smsp__issue_active.avg.per_cycle_active | 0.29 | 29% cycles issue instruction |
| sm__ctas_active.avg | 5.86 blocks/SM | Occupancy ~6 ✓ |

**Ratio eligible/active = 3.28 / 23.45 = 14%** — 86% resident warps застряли в стойлах.

### Причина расхождения 024 vs 028

**024 запись "4.9 warps/SM"** — **арифметическая ошибка интерпретации**: я умножил 48.86% на 10, полагая peak = 10 (устаревшая архитектура). Правильно peak = 48 warps/SM (Blackwell sm_120a) → 23.45 warps active. 028 запись 23.45 верна.

**НО**: цифра "4.9" случайно попала в правильный диапазон eligible warps (3.28 ~ близко). **Термины были перепутаны**, диагноз "конвой cp.async" оставался верным.

### В техлог (Vugar-формулировка):

> **«Occupancy высокий (23.45 warps active, 98% от теор.), eligible низкий (3.28 warps/SM, 14% от active) = конвой cp.async, диагноз 028 подтверждён с уточнением терминов. 024 запись 4.9 warps/SM была арифметической ошибкой (× 10 вместо × 48); совпадение с eligible ~3.28 — случайность.»**

Расхождение <15% через объяснение → **пункт 0 закрыт, продвижение разрешено**.

---

## 1. D5-lite правка

Из fa_bwd_dq_new.cu (D5-lite поверх pack+π архива `fe02e056…`):

**Изменение**: single commit → **два commit_group** (K, dS), wait<1> для K → Phase 1.5 → wait<0> для dS.

```c
cp.async K (8 KB); cpa_commit();      // group_0
cp.async dS_nat (4 KB); cpa_commit(); // group_1

cpa_wait<1>();      // K ready, dS in flight
__syncthreads();    // BARRIER #1: K ready

Phase 1.5 K→K_T [read + write to smK_area]
                    [overlap с dS trailing cp.async]

cpa_wait<0>();      // dS ready
__syncthreads();    // BARRIER #3: K_T + dS ready

MMA-C loop
__syncthreads();    // BARRIER #4: end of kt
```

**Инварианты**:
- SMEM Δ = 0 (существующие буферы smK_area + smdS)
- Blocks/SM: 6 сохраняется
- Barrier count: 4 (unchanged)
- fp16-acc порядок kt/kb/ni: **не тронут** ✓

## 2. pack+π реанимация

Из архива `runs/archive/027_post_pack_pi_dq/` (md5 `fe02e0567b2341d81126e854cc208a69` сверен ✓). Дифф D5-lite наложен поверх.

**Итоговый диф**: D5-lite + pack+π **одним диффом**, source md5 `d7a11a3d788eb4c396d892bc9c8ab754` (архив 029_d5lite_pack_pi/).

---

## 3. Гейты (a)-(e)

### (a) ptxas

- **69r** / 0 spill / 0 stack / 1 barrier
- Прогноз 70-76r → факт **69r** (ниже прогноза на 1r vs pack+π-only 70r)
- Потолок 85r для 6 blocks → **запас +16r**
- **π_V + D5-lite compact**: -1r vs pack+π only (компилятор оптимальнее с split commit)

### (b) fingerprint

```
FINGERPRINT kernel_d_precompute    numRegs= 38 (expected  38) OK
FINGERPRINT kernel_merged_v1       numRegs=253 (expected 253) OK
FINGERPRINT kernel_dk_new          numRegs=124 (expected 124) OK    ← unchanged (соседи)
FINGERPRINT kernel_dq_new          numRegs= 69 (expected  69) OK    ← 56→69
```
SMEM 13824 unchanged ✓. 6 blocks/SM ✓. dk_new+merged not touched ✓.

### (c) triple bit-exact + fp16-acc + sanitizer

- **BIT-EXACT 11/11 + CANARY** (dK+dV+dQ) ✓
- **fp16-acc order kt→kb→ni preserved** (MMA-C loop не тронут) ✓
- **Sanitizer 0 errors** ✓

### (d) Wall session-pair 5+5

**Baseline (sealed pre-pack, md5 683396f8…)**:
```
Runs: 8.521 8.534 8.524 8.522 8.529
Median: 8.524 ms (CV 0.06%)
```

**Post D5-lite+pack+π (69r)**:
```
Runs: 8.394 8.394 8.396 8.395 8.392
Sorted: 8.392 8.394 8.394 8.395 8.396
Median: 8.394 ms (CV 0.02%)
```

**Δ vs baseline: -0.130 ms = -1.52%**

**Правило-2/3 v2**:
- ≥3% keep → 8.269 — не пройден
- 2-3% ABBA → 8.270-8.353 — не в вилке (8.394 > 8.353)
- **<2% → откат связки + **большой стоп-разбор** (по TZ пункт 4)

### (e) NCu-сверка предсказаний

| Метрика | pre (024) | pack+π (027) | **D5-lite+pack+π (029)** | Прогноз | Verdict |
|:--|--:|--:|--:|:-:|:-:|
| **long_sb** | 10.22% | 14.88% | **2.37%** | 8-11% | ✓✓ **побит на порядок** |
| barrier | 10.93% | 9.54% | **11.90%** | ≤ +1.5 pp | **✗ +2.36 pp** (превышение +0.86 pp) |
| **mio_throttle** | 46.73% | 35.47% | **41.44%** | ~36 | **✗ вырос обратно +5.97 pp** |
| wait | 9.11% | 10.91% | 11.32% | — | +0.41 pp |
| **not_selected** | 8.94% | 11.52% | **13.25%** | — | ✗ +1.73 pp (scheduler pressure) |
| short_sb | 5.97% | 3.26% | 4.69% | — | +1.43 pp |
| **LD conflicts** | 541 M | 1.95 M | **2.17 M** | ~2M | ✓ π работает |
| **ST conflicts** | 60.9 M | 30.3 M | **36.6 M** | ~30 M | ~ ✓ |
| shared_st inst | 537 M | 134 M | 134 M | unchanged | ✓ |
| shared_ld inst | 889 M | 889 M | 889 M | unchanged | ✓ |

**Ключевое**:
- long_sb упал на **12.5 pp** (14.88 → 2.37) — **побит на порядок**, прогноз (8-11) сильно занижен
- Но wall drop **-1.52%** практически совпал с pack+π-only (-1.47%)
- **Выигрыш long_sb полностью поглощён MIO+barrier+not_selected+short_sb**

---

## 4. **Большой стоп-разбор** (по TZ пункт 4)

### 4.1 Что произошло

D5-lite ФАКТИЧЕСКИ сработал — long_sb упал на порядок. Но wall не сдвинулся. Значит **long_sb не был единственным критпутём**.

### 4.2 Ре-атрибуция: dq_new — balanced kernel

Распределение stall в baseline:
- MIO 46.7% + long_sb 10.2 + barrier 10.9 + wait 9.1 + not_sel 8.9 + short_sb 6.0 + selected 5.0 = 96.8%
- **Никакой одиночный класс > 50% — все делят критпуть примерно поровну (кроме MIO)**

Каждая правка убирает один класс, но открывает другие:
- **pack соло**: MIO -10.4, long_sb +4.65, ST conflicts +146M → net -1.43%
- **pack+π**: LD -99.6% ST -80%, но long_sb 14.88% (не улучшилось), MIO 35.47 → net -1.47%
- **D5-lite+pack+π**: long_sb -12.5 pp, но MIO +5.97, barrier +2.36, not_selected +1.73 → net -1.52%

**Каждая итерация даёт ~1.5% wall drop**, независимо от которое лекарство. **Это структурная особенность dq_new**: балансирован между 5 классами стойлов.

### 4.3 Что это значит

**Long_sb НЕ был критпутём**. Диагноз 028 «cp.async конвой» был **симптомом**, а не первопричиной. Первопричина — **балансированное распределение stall'ов на 5 классов**.

Ни одно локальное лекарство не преодолеет 3% keep-порог. Нужен **системный подход**:

**Кандидаты для большого стоп-разбора**:

- **A) Переработать SMEM tile size**: Bc=64 → 128 или другое. Изменит balance, может open большое горло.
- **B) Изменить acc type**: fp16 → fp32 (жёсткий инвариант bit-exact, не для этой сессии)
- **C) Reshape MMA structure**: kb outer/ni inner → альтернатива. Тоже инвариант bit-exact.
- **D) Двойной буфер K/dS (D5-full)**: 6→4 blocks регресс, требует явного решения Vugar
- **E) Registration через отдельное ядро**: **переключение на другое ядро** (dv_mma_p1, merged_v1) — уйти от dq_new

**Рекомендация**: **E — переключение на другое ядро**. dq_new достиг локального оптимума (~8.4 ms, ~262 T). Дальше требуется либо структурная перестройка (A/C/D) с высоким риском, либо смена фронта.

### 4.4 Морозилка

Все 4 архива сохранены на случай возврата:
- `024_pre_pack_dq/` (sealed base)
- `025_post_pack_dq/` (pack соло, md5 `8811e45d…`)
- `027_post_pack_pi_dq/` (pack+π, md5 `fe02e056…`)
- `029_d5lite_pack_pi/` (composite, md5 `d7a11a3d…`)

Ни один не активирован в production. dq_new остаётся на sealed baseline 56r.

---

## 5. Откат штатный

- `libs/fa_bwd_dq_new.cu` ← `runs/archive/024_pre_pack_dq/` (md5 `683396f8…`)
- `libs/r1c_dq_wall` rebuilt: **56r ✓**
- `libs/bench_r2c_e2e` rebuilt: fingerprint 56 restored, chain 11/11 OK
- **No 029_sealed архив** (не KEEP по правилу)

---

## 6. Вердикт в техлог

- **D5-lite работает по механике** (long_sb -12.5 pp, побил прогноз на порядок), но **wall gain съедается**
- **dq_new — balanced kernel**: 5 классов стойлов делят критпуть; локальные лекарства выигрывают ~1.5% wall и остаются под 2%
- **Пункт 4 стоп-разбора**: long_sb не был критпутём, была симптомом
- **Рекомендация**: смена ландшафта (E) — переключение на другое ядро (dv_mma_p1 / merged_v1 / dv_baseline)

Chain md5: 028 `e8e5b806…` → **029 `<computed>`**

---

**End 029.**
Большой стоп-разбор активирован. dq_new паркуется на sealed baseline (56r, 8.52 ms). Ожидаю решение Vugar о смене фронта.
