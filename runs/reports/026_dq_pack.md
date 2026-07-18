# 026 — dq_new pack K_T scatter: правка + гейты → **откат штатный** (правило-2/3 v2)

**Chain**:
- 024_dq_census.md md5: `8462480a45d88dfcbcaabac3599dd2ff`
- 025_dq_pack_paper.md md5: `23425012f92d20f5f3da35267accbc06`

**Artifact header** (после отката):
```
-rw-r--r-- 13352  Jul  5 17:51  libs/fa_bwd_dq_new.cu       (sealed pre-pack, md5 683396f8…)
-rwxr-xr-x         Jul  7        libs/r1c_dq_wall            (56r restored)
-rwxr-xr-x         Jul  7        libs/bench_r2c_e2e          (expected 56 restored)
-rw-r--r-- 14567  Jul  7        runs/archive/025_post_pack_dq/fa_bwd_dq_new.cu  (md5 8811e45d…, сохранён)
-rw-r--r--         Jul  7        runs/archive/024_pre_pack_dq/fa_bwd_dq_new.cu   (md5 683396f8…)
```

---

## 1c. Production правка

Изменён Phase 1.5 write (fa_bwd_dq_new.cu:187-195) — 64 STS.U8/lane → 12 SHFL + 16 STS.32.
Все прочие места (feeder, MMA-C, epilogue, cp.async) не тронуты.
fp16-acc order kt→kb→ni сохранён (жёсткий инвариант).

Post-pack source md5: **`8811e45d440622b9c49d89c9ee687ad6`** (архив `runs/archive/025_post_pack_dq/`).

---

## 2. Гейты (a-d)

### (a) ptxas — PASS

- **72r** / 0 spill / 0 stack / 1 barrier
- Прогноз 73-78r → факт **72r** (на 1 регистр НИЖЕ вилки, лучше)
- Потолок 85r для 6 blocks — **под потолком, запас +13r**
- 6 blocks/SM maintained ✓

### SASS gates (pack scatter K_T)

```
SHFL:    12   ✓ (target 12)
STS.32:  16   ✓ (target 16, pack)
STS.U8:   1   ★ pre-existing в partial-CANARY-fallback (не в K_T pack)
LDL/STL:  0   ✓
```

STS.U8=1 — не в K_T scatter (см. fa_bwd_dq_new.cu:154-163, partial dS_nat OOB fallback, был до правки). Инвариант "0 STS.U8 в pack" пройден.

### (b) fingerprint — PASS

```
FINGERPRINT kernel_d_precompute    numRegs= 38 (expected  38) OK
FINGERPRINT kernel_merged_v1       numRegs=253 (expected 253) OK
FINGERPRINT kernel_dk_new          numRegs=124 (expected 124) OK    ← unchanged (соседи)
FINGERPRINT kernel_dq_new          numRegs= 72 (expected  72) OK    ← updated 56→72
```
- SMEM 13824 B unchanged ✓
- 6 blocks/SM ✓
- dk_new + merged соседи не тронуты (fingerprint unchanged)

### (c) triple bit-exact + CANARY + sanitizer — PASS

```
=== CHAIN BIT-EXACT SUMMARY ===
  forms all-3 bit-exact: 11 / 11   (все dK+dV+dQ по 10 форм + CANARY)

========= ERROR SUMMARY: 0 errors (compute-sanitizer memcheck)
```

fp16-acc floor-константы: kt→kb→ni order preserved (правка не тронула MMA loop) ✓.
Chain bit-exact ×1 (dk_new/merged соседи не тронуты) ✓.

### (d) Wall session-pair 5+5 — **Δ = -1.43%**

**Baseline (pre-pack, sealed 683396f8…)**:
```
Runs: 8.504 8.504 8.511 8.525 8.526
Median: 8.511 ms (CV 0.11%)
```

**Post-pack (72r)**:
```
Runs: 8.390 8.390 8.389 8.389 8.391
Median: 8.389 ms (CV 0.03%)
```

**Δ vs baseline: -0.122 ms = -1.43%**

**Правило-2/3 v2**:
- ≥3% keep → **8.256** — не пройден
- 2-3% ABBA → 8.257 - 8.341 — вне вилки
- **<2% → откат штатный** — **8.389 > 8.341** → откат

### (e) NCu-сверка предсказаний поимённо

**Post-pack на r1c_dq_wall (post) vs 024 baseline (pre-pack)**:

| Метрика | pre (024) | post | Δ | Прогноз (025) | Verdict |
|:--|--:|--:|:-:|:-:|:-:|
| **shared_st inst** | 537 M | **134 M** | **-75%** | -70..75% | ✓ **прогноз точно** |
| **mio_throttle** | 46.73% | **36.38%** | **-10.4 pp** | 34-38% | ✓ **в вилке** |
| shared_ld inst | 889 M | 889 M | 0% | unchanged | ✓ |
| LD conflicts | 541 M | 543 M | +0.4% | unchanged | ✓ |
| **ST conflicts** | 60.9 M | **150 M** | **+146%** | (не прогноз) | ↑ pack layout |
| short_sb | 5.97% | **2.94%** | -3.03 pp | — | ✓ |
| barrier | 10.93% | 9.52% | -1.4 pp | — | ✓ |
| long_sb | 10.22% | **14.87%** | **+4.65 pp** | — | ✗ вышло вперёд |
| wait | 9.11% | 11.05% | +1.94 pp | — | ↑ |
| not_selected | 8.94% | 11.28% | +2.34 pp | — | ↑ |

**Ключевое**:
- **shared_st inst -75% + MIO -10.4 pp — оба прогноза попали точно.**
- **wall drop только -1.43%** (прогноз 2.2..4.9%) — **не соответствует MIO drop**.

---

## 3. Диагноз перераспределения стойлов

MIO упал на 10.4 pp, но остальные стойлы выросли: long_sb +4.65, wait +1.94, not_selected +2.34, ST conflicts +146%. Net drop ≈ 10.4 - 8.9 (сумма ростов) ≈ 1.5 pp — соответствует наблюдаемому wall -1.43%.

**Причины роста**:
- **ST conflicts удвоились (60→150M)**: post-pack STS.32 layout по col_words c KT_STRIDE=68 попадает в bank-конфликтный паттерн (P16-класс без π).
- **long_sb вырос**: LDG cp.async wait стал заметнее после MIO drop.
- **not_selected вырос**: warp scheduler starvation (меньше issue-ready warps).

**Vugar decision tree**: 
- «π-цикл если conflicts/short_sb вышли вперёд» — **ST conflicts выросли +146% → π-цикл релевантен** (post-pack раскладка требует π-фикса)
- «barrier-разбор D4» — barrier наоборот упал (9.52), не релевантно.

---

## 4. Откат штатный

- `libs/fa_bwd_dq_new.cu` ← `runs/archive/024_pre_pack_dq/` (md5 `683396f8e6867e9fc2e26f8b628774f3`)
- `libs/r1c_dq_wall` rebuilt: 56r ✓
- `libs/bench_r2c_e2e` rebuilt: fingerprint expected 72→56 restored, chain OK
- Post-pack источник сохранён в `runs/archive/025_post_pack_dq/` (md5 `8811e45d…`) на случай будущих ревизий

---

## 5. Вердикт в техлог

- **Pack scatter K_T в dq_new**: **инструктивно работает** (shared_st inst -75%, MIO -10.4 pp — оба NCu-прогноза попали точно), но **wall выигрыш поглощён перераспределением стойлов** (long_sb, wait, not_selected, ST conflicts все выросли).
- **Правило-2/3 v2 отрабатывает**: <2% wall → откат без спорa.
- **Новый факт для техлога**: в dq_new (в отличие от dk_new) drop MIO **более чем на 60% съедается** ростом cp.async + ST-conflict пула. Т.е. dq_new **не MIO-bound**, а MIO-**и**-LDG-bound (совместный горло).
- **Следующее лекарство (по TZ)**: **π-цикл** под фактическую пост-pack раскладку — целится в 150M ST-conflicts. Возможен keep, если π-compute overhead будет меньше чем в dk_new (или compensate at short_sb / long_sb).

---

## 6. π-бумага (paper-only, для 027 стартового цикла)

**Из TZ**: "π-бумага из прежнего ТЗ — **после зелёного юнита, под фактическую новую раскладку**." Юнит зелёный (8192/8192), pack применён. Разрабатываю π-бумагу отдельно, будет в 027 (не в этом отчёте — правка отката приоритетнее).

Кандидат-формула: PI_V (bit-perm r0→2, r1→3, r2→4, r3→1, r4→0) применена к базовому row K_T:
- row_base_new = ks*32 + 4c + p + 16*slot_half → PI_V(row_base_new)
- Ожидаемый ST conflict drop: 150M → <25M
- Ожидаемый LD conflict drop: 543M → ~150M (B-load тоже π_V)
- Register cost: +5-8r (в окне 85r для 6 blocks)
- Wall прогноз (paper): TBD после CPU-судьи (метод 021).

---

## 7. Файлы

- Prod dq_new: sealed pre-pack (056r)
- Prod dk_new: sealed π_V (124r, 023 KEEP)
- Post-pack dq архив: `runs/archive/025_post_pack_dq/` (для π-цикла reference)
- Pre-pack dq архив: `runs/archive/024_pre_pack_dq/`

Chain md5: 025 `23425012…` → **026 `<computed>`**

---

**End 026.**  
Откат dq_new штатный. Ожидаю решение Vugar: (i) π-цикл под пост-pack раскладку в 027, (ii) barrier D4 разбор, (iii) переключение на dv_mma_p1 или merged_v1.
