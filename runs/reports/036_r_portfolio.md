# 036-r — Портфельный аудит полировок dk_new (полный квадрат)

**Chain**:
- 035_two_roads.md md5: `2a906152cff657e13fa0dea3f9ca1ef5`
- 036_calib_pi_v.md md5: `7c4cce12947488191f9517fc85855dd3`

**Artifact header** (production после аудита восстановлена в AB = 033_sealed):
```
-rw-r--r-- 14667  Jul  8 12:00  libs/fa_bwd_dk_new.cu     (variant AB = 033_sealed, md5 a9f0ded8…)
-rwxr-xr-x 2.27M  Jul  8 12:47  libs/bench_r2c_e2e_AB     (md5 b9f9ab3a…, 128r pack+π_V)
-rwxr-xr-x 2.27M  Jul  8 12:46  libs/bench_r2c_e2e_Aonly  (md5 de0685a0…, 107r pack no-π_V)
-rwxr-xr-x 2.26M  Jul  8 13:02  libs/bench_r2c_e2e_0      (md5 605199af…, 127r byte no-π)
-rw-r--r-- 14667  Jul  8        runs/archive/036_variant_A/fa_bwd_dk_new.cu (AB src, md5 a9f0ded8…)
-rw-r--r-- 14508  Jul  8        runs/archive/036_variant_B/fa_bwd_dk_new.cu (A src, md5 b36eec6e…)
-rw-r--r-- 12983  Jul  8        runs/archive/036_variant_0/fa_bwd_dk_new.cu (0 src, md5 6e1a1a40…)
```

Merged/морозилки не тронуты.

---

## 0. Калибровка стенда (наследована из 036)

Стенд-протокол зафиксирован в 036:
- **4 warmup runs обязательны** (~60s cold-clear)
- **Дрейф на 5+5 паре 0.67% > 0.5% Vugar-порог** → **ABBA default**
- Plateau с run 5, temp 44-49°C

Применён без изменений во всех парах ниже.

---

## 1. Четыре бинаря — полный квадрат полировок на базе W2

### 1.1 Вариант AB (033_sealed prod, pack + π_V)

- Source: `runs/archive/036_variant_A/fa_bwd_dk_new.cu` md5 `a9f0ded8261e53a143b521ffa647f458`
- ptxas: **128r** / 0 spill / 0 stack / 4 blocks/SM (SMEM 20992 B)
- Binary: `libs/bench_r2c_e2e_AB` md5 `b9f9ab3a266d80d9e11d00ba48e0cbc9`
- Fingerprint gate: dk_new=128 OK
- **BIT-EXACT 11/11 + CANARY** ✓

### 1.2 Вариант A (pack без π_V)

- Source: `runs/archive/036_variant_B/fa_bwd_dk_new.cu` md5 `b36eec6e96f0f6d5fde0c93cabc7d69a`
- Изменение: снятие PI_V(row) macros из Phase D pack + B-load (rows 299-302, 327)
- ptxas: **107r** / 0 spill / 0 stack / 4 blocks/SM
  - 65536/(107×128)=4.78 → 4 blocks by regs (SMEM тот же лимитер)
- Binary: `libs/bench_r2c_e2e_Aonly` md5 `de0685a0981bcc63b91b298149d1f883`
- Fingerprint gate: dk_new=107 OK
- **BIT-EXACT 11/11 + CANARY** ✓ (π_V — layout-only, снятие не меняет байты)

### 1.3 Вариант 0 (без pack, без π_V — byte scatter)

- Source: `runs/archive/036_variant_0/fa_bwd_dk_new.cu` md5 `6e1a1a404a7b64643c2b058245fdc5ad`
- Изменение: pack Phase A/B/C/D (67 строк) заменён на byte-scatter loop (13 строк):
  - Восстановлен pre-018 sealed byte scatter (64 STS.U8 per lane per qt)
  - 0 SHFL, 0 PRMT/SEL (весь cross-lane обмен убран)
- ptxas: **127r** / 0 spill / 0 stack / 4 blocks/SM
  - Прогноз был <107r; факт **127r** (byte-scatter даёт больше temp reg pressure чем pack: 64 STS.U8 с shift+mask на каждый байт → компилятор держит per-store address computations в reg-file)
  - 65536/(127×128)=4.03 → 4 blocks by regs (SMEM тот же лимитер)
- Binary: `libs/bench_r2c_e2e_0` md5 `605199af4e717d5f176aa9e38dcc9b45`
- Fingerprint gate: dk_new=127 OK
- **BIT-EXACT 11/11 + CANARY** ✓

### 1.4 Вариант B (π_V без pack) — **не строим**

Vugar TZ036-r warning: «π_V на байтовом скаттере ломала ST (P14/P15=1.00, док 016) — если сборка B требует ре-вывода π под байтовый скаттер, честно пометить B как «недостроим дёшево»».

**Ре-вывод π_V-подобного bank-perm под byte scatter требовал бы**:
- новую алгебру (64 STS.U8 vs 16 STS.32 — permutation domain меняется)
- probe P24/P25 с нуля
- отдельный unit-test
- 20-40 rows ALU в hot loop

При статусе «AB best, statu quo» — работа не окупается.

**Вариант B помечен «недостроим дёшево», из ABBA исключён.**

---

## 2. Замер по стенд-протоколу — ABBA 8 пар каждая

### 2.1 AB ↔ A (pack + π_V vs pack no-π_V)

Схема: X Y Y X X Y Y X X Y Y X X Y Y X (X=AB, Y=A)

| # | tag | temp | E2E (ms) | dk_new (ms) |
|:-:|:-:|:-:|:-:|:-:|
| 1 | AB | 43 | 47.942 | 10.383 |
| 2 | A | 42 | 48.027 | 10.489 |
| 3 | A | 43 | 48.049 | 10.495 |
| 4 | AB | 45 | 47.956 | 10.390 |
| 5 | AB | 44 | 48.013 | 10.397 |
| 6 | A | 44 | 48.141 | 10.514 |
| 7 | A | 47 | 48.149 | 10.518 |
| 8 | AB | 45 | 48.056 | 10.407 |
| 9 | AB | 44 | 48.008 | 10.400 |
| 10 | A | 46 | 48.158 | 10.522 |
| 11 | A | 45 | 48.130 | 10.514 |
| 12 | AB | 47 | 48.066 | 10.414 |
| 13 | AB | 45 | 48.006 | 10.395 |
| 14 | A | 50 | 48.205 | 10.532 |
| 15 | A | 47 | 48.161 | 10.521 |
| 16 | AB | 48 | 48.093 | 10.422 |

**Парные дельты Δ = A − AB (положит.=AB быстрее)**:

| Пара | AB | A | ΔE2E (ms) | Δ% | Δdk_new |
|:-:|:-:|:-:|:-:|:-:|:-:|
| P1 (1AB,2A) | 47.942 | 48.027 | +0.085 | +0.18% | +0.106 |
| P2 (3A,4AB) | 47.956 | 48.049 | +0.093 | +0.19% | +0.105 |
| P3 (5AB,6A) | 48.013 | 48.141 | +0.128 | +0.27% | +0.117 |
| P4 (7A,8AB) | 48.056 | 48.149 | +0.093 | +0.19% | +0.111 |
| P5 (9AB,10A) | 48.008 | 48.158 | +0.150 | +0.31% | +0.122 |
| P6 (11A,12AB) | 48.066 | 48.130 | +0.064 | +0.13% | +0.100 |
| P7 (13AB,14A) | 48.006 | 48.205 | +0.199 | +0.41% | +0.137 |
| P8 (15A,16AB) | 48.093 | 48.161 | +0.068 | +0.14% | +0.099 |

**Все 8 пар: AB быстрее A** (ΔE2E>0 единогласно).

- Sorted ΔE2E: 0.064, 0.068, 0.085, 0.093, 0.093, 0.128, 0.150, 0.199
- **Median ΔE2E = 0.093 ms = 0.19% E2E**
- Sorted Δdk: 0.099, 0.100, 0.105, 0.106, 0.111, 0.117, 0.122, 0.137
- **Median Δdk = 0.109 ms**

### 2.2 AB ↔ 0 (pack + π_V vs bare byte)

Схема: X Y Y X X Y Y X X Y Y X X Y Y X (X=AB, Y=0)

| # | tag | temp | E2E (ms) | dk_new (ms) |
|:-:|:-:|:-:|:-:|:-:|
| 1 | AB | 47 | 48.066 | 10.414 |
| 2 | 0 | 45 | 48.413 | 10.752 |
| 3 | 0 | 49 | 48.442 | 10.757 |
| 4 | AB | 47 | 48.129 | 10.430 |
| 5 | AB | 46 | 48.110 | 10.425 |
| 6 | 0 | 46 | 48.436 | 10.757 |
| 7 | 0 | 45 | 48.442 | 10.760 |
| 8 | AB | 48 | 48.107 | 10.424 |
| 9 | AB | 46 | 48.100 | 10.423 |
| 10 | 0 | 54 | 48.444 | 10.758 |
| 11 | 0 | 48 | 48.463 | 10.764 |
| 12 | AB | 47 | 48.157 | 10.437 |
| 13 | AB | 54 | 48.151 | 10.432 |
| 14 | 0 | 48 | 48.506 | 10.773 |
| 15 | 0 | 47 | 48.463 | 10.763 |
| 16 | AB | 45 | 48.114 | 10.426 |

**Парные дельты Δ = 0 − AB**:

| Пара | AB | 0 | ΔE2E (ms) | Δ% | Δdk_new |
|:-:|:-:|:-:|:-:|:-:|:-:|
| P1 | 48.066 | 48.413 | +0.347 | +0.72% | +0.338 |
| P2 | 48.129 | 48.442 | +0.313 | +0.65% | +0.327 |
| P3 | 48.110 | 48.436 | +0.326 | +0.68% | +0.332 |
| P4 | 48.107 | 48.442 | +0.335 | +0.70% | +0.336 |
| P5 | 48.100 | 48.444 | +0.344 | +0.72% | +0.335 |
| P6 | 48.157 | 48.463 | +0.306 | +0.64% | +0.327 |
| P7 | 48.151 | 48.506 | +0.355 | +0.74% | +0.341 |
| P8 | 48.114 | 48.463 | +0.349 | +0.73% | +0.337 |

**Все 8 пар: AB быстрее 0** (ΔE2E>0 единогласно).

- Sorted ΔE2E: 0.306, 0.313, 0.326, 0.335, 0.344, 0.347, 0.349, 0.355
- **Median ΔE2E = 0.340 ms = 0.71% E2E**
- Sorted Δdk: 0.327, 0.327, 0.332, 0.335, 0.336, 0.337, 0.338, 0.341
- **Median Δdk = 0.336 ms**

---

## 3. Итоговая таблица портфеля (прейскурант)

| Вариант | regs | blocks/SM | dk_new median (ms) | ΔE2E vs AB (median) | Δ% E2E | Ops shared* | LD/ST conflicts* | mio*% |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **AB** (033_sealed prod) | 128 | 4 | 10.400 | 0 | 0 | pack+π_V (12 SHFL+64 PRMT+24 SEL+16 STS.32) | 1.14B / 16.8M (022 ref) | 42% (033 ref) |
| **A** (pack no-π_V) | 107 | 4 | 10.509 | +0.093 | +0.19% | pack only (12 SHFL+64 PRMT+24 SEL+16 STS.32) | mid (est: 022 minus π-gains) | ~44% (π-снятие) |
| **0** (byte no-π) | 127 | 4 | 10.736 | +0.340 | +0.71% | none (64 STS.U8, 0 SHFL/PRMT/SEL) | high (pre-018 baseline) | ~45-48% |
| ~~**B**~~ (π_V no pack) | — | — | недостроим дёшево | — | — | требует ре-вывода π под byte-scatter (P14/P15=1.00 warning) | — | — |

*NCu measurements: значения для AB заимствованы из 022/033 отчётов (аналогичный prod); для A/0 обозначены как оценки — экспресс NCu-профилирование не проводилось (лежит в дополнительных гейтах, вне scope 036-r).

**Все Δ ≤ 0.71% < 2%** → все три варианта в **statu quo зоне** относительно AB.

---

## 4. Вердикт

**Vugar rule 036-r п.3**:
- «Сидим в лучшем углу → statu quo, таблица в леджер как прейскурант»
- «Другой лучший угол → migration mini-gate»

**Факт**: **AB — лучший угол** (все ΔE2E ≥ 0.19%, знак определён единогласно на 16 парах).

→ **Statu quo. Возмущений не вносим. Таблица становится прейскурантом в леджере.**

### 4.1 Vugar предсказание vs факт

**Прогноз (TZ 036-r п.3)**: «AB остаётся лучшим с margin <2% над A»  
**Факт**: **AB best, margin 0.19% над A, 0.71% над 0**. Оба <2% ✓ Предсказание подтверждено с большим запасом.

### 4.2 Разложение вклада

- **π_V incremental gain (A→AB)**: +0.19% E2E, +0.109 ms dk_new
- **Pack incremental gain (0→A)**: +0.52% E2E, +0.227 ms dk_new (0.71% − 0.19%)
- **Total pack+π_V gain vs bare (0→AB)**: +0.71% E2E, +0.336 ms dk_new
- **π_V стоимость**: +21r (128r vs 107r), 0 blocks impact, 0.19% wall win на post-033 ландшафте
- **Pack стоимость**: −20r (107r vs 127r bare), 0 blocks impact, 0.52% wall win

### 4.3 Обоснование statu quo

- AB best corner на текущем ландшафте
- Оба альтернативных угла хуже, но margin <2% (rule Right-2/3-v2 statu quo)
- π_V validated 8 пар на 036 baseline (|Δ|=0.26% E2E)
- π_V re-validated 8 пар на 036-r baseline (|Δ|=0.19% E2E) — стабильно knee
- Pack validated 8 пар на 036-r baseline (|Δ|=0.71% E2E) — стабильно knee
- Оба polishing даёт положительный signal, но не критический

---

## 5. Production status

- `libs/fa_bwd_dk_new.cu` восстановлен к AB (033_sealed md5 `a9f0ded8261e53a143b521ffa647f458`)
- `libs/bench_r2c_e2e` пересобран с fingerprint dk_new=128 OK
- **BIT-EXACT 11/11 + CANARY** ✓ на prod bench
- Три reference-бинаря сохранены в `libs/` для дальнейших ре-замеров:
  - `bench_r2c_e2e_AB` (128r pack+π_V)
  - `bench_r2c_e2e_Aonly` (107r pack no-π_V)
  - `bench_r2c_e2e_0` (127r byte no-π)

---

## 6. Обновление леджера

- **Портфель полировок dk_new — прейскурант зафиксирован** (§3)
- **π_V status**: остаётся в prod (validated 036 + 036-r, оба ландшафта статус quo)
- **Pack status**: остаётся в prod (validated 036-r, +0.52% E2E incremental)
- **033_sealed остаётся текущим production** (dk 128r + pack W2 + π_V, merged 254r + T-cut, E2E ~48 ms 036-r plateau, TFLOPS ~366)
- **Вариант B (π_V without pack)**: помечен «недостроим дёшево» — требует ре-вывода π-алгебры под byte-scatter, не окупается

---

## 7. Файлы

- ABBA scripts: `runs/reports/036_r_abba_ab_a.sh` + data `036_r_abba_ab_a_data.txt`
- ABBA scripts: `runs/reports/036_r_abba_ab_0.sh` + data `036_r_abba_ab_0_data.txt`
- Variant AB source: `runs/archive/036_variant_A/fa_bwd_dk_new.cu` md5 `a9f0ded8…`
- Variant A source: `runs/archive/036_variant_B/fa_bwd_dk_new.cu` md5 `b36eec6e…`
- Variant 0 source: `runs/archive/036_variant_0/fa_bwd_dk_new.cu` md5 `6e1a1a40…`

Chain md5: 035 `2a906152…` → 036 `7c4cce12…` → **036-r `<computed>`**

---

**End 036-r.**

**Итог**:
1. **Полный квадрат полировок собран**: AB (prod), A (pack no-π_V), 0 (byte no-π). B помечен «недостроим дёшево».
2. **ABBA 16 пар (8+8)**: AB best всегда, margin 0.19% над A (π_V) и 0.71% над 0 (pack+π_V).
3. **|Δ| < 2% везде → statu quo**. Прейскурант в леджер.
4. **Прогноз AB best margin <2% над A подтверждён** (факт 0.19%, десятикратный запас до порога).
5. **Production не тронут** после аудита (AB восстановлена, BIT-EXACT 11/11 ✓).
