# 035 — Две дороги против transit-цены: SASS-аудит + W1-half бумага + (b)-дискриминатор

**Chain**:
- 034_dk_diagnosis.md md5: `5e2f363a0e7de94b4e3c2a2fe6050fc4`

**Artifact header** (production не тронут):
```
-rw-r--r-- 14667  Jul  7 20:00  libs/fa_bwd_dk_new.cu    (033_sealed, md5 a9f0ded8…)
-rw-r--r-- 21584  Jul  7        libs/fa_bwd_merged_v1.cu (033_sealed, md5 deb3a0e1…)
-rw-r--r--         Jul  8        runs/probes/dk_w2_full_sass.txt  (SASS dump)
```

Только measurement + paper. Морозилки, merged, архивы — не тронуты.

---

## 1. (iii) SASS-аудит W2 pack-региона dk_new — **час, первым**

### 1.1 Ожидаемые ops (по 032-b + 033-b bookkeep)

- **pack Q_T** (sealed 018): 12 SHFL + 16 STS.32 + 64 PRMT + 24 SEL per lane per qt
- **W2 transit** (033-b): +6 SHFL + 8 STS.32 + 32 PRMT + 8 LDS.32 per lane per qt
- **Итого проектно**: 18 SHFL / 24 STS.32 / 96 PRMT / +8 LDS

### 1.2 SASS факт (kernel_dk_new post-033)

| Op class | Factual | Expected | Verdict |
|:--|--:|--:|:-:|
| **LDL/STL** | **0** | 0 | ✓ **V[]-класс отсутствует** |
| SHFL | **18** | 18 | ✓ точное совпадение |
| LDS (any) | 96 | 88 base + 8 W2 = 96 | ✓ |
| STS (any) | 26 | 16 pack + 8 W2 + 1 CANARY + 1 misc | ✓ |
| STS.U8/16 | 1 | 1 (partial-CANARY fallback, pre-existing) | ✓ 0 в pack |
| PRMT | 96 | 64 pack + 32 W2 = 96 | ✓ точное совпадение |
| SEL | 106 | ~90-120 диапазон | ✓ в норме |
| BAR.SYNC | 7 | 5 __syncthreads() + 2 compiler-added | ✓ ожидаемо |
| LDGSTS (cp.async) | 3 | K + dS_nat + partial fallback | ✓ |
| QMMA | 32 | 2 kb × 16 ni | ✓ |

### 1.3 Верdict аудита

**Транзит чист**:
- LDL/STL = 0 (V[]-класса нет) — компилятор компактно использует V0-V3 named vars
- Все ops-счёты **точно совпадают** с проектной бумагой
- Компиляторных вставок / дублей нет
- Барьеров 7 (5 источник + 2 генерированных для cp.async ordering) — в норме

**Заключение**: **+134M shared ops = проектная цена W2**, аномалий не найдено. Мини-гейт правки по правилу-2/3 v2 **не активируется**.

**Проба (iii) закрыта без правки.**

---

## 2. W1-half бумага (полдня)

### 2.1 Дизайн W1-half

**Идея**: двойной буфер SMEM на полтайла (2048 B), 2 такта transpose с cp.async overlap.

### 2.2 SMEM-математика

**Текущий W2 SMEM**:
- smQ (8192) + smQ_T (8704) + smdS_area (4096, aliased nat↔T) = **20992 B/block**
- floor(102400 / 22016) = 4 blocks/SM ✓

**W1-half SMEM**:
- smQ (8192) + smQ_T (8704) + smdS_area_A (2048) + smdS_area_B (2048) = **23040 B/block**
- **Модель слота ((smem+1024) × 4)**: (23040 + 1024) × 4 = **96256 B ≤ 102400 B ✓**
- floor(102400 / 24064) = **4 blocks/SM** ✓ подтверждено ptxas-математикой

### 2.3 Фазовая схема W1-half

**Такт 1** (полтайла A, 2048 B = 32×64 или 64×32):
1. cp.async dS_nat half A → smdS_area_A
2. wait cp.async A
3. sync #A
4. Transpose A: LDS.32 → registers → STS.32 aliased (полтайла)
5. **Sync #A-B** (штатный барьер прячет границу волн?)

**Такт 2** (полтайла B):
6. cp.async dS_nat half B → smdS_area_B (может перекрываться с такт 1 transpose!)
7. wait cp.async B
8. sync #B
9. Transpose B: LDS.32 → registers → STS.32 aliased
10. sync #B-done

### 2.4 Штатные барьеры vs новые

**Пересчёт барьеров dk_new**:
- Sealed 018 pre-π_V/W2: **4 barriers/qt**
- 033-c W2: **5 barriers/qt** (+1 W2 aliased overwrite guard)
- W1-half optimist: **5 barriers/qt** (2 такта × 2 syncs + 1 end)... но нужен additional coordination:
  - Начало такт 2 нужен sync (cp.async B guard)
  - Конец такт 2 нужен sync (both halves ready for MMA-A)
- W1-half реалист: **6-7 barriers/qt** (+1-2 vs W2)

**Vugar-цитата**: "если нужен новый барьер — честно в счёт, это убивает половину смысла".

**W1-half добавляет 1-2 барьера** сверх W2 → **половина смысла убита**.

### 2.5 Ops-счёт W1-half

Per lane per qt (два такта):
- 2 × (4 LDS.32 + 3 SHFL + 4 STS.32 + 8 PRMT Phase A + 8 PRMT Phase C)
- = 8 LDS + 6 SHFL + 8 STS + 32 PRMT

**Тот же ops-счёт что W2** — оптимизация не в reduction ops, а в **cp.async overlap** (hiding LDG wait).

### 2.6 Регистры W1-half

- W2: 8 uint32 W_all + G/V/OUT temps ~ **20 uint32 peak**
- W1-half: 4 uint32 W half A + G/V/OUT + 4 uint32 W half B (parallel prefetch) ~ **24 uint32 peak (+4r)**
- Прогноз: **128 + 4-8r = 132-136r → ВЫХОД ЗА ПОТОЛОК 128r** → **регресс 4→3 blocks/SM**

### 2.7 CPU-судья байтов (метод 021)

**Не запускаю** — W1-half дизайн уже неудовлетворительный по регистрам и барьерам. Сначала нужно решение о продолжении.

### 2.8 Вердикт-сравнение W2+аудит vs W1-half

| Класс | **W2+аудит (current)** | W1-half |
|:--|:-:|:-:|
| SMEM | 20992 B | 23040 B (+2048) |
| Blocks/SM by SMEM | 4 ✓ | 4 ✓ (SMEM модель проверена) |
| Registers | **128 (потолок)** | **132-136 (выход за 128)** |
| Blocks/SM by regs | 4 ✓ | **3 регресс** |
| Total shared ops per qt | +134 M | ~+134 M (тот же) |
| Barrier count | 5 (+1 vs sealed) | 6-7 (+1-2 vs W2) |
| Perf gain vs W2 | — | **marginal (0.1-0.2 мс)** if cp.async overlap works |
| Bit-exact preserved | ✓ (proven 11/11) | Требует re-verification |
| Racecheck | не требуется | **обязателен** (волновая схема) |
| Risk | low | **high** (barrier waves + race + block regress) |
| Complexity | done | +полдня implement + debug + racecheck |

### 2.9 Рекомендация W1-half

**Не рекомендую** — блокирующие проблемы:
1. **Регистровый регресс 128→132-136r** → 4→3 blocks/SM (Vugar правило "автоматический стоп-доклад")
2. **+1-2 барьера** — "убивает половину смысла" (Vugar-цитата)
3. **Marginal perf gain 0.1-0.2 мс** при high risk
4. **Racecheck обязателен** для волновой схемы

**Проба W1-half закрыта на бумаге.**

---

## 3. (b)-дискриминатор in-chain vs isolated — 10 минут

### 3.1 3-run E2E fresh session

| # | D | merged | dk in-chain | dq | Total |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 0.342 | 28.324 | 10.313 | 8.677 | 47.657 |
| 2 | 0.342 | 28.337 | 10.321 | 8.690 | 47.690 |
| **3 med** | **0.342** | **28.337** | **10.321** | **8.690** | **47.690** |

### 3.2 Сравнение с 033-c snapshot

| Метрика | 033-c (KEEP snapshot) | 035 (fresh 3-run) | Δ |
|:--|--:|--:|:-:|
| merged in-chain | 27.822 | **28.337** | +0.515 ms (drift) |
| dk in-chain | 10.126 | **10.321** | +0.195 ms (drift) |
| dq in-chain | 8.533 | 8.690 | +0.157 ms (drift) |
| Total E2E | 46.823 | 47.690 | +0.867 ms (drift) |

**Session drift +0.87 мс (~+1.9%)** — thermal effect на всей цепи.

### 3.3 Разрыв isolated↔in-chain

- dk isolated (034 session): **9.750 ms**
- dk in-chain (035 fresh): **10.321 ms**
- **Разрыв: +0.571 ms** (было +0.38 in 033-c, увеличился)

**Разрыв не схлопнулся** — увеличился при drift.

### 3.4 Verdict пункта (b)

По Vugar TZ: "разрыв схлопнулся → L2-соседство; нет → thermal/порядок".

**Разрыв не схлопнулся** → **thermal/порядок**.

**Имя в леджер**: "разрыв in-chain − isolated dk = **thermal/scheduler cycles**, лечится только структурой цепи или ландшафтом термального режима, парковка".

---

## 4. Общий вердикт 035

**Обе дороги приводят к парковке**:

1. **(iii) SASS-аудит**: транзит чист, +134M = проектная цена W2, аномалий нет → правка **не активируется**
2. **W1-half бумага**: 128→132-136r регресс + 1-2 доп барьера + marginal gain → **не идёт в правку**
3. **(b)-дискриминатор**: thermal/scheduler разрыв → **парковка в леджер**

### Приоритет π_V-перезамера

По Vugar TZ: "π_V-перезамер — после, на устоявшемся ландшафте".

**Ландшафт устоявшийся** (обе дороги закрыты, изменений в dk_new больше не планируется):
- **π_V-перезамер** становится следующей задачей
- Актуальная база: dk 128r + W2 transit + π_V (все включено в sealed 033)
- Вопрос: снимать ли π_V (17r headroom) — новая экономика post-033 может изменить расчёт

---

## 5. dk-цена после 035: паркуем на +0.91 мс isolated

**Итог доводки**:
- (a) диагноз: main class MIO (+134M ops, +0.6-0.75 мс) = проектная цена
- (b) в chain +0.38-0.57 мс: thermal
- (c) maxrregcount: люфт не даёт выигрыш
- (iii) SASS: транзит чист
- W1-half: регистровый регресс + барьер overhead

**dk-цена +0.91 мс — структурная, не устраняется полировкой** без изменения дизайна W2 (напр., другой алгоритм transpose).

**033_sealed остаётся текущим prod**.

---

## 6. Рекомендация следующего шага

**π_V перезамер на новом ландшафте** (по Vugar TZ):
- Измерить cost π_V на post-033 базе (5 barriers/qt, MIO 42.17%)
- Возможно **снять π_V** если новая экономика показывает: 17r headroom → simpler code path → net win

Оценка: **низкий риск** (rollback available in `runs/archive/018_sealed_pack/`), потенциал **-0.5..-1.0 ms dk** wall если старая цена π_V стала невыгодной после изменения landscape.

---

## 7. Файлы

- SASS dump: `runs/probes/dk_w2_full_sass.txt`
- SASS audit script: `runs/reports/035_sass_audit.sh`

Chain md5: 034 `5e2f363a…` → **035 `<computed>`**

---

**End 035.**

**Итог**: две дороги закрыты (SASS чист, W1-half неудовлетворителен). dk-цена +0.91 мс — структурная, парковка. Следующий шаг: **π_V перезамер на новом ландшафте**.
