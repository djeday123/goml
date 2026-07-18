# 012 — dk_new QT_STRIDE fix — HALT перед правкой (Vugar-правило model reconcile) (2026-07-06)

## Статус: ПРАВКА НЕ СДЕЛАНА. Model reconcile гейт провалился.

## ARTIFACT-HEADER

### Кросс-верификация цепочки отчётов
```
md5 011_dk_conflicts.md: 766f8d6cd592a0c956d5414d2c2b1379  (13384 bytes)
md5 010_dk_sass.md:      bbdb223ea2c452333f7f1ff8eb625279  (11620 bytes)
md5 009_ncu.md:          62977f103b01436c877c56d4b7a48a4a  (11748 bytes)
```

### ls -la runs/reports/ (свежая часть, 012-related)
```
-rw-r--r--   9320 Jul 6 07:21  008_R2C.md
-rw-r--r--  11748 Jul 6 09:26  009_ncu.md
-rw-r--r--  11620 Jul 6 09:33  010_dk_sass.md
-rw-r--r--  13384 Jul 6 15:15  011_dk_conflicts.md
-rw-r--r-- 1226067 Jul 6 09:27 010_dk_new_sass_full.txt   (SASS дамп источник)
```

Артефакт правки (libs/fa_bwd_dk_new.cu:23 QT_STRIDE) **не изменён**. ptxas/bit-exact/wall гейты **не запущены**. 

## 012-a — Model reconcile (RESULT: gap 129% > threshold 15% = STOP)

Формула paper (TZ шага 2 пункт a):
```
blocks × warps × qt-iters × excess-per-qt = 16384 × 4 × 128 × 88 = 738 M events
```

Measured NCu (011_dk_ld_st_split.sh, kernel_dk_new isolated, canonical):
```
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum = 1,694,738,730 events
```

### Reconcile-таблица

| Показатель | Paper (011) | Measured (011) | Ratio | Vugar-порог |
|-----------|:-----------:|:--------------:|:-----:|:-----------:|
| **Conflicts LD events per launch** | **738 M** | **1.69 B** | **2.296** (**+129.6%**) | **>15% = STOP** ✗ |
| Conflicts events per qt per warp | 88 | 202.0 | 2.30 | |
| shared_ld inst per launch | 738 M (88×qt×grid) | 889 M | 1.205 (+20.5%) | >15% (тоже вне окна) |
| Dynamic LDS per qt per warp | 88 (SASS static) | 106 | +18 extra LDS/qt | |

### Гипотезы источника расхождения (для ре-атрибуции)

Модель 011 идентифицировала три класса LDS (A dS_T 8 + B Q_T 64 + Q feeder 16 = 88 per qt) и предсказала:
- A: 4-way conflict × 8 loads × 3 excess = 24 events/qt/warp
- B: 2-way conflict × 64 loads × 1 excess = 64 events/qt/warp
- Q feeder: swz_byte swizzle → 0 conflicts
- **Sum: 88 events/qt/warp**

Measured 202/qt/warp = **114 unaccounted events per qt per warp**. Возможные источники:
1. **Missing 18 dynamic LDS/qt** (SASS static 88 → measured 106). Если они конфликтуют в среднем 6.3-way each → +114 events. Не идентифицировано в SASS-грепе.
2. **A operand 8-way не 4-way**: если 32 lanes все хитают банки {0..3} + {16..19} (по 4 lanes/bank), но с mask-based sub-broadcast → nvidia counting 4-way = 3 events. Если счётчик работает per-bank (не per-warp), 4 banks × 3 events = 12 events per A LDS → 96/qt vs paper 24. **Overshoots by 72** — плюс B unchanged 64 → 160 total, closer but still short 42.
3. **B pattern is 4-way not 2-way**: Если min-diff расчёт неверен (например nvidia считает по кольцу differently), B could give 3 excess/load → 64×3 = 192 events. Plus A 24 = 216. Match to 202 within noise. **Требует эмпирической проверки на MINIMAL micro-kernel** (LDS-only с известным паттерном).
4. **Q feeder имеет non-zero conflicts на некотором ks**: swz_byte байт-масштабная свизла — при чтении uint32 (4 байта) mask-flow может нарушить XOR-паттерн. Проверить перерасчётом на word-level.
5. **NCu counting metric definition**: возможно `l1tex__data_bank_conflicts...op_ld.sum` считает event × parallel-lane pairs, не просто additional passes. Требует NCu-doc verification.

### Что НЕ помогает close the gap

- **Q feeder — verified 0 conflicts** для всех ks через полный XOR-расчёт (011.3).
- **STS conflicts 30.9 M** — 1.8% от LD, не помогают explain LD gap.
- **Dead LDS (@!PT LDS RZ, 6 штук)** — predicated off, не считаются.

## 012-b — Vugar-правки приняты, зарегистрированы БЕЗ применения

### SMEM slot 4 blocks = 24576 (моя цифра 25600 была без driver = ошибка)
```
Правильно: (dyn_smem + driver_1024) × blocks ≤ 102400
Для 4 blocks: dyn_smem ≤ 24576 B
```
Current dyn: 20992 B (from libs/fa_bwd_dk_new.cu:243 launcher).
Headroom current: 24576 − 20992 = **3584 B**.
После правки (STRIDE 80): dyn = 22528 B. Headroom: 24576 − 22528 = **2048 B**.

Vugar-цифра 1000 B — округление вниз (запас с margin). Мой exact 2048 B ≥ 1000 B → правка помещается **если** правка делается.

### Порог 33024 (Vugar-контекст)
```
Slot 3 blocks: (dyn + 1024) × 3 ≤ 102400 → dyn ≤ 33109 (≈33024 округлённо)
```
Значит 33024 = максимум dyn для 3 blocks/SM. Наш 22528 (после fix) far below → 4 blocks safe (запас 2048 B ± threshold interpretation).

### Predictions (зарегистрированы, но НЕ проверены — правка не делалась)
- **Post-fix excessive LD**: 0.4-0.5 B (paper предсказание сдвига B-класса с 2-way → 1-way = perfect distribution).
- **Post-fix wall dk_new**: 7.5-8.5 ms (from current 9.42 ms, экономия -1..-2 ms).
- **Model-failure criterion**: excessive > 1 B post-fix = paper model wrong в attribution.

### Таблица стридов — правка (метка stride 76)
```
| STRIDE | S=word_stride | min diff | Verdict |
| 68 (current) | 17 | 2 | 2-way |
| 72 | 18 | 2 (wrap 32→0) | 2-way |
| 76 | 19 | **1** | **2-way** (правильная метка per Vugar; было "1-way" — исправлено) |
| 80 | 20 | 4 | perfect 1-way |
```

## 012-c — EXPECT-dict fingerprint dk_new (регистрируется НО не обновляется — правка не сделана)

**Планировалось** (если бы правка прошла reconcile):
| Атрибут | Current (68) | Post-fix (80) |
|---------|:------------:|:-------------:|
| numRegs | 96 | 96 (expect unchanged) |
| sharedSizeBytes | 0 (dynamic) | 0 |
| localSizeBytes | 0 | 0 |
| dynamic_smem | 20992 | 22528 |
| blocks/SM | 4 | 4 (expect held) |

Fingerprint EXPECT-dict в bench_r1_e2e/bench_r2c_e2e и других harness'ах **не обновлён**.

## 012-d — Гейты не запущены (правка не сделана)

- (a) ptxas: не запущен
- (b) fingerprint 4 blocks: не проверен
- (c) BIT-EXACT: не проверен
- (d) Wall (a)-baseline: не собран
- (e) NCu post-fix: не снят

## Резюме 012 (STOP)

- ⛔ **Model reconcile gap 129% > Vugar-порог 15%**
- ⛔ **Правка QT_STRIDE 68→80 НЕ сделана** (по правилу шага 2 пункт a)
- ✅ Vugar-правки на бумаге приняты (slot 24576, headroom 2048 после fix, stride 76 = 2-way)
- ✅ Predictions зарегистрированы (0.4-0.5 B excess, 7.5-8.5 ms wall) — не проверены
- ⚠️ **Ре-атрибуция требуется**: 5 гипотез перечислены; наиболее вероятная — B pattern 3-excess vs 1-excess (не 2-way, а более сложный паттерн)

### Ре-атрибуция путей (для Vugar-выбора)

1. **Микро-kernel LDS-only probe**: minimal SASS-controlled kernel с известным паттерном stride 68 vs stride 80 → mesure conflicts empirically, verify counting model. **Стоимость**: 30 min, isolated.
2. **NCu docs deep-dive**: точное определение `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`. Возможно per-bank vs per-warp counting mismatch.
3. **SASS ILP-audit**: найти 18 extra LDS/qt (dynamic 106 - static 88). Compiler-inserted vs source-inserted.
4. **Alternative approach**: пропустить model reconcile, сделать fix **эмпирически** и измерить delta (Vugar-правило требует reconcile, но при 20-30% паперовой уверенности можно попробовать — только по прямому решению Vugar).

### Что подтверждено на числах
- **LD excess 1.69 B — реальный** (measured, не оценка)
- **B pattern stride 68 vs 80** — верифицирован per-bank calculation (011.3)
- **SMEM budget** — 4 blocks запас 2048 B post-fix
- **Bit-exact risk fix**: 0 (layout-only change, byte content preserved)

**Правка технически безопасна и корректна. Model gap блокирует по правилу шага-2 пункт (a). Жду Vugar-решения: (1) микро-kernel probe для reconcile, (2) выбор из 5 гипотез, (3) прямое разрешение на empirical fix без reconcile, (4) переход к другому кандидату.**
