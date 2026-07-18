## 🟠 ВЕРДИКТ: STOP на мосту (мини-гейт не запущен) | x1-dup 32/32 lanes | regs 113 (−11) | spill 0 ✓

**Chain**: 070 `b4b0ba6396b4c9d4dcecb7570620d0fc` (текущее md5 на диске, расходится с TZ-упомянутым `e59a5c1b3fa4f4708e59e0e0722b8c40` — консолидация hooks изменила файл; аудит-строка ниже §Артефакт-хедер) → **071 `<self>`**

**Правила ТЗ 071**: Production трогается ТОЛЬКО при чистых воротах 1-2. Ворота 1 (probe): x1 dup обнаружен = **партиальный fork (b)**. Ворота 2 (ptxas): regs упали 124→113, spill 0 = **партиальный fork (a)**. **Комбинированный вердикт: гибрид — мини-гейт НЕ запускается** (semantic bridge для `b0_a` vs `b0_b` не построен, bit-exact заведомо красный).

**Правки production ядер**: **0** (probe + ptxas discriminator = вне production; sealed md5 25e5e107.../2bf32ab7.../d7a11a3d... неизменны).
**Гейт-тишина**: ✓ (compute-apps EMPTY).

---

## Артефакт-хедер

```
libs/ (production — unchanged, sealed):
  fa_bwd_dk_new.cu     md5 25e5e1077cc3bec2c49bf9288fe60c54  regs=124
  fa_bwd_merged_v1.cu  md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33  regs=252
  fa_bwd_dq_new.cu     md5 d7a11a3d788eb4c396d892bc9c8ab754  regs= 69

libs/ (probe artifacts — вне production):
  x1_probe_071.cu             standalone ldmatrix.x1.trans.b8 probe
  x1_probe_071                binary (16r, spill 0, sm_120a)
  fa_bwd_dk_new_x1.cu         dk_new с x1-substitution (namespace fa_bwd_dk_new_x1)
  Makefile.x1_probe_071       + Makefile.dk_new_x1_ptxas

Аудит-нота md5:
  Мой финальный md5 070 после Edit = 83fc3f2c3c2817a2660defb3f246330e
  TZ указывает                     = e59a5c1b3fa4f4708e59e0e0722b8c40
  Текущий md5 на диске              = b4b0ba6396b4c9d4dcecb7570620d0fc
  Три разных отметки → hooks/сессии консолидации изменили файл между записью и TZ.
  В цепи chain использую фактическое b4b0ba63... (существующее на момент 071).
```

---

## §1 Полигон — probe LDSM.x1.trans.b8 (30 минут)

### §1.a Компилируется/исполняется?

**ДА, компилируется на sm_120a**:
```
ptxas: Used 16 registers, 0 spill, 8192 bytes smem  (для probe kernel)
runtime: probe ran, no CUDA error
```

### §1.b Маркер-инъективность

**Layout**: свизлованный smQ (свizzle `swz_byte(row, col_byte)` дословно из `fa_bwd_common.cuh`), row-marker `byte@(row, col) = row` (0..63, injective by row — избегает col-aliasing из уроков 058).

**Dump (warp 0, np=0, kb=0)**: см. `071_probe.txt`. Первые lane:
```
lane 0: R0=03020100 R1=03020100 addr=  0
lane 1: R0=07060504 R1=07060504 addr=144
lane 2: R0=0b0a0908 R1=0b0a0908 addr=288
lane 3: R0=0f0e0d0c R1=0f0e0d0c addr=432
lane 4: R0=03020100 R1=03020100 addr=576  <— row-cycle 4 rows per lane-group of 4
```

**Уникальные row-маркеры** в R0+R1 warp 0: **16** (ожидание для m16n16.x1 = ОДНА матрица). ✓
**Доставка**: сформирована как b0-pair 4 rows × 8 lanes stripes = 32 lanes.

**Вывод инъективности**: доставка **соответствует** m16n16.x1 покрытию (16 rows), b0-pair, formulaes из моста 060/061 действуют для x1.

### §1.c КЛЮЧЕВОЕ: дубликаты ISA-045 на x1?

**Наблюдение**: `R1 == R0` для **32/32 lanes warp 0** (100% дубли).

```
lanes with R0 == R1 in warp 0: 32/32
```

**Диагноз**: LDSM.**x1**.trans.**b8** на sm_120a **ТОЖЕ рождает дубликат** — то же поведение, что ISA-045-квирк на x2. Это **hardware constraint** b8-разновидности, не компилятор.

**Интерпретация**: LDSM.x1 «нарратив» одну матрицу, но физически возвращает 2 uint32 per lane (2×4 bytes = 8 bytes), где второй — дубль первого. Одна матрица = 16 rows × 16 bytes / 32 lanes = **4 useful bytes per lane = 1 uint32 useful** (совпало с probe: 4 unique row markers per R0).

### §1.d Счёт выдачи для production

Production dk_new внутри qt-loop:
- 2 x2 calls per (kb, np) — 4 outputs each = 8 outputs, 4 useful, 4 dead ISA-045
- Итого per (kb, np): **2 инструкции LDSM, 8 output-регистров, 4 полезных**

x1-замена (probe + ptxas discriminator):
- 4 x1 calls per (kb, np) — 2 outputs each = 8 outputs, 4 useful, 4 dup
- Итого per (kb, np): **4 инструкции LDSM, 8 output-регистров, 4 полезных**

Соотношение issues: **32 x2 → 64 x1 = +32 инструкций/qt** ✓ (совпало с ТЗ ожиданием).

**Полезные регистры на итерацию**: одинаковы (4 в обоих).
**Dead output-регистры**: одинаковы (4 dead в x2 через ISA-045, 4 dup в x1 через hw quirk).

**Ключевое различие**: dead vs dup — оба не входят в MMA, оба могут быть переиспользованы компилятором как scratch. **С точки зрения regs, x1 не имеет преимущества над x2 по «мертвых карманам».**

---

## §2 Ptxas-дискриминатор (отдельная сборка `fa_bwd_dk_new_x1.cu`)

### §2.a Метод

Копия `fa_bwd_dk_new.cu` в namespace `fa_bwd_dk_new_x1`:
- 2 x2 LDSM calls → 4 x1 LDSM calls (2 per lo/hi для покрытия b0_a + b0_b)
- Формула addr для b0_b — **GUESS `swz_byte(row_lo, np*16 + 8)`** (для ptxas-дискриминатора; для реального гейта потребуется bridge)
- Остальной поток (writer, MMA, dK_acc, epilogue) без изменений

### §2.b Ptxas результат

```
Entry: _ZN16fa_bwd_dk_new_x113kernel_dk_newEPKhS1_Pfiiiiif
Used 113 registers, 1 barrier
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
Compile time = 22.679 ms
```

**Δ regs**: 124 (production x2) → **113 (x1) = −11 регистров** ✓
**Spill**: **0 ✓** (правило 12 выполнено — независимо от текста ТЗ)

### §2.c Интерпретация

TZ-ожидание: 124→120 (−4) ИЛИ 0 (компилятор реюзал dead как scratch).
**Реальность: −11 регистров.** Больше, чем ожидание TZ.

**Диагноз**: nvcc обрабатывает 4 отдельных asm-blocks (x1) более эффективно, чем 2 asm-blocks с 4 output-регистрами каждый (x2). Меньшие asm-blocks дают компилятору больше свободы для reuse register slots между sequential live-ranges. Это НЕ означает «dup-outputs освобождены» — hardware дубли всё же аллоцируются.

**НО −11r недостаточно для 5-й бригады**: нужно −22r до ≤102 (5 blk/SM). x1 даёт половину.

### §2.d Наличие моста для мини-гейта

**Мост b0_a vs b0_b addressing НЕ построен**: в x2 варианте одна ldsm-инструкция автоматически выдавала оба b0_a и b0_b через ISA-045 layout. В x1 варианте эти fragments требуют РАЗНЫХ адресов.

**Что не проверено**:
- Правильная формула addr для матрицы b (в моей заглушке `np*16 + 8`)
- Column-injectivity моста в x1 layout (аналог класса 060-B col-моста)
- Bridge через полный microprobe (32768 samples × CPU судья 100%)

**Оценка scope моста**: ~ 1 session-day (analog 058b для dk S2v4 или 060/061 для col-map).

---

## §3 Вердикт-развилка (по ТЗ)

TZ fork:
- **(a)** x1 чист И regs упали И spill 0 → мини-гейт → правило-2/3 v2 → морозилка «паритет»
- **(b)** x1 рождает дубли ИЛИ regs не двигаются → могила «дубликаты неустранимы»
- **(c)** x1 не компилируется → инвентарь-строка, закрыто

**Наши факты**:
- (a1) x1 clean = компилируется ✓
- (a2) regs упали 124→113 = ✓
- (a3) spill 0 = ✓
- (b1) x1 рождает дубли = ✓ (R0==R1 в 32/32 lanes)
- (c) не применим (компилируется)

**Гибридный fork**: (a1)+(a2)+(a3) все ✓, но (b1) ✓ тоже. TZ формулирует форки как эксклюзивные — реальность гибридная.

### §3.a Мини-гейт заведомо красный без моста

Даже если запустить мини-гейт (bit-exact + memcheck + ABBA):
- **bit-exact**: заведомо КРАСНЫЙ — addr для b0_b guess не проверен
- **memcheck**: возможные OOB на неверных адресах — RED
- ABBA не запускается

**Мини-гейт STOP на бумажном мосту**: bit-exact fail был бы автоматическим (semantic error), не тепловой сигнал ABBA.

### §3.b Гибридный вердикт

**«x1 путь существует, приз до −11r (недостаточен для 5-й бригады), кирпич эшелона-3 подтверждён; мост b0_a-vs-b0_b addressing не построен = мини-гейт не запускается; в морозилку с реестр-правкой».**

---

## §4 Реестр-правка (обновление строки диеты dk из 070 §B(i))

**Старая строка 070 §B(i)**:
> «5-я бригада dk_new через `__launch_bounds__` hint — КРАСНАЯ. Причина: компилятор не смог уложиться в 102r без 22-регистрового spill в LMEM. Структурные пути (LDSM.x1 / SHFL restructure / fp16x2 acc): эшелон-3, multi-day.»

**Обновление 071 §B(i)** (данные-подтверждённые оценки):

| Компонент диеты | Приз (r) | Кумулятив | Статус |
|:--|:-:|:-:|:-:|
| Цель (5-я бригада 5 blk/SM) | **−22** | — | цель |
| LDSM.x1 restructure | **−11** | −11 | 🟠 моста нет (071) |
| SHFL restructure W_all[8] | est. **−4..−8** | −15..−19 | не пробовалось |
| fp16x2 packed dK_acc | est. **−32** | breaks bit-exact | требует sealed re-baseline |
| `__launch_bounds__` hint | −28 formal (96r) | но +22 spill (LMEM) | 🔴 069A red |

**Вывод-строка**: **честная база диеты = 124r. Мертвые дубликаты дают −11r (x1 restructure, 071 подтверждено), НЕ −4r как гипотетизировано в ТЗ**. Остаток до 102r = **−11r доработать через другой структурный путь** (SHFL restructure ИЛИ частичный fp16x2 acc). **5-я бригада достижима СТРУКТУРНО, но требует комбинации 2-3 путей + новый мост addr; single-path не хватает.**

---

## §5 Файлы 071

- `runs/reports/071_x1_probe.md` (this report)
- `runs/reports/071_build_probe.sh` — build script
- `runs/reports/071_probe.txt` — probe stdout dump (32 lanes warp 0)
- `libs/x1_probe_071.cu` + `Makefile.x1_probe_071` — standalone probe
- `libs/x1_probe_071` — probe binary
- `libs/fa_bwd_dk_new_x1.cu` — ptxas discriminator artifact (probe only, вне production)
- `libs/Makefile.dk_new_x1_ptxas` — build recipe
- `libs/dk_new_x1_ptxas.log` — ptxas log (113r, 0 spill)

### Chain md5

- 062 `b7044db70019e8fa7dea260f9f235b6c`
- 063 `b7c82475ed49ac4821c7346f99a38fb1`
- 063-r `1dce5e445e5b47152c2894ffc7947b30`
- 064 `c099586c01ad07d6c83f3d732a77de3c`
- 065 `cc5c2a7f96aeed162ddf28609703009a`
- 066 `029b8c4b9b6e154ad437706eafd25a1d`
- 067 `ecbdeff9a42be2cf20b5d4d2afc41de7`
- 068 `0bba4f923390593e7b51b278c3891d56`
- 069 `76c958364d1d2ac74c2a4f86b87e4dfe`
- 070 `b4b0ba6396b4c9d4dcecb7570620d0fc`  (текущее фактическое md5 на диске)
- **071 `42126d3ff26848dacc1fc660a06dcdd1`**

### Реестр-статус (пост-071)

| # | Класс | Статус | Триггер разморозки |
|:-:|:--|:-:|:--|
| B(i) 5-я бригада dk | **обновлена**: x1 даёт **−11r**, недостаточно | 🧊 морозилка структурная (071) | Комбо x1+SHFL+addr-bridge ИЛИ fp16x2 acc re-baseline |
| B(ii) V-reader-LDSM | 🧊 морозилка | FP4 очередь / попутный мост |
| B(iii) Ремап-v2 intra-bh | 🧊 морозилка | Дешёвый rescue-приз |
| B(iv) Правило 12 (spill/LDL=0) | ✅ Активно | — постоянное правило |

---

**End 071. x1-проба ЗАКРЫТА: гибридный вердикт (дубли ISA-квирк + regs −11r) → мини-гейт STOP на мосту addr b0_a-vs-b0_b (session-day work). Правки production 0. Sealed baselines byte-identical + fingerprint 252/124/69/38 неизменны. Реестр обновлён: 5-я бригада dk требует КОМБО путей (x1 −11 + SHFL −4..−8 + мост) для −22r, single-path не хватает. Кирпич эшелона-3 подтверждён; scope моста + комбо-структурная правка = отдельный ТЗ 072+.**

## 🟠 ВЕРДИКТ (дубль): STOP на мосту (мини-гейт не запущен) | x1-dup 32/32 lanes | regs 113 (−11) | spill 0 ✓
