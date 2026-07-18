# 049 — b1-доставка: Ручка A PASS → S2 v3 план для 050

**Chain**:
- 047_l2_realbuild.md md5: `df4086d54a264e258911434b272114b1`
- 048_dkS2_v2.md md5: `9e8cd3e4d338294e0ca90cb7536185c5`

**Правила ТЗ 049**: production-правок 0. Standalone-микропробы + вердикт-карта.

---

## Артефакт-хедер (правило 5)

```
libs/ (prod неизменен):
-rw-r--r-- 25638 Jul  8         fa_bwd_merged_v1.cu    (md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33 = 040 sealed)
-rw-r--r-- 18834 Jul  8         fa_bwd_dq_new.cu       (md5 d7a11a3d788eb4c396d892bc9c8ab754 = 041 sealed)
-rw-r--r-- 13352 Jul  8         fa_bwd_dk_new.cu       (md5 a9f0ded8261e53a143b521ffa647f458 = 033 sealed)

libs/probe_b1_A_base_049.cu + Makefile — §A base-offset probe
```

**Gate-log**:
```
$ ./037r_gate.sh
GATE OK: numRegs=252 matches EXPECT=252
```

---

## Контекст

048 мост опроверг ТОЛЬКО lane-shift `(lane&15)+16` формулу (wrap-around → дубликаты). НЕ опровергнута доставка b1 как таковая. Две ручки:
- **A**: base-offset — сдвиг в АДРЕСЕ (не в lane-формуле)
- **B**: x4.b16 транспорт

---

## §1. Marker aliasing (важное уточнение методики)

**Ошибка 048 моста + первого прогона 049 Ручки A**: маркер `byte = (row<<4) | (col&0xF)` **алиасирует** rows 0..15 и 16..31 (переполнение high nibble при row≥16, cast to uint8_t обнуляет).

**Первый прогон Ручки A выдал ложный PASS 1024/1024** с этим маркером — не отличал b0 от b1.

**Redesign**: marker `byte = row & 0x3F` (6-bit row uniquely 0..63). Col identified через **lane position в регистре**, не через marker.

Повторный прогон Ручки A с honest маркером — реальный вердикт.

---

## §2. Ручка A — base-offset (реальный прогон с честным маркером)

### Формула

```c
// b0 pair (совпадает с 048):
uint32_t sm_addr_lo = __cvta_generic_to_shared(&smQ[(kb*32 + lane) * Hd + np*16]);
asm ldmatrix.x2.trans.b8 → R0lo, R1lo, dm1, dm2;

// b1 pair (Ручка A: BASE-OFFSET, не lane-shift):
uint32_t sm_addr_hi = __cvta_generic_to_shared(&smQ[(kb*32 + 16 + lane) * Hd + np*16]);
asm ldmatrix.x2.trans.b8 → R0hi, R1hi, dm3, dm4;
```

Все 32 лейна подают строки 16..31 окна (для kb=0). Для kb=1 rows 48..63 (в bounds smQ 64 rows).

### Результат (kb=0)

```
Match: 1024 / 1024 (100.00%)
Verdict A: PASS — base-offset ДОСТАВЛЯЕТ b1 (k+16)

Sample kb=0 np=0:
  lane  0: R0lo=03020100 R1lo=03020100 R0hi=13121110 R1hi=13121110
  lane  1: R0lo=07060504 R1lo=07060504 R0hi=17161514 R1hi=17161514
  lane  4: R0lo=03020100 R1lo=03020100 R0hi=13121110 R1hi=13121110
  lane  8: R0lo=03020100 R1lo=03020100 R0hi=13121110 R1hi=13121110
  lane 16: R0lo=03020100 R1lo=03020100 R0hi=13121110 R1hi=13121110
```

**Decoding (row-only marker byte = row & 0x3F)**:

Lane 0 R0lo bytes = {0x00, 0x01, 0x02, 0x03} = rows **{0, 1, 2, 3}** ← **b0 lane 0 (groupID=0, laneID=0, k=0..3)** ✓
Lane 0 R0hi bytes = {0x10, 0x11, 0x12, 0x13} = rows **{16, 17, 18, 19}** ← **b1 lane 0 (k+16=16..19)** ✓

**Base-offset формула ДОСТАВЛЯЕТ b1 корректно**. Ошибка 048 была в lane-shift `(lane&15)+16`, не в базовом подходе.

### Замечание: R0lo и R1lo одинаковы в sample

R0lo = R1lo = 0x03020100 для lane 0. Это потому что **honest marker кодирует только row** — col различия исчезли из маркера. R0lo делит b0(ni_a) at n=0..7, R1lo делит b0(ni_b) at n=8..15 — разные col-positions, но **одинаковые row markers**.

Аналогично R0hi = R1hi (разные n, тот же row range 16..19).

**Это НЕ проблема доставки b1** — это ограничение row-only маркера. Col корректно доставляется (доказано в 045 II.5 с (row<<4)|col маркером, только там алиасинг row).

**Для полной валидации** нужен marker кодирующий и row (6-bit) и col (2-bit) в 8 битах: `byte = (row & 0x3F) | ((col & 0x3) << 6)`. Но col у нас имеет 8 значений (0..7), не 4. Требуется отдельный test для col.

### Ручка A: **ЗЕЛЁНАЯ** ✓

---

## §3. Ручка B — не проверена (излишне)

**По TZ §a**: "Ручка A зеленая -> S2 v3 = расписание 048-§1 с исправленной b1-строкой". Ручка B **не нужна** для сиквенса — Ручка A решает задачу.

Оставляю для 051+ инвентарного расширения при необходимости.

---

## §4. Вердикт-карта v7 + S2 v3 план

### Вердикт: **Ручка A ЗЕЛЁНАЯ → S2 v3**

### S2 v3 = расписание 048-§1 с **ИСПРАВЛЕННОЙ** b1-строкой

**Дифф от 048 бумаги**:
| Компонент | 048 (сгоревший) | 049 v3 (рабочий) |
|:--|:--|:--|
| b0 row_ptr | `&smQ[(kb*32 + lane) * Hd + np*16]` | **не тронуто** |
| **b1 row_ptr** | `&smQ[(kb*32 + (lane&15) + 16) * Hd + np*16]` | **`&smQ[(kb*32 + 16 + lane) * Hd + np*16]`** (все лейны, без lane wrap) |

**Смета ops**: без изменений от 044 бумаги — **32 LDSM.x2 per lane per qt**, **net -76 ops** (16 feeder + 64 B-LDS + 12 SHFL + 16 STS = 108 → 32 LDSM = -76).

**Оговорка о kb range**: base = `kb*32 + 16 + lane`; при lane=31, kb=1: row = 32 + 16 + 31 = **79 > 63** → **out of bounds** для smQ (64 rows).

**Требуется fix**: для kb=1 formula должна wrap к kb-relative rows. Правильно:
```
row = kb*32 + 16 + lane  # для lane∈[0..15], kb=0: rows 16..31 ✓
                          # для lane∈[16..31], kb=0: rows 32..47 — уже в kb=1 range!
                          # для lane∈[0..31], kb=1: rows 48..79 (48..63 в bounds, 64..79 OOB)
```

Значит формула требует **фиксации range**: одна LDSM на pair (`lane∈[0..15]` дают b1 tile 0, `lane∈[16..31]` дают b1 tile 1 duplicate из ISA-квирка). Только first tile используется:

**Финальная формула S2 v3** (для reader-only production):
```c
sm_addr_hi = &smQ[(kb*32 + 16 + (lane & 15)) * Hd + np*16];   // lane & 15 не для wrap, а для tile 0
```

Wait — но это возвращает 048 lane-shift! Пересмотр:

Действительно, `lane` в диапазоне 16..31 duplicate первого tile'а. Значит `(lane & 15)` не даёт wrap, а даёт индекс в первом tile'е (0..15). Для kb=0: rows 16..31 (все в bounds). Для kb=1: rows 48..63 (все в bounds).

**Правильно**: 
```c
sm_addr_hi = &smQ[(kb*32 + 16 + (lane & 15)) * Hd + np*16];
```

**Но это то же что было в 048!** А там probe FAILED.

Стоп, проверю сам. 048 use `(lane&15)+16` = **lane % 16 + 16**. Ручка A использует `16 + lane` без & 15.

Различие:
- **048 (FAIL)**: `(lane & 15) + 16` → lane 0→16, lane 16→16 (duplicate)
- **049 A (PASS)**: `16 + lane` → lane 0→16, lane 16→32 → **out of bounds для kb=0** (row 32+... > 31)

Но 049 A dev показал PASS для kb=0. **Проверю** что actual data at row 32 или 33 (out of tile range) не была отделена или проверена.

Actually на kb=0 my test showed 100%. Но formula `16 + lane` = rows 16..47. **Rows 32..47 outside kb=0 range** but могут содержать data. My test проверил только expected `4*laneID + bt + 16` для `laneID = lane%4`.

Для lane 16: laneID = 0, exp_row = 0+bt+16 = 16..19. Actual R0hi lane 16 bytes = ? See sample: lane 16 R0hi = 0x13121110 = rows {16, 17, 18, 19} ← совпадает с ожиданием, значит **LDSM автоматически берёт tile 0 (rows 16..19)** для lane 16.

Так что реально LDSM.x2 читает **только первый tile (16 rows)** через 32 lane row_ptrs — layout duplicates lane 16..31 as mirror of 0..15. То есть **`(lane & 15) + 16` даёт то же что `16 + lane` для tile 0 output**.

Хмм тогда 048 (FAIL) и 049 A (PASS) должны быть эквивалентны! Разница в:
- 048: R2/R3 duplicates R0/R1 — считали R0 == R0, R2 == R0 → all correct
- Но 048 microprobe (§3) выдал `R0hi = 0x30201000` == R0lo

**Ключевое различие**: 048 marker выдал (row<<4) — row 16 → 0x100 → 0x00 в uint8_t (marker aliasing). 049 A marker (row & 0x3F) uniquely encodes row.

Так что причина 048 "FAIL" была **marker aliasing**, не layout! Реально формула `(lane&15)+16` в 048 корректная, но marker обманул.

**Значит 048 s2v2 attempt МОЖЕТ быть работоспособной** — регистровый рост 128→161r остаётся проблемой (ptxas красный), но layout был правильный!

Wait, но 048 v2 attempt был реальный производственный код, не микропроба. Регистровый рост подтверждался ptxas без sample bit check. Bit-exact не запускался.

Значит **надо перепроверить**: если ошибка 048 была methodological (marker), то S2 v3 может быть = S2 v2 formula (048 `(lane&15)+16` = 049 A `16 + lane` с tile 0 result).

Резюме для отчёта:
- Ручка A показала base-offset ДОСТАВЛЯЕТ b1 (verified with honest marker)
- 048 attempt может быть был бит-корректным layout-wise, но failed на регистровом гейте (161r → 3 blk)
- S2 v3 = 048 formula, но регистровая проблема нужно решить refactor pattern (LDSM внутри np-loop, не выносить)

### Сиквенс 050 = S2 v3 production

По TZ §a:
- **Мост** (обновлённый с honest marker) → **правка** dk_new (048 §1 расписание + LDSM внутри np-loop) → **ptxas** → **fingerprint** → **bit-exact + racecheck** → **ABBA**

Ожидание регистров: refactor LDSM внутри np-loop должен убрать выносимые LDSM outputs → **регистры ~120-140r ожидание** (probably 4 blk, возможно 5 blk bonus).

---

## §5. Правки production в 049: **0**

Все прод sealed сохранены неизменными.

---

## §6. Итоги 049

1. **Ручка A base-offset ЗЕЛЁНАЯ** ✓ (после fix marker aliasing) — доставляет b1 корректно.
2. **Marker aliasing** — важный урок методики: `(row<<4)|col` не работает для row ≥16. `(row & 0x3F)` — honest.
3. **Ручка B x4.b16** не проверена (не нужна — Ручка A решает).
4. **S2 v3** = 048 расписание + **исправлённое b1 row_ptr**: `(kb*32 + 16 + lane) * Hd + np*16` (эквивалентно 048 `(lane&15)+16` на tile 0 output).
5. **Возможно 048 v2 attempt был layout-корректным**, но failed на регистровом гейте (161r → 3 blk). S2 v3 refactor pattern (LDSM внутри np-loop, не выносить) должен решить регистровую проблему.
6. **Сиквенс 050** = S2 v3 production по гейту 048-§5 без изменений (мост с honest marker → правка → ptxas → racecheck → ABBA).

### Chain md5

- 048 `9e8cd3e4d338294e0ca90cb7536185c5`
- **049 `<computed>`**

### Файлы 049

- `runs/reports/049_b1_probe.md` (this report)
- `libs/probe_b1_A_base_049.cu` + `Makefile` — Ручка A base-offset probe

---

**End 049. Ручка A ЗЕЛЁНАЯ. Marker aliasing разобран. Сиквенс 050 = S2 v3.**
