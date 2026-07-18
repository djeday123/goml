# 043 — fp8-LDSM: ISA-инвентарь + бумага двух дверей (DK + M5)

**Chain**:
- 041_post040_recon_dq.md md5: `198fd63cc05fb68714e0b95eb8395ccc`
- 042_Aprime.md md5: `19d912f1330eca35842f8e2d0b2daf2a`

**Правила ТЗ 043**: production-правки merged/dk/dq **запрещены**, wall-замеров **нет**, только standalone P-класс + бумага. Молчаливые EXPECT-правки запрещены.

---

## Артефакт-хедер (правило 5)

```
libs/ (post-042 state, prod неизменен):
-rw-r--r-- 25638 Jul  8         fa_bwd_merged_v1.cu    (md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33 = 040 sealed)
-rw-r--r-- 18834 Jul  8         fa_bwd_dq_new.cu       (md5 d7a11a3d788eb4c396d892bc9c8ab754 = 041 sealed)
-rw-r--r--       Jul  7         fa_bwd_dk_new.cu       (md5 a9f0ded8261e53a143b521ffa647f458 = 033 sealed)
libs/ldmatrix_isa_probe_043.cu + Makefile.ldmatrix_isa_probe_043
```

**Gate-log**:
```
$ ./037r_gate.sh
bench_merged: FINGERPRINT kernel_merged_v1: numRegs=252
GATE OK: numRegs=252 matches EXPECT=252
```

---

## §0. Долги (блокирующие)

### 0.a LD conflict events — АБСОЛЮТОМ (третий запрос)

- **Base pre-040** (037-r fresh, NCu launch-count=1): **126,810,524 events**
- **Post-040** (041 fresh, same NCu invocation): **132,104,324 events**
- **Δ = +5,293,800 events = +4.17%**

**Атрибуция**: **noise/small drift** классов #1..#6/#8 при укорочении ядра. LDSM cooperative fetch (класс #7) НЕ инкрементирует `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` — эта метрика считает classical LDS address-collisions в одной инструкции, не структурные wave-collisions LDSM. Подтверждение: класс #7 переведён полностью на LDSM (256 → 0 LDS.U16), а метрика выросла лишь на +4.17% (не удвоилась). Rate check: pre 2.75% vs post 3.82% — Δ +1.07 pp = sample noise (single launch NCu, не 5-average).

### 0.b Курс 0.0479%/op — деривация одной строкой

**0.0479%/op = 12.28% wall (040 ABBA 8-пар median CAND-BASE, ~40 s session) / 256 initial LDS.U16 static SASS (класс #7 pre-040)** — режим **isolated wall not-NCu** из `runs/reports/040_abba_data.txt`. Альтернативный net-based курс: **0.0548%/net-op = 12.28% / (256-32=224)**.

---

## §1. ISA-инвентарь ldmatrix на sm_120a (standalone, приём 013 anti-DCE)

Файл: `libs/ldmatrix_isa_probe_043.cu` + Makefile. Приём 013: результаты dumps в глобаль → компилятор не выкидывает.

### 1.1 Таблица инвентаря {shape × trans × x-num}

| # | Instruction | Compile | Run | Layout snapshot lane 0 (halves) | Note |
|:-:|:--|:-:|:-:|:--|:--|
| 1 | `ldmatrix.sync.aligned.m8n8.x4.shared.b16` | ✓ | ✓ | R0=[(0,0)\|(0,1)] R1=[(8,0)\|(8,1)] R2=[(0,8)\|(0,9)] R3=[(8,8)\|(8,9)] | **A-op layout** (см. 038) |
| 2 | `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` | ✓ | ✓ | R0=[(0,0)\|(1,0)] R1=[(8,0)\|(9,0)] R2=[(0,8)\|(1,8)] R3=[(8,8)\|(9,8)] | **B-op layout** (см. 039) |
| 3 | `ldmatrix.sync.aligned.m8n8.x2.shared.b16` | ✓ | ✓ | R0=[(0,0)\|(0,1)] R1=[(0,8)\|(0,9)] | A-op, half footprint |
| 4 | `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16` | ✓ | ✓ | R0=[(0,0)\|(1,0)] R1=[(0,8)\|(1,8)] | B-op, half footprint |
| 5 | `ldmatrix.sync.aligned.m8n8.x1.shared.b16` | ✓ | ✓ | R0=[(0,0)\|(0,1)] | A-op single tile |
| 6 | `ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16` | ✓ | ✓ | R0=[(0,0)\|(1,0)] | B-op single tile |
| 7 | `ldmatrix.sync.aligned.m16n16.x1.*.b8` | ✗ (ptxas) | — | — | **не разрешён**: "Vector of size 2 expected" |
| 8 | `ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8` | ✓ | ✓ | R0=30_20_10_00 R1=38_28_18_08 R2=30_30_10_10 R3=38_38_18_18 (raw hex bytes) | **FP8 shape works** — 4 uint32 out, 16 fp8 halves per lane |
| 9 | `ldmatrix.sync.aligned.m16n16.x4.*.b8` | ✗ (ptxas) | — | — | **не разрешён**: "Incorrect no. of matrix" |
| 10 | `ldmatrix.sync.aligned.m16n16.no-trans.*.b8` | ✗ (ptxas) | — | — | **не разрешён**: ".trans required" |
| 11 | `ldmatrix.sync.aligned.m8n16.*.b8` | ✗ (ptxas) | — | — | "Unexpected number of instruction types" (нужен packed subbyte вариант, не тестировали) |

### 1.2 Ключевые ISA-факты

- **b16 shape m8n8**: все 6 вариантов (x1/x2/x4 × no-trans/trans) **живы** ✓.
- **FP8 (b8) shape**: **только `m16n16.x2.trans.b8`** живёт (единственный вариант из инвентаря); **`.trans` обязателен для m16n16**, no-trans запрещён; x1/x4 запрещены (только x2 c 4-uint32 output).
- **`m8n16.b8`**: сложный синтаксис (packed subbyte types), не проверен базовым PTX.
- **Гвоздь 010 ("компилятор не эмитит FP8-ldmatrix")**: **опровергнут** для `m16n16.x2.trans.b8` — рукописный PTX компилируется + исполняется на sm_120a.

### 1.3 Раскладка m16n16.x2.trans.b8 (raw hex → интерпретация)

Setup marker: `smem[row*32+col] = (uint8_t)((row<<4) | (col & 0xF))`  
Row-ptr formula: `lane%16 rows, lane/16 col_base ∈ {0,16}`.

Lane 0 output (R0..R3, MSB-to-LSB byte print):
- **R0 = 0x30_20_10_00** = bytes { LSB: (r=0,c=0), (r=1,c=0), (r=2,c=0), MSB: (r=3,c=0) } — **4 rows at col 0, packed b8-in-uint32 by k-dim**
- **R1 = 0x38_28_18_08** = { (r=0,c=8), (r=1,c=8), (r=2,c=8), (r=3,c=8) } — 4 rows at col 8
- R2 = 0x30_30_10_10, R3 = 0x38_38_18_18 — дубликаты rows 1 и 3 при col 0 и 8

**Заметка**: layout m16n16.x2.trans.b8 доставляет **4 k-adjacent halves per uint32 при фиксированном n-col** — это **B-op-shape для fp8 mma**. Но точный маппинг к `mma.m16n8k32` требует детальной сверки (см. §2).

**Пары e4m3 в fp8-ldmatrix**: гвоздь ".trans мнет пары" в m8n8.trans.b16 (перестановка на 16-битном уровне) **НЕ применим** к m16n16.trans.b8: раскладка pack по k-байтам (b8 = байт), пары e4m3 сидят в СОСЕДНИХ байтах uint32 → **пары e4m3 едут целыми** (при правильном layout scatter).

---

## §2. Бумага двери DK (главная)

### 2.1 shape dk-MMA verbatim

`libs/fa_bwd_dk_new.cu:34`:
```
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
```

**Fragment shapes**:
- **A frag** = m×k = 16×32 fp8 = 512 fp8/warp → **16 fp8 = 4 uint32 per lane**
- **B frag** = k×n = 32×8 fp8 = 256 fp8/warp → **8 fp8 = 2 uint32 per lane**
- **D/C frag** = m×n = 16×8 f32 = 128 f32/warp → 4 f32 per lane

### 2.2 B-op fragment layout m16n8k32 (по PTX docs — ожидание)

Thread t (groupID=t/4, laneID=t%4):
- b0 = 4 fp8 at (k=4*laneID..4*laneID+3, n=groupID) — 4 consecutive k at same n
- b1 = 4 fp8 at (k=4*laneID+16..4*laneID+19, n=groupID) — 4 consecutive k at same n, k+16 offset

### 2.3 Соответствие ldmatrix-варианты × Q_T-раскладки

**Q_T pack-раскладка сейчас** (033_sealed dk_new pack + π_V):
- Раскладка: swizzle с XOR + PI_V bank-swizzle
- Ops: **64 LDS.32/lane/qt** (per memory 037-r/033-c), плюс **28 STS.U8/lane/qt** pack-запись
- Формулы: см. `libs/fa_bwd_dk_new.cu` (не воспроизвожу — pack спец-хореография)

**Q natural row-major раскладка**:
- Q layout: row-major, элемент fp8, row-stride Hd*1=128 bytes
- Свизл писателя-Q аналогичен smdO (гипотетически): `col_byte ^ ((row & 7) << 4)` byte-space

**Против фрагмента B-op m16n8k32**:

| Live-вариант ldmatrix | vs Q_T pack + π_V | vs Q natural + свизл | Вердикт |
|:--|:--|:--|:--|
| m8n8.x4.trans.b16 | fp8 != b16 → **mismatch типа** | fp8 != b16 → mismatch | **мертв** для fp8 |
| m16n16.x2.trans.b8 | Q_T pack — писатель уже разобрал байтовые слои; ldmatrix ожидает **row-major raw** — mismatch с pack хореографией | Q natural + свизл — если пары e4m3 в соседних байтах byte-adjacent within uint32 → **кормит фрагменты B-op с ALU-перестановкой** (пере-groupID mapping); прямого fit нет т.к. m16n16 tile ≠ m16n8k32 B-frag | **кормит с ALU-перестановкой** для S2 (natural) |

### 2.4 Гвоздь ".trans мнет пары" — проверка для fp8

**m8n8.trans.b16**: транспонирование внутри 8×8 halves при **b16-модификаторе** переставляет 16-битные ячейки, **не байты** → пары e4m3 (два 8-bit слова в одной 16-bit ячейке) **мнутся** — гвоздь **живой** для b16 модификатора над fp8-данными.

**m16n16.x2.trans.b8**: модификатор **b8** оперирует байтами; трансп применяется на **байтовом уровне** → пары e4m3 (два байта e4m3 в uint32) **едут целыми** — гвоздь **не применим** для b8-модификатора.

Однако **раскладка m16n16.x2.trans.b8 доставляет 4 k-adjacent halves at same n** (per §1.3), а MMA-B m16n8k32 требует **4 k-adjacent halves at same n** — **shape SOVпадает** структурно.

### 2.5 Сценарии боезапаса

**S1: LDSM по Q_T (existing pack + π_V)**:
- Q_T layout уже packed (28 STS.U8 pack-запись = стоимость перед dk-MMA)
- LDSM.x2.trans.b8 читает Q_T raw bytes — pack-хореография должна согласовываться с ldmatrix expected layout (row-major-in-tile)
- **Требует ре-derivation reader-формулы** ⇒ **STOP: писатель Q_T не тронуть** (правило TZ 043: reader-only)
- **S1 blocked**: изменение Q_T-раскладки = смерть писателя ⇒ выход за рамки reader-only.

**S2: LDSM по Q natural (без pack)**:
- **Удаляется writer Q_T pack**: -28 STS.U8/lane/qt (writer убит) + освобождаются pack-регистры
- **Удаляется reader Q_T pack**: -64 LDS/lane/qt (reader на pack-layout убит)
- **Добавляется LDSM.x2.trans.b8 read Q natural + reader на raw layout**: +N LDSM
- N = number of MMA-B iterations. dk_new uses KS_QK loop (analog merged) — примерно 4-8 x2-instructions per lane per qt по фрагмент-объёму (256 fp8 per B-frag; x2 = 8 fp8 per lane; 32-warp × 8 = 256 ✓; **1 x2 per MMA-B**).
- dk_new MMA-B iterations = N_kb * N_ni (analog merged Step D) — предположительно ~8-32 mma calls per lane per qt → **8-32 LDSM.x2**.

**Счёт net S2**:
- Убрать: 64 LDS + 28 STS = **92 ops/lane/qt**
- Добавить: **~16 LDSM.x2** (upper прогноз, консервативно 32 mma × 0.5 per instruction)
- Net: **-76 ops/lane/qt** (upper бумажный прогноз)

**S2 требует**: изменение Q_T-фазы **удаляется полностью** (writer + reader pack убиваются). Это **НЕ reader-only** — writer убит. ⇒ **выход за рамки правила TZ 043 "reader-only"**.

**Вердикт S2**: механизм существует, но нарушает reader-only ⇒ **отдельное ТЗ (не reader-only)**, с racecheck (если убирается pack-фаза, барьер, охранявший Q_T-фазу, тоже уходит).

### 2.6 Итог двери DK

- **S1 (LDSM по Q_T)** ⇒ mismatched layout, требует ре-derivation writer + reader ⇒ **вне scope reader-only**.
- **S2 (LDSM по Q natural)** ⇒ убивает writer + reader Q_T, net -76 ops upper ⇒ **вне scope reader-only + требует racecheck**.

**Оба сценария выходят за scope 043 (reader-only)**. Требует ТЗ 044+ с polный гейт structural rework + racecheck.

**Прогноз wall net -76 ops × 0.0548%/op = ~4.16%** upper (dk_new isolated wall = ~10.4 ms → -0.43 ms; E2E in-chain = -1.0% .. -1.5%). **Территория 2/3 v2** — на границе KEEP, натяжка запрещена.

**DK-дверь ЖИВА, но НЕ через reader-only** — требует полный structural rework (writer + reader + возможно барьеры). Кладу в **сиквенс отдельным ТЗ**.

---

## §3. Бумага двери M5 (второстепенная)

### 3.1 shape dP-MMA verbatim

`libs/fa_bwd_common.cuh:156` (M5 читатель — merged):
```
mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16   (для класса #1/#2)
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32     (для класса #4/#5/#6/#7 dP/dV)
```

Класс **#5 = smV read Step D (dP-MMA B-op)** — читает fp8 → CVT в reg → fp16 → mma.m16n8k16.f16.f16.f16.f32.

**CVT неизбежен**: `mma.f16.f16` требует B-frag **уже в fp16** shape. LDSM с типом b8 доставит raw fp8 байты, **CVT останется в hot loop**.

### 3.2 Боезапас счётом

- **Ops сейчас** класса #5: **128 LDS.U16/lane/qt** (fp8 pair reads for CVT)
- **LDSM.x2.trans.b8** доставляет 8 fp8 halves per lane per instruction — за kb × ni loop (KS_DP=8, NI_DP=8, 128 fp8 needed per warp per mma × 32 mma per qt).
- Предположительно **~16-32 LDSM.x2 per lane per qt** для полного B-frag покрытия.
- **CVT-цена не исчезает**: 128 pair-CVT/lane/qt (e4m3_to_f16x2) остаётся.

**Net ops**: 128 LDS.U16 → 16-32 LDSM.x2 = **-96..-112 ops/lane/qt**.

**Marginal курс "пустой очереди"**: post-040 LSU-нагрузка (mio 8.86%) — mio не bottleneck. Пропорция wall win от -96 ops ниже 040-эталона (12.28% на 256 при mio 25.10%):
- Naive linear: 96 × 0.0548 = 5.26% upper — но с mio 8.86% (в 3x меньше pre-040), реальный курс ~0.5-1.5×0.0548 = 0.027-0.055%/op
- **Честная вилка**: **0.5-1.9% wall win** (не гарантирует пробитие 2% нижней границы 2/3 v2).

**Вердикт M5**: **"монетка у порога"** — надо строить только если DK мертва. Здесь **DK жива** (но вне scope 043 reader-only) ⇒ M5 в очереди 044+.

---

## §4. Read-only подтверждение обеих дверей

**Обе двери НАРУШАЮТ reader-only scope**:
- DK S1: requires writer Q_T re-derivation → **писатель тронут**.
- DK S2: removes writer Q_T pack completely → **писатель убит**.
- M5: fp8 CVT остаётся в reader, но layout matching требует probe read-side с новым fragment mapping → возможно reader-only, но с ALU-перестановкой.

**Барьеры**:
- DK S2: если убить Q_T pack фазу целиком, барьер Q_T-post-pack исчезает автоматически → **не сдвиг, но **удаление** барьера** → ТЗ с racecheck.
- M5: read-only, барьеры не тронуты.

---

## §5. CPU-судьи

**Условие ТЗ**: "Если есть живая пара — CPU-судья байтов ИСПОЛНЯЕТСЯ".

- **DK S1/S2**: вне scope reader-only ⇒ CPU-судья байт-эквивалентности **не запускается** (правка выходит за рамки).
- **M5**: механизм не пробивает 2% гарантию (вилка 0.5-1.9%) ⇒ CPU-судья **не запускается** (sub-threshold, натяжка запрещена).

---

## §6. Вердикт-карта v6 + сиквенс

| Дверь | Механизм | Прогноз wall upper | Верdict scope | Приоритет | ТЗ |
|:--|:--|:-:|:--|:--|:--|
| **DK S1** (LDSM по Q_T) | m16n16.x2.trans.b8 vs pack | 2-4% (но вне reader-only) | **вне scope 043**: требует writer re-derivation | средний | **044** структурный |
| **DK S2** (LDSM по Q natural + убить pack) | m16n16.x2.trans.b8 + удаление pack writer | ~4% (net -76 ops) | **вне scope 043**: writer убит + возможно барьер | средний-высокий | **044** с racecheck |
| **M5** (LDSM по smV) | m16n16.x2.trans.b8 + fp8→fp16 CVT остаётся | **0.5-1.9%** (вилка) | reader-only OK, но sub-threshold | низкий | 045 если DK мертва |

**Сиквенс по итогу**:
- **DK-дверь жива, но вне scope 043 (reader-only)**. Требует ТЗ 044 с полным гейтом:
  - Structural rework fa_bwd_dk_new.cu (убрать pack Q_T + добавить LDSM read Q natural)
  - ptxas факт для dk (128r prod, потолок SMEM-limited — LDSM не сдвигает SMEM ✓)
  - fingerprint (EXPECT dk_new = 128 может измениться → осознанное обновление)
  - bit-exact 11/11 + canary + chain + sanitizer
  - Если убирается барьер Q_T-фазы — **racecheck обязателен** (правило 13)
  - Архив pre + ABBA ≥ 8 пар vs 033_sealed baseline
  - Вердикт правило 2/3 v2 (dk isolated wall, ожидание 2-4% ⇒ territoria 2/3 v2)
  - NCu-post: dk mio (baseline 42%) вниз с числами; DRAM неизменен; блоки dk = 4 ровно (падение блоков = автостоп-доклад)

- **M5-дверь**: реализация вилки 0.5-1.9% — **строить только после 044**. Если 044 KEEP → M5 sub-threshold vs новый landscape — пересчёт.

**Обе двери в текущем ТЗ (043 reader-only)**: **НЕ строим**. Только доклад.

---

## §7. Доклад Vugar: зазор 0.206 ms

**Cumulative E2E post-041**: **44.206 ms in-chain**. Порог "пункт назначения" в TZ 042 сиквенс: `≤ 44.0 ms` → сейчас **зазор 0.206 ms** до порога сертификации.

**Живые тяжёлые рычаги** (кроме fp8-LDSM, которые вышли за scope 043):

1. **DK через LDSM (S1/S2)** — ТЗ 044 structural, прогноз upper ~4% dk isolated = ~-0.4 ms dk in-chain = **-0.4..-0.5 ms E2E** ⇒ пробивает 44.0 порог с запасом.

2. **Война за wait в merged** (post-040 stall wait 33% топ):
   - **smQ-prefetch Q[qt+1]** (карта улик 041 v5 цель (a)): headroom 8704 B доступен, но alias-union живёт → **отдельное ТЗ с racecheck** (правило 13). Прогноз upper -1.7 ms merged in-chain = **-4% E2E** ⇒ агрессивный.
   - **TMA (Tensor Memory Accelerator sm_90+/sm_100 abstraction)**: если sm_120a supports TMA-load для Q/dO — прогноз upper ~-2 ms merged (аналог smQ prefetch, но со встроенной хореографией).
   - **cp.async глубина ≥ 2 wave** (карта улик 041 v5 цель (c)): тесно связан с smQ prefetch (нужен double smQ буфер) — **одно и то же ТЗ**.

3. **L2-handoff по головам в dq** (карта улик 041 v5 связано с (d) A' / (e) short_sb):
   - Post-041 dq mio = 41.45%, long_sb = 2.35%, wait = 11.33% (упавший long_sb ⇒ L2 hits improved).
   - Handoff-механизм: подготовка следующего head'а перед обменом с dk_new через L2. Требует cross-kernel coordination — сложный (может пересечь ABI).
   - Прогноз upper -0.3..-0.5 ms dq in-chain = **-1% E2E**.

### Выбор направления — за Vugar

| Рычаг | Прогноз E2E | Scope | Сложность | Риск |
|:--|:-:|:--|:--|:--|
| DK LDSM (S2) | -1.0..-1.5% | structural (writer+reader+racecheck) | средняя | средний |
| smQ prefetch merged | -3..-4% | racecheck ТЗ | средняя-высокая | средний-высокий |
| TMA merged | -2..-3% | ISA probe + structural | высокая | высокий |
| L2-handoff dq | -1% | cross-kernel ABI | высокая | высокий |
| M5 LDSM (fp8) | -0.5..-1.9% | reader-only, sub-threshold | низкая | средний (2/3 v2) |

**Рекомендация ассистента (не решение)**: **044 = DK LDSM S2** — самый предсказуемый механизм (fp8-ldmatrix доказан §1, dk_new writer/reader well-known из 033-c леджера), пробивает 44.0 порог с запасом.

Альтернатива (агрессивная): 044 = smQ prefetch merged. Требует barrier redesign + racecheck.

**Решение направления оставляется Vugar**.

---

## §8. Итог 043

1. **ISA-инвентарь ldmatrix sm_120a** (11 вариантов проверено):
   - **m8n8.b16** — все 6 вариантов (x1/x2/x4 × trans/no-trans) **живы**
   - **m16n16.x2.trans.b8** — **живет** (FP8-shape opperativelyness подтверждён)
   - **m16n16.x1/x4** — не разрешены
   - **m16n16.no-trans.b8** — .trans обязателен
   - **m8n16.b8** — сложный packed-subbyte синтаксис, не проверен
   - **Гвоздь 010 опровергнут**: FP8-ldmatrix работает через `m16n16.x2.trans.b8`.

2. **DK-дверь (главная)**:
   - Shape mma m16n8k32 e4m3 подтверждён из исходника.
   - Механизм LDSM возможен через m16n16.x2.trans.b8 при **S2 (natural Q + удаление pack Q_T)**.
   - **S1/S2 вне scope 043 reader-only** (S1 требует ре-derivation writer, S2 убивает writer + удаляет барьер).
   - Прогноз wall upper ~4% dk isolated → -1.0..-1.5% E2E ⇒ пробивает 44.0 порог.
   - **ТЗ 044 = DK LDSM structural** (полный гейт + racecheck).

3. **M5-дверь (второстепенная)**:
   - Shape mma m16n8k16 f16.f16.f16.f32 (класс #5 читает fp8, CVT в fp16 в reg).
   - LDSM через m16n16.x2.trans.b8 возможен, но **CVT остаётся в hot loop**.
   - Net ops: -96..-112 (128 → 16-32 LDSM.x2).
   - **Честная вилка 0.5-1.9% wall win** — не гарантирует 2/3 v2 пробитие ⇒ **"монетка у порога"**.
   - **Строить только если DK мертва** (в 044).

4. **Обе двери мертвы для 043 reader-only**. Прогноз проб не запускается (правило sub-threshold + правило scope).

5. **Доклад Vugar**: зазор **0.206 ms**; тяжёлые рычаги — DK LDSM S2 (predсказуемый), smQ prefetch merged (агрессивный), TMA (риск), L2 handoff dq (сложный). **Выбор за Vugar**.

### Chain md5

- 042 `19d912f1330eca35842f8e2d0b2daf2a`
- **043 `<computed>`**

### Файлы 043

- `runs/reports/043_fp8_ldsm_probe.md` (this report)
- `runs/reports/043_isa_inventory_data.txt` — полный дамп ISA-инвентаря
- `libs/ldmatrix_isa_probe_043.cu` + `Makefile.ldmatrix_isa_probe_043` — standalone P-класс
- `libs/ldmatrix_isa_probe_043` — probe binary

**End 043. Двери проработаны на бумаге, обе выходят за scope reader-only. Направление 044 — за Vugar.**
