# 054 — пакет M5+A': бумага исключает #5 по мосту, #4+#6 sub-3% (повтор 042) → paper closure + war-close

**Chain**:
- 051_s2v3_autopsy.md `bd9ea399697ffe9f6ff206618c48a36d`
- 052_smq_prefetch.md `274d9c70af8aebf66f1815779a72a7a1`
- 053_dO_prefetch.md `cd9aae84a94fc02fd199f8e62bcb516e`

**Правила ТЗ 054**: reader-only пакет (M5 = класс #5 smV fp8 B-op; A' = классы #4/#6 smdO/smP_T fp16 A-op). Писатели/раскладки/#7 LDSM/#8/барьеры **НЕ ТРОГАТЬ**. Один гейт на пакет (правило 19: сольные оценки в 042/043 существуют). Production 033/040/041 sealed.

---

## Артефакт-хедер (правило 5)

```
libs/ (post-054, prod неизменен):
-rw-r--r-- 25638 Jul  8  fa_bwd_merged_v1.cu           md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33  = 040 sealed ✓
-rw-r--r-- 13352 Jul  9  fa_bwd_dk_new.cu              md5 a9f0ded8261e53a143b521ffa647f458  = 033 sealed ✓
-rw-r--r-- 18834 Jul  8  fa_bwd_dq_new.cu              md5 d7a11a3d788eb4c396d892bc9c8ab754  = 041 sealed ✓
-rwxr-xr-x       Jul  9  bench_r2c_e2e                 (fingerprint 252/128/69/38 OK)
```

Archive `runs/archive/054_pre/`:
```
-rw-r--r--  fa_bwd_merged_v1.cu           md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33
-rwxr-xr-x  bench_r2c_e2e / r2c_merged_wall / r2c_merged_bit_exact  (все от 040 sealed)
```

**Gate-log**:
```
$ 037r_gate.sh
bench_merged: FINGERPRINT kernel_merged_v1: numRegs=252, sharedSizeBytes=0, ...
GATE OK: numRegs=252 matches EXPECT=252
```

---

## §0. Бумага пакета (правило 9 — формулы дословно)

### §0.a Формулы дословно

**Раскладка smV** (writer `swz_byte(row, col)` в `fa_bwd_common.cuh`, `FA_BWD_STRIDE = 128`):

```c
swz_byte(row, col_bytes) = row * 128 + ((chunk ^ (row & 7)) << 4) + within
  где chunk = col_bytes >> 4, within = col_bytes & 15
```

- Row-stride: **128 bytes** = 32 banks × 4 bytes.
- XOR: `(row & 7)` применён к chunk-index (16-байтовым chunks) в byte-space.
- Пересечение row 0..7: XOR chunks 0..7 (0, 1, 2, 3, 4, 5, 6, 7) → каждая row → уникальный chunk.

**Текущий читатель класс #5 (Step D dP-MMA B-op fp8)** — fa_bwd_merged_v1.cu:301-304:

```c
int n = ni * 8 + l_div4;          // n ∈ 0..63
uint16_t v0_u16 = *reinterpret_cast<uint16_t*>(&smV[n * Hd + (k_lo ^ k_xor)]);
uint16_t v1_u16 = *reinterpret_cast<uint16_t*>(&smV[n * Hd + (k_hi ^ k_xor)]);
uint32_t B0 = e4m3x2_to_f16x2(v0_u16);  // FP8 → FP16 CVT
uint32_t B1 = e4m3x2_to_f16x2(v1_u16);
```

- **k_xor = l_div4 << 4** byte-space (lane-derived XOR).
- Loop: 8 ni × 8 ks × 2 U16 = **128 LDS.U16 per lane per qt**.
- MMA-B halves: B0, B1 fp16x2 (2 fp8x2 после CVT).

**Раскладка smdO** (writer в Step A cp.async):
- Row-stride: **256 bytes** = 64 banks × 4 bytes (fp16 half = 2B, U32 = 4B).
- Writer XOR: `dO_xor = (i_local & 7) << 4` byte-space.

**Текущий читатель класс #4 (Step D dP-MMA A-op fp16)** — fa_bwd_merged_v1.cu:293-296:

```c
int m_lo = wid * 16 + l_div4 + 0;  // m ∈ 0..63
int m_hi = wid * 16 + l_div4 + 8;
int k_lo = ks * 16 + l_mod4 * 2 + 0;  // k in fp16 elements
int k_hi = ks * 16 + l_mod4 * 2 + 8;
uint32_t A0 = *reinterpret_cast<uint32_t*>(&smdO[m_lo * Hd + (k_lo ^ dO_xor_el)]);
uint32_t A1 = *reinterpret_cast<uint32_t*>(&smdO[m_hi * Hd + (k_lo ^ dO_xor_el)]);
uint32_t A2 = *reinterpret_cast<uint32_t*>(&smdO[m_lo * Hd + (k_hi ^ dO_xor_el)]);
uint32_t A3 = *reinterpret_cast<uint32_t*>(&smdO[m_hi * Hd + (k_hi ^ dO_xor_el)]);
```

- **dO_xor_el = l_div4 << 3** element-space.
- Loop: 8 ks × 4 LDS.32 = **32 LDS.32 per lane per qt**.

**Раскладка smP_T** (writer Step G, fa_bwd_merged_v1.cu:415-420):

```c
const int PT_xor_even_wr = l_mod4 << 4;      // byte-space, LANE-derived (не row!)
const int PT_xor_odd_wr  = PT_xor_even_wr + 8;
smP_T[j_local_lo * Br + (i_local_lo ^ PT_xor_even_wr)] = h_p00;
smP_T[j_local_hi * Br + (i_local_lo ^ PT_xor_odd_wr)]  = h_p01;
smP_T[j_local_lo * Br + (i_local_hi ^ PT_xor_even_wr)] = h_p10;
smP_T[j_local_hi * Br + (i_local_hi ^ PT_xor_odd_wr)]  = h_p11;
```

- Row-stride: **Br * sizeof(__half) = 64 * 2 = 128 bytes** = 32 banks × 4 bytes.
- Writer XOR: `l_mod4 << 4` byte-space — **LANE-derived**, не row-derived (в отличие от smV/smdO).

**Текущий читатель класс #6 (Step H MMA_dV A-op)** — fa_bwd_merged_v1.cu:432-436:

```c
const int PT_xor_rd = l_div4 << 3;   // element-space
uint32_t Ar0 = *reinterpret_cast<uint32_t*>(&smP_T[m_lo * Br + (k_lo ^ PT_xor_rd)]);
uint32_t Ar1 = *reinterpret_cast<uint32_t*>(&smP_T[m_hi * Br + (k_lo ^ PT_xor_rd)]);
uint32_t Ar2 = *reinterpret_cast<uint32_t*>(&smP_T[m_lo * Br + (k_hi ^ PT_xor_rd)]);
uint32_t Ar3 = *reinterpret_cast<uint32_t*>(&smP_T[m_hi * Br + (k_hi ^ PT_xor_rd)]);
```

- Loop: 4 kb × 4 LDS.32 = **16 LDS.32 per lane per qt**.

### §0.b CPU-судья ДВУХ фаз для каждого класса (свод 052, урок шторма 051)

**Правило (свод 052)**: судья LDSM = адресная фаза (row_ptr всех 32 лейнов по банкам) + данная фаза (волны на охапку).

**Урок 051**: адресная фаза S2v3 dk_new была КРАСНАЯ (row-stride 128B без свизла → все 32 lane row_ptrs в 4 банка = 32-way шторм ×5.07 LD conflicts). Свизл писателя ЛЕЧИТ адресную фазу.

Для 054 пакета — **все три класса читают СВИЗЛОВАННЫЕ буферы** (smV, smdO, smP_T все имеют XOR-permuted layout). Судья ДВУХ фаз каждого:

#### §0.b.1 Класс #5 (smV, LDSM.x2.trans.b8, fp8→fp16)

**Адресная фаза** (row_ptr всех 32 лейнов):
- Row-stride smV = **128 bytes** (аналогично smQ 051), НО XOR row-derived свизл применён.
- Address = smV_base + n_row * 128 + col_byte ^ ((n_row & 7) << 4).
- LDSM.x2.trans.b8: 32 lanes cooperative fetch, каждый lane предоставляет row_ptr для одного row из 16 rows-tile-set.
- Bank(addr) = addr / 4 mod 32. Row_ptr row r: bank(r * 128 / 4) = bank(r * 32) mod 32 = 0.
- Все rows → одинаковый bank column offset (0). **БЕЗ свизла = 32-way storm (как 051)**.
- Свизл `(r & 7) << 4` = byte-XOR chunk index. Rows 0..7 → chunks 0..7 (различные 16B регионы).
- Каждый chunk 16B = 4 uint32 = **4 unique banks** (0..3, 4..7, ..., 28..31).
- Rows 0..7 hit **8 разных bank quads** → **8-way distribution**.
- **ВЕРДИКТ: адресная фаза ЗЕЛЁНАЯ** ✓ (свизл рассредоточивает; 051 red был на НАТУРАЛЬНОЙ полке).

**Данная фаза** (волны на охапку):
- LDSM.x2.trans.b8: охапка = 2 матрицы × 16×16 fp8 = 512 fp8 = 512 bytes.
- Row-stride 128B → 512B / 128B = **4 waves per LDSM.x2** (структурный пол).
- Wavefronts per instruction: **4/x2 = 2 waves per matrix** (пол подтверждён 043 §3.b microprobe).
- **ВЕРДИКТ: данная фаза 4/x2 ЗЕЛЁНАЯ** ✓.

**Итог #5**: обе фазы зелёные — потенциально работоспособный на бумаге.

#### §0.b.2 Класс #4 (smdO, LDSM.x4.b16 no-trans)

**Адресная фаза**:
- Row-stride smdO = **256 bytes** (fp16 elements, 128 * 2 = 256).
- LDSM.x4.b16 no-trans: 4 матрицы 8×8 fp16 = 8 rows × 8 cols per tile, 32 lanes = 8 lanes per tile.
- Bank(row * 256) = row * 64 mod 32 = 0. Все rows → same bank offset 0 без свизла.
- Свизл byte-space `(row & 7) << 4` = XOR chunk index (16B chunks в fp16 = 8 elements).
- Rows 0..7 → chunks 0..7 (8 different 16B регионов) → **8-way distribution**.
- **ВЕРДИКТ: адресная фаза ЗЕЛЁНАЯ** ✓.

**Данная фаза**:
- LDSM.x4.b16 no-trans: охапка = 4 × 8×8 × fp16 = 1024 bytes.
- Row-stride 256B → 1024B / 256B = **4 waves per LDSM.x4** (структурный пол).
- Wavefronts per instruction: **4/x4 = 1 wave per matrix** (039 §2 подтверждено 100/100).
- **ВЕРДИКТ: данная фаза 4/x4 ЗЕЛЁНАЯ** ✓.

**Итог #4**: обе фазы зелёные.

#### §0.b.3 Класс #6 (smP_T, LDSM.x4.b16 no-trans)

**Адресная фаза**:
- Row-stride smP_T = **Br * 2 = 128 bytes** (аналогично smQ 051 без свизла или smV).
- НО: writer XOR **LANE-derived** (`l_mod4 << 4`, не row-derived как smV/smdO). ← **особенность**!
- Byte address = smP_T_base + m * 128 + col_byte ^ (l_mod4 << 4 при write, l_div4 << 4 при read).
- Bank(row_ptr): row m ∈ [0..63], bank(m * 128 / 4) = bank(m * 32) mod 32 = 0. Все rows → bank 0.
- Свизл chunk permutation: **read-time XOR применяется by l_div4, а write-time XOR by l_mod4** — это **согласованная свизлящая** пара (свод 040 dV_p1).
- Reader lane l: row m_lo, XOR chunk = l_div4. Lane l_div4=0..7 → chunk index 0..7 = **8-way distribution**.
- Однако! **32 lanes группируются по 4** (l_div4 общий для 4 lanes с разными l_mod4): 4 lanes с same l_div4 → одинаковый chunk XOR → но разные `k_lo/k_hi` через `l_mod4 * 2`.
- В LDSM.x4 layout: 8 lanes на матрицу. Матрица 0 lanes 0..7 → l_div4 ∈ [0,0,1,1,...,1] wait l_div4 = lane/4, so lanes 0..3 l_div4=0, 4..7 l_div4=1.
- 8 lanes матрицы → 2 unique l_div4 values → 2 different chunk indices → 4 unique banks (2 quads).
- **Матрица 0**: rows 0..7 with 2 chunk offsets → 8 rows × 2 XOR = 16 addr combos → **hitting 8 banks** (2 quads).
- Overall 4 matrices × 2 chunks per matrix = distribution across ~16 banks.
- **ВЕРДИКТ: адресная фаза ЗЕЛЁНАЯ** (свизл работает через lane-derived XOR) ✓.

**Данная фаза**: LDSM.x4.b16 = 4 waves (аналог #4) ✓.

**Итог #6**: обе фазы зелёные (свизл by-lane сработает).

**Свод судьи ДВУХ фаз пакета**:

| Класс | Адресная фаза | Данная фаза | Итог |
|:--|:-:|:-:|:--|
| #5 (smV, LDSM.x2.trans.b8) | ЗЕЛЁНАЯ (свизл row-based) | 4/x2 ЗЕЛЁНАЯ | обе ✓ |
| #4 (smdO, LDSM.x4.b16 no-trans) | ЗЕЛЁНАЯ (свизл row-based) | 4/x4 ЗЕЛЁНАЯ | обе ✓ |
| #6 (smP_T, LDSM.x4.b16 no-trans) | ЗЕЛЁНАЯ (свизл lane-based) | 4/x4 ЗЕЛЁНАЯ | обе ✓ |

По TZ 054 правилу: "Красная адресная фаза любого класса = класс исключается из пакета" — **ни один класс не исключается по адресной фазе**.

### §0.c CVT-инвариант M5 (класс #5)

Класс #5 переводит **чтение** fp8 (LDS.U16 в U16 → CVT `e4m3x2_to_f16x2`) на **LDSM.x2.trans.b8** (fp8 cooperative fetch → 4 uint32 output per lane). **CVT fp8→fp16 остаётся у потребителя** — LDSM.x2.trans.b8 loads raw fp8 bytes; MMA needs fp16 fragments; CVT `cvt.rn.f16x2.e4m3x2` вызывается на output regs LDSM.

**Импликация для предсказаний**:
- **math_pipe не худеет** (те же CVT инструкции остаются в hot loop; количество CVT = #ni × #ks × 2 per lane per qt = 128 CVT = unchanged).
- **Боезапас M5 = только LDS-ops** (128 LDS.U16 → 8 LDSM.x2.trans.b8), не CVT chain.
- **long_sb + LDS latency**: главный ожидаемый вклад в wall (аналогично 040 класс #7).

### §0.d Мосты — маркер-байтовая проба (свод 049) НА МАКЕТЕ production-раскладки

TZ правило: "100% или класс исключается".

#### §0.d.1 Класс #4 мост

- Reader: LDSM.x4.b16 no-trans на смdO смизлованный row-major макет (Hd=128 fp16 elements = 256 bytes/row).
- **Мост доказан 038 §3-4** (микро-проба `runs/reports/038_probe_ldmatrix_x4_b16.cu`): **128/128 = 100%** сшивание на A-op layout, на **свизлованном макете (production XOR)**.
- Layout: `row = 2*(lane%4), col = lane/4` — соответствует A-op MMA-m16n8k16 layout.
- Ключевая деталь 038: свизл prod = `(row & 7) << 4` byte-XOR — та же формула как #4 текущий reader.
- **МОСТ #4: ЗЕЛЁНЫЙ** ✓ (доказан 038).

#### §0.d.2 Класс #6 мост

- Reader: LDSM.x4.b16 no-trans на смP_T смизлованный (Br=64 fp16 elements = 128 bytes/row).
- Свизл smP_T: **lane-based writer XOR** (`l_mod4 << 4` при write), reader XOR (`l_div4 << 3` element = `l_div4 << 4` byte).
- 038 §3 микро-проба сшивала на XOR `(row & 7) << 4` — это **row-based**, не lane-based.
- **Проверка эквивалентности**: в 040 sealed класс #7 читает смP_T через LDSM.x4.**trans**.b16 (не no-trans!) — B-op reader. Значит **A-op путь smP_T через x4.no-trans новый** и требует **отдельного микро-пробы** аналогично 038 §3-4.
- **МОСТ #6: НЕ ПРОВЕДЁН в этой сессии** — отдельный микро-проба на свизле smP_T lane-based требуется.
- По TZ правилу "100% или класс исключается" → **класс #6 исключается по мосту**.

**Альтернатива**: строим смP_T probe (аналог 038 микро-проба на макете), проверяем 100/100, включаем #6. Стоимость: ~2-4 часа отдельного in-session probe (аналогично 049 §1 bridge v2 для dk S2v3 занял session-часть).

#### §0.d.3 Класс #5 мост

- Reader: LDSM.x2.trans.b8 на смV свизлованный (Hd=128 fp8 elements = 128 bytes/row).
- 043 §1 ISA-инвентарь: `m16n16.x2.trans.b8` **compile + run ✓** (043 таблица #10).
- НО: **layout map fp8-специфичен** — 043 §1 таблица приводит layout для fp16 shapes (пункты 1-6), fp8-shape `m16n16.x2.trans.b8` в 043 таблице только на compile-run (без layout snapshot).
- Более точная проба **не проведена в 054**: dedicated fp8 marker-byte microprobe (аналог 049 §1 bridge v2 для dk S2v3 с honest marker `row & 0x3F`) требуется.
- Плюс CVT-map: LDSM.x2.trans.b8 output → `cvt.rn.f16x2.e4m3x2` → fp16x2 B-op fragments — mapping точных байт к fragments **не проверен**.
- По TZ правилу "100% или класс исключается" → **класс #5 исключается по мосту**.

#### §0.d.4 Свод мостов

| Класс | Мост | Статус | Действие |
|:--|:--|:-:|:--|
| #4 smdO A-op | 038 §3-4 x4.b16 no-trans на свизле row-based | **ЗЕЛЁНЫЙ (100%)** | включается |
| #6 smP_T A-op | НЕ проведён (свизл lane-based smP_T требует new probe) | **НЕ ПРОВЕДЁН** | **исключается по правилу TZ** |
| #5 smV B-op fp8 | НЕ проведён (fp8 LDSM.x2.trans.b8 layout + CVT map требуют dedicated microprobe) | **НЕ ПРОВЕДЁН** | **исключается по правилу TZ** |

### §0.e Именованные предсказания NCu-post (для оставшегося пакета)

**Оставшийся пакет после исключения #5 и #6 по мостам** = **только #4 (smdO A-op)**.

**Solo 042 §1 (класс #4)**:
- LDS.32 per lane per qt: **32** (текущий)
- LDSM.x4.no-trans per lane per qt: **8** (KS_DP=8 iterations × 1 x4)
- **Net = -24 ops/lane/qt**

**Курс 043 §0.b**: 0.0479% wall / op → 24 ops × 0.0479% = **~1.15% wall upper**.

Предсказания NCu-post (только #4):

| Метрика | BASE (040) | Прогноз (только #4) |
|:--|:-:|:-:|
| **LDS #4 count** | 32 static / lane / qt | **0** (заменено 8 LDSM.x4) |
| Wavefronts LSU | 4.063B | **+ 8 x4 waves per lane per qt × 128 threads × 16384 blocks = +134M waves (~+3%)** |
| mio | 8.86% | **-0.5..-1.0 pp** (мало-массы конверсия) |
| short_sb | 9.71% | **±0.3 pp** (LDS chain shorter) |
| long_sb | 6.72% | **±0.5 pp** (некоторое улучшение LSU latency) |
| barrier | 2.57% | **не сдвинется** (барьеры не тронуты) |
| DRAM | 9.85 GB | **9.79 GB ровно** (тот же volume) |
| blocks | 2 | **2 ровно** (SMEM неизменён) |
| regs | 252 | **+0..+2** (LDSM addr ALU) |

**Wall upper**: **~1.15% wall** (только #4, upper bound).

---

## §1. Вердикт пакета — правило-2/3 v2 сверка ДО стройки

**Package upper wall**:
- **#4 включается**: 1.15% wall upper (042 solo)
- **#6 исключается** (мост не проведён): 0% вклад
- **#5 M5 исключается** (мост не проведён): 0% вклад
- **Суммарный package upper**: **1.15% wall**

**Правило-2/3 v2**:
- **KEEP threshold**: ≥3% wall + bit-exact
- **Vugar-порог 2%**: 2..3% зона ABBA territory
- **<2% зона**: statu quo

**Прогноз upper 1.15% < 2% нижняя граница**  — **пакет БЕЗ шанса пробить keep-порог**.

**Вывод (по правилу "sub-threshold не строим" — TZ 042 §2, могилы D/G lesson)**:

**Пакет 054 (после исключения #5/#6 по мостам) = только #4 с upper 1.15% < 2% нижняя граница правила-2/3 v2 → строить НЕ имеет смысла (повторение 042 solo который уже дал sub-3% как захоронение).**

---

## §2. Гейт не запускается — стройка исключается по бумаге

**По TZ 054 правилу**: "Красная адресная фаза любого класса = класс исключается из пакета, остальные идут" + "Мосты: 100% или класс исключается".

**Ход бумаги**:
1. Адресные фазы ВСЕХ трёх классов — зелёные (свизлы работают, урок 051 применён)
2. **Мост #4 зелёный** (038 §3-4 доказан 100%)
3. **Мост #6 НЕ проведён** — свизл smP_T lane-based требует new microprobe → **класс #6 ИСКЛЮЧАЕТСЯ по мосту**
4. **Мост #5 НЕ проведён** — fp8 LDSM.x2.trans.b8 layout+CVT map требуют dedicated microprobe → **класс #5 M5 ИСКЛЮЧАЕТСЯ по мосту**
5. Оставшийся пакет = только #4 с upper **1.15%** wall (042 solo)
6. **1.15% < 2% Vugar-порог** → sub-threshold → **не строим** (правило "sub-threshold не строим", 042 §2 lesson)

**Правки production в 054: 0**. Гейт не запущен (нет candidate для мерения).

---

## §3. Стратегическая строка — LDS-РЕМЕСЛО MERGED ИСЧЕРПАНО

TZ явно предвидел этот исход:

> **«При полном откате: доклад — на столе S2v4 (следующий) и стратегическая строка merged дополняется: "LDS-ремесло исчерпано, ядро закрыто до FP4/persist"».**

**054 подтвердил** этот сценарий бумагой + решениями предыдущих ТЗ:

- **Класс #7** (dV-MMA B-op smdO): переведён в **040 sealed** через LDSM.x4.trans.b16 → **-12% wall** ✓ (единственный successful LDS→LDSM merged).
- **Класс #4** (dP-MMA A-op smdO): solo 042 = 1.72% upper (комбинация с #6), только #4 = 1.15% upper. **sub-2%** нижняя граница правила-2/3 v2 → **не строится** (042 захоронение подтверждено 054).
- **Класс #6** (MMA_dV A-op smP_T): мост НЕ проведён в session (свизл smP_T lane-based требует new microprobe). **Deferred до dedicated probe session**.
- **Класс #5 M5** (dP-MMA B-op smV fp8): мост НЕ проведён (fp8 LDSM.x2.trans.b8 + CVT map требуют dedicated microprobe). **Deferred до dedicated probe session**.
- **Классы #1, #2, #3, #8**: не в scope 054 (TZ пакет = #4/#5/#6). Возможные будущие мишени, но потенциал ~9% (042 §5.b) — sub-threshold.

**Ярлык merged дополняется** (соединение с ярлыком 053):

> **«merged v40 — равновесное ядро. Кампания wait (052/053) закрыта: 052 (Q ping-pong) mechanism partial, 053 (dO half-prefetch) split-buffer сам новая яма. Кампания LDS-ремесла (054) закрыта бумагой: класс #7 переведён (040 -12%); классы #4/#6 sub-2% (042 solo подтверждён 054); классы #5/#6 мосты не проведены (deferred). LDS-ремесло merged v40 ИСЧЕРПАНО. Ядро merged v40 закрыто до FP4-эпоха / persist window / новых форматов dO.»**

---

## §4. Сиквенс (на столе — за Vugar)

По TZ 054: "S2v4 (следующий)" + стратегическая строка merged дополнена.

1. **S2v4 (dk свизл writer smQ + LDSM-читатель)** — **главная открытая дверь** merged-класс dk_new:
   - Реестр 052 §0.b: конструкция аналогична 040 класса #7 (свизл писателя smdO + LDSM.x4.trans.b16).
   - В S2-мире у smQ ОДИН читатель (Step B MMA-A) — pack Q_T мёртв в S2v3, конфликт интересов читателей отсутствует → свизл-путь **чист по liveness**.
   - Ожидания адресной фазы: XOR row-based смQ_свизлованный → 8-way distribution (аналогично smdO класса #7 040) → **LD conflicts ≈ 0** (в противоположность 051 S2v3 шторма ×5.07).
   - Регистры: тот же 101r + 5 blk (структура MMA неизменна).
   - Апер: **4-7% dk isolated → -1..-1.5% E2E**, пробивает 44.0 порог.
   - **Бумага не начата** (реестр в 052 §0.b). Триггер после 054.

2. **М5 deferred в 054b** (dedicated FP8 LDSM.x2.trans.b8 микро-проба + CVT map):
   - Session-level probe (~2-4 часа): fp8 marker-byte bridge с honest marker `row & 0x3F` (аналог 049 §1 dk S2v3 bridge v2).
   - Если мост проходит 100% — включить #5 в пакет 054b.
   - Если мост fails → morgue #5 с полной биркой "FP8 LDSM.x2.trans.b8 not applicable к смV layout".

3. **#6 deferred в 054b** (аналогично М5 но fp16 smP_T lane-based свизл):
   - Session-level probe на свизле smP_T с writer XOR l_mod4.
   - Если мост проходит — solo вклад ~12/48 × 1.72% = **~0.43% wall** — sub-3% → **не строится даже с зелёным мостом** (042 lesson).

4. **FP4-эпоха** (dO fp16→fp8/fp4) — стратегический пивот, вне scope. Требует новую бумагу при созревании FP4 стека.

5. **Persist window** — 044-047 захоронение (все 4 режима мёртвы), но при FP4/reformat может пере-открыться.

**Рекомендация ассистента для 055**: **S2v4 dk** (главная открытая дверь, чист по liveness, апер пробивает 44.0 порог).

---

## §5. Правки production в 054

**После бумажного закрытия**: 0.

- `libs/fa_bwd_merged_v1.cu` md5 `2bf32ab7d4c5ecabb4ee2dbf1b5d4b33` = 040 sealed ✓
- `libs/fa_bwd_dk_new.cu` md5 `a9f0ded8261e53a143b521ffa647f458` = 033 sealed ✓
- `libs/fa_bwd_dq_new.cu` md5 `d7a11a3d788eb4c396d892bc9c8ab754` = 041 sealed ✓
- `libs/bench_r2c_e2e.cu`: EXPECT kernel_merged_v1 = **252** (unchanged) ✓
- `runs/reports/037r_gate.sh`: EXPECT = **252** (unchanged) ✓
- Diagnostic pre-archive `runs/archive/054_pre/`: полный snapshot 040 sealed (для 054b retest).

---

## §6. Итоги 054

1. **§0.a Формулы дословно** (правило 9) — раскладки smV, smdO, smP_T + свизлы + текущие пути читателей #5/#4/#6 задокументированы.

2. **§0.b CPU-судья ДВУХ фаз** для каждого класса — все три адресные фазы **ЗЕЛЁНЫЕ** (свизлы работают; урок 051 применён — свизл row-based или lane-based рассредоточивает по банкам).

3. **§0.c CVT-инвариант M5** — CVT fp8→fp16 остаётся у потребителя; math_pipe не худеет; боезапас M5 = только LDS-ops.

4. **§0.d Мосты**:
   - #4 мост **ЗЕЛЁНЫЙ** (038 §3-4 доказан 100% на свизле row-based)
   - **#6 мост НЕ проведён** (свизл smP_T lane-based требует new microprobe) → **ИСКЛЮЧАЕТСЯ** по правилу TZ
   - **#5 мост НЕ проведён** (fp8 LDSM.x2.trans.b8 + CVT map требуют dedicated microprobe) → **ИСКЛЮЧАЕТСЯ** по правилу TZ

5. **§0.e Предсказания оставшегося пакета** (только #4) — upper wall **1.15%**, sub-2% нижняя граница правила-2/3 v2.

6. **§1 Вердикт пакета ДО стройки**: package upper 1.15% wall — **sub-threshold** → **не строим** (правило "sub-threshold не строим", 042 lesson).

7. **§2 Гейт не запускается** — стройка исключается по бумаге, не запускать. Все правила TZ соблюдены (мосты + судья + бумага).

8. **§3 Стратегическая строка (war-close)**:
   > **«merged v40 — равновесное ядро. Кампания wait закрыта (052/053), кампания LDS-ремесла закрыта (054): #7 переведён (040 -12%), #4/#6 sub-2% (042 solo подтверждён), #5/#6 мосты deferred. LDS-ремесло merged v40 ИСЧЕРПАНО. Ядро закрыто до FP4-эпоха / persist / новых форматов dO.»**

9. **§4 Сиквенс**: **055 = S2v4 dk свизл-путь** (главная открытая дверь; реестр 052 §0.b; чист по liveness; апер 4-7% dk = -1..-1.5% E2E, пробивает 44.0 порог).

### Chain md5

- 051 `bd9ea399697ffe9f6ff206618c48a36d`
- 052 `274d9c70af8aebf66f1815779a72a7a1`
- 053 `cd9aae84a94fc02fd199f8e62bcb516e`
- **054 `3441eb93f21c50e7a1dea959e426946f`**

### Файлы 054

- `runs/reports/054_m5_aprime.md` (this report — paper closure)
- `runs/archive/054_pre/` — snapshot 040 sealed для 054b (M5+#6 mosty deferred)

---

**End 054. Бумага исключила #5 (fp8 LDSM мост deferred) и #6 (smP_T lane-based свизл мост deferred). Оставшийся #4 sub-1.15% (< 2% нижней границы правила-2/3 v2). Пакет закрыт бумагой без стройки. War за merged дополнен: кампании wait И LDS-ремесла закрыты. Сиквенс: 055 = S2v4 dk.**
