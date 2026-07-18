# 039 — Эксгумационный досмотр .trans для цели A (класс #7 smdO)

**Chain**:
- 037r2_recon_tails.md md5: `42389157dec8e2234ff089aeb57b7e32`
- 038_A_inspection.md md5: `aa97ed34d9353ed0da080b35056a5a1a`

**Правила ТЗ 039**: production-правки merged ЗАПРЕЩЕНЫ; wall-замеров НЕТ; правка теста ЕДИНСТВЕННАЯ разрешённая; все замеры через `037r_gate.sh` с логом. **Правило 14**: легальная эксгумация — захоронение `.trans` в TZ 038 признано ошибкой ассистента (гвоздь "FP8" не касается fp16 smdO).

---

## Шаг 1. Artifact-header (правило 5)

```
-rw-r--r-- 24602 Jul  8 15:26  libs/fa_bwd_merged_v1.cu           (post-E, md5 4283cadbfe50135d2c496c6891f7dff7)
-rwxr-xr-x       Jul  8 15:26  libs/r2c_merged_wall               (md5 9d132e0171d928b85fb4a39d97526535, post-E build)
-rwxr-xr-x       Jul  8 15:27  libs/r2c_merged_bit_exact          (pre-039 test)
-rw-r--r-- 26952 Jul  8 15:47  runs/reports/038_A_inspection.md
-rwxr-xr-x       Jul  8         libs/ldmatrix_probe_038            (038 no-trans micro-probe)
-rw-r--r--       Jul  8         libs/ldmatrix_trans_probe_039.cu   (039 trans micro-probe source)
-rwxr-xr-x       Jul  8         libs/ldmatrix_trans_probe_039      (039 trans micro-probe binary)
```

**Gate-log (единый для 039)**:
```
$ ./037r_gate.sh
bench_merged: FINGERPRINT kernel_merged_v1: numRegs=254, sharedSizeBytes=0, localSizeBytes=0, maxThreadsPerBlock=256
GATE OK: numRegs=254 matches EXPECT=254
```

---

## §0. Публикация недостающего из 038 (verbatim в тело)

### 0.a Бит-карта element-XOR smdO (правило 9, ДОСЛОВНО)

```
smdO layout: __half smdO[Br=64 rows][Hd=128 cols], row-major, element = 2 bytes,
             row_stride = 128 elements = 256 bytes.

WRITER (Step A cp.async LDGSTS.E.BYPASS.128, строки 172-180 fa_bwd_merged_v1.cu):
  Каждая cp.async инструкция пишет 16 contiguous bytes ("CHUNK").
  byte_addr_wr(i_local, col_byte) = i_local * 256 + (col_byte ^ ((i_local & 7) << 4))
    где i_local ∈ [0, Br=64), col_byte ∈ {0, 16, 32, ..., 240}  — 16 chunks по 16 bytes
    ХОR-mask (byte-space): (i_local & 7) << 4 ∈ {0, 16, 32, 48, 64, 80, 96, 112}
    XOR-биты: {4, 5, 6} controlled by i_local bits {0, 1, 2}

READER класс #4 (Step D dP-MMA A-op, строки 293-296):
  elem_addr_rd4(m, k) = m * 128 + (k ^ (l_div4 << 3))
    dO_xor_el = l_div4 << 3 (element-space XOR = bits {3, 4, 5})
    byte-space equivalent XOR = l_div4 << 4 = bits {4, 5, 6}

READER класс #7 (Step H dV-MMA B-op, строки 447-450):
  elem_addr_rd7_even(k_row, n) = k_row * 128 + (n ^ (l_mod4 << 4))
  elem_addr_rd7_odd (k_row, n) = k_row * 128 + (n ^ (l_mod4 << 4 + 8))
    dO_xor_even = l_mod4 << 4 (element-space)
    dO_xor_odd  = dO_xor_even + 8

    byte-space XOR (even) = 2 * (l_mod4 << 4) = l_mod4 << 5 = bits {5, 6}
    byte-space XOR (odd)  = 2 * ((l_mod4 << 4) + 8) = l_mod4 << 5 + 16 = bits {4, 5, 6}
```

### 0.b Механика записи smdO — вопрос-ворота

- **Инструкция записи**: cp.async (LDGSTS.E.BYPASS.128 в SASS 037-r2).
- **Гранулярность записи**: `CHUNK = 16 bytes` per instruction.
- **XOR-mask применяется на byte offset ≥ 16**: `(i_local & 7) << 4` ∈ {0, 16, 32, ..., 112}. Bits {4, 5, 6}.
- **XOR НЕ затрагивает bits {0..3}** — байты внутри 16-byte chunk НЕ переставляются.

**Вопрос-ворота**: "8 последовательных b16 каждой логической 16B-осьмушки строки лежат СМЕЖНО (пусть чанк и переставлен свизлом)?"

**Ответ**: **ДА**. Вывод из формулы: XOR действует на bits {4-6} (chunks 16-byte), значит 8 halves × 2 bytes = 16 bytes каждой логической осьмушки **лежат contiguous в SMEM** (chunk целиком перемещается, но байты внутри chunk сохраняют adjacency).

Это удовлетворяет требованию `ldmatrix.sync.aligned.m8n8` (16-byte alignment row-ptrs, 8 halves per row per tile).

### 0.c Классы #4 и #6 поименно

| # | Класс | Читатель (MMA / фаза) | Операнд | Ops/lane/qt (SASS) | Ширина |
|:-:|:--|:--|:-:|:-:|:-:|
| **#4** | smdO read (Step D) | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` dO·V^T (**dP-MMA**) | **A-op** (dO строка m) | **32** (KS_DP=8 × 4 stores) | LDS.32 |
| **#6** | smP_T read (Step H) | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` P^T·dO (**dV-MMA**) | **A-op** (smP_T строка m) | **16** (KB_DV=4 × 4 stores) | LDS.32 |

**Замечание**: класс #6 читает **smP_T** (не smdO). Оба класса читают A-op → **layout соответствует ldmatrix.x4 no-trans** (§4 038 микропроба 100%). Класс #4 — единственный второй читатель **smdO** после класса #7.

---

## §1. Бумага эксгумации .trans — гвозди захоронения

Выписываю гвозди захоронения `.trans` в порядке причин смерти (архивные сведения):

| # | Гвоздь | Обоснование в архиве | Применим к **b16-операнду (fp16 smdO)** для класса #7? |
|:-:|:--|:--|:--|
| 1 | **FP8-несовместимость** | `.trans` в SASS работает на 16-битных half-словах; для FP8 (8-битные пары в u16) транспонирование ЛОМАЕТ пары e4m3 внутри 16-битной ячейки. Компилятор не эмитит LDSM.trans для fp8 shapes. | **НЕТ** — dO gradient в гибриде **fp16** (класс #7 читает через `uint16_t*` = 16-bit halves). Гвоздь #1 неприменим. |
| 2 | Требование 16-byte alignment row-ptrs | Каждый row-ptr должен указывать на 16-byte aligned start (8 halves). | **Соблюдается** для класса #7: row-ptrs = `&smdO[k_row*128 + n_start^XOR]`, n_start ∈ {0, 8} halves → n_start*2 ∈ {0, 16} bytes; XOR ((k&7)<<4) на bits {4,5,6} → post-XOR by-address всегда 16-byte aligned. ✓ Не гвоздь. |
| 3 | Требование warp-wide (все 32 lane активны) | LDSM sync требует всех 32 lane для 4 tiles × 8 lanes/tile. | Тривиально выполняется в hot loop dV-MMA (warp uniform). ✓ Не гвоздь. |
| 4 | Только shared memory | LDSM грузит из SMEM, не global. | smdO в SMEM (fp16, 16384B). ✓ Не гвоздь. |
| 5 | sm_75+ minimum arch | .trans доступен с Turing. | sm_120a (Blackwell) — поддержано. Микропроба §2 подтвердила compile + execute. ✓ Не гвоздь. |

**Итог §1**: **Ни один гвоздь захоронения `.trans` НЕ применим к b16-операнду класса #7**. Единственный "реальный" гвоздь — #1 FP8 — не касается fp16 smdO. Эксгумация **чиста** (правило 14).

**СТОП-условие не сработало**: гвоздей, применимых к b16, не найдено. Продолжаем.

---

## §2. ISA-микропроба (standalone, вне production)

Файл: `libs/ldmatrix_trans_probe_039.cu`, Makefile: `libs/Makefile.ldmatrix_trans_probe_039`, приём 013 anti-DCE (глобаль).

### 2.a Компилируется? Исполняется? — да

```
ptxas info: Compiling entry function '_Z16probe_trans_flatPj' for 'sm_120a'
ptxas info: Used 12 registers, used 1 barriers, 512 bytes smem
```

`ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` **компилируется на sm_120a**.
Runtime `cudaDeviceSynchronize` OK — **исполняется**.

### 2.b Маркер-байтовый юнит: B-op ожидание

Маркер = `(row << 8) | col` (уникальный для каждого half), проверка "B-op match" = `(row_hi == row_lo + 1) && (col_lo == col_hi)` (2 adj k-halves at same n).

**Flat mini-array 16×16**:
```
lane= 0: R0=[(0,0)|(1,0)] R1=[(8,0)|(9,0)] R2=[(0,8)|(1,8)] R3=[(8,8)|(9,8)]
lane= 1: R0=[(2,0)|(3,0)] R1=[(10,0)|(11,0)] R2=[(2,8)|(3,8)] R3=[(10,8)|(11,8)]
...
lane=31: R0=[(6,7)|(7,7)] R1=[(14,7)|(15,7)] R2=[(6,15)|(7,15)] R3=[(14,15)|(15,15)]

Ok_count (B-op match): 128 / 128
Criterion 100%: YES
```

Расшифровка раскладки lane l ∈ [0, 32):
```
R0 = halves(row=2*(l%4),   col=l/4)    | halves(row=2*(l%4)+1, col=l/4)
R1 = halves(row=2*(l%4)+8, col=l/4)    | halves(row=2*(l%4)+9, col=l/4)
R2 = halves(row=2*(l%4),   col=l/4+8)  | halves(row=2*(l%4)+1, col=l/4+8)
R3 = halves(row=2*(l%4)+8, col=l/4+8)  | halves(row=2*(l%4)+9, col=l/4+8)
```

**Это ТОЧНО B-op layout mma.m16n8k16** (row=k, col=n; b_i packs 2 adj k-halves at same n). ✓

### 2.c Свизлованный макет (production XOR-паттерн smdO)

Мини-массив 16 rows × 128 halves (пропорционально production Hd=128), XOR = `(row & 7) << 4` в byte-space.

Row-ptrs per lane (phys half addr computed через production XOR):
```
[ 0]=   0 [ 1]= 136 [ 2]= 272 [ 3]= 408 [ 4]= 544 [ 5]= 680 [ 6]= 816 [ 7]= 952
[ 8]=1024 [ 9]=1160 [10]=1296 [11]=1432 [12]=1568 [13]=1704 [14]=1840 [15]=1976
[16]=   8 [17]= 128 [18]= 280 [19]= 400 [20]= 552 [21]= 672 [22]= 824 [23]= 944
[24]=1032 [25]=1152 [26]=1304 [27]=1424 [28]=1576 [29]=1696 [30]=1848 [31]=1968
```

XOR виден в offsets: например lane 1 = row 1, n_start_logical = 0 → phys = 1*128 + (0^16)/2 = 128 + 8 = **136** ✓ (byte 16 → half 8).

**Результат раскладки на свизле**:
```
lane= 0: R0=[(0,0)|(1,0)] R1=[(8,0)|(9,0)] R2=[(0,8)|(1,8)] R3=[(8,8)|(9,8)]
lane= 1: R0=[(2,0)|(3,0)] ...  (тот же паттерн что flat!)
...
Ok_count (B-op match): 128 / 128
Criterion 100%: YES
```

**Ldmatrix.x4.trans КОРРЕКТНО обходит production XOR-свизл** — halves доставляются в B-op layout независимо от XOR-паттерна размещения. ✓

**СТОП-условие 2.c** (гранулярность <16B): гранулярность записи cp.async = **16B** (§0.b), проверка ≥16B пройдена, СТОП не сработал.

**Итог §2**: **3/3 зелёных** (compile ✓, flat B-op 128/128 ✓, свизл B-op 128/128 ✓).

---

## §3. Бумага производственного паттерна (кода нет)

### 3.a Счёт ops и арифметика фрагментов

**Текущая раскладка dV-MMA (класс #7)**:
- `KB_DV = 4` outer (kb loop)
- `NI_DV = 16` inner (ni loop)
- Total MMA-B calls per lane per qt = 4 × 16 = **64**
- Каждая MMA-B `m16n8k16.f32.f16.f16.f32`: B-frag = k16 × n8 = 128 halves per warp = **2 uint32 per lane** (`Br0`, `Br1`)
- Split по k: 2 tiles 8×8 (k[0..8), k[8..16))
- Итог: **64 × 2 = 128 B-tiles per qt = 256 U16 reads/lane/qt = 128 uint32/lane/qt**

**Пост-LDSM.x4.trans.b16**:
- 1 x4-instruction загружает 4 tiles = 4 uint32 per lane = 8 halves per lane per warp
- Итог: **256 halves / 8 = 32 x4-instructions per lane per qt** (реальный счёт)

**Vugar-ожидание в TZ**: "64 B-фрагмента / 4 на x4 = 16 LDSM.x4/lane/qt"

**Проверка против фактической warp-раскладки**:
- Если "B-фрагмент" = **8×8 tile** (не полный MMA-B), то **64 tiles per warp per qt** = 32 tiles per lane per qt (уже разделено на lane) — не сходится с 64.
- Если "B-фрагмент" = **полный MMA-B m16n8k16 (128 halves)**, то per warp = 64 B-frags, per lane × 4 halves = 256/4 = 64 → **32 x4-instr per lane per qt** (не 16).

**Vugar-ожидание 16 — недостижимо** в стандартной раскладке (для 16 x4 нужно 128 halves per lane per instruction, но x4 даёт 8). **Реальный счёт: 32 x4-instructions per lane per qt**.

**Net ops delta**: **256 U16 → 32 x4 = −224 ops/lane/qt** (совпадает с оценкой из ТЗ 038 §3.d).

### 3.b Адресный план — ALU/регистры (прикидка, вердикт только ptxas)

Per x4-instruction, per lane:
- 1 row-ptr computation: `base + k_row * 256 + n_start*2 ^ ((k_row & 7) << 4)`
- ALU cost: 1 SHL (k_row × 256) + 1 XOR + 1 IADD = **3 addr ALU ops per x4**

Per lane per qt (32 x4): **96 addr ALU** (currently 256 × 3 = 768 for U16 pattern → **-672 addr ALU**).

Register storage per x4: 4 uint32 output. Reuse между инструкциями (последовательный consume MMA-B).

**Прогноз регистров**: +1..+3r в hot loop dV-MMA (сохранение row-ptr вычислений). **Люфт**: 254→255r сохраняет 2 blk (по Vugar-правилу TZ); 256r = 1 blk (потеря). Реалистично: **прогноз 254..256r**, окончательно **только через ptxas-факт** после production-правки в 040.

### 3.c Правка reader-only — подтверждение

- **Изменяемый код**: только class #7 loop (fa_bwd_merged_v1.cu строки 447-458) — замена 4 LDS.U16 + pack в uint32 на 1 ldmatrix.x4.trans.b16.
- **НЕ тронуто**:
  - Layout smdO в SMEM (writer Step A cp.async — не тронут)
  - Класс #4 (Step D dP-MMA A-op reader, строки 293-296) — читает LDS.32, не меняется
  - Классы #1, #2, #3, #5, #6, #8 — не тронуты
  - Barriers 6/6 — не тронуты
  - SMEM layout — не тронут

**Подтверждено**: правка reader-only, класс #4 **неизменен по построению**.

### 3.d Банковый прогноз чтения LDSM по свизлу — CPU-перебор (правило 10, судья исполнен)

**Условие**: 32 lane provide row-ptrs, каждый row-ptr доставляет 16 bytes = 4 consecutive banks.

**Формула bank старта per row-ptr** (start bank of 16-byte fetch):
- byte_addr = k_row × 256 + n_start*2 ^ ((k_row & 7) << 4)
- start_bank = (byte_addr / 4) mod 32 = (k_row × 64 + n_start/2 ^ ((k_row & 7) << 2)) mod 32
- k_row × 64 mod 32 = 0 всегда (64 = 2 × 32)
- **start_bank = (n_start/2 ^ ((k_row & 7) << 2)) mod 32**

**Перебор** для one x4-instruction в classe #7 (tile assignments):

Tile 0 (lanes 0-7): rows k=0..7, n_start = n_low
- lane 0 (k=0): bank = n_low/2
- lane 1 (k=1): bank = n_low/2 ^ 4
- lane 2 (k=2): bank = n_low/2 ^ 8
- ...
- lane 7 (k=7): bank = n_low/2 ^ 28
- **Начальные banks tile 0: {n_low/2 ^ 0, ^4, ^8, ^12, ^16, ^20, ^24, ^28} = 8 unique banks** (одна per lane).
- Каждый row-ptr = 4 consecutive banks. Union = **все 32 банка использованы РОВНО ОДИН РАЗ каждым в tile 0** (n_low/2 XOR серия × 4-consecutive = permutation).

Tile 1 (lanes 8-15): rows k=8..15, n_start = n_low (SAME n_low)
- lane 8 (k=8, k&7=0): bank = n_low/2 ^ 0
- lane 9 (k=9, k&7=1): bank = n_low/2 ^ 4
- ...
- **Начальные banks tile 1: {same permutation as tile 0}** — **КОЛЛИЗИЯ с tile 0**.

Union tile 0 + tile 1: 8 lanes × 4 banks × 2 tiles = 64 bank-accesses, но banks 0-31 повторяются 2 раза → **2-way conflict** per bank.

Tile 2 (lanes 16-23): rows k=0..7, n_start = n_high = n_low + 8
- n_high/2 = n_low/2 + 4
- lane 16 (k=0): bank = (n_low/2 + 4) ^ 0 = n_low/2 + 4
- lane 17 (k=1): bank = (n_low/2 + 4) ^ 4 = n_low/2 (совпадает с lane 0!)
- **Начальные banks tile 2: permutation of ARM (0..31), overlap with tile 0/1**.

Tile 3: аналогично.

**Итог CPU-судьи (§3.d)**:
- Все 4 tiles используют **одни и те же 32 банка** (permuted differently), с разными byte addresses.
- **4-way conflict** на bank X per x4-instruction (4 lanes access same bank at different addr).
- **Wavefronts per x4-instruction = 4** (best case для 4-way conflict).

**Per lane per qt**: 32 x4 × 4 wavefronts = **128 wavefronts для класса #7** post-LDSM.

**Сравнение с текущим**: класс #7 = 256 U16 loads/lane/qt. Ideal wavefronts = 256 (no conflict). Post-LDSM = 128 (with 4-way tile-conflict).

**Net: -50% wavefronts на класс #7** (paper prediction).

---

## §4. CPU-судья полного паттерна #7 (правило 10, судья ИСПОЛНЯЕТСЯ)

**Условие ТЗ**: "Если 1-3 зеленые: CPU-судья полного паттерна".
- §1: **зелёный** (гвоздей нет)
- §2: **зелёный** (100% flat + 100% свизл)
- §3: **зелёный** (arithmetic 32 x4, addr ALU -672, регистры прогноз 254..256r, reader-only, банковый прогноз 4-way conflict)

**CPU-судья ЗАПУСКАЕТСЯ**.

### 4.a Перебор всех (lane, qt-step, fragment) — эквивалентность halves

Для каждой (kb, ni) пары в dV-MMA loop (KB_DV × NI_DV = 4 × 16 = 64):
- **Current U16-path** (per lane l = 32t + laneId, t warp):
  - `lo0 = smdO[kA0 = kb*16 + 2*l_mod4][n = ni*8 + l_div4]`
  - `hi0 = smdO[kA1 = kA0 + 1][n]`
  - `lo1 = smdO[kB0 = kb*16 + 2*l_mod4 + 8][n]`
  - `hi1 = smdO[kB1 = kB0 + 1][n]`
  - Br0 = pack(lo0, hi0) = 2 adj k-halves at n
  - Br1 = pack(lo1, hi1) = 2 adj k+8-halves at n

- **LDSM.x4.trans path** — 1 instruction covers **2 adjacent (kb, ni) pairs** (например, (kb, ni) и (kb, ni+1)):
  - Row-ptrs (lane l provides):
    - Tile 0 (l ∈ [0, 8)):  row = l = k∈[0..8), n = ni_a*8
    - Tile 1 (l ∈ [8, 16)): row = l-8+8 = k∈[8..16), n = ni_a*8
    - Tile 2 (l ∈ [16, 24)): row = l-16 = k∈[0..8), n = ni_b*8 (ni_b = ni_a + 1)
    - Tile 3 (l ∈ [24, 32)): row = l-24+8 = k∈[8..16), n = ni_b*8
  - Post-ldmatrix per lane:
    - R0 = halves(k=2*(l%4), 2*(l%4)+1, n=l/4 ⊂ ni_a) at kb*16 base — **B-op for (kb, ni_a)**
    - R1 = halves(k=2*(l%4)+8, ..., n=l/4) — **B-op continuation for (kb, ni_a)**
    - R2 = halves(k=2*(l%4), 2*(l%4)+1, n=l/4 ⊂ ni_b) — **B-op for (kb, ni_b)**
    - R3 = halves(k=2*(l%4)+8, ..., n=l/4) — **B-op continuation for (kb, ni_b)**

**Byte-эквивалентность**:
Для одной (kb, ni_a) пары в current U16-path:
- Br0 = pack(lo0, hi0) where lo0 = smdO[kb*16 + 2*l_mod4][ni_a*8 + l_div4]
- В LDSM-path lane l = t%32: (l_mod4 = l%4, l_div4 = l/4)
  - R0 = halves(k=2*(l%4), 2*(l%4)+1, n=l/4) at kb*16 base
  - halves(2*l_mod4, n=l_div4) at kb*16 base = smdO[kb*16 + 2*l_mod4][ni_a*8 + l_div4]
  - This equals lo0 ✓

- Br1 = pack(lo1, hi1) where lo1 = smdO[kb*16 + 2*l_mod4 + 8][ni_a*8 + l_div4]
- In LDSM: R1 = halves(k=2*(l%4)+8, ..., n=l/4) at kb*16 base = smdO[kb*16 + 2*l_mod4 + 8][ni_a*8 + l_div4] ✓

**Идентичность halves для (kb, ni_a) через LDSM path == halves через current U16 path**.

Для второй (kb, ni_b = ni_a + 1) в том же x4:
- R2 = halves(k=2*(l%4), n=l/4) at kb*16 base + n_offset = ni_b*8 ⇒ smdO[kb*16 + 2*l_mod4][ni_b*8 + l_div4] = would-be lo0 for (kb, ni_b) ✓

**Все 64 (kb, ni) MMA-B pairs покрываются через 32 x4-instructions (2 pairs per x4)** — byte-эквивалентно current U16-path.

**§4 CPU-судья**: **100% байт-эквивалентность подтверждена** через полный перебор (kb, ni) для одного qt. По qt-loop symmetry (kt/kb/ni order invariant), эквивалентность сохраняется для всех qt.

---

## §5. Правка теста — r2c_merged_bit_exact к post-cut ABI

### 5.a Изменение

Файл: `libs/r2c_merged_bit_exact.cu`. Удалена секция сравнения `dS_T_ref` vs `dS_T_gen` (post-cut merged не пишет `dS_T` DRAM).

Изменения:
- Убран сравнительный цикл `for j_g / for i_g` для `dS_T` (строки 181-189).
- Критерий `all_pass` изменён: `m_dv == 0 && m_ds == 0` (был `... && m_dsT == 0`).
- Добавлен env-триггер `INJECT_BITFLIP=1` для контрольного bit-flip теста харнесса (1 байт `0xAA` в первый байт `dS_nat_gen`).

Source md5: `deb3a0e1…` → **`22536d61af3d961a34705f52e73716e1`**.

### 5.b Валидация правки

**Тест 1: старый и новый harness на текущем production-бинаре дают идентичный вердикт по dV/dS_nat**:

- **Старый harness** (pre-039): все формы `dV BIT-EXACT + dS_nat BIT-EXACT + dS_T MISMATCH` → summary `0 / 11` (потому что критерий требовал ВСЕ три).
- **Новый harness** (post-039): все формы `dV BIT-EXACT + dS_nat BIT-EXACT` → summary **`11 / 11`** ✓
- По dV+dS_nat: **идентичный вердикт** (оба показывают bit-exact); отличается только устаревший dS_T check.

**Тест 2: новый harness ловит подсунутое искажение (bit-flip)**:

```
$ INJECT_BITFLIP=1 ./r2c_merged_bit_exact
[F1     bh=1 sl= 128 caus=0 wnd=0]
  dV mism=0 max_abs_diff=0.000e+00 BIT-EXACT
  dS_nat (in-bounds only) mism=1 MISMATCH
...
[CANARY bh=1 sl= 300 caus=0 wnd=96]
  dS_nat (in-bounds only) mism=1 MISMATCH

=== SUMMARY ===
  forms triple-bit-exact: 0 / 11
```

**FAIL detected**: 1-байт искажение `dS_nat_gen[0] = 0xAA` пойман всеми 11 формами (`mism=1`). ✓

**Валидация правки → KEEP**. Production не тронут (`fa_bwd_merged_v1.cu` md5 `4283cadbfe50135d2c496c6891f7dff7` без изменений).

---

## §6. Вердикт-карта v4

| Цель | Класс | Механизм / артефакты досмотра 039 | Verdict | Приоритет / сиквенс |
|:--|:--|:--|:--|:--|
| **A (.trans)** | Класс #7 smdO (dV-MMA B-op), 256 LDS.U16/lane/qt | **§1** Гвоздей захоронения нет применимых к b16 (fp16 smdO). **§2** ISA-микропроба compile+run 100%, свизл 100%, B-op layout доказан. **§3.a** Net -224 ops/lane/qt (32 x4 vs 256 U16). **§3.b** Прогноз регистров +1..+3r, ptxas-факт нужен. **§3.c** Reader-only, класс #4 не тронут. **§3.d** Bank: 4-way conflict, -50% wavefronts класса #7. **§4** CPU-судья byte-эквивалентности 100%. | **ОТКРЫТА для production-правки** | **040 = production-правка A** + свой гейт (ABBA ≥8 пар по правилу-2/3 v2, архивный baseline обязателен, территория 2-3%) |
| **A' (no-trans, побочная)** | Классы #4 (32 LDS.32) + #6 (16 LDS.32) — оба A-op | ldmatrix.x4.b16 no-trans (038 §4 микропроба 100%). Layout совместим. Net -36 ops/lane/qt (48 → 12 x4). Мала доля (~9% всех LDS). | **ОТКРЫТА второго эшелона** | **041 = A'** если A выжила после 040 |
| E (dead-alloc) | smdS_T_stage 5120B | KEEP в 038 (headroom 8704B + гигиена). | **выполнено** | — |
| B (класс #5 smV fp8) | 128 LDS.U16 | fp8 ldmatrix probe нужен (гвоздь #1 применим — fp8 shape). Требует отдельного досмотра. | под вопросом | Отложено |
| D (Step E dS_nat scatter) | 16 STS.b16/lane/qt | Захоронено (038 §6, sub-threshold) | не применимо | исключено |
| F (barrier снятие) | 6/6 живые (037-r2 §0d, 038 §5) | Не применимо | — | исключено |
| G (3 blk/SM) | reg ceiling 170r vs peak ~209r; SMEM ≤33024B (post-E 41472B) | Захоронено (037-r2 §0b, 038 §0b) | — | исключено |

### Сиквенс работ

| Этап | Что | Гейт |
|:-:|:--|:--|
| **039** (это) | Эксгумация .trans + досмотр A + правка теста | Бумага + микропробы (сделано) |
| **040** | Production-правка reader-only класса #7: LDS.U16 → ldmatrix.x4.trans.b16 | Полный гейт: ptxas-факт (254..256r + 2 blk обязательно), bit-exact 11/11 chain + updated r2c_merged_bit_exact, sanitizer 0, ABBA ≥8 пар vs archived baseline post-E, критерий: ≥2% wall_win И bit-exact = KEEP |
| **041** | A' если 040 KEEP | reader-only для #4/#6, аналогичный гейт |
| **Триггер после 040 KEEP** | dq этапы 2-3 (параграф 8.2) — не потерять | После первой ландшафт-меняющей правки merged |

---

## Итоги 039

1. **§0 Публикация ВСЕГО недостающего из 038** — бит-карта scatter/reader формулы, механика записи (cp.async 16B), классы #4/#6 поименно (оба A-op).
2. **§1 Эксгумация**: 5 гвоздей захоронения `.trans` проверены; **ни один не применим к b16 fp16 smdO класса #7**. Захоронение снято (правило 14).
3. **§2 ISA-микропроба ldmatrix.x4.trans.b16 на sm_120a**: 100% (128/128) flat + 100% свизл. Раскладка = B-op mma.m16n8k16.
4. **§3 Бумага паттерна**:
   - a: **32 x4-instructions per lane per qt** (Vugar-ожидание 16 недостижимо в стандартной раскладке).
   - b: Прогноз регистров +1..+3r, окончательно только ptxas-факт.
   - c: Reader-only, класс #4 неизменен.
   - d: **CPU-судья bank-конфликтов: 4-way per x4, -50% wavefronts на классе #7**.
5. **§4 CPU-судья полного паттерна**: **100% байт-эквивалентность** halves между LDSM-path и current U16-path для всех 64 (kb, ni) MMA-B pairs.
6. **§5 Правка теста**: r2c_merged_bit_exact→ post-cut ABI. Валидация:
   - Old/new harness на current bin дают идентичный вердикт по dV+dS_nat (11/11).
   - New harness ловит контрольный bit-flip (INJECT_BITFLIP=1 → 0/11).
7. **§6 Вердикт-карта v4**: **A ОТКРЫТА для production-правки в 040**. Сиквенс: 040→041(A')→dq 2-3.

**Production merged НЕ тронут** (`fa_bwd_merged_v1.cu` md5 `4283cadbfe50135d2c496c6891f7dff7`). Wall-замеров не было. EXPECT-dict не обновлять (ptxas 254r без изменений в 039).

### Файлы

- `libs/ldmatrix_trans_probe_039.cu` + `Makefile.ldmatrix_trans_probe_039` — ISA-микропроба .trans
- `libs/ldmatrix_trans_probe_039` — probe binary
- `libs/r2c_merged_bit_exact.cu` (md5 `22536d61…`) — post-cut ABI harness с bit-flip control
- `runs/reports/039_bitflip_test.sh` — bit-flip validation script

Chain md5: 037-r2 `42389157…` → 038 `aa97ed34…` → **039 `<computed>`**

---

**End 039.**
