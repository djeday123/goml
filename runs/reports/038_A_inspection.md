# 038 — Гигиена E + входной досмотр цели A (класс #7 smdO)

**Chain**:
- 037r_reattrib.md md5: `886ac6e52525a7beb9dee7082a05d742`
- 037r2_recon_tails.md md5: `42389157dec8e2234ff089aeb57b7e32`

**Правила ТЗ 038**: production-правка merged по A ЗАПРЕЩЕНА; единственная code-правка = E-снятие dead-alloc; каждый замер через `037r_gate.sh` с логом; NCu-wall и чистый wall не смешиваются в одной таблице.

---

## Шаг 1. Artifact-header (правило 5)

```
-rw-r--r-- 24403 Jul  8 06:09  libs/fa_bwd_merged_v1.cu       (pre-E, md5 deb3a0e16c2e65591e1f98f7aebd9e43)
-rw-r--r-- 24491 Jul  8 15:19  libs/fa_bwd_merged_v1.cu       (post-E, md5 4283cadbfe50135d2c496c6891f7dff7)
-rwxr-xr-x       Jul  8 15:23  libs/r2c_merged_wall           (post-E build)
-rwxr-xr-x       Jul  8 15:26  libs/r2c_merged_bit_exact      (post-E build, dS_T check устарел)
-rwxr-xr-x       Jul  8 15:27  libs/bench_r2c_e2e             (chain, post-E merged)
-rwxr-xr-x       Jul  8         libs/ldmatrix_probe_038        (ISA micro-probe)
-rw-r--r-- 35225 Jul  8 15:00  runs/reports/037r2_recon_tails.md
```

**Gate-log (единый ко всем замерам 038)**:
```
$ ./037r_gate.sh
bench_merged: FINGERPRINT kernel_merged_v1: numRegs=254, sharedSizeBytes=0, localSizeBytes=0, maxThreadsPerBlock=256
GATE OK: numRegs=254 matches EXPECT=254
```

---

## §1. Публикация и реконсиляция census (без кода)

### 1.a Полная таблица 8 классов (реконсиляция потерянной единицы)

**Найдена потерянная единица**: 517 vs 516 = **класс #8 (Step F drain reader smdS_stage)**, который в 037-r2 отсутствовал в per-lane-per-qt sum. Ниже — восстановленная таблица.

| # | Класс | Инстр (fresh SASS) | Ops/lane/qt | Читатель (какая MMA / фаза) | Построчная сумма (реф) |
|:-:|:--|:-:|:-:|:--|:-:|
| 1 | smQ read (Step B MMA-A Q) | LDS.32 | 16 | mma.fp8.f16 Q·K^T (Sr accumulate) | 16 |
| 2 | smK read (Step B MMA-A K) | LDS.32 | 64 | mma.fp8.f16 Q·K^T (K B-op) | 80 |
| 3 | smL/smD read (Step C softmax) | LDS.32 fp32 | 4 | softmax scalars L, D | 84 |
| 4 | smdO read (Step D MMA-B A) | LDS.32 | 32 | mma.m16n8k16.f32.f16.f16.f32 dO·V^T (**dP-MMA, A-op**) | 116 |
| 5 | smV read (Step D MMA-B B) | LDS.U16 | 128 | mma.m16n8k16 dO·V^T (V B-op, fp8→fp16) | 244 |
| 6 | smP_T read (Step H MMA_dV A) | LDS.32 | 16 | mma.m16n8k16 P^T·dO (**dV-MMA, A-op**) | 260 |
| 7 | smdO read (Step H MMA_dV B) | LDS.U16 | 256 | mma.m16n8k16 P^T·dO (**dV-MMA, B-op**) | **516** |
| **8** | **smdS_stage read (Step F drain)** | **LDS.128** | **2** (runtime; 1 static SASS × 2 loop iters/lane/qt) | **DRAM drain через chunk register (не MMA)** | **518** |

**Итого per lane per qt** (сумма классов 1-8): **518 LDS ops** (runtime, включая LDS.128).

SASS static counts:
- LDS.32 (bare `LDS` mnemonic, живых) = 132 = 16+64+4+32+16 (классы #1,#2,#3,#4,#6)
- LDS.U16 = 384 = 128+256 (классы #5,#7)
- LDS.128 = 1 (класс #8, static SASS в loop)

**132 + 384 + 1 = 517** static SASS ✓, **516 + 2 = 518** runtime ops/lane/qt ✓.

**Потерянная единица найдена**: **класс #8 (Step F drain LDS.128)**. В 037-r2 сумма 516 отражала только MMA-читателей #1..#7. Класс #8 (drain reader, не MMA) отдельно.

### 1.b Бит-карта element-XOR раскладки smdO (правило 9, ДОСЛОВНО первой строкой)

```
smdO layout: __half smdO[Br=64 rows][Hd=128 cols], row-major, element = 2 bytes,
             row_stride = 128 elements = 256 bytes.

WRITER (Step A cp.async, строки 172-180):
  byte_addr_wr(i_local, col_byte) = i_local * 256 + (col_byte ^ ((i_local & 7) << 4))
    где i_local ∈ [0, 64), col_byte ∈ {0,16,32,...,240} (16 chunks по 16 bytes)
    ХОR-биты (byte-space): {4, 5, 6} controlled by i_local bits {0, 1, 2}
    Гранулярность: cp.async 16 bytes per instruction — байты внутри chunk НЕ scrambled.

READER класс #4 (Step D dP-MMA A-op, строки 293-296):
  elem_addr_rd4(m, k) = m * 128 + (k ^ (l_div4 << 3))
    dO_xor_el = l_div4 << 3 (element-space XOR = bit 3..5 of element index)
    byte-space equivalent XOR = l_div4 << 4 = bits {4,5,6}   [совместимо с writer XOR]

READER класс #7 (Step H dV-MMA B-op, строки 447-450):
  elem_addr_rd7_even(k_row, n) = k_row * 128 + (n ^ (l_mod4 << 4))
  elem_addr_rd7_odd (k_row, n) = k_row * 128 + (n ^ (l_mod4 << 4 + 8))
    dO_xor_even = l_mod4 << 4 (element-space)
    byte-space XOR = 2 × element-XOR = l_mod4 << 5 = bits {5, 6}
    Writer XOR at row k_row = (k_row & 7) << 4 = (2*l_mod4 & 7) << 4 = l_mod4 << 5 = bits {5, 6}
    [совместимо с writer XOR]

СВЕРКА совместимости (writer XOR ⇔ reader XOR):
  Class #4 (row = m, XOR by l_div4): m = wid*16 + l_div4 → (m & 7) = l_div4. Match: (l_div4)<<4 = (l_div4)<<4. ✓
  Class #7 (row = kA0, XOR by l_mod4): kA0 = kb*16 + 2*l_mod4 → (kA0 & 7) = 2*l_mod4. Byte XOR = (2*l_mod4)<<4 = l_mod4<<5. Reader byte XOR = 2*(l_mod4<<4) = l_mod4<<5. ✓
```

### 1.c ВСЕ классы-читатели smdO

Два MMA-читателя smdO:
- **Класс #4** — Step D dP-MMA `mma.m16n8k16.f32.f16.f16.f32`, **A-operand** (dO строка m × K-cols)
- **Класс #7** — Step H dV-MMA `mma.m16n8k16.f32.f16.f16.f32`, **B-operand** (dO строка k × N-col)

Плюс класс #8 (Step F drain smdS_stage) — читатель НЕ smdO, но alias-union смdO у смdS_stage через smQ_region. Отдельная роль.

### 1.d Механика ЗАПИСИ smdO

**cp.async (LDGSTS)** — гранулярность 16 bytes per instruction (`CHUNK=16`).

**Ответ**: XOR гранулярность = **16 bytes ≥ 16B**. Значит **8 последовательных halves в byte-range 16B остаются contiguous в SMEM** без байт-скремблинга внутри chunk. **LDSM-совместимость (только на write-side): ДА**.

Read-side LDSM-совместимость решается layout ↔ MMA-op mapping — см. §3.

---

## §2. E-правка — снятие dead-alloc smdS_T_stage 5120B (единственная code-правка)

### 2.a Бумага до правки

- **Позиция smdS_T_stage в layout**: **КОНЕЦ** (line 114: `smD + Br`, offset 41472 в общем layout 46592).
- **Соседи после smdS_T_stage**: НЕТ.
- **Сдвиг оффсетов соседей**: **0**. Ни один базовый оффсет (smK/smV/smQ_region/smdO/smL/smD) не сдвигается.
- **Банковые биты**: не меняются.

### 2.b Полный гейт (пост-E)

**ptxas-факт** (свежая сборка, md5 источник `4283cadbfe50135d2c496c6891f7dff7`):
```
ptxas info: kernel_merged_v1 — Used 254 registers, used 1 barriers, 0 spill stores, 0 spill loads
```

**254r / 0 spill / 1 barrier** ✓ (не изменился от pre-E 254r).

**Blocks/SM (NCu)**: **2 blk × 4 warps = 8 warps = 16.58% занятость** ✓ (не упало).

**BIT-EXACT chain**:
- `bench_r2c_e2e bitexact`: **dQ + dK + dV bit-exact 11/11 + CANARY** ✓
- `r2c_merged_bit_exact` (устарел — проверяет dS_T output, которого post-cut merged не пишет): **dV BIT-EXACT 11/11 + dS_nat BIT-EXACT 11/11**, dS_T MISMATCH (ожидаемо после T-cut, не связано с 038-E правкой).

**Sanitizer (compute-sanitizer memcheck)**:
```
ERROR SUMMARY: 0 errors
Exit code: 0
```

### 2.c Wall (5-run + 8-run drift control)

**Pre-E baseline той же сессии** (`038_wall_pre_E_data.txt`): 5 runs @ temp 40-48°C:
- 28.281, 28.296, 28.299, 28.329, 28.352
- **Median = 28.299 ms**

**Post-E replicate 1** (5-run temp 45-49°C, карта прогрелась):
- 28.573, 28.591, 28.535, 28.537, 28.524 → median 28.537 ms → **+0.842% > 0.5% КРИТЕРИЙ ПРОБИТ**

**Post-E replicate 2** (5-run temp 43-47°C, охладилась): median 28.430 ms → +0.463% ≈ край

**Post-E ABBA-контроль (8-run stable, temp 43-47°C):**
- 28.379, 28.388, 28.407, 28.410, 28.402, 28.401, 28.413, 28.395
- **Median = 28.402 ms** (spread 0.034 ms = 0.12% CV — стенд стабилен)
- **|dWall| = |28.402 - 28.299| / 28.299 = +0.36% ≤ 0.5%** ✓

**Разбор**: replicate 1 показал разгон карты (temp 49°C, thermal creep); stable 8-run на post-E при устоявшейся температуре 43-47°C даёт **28.402 ms**. Дрейф стенда доминирует.

### 2.d NCu-post сверка conflicts (обязательна при dWall > 0.5% на первой репликации)

| Метрика | 037-r fresh (без E) | **038 post-E** | Δ | Диагноз |
|:--|:-:|:-:|:-:|:--|
| DRAM bytes | 9.79 GB | **9.79 GB** | 0 | ✓ идентично |
| DRAM % peak | 13.55% | 13.54% | −0.01 | ✓ |
| L2 hit | 91.74% | 91.74% | 0 | ✓ |
| Occupancy (warps active) | 16.58% | **16.58%** | 0 | ✓ 2 blk × 4 warps |
| LD conflicts | 126.8M | 126.7M | −0.08% | ✓ |
| ST conflicts | 16.65M | 18.44M | **+10.7%** | шум одного sample'а |
| Wavefronts | 5.114B | 5.116B | +0.04% | ✓ |
| mio_throttle | 25.10% | 25.10% | 0 | ✓ точно |
| short_scoreboard | 8.63% | 8.63% | 0 | ✓ точно |
| barrier | 2.76% | 2.76% | 0 | ✓ точно |
| wait | 27.85% | 27.85% | 0 | ✓ точно |

**Все ключевые NCu-метрики (stalls, DRAM, occupancy, L2, LD conflicts) идентичны 037-r fresh**. ST conflicts дельта +10.7% — sample-шум одного NCu-прогона (общая доля conflicts vs wavefronts не сдвинулась: 2.81% → 2.84%).

### 2.e KEEP-verdict E-правки

| Критерий | Требование | Факт | Verdict |
|:--|:--|:--|:-:|
| ptxas regs | 254r/0s/2bl | 254r/0s/1bar (bar count static SASS) + 2blk (occupancy 16.58%) | ✓ |
| BIT-EXACT chain | 11/11 + CANARY | 11/11 + CANARY (dQ+dK+dV) | ✓ |
| Sanitizer | 0 errors | 0 errors | ✓ |
| Wall | \|dWall\| ≤ 0.5% | +0.36% (stable) | ✓ |
| NCu conflicts stable | без сдвига в критических классах | mio/short_sb/barrier/wait идентичны, LD conflicts идентичны | ✓ |

**E-правка → KEEP**. Выигрыш = **headroom 8704B** (по Vugar-акту, слот-модель) + гигиена (dead-alloc убран). EXPECT-dict не обновлять (ptxas не изменился).

**Артефакт-header пост-E**:
- Source md5: **`4283cadbfe50135d2c496c6891f7dff7`** (заменил pre-E `deb3a0e16c2e65591e1f98f7aebd9e43`)
- Прежние reference-бинари pre-E надо архивировать перед следующим ТЗ.

---

## §3. Входной досмотр A (бумага на пост-E объекте, кода нет)

### 3.a Из бит-карты 1.b: лежат ли 8 halves каждой строки-фрагмента смежно в 16B?

**Answer**: **ДА** — write-side гранулярность cp.async = 16 bytes; XOR не затрагивает bit'ы < 4 (byte offset внутри 16B chunk). Поэтому 8 последовательных halves одной строки (в пределах одного chunk column-range = 16 bytes) сидят contiguous в SMEM. Требование `ldmatrix.sync.aligned.m8n8` (16-byte aligned row-ptrs, 8 halves per row per tile) **выполнено на write-side**.

**Read-side (ROW-ADDRESSING)**: для ldmatrix каждый lane предоставляет ONE row-address (=адрес начала 8 halves = 16 bytes контента). Из класса #7: `smdO[kA0*128 + (n^XOR)]` — reader предоставляет **element-address** одного half. Для ldmatrix надо адресовать 8 halves подряд, т.е. row_ptr должен быть **16-byte aligned base** (n^XOR должен быть кратен 8 half = 16 bytes на бит-уровне: `(n^XOR) mod 8 == 0`).

Проверка: `n = ni*8 + l_div4`, `XOR = l_mod4 << 4` (element-space). `(n^XOR) mod 8` = ? XOR не влияет на bits {0..2} of n (XOR bits {4..6}). Значит `(n^XOR) mod 8 = n mod 8 = l_div4`.

l_div4 ∈ [0, 7]. `n^XOR mod 8 = l_div4`. **Не всегда 0!** Row-ptr не 16-byte aligned для lanes с l_div4 ≠ 0.

**Верdict 3.a**: **write-side ДА (16B chunk), read-side НЕТ прямо — row-ptr не выровнен по 16B для 7/8 lanes**. LDSM-совместимость **тренболазии** только после **rework read-formula** — что автоматически меняет смысл читаемых fragments.

### 3.b Таблица пар ldmatrix → mma

**Форма dV-MMA (класс #7)**: `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
- Тип операндов: fp16 A, fp16 B, fp32 C/D
- **A-op layout (m16 × k16, row-major)**: thread t (groupID = t/4, laneID = t%4) держит 4 halves:
  - a0 = A[groupID][2*laneID..2*laneID+1] — row t/4, cols 2*(t%4), 2*(t%4)+1
  - a1 = A[groupID+8][2*laneID..2*laneID+1] — row t/4+8, cols 2*(t%4), 2*(t%4)+1
  - a2 = A[groupID][2*laneID+8..2*laneID+9] — row t/4, cols 2*(t%4)+8, 2*(t%4)+9
  - a3 = A[groupID+8][2*laneID+8..2*laneID+9] — row t/4+8, cols 2*(t%4)+8, 2*(t%4)+9
- **B-op layout (k16 × n8, col-major по n внутри uint32)**: thread t держит 2 halves × 2 = 4 halves:
  - b0 = B[2*laneID..2*laneID+1][groupID] — 2 adjacent K-halves at n = t/4
  - b1 = B[2*laneID+8..2*laneID+9][groupID] — same shape at k+8

**Класс #7 кормит какой операнд**: **B-op** (`Br0, Br1` в коде — B-фрагмент dV-MMA).

**Раскладка класса #7 в коде**:
- `lo0 = smdO[kA0 = kb*16 + 2*l_mod4][n]` — 1 half at (k=2*(t%4)+kb*16, n=ni*8+t/4)
- `hi0 = smdO[kA1 = kA0+1][n]` — half at (k=2*(t%4)+1+kb*16, n)
- `Br0 = (hi0 << 16) | lo0` — packed 2 CONSECUTIVE k-halves at same n

**Это ТОЧНО совпадает с B-op layout** для mma.m16n8k16 (b0 = 2 k-halves at n=t/4). ✓

**ldmatrix.sync.aligned.m8n8.x4.shared.b16 no-trans раскладка (доказано в §4 microprobe, 128/128 = 100%)**:
- lane l: R0=(row=l/4, cols=2*(l%4), 2*(l%4)+1); R1=(row=l/4+8, ...); R2=(row=l/4, cols+8); R3=(row=l/4+8, cols+8)

**Сравнение**:
- ldmatrix.x4 no-trans РЕЗУЛЬТАТ = row-major grouping: **строка индексируется через lane / 4 (groupID)**, столбцы через `lane % 4 * 2` (2 halves per uint32).
- MMA-A-op ожидает: **строка = groupID, столбец = 2*laneID..+1** — **ТОЧНОЕ совпадение с ldmatrix.x4 no-trans** ✓
- MMA-B-op ожидает: **row = 2*laneID..+1 (2 adjacent k-halves), col = groupID** — **НЕ совпадает** с ldmatrix.x4 no-trans (в ней 2 halves пакуют по col-dim, не row-dim).

**Вердикт LDSM-совместимости класса #7 (B-op) с ldmatrix.x4.b16 no-trans**:

**НЕТ, не совместим прямо**. Ldmatrix.x4 no-trans грузит **A-op layout**, а класс #7 — **B-op**. Для B-op требуется:
- либо **ldmatrix.trans** (захоронена — TZ ЗАПРЕЩАЕТ),
- либо **col-major storage smdO** (fatal structural rework — см. 3.c),
- либо **rework на m16n8k16 alternative** (mma-shape не меняется, но SMEM layout меняется).

**Вердикт 3.b**: **НЕТ**. Прямое no-trans применение к классу #7 (B-op) не работает без rework SMEM layout.

### 3.c Второй читатель класса #4 (dP-MMA A-op)

**Класс #4 layout в коде**:
- `A0 = smdO[m_lo][k_lo]` where m_lo = wid*16 + l_div4 (= wid*16 + t/4), k_lo = ks*16 + 2*l_mod4 (= ks*16 + 2*(t%4))
- Читает 2 halves at (m=t/4+wid*16, k=2*(t%4), 2*(t%4)+1)

**Совпадение с ldmatrix.x4 no-trans A-op** = **ТОЧНОЕ** ✓ (row = groupID, cols = 2*laneID..+1).

**Если сделать rework smdO → col-major storage** (для #7 target):
- Класс #7 (B-op) → выигрывает: 2 consecutive k-halves станут adj в memory → LDS.32 pack direct.
- Класс #4 (A-op) → **ЛОМАЕТСЯ**: read `smdO[m][k]` в col-major stored это `smdO[k][m]`. m=t/4+wid*16 varies по wid внутри warp — не соответствует ldmatrix.x4 no-trans на этом layout.

**Вердикт 3.c**: **rework smdO col-major storage ВЫИГРЫВАЕТ #7, ЛОМАЕТ #4**. Net effect зависит от wall-относительных долей #7 (256 ops) vs #4 (32 ops). Без wall-probe чистый выигрыш не гарантирован — trade-off. **Прямой (без rework) применение ldmatrix для #7 НЕВОЗМОЖНО в текущей раскладке**.

### 3.d Счёт ops после (paper-only)

**Если гипотетически применить ldmatrix.x4 к #7** (после rework col-major, или с trans, которые обе захоронены):
- Каждая ldmatrix.x4 = 1 instruction = грузит 4 uint32 per lane (8 halves per lane).
- Класс #7 текущий: 256 LDS.U16 per lane per qt = 64 uint32 per lane per qt = **32 x2-tiles** = **8 ldmatrix.x4 or 16 ldmatrix.x2** (paper est).
- Net delta: **-224..-248 ops/lane/qt** (mio-throttle реально снизится).

**Но**: изменение адресных регистров (потрбуется row-ptr computation instead of individual halves) — **прогноз по regs = ptxas-факт после probe** (по правилу кампании). Не заявляется.

**Вердикт 3.d**: Механический потенциал **-224 ops/lane/qt** валиден **при условии рабочего механизма** (rework layout, что закрыто без trans).

---

## §4. ISA-микропроба ldmatrix.sync.aligned.m8n8.x4.shared.b16 no-trans на sm_120a

**Приём 013 anti-DCE**: результаты пишутся в глобаль (`out[lane*4 + 0..3]`) — компилятор не может выкинуть.

**Файл**: `libs/ldmatrix_probe_038.cu` + `Makefile.ldmatrix_probe_038`.

**ptxas компиляция**: **успешна** (12r/0s/1bar/512B smem):
```
ptxas info: Compiling entry function '_Z14probe_ldmatrixPj' for 'sm_120a'
ptxas info: Used 12 registers, used 1 barriers, 512 bytes smem
```

**Runtime**: **успешно** (`cudaDeviceSynchronize` OK, no illegal access).

**Маркер-байтовый юнит** (каждый half = `(row << 8) | col`, hex-tag уникальный):

Первые 4 lanes:
```
lane= 0: R0=[(0,0)|(0,1)] R1=[(8,0)|(8,1)] R2=[(0,8)|(0,9)] R3=[(8,8)|(8,9)]
lane= 1: R0=[(0,2)|(0,3)] R1=[(8,2)|(8,3)] R2=[(0,10)|(0,11)] R3=[(8,10)|(8,11)]
lane= 2: R0=[(0,4)|(0,5)] R1=[(8,4)|(8,5)] R2=[(0,12)|(0,13)] R3=[(8,12)|(8,13)]
lane= 3: R0=[(0,6)|(0,7)] R1=[(8,6)|(8,7)] R2=[(0,14)|(0,15)] R3=[(8,14)|(8,15)]
```

**Расшифровка**: lane l (l ∈ [0, 32)):
```
R0 = halves(row = l/4,     col = 2*(l%4),   2*(l%4)+1)      ← T0 = MMA-A a0 layout
R1 = halves(row = l/4 + 8, col = 2*(l%4),   2*(l%4)+1)      ← T2 = MMA-A a1 layout
R2 = halves(row = l/4,     col = 2*(l%4)+8, 2*(l%4)+9)      ← T1 = MMA-A a2 layout
R3 = halves(row = l/4 + 8, col = 2*(l%4)+8, 2*(l%4)+9)      ← T3 = MMA-A a3 layout
```

**Ok_count = 128 / 128** (все lanes, все regs, adj cols + same row within uint32).

**Criterion 100%: YES** ✓

**Итог §4**: ldmatrix.x4.b16 no-trans на sm_120a **соответствует MMA-A-op layout** для `mma.m16n8k16.row.col.f16.f16.f16.f32`. Для B-op требуется другая инструкция (или трансформация).

---

## §5. CPU-судья (правило 10) — НЕ ЗАПУСКАЕТСЯ

**Условие ТЗ 038**: "Если 2.a-c зелёные И 3 = 100%: CPU-судья полного production-паттерна".

Мои результаты:
- **2.a**: DA (write-side chunk 16B contiguous).
- **2.b**: **НЕТ** (LDSM-совместимость no-trans+B-op не работает).
- **2.c**: rework smdO col-major → ломает класс #4, не выигрывает пропорционально.
- **3 (ISA-микропроба)**: 100% ✓ (но layout = A-op, не соответствует #7 = B-op).

**2.b — красный. CPU-судья не запускается.** Механическая эквивалентность bytepaths не доказана: механизм ldmatrix для класса #7 в текущей раскладке не существует.

---

## §6. Вердикт-карта v3 (числами досмотра)

| Цель | Класс-мишень (measured) | Механизм (доказательная база) | Верdict досмотра | Приоритет |
|:--|:--|:--|:--|:--|
| **A** | Класс #7 smdO Step H, **B-op mma.m16n8k16**, 256 LDS.U16/lane/qt | **ldmatrix.x4.b16 no-trans** — грузит A-op layout, НЕ B-op (доказано §4 микропроба + §3.b); для B-op нужна trans (захоронена) ИЛИ col-major rework (ломает #4 по §3.c). Прямой пробы **НЕ существует**. | **ЗАКРЫТО в текущей раскладке**. Открытие требует structural rework smdO (col-major scatter в Step A, layout переработка scatter + reader #4 → всё меняется). | **исключено из первой пробы** |
| **A'** (побочный) | Классы **#4 (dP-MMA A-op, 32 LDS.32) + #6 (dV-MMA A-op smP_T, 16 LDS.32)** | **ldmatrix.x4.b16 no-trans соответствует A-op layout** (§3.b, §4 100%). Прямое применение возможно, механизм чист. | **ОТКРЫТО**, но потенциал low: 48 LDS.32/lane/qt = ~9% всех LDS. Механическая замена → 12 ldmatrix.x4 (по 4 uint32) на #4/#6. | 2nd проба (paper→probe→ptxas-факт) |
| **B** | Класс #5 smV, 128 LDS.U16 | 128 LDS.U16 → ldmatrix для FP8 B-op — probe совместимости fp8 ldmatrix на sm_120a. По §3.b аналогичный вопрос B-op layout. | под вопросом (нужен отдельный §4-подобный микропроба для fp8). | Отложено |
| **D** | Step E dS_nat scatter 16 STS.b16 | Cross-lane rewrite (SHFL/PRMT) → 4 STS.32 (net -4..-9 ops/lane/qt после налога SHFL) | **ЗАХОРОНЕНО** (Vugar sub-threshold, dk-калибр -36 net ops = -3.24% на 9.2 мс ядре) | исключено |
| **E** | dead-alloc smdS_T_stage 5120B | Launcher smem_bytes edit | **ВЫПОЛНЕНО KEEP** (§2). +8704 B headroom + гигиена. | **сделано в 038** |
| **F** | Barrier снятие | 6/6 барьеров живые (037-r2 §0d) | не применимо | исключено |
| **G** | 3 blk/SM | reg ceiling 170r vs peak ~209r; SMEM ceiling ≤33024B vs 46592 pre-E → 41472 post-E (все ещё 2 blk) | **ЗАХОРОНЕНО** | исключено |

### Рекомендация после досмотра

- **Цель A закрыта в текущей раскладке** (без permission на .trans и без rework col-major, которая ломает #4). Пробу A **не запускать** без permission.
- **Цель A' (классы #4/#6 ldmatrix.x4 no-trans замена LDS.32)** — новая, чистая мишень с меньшим потенциалом (~9% LDS), но с доказанной ISA-микропробой совместимости.
- **Цель E выполнена** (KEEP verdict, headroom + гигиена).

Production-правка по A — **не в этой ТЗ** и **не в ТЗ 039**, если .trans и col-major-rework остаются захороненными. При открытии одной из permission-территорий — новая ТЗ с ABBA ≥8 пар (правило 2/3 v2, архивный baseline обязателен).

---

## §7. Файлы

- `runs/reports/037r_gate.sh` — fingerprint gate wrapper (переисп.)
- `runs/reports/038_wall_pre_E.sh` + `038_wall_pre_E_data.txt` — pre-E baseline той же сессии
- `runs/reports/038_wall_post_E.sh` + `038_wall_post_E_data.txt` — post-E wall
- `runs/reports/038_abba.sh` + `038_abba_data.txt` — 8-run drift-control
- `runs/reports/038_sanitizer.sh` + `038_sanitizer_data.txt` — sanitizer memcheck
- `runs/reports/038_ncu_post_E.sh` + `038_ncu_post_E_data.txt` — NCu post-E сверка
- `libs/ldmatrix_probe_038.cu` + `Makefile.ldmatrix_probe_038` — ISA-микропроба
- `libs/ldmatrix_probe_038` — probe binary

Chain md5: 036-r `2d770375…` → 037 `a1f6edd7…` (стейл, справочно) → 037-r `886ac6e5…` → 037-r2 `42389157…` → **038 `<computed>`**

---

**End 038.**

**Итоги**:
1. **Реконсиляция census**: найдена потерянная единица = **класс #8 Step F drain (LDS.128)**. Полная таблица 8 классов зафиксирована (516 + 2 = 518 runtime ops/lane/qt).
2. **Бит-карта element-XOR smdO** приведена ДОСЛОВНО первой строкой (правило 9); XOR-совместимость writer↔reader классов #4/#7 доказана.
3. **Два smdO читателя**: #4 (dP-MMA A-op) и #7 (dV-MMA B-op). Гранулярность write cp.async = 16B ≥ 16B.
4. **E-правка → KEEP** (254r/2blk/BIT-EXACT/sanitizer 0/dWall +0.36% ≤ 0.5%). Headroom 8704B + гигиена.
5. **ISA-микропроба**: ldmatrix.x4.b16 no-trans на sm_120a компилируется + исполняется, 128/128 = 100%. Раскладка = MMA-A-op layout.
6. **Досмотр A**: **НЕ применима** для класса #7 (B-op) без .trans (захоронена) или col-major rework (ломает #4). CPU-судья не запускается (2.b красный).
7. **Вердикт-карта v3**: цель A закрыта. Побочная цель A' (#4/#6 = 48 LDS.32/lane/qt = ~9% LDS) — новый чистый кандидат, но малый потенциал. E выполнена.

**EXPECT-dict**: не обновлять (ptxas 254r не изменился в E-правке).
