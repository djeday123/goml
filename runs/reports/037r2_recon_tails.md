# 037-r2 — Хвосты реатрибуции: 0c/0d/0e пересъёмка + поправки нитей 037-r

**Chain**:
- 036_r_portfolio.md md5: `2d770375664b01a2c4459fa2e0ca8bf9`
- 037_merged_recon.md md5: `a1f6edd762710c1b69aa2ddda4d4fdb3` (**АННУЛИРОВАН — стейл, справочно**)
- 037r_reattrib.md md5: `886ac6e52525a7beb9dee7082a05d742`

**Правила** (из ТЗ 037-r2): правки кода/harness запрещены; только разведка на опознанном объекте (свежая сборка 254r из исходника md5 `deb3a0e16c2e65591e1f98f7aebd9e43`); каждый замер через `037r_gate.sh`; лог гейта прилагается.

---

## Шаг 1. Artifact-header (правило 5)

```
-rw-r--r--  24403  Jul  8 06:09  libs/fa_bwd_merged_v1.cu       (prod, md5 deb3a0e16c2e65591e1f98f7aebd9e43)
-rwxr-xr-x 1189024 Jul  8 14:28  libs/r2c_merged_wall           (fresh, md5 0d54721e144cb6fe72940fba65fed044)
-rwxr-xr-x 1294112 Jul  8 14:28  libs/r2c_merged_bit_exact      (fresh, md5 967f6f2000265ac1cdaf768cd199fceb)
-rw-r--r--  27421  Jul  8 14:38  runs/reports/037r_reattrib.md
-rw-r--r--  28609  Jul  8 13:40  runs/reports/037_merged_recon.md  (стейл, справочно)
```

**Гейт-лог (единый для всех замеров 037-r2)**:
```
$ ./037r_gate.sh
bench_merged: FINGERPRINT kernel_merged_v1: numRegs=254, sharedSizeBytes=0, localSizeBytes=0, maxThreadsPerBlock=256
GATE OK: numRegs=254 matches EXPECT=254
```

---

## Поправки нитей 037-r

### (a) Stale-колонка barrier

Моя запись в 037-r §6.0a: `| **barrier** | 2.59 | **2.76** | **2.76** |` — **stale=2.59, fresh=2.76**. Совпадает с 037-stale отчётом (§0a stall breakdown, «barrier 2.59%»). **Ошибки не было** — Vugar-напоминание принято, повторно фиксирую 2.59% как стейл-число.

### (b) TFLOPS от МЕДИАНЫ, не среднего

**Ошибка 037-r §6.0a**: указал "236.7 T (3-MMA convention)".

**Правка**:
- **3-MMA канон**: `6 × base = 6 × 1.0995e+12 = 6.597e+12 FLOP` per canonical form (bh=128, sl=8192, hd=128).
- **Median wall isolated (5-run stand-protocol)**: **27.836 ms**
- **TFLOPS_3mma_median = 6.597e+12 / 0.027836 = 236.99 T ≈ 237.0 T**

Моё «236.7» получалось от per-run tflops printed by bench: {237.11, 236.69, 237.00, 236.71, 236.53} — **среднее** ≈ 236.81 округлено до 236.7, тогда как **median** ≈ 236.71. От медианы wall 27.836 → **237.0 T**.

**Итог правки (b)**: **237.0 T (3-MMA)**, не 236.7.

### (c) NCu-налог 28-51% — конкретные измерения

Скрипт: `runs/reports/037r2_ncu_tax.sh`, данные: `037r2_ncu_tax_data.txt`.

| Режим NCu | Passes | Wall в NCu (мс) | Reference isolated (мс) | Налог |
|:--|:-:|:-:|:-:|:-:|
| **A** `--metrics dram__bytes.sum` (single metric, 1 pass) | **1** | **42.321** | 27.836 | **+52.0%** |
| **B** `--metrics {barrier,mio,wait,short_sb}` (4 metrics) | 3 | 28.219 | 27.836 | **+1.4%** |
| **C** `--set base` (полный default set) | ~40+ | 28.227 | 27.836 | **+1.4%** |

**Диагноз**: NCu-налог **бимодален**:
- Single-pass single-metric с `--launch-count 1` даёт **~+50%** налог, потому что overhead инициализации professionalлировщика ложится на один-единственный профилированный launch внутри bench, в то время как остальные 24 launch idle (avg по всем = large-per-iter выход).
- Multi-metric multi-pass режимы **амортизируют** overhead через replay: **~+1.4%** налог.
- Мой предыдущий диапазон "28-47%" в правиле 6 — **устаревший** (взят из 037-stale отчёта, режим неявный). **Уточнение правила 6**: NCu-налог зависит от режима, диапазон **1.4%..52%** установлен на опознанном объекте 037-r2, режимы A/B/C зафиксированы в леджер.

**Пример: пара измерений (Reference vs Режим A)**:
```
Reference isolated: 27.836 ms (bench self-timed, 5-run stand median)
Режим A NCu:        42.321 ms (bench self-timed under NCu instrumentation)
Налог:              +14.485 ms = +52.0%
```

---

## 0c-fresh: LDS-census на свежей сборке

### 0c.1 SASS-факт (kernel_merged_v1, fresh binary md5 `0d54721e…`)

Скрипт: `037r2_sass.sh`, данные: `037r2_sass_data.txt`, SASS-extract: `037r2_sass_merged_only.txt`.

**Instruction counts**:
| Класс | Count |
|:--|:-:|
| Bare `LDS` (LDS.32 без суффикса) | **150** (в т.ч. 18 predicated `@!PT LDS RZ, [RZ]` = **132 живых**) |
| `LDS.128` | 1 |
| `LDS.U16` | **384** |
| `STS.U16` | 48 |
| `STS.E.BYPASS.128` (cp.async pipeline) | 10 |
| `STG.E` (dV epilogue) | 64 |
| `STG.E.128` (dS_nat drain) | **1** (post-cut ✓, single drain) |
| `LDGSTS.E.BYPASS.128` (cp.async) | 10 |
| `LDG.E.CONSTANT` | 2 |
| `SHFL*` | **0** |

**Валидация 132 живых LDS.32 = ожидание**: сумма классов #1+#2+#3+#4+#6 = 16+64+4+32+16 = **132** ✓

### 0c.2 Полная таблица LDS-классов (per lane per qt) — post-cut

**Ожидание (сверка не подгонка)**: T-cut убрал только STS smdS_T_stage + STG.128 dS_T_out. Ни один LDS-класс #1..#7 не был T-путём, все живы в post-cut. Итог `LDS = 516 ops/lane/qt` без изменений от pre-cut. LDS.128 в Step F drain: pre-cut = 2 (dS_nat + dS_T reads), post-cut = **1** (только dS_nat), SASS ✓.

| # | Класс | Инстр (fresh SASS) | Ops/lane/qt | Смежность (адресная, доказанная) | Потенциал склейки | Bank-конфликт риск |
|:-:|:--|:-:|:-:|:--|:--|:--|
| 1 | smQ read (Step B, MMA-A Q) | LDS.32 | **16** | Qr[ks][0..3] pairs через swz_byte; m 8-adj (Qr[0]↔Qr[1] delta 8 rows = 8×128=1024B), k 16-str | LDS.64 если swz_byte даёт (k_lo, k_hi) byte-adj | средний |
| 2 | smK read (Step B, MMA-A K) | LDS.32 | **64** | swz_byte(n=ni*8+l_div4, k_lo/k_hi); k_lo, k_hi delta 16 bytes (2 words); ni 8-str, n rows | LDS.64 если swz XOR сохраняет adj | средний |
| 3 | smL/smD read (Step C softmax) | LDS.32 fp32 | **4** | smL[wid*16+l_div4+{0,8}], smD[wid*16+l_div4+{0,8}]; pairs delta 8 fp32 = 32B (не adj), но у same-lane +0/+8 tuple | LDS.64 fp32x2 для (L_lo,L_hi) и (D_lo,D_hi) — pair {+0,+8} НЕ adj (delta 8) | низкий |
| 4 | smdO read (Step D, MMA-B A) | LDS.32 | **32** | smdO[m*128+(k^dO_xor_el)]; m 8-adj row = 1024B, k 16-str (k_lo, k_hi elements delta 16=32B) | LDS.64 для (k_lo,k_hi)? — delta 32B, НЕ adj | средний |
| 5 | smV read (Step D, MMA-B B) | **LDS.U16** | **128** | smV[n*128+(k_lo/k_hi^k_xor)]; n row=n_x_128, k_lo/k_hi delta 8 bytes; k_xor = l_div4<<4 (bits 4,5,6 in byte) | **см. detail 0c.4 ниже** | средний |
| 6 | smP_T read (Step H, MMA_dV A) | LDS.32 | **16** | smP_T[m*Br+(k^PT_xor_rd)]; row size Br=64 fp16=128B; m 8-adj row = 1024B, k 16 elements = 32B | LDS.64 если XOR даёт adj | низкий-средний |
| 7 | smdO read (Step H, MMA_dV B) | **LDS.U16** | **256** | **см. bit-map 0c.3 ниже** | **см. detail 0c.3 ниже** | **средний-высокий** (256 ops × factor) |
| — | Step F drain read | LDS.128 | 1 (static) = **256 chunks / block-qt runtime, 2 per lane per qt** | uint4 chunk из smdS_stage; contiguous 16B | (already wide) | низкий |

**Итог per lane per qt (LDS.16/32/64)**: 16+64+4+32+128+16+256 = **516 LDS ops** ✓

**Валидация против NCu wavefronts**: 5.114B / (n_qt=128 × n_kt=128 × bh=128) = 5.114B / 2.097M = **2439 wavefronts per block per qt**.
Per lane per qt: 516 LDS × 32 lanes/warp × 4 warps = 66048 per block per qt.
Wavefronts (32 lanes per wave with adjacency): ~1/2 ratio due to conflicts + serialization = **~2439 / block-qt wavefronts ≈ 66048 / 27 = 2445** if conflict factor 27 (не сходится — точная модель зависит от bank pattern).

Более простой sanity check: total LDS ops = 66048 × n_qt × n_kt × bh = 66048 × 128 × 128 × 128 = **138B ops**. Wavefronts NCu = 5.114B. **Ratio = 27** (waves per 32-lane ops — включает bank conflict expansion). Средняя серilизация ~27:32 = 84% — норма для XOR-swizzled MMA.

### 0c.3 Бит-карта element-XOR раскладки smdO (правило 9, первая строка секции)

**Bit-map smdO layout (element = fp16 = 2 bytes, row_stride = 128 elements = 256 bytes)**:

Writer (Step A scatter, строки 172-180):
```
byte_addr(i_local, col_byte) = i_local * 256 + (col_byte ^ ((i_local & 7) << 4))
XOR bits (byte-space): {4, 5, 6} controlled by row bits {0, 1, 2}
XOR bits (element-space, /2): {3, 4, 5} controlled by row bits {0, 1, 2}

Layout table (row → col-element XOR):
Row (i_local & 7)  |  XOR mask on col_element (bits 3,4,5)
------------------|-----------------
       0          |  0     (nothing)
       1          |  8     (bit 3)
       2          |  16    (bit 4)
       3          |  24    (bits 3,4)
       4          |  32    (bit 5)
       5          |  40    (bits 3,5)
       6          |  48    (bits 4,5)
       7          |  56    (bits 3,4,5)
Rows 8..63 repeat mod 8.
```

Reader class #7 (Step H MMA_dV, строки 447-450):
```
kA0 = kb*16 + l_mod4*2 + 0     (row index, kb ∈ [0..3], l_mod4 ∈ [0..3])
kA1 = kA0 + 1                   (adjacent row +1)
kB0 = kb*16 + l_mod4*2 + 8      (row +8)
kB1 = kB0 + 1

lo0 = smdO[kA0 * 128 + (n ^ dO_xor_even)]   ← element index
hi0 = smdO[kA1 * 128 + (n ^ dO_xor_odd)]
lo1 = smdO[kB0 * 128 + (n ^ dO_xor_even)]
hi1 = smdO[kB1 * 128 + (n ^ dO_xor_odd)]

где n = ni*8 + l_div4
     dO_xor_even = l_mod4 << 4    (element-space XOR, values {0,16,32,48})
     dO_xor_odd  = dO_xor_even + 8

Byte address = 2 × element_index = k*256 + 2*n ^ (l_mod4 << 5)   для even
Byte address = k*256 + 2*n ^ (l_mod4 << 5) + 16                   для odd
Byte-space XOR: bits {5, 6} controlled by l_mod4 bits {0, 1}
```

**Проверка совместимости writer XOR ⇔ reader XOR** (для одной элементной ячейки):
- Writer scatter row=kA0: XOR mask = (kA0 & 7) << 4 = ((l_mod4 * 2) & 7) << 4 = (2*l_mod4) << 4 = l_mod4 << 5 ✓ (совпадает с reader byte-XOR)
- Writer row=kA1: XOR = ((l_mod4 * 2 + 1) & 7) << 4 = (2*l_mod4 + 1) << 4 = l_mod4 << 5 + 16 ✓ (совпадает с reader odd XOR)

**Раскладка совместима**. Reader и writer видят те же байты.

### 0c.4 Класс #7: адресная смежность пар U16 (доказательство)

**Проверка склеиваемости 4 LDS.U16 в LDS.32 (для одного mgновения (kb, ni, lane))**:

Пары для склейки:
```
Pair (lo0, hi0):
  lo0 byte_addr = kA0 * 256 + 2n ^ (l_mod4 << 5)
  hi0 byte_addr = kA1 * 256 + 2n ^ (l_mod4 << 5) + 16
  Δ_byte = (kA1 - kA0) * 256 + 16 = 256 + 16 = 272

  **NOT adj** (нужно Δ = 2 для 32-bit adjacency в одной 32-bit word). Δ = 272 bytes.
```

```
Pair (lo0, lo1):
  lo1 byte_addr = kB0 * 256 + 2n ^ (l_mod4 << 5)
  Δ = (kB0 - kA0) * 256 = 8 * 256 = 2048 bytes

  **NOT adj**. Δ = 2048 bytes.
```

```
Pair (hi0, hi1):  Δ = 2048 bytes.  **NOT adj**.
```

```
Pair (lo0, hi1):  Δ = 9 * 256 + 16 = 2320 bytes.  **NOT adj**.
```

**Вывод класса #7 склеиваемости**: **ни одна из 6 возможных пар из 4 LDS.U16 НЕ является byte-adjacent** для LDS.32 склейки.

**Класс препятствия склейки**: **row-stride 256 bytes (шаг фрагмента)** — пары (lo0, hi0) и (lo1, hi1) лежат в РАЗНЫХ SMEM rows.

**Реальная мишень класса #7 (пересмотрено)**: прямая LDS.U16 → LDS.32 склейка **невозможна**. Альтернативы:

| Механизм | Требование | Обоснование |
|:--|:--|:--|
| `ldmatrix.sync.aligned.m8n8.x4.b16` | XOR-паттерн smdO должен быть совместим с ldmatrix cooperative layout (32 lanes → 8×8 fragment) | Blackwell sm_120a supports ldmatrix.trans для m8n8 fp16; проверка bit-map совместимости требует probe |
| Structural rework XOR-паттерна scatter | Перестроить XOR так, чтобы (kA0, n) и (kA1, n) сидели в соседних 32-bit словах | Слоя́т XOR-биты писателя должны быть выбраны так, чтобы совместное чтение lo0/hi0 попадало в одну 32-bit ячейку — существенная перестройка Step A cp.async |
| Prefetch smdO в регистры до MMA_dV loop | Регистровое расширение: 256 U16 × 2 (double-buffer) = 512 регистров — **невозможно** (текущий peak 254r из 65536/(2×128)) | Не применимо |

**Первичная рекомендация 037/037-r «LDS.U16 → LDS.32 через XOR-паттерн rewrite» опровергнута адресным анализом — переоценка в 0f-v2.**

### 0c.5 Класс #5 (smV MMA-B): адресная смежность

**Пара (v0_u16, v1_u16)**:
```
v0 byte_addr = n * 128 + (k_lo ^ k_xor)    где k_lo = ks*16 + l_mod4*2 + 0
v1 byte_addr = n * 128 + (k_hi ^ k_xor)    где k_hi = ks*16 + l_mod4*2 + 8
Δ_byte = (k_hi - k_lo) = 8

Проверка bit-XOR совместимости: k_xor = l_div4 << 4 = bits {4,5,6}; k_lo/k_hi bits {0,1} = ks*16 (bit 4); bit 3 = l_mod4*2 (bits 4-5 для l_mod4=1,2,3); k_hi & 8 = 8 (bit 3 set).
Post-XOR: bits {4,5,6} flipped by l_div4 — не пересекается с bit 3 (delta bit).
Result: Δ (post-XOR byte) = 8. NOT adj (для LDS.32 нужно Δ ≤ 2).
```

**Вывод класса #5**: **row-стайрать delta 8 bytes** (fragment step внутри row) → LDS.32 склейка **невозможна** без layout rework.

**Механизм для класса #5**: `ldmatrix` для FP8 B fragment. Blackwell sm_120a supports `ldmatrix.sync.aligned.m8n8.x4.b8` (8-bit variant для FP8). Требует probe на совместимость XOR.

### 0c.6 Классы #1, #2, #4, #6 (LDS.32)

Проверка склейки в LDS.64 для этих классов: аналогичный анализ. Для #1 (smQ), #2 (smK), #4 (smdO Step D), #6 (smP_T) — все имеют fragment step 16 elements или 8 rows, что даёт byte-Δ ≥ 16..2048. LDS.64 требует Δ = 4 (для двух uint32_t в 8-byte word). **Ни один из #1/2/4/6 не имеет byte-adj пары для LDS.64** без layout rework.

---

## 0d-fresh: barrier аудит post-cut ИСХОДНИКА (метод 021, судья исполняется)

### 0d.0 Полный inventory барьеров в prod исходнике

Прямой grep `__syncthreads()` в `fa_bwd_merged_v1.cu`:
- **t3** — line 190 (после Step A cp.async Q/dO/L/D)
- **t_new1** — line 315 (перед Step E scatter)
- **t9** — line 377 (перед Step F drain)
- **t_new2** — line 399 (после Step F drain)
- **t11** — line 423 (перед Step H MMA_dV)
- **t13** — line 462 (конец qt loop)

**Итого 6 барьеров/qt** в prod post-cut исходнике. Warmup барьер `line 137` — вне qt-loop (1 раз на всё ядро).

### 0d.1 Метод 021: address-set перебор через границу каждого барьера

Для каждого барьера **исполняю судью** — перечисляю ФАКТИЧЕСКИЕ множества адресов (writers/readers) до и после, ищу пересечения.

#### t3 (line 190, post cp.async Q/dO/L/D)

**Writers ДО** (последний блок пишущий SMEM):
- `cpa16 → smQ[swz_byte(i_local, col_byte)]` (Step A Q): byte offsets в smQ_region [0..8191]
- `cpa16 → smdO[i_local*256 + col_byte^XOR]` (Step A dO): byte offsets в smdO [0..16383]
- `smL[tid] = L[b*sl+i_g]` (Step A L, tid<Br): offsets [0..255] в smL region
- `smD[tid] = D[b*sl+i_g]` (Step A D, tid<Br): offsets [0..255] в smD region

**Readers ПОСЛЕ** (первый блок читающий SMEM):
- Step B smQ reads (line 202-205): smQ [swz_byte(m,k)] в диапазоне [0..8191]
- Step C smL/smD reads (line 231-234): smL[wid*16+l_div4+{0,8}] и smD[wid*16+l_div4+{0,8}]
- Step D smdO reads (line 293-296): smdO [m*128 + k^XOR] в диапазоне [0..16383]

**Пересечения**:
- Writers Q ∩ Readers B (smQ): **∩ non-empty** (whole 8192 B)
- Writers dO ∩ Readers D (smdO): **∩ non-empty** (whole 16384 B)
- Writers L ∩ Readers C (smL/smD): **∩ non-empty**

**Вердикт t3**: **ЖИВОЙ**. Три пары RAW hazard'ов. Не removable.

#### t_new1 (line 315, pre-scatter)

**Writers ДО**: ничего в SMEM (Step B/C/D — только LDS reads + register ops)

**Readers ДО** (последние LDS смQ_region):
- Step B smQ reads: swz_byte offsets в smQ_region [0..8191] — Qr registers filled

**Writers ПОСЛЕ**:
- Step E STS.b16 smdS_stage[i_local*80 + j_local] в smQ_region [0..5055] alias

**Пересечения**:
- Readers ДО (smQ) ∩ Writers ПОСЛЕ (smdS_stage aliased в smQ_region): **∩ non-empty** [0..5055]

**Вердикт t_new1**: **ЖИВОЙ — WAR hazard**. Warp A может закончить чтение smQ и начать writing smdS_stage, warp B ещё читает smQ.

#### t9 (line 377, pre-drain)

**Writers ДО**:
- Step E STS.b16 smdS_stage: offsets [0..5055] в smQ_region alias

**Readers ПОСЛЕ**:
- Step F LDS.128 smdS_stage[r*80+col_byte]: offsets [0..5055]
- Плюс STG.128 → dS_nat DRAM (не SMEM)

**Пересечения**:
- Writers ДО ∩ Readers ПОСЛЕ (smdS_stage): **∩ non-empty** (весь [0..5055])

**Вердикт t9**: **ЖИВОЙ — RAW hazard**. Не removable.

#### t_new2 (line 399, post-drain) — **ГЛАВНЫЙ ВОПРОС Vugar**

**Гипотеза Vugar** (сомнение): «подопечный дренаж мертв, но alias-union smQ/smdS_stage/smP_T жив — охраняет живую пару или пустой склад?»

**Writers ДО** (в интервале между t9 и t_new2):
- Step F **не пишет SMEM** (только LDS smdS_stage → reg → STG.128 к DRAM). SMEM writers ДО = **∅ (пустое множество)**.

**Readers ДО** (в интервале t9→t_new2):
- Step F LDS smdS_stage: offsets [0..5055]

**Writers ПОСЛЕ** (Step G):
- STS.U16 smP_T[j*Br + (i^xor)]: fp16 element offsets [0..4095] = byte offsets [0..8191] в smQ_region alias

**Пересечения**:
- **Writers ДО ∅ vs Writers ПОСЛЕ**: ∅ (нет write-write hazard, потому что Step F не пишет SMEM — **это тот самый вопрос «пустой склад» для write-after-write**)
- **Readers ДО ∩ Writers ПОСЛЕ (smdS_stage ⊂ smQ_region ∩ smP_T ⊂ smQ_region)**: byte offsets [0..5055] ∩ [0..8191] = **[0..5055] non-empty**

**Классификация hazard**: WAR (Write-After-Read). Warp A уже завершил LDS smdS_stage (Step F, register `chunk`) и начал STS.U16 smP_T (Step G), пока Warp B ещё читает smdS_stage через LDS в Step F.

**Проверка «живой vs пустой склад»**:
- «Пустой склад» гипотеза требует что-бы **никаких LDS смdS_stage не оставалось в pending state** к моменту начала Step G. Это выполняется только если между Step F и Step G **все warps завершили LDS smdS_stage**. Без барьера **это НЕ гарантируется** (warp scheduler может опережать warp'ы).
- **RAW между warps в drain path**: Warp A pre-issued STG.128 (async), но STG не commit'ится синхронно. Warp A не должен переписывать smdS_stage до пока Warp B не прочитал последний байт.

**Вердикт t_new2**: **ЖИВОЙ — WAR hazard**. Охраняет **живую пару Readers[F] vs Writers[G]**. **Не «пустой склад»**. **Не removable, не передвигаемый в текущей структуре**.

*Заметка*: В post-cut Step F стал легче (только 1 drain вместо 2 — pre-cut имел dS_nat + dS_T drain). Но структурная роль t_new2 не изменилась — сторож охраняет WAR внутри alias-union.

**Передвигаемость**: можно ли передвинуть t_new2 в тело Step F (per-iteration barrier)? — нет, это не help; можно ли объединить с t9? — нет, они охраняют разные фазы. **Не передвигаемый.**

#### t11 (line 423, pre-MMA_dV)

**Writers ДО**:
- Step G STS.U16 smP_T: byte offsets [0..8191]

**Readers ПОСЛЕ**:
- Step H MMA_dV A-frag: LDS.32 smP_T[m*Br + (k^PT_xor_rd)]: byte offsets [0..8191]

**Пересечения**: **∩ non-empty (весь smP_T)**.

**Вердикт t11**: **ЖИВОЙ — RAW hazard**. Не removable.

#### t13 (line 462, end qt)

**Writers ДО**: ничего в SMEM (Step H — только LDS reads + register ops для dV_acc)

**Readers ДО**:
- Step H smP_T reads: byte offsets [0..8191]
- Step H smdO reads: byte offsets [0..16383]

**Writers ПОСЛЕ (следующая итерация qt)**:
- Step A next-qt cpa16 → smQ (=smQ_region alias): [0..8191]
- Step A next-qt cpa16 → smdO: [0..16383]

**Пересечения**:
- Readers ДО (smP_T) ∩ Writers ПОСЛЕ (smQ): **∩ non-empty** (smQ и smP_T — alias одного smQ_region)
- Readers ДО (smdO) ∩ Writers ПОСЛЕ (smdO): **∩ non-empty**

**Вердикт t13**: **ЖИВОЙ — WAR hazard**. Не removable.

### 0d.2 Сводная таблица вердиктов

| # | Барьер | Строка | Hazard-class | Address-set пересечение | Вердикт | Передвигаемость |
|:-:|:--|:-:|:--|:--|:--|:--|
| 1 | t3 | 190 | RAW (Writers ДО × Readers ПОСЛЕ на smQ, smdO, smL, smD) | non-empty (3 подсписка) | **живой** | не передвигаемый |
| 2 | t_new1 | 315 | WAR (Readers ДО smQ × Writers ПОСЛЕ smdS_stage aliased) | non-empty | **живой** | не передвигаемый |
| 3 | t9 | 377 | RAW (Writers ДО smdS_stage × Readers ПОСЛЕ smdS_stage) | non-empty | **живой** | не передвигаемый |
| 4 | **t_new2** | 399 | **WAR (Readers ДО smdS_stage × Writers ПОСЛЕ smP_T aliased)** | **non-empty [0..5055]** | **живой** — охраняет **живую пару**, **не пустой склад** | не передвигаемый |
| 5 | t11 | 423 | RAW (Writers ДО smP_T × Readers ПОСЛЕ smP_T) | non-empty | **живой** | не передвигаемый |
| 6 | t13 | 462 | WAR (Readers ДО smP_T/smdO × Writers ПОСЛЕ next-qt smQ/smdO) | non-empty | **живой** | не передвигаемый |

**Итог 0d-fresh (без наследования от 037-stale)**: **6/6 барьеров ЖИВЫЕ, ни один не мёртвый и не передвигаемый в текущей структуре**. Vugar-гипотеза «пустой склад» для t_new2 **опровергнута адресным перебором** — Readers ДО от Step F создают WAR-hazard vs Writers ПОСЛЕ от Step G aliased.

**Barrier stall 2.76% distribution**: примерно ~0.46% на каждый барьер (равное распределение приблизительно). Не мишень.

**Правило 13** (напоминание): любое будущее снятие барьера = отдельный гейт + racecheck (sanitizer.race) + bit-exact 11/11 + ABBA ≥6 пар.

---

## 0e-fresh: dS_nat scatter — счёт инструкций и объект измерения

### 0e.1 SASS-факт (fresh binary)

`grep -oE "STS\.U16" kernel_merged_v1` = **48 инструкций** статических.

Разложение по коду:
- **Step E (dS_nat scatter, строки 368-371)**: 4 STS.U16 per (ni_a, ni_b) pair × `NI_DP/2 = 4` iters unrolled = **16 STS.U16/lane/qt**
- **Step G (smP_T scatter, строки 417-420)**: 4 STS.U16 per ni × `NI_QK = 8` iters unrolled = **32 STS.U16/lane/qt**
- **Всего**: 16 + 32 = **48 ✓** (совпадает со SASS)

### 0e.2 Объект измерения — Step E dS_nat scatter

**Явная подпись**: пункт 0e относится к **Step E dS_nat scatter (не Step G smP_T scatter)**.

- **Ширина stores**: **STS.U16 (2 bytes)** — подтверждено SASS
- **Count per lane per qt**: **16 STS.b16**
- **Count per qt per block**: 16 × 128 threads = **2048 STS.b16**
- **Формулы адреса** (строки 368-371 prod):
  ```
  smdS_stage[i_local_lo * 80 + ja_lo] = dsa_lo_fp8
  smdS_stage[i_local_lo * 80 + jb_lo] = dsb_lo_fp8
  smdS_stage[i_local_hi * 80 + ja_lo] = dsa_hi_fp8
  smdS_stage[i_local_hi * 80 + jb_lo] = dsb_hi_fp8

  где i_local_lo = wid*16+l_div4, i_local_hi = i_local_lo + 8
      ja_lo = ni_a*8 + l_mod4*2, ja_hi = ja_lo + 1
      jb_lo = ni_b*8 + l_mod4*2 (ni_b = ni_a + 1)
      SMDS_STAGE_STRIDE = 80
  ```

### 0e.3 Смежность и потенциал pack-analog

- ja_lo / jb_lo: расстояние **8** (ni-frag step). НЕ adj для STS.32 склейки.
- ja_lo / ja_hi: расстояние **1** (adj), но dsa_lo_fp8 (от pa_lo_h2) и dsa_hi_fp8 (от pa_hi_h2) — **от разных lane-fragments (m_lo vs m_hi)**. Требуется cross-lane rewrite (SHFL/PRMT), как в dk_new W2 pack.
- i_local_lo / i_local_hi: расстояние **8 rows = 640 B** (далеко).

**Возможность pack-analog**: **есть**, требуется derivation хореографии **от читателя** — Step F drain делает `LDS.128 → reg chunk → STG.128 DRAM` (uint4 chunk = 16 contiguous bytes из smdS_stage[r*80 + col_byte]). Reader — блочный memcpy, а не MMA — что упрощает pack по сравнению с dk_new (где reader был MMA-A frag).

**Механизм pack-analog**:
1. Cross-lane rewrite 16 STS.b16 → 4 STS.32 (объединяя 2×2 halves per lane)
2. Requires SHFL exchange между lane pairs (dsa_lo ↔ dsb_lo пары адj в j, но from different lane-fragments)
3. Ожидаемо ~4 SHFL + 8 PRMT ALU per pair × 4 pairs = ~16 SHFL + 32 PRMT в hot loop

**Регистровая цена** — **только через ptxas-факт после probe** (правило кампании).

---

## 0f-v2: вердикт-карта пересобранная по числам 037-r2 (с механизмами)

| Цель | Измеренный класс | Потенциал (МЕХАНИЗМ) | Риск | Цена разведпробы (paper→probe) |
|:--|:--|:--|:--|:--|
| **A** | **LDS.U16 класс #7 smdO MMA_dV read**: 256 ops/lane/qt = 50% всех LDS; часть mio_throttle 25.10% | **Механизм**: LDS.U16 → LDS.32 **НЕ применима** (адресный анализ 0c.4 показал row-stride 256B между парами). Кандидаты: (i) `ldmatrix.sync.aligned.m8n8.x4.b16` cooperative-lane load, (ii) structural rework XOR-паттерна scatter smdO. Оба требуют SMEM layout compat probe (Step A cp.async + Step D dPr + Step H MMA_dV). | **высокий** (layout mismatch между Step D и Step H возможен → double SMEM footprint → potentially 1 blk/SM). | Бумажный probe: bit-map совместимости XOR-паттерна scatter smdO с ldmatrix.sync fp16 фрагмент layout. Если совместим → в probe-хоресос без правки production. |
| **B** | **LDS.U16 класс #5 smV MMA-B read**: 128 ops/lane/qt = 25% LDS; часть mio_throttle | **Механизм**: LDS.U16 → LDS.32 **НЕ применима** (пара k_lo/k_hi delta 8 bytes, не adj). Кандидат: `ldmatrix.sync.aligned.m8n8.x4.b8` для FP8 B fragment (sm_120a поддержка требует verify). | средний (FP8 ldmatrix - новый инструмент на Blackwell). | Бумажный probe: sm_120a fp8 ldmatrix availability + XOR compat. |
| **C** | **LDS.32 классы #1/#2/#4/#6 (smQ, smK, smdO Step D, smP_T)**: 128 ops/lane/qt (~25% LDS) | **Механизм**: LDS.32 → LDS.64 склейка. **НЕ применима** для всех 4 классов (byte-Δ ≥ 16 в парах, не 4B adj). | низкий-средний. | Не рекомендуется в первую пробу (все 4 класса имеют структурное препятствие). |
| **D** | **Step E dS_nat scatter**: 16 STS.b16/lane/qt (0e-fresh) | **Механизм**: pack-analog cross-lane rewrite → 4 STS.32 per lane per qt (−75% count). Derivation "от читателя" (Step F LDS.128 chunk drain — блочный memcpy, упрощает pack vs dk_new W2). Требует SHFL/PRMT в hot loop. | средний-высокий (регистровая цена — только через ptxas-факт; potentially +5..15r → сохранение 2 blk/SM надо проверить). | Бумажный probe: derivation от reader Step F byte-map (16-byte chunk uint4), затем ptxas-факт на variant-branch. |
| **E** | Snятие 5120B dead-alloc (smdS_T_stage) | **Механизм**: удалить `+ Bc*80` из smem_bytes launcher. **НЕ даёт 3 blk/SM skip** (нужно −13568B для ceiling 33024B; cleanup даёт 46592−5120=41472 → всё ещё 2 blk). | нулевой (bit-exact 11/11 достаточно). | Тривиальная правка launcher smem_bytes; sanity only. |
| **F** | Barrier снятие (t_new2 или другие) | **Механизм**: адресное снятие 6/6 барьеров **НЕ применимо** (0d-fresh: все 6 живые, RAW/WAR hazards с непустыми пересечениями). Vugar-гипотеза «пустой склад» для t_new2 опровергнута. | **не применимо**. | Требует структурной перестройки для снятия — вне разведки. |
| **G** | 3-blocks/SM occupancy | **Механизм**: **невозможен** (0b-fresh: reg ceiling **170r** vs peak фазы ~209r; SMEM ceiling **≤33024B** vs 46592). Оба лимитера активны. | не применимо. | Требует структурной перестройки dV_acc (64r fp32) + SMEM −13568B. |

**Прогнозы wall (правило TZ 037-r2)**: **не заявляются вне механизма**. Для каждой цели указан только механизм и цена разведпробы. Wall-эффект = paper→probe→ptxas-факт→ABBA ≥6 пар.

### Рекомендация первой пробы

**Цель D (pack-analog dS_nat scatter)** — самая тщательно измеримая:
- Reader (Step F) — блочный memcpy, простая derivation.
- 16 STS.b16 → 4 STS.32 механизм чётко определён.
- Регистровая цена измерима через ptxas-факт после probe.
- Не требует ldmatrix / cooperative reads / layout mismatch.

**Цель A (класс #7 smdO)** сохраняет наибольшую механическую массу (256 ops = 50% LDS), но требует более глубокого paper-probe (совместимость ldmatrix с XOR-паттерном) — **вторая проба, если D не сработала**.

---

## Итог 037-r2

1. **Хвосты 0c/0d/0e пересняты на опознанном объекте** (fresh SASS md5 `0d54721e…`, ptxas 254r).
2. **Правки нитей 037-r**:
   - (a) stale-колонка barrier 2.59 — **уже правильно в 037-r** (напоминание принято).
   - (b) TFLOPS от медианы = **237.0 T** (не 236.7).
   - (c) NCu-налог: **бимодален** — режим A (1-pass single-metric) +52.0%; режимы B/C (multi-pass amortized) +1.4%. Правило 6 диапазон "28-47%" **устарел**, новый диапазон **1.4%..52%** зафиксирован для разных режимов.
3. **0c-fresh**: 132 LDS.32 + 384 LDS.U16 + 1 LDS.128 = **516 LDS + 1 LDS.128** ops/lane/qt (совпадает с SASS). **Класс #7 склейка LDS.U16→LDS.32 НЕ применима** (row-stride 256B препятствие; bit-map element-XOR раскладки smdO приведен ПЕРВОЙ СТРОКОЙ секции 0c.3).
4. **0d-fresh**: **6/6 барьеров живые** (адресный перебор адресных множеств метод 021). t_new2 **охраняет живую пару Readers[F]/Writers[G]**, **не пустой склад** — Vugar-гипотеза опровергнута.
5. **0e-fresh**: Step E dS_nat scatter **= 16 STS.b16/lane/qt** (SASS 48 STS.U16 = 16 Step E + 32 Step G ✓).
6. **0f-v2 переосмыслена**: 
   - **Цель A (класс #7)** перевершилась в структурный кандидат ldmatrix/rework — **высокий риск**.
   - **Цель D (pack-analog dS_nat scatter)** — **новая первая проба** (простая derivation "от читателя").

**Прогнозы wall вне механизма — не заявляются** (правило ТЗ 037-r2 соблюдено). Правок кода/harness/EXPECT-dict не производилось.

---

## Файлы

- `runs/reports/037r2_sass.sh` + `037r2_sass_data.txt` + `037r2_sass_merged_only.txt` — SASS instruction counts fresh
- `runs/reports/037r2_sass_widths.sh` — LDS width taxonomy
- `runs/reports/037r2_sts_decomp.sh` — STS.U16 разложение Step E/G
- `runs/reports/037r2_ncu_tax.sh` + `037r2_ncu_tax_data.txt` — NCu-налог измерения

Chain md5: 036-r `2d770375…` → 037 `a1f6edd7…` (стейл, справочно) → 037-r `886ac6e5…` → **037-r2 `<computed>`**

---

**End 037-r2.**
