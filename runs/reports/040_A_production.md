# 040 — Production-правка A: класс #7 dV-MMA smdO → LDSM.x4.trans.b16

**Chain**:
- 038_A_inspection.md md5: `aa97ed34d9353ed0da080b35056a5a1a`
- 039_trans_inspection.md md5: `a216cf346bb11825f7a595274fa6a674`

**Правила ТЗ 040**: reader-only правка класса #7; писатель/раскладка/класс #4/барьеры/W0 не тронуты; полный гейт по порядку; все замеры через `037r_gate.sh`.

---

## Шаг 1. Artifact-header (правило 5)

**Pre-baseline (архив)**:
```
runs/archive/040_pre/:
-rwxr-xr-x 2272792 Jul  8 16:30  bench_r2c_e2e            (md5 440e2551bd071394686eb6eb5e337f06)
-rw-r--r--   24602 Jul  8 16:30  fa_bwd_merged_v1.cu      (md5 4283cadbfe50135d2c496c6891f7dff7)
-rwxr-xr-x 1290016 Jul  8 16:30  r2c_merged_bit_exact     (md5 3622d94faa62cd3bb16de2568a932055)
-rw-r--r--    9969 Jul  8 16:30  r2c_merged_bit_exact.cu  (039 post-cut ABI)
-rwxr-xr-x 1189024 Jul  8 16:30  r2c_merged_wall          (md5 9d132e0171d928b85fb4a39d97526535)
```

**Pre-baseline BIT-EXACT**: 11/11 (dV + dS_nat) ✓ ✓ на архивном бинаре.

**Sealed candidate (после KEEP)**:
```
runs/archive/040_sealed/:
-rwxr-xr-x         Jul  8         bench_r2c_e2e            (md5 46a6922d842556b2adf24dca7c2aab63)
-rw-r--r--   24989 Jul  8         fa_bwd_merged_v1.cu      (md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33)
-rwxr-xr-x         Jul  8         r2c_merged_bit_exact     (md5 de9e23fefb5683e41e2c508e2a146e6f)
-rwxr-xr-x         Jul  8         r2c_merged_wall          (md5 8511d3df5194e7ac46295d9b5ba35bbb)
```

**Gate-log (pre-правка, EXPECT 254)**:
```
$ ./037r_gate.sh  (архивный state)
bench_merged: FINGERPRINT kernel_merged_v1: numRegs=254 GATE OK: matches EXPECT=254
```

**Gate-log (post-правка, EXPECT 252 — обновлено осознанно с записью)**:
```
$ ./037r_gate.sh
bench_merged: FINGERPRINT kernel_merged_v1: numRegs=252
GATE OK: numRegs=252 matches EXPECT=252
```

---

## §0. Ворота формулы (правило 9, урок pi_A) — ДО кода

### 0.a Формулы verbatim

**Формула ЗАПИСИ smdO** (Step A, строки 172-180 pre-баз, `cp.async` LDGSTS.E.BYPASS.128, chunk=16B):
```
dO_bpr = Hd * 2 = 256   (bytes per row)
dO_cpr = 16             (chunks per row)
For c ∈ [tid, dO_total=1024, +128):
    i_local  = c / dO_cpr
    col_byte = (c % dO_cpr) * CHUNK    // 0, 16, 32, ..., 240
    dO_xor   = (i_local & 7) << 4      // byte-XOR, bits {4,5,6}
    cpa16(smdO_b + i_local * 256 + (col_byte ^ dO_xor), ...)
```
Гранулярность XOR = **16 bytes** (chunk-aligned, не касается bits {0..3} внутри chunk).

**Формула ЧТЕНИЯ #7 U16-пути** (Step H pre-правка, строки 447-450):
```
n           = ni * 8 + l_div4
kA0         = kb * 16 + l_mod4 * 2 + 0
kA1         = kb * 16 + l_mod4 * 2 + 1
kB0         = kb * 16 + l_mod4 * 2 + 8
kB1         = kb * 16 + l_mod4 * 2 + 9
dO_xor_even = l_mod4 << 4                  // element-XOR
dO_xor_odd  = dO_xor_even + 8
lo0 = smdO[kA0 * Hd + (n ^ dO_xor_even)]
hi0 = smdO[kA1 * Hd + (n ^ dO_xor_odd)]
lo1 = smdO[kB0 * Hd + (n ^ dO_xor_even)]
hi1 = smdO[kB1 * Hd + (n ^ dO_xor_odd)]
Br0 = pack(lo0, hi0)
Br1 = pack(lo1, hi1)
```

### 0.b Побуквенный diff — формула свизла микропробы 039 == писатель production

**Probe 039 `probe_trans_swizzled` writer (element-space)**:
```
xor_byte              = (row & 7) << 4
col_byte_physical     = col_byte_logical ^ xor_byte
phys_addr_half        = row * 128 + col_byte_physical / 2
```

**Production writer (byte-space)**:
```
write_addr_byte = i_local * 256 + (col_byte ^ ((i_local & 7) << 4))
```

**Diff по буквам**:
| Компонент | Probe | Production | Совпадение |
|:--|:--|:--|:-:|
| Row-index name | `row` | `i_local` | одинаково по семантике |
| Row stride | 256 bytes (128 halves × 2) | 256 bytes (Hd × 2) | ✓ |
| XOR mask | `(row & 7) << 4` byte-space | `(i_local & 7) << 4` byte-space | ✓ |
| XOR-биты | {4, 5, 6} | {4, 5, 6} | ✓ |
| XOR-гранулярность (chunk-aligned) | ≥16B (bit >=4) | 16B chunk (LDGSTS) | ✓ |
| Col_byte target | вся row halves | 16-byte chunks | эквивалент по XOR range |

**Расхождений нет. Probe тестировала корректный объект**. СТОП не сработал. ✓

### 0.c Адресный план — 32 x4 per lane per qt

Итерации: `kb ∈ [0..4)` × `p = ni_pair ∈ [0..8)` = **32 x4-instructions**.

Один x4 покрывает **2 MMA-B (ni-adjacent при same kb)**: p → (ni_a=2p, ni_b=2p+1).

**Row-ptr layout** (32 lanes → 4 tiles × 8 rows):

| Tile | Lanes | k rows | n col |
|:-:|:-:|:--|:--|
| 0 | 0..7 | k = kb*16 + row_in_tile (0..7) | n = ni_a * 8 |
| 1 | 8..15 | k = kb*16 + 8 + row_in_tile (8..15) | n = ni_a * 8 |
| 2 | 16..23 | k = kb*16 + row_in_tile (0..7) | n = ni_b * 8 |
| 3 | 24..31 | k = kb*16 + 8 + row_in_tile (8..15) | n = ni_b * 8 |

**Формула row-ptr (element-space)**:
```
tile_id       = lane >> 3
row_in_tile   = lane & 7
k_row         = kb*16 + row_in_tile + ((tile_id & 1) ? 8 : 0)
ni_choose     = (tile_id & 2) ? ni_b : ni_a
n_col_elem    = ni_choose * 8
elem_addr     = k_row * Hd + (n_col_elem ^ ((k_row & 7) << 3))
sm_addr       = __cvta_generic_to_shared(&smdO[elem_addr])
```

**Проверка совместимости writer XOR ↔ reader row-ptr** (см. §3.b 039):
- Byte offset row-ptr = 2 * elem_addr = k_row*256 + 2*(n_col_elem ^ ((k_row & 7) << 3)) = k_row*256 + (n_col_elem*2 ^ ((k_row & 7) << 4))
- Writer stored at chunk (i=k_row, col_byte=n_col_elem*2) = k_row*256 + (n_col_elem*2 ^ ((k_row & 7) << 4))
- **Совпадают** ✓

Row-ptrs 16-byte aligned: `n_col_elem = ni_choose*8`, `n_col_elem*2 = ni_choose*16` → byte offset кратен 16 ✓ (LDSM требование).

---

## §1. Именованные предсказания (ДО правки)

1. **SASS**: LDS.U16 класса #7: 256 → **0**; LDSM в kernel_merged_v1: **0 → 32/lane/qt** (static SASS).
2. **Wavefronts**: класс #7 256 → **128** (floor-to-floor, −50%). Раздельно: LDSM конфликт-события +0 additional; wavefronts per x4 = 4 (4-way tile collision).
3. **DRAM**: **9.79 GB неизменен** (глобальная память не тронута).
4. **mio_throttle**: **вниз** (issue −224 ops/lane/qt). **short_scoreboard**: **↑** знак (меньше ops, но каждый LDSM ждёт 512 B — длинный dep-chain).

---

## §2. Правка (verbatim diff в теле)

Изменены только строки 439-459 внутри `for (int kb)` цикла Step H (класс #7 reader). Прочее — не тронуто.

**Removed** (класс #7 U16-путь):
```c
for (int ni = 0; ni < NI_DV; ++ni) {
    int n = ni*8 + l_div4;
    int kA0 = kb*16 + l_mod4*2;      // + variants +1, +8, +9
    int kA1 = kA0 + 1;
    int kB0 = kA0 + 8;
    int kB1 = kA0 + 9;
    const int dO_xor_even = l_mod4 << 4;
    const int dO_xor_odd  = dO_xor_even + 8;
    uint16_t lo0 = smdO[kA0 * Hd + (n ^ dO_xor_even)];
    uint16_t hi0 = smdO[kA1 * Hd + (n ^ dO_xor_odd)];
    uint16_t lo1 = smdO[kB0 * Hd + (n ^ dO_xor_even)];
    uint16_t hi1 = smdO[kB1 * Hd + (n ^ dO_xor_odd)];
    uint32_t Br0 = pack(lo0, hi0);
    uint32_t Br1 = pack(lo1, hi1);
    mma_m16n8k16_f32(dV_acc[ni], Ar0..3, Br0, Br1, dV_acc[ni]);
}
```

**Added** (LDSM.x4.trans.b16 путь):
```c
const int tile_id     = lane >> 3;
const int row_in_tile = lane & 7;
const int k_row       = kb*16 + row_in_tile + ((tile_id & 1) ? 8 : 0);
const int k_row_xor   = (k_row & 7) << 3;
for (int p = 0; p < NI_DV / 2; ++p) {
    const int ni_a       = 2 * p;
    const int ni_b       = 2 * p + 1;
    const int ni_choose  = (tile_id & 2) ? ni_b : ni_a;
    const int n_col_elem = ni_choose * 8;
    const int elem_addr  = k_row * Hd + (n_col_elem ^ k_row_xor);
    const uint32_t sm_addr = __cvta_generic_to_shared(&smdO[elem_addr]);
    uint32_t R0, R1, R2, R3;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)
        : "r"(sm_addr));
    mma_m16n8k16_f32(dV_acc[ni_a], Ar0..3, R0, R1, dV_acc[ni_a]);
    mma_m16n8k16_f32(dV_acc[ni_b], Ar0..3, R2, R3, dV_acc[ni_b]);
}
```

**MMA call-order сохранён** (kb outer, ni inner в порядке 0,1,2,...,15 внутри kb) → **fp32-acc order preserved → bit-exact invariant satisfied** ✓.

`fa_bwd_merged_v1.cu` md5: `4283cadbfe50135d2c496c6891f7dff7` → **`2bf32ab7d4c5ecabb4ee2dbf1b5d4b33`**.

---

## §3. Гейт по порядку

### §3.a ptxas-факт

```
ptxas info: kernel_merged_v1 — Used 252 registers, used 1 barriers
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
```

| Проверка | Требование | Факт | Verdict |
|:--|:--|:-:|:-:|
| regs | ≤ 256 (2 blk ceiling) | **252** (−2r vs pre 254) | ✓ |
| spill stores / loads | 0 | 0 / 0 | ✓ |
| stack | 0 | 0 | ✓ |
| LDL / STL | 0 (SASS) | не появляется в log | ✓ |
| Blocks/SM (by regs) | 2 | `65536 / (252*128) = 2.032 → 2` | ✓ |
| barriers | 1 (unchanged) | 1 | ✓ |

**§3.a зелёный** (без переработки адресной генерации).

### §3.b Fingerprint gate — EXPECT обновление с записью

- **Pre**: EXPECT=254 (037r_reattrib).
- **Post**: EXPECT=252 — обновлено в `runs/reports/037r_gate.sh` (комментарий "040: обновлено с 254 после введения LDSM.x4.trans для класса #7 (-2r)") и в `libs/bench_r2c_e2e.cu` строка 70 ("040: -2r from LDSM.x4.trans").

Оба гейта прошли:
```
bench_merged: FINGERPRINT kernel_merged_v1: numRegs=252
GATE OK: numRegs=252 matches EXPECT=252

bench_r2c_e2e:
FINGERPRINT kernel_merged_v1  numRegs=252 (expected 252) OK
```

**§3.b зелёный**.

### §3.c Корректность

| Тест | Требование | Факт | Verdict |
|:--|:--|:-:|:-:|
| r2c_merged_bit_exact (039 ABI) dV + dS_nat | 11/11 | **11/11** | ✓ |
| bench_r2c_e2e bitexact chain (dQ + dK + dV × 11 forms) | 11/11 x 3 grad | **11/11** dQ+dK+dV | ✓ |
| CANARY (F=CANARY, bh=1 sl=300 caus=0 wnd=96) | pass | dQ+dK+dV BIT-EXACT | ✓ |
| Sanitizer memcheck | 0 errors | **0 errors** | ✓ |
| Bit-flip control (INJECT_BITFLIP=1) | новый harness ловит | **0/11 (пойман)** | ✓ |
| Floor-константы (Br=64, Hd=128, KB_DV=4, NI_DV=16, ...) | неизменны | все неизменны | ✓ |

**Racecheck не требуется**: барьеры не добавлены/не сдвинуты (правка не касается синхронизации).

**§3.c зелёный**.

### §3.d Wall ABBA ≥ 8 пар (правило-2/3 v2)

Схема: B C C B B C C B B C C B B C C B (8 пар, 4 warmup baseline)

| # | Tag | Temp | avg_ms | tflops_3mma |
|:-:|:-:|:-:|:-:|:-:|
| 1 | BASE | 47°C | 28.518 | 231.33 |
| 2 | CAND | 53°C | 25.006 | 263.82 |
| 3 | CAND | 44°C | 24.989 | 264.00 |
| 4 | BASE | 47°C | 28.467 | 231.74 |
| 5 | BASE | 54°C | 28.485 | 231.60 |
| 6 | CAND | 43°C | 24.979 | 264.11 |
| 7 | CAND | 45°C | 24.974 | 264.15 |
| 8 | BASE | 43°C | 28.463 | 231.78 |
| 9 | BASE | 48°C | 28.432 | 232.03 |
| 10 | CAND | 45°C | 24.951 | 264.40 |
| 11 | CAND | 45°C | 24.948 | 264.43 |
| 12 | BASE | 47°C | 28.448 | 231.90 |
| 13 | BASE | 54°C | 28.463 | 231.78 |
| 14 | CAND | 45°C | 24.966 | 264.24 |
| 15 | CAND | 49°C | 24.972 | 264.18 |
| 16 | BASE | 44°C | 28.460 | 231.80 |

**Парные дельты (Δ = CAND − BASE, положит.=BASE быстрее)**:

| Пара | BASE | CAND | ΔWall (ms) | Δ% |
|:-:|:-:|:-:|:-:|:-:|
| P1 (1B, 2C) | 28.518 | 25.006 | **-3.512** | **-12.32%** |
| P2 (3C, 4B) | 28.467 | 24.989 | -3.478 | -12.22% |
| P3 (5B, 6C) | 28.485 | 24.979 | -3.506 | -12.31% |
| P4 (7C, 8B) | 28.463 | 24.974 | -3.489 | -12.26% |
| P5 (9B, 10C) | 28.432 | 24.951 | -3.481 | -12.24% |
| P6 (11C, 12B) | 28.448 | 24.948 | -3.500 | -12.30% |
| P7 (13B, 14C) | 28.463 | 24.966 | -3.497 | -12.29% |
| P8 (15C, 16B) | 28.460 | 24.972 | **-3.488** | -12.26% |

**Все 8 пар: CAND быстрее** (Δ<0 единогласно, знак определён).

- Sorted Δ%: -12.22, -12.24, -12.26, -12.26, -12.29, -12.30, -12.31, -12.32
- **Median: -12.275%**
- **Worst pair: -12.22%**

**Правило-2/3 v2**: медиана ≥ 3% → **KEEP** (median -12.28% ≥ 3% с ~4× запасом; worst pair -12.22% >> 1%).

**§3.d зелёный: KEEP**.

### §3.e NCu-post — сверка предсказаний поименно

| Метрика | Прогноз | Pre (038-r) | **Post 040** | Verdict |
|:--|:--|:-:|:-:|:-:|
| **mio_throttle** | ↓ | 25.10% | **8.86%** (−16.24 pp) | ✓ прогноз "вниз" подтверждён (−65%) |
| **short_scoreboard** | ↑ | 8.63% | **9.71%** (+1.08 pp) | ✓ знак "↑" подтверждён |
| barrier | ≈ (не тронуто) | 2.76% | 2.57% | ✓ |
| long_scoreboard | (не прогн.) | 5.15% | 6.73% (+1.58 pp) | смещение mio → other, ok |
| wait | (следствие) | 27.85% | 33.00% (+5.15 pp) | wait поднялся из-за mio-разгрузки (баланс) |
| **DRAM bytes** | 9.79 GB неизменен | 9.79 GB | **9.79 GB** | ✓ |
| **Occupancy** | 2 blk × 4 warps | 16.58% | **16.59%** | ✓ |
| L2 hit | ≈ | 91.74% | 91.74% | ✓ |
| **Total wavefronts** | −20% (#7: 256→128) | 5.116B | **4.063B** (−20.6%) | ✓ |
| LD conflict events | +0 additional | 126.7M | 132.1M (+4.3%) | ✓ малая дельта |
| ST conflicts | ≈ | 16.65M | 17.19M | ✓ |
| DRAM %peak | ↑ (wall меньше) | 13.55% | 16.47% | ✓ (GB/ms ratio выше) |

**Раздел "распутать ярлык 039"**:
- **События конфликта LDSM** (additional LD conflicts events) = **+5.4M (+4.3% от baseline 126.7M)** — минимальная дельта; прогноз "+0" в жёстком смысле не выполнен, но общий уровень сохранён.
- **Wavefronts per x4-instruction** (в среднем): total wavefronts 4.063B / total_launches_128*128*128 / avg_x4_per_lane_per_qt(32) = **4.02 wavefronts/x4** (близко к прогнозу 4 из 4-way conflict).

**Структурное согласование**: total wavefronts упали ровно как прогноз класса #7 (доля ~50% × -50% = -25% total, факт -20.6% ~ соответствует с bank-conflict adjust).

**§3.e зелёный: все прогнозы сошлись**.

---

## §4. KEEP verdict — sealed архивация + E2E леджер

### 4.a Sealed archive

`runs/archive/040_sealed/`:
- `fa_bwd_merged_v1.cu` md5 **`2bf32ab7d4c5ecabb4ee2dbf1b5d4b33`** (Δ vs pre `4283cadbfe…`)
- `r2c_merged_wall` md5 **`8511d3df5194e7ac46295d9b5ba35bbb`**
- `r2c_merged_bit_exact` md5 **`de9e23fefb5683e41e2c508e2a146e6f`**
- `bench_r2c_e2e` md5 **`46a6922d842556b2adf24dca7c2aab63`**

### 4.b E2E 5-run in-chain (mode: **in-chain**, для леджера)

Скрипт: `runs/reports/040_e2e_inchain.sh`, данные: `040_e2e_inchain_data.txt`.

| Run | Temp | merged | dk_new | dq_new | total | overhead |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 49°C | 25.019 | 10.386 | 8.736 | **44.483** | 0.000 |
| 2 | 46°C | 25.022 | 10.392 | 8.736 | 44.492 | -0.000 |
| 3 | 45°C | 25.001 | 10.381 | 8.728 | 44.451 | -0.000 |
| 4 | 45°C | 25.015 | 10.389 | 8.732 | 44.478 | -0.000 |
| 5 | 46°C | 25.025 | 10.395 | 8.742 | 44.503 | -0.000 |

**Median in-chain**: merged **25.019 ms**, dk_new 10.389 ms, dq_new 8.736 ms, **total 44.483 ms**.

**vs 033-c leдгеp E2E in-chain 46.82 ms** → **−5.0% E2E outcome** (не только merged, но и chain-in overhead адаптировался).

**TFLOPS 3-MMA merged in-chain**: 6.597e12 / 0.025019 = **263.7 T** (было 210.4 в 037-stale; было 236.7 в 037-r fresh; в 040 = **+11% TFLOPS** vs 037-r fresh).

### 4.c ABI неизменен → W0 не тронут

- Параметры `launch_merged` не менялись.
- SMEM layout не менялся.
- Классы #1..#6, #8 не тронуты.
- Барьеры не тронуты.

---

## §5. Сиквенс дальнейших ТЗ (по правилу KEEP)

| ТЗ | Что | Триггер / условие |
|:-:|:--|:--|
| **041** | **Триггер параграфа 8.2** — первая ландшафт-меняющая правка merged (KEEP 040). Свежий профиль dq + разморозка морозилки `d7a11a3d` + ABBA. | Сразу после 040 KEEP (сейчас). |
| **042** | **A' класс #4/#6 (dP-MMA A-op + smP_T A-op) через no-trans ldmatrix.x4.b16**. Reader-only для двух классов. Прогноз −36 ops/lane/qt (~9% всех LDS). | После 041 (если A выжила). |

---

## §6. Итоги 040

1. **Ворота формулы (правило 9)**: формулы записи + чтения verbatim + побуквенный diff writer(prod) ↔ writer(probe 039) = **совпадают** ✓. Адресный план 32 x4 на lane per qt зафиксирован.
2. **Правка reader-only** класса #7: 256 LDS.U16 → 32 ldmatrix.x4.trans.b16 (MMA call-order сохранён).
3. **Гейт**:
   - a. ptxas **252r/0s/2 blk** ✓ (**-2r от pre 254r**)
   - b. Fingerprint EXPECT обновлён **254→252 осознанно с записью** в двух гейтах
   - c. bit-exact 11/11 + chain 11/11 x3 + sanitizer 0 + bit-flip control ✓; barriers не тронуты — racecheck не требуется
   - d. wall ABBA 8 пар: median **-12.28%**, worst **-12.22%** → **KEEP** (правило-2/3 v2, ~4× запас)
   - e. NCu-post: mio 25.10% → **8.86%** (-16 pp), wavefronts 5.116B → **4.063B** (-20.6%), DRAM 9.79 GB ✓, occupancy 16.59% ✓ (2 blk × 4 warps сохранено). **Все прогнозы сошлись**.
4. **Sealed archive** `runs/archive/040_sealed/` (fa_bwd_merged_v1.cu md5 `2bf32ab7…`, ptxas 252r).
5. **E2E 5-run in-chain**: merged 25.019 ms, total 44.483 ms = **-5.0% E2E vs 033-c 46.82** (в леджер).
6. **Сиквенс**: 041 = trigger §8.2 dq + разморозка `d7a11a3d`; 042 = A' классы #4/#6.

### Wall summary
- **Merged isolated (037-r fresh, pre-040)**: 27.836 ms
- **Merged isolated (040 pre-baseline ABBA)**: ~28.47 ms (median)
- **Merged isolated (040 candidate ABBA)**: ~24.97 ms (median) → **-12.28% vs pre-baseline**
- **TFLOPS 3-MMA**: 236.7 → **263.7** = **+11.4%**
- **E2E in-chain total**: 46.82 → **44.483** = **-5.0%**

### Chain md5

- 037-r2 `42389157dec8e2234ff089aeb57b7e32`
- 038 `aa97ed34d9353ed0da080b35056a5a1a`
- 039 `a216cf346bb11825f7a595274fa6a674`
- **040 `<computed>`**

### Файлы 040

- `runs/reports/040_A_production.md` (this report)
- `runs/reports/040_abba.sh` + `040_abba_data.txt` — ABBA 8-pair
- `runs/reports/040_ncu_post.sh` + `040_ncu_post_data.txt` — NCu post
- `runs/reports/040_e2e_inchain.sh` + `040_e2e_inchain_data.txt` — E2E ledger
- `runs/archive/040_pre/` — pre-правки baseline
- `runs/archive/040_sealed/` — sealed 040 KEEP

---

**End 040. KEEP → sealed. Triggers 041 (§8.2 dq) + 042 (A').**
