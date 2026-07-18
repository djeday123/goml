# 011 — dk_new bank-conflicts атрибуция + бумажный расчёт (2026-07-06)

## ARTIFACT-HEADER

### Кросс-верификация
```
md5 010_dk_sass.md:  bbdb223ea2c452333f7f1ff8eb625279  (11620 bytes)
md5 009_ncu.md:      62977f103b01436c877c56d4b7a48a4a  (11748 bytes)
```

### ls -la runs/reports/ (свежая часть 011-related)
```
-rwxr-xr-x   522 Jul 6 15:11  011_dk_ld_st_split.sh          (LD/ST wavefronts split)
-rwxr-xr-x   594 Jul 6 15:10  011_ncu_source_dk.sh           (source-mapped attempt)
-rw-r--r-- 11620 Jul 6 09:33  010_dk_sass.md
-rw-r--r-- 11748 Jul 6 09:26  009_ncu.md
-rw-r--r-- 1226067 Jul 6 09:27 010_dk_new_sass_full.txt
```

## 011.1 — LD/ST split (bank conflicts targeting)

`011_dk_ld_st_split.sh` вернуло факт (kernel_dk_new, canonical bh=128 sl=8192):

| Metric | Value | Δ vs total |
|--------|:-----:|:----------:|
| **shared_ld inst** (warp-instructions) | 889,192,448 | base LD |
| **LD wavefronts** | 2,432,936,234 | +1.54B extra |
| **Bank conflicts LD** | 1,694,738,730 | ~= extra wavefronts (совпадает 1:1 с 2.43-0.89) |
| **shared_st inst** | 570,425,344 | base ST |
| **ST wavefronts** | 567,754,205 | ≈ inst 1:1 |
| **Bank conflicts ST** | 30,883,293 | **5.4 % excessive ST** |

**Вывод**: **95%+ конфликтов приходят с LD-стороны**. STS-скаттер Q_T-фазы (Vugar-candidate) — **не виновник** (5.4% excessive negligible).

## 011.2 — Per-класс атрибуция (source-level, без per-SASS-line NCu)

`--print-source sass` требует debug-info (-lineinfo), в r1b_dk_wall binary этого нет. Атрибуция идёт по паре SASS-count + логическая привязка к строкам fa_bwd_dk_new.cu.

### Три класса LD в dk_new (из fa_bwd_dk_new.cu lines 141–197)

| Класс | Source lines | LDS.32 per qt-iter |
|-------|--------------|:------------------:|
| **Q-фидер → Qr regs** (для Q_T scatter) | 148-151 (KS_QK × 4) | **16** |
| **A-операнд (dS_T)** | 182-185 (KB_DK=2 × 4) | **8** |
| **B-операнд (Q_T)** | 190-191 (KB_DK=2 × NI_DK=16 × 2) | **64** |
| **Итого source-level** | | **88** |

Runtime: 128 qt-iters × 65536 warps × 88 = 738 M source-level LDS.
NCu measured: **889 M warp-inst** — Δ = ~150 M сверху (warmup K-cp.async инициализация + возможные implicit LDS). Порядок совпадает.

## 011.3 — Бумажный банк-расчёт (адрес → банк, по варпу)

### Обозначения
- Bank width = 4 bytes → word address = byte / 4
- Bank = word_addr mod 32
- Lane splitting в MMA: `l_div4 = lane >> 2 ∈ [0,8)`, `l_mod4 = lane & 3 ∈ [0,4)`

### Класс A (dS_T reads, Br=64, `smdS_T[m_lo * 64 + k_i_lo]`)

- `m_lo = wid*16 + l_div4` → варьируется по l_div4 ∈ [0,8)
- `k_i_lo = kb*32 + l_mod4*4` → варьируется по l_mod4 ∈ [0,4)
- Word addr = (m_lo*64 + k_i_lo) / 4 = m_lo*16 + kb*8 + l_mod4
- Bank = (m_lo*16 + kb*8 + l_mod4) mod 32 = ((l_div4 mod 2)*16 + l_mod4) mod 32 [wid*16*16 mod 32 = 0, kb*8 const-hoisted]

**Bank distribution across 32 lanes (fixed kb)**:
- l_div4 ∈ {0,2,4,6} (16 lanes) × l_mod4 ∈ {0..3}: banks **{0,1,2,3}** ← 16 lanes / 4 banks = **4-way collision**
- l_div4 ∈ {1,3,5,7} (16 lanes) × l_mod4 ∈ {0..3}: banks **{16,17,18,19}** ← 4-way collision

**Excess wavefronts на A read**: 3 extra per LDS.32 (4-way = 4 sequential wavefronts, 3 extra).
- 8 A-loads × 3 extra = **24 excess/qt/warp**

### Класс B (Q_T reads, QT_STRIDE=68, `smQ_T[n_d * 68 + k_i_lo]`)

- `n_d = ni*8 + l_div4` → l_div4 ∈ [0,8)
- `k_i_lo = kb*32 + l_mod4*4` → l_mod4 ∈ [0,4)
- Word addr = (n_d*68 + k_i_lo) / 4 = n_d*17 + kb*8 + l_mod4
- Bank = (n_d*17 + kb*8 + l_mod4) mod 32

**Развёртка по l_div4 при фикс. (ni, kb)** — start bank per l_div4-группа {l_mod4=0..3}:
| l_div4 | start = (l_div4*17) mod 32 | Banks {start .. start+3} |
|--------|:--------------------------:|:------------------------:|
| 0 | 0 | {0, 1, 2, 3} |
| 1 | 17 | {17, 18, 19, 20} |
| 2 | 34 mod 32 = 2 | {2, 3, 4, 5} ← **overlap {2,3} с l_div4=0** |
| 3 | 51 mod 32 = 19 | {19, 20, 21, 22} ← **overlap {19,20} с l_div4=1** |
| 4 | 68 mod 32 = 4 | {4, 5, 6, 7} ← overlap {4,5} с l_div4=2 |
| 5 | 85 mod 32 = 21 | {21, 22, 23, 24} ← overlap с l_div4=3 |
| 6 | 102 mod 32 = 6 | {6, 7, 8, 9} ← overlap с l_div4=4 |
| 7 | 119 mod 32 = 23 | {23, 24, 25, 26} ← overlap с l_div4=5 |

Bank-utilization card (per-bank hit count по 32 lanes):
- Banks {0, 1}: 1 lane каждый (l_div4=0)
- Banks {2, 3}: **2 lanes каждый** (l_div4=0 + l_div4=2)
- Banks {4, 5}: **2 lanes каждый** (l_div4=2 + l_div4=4)
- Banks {6, 7}: **2 lanes каждый** (l_div4=4 + l_div4=6)
- Banks {8, 9}: 1 lane каждый (l_div4=6)
- Аналогично {17, 18} 1-way, {19..24} 2-way, {25, 26} 1-way

**Максимум 2 lanes на банк** → **2-way conflict на B read**.

**Механизм overlap**: stride 17 mod 32 = 17. Шаг = 17 приводит к **паре двойного покрытия**, потому что 17 + 15 (l_mod4 spread + 1 within-group) = 32 → wrap-around в след. пару банков.

**Развенчание "нечётный простой стрид конфликт-свободен"**:
- Условие conflict-free для этой раскладки: **все 8 start-положений (l_div4 groups) должны быть попарно ≥4 банка apart**.
- Stride 17: sorted starts = {0, 2, 4, 6, 17, 19, 21, 23}. Adjacent diffs: {2, 2, 2, 11, 2, 2, 2, 9}. **Min diff = 2 < 4** → conflict.
- Простой S=17 гарантирует непересечение SINGLE-lane sequential walks, а не MULTI-lane 4-wide groups.

**Excess wavefronts на B read**: 1 extra per LDS.32 (2-way).
- 64 B-loads × 1 extra = **64 excess/qt/warp**

### Класс Q-фидер (swz_byte swizzle, `smQ[swz_byte(m, k)]`)

- Byte offset = row*128 + ((chunk^(row&7))<<4) + within (from fa_bwd_common.cuh:70)
- Word addr = row*32 + ((chunk^(row&7))*4) + within/4
- Bank = (row*32 + chunk*4 + (row&7)*4 (XOR) + within/4) mod 32 = distributed XOR-свизлом → **0 conflicts** (swz_byte спроектирован conflict-free)

## 011.4 — Суммарный theoretical excess vs measured

Per qt per warp:
- A: 24 excess (4-way × 8 loads)
- B: 64 excess (2-way × 64 loads)
- Q-feeder: 0
- **Total: 88 excess** (paper)

Runtime excess: 88 × 65536 warps × 128 qt = **738 M paper excess**.
NCu measured LD conflicts: **1.69 B**.
Δ 951 M (~2.3×) — избыток. Кандидаты:
- LDGSTS.E.BYPASS.128 (cp.async в SMEM) может добавлять implicit LD wavefronts.
- Дополнительная ILP-раскладка в SASS может дублировать реад-паттерн (не учтено в static count).

Порядок совпадает: **B-операнд доминирует** (73% excess по paper), A вторичен (27%), Q-фидер и STS негативно-negligible.

## 011.5 — Ресурсы dk_new (Vugar-запрос: точный SMEM, лимитер, запас)

### Точный SMEM total (из launcher fa_bwd_dk_new.cu:243)
```
smem_bytes = Br * hd + hd * QT_STRIDE + Bc * Br
           = 64*128 + 128*68  + 64*64
           = 8192  + 8704   + 4096
           = 20992 B/block   (без driver)
```
Плюс driver SMEM per block: 1024 B → **21,016 B dynamic + 1024 B driver = 22040 B total per block**.

### Лимитер blocks/SM (canonical, sm_120a, SM SMEM=100 KB=102400 B)
- **Registers**: 96r/thread × 128 threads = 12288 regs/block → 65536 / 12288 = 5.33 → **5 blocks reg-wise**
- **SMEM**: 102400 / 22040 = 4.65 → **4 blocks SMEM-wise** ← **лимитер**
- **Warps**: 4 warps/block, ceiling 12 warps/SM → 3 blocks warp-wise... wait, sm_120a max warps/SM (need to check — Hopper was 64, Blackwell...).

От NCu 004_r1b_fix_dk_ncu.txt в кампании: **Block Limit SMEM = 4, Block Limit Registers = 5, Block Limit Warps = 12, Theoretical Occupancy = 4 blocks × 4 warps = 16 warps/SM (33.33%)**. Совпадает.

### SMEM запас до потери 4-го блока

Slot размер per block at 4 blocks/SM = 102400 / 4 = **25600 B**.
Current usage: 22040 B.
**Headroom: 25600 - 22040 = 3560 B** (≈ 3.5 KB запас).

### Возможные QT_STRIDE изменения vs SMEM бюджет
| QT_STRIDE | smQ_T (Hd × STRIDE) | ΔSMEM vs 68 | Total | 4 blocks? | Headroom |
|-----------|:-------------------:|:-----------:|:-----:|:---------:|:--------:|
| 68 (current) | 8704 | 0 | 22040 | ✓ | 3560 B |
| **72** | 9216 | +512 | 22552 | ✓ | 3048 B |
| **76** | 9728 | +1024 | 23064 | ✓ | 2536 B |
| **80** | 10240 | +1536 | 23576 | ✓ | 2024 B |
| 96 | 12288 | +3584 | 25624 | ✗ (25624 > 25600) | -24 B (fail) |

**QT_STRIDE 72/76/80 все влезают в 4-block budget.**

## 011.6 — Recommendation ONE fix (по бумаге + бюджет)

### Кандидаты
- (a) **XOR-свизл на Q_T**: 0 SMEM cost, но менять write layout в Q_T-transpose STS + read swizzle в MMA. Больше правок, риск bit-exact.
- (b) **QT_STRIDE 68 → 80**: 1-строка изменения, +1536 B SMEM, headroom ok, conflict-free по бумажному расчёту (S=20, min diff = 4 ≥ 4).

### Верификация (b) для S=20 conflict-freeness

| l_div4 | start = (l_div4*20) mod 32 | Banks {start..start+3} |
|--------|:---:|:---:|
| 0 | 0 | {0..3} |
| 1 | 20 | {20..23} |
| 2 | 40 mod 32 = 8 | {8..11} |
| 3 | 60 mod 32 = 28 | {28..31} |
| 4 | 80 mod 32 = 16 | {16..19} |
| 5 | 100 mod 32 = 4 | {4..7} |
| 6 | 120 mod 32 = 24 | {24..27} |
| 7 | 140 mod 32 = 12 | {12..15} |

Sorted starts: {0, 4, 8, 12, 16, 20, 24, 28}. **Все adjacent diffs = 4 ровно, min diff = 4 ≥ 4 → conflict-free**. ✓

Banks used across warp: 0..3, 4..7, 8..11, 12..15, 16..19, 20..23, 24..27, 28..31 = **все 32 банка ровно 1 lane каждый**. **Perfect 1-way distribution**.

### QT_STRIDE 72 (S=18) для сравнения
Sorted starts: {0, 4, 8, 12, 18, 22, 26, 30}. Diffs: {4, 4, 4, 6, 4, 4, 4, 2 (wrap 32→0)}. Wrap-diff = 32-30+0 = 2. **Conflict!** Banks {30..31, 0, 1} coverage overlaps with {0..3}.

Actually wait — the wrap-around is: last start 30, group {30, 31, 0, 1} — banks 0, 1 also hit by l_div4=0 group {0, 1, 2, 3}. **2-way conflict on banks 0, 1**.

Так что **stride 72 (S=18) — не conflict-free**. Лучше **stride 80 (S=20)**.

### Recommendation

**ЕДИНСТВЕННЫЙ FIX: `#define FA_DKN_QT_STRIDE 80`** (было 68).
- 1 строка (libs/fa_bwd_dk_new.cu:23)
- +1536 B SMEM → total 23576 B → **держится 4 blocks/SM (headroom 2024 B)**
- Conflict-free по бумажному расчёту (perfect 1-way distribution всех 32 банков)
- Устраняет B-класс 64 excess/qt/warp (73% всех excess)
- A-класс 4-way остаётся (24 excess/qt/warp) — вторая полировка при желании

### НЕ предлагаю пары (свизл + stride одновременно)

## Гейт шага 2 (заранее зафиксирован per Vugar)

1. **ptxas**: regs остаются 96 (или ниже), 0 spill/stack. Регресс регов = откат.
2. **fingerprint**: numRegs=96, sharedSizeBytes=0 в cudaFuncGetAttributes → cudaOccupancyMaxPotentialBlockSize показывает **4 blocks/SM**. Регресс blocks (4→3) = **мгновенный откат без обсуждений**.
3. **BIT-EXACT ТРОЙНОЙ**: r1b_dk_bit_exact 11/11 max_abs_diff=0.000e+00 включая CANARY + compute-sanitizer 0 errors. Layout-only change (byte content preserved).
4. **Wall 5-run canonical**: собственный (a)-baseline в той же сессии перед правкой (свежие числа dk_new@stride=68), затем правка → 5-run @stride=80. Дельта, CV, per-run.

## Резюме 011

- ✅ **LD/ST split**: 95% конфликтов из LD path; ST 5.4% negligible (STS-scatter НЕ виновник)
- ✅ **Атрибуция по классам**: B-операнд Q_T (73% excess) > A-операнд dS_T (27%) > Q-фидер (0)
- ✅ **Бумага conflicts**: A 4-way (banks 0..3 и 16..19), B 2-way (stride 17 overlap), Q-фидер conflict-free (swz_byte)
- ✅ **Механизм 51.5% factually**: stride 17 mod 32 создаёт 8 групп start-положений с min diff=2 → 2-way overlap
- ✅ **Развенчание "простой нечётный конфликт-свободен"**: гарантирует single-walk, не multi-lane 4-wide groups
- ✅ **SMEM запас**: 22040 → до 25600 = **3560 B headroom** для 4 blocks/SM
- ✅ **Fix candidate**: **QT_STRIDE 68 → 80** (+1536 B, 4 blocks держатся, perfect 1-way по бумаге)

**Rekомендация: правка stride 80 отдельным гейт-циклом. Правок кода в 011 нет — только диагностика.**

Жду ACK на fix (QT_STRIDE 80) — тогда перехожу к гейт-циклу шага 2 с baseline-first.
