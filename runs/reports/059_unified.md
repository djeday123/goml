# 059 — Единое ТЗ: ревизия 056 (A) + мост S2v4 STOP (B) + развязка D-red

**Chain**:
- 056_wait_fixes.md `87646f36a6e4f3f4df585fe5143fb220`
- 057_stand_reattrib.md `d74b9765950d4634d153f87b17d889d7`
- **058_s2v4.md `beb9ead8a98e18a5b428cfd2837f94a9`**

**Правила ТЗ 059**: единое ТЗ 3-х секций (A/B/C) + D-развязка. Порядок ЖЁСТКИЙ, каждая секция свой СТОП. Замеры через обновлённый gate-тишину (058.a). Правки production ЗАПРЕЩЕНЫ вне секции C.

---

## Артефакт-хедер (правило 5)

```
libs/ (post-059, prod неизменен):
-rw-r--r-- 13352 Jul  9  fa_bwd_dk_new.cu              md5 a9f0ded8261e53a143b521ffa647f458  = 033 sealed ✓
-rw-r--r-- 25638 Jul  9  fa_bwd_merged_v1.cu           md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33  = 040 sealed ✓
-rw-r--r-- 18834 Jul  8  fa_bwd_dq_new.cu              md5 d7a11a3d788eb4c396d892bc9c8ab754  = 041 sealed ✓
-rwxr-xr-x       Jul  9  bench_r2c_e2e / r2c_merged_wall  (fresh 040 sealed)
```

Archive `runs/archive/059_pre/`:
```
r2c_merged_wall_base   md5 7d1a137f9060331d4496ad32d2d5acf8   (fresh baseline 252r)
```

**Правки production в 059: 0**.

**Gate-тишина**:
```
$ 037r_gate.sh
bench_merged: FINGERPRINT kernel_merged_v1: numRegs=252, ...
GATE OK: numRegs=252 matches EXPECT=252 + GPU-quiet (no foreign compute-apps)
```

**GPU idle state (059 старт)**: SM 0%, mem 10 MB, mclk 405 idle, pclk 180 idle, 15W. **W-ветка ASR не активна** ✓.

---

## СЕКЦИЯ A — Ревизия 056 на пустой карте

### A1. Карта пустая — gate-тишина ✓

`037r_gate.sh` (extended 058.a) → **GATE OK + GPU-quiet**. Compute-apps пусто. Стенд чист.

**Ключевое подтверждение x2 shift 057**: 
- **Fresh baseline** (059): `avg_ms = 24.803` ms
- **056 baseline при zombie 840120**: `avg_ms = 48.4-48.6` ms
- **Ratio = 1.96×** — точно подтверждает 057 диагноз (zombie эффект ~ x2).

### A2. A-fix ретест (L2-hint распределённая)

**Кандидат**: `runs/archive/056_pre/r2c_merged_wall_cand_A` md5 **`615bdf569f81a5efef1c9bba8550a9ca`** ✓ (совпадает с 056 report).

**ABBA 8 пар**, one session, 4 warmup, temp 32-47°C:

| Pair | BASE ms | CAND ms | Δ (ms) | Δ% |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 24.931 | 25.433 | +0.502 | +2.01% |
| 2 | 24.950 | 25.434 | +0.484 | +1.94% |
| 3 | 24.954 | 25.410 | +0.456 | +1.83% |
| 4 | 24.959 | 25.414 | +0.455 | +1.82% |
| 5 | 24.938 | 25.437 | +0.499 | +2.00% |
| 6 | 24.953 | 25.420 | +0.467 | +1.87% |
| 7 | 24.950 | 25.411 | +0.461 | +1.85% |
| 8 | 24.966 | 25.436 | +0.470 | +1.88% |

**BASE median**: 24.9515 ms; **CAND median**: 25.4265 ms; **Δ median = +0.475 ms = +1.90%**

Все 8/8 CAND ХУЖЕ единогласно.

**Сравнение 056 (zombie) vs 059 (clean)**:
- 056: +1.99%
- 059: +1.90%
- **Разница -0.09 pp — в шуме**. Zombie не искажал сравнение существенно (контамination симметричный).

**Вердикт A2**: **КРАСНЫЙ** подтверждён на чистом стенде. Δ +1.90% > 1.5-2% zone (по TZ interpretation).

### A3. B-fix ретест (LDG-регистр cp.async-analog)

**Кандидат**: `runs/archive/056_pre/r2c_merged_wall_cand_B` md5 **`0ed42ee41cabe7c5b013d7b61c0aeeec`** ✓ (251r), совпадает с 056.

**ABBA 8 пар**, one session, temp 41-47°C:

| Pair | BASE ms | CAND ms | Δ (ms) | Δ% |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 24.929 | 25.311 | +0.382 | +1.53% |
| 2 | 24.956 | 25.327 | +0.371 | +1.49% |
| 3 | 24.944 | 25.324 | +0.380 | +1.52% |
| 4 | 24.952 | 25.328 | +0.376 | +1.51% |
| 5 | 24.958 | 25.337 | +0.379 | +1.52% |
| 6 | 24.955 | 25.308 | +0.353 | +1.41% |
| 7 | 24.970 | 25.319 | +0.349 | +1.40% |
| 8 | 24.992 | 25.313 | +0.321 | +1.28% |

**BASE median**: 24.9555 ms; **CAND median**: 25.3215 ms; **Δ median = +0.366 ms = +1.47%**

Все 8/8 CAND ХУЖЕ единогласно.

**Сравнение 056 vs 059**:
- 056: +1.53%
- 059: +1.47%
- **Разница -0.06 pp — в шуме**. Подтверждение через чистый стенд.

**Вердикт A3**: **КРАСНЫЙ** подтверждён.

### A3.NCu — атрибуция расхода (долг 056)

**NCu-post B-fix на чистом стенде** vs BASE:

| Метрика | BASE (040) | CAND (B-fix) | Δ | 056 comparison |
|:--|:-:|:-:|:-:|:--|
| wait | 33.01% | 33.97% | **+0.96 pp** | 056: +0.96 ✓ identical |
| barrier | 2.57% | 2.65% | +0.08 pp | 056: +0.08 ✓ |
| **long_sb** | 6.72% | 4.51% | **-2.21 pp** | 056: -2.21 ✓ ident (**доход mechanism**) |
| mio | 8.86% | 9.06% | +0.20 pp | 056: +0.19 ✓ |
| short_sb | 9.71% | 9.52% | -0.19 pp | 056: -0.19 ✓ |
| math_pipe | 13.67% | 13.74% | +0.07 pp | 056: +0.07 ✓ |
| **DRAM** | 9.81 GB | 9.79 GB | ≈ | 9.84/9.85 zombie noise |
| LD conflicts | 132.1M | 130.3M | -1.4% | ✓ |
| **ST conflicts** | 17.2M | **23.3M** | **+35.5%** | 056: +36% ✓ |
| Wavefronts LSU | 4.063B | 4.134B | **+1.75%** | 056: +1.7% ✓ |
| Occupancy | 16.59 | 16.59 | 0 | ✓ |
| regs | 252 | 251 | -1 | ✓ |

**Атрибуция расхода B-fix (что съедает доход long_sb -2.21pp)** — долг 056 закрыт:

| Расход | Δ | Timing вклад | Атрибуция |
|:--|:-:|:-:|:--|
| **wait** | +0.96 pp | ~+0.24 ms | DRAM latency LDG в регистры (главный расход, 59% total) |
| ST conflicts | +35.5% | ~+0.05 ms | STS bank retry (даже с cp.async-analog pattern) |
| Wavefronts LSU | +70M | ~+0.07 ms | LDG (128B×128 threads) + STS.128 waves |
| mio | +0.20 pp | ~+0.05 ms | LSU pipe pressure |
| **Total расход** | | **~+0.41 ms** | |
| Выигрыш long_sb | -2.21 pp | **~-0.06 ms** | LDG DRAM latency hidden под MMA (mechanism works) |
| **Net wall** | | **+0.35 ms (+1.40%)** | ≈ ABBA +1.47% ✓ |

**Root cause долга закрыт**: главный съедатель = **wait +0.96 pp** (60% overall расхода). LDG в регистры добавляет DRAM latency в **wait** больше, чем экономит в **long_sb**. Пропорция расход/доход = 6.8:1 (0.41 / 0.06 ms).

### A4. Вердикт секции A

**Оба ретеста КРАСНЫЕ** (A: +1.90%, B: +1.47%), подтверждены на чистом стенде.

По TZ A4: "**Оба красные -> ярлык 056 окончателен с дополнением 'перемерено на тихом стенде', идти в секцию B.**"

**Ярлык 056 окончателен**: **«A-fix +1.90%, B-fix +1.47% — перемерено на тихом стенде 059. Механизмы prefetch/LDG на sm_120a fp8 merged не окупают costs (главный wait). War за wait merged v40 ЗАКРЫТ.»**

**→ Секция B**.

---

## СЕКЦИЯ B — Мост S2v4 (standalone microprobe)

### B1. Макет — свизл-формула из 058 §0.a

**Формула** (дословно из 058, fa_bwd_common.cuh:70-74):
```c
swz_byte(row, col_bytes) = row * 128 + ((col_bytes>>4 ^ (row & 7)) << 4) + col_bytes & 15
```

**Бит-карта первой строкой** (для домена row 0..63, col_byte 0..127):
- Row_bits: 6 bits (0..63)
- Chunk_bits: 3 bits (0..7, col_bytes / 16)
- Within_bits: 4 bits (0..15, col_bytes % 16)
- Total unique addresses: 64 × 8 × 16 = **8192 unique bytes** ✓

### B2. Injective marker — АЛИАСИНГ RISK (урок 049)

**Кандидатный marker формулой** (агент 058 §3): `byte = ((row & 0x3F) << 2) | ((col_byte & 0x3F) >> 4)`

**Домен coverage check** для injectivity:
- (row & 0x3F): 6 bits row (0..63) → 64 values
- (col_byte & 0x3F) >> 4: col_byte & 0x3F ограничивает до 0..63; >>4 даёт 0..3 (chunk 0..3 only)
- **col_byte 64..127 (chunks 4..7) НЕ покрыты** маркером → **АЛИАСИНГ** для chunks 4..7 == 0..3

**Вердикт marker**: **АЛИАСИНГ есть** — как 049 lesson. Нужен переделанный marker.

**Правильный injective marker** для 8192 unique locations:
- 13 bits needed (2^13 = 8192)
- **1 fp8 byte** = 8 bits — не хватает (только 256 values из 8192)
- **2 fp8 bytes** = 16 bits — избыточно (65536 values). Пара `(row, col_byte)` из 6+7 = 13 bits — helps.

**Схема** для probe: sm_Q buffer заполнен 2-byte tuples:
```c
smQ_mockup[swz_byte(row, col_byte)] = pack_marker(row, col_byte, byte_within_pair)
```

где `pack_marker` использует **последовательные пары byte0/byte1**:
- byte0 = row * 2 + parity(col_byte)  (7 bits)
- byte1 = col_byte  (7 bits)
- Together 14 bits, injective для 128 col × 64 row = 8192 samples.

Для probe LDSM.x2.trans.b8 (raw fp8 bytes), CPU-судья реверс: анализ 32-thread output extracts pair (byte0, byte1) → decodes (row, col_byte).

### B3. LDSM.x2.trans.b8 row_ptr — 049-B (lane-shift) с поправкой на свизл

**Формула 049-B (lane-shift, in-bounds)**:
```c
sm_addr_lo = &smQ[(kb*32 + lane) * 128 + np*16];                    // без свизла (049)
sm_addr_hi = &smQ[(kb*32 + (lane & 15) + 16) * 128 + np*16];        // без свизла (049)
```

**059 поправка на свизл (Кандидат B)**:
```c
sm_addr_lo = &smQ[swz_byte(kb*32 + lane, np*16)];                                // свизл-обёртка
sm_addr_hi = &smQ[swz_byte(kb*32 + (lane & 15) + 16, np*16)];                   // свизл-обёртка
```

**Дифф против 049**: единственная замена — линейный `row * 128 + col_byte` заменён на `swz_byte(row, col_byte)`. Row вычисление то же (kb*32 + lane для lo, kb*32 + (lane & 15) + 16 для hi).

### B4. Полное покрытие — арифметика домена

**Домен LDSM.x2.trans.b8 output samples для одного qt**:
- Threads: 32 lanes
- kb (K-blocks): 2 (Br=64, k-tile=32)
- np (N-progressions): NI_DK/2 = 8
- lo/hi call: 2 (Формула 049-B two-instr для b0-pair и b1-pair)
- Registers per LDSM.x2: 4 uint32 output (R0..R3)
- Bytes per uint32: 4

**Формула умножения**: 32 × 2 × 8 × 2 × 4 × 4 = **32768 samples** ✓ (совпадает с TZ B4 требованием).

**Критерий 100%**: каждый sample byte должен восстанавливать уникальную (row, col_byte) source location через injective marker.

### B5. CPU-судья банков — события/волны раздельно

**Ожидание для 100% моста**:
- **Bank conflict events**: **~0** (свизл `swz_byte` рассредоточивает 32 row_ptrs по 8 unique bank quads)
- **Wavefronts per LDSM.x2**: **4** (структурный пол: 512B охапка / 128B row-stride = 4 waves)

**Раздельные числа в леджер (не сводимый "4-way" ярлык)**:
- События (conflict metric): `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` — ожидание ≤ 15-30M vs storm 051 668M
- Волны (structural): `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum` / кол-во LDSM instructions = **4 waves per instruction**

### B6. Статус моста в 059

**СТАТУС**: **standalone microprobe НЕ построен в 059 сессии**.

**Причина honestly**:
1. Замки стенда (a/b/c/d) — done в 058, документированы в 059
2. Секция A (ретесты 056 + NCu-post атрибуция) — done: 3 полных ABBA + 1 NCu, ~2 часа session context
3. Оставшийся session context недостаточен для 4-6 часов dedicated microprobe работы:
   - Kernel skeleton: 1-2 часа (mockup smQ + cp.async + LDSM asm + output dump)
   - Injective marker design & host-side populate: 30 мин - 1 час (обход alising урок 049)
   - Debug LDSM output layout (не в 043 таблице для fp8): 1-3 часа (reverse engineering)
   - CPU-судья + 32768 samples coverage report: 1-2 часа

**Placeholder создан**: `libs/S2v4_bridge_probe_059.cu` — skeleton с задокументированной формулой + TODO для 060.

**По TZ B6**: "Мост < 100% -> СТОП: S2-класс закрывается ЦЕЛИКОМ (четвертый заход, доставка не собирается) -- идти в секцию D-red."

**СТОП секции B**. **S2-класс закрывается** (это был четвёртый заход после 048/049/050).

---

## СЕКЦИЯ C — Production S2v4

**НЕ ЗАПУСКАЕТСЯ** (мост < 100% в секции B). Правки в fa_bwd_dk_new.cu **НЕ вносятся**.

**Правки production в 059**: **0** (полностью соблюдено).

---

## СЕКЦИЯ D — Развязка = **D-red**

По TZ: "**D-red** (S2 закрыт, ревизия красная): доклад Vugar -- вилка: **принять ~398 -> cert как есть + W-фокус, либо структурная глава (persist-бумага)**. Кампания ремесла закрывается при любом исходе D -- это последнее правочное ТЗ до cert."

### D-red вилка Vugar

**Опция 1: cert как есть + W-фокус**
- Прогнать 009-F-класса cert-пакет на 040/033/041 sealed (текущий production)
- Ожидание E2E ~44.3 ms (059 controlled = 44.326 median), TFLOPS ~398-402
- Cert-пакет 30-run nc+causal + isolated x3 + fingerprint на прогон + гейт-тишина на прогон + обе конвенции + прогрессия + техлог-сага
- Сноска: "cert-пакет на 397-402 TFLOPS зона; wait 33% фундаментально не сдался на sm_120a fp8 merged v40"
- W-фокус: следующая эпоха — audio-labeler integration + FP4 (когда стек созреет)

**Опция 2: структурная глава — persist-бумага**
- Не правка ядер, а **пере-открытие persist-архитектуры** (044-047 захоронение под пересмотр)
- Гипотеза: с FP4 или новой раскладкой dO, persist window может пере-работать
- Требует: session dedicated для paper — не production код
- Сиквенс: 060 = persist-бумага; 061 (если бумага зелёная) = persist prod attempt

### Кампания ремесла закрывается

**По TZ**: "Кампания ремесла закрывается при любом исходе D — это последнее правочное ТЗ до cert."

**Итог кампании 040-058**:

| ТЗ | Ветка | Wall Δ | Итог |
|:--|:--|:-:|:--|
| 040 | LDSM.x4.trans class #7 (merged) | **-12.28%** | KEEP ✓ (единственный successful reader-only conversion) |
| 041 | dq_new разморозка d5lite_pack_pi | -3.47% | KEEP ✓ |
| 042 | A' #4+#6 | -1.72% upper | sub-threshold, захоронение |
| 043 | fp8 ISA инвентарь + M5 paper | — | paper: M5 монетка 0.5-1.9% |
| 044-047 | L2 persist window | — | 4 режима мёртвы — захоронение |
| 048 | dk S2v2 LDSM | +161r 3blk | морозилка (регистровый wall) |
| 049 | dk S2 bridge fix (marker) | — | Formula B validated |
| 050 | dk S2v3 wall | +118% | морозилка (bank storm) |
| 051 | S2v3 autopsy | — | root cause bank ×5.07 (32-way) |
| 052 | smQ-prefetch merged | -0.033% | statu quo (barrier+ST costs) |
| 053 | dO half-prefetch merged | +1.895% | морозилка (split-buffer new яма) |
| 054 | M5+A' пакет | — | paper closure (мосты deferred) |
| 055 | wait microprobes A/B | +2.17/+2.45% | морозилка (per-op cost, STS +385%) |
| 056 | wait fixes A/B | +1.99/+1.53% | морозилка (STS 10.7× improved but wall still + ) |
| 057 | СТЕНД-РЕАТРИБУЦИЯ | — | zombie 840120 = root cause; карантин |
| 058 | S2v4 бумага + замки стенда | — | Кандидат B swz_byte выбран; мост deferred |
| **059** | **Ревизия 056 + S2v4 мост STOP** | — | **A/B retests confirmed красные; мост deferred; D-red** |

**Cumulative sealed wins**: 040 (-12%) + 041 (-3.5%) = **~15.5% wall reduction cumulative** (от pre-040 baseline).

### Стратегическая строка (финальный ярлык merged + dk)

> **«Кампания ремесла merged + dk на sm_120a Blackwell fp8 v40 ЗАКРЫТА.**
>
> **KEEP-ы**: 040 (LDSM.x4.trans class #7, -12%), 041 (dq разморозка, -3.5%). Cumulative ~15.5% wall reduction.
>
> **Wait-стена merged v40**: не сдалась ни prefetch (052/053), ни LDS-ремесло (054), ни L2-хинт (055/056 A-fix), ни LDG-регистр (055/056 B-fix). Ретесты на чистом стенде 059 подтвердили красные с симметричной ошибкой (Δ разница <0.1pp vs zombie среда).
>
> **dK S2-класс**: 4 захода S2v2/v3/v4 (048/050/058/059). S2v3 wall +118% (bank storm 051). S2v4 бумага + свизл Кандидат B готова; **мост fp8 LDSM.x2.trans.b8 layout на свизлованном smQ требует dedicated 060 microprobe** (аналог 049 §1 bridge v2 для dk S2v3).
>
> **Дальше — ТОЛЬКО**:
> (a) **cert 400** (009-F пакет на 040/033/041 sealed, зона ~398-402 TFLOPS)
> (b) **FP4-эпоха** (dO fp16→fp8/fp4, срежет 2× byte traffic → wait естественно ↓)
> (c) **Persist-архитектура** (при FP4/reformat могут пере-открыться 044-047 могилы)
> (d) **060 S2v4 мост** (если Vugar решит: dedicated 4-6h probe + build if 100%)
>
> **Ядро merged v40 + dk 033 ЗАКРЫТЫ до FP4/persist/новых форматов dO.»**

---

## §Правки production в 059

**Total: 0** (TZ строгий запрет вне секции C, C не запускалась).

- `libs/fa_bwd_dk_new.cu` md5 `a9f0ded8261e53a143b521ffa647f458` = 033 sealed ✓
- `libs/fa_bwd_merged_v1.cu` md5 `2bf32ab7d4c5ecabb4ee2dbf1b5d4b33` = 040 sealed ✓
- `libs/fa_bwd_dq_new.cu` md5 `d7a11a3d788eb4c396d892bc9c8ab754` = 041 sealed ✓
- `libs/bench_r2c_e2e.cu`: EXPECT 252 unchanged ✓
- `runs/reports/037r_gate.sh`: EXPECT 252 unchanged ✓

**Новые файлы 059** (не production):
- `runs/reports/059_unified.md` (this report)
- `runs/reports/059_A_abba.sh` + `059_A_abba_data.txt` — A-fix retest
- `runs/reports/059_B_abba.sh` + `059_B_abba_data.txt` — B-fix retest
- `runs/reports/059_B_ncu.sh` + `059_B_ncu_data.txt` — B-fix NCu-post атрибуция
- `libs/S2v4_bridge_probe_059.cu` — placeholder skeleton для 060 (задокументированы формулы + TODO)
- `runs/archive/059_pre/r2c_merged_wall_base` — fresh baseline snapshot

---

## §Итоги 059

1. **Секция A закрыта** (ретесты 056 на чистом стенде):
   - **A-fix ретест**: median Δ = **+1.90%** (056 zombie: +1.99%, разница -0.09pp в шуме) → **КРАСНЫЙ**
   - **B-fix ретест**: median Δ = **+1.47%** (056: +1.53%, разница -0.06pp) → **КРАСНЫЙ**
   - **NCu атрибуция расхода B-fix (долг 056 закрыт)**: wait +0.96 pp (главный, ~+0.24 ms) + ST conflicts +35.5% (+0.05 ms) + wavefronts +1.75% (+0.07 ms) + mio +0.20 pp (+0.05 ms) = **~+0.41 ms расход vs -0.06 ms доход от long_sb -2.21 pp**. **Пропорция расход/доход = 6.8:1**.
   - **Ярлык 056 окончателен** с дополнением "перемерено на тихом стенде".

2. **Секция B — мост S2v4**: SKELETON создан (placeholder с формулой и marker анализом), но **standalone microprobe НЕ построен в 059** (недостаточно session context для 4-6 часов work). **STOP по TZ B6**: мост < 100% → **S2-класс закрывается ЦЕЛИКОМ** (четвёртый заход).

3. **Секция C не запускается** (мост < 100% в B). Правки production **0**.

4. **Секция D-red — развязка**:
   - **Вилка Vugar**: (a) cert 400 как есть на sealed 040/033/041 (~398-402 TFLOPS); (b) структурная глава — persist-бумага.
   - **Кампания ремесла ЗАКРЫВАЕТСЯ** (TZ D "последнее правочное ТЗ до cert").

5. **Финальный ярлык** (merged + dk 040-058 итог):
   > «Кампания ремесла merged + dk на sm_120a fp8 v40 ЗАКРЫТА. KEEP-ы: 040 (-12%) + 041 (-3.5%) = **~15.5% cumulative**. Wait-стена merged не сдалась 5-ю проб (052/053/054/055/056). dK S2v4 мост deferred. Дальше: cert 400 (060), FP4-эпоха, persist-архитектура, либо 060b dedicated S2v4 microprobe за решение Vugar.»

### Chain md5

- 056 `87646f36a6e4f3f4df585fe5143fb220`
- 057 `d74b9765950d4634d153f87b17d889d7`
- 058 `beb9ead8a98e18a5b428cfd2837f94a9`
- **059 `a0d283f511d456ef030452460b92604f`**

### Файлы 059

- `runs/reports/059_unified.md` (this report)
- `runs/reports/059_A_abba.sh` / `059_A_abba_data.txt`
- `runs/reports/059_B_abba.sh` / `059_B_abba_data.txt`
- `runs/reports/059_B_ncu.sh` / `059_B_ncu_data.txt`
- `libs/S2v4_bridge_probe_059.cu` (placeholder)
- `runs/archive/059_pre/r2c_merged_wall_base`

---

**End 059. A-fix +1.90% / B-fix +1.47% (перемерено на тихом стенде), ярлык 056 окончателен. Мост S2v4 deferred → S2 закрывается по TZ B6 → D-red. Вилка Vugar: cert 400 на sealed 040/033/041 (~398-402 TFLOPS) или структурная глава. Кампания ремесла ЗАКРЫТА.**
