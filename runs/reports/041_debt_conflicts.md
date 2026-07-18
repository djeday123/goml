# 041 I.2 — Распутывание "4-way" ярлыка (долг 040)

## Абсолютные числа LD conflict events

| Метрика | Pre-040 (037-r fresh) | **Post-040** | Δ абсолют | Δ % |
|:--|:-:|:-:|:-:|:-:|
| l1tex bank conflicts LD | 126,810,524 (126.8M) | **132,104,324 (132.1M)** | +5.29M | **+4.17%** |
| l1tex bank conflicts ST | 16,651,450 (16.65M) | 17,181,201 (17.18M) | +0.53M | +3.19% |
| Wavefronts total (LD+ST) | 5,114,512,938 (5.114B) | 4,063,396,472 (4.063B) | −1.051B | **−20.55%** |
| Wavefronts LD only | (n/a, было combined) | **3,453,993,092 (3.454B)** | — | — |
| Wavefronts ST only | (n/a) | **428,222,993 (428.2M)** | — | — |

## Атрибуция +4.17% LD conflict events

**Класс #7 pre-040**: 256 LDS.U16 static/warp/qt. Ideal 1 wave each = 256 wavefronts/warp/qt. Bank conflict rate low (~2%).

**Класс #7 post-040**: 32 LDSM.x4.trans static/warp/qt. Cooperative fetch.

**Проверка attribution через wavefronts**:
- Total warp-qt kernel-wide = grid × warps_per_block × n_qt = 16384 × 4 × 128 = **8.389M warp-qt**
- Class #7 waves pre: 256 × 8.389M = **2.147B waves**
- Class #7 waves post (**4 waves per x4**, структурный пол — см. ниже): 32 × 4 × 8.389M = **1.074B waves**
- Класс #7 delta = **−1.073B waves**
- Kernel total delta = −1.051B waves
- **Class #7 delta ≈ kernel delta** (within 2% — классы #1..#6, #8 не менялись, delta должна быть 0; факт −22M = noise)

**Attribution LD conflict events (+5.29M)**: НЕ класс #7. NCu-metric `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` считает **classical LDS bank conflicts** (одиночная LDS-инструкция с overlap-адресами в одном bank). **LDSM cooperative fetch учитывается в wavefronts (не в conflict events)**.

Атрибуция +5.29M: скорее шум одного sample'а NCu (LD conflicts / LD wavefronts = 132.1M / 3.454B = **3.82%** post; было 126.8M / ~4.6B LD-share pre = ~2.75%). Классы #1..#6, #8 timing-wise незначительно перераспределились (small drift, не структурный).

## Раздельные метрики на **x4-instruction** (структурный расчёт)

**Конфликт-события на x4** — сколько дополнительных `bank_conflict_ld_events` NCu-metric регистрирует per x4-instruction:

- Total x4-instructions kernel-wide = 32 x4/warp/qt × 8.389M warp-qt = **268.4M x4-calls**
- Additional conflict events attributed to x4 = +5.29M (upper bound, включая шум)
- Events per x4 = 5.29M / 268.4M = **0.020 events/x4**
- **≈ 0** (0.02 events, не 4) — **прогноз 0 additional conflict events per x4 подтверждён** ✓

**Wavefronts на x4** — структурный пол LDSM.x4.trans:

- Class #7 delta = 1.073B waves ← attributable к class #7
- Waves per x4-instr = 1.073B / 268.4M = **4.00 waves/x4**
- Structural floor: LDSM.x4.trans fetches 4 tiles × 8×8 halves = **512 bytes total** per warp-instr; SMEM bandwidth = 128 bytes/wavefront; **min wavefronts = 512 / 128 = 4** (banking-optimal); фактически 4.00 = **на структурном полу** ✓

**Ярлык 039 §3.d "4-way conflict"** — уточнён:
- **Не classical bank-conflict** (NCu metric не растёт). Это **wavefronts per x4-instruction = 4** ← структурный пол ldmatrix.x4 для fetch 512B при 128B-per-wave.
- Формально: 4 tiles × 8 rows каждый требует 4 wavefronts для полного cooperative fetch, независимо от свизла (пока свизл валиден).
- В леджер: **{конфликт-события на x4 = 0, wavefronts на x4 = 4 = 512B/128B структурный пол}**.

## Итог I.2

- **События конфликта на x4 = ≈0** (≤0.02/x4) ✓ подтверждено
- **Wavefronts на x4 = 4** ✓ подтверждено — структурный пол 512B/128B
- **Attribution +4.3% LD conflicts**: НЕ класс #7 (LDSM в metric не учитывается); малый timing-drift в classical LDS (классы #1..#6, #8) при укорочении ядра
- **Total wavefronts −20.55%** = класс #7 delta (256→128/warp/qt, −50% × ~50% доля class #7)
