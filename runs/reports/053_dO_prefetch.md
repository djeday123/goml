# 053 — dO half-prefetch merged: обходы ям 052 работают, split-buffer сам новая яма → откат + war-close

**Chain**:
- 050_dkS2_v3.md `9269bcf280c5ebf1a67aa9e91bbd3933`
- 051_s2v3_autopsy.md `bd9ea399697ffe9f6ff206618c48a36d`
- 052_smq_prefetch.md `274d9c70af8aebf66f1815779a72a7a1`

**Правила ТЗ 053**: одна структурная правка (dO half-prefetch с обходами двух ям 052) + полный гейт **ВКЛЮЧАЯ racecheck**. Production 033/040/041 sealed.

---

## Артефакт-хедер (правило 5)

```
libs/ (post-053):
-rw-r--r-- 25638 Jul  9         fa_bwd_merged_v1.cu           (md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33 = 040 sealed) ✓
-rw-r--r-- 13352 Jul  9         fa_bwd_dk_new.cu              (md5 a9f0ded8261e53a143b521ffa647f458 = 033 sealed) ✓
-rw-r--r-- 18834 Jul  8         fa_bwd_dq_new.cu              (md5 d7a11a3d788eb4c396d892bc9c8ab754 = 041 sealed) ✓
-rwxr-xr-x       Jul  9         bench_r2c_e2e                 (post-053 rollback, fingerprint 252/128/69 OK)
```

**Diagnostic binaries** в `runs/archive/053_pre/`:
```
-rwxr-xr-x r2c_merged_wall_base   md5 3d4ce72183fe0a3a757457f15c06c89f   (040 252r)
-rwxr-xr-x r2c_merged_wall_cand   md5 4ac22431851a18498f56973eeee173f4   (053 254r dO half-prefetch)
-rw-r--r-- fa_bwd_merged_v1.cu.053_cand   md5 a0b4b1905ff39aba2cd597e0fbca3012  (candidate source)
```

**Gate-log** (post-rollback):
```
$ 037r_gate.sh
bench_merged: FINGERPRINT kernel_merged_v1: numRegs=252, sharedSizeBytes=0, ...
GATE OK: numRegs=252 matches EXPECT=252
```

---

## §0. Бумага (SMEM-баланс + обходы ям 052)

### §0.a Фазовая карта dO-стрима + SMEM-баланс

**Живость dO в qt-шаге** (16384B = Br*Hd*sizeof(__half)):

| Фаза qt | Использование smdO | Барьер после |
|:--|:--|:--|
| **Step A** cp.async Q + dO + L/D | **Writer smdO** (cp.async 16KB) | t3 |
| **Step B** MMA-A Q·K^T | idle (smQ used) | — |
| **Step C** softmax (regs) | idle | — |
| **Step D** MMA-B dO·V^T | **Reader smdO** (fp8→fp16 CVT + A0..A3 fragments) | t_new1 |
| **Step E** STS smdS_stage | idle | t9 |
| **Step F** STG.128 drain dS_nat | idle | t_new2 |
| **Step G** STS smP_T | idle | t11 |
| **Step H** MMA_dV P^T·dO (LDSM.x4.trans) | **Reader smdO** | t13 |

**dO живёт весь qt**: Writer Step A → Reader Step D → Reader Step H → t13 → next qt Writer. **Ни одной мёртвой фазы**.

**SMEM-баланс**:
- Base 040: 41472B (smK+smV+smQ_region+smdO+smL+smD)
- Threshold 2 blk/SM: `(smem + 1024) * 2 ≤ 102400 → smem ≤ 50176`
- Headroom: `50176 - 41472 = 8704 B` ← совпадает с формулировкой ТЗ

**Механики (три обходов SMEM-затыка)**:

| Механика | Формулировка | Оценка | Вердикт |
|:--|:--|:-:|:--|
| **(i) alias на мёртвые фазы union** | dO 16KB в union {smQ/smdS_stage/smP_T} | dO нет мёртвой фазы 16KB | **режется структурно** |
| **(ii) полу-нога 8KB** | first-half prefetch, second-half в Step A | 41472+8192=49664 ≤ 50176, 2 blk ✓; upper 50% latency | **выбрана** |
| **(iii) отказ от Q-стейджа как отдельной ноги** | Q в union уже; убрать Q → union free для dO alias | Q в union уже (не отдельная нога); отказ Q даёт лишь 8KB (union) не 16KB | не даёт 16KB для полной ноги dO |

**Выбор: (ii) полу-нога 8KB** — единственная работоспособная механика по SMEM-бумаге.

**Split-buffer readers для (ii)**:
- Step D читает smdO по `m_lo = wid*16 + l_div4`. warp 0/1 → rows [0..31], warp 2/3 → rows [32..63]. **Split по warp: wid<2 → first-half; wid≥2 → second-half. `m_rel = m & 31`.**
- Step H читает smdO по `k_row = kb*16 + row_in_tile + (tile_id&1 ? 8 : 0)`. kb=0/1 → k_row [0..31] (first-half); kb=2/3 → k_row [32..63] (second-half). **Split по kb: kb<2 → first; kb≥2 → second. `k_row_rel = k_row & 31`.**

Bit-exact invariant (проверено): `(m_rel & 7) = (m & 7) = l_div4` — swizzle unwind корректно на обеих ногах.

### §0.b Обход ямы-1 (barrier +2.29 из 052) — cp.async-commit-group scheme

**052 diagnosis**: `cpa_wait<0>` перед `__syncthreads()` t13 → barrier +2.29pp (thread synchronization stall от cp.async completion).

**Обход**:
- Prefetch cp.async группа НЕ waited перед t13.
- В Step A qt+1: `cpa_commit()` для current qt loads (второй group), затем `cpa_wait<0>()` waits **ОБА** groups (prev prefetch + current).
- `__syncthreads()` t3 syncs threads.
- Barrier t13 = чистый __syncthreads без cp.async ожидания.

**Address sets всех 6 барьеров** (метод 021):

| Barrier | Waits on | Address set | 053 изменение? |
|:--|:--|:--|:--|
| t3 (post-loads) | dO_second cp.async + Q cp.async + L/D + **prefetch из prev qt** | smdO_second, smQ, smL, smD, smdO_first_curr | dO_second only writer + prefetch prev-qt дождан **тут** |
| t_new1 (pre-scatter) | smQ reads завершены → alias overlay | smQ_region | ✓ (unchanged) |
| t9 (pre-drain) | smdS_stage STS завершены | smdS_stage | ✓ (unchanged) |
| t_new2 (post-drain) | STG drain завершён | smdS_stage читатель Step F | ✓ (unchanged) |
| t11 (pre-MMA_dV) | smP_T STS завершены | smP_T | ✓ (unchanged) |
| t13 (end qt) | dV_acc обновлены; **prefetch outstanding НЕ ждётся** | smdO_first_next (outstanding) | **обход: cpa_wait НЕ здесь**, дождётся в Step A qt+1 |

**Racecheck-фокусы**:
1. Prefetch destination smdO_first_next ≠ smdO_first_curr (ping-pong disjoint) — reader qt qt+1 читает smdO_first_curr (загруженный prefetch prev qt) → GUARANTEED через cpa_wait<0> в Step A qt+1.
2. cp.async issue после Step H (все LDSM reads завершены) — не конкурирует с shared reads.
3. t13 barrier syncs threads (dV_acc updates ready) но НЕ ждёт cp.async — safe потому что Step B qt+1 (reader) находится ЗА cpa_wait<0>.

### §0.c Обход ямы-2 (ST-conflicts +123% из 052) — CPU-судья ДВУХ фаз

**052 diagnosis**: concurrent cp.async STS → smQ_next_buf параллельно с Step E/G STS → smQ_curr = **+123% ST conflicts**.

**CPU-судья bank ДВУХ фаз для 053 dO prefetch** (свод 052):

**Адресная фаза** (row_ptr всех 32 лейнов по банкам):
- prefetch destination smdO_first_next 8KB, base-aligned.
- cp.async chunk stride 16B (CHUNK) × 8 rows_first × 32 threads.
- Row-stride в smdO_first_next = Hd*2 = 256B = 64 banks × 4B.
- 256B / 4B = 64 banks — **not** kратно 32 lane_count × 4 (like 128B was).
- Lane l (l ∈ 0..31) row = l/16 (2 rows per warp iteration in cp.async loop), col_chunk = (l % 16) * 16.
- Address = smdO_first_next_base + row*256 + col*16 = base + (l/16)*256 + (l%16)*16.
- Bank(l) = (base + (l/16)*256 + (l%16)*16) / 4 mod 32.
- (l/16)*256/4 = (l/16)*64 mod 32 = 0.
- (l%16)*16/4 = (l%16)*4 mod 32 = (l%16)*4 mod 32.
- l=0..15: bank = 0, 4, 8, 12, 16, 20, 24, 28, 0, 4, 8, 12, 16, 20, 24, 28.
- Аналогично l=16..31.
- **8 banks unique × 4 hits per bank = 4-way conflict per cp.async chunk (не 32-way шторм)**.

**Данная фаза** — cp.async LDGSTS.128 = 16B per transfer, distributed via HW pipeline. Bank pattern determined by writer's chunk destination (аналогично 052).

**Обход**: **отсрочка cp.async issue до конца qt после Step G STS**. Тогда prefetch STS **serialized** с core kernel STS (не concurrent). Ожидание: **ST conflicts +0..+30%** (vs 052 +123%).

### §0.d Именованные предсказания NCu-post

| Метрика | BASE (040) | Прогноз | Механизм |
|:--|:-:|:-:|:--|
| **wait** | 33.00% | **-3..-8 pp** (25..30%) | Half dO 8KB → -50% dO latency (16KB→8KB effective) → wait −доля × 50% |
| **barrier** | 2.57% | **+0..+0.5 pp** (052: +2.29) | Обход-1: no cpa_wait перед t13 → barrier не растёт |
| **ST conflicts** | 17.1M | **+0..+30%** (052: +123%) | Обход-2: prefetch issue после Step G → serialized |
| long_sb | 6.72% | ~4-5% (-2..-3 pp) | half of dO DRAM latency hidden |
| mio | 8.86% | +0..+1 pp | extra cp.async issues |
| short_sb | 9.71% | ~10% (знак ±0.5 pp) | LDS chain unchanged |
| DRAM | 9.85 GB | **9.79 ровно** | тот же volume |
| blocks | 2 | **2 ровно** | SMEM 49664 < 50688 |
| regs | 252 | не предсказан (потолок 256) | ptxas факт |

---

## §1. Правки kernel + launcher (8 hunks)

1. Header comment — обновление SMEM плана + описание half-prefetch mechanism
2. SMEM layout: `smdO` 16KB → `smdO_first_A` (8KB) + `smdO_first_B` (8KB) + `smdO_second` (8KB)
3. Прелюдия (в K/V warmup block): **prefetch dO[qt_start] first-half** cp.async → smdO_first_leg (по `qt_start & 1`); wait в prelude commit
4. Step A: **remove full dO cp.async**; keep Q + L/D + only **dO SECOND half** (rows 32..63) → smdO_second; `cpa_wait<0>()` waits current + previous prefetch. Ping-pong pointers `smdO_first_curr` и `smdO_first_next`
5. Step D reader: split by wid (wid<2 → `smdO_first_curr`, wid≥2 → `smdO_second`), `m_lo_rel = m_lo & 31`
6. Step H reader: split by kb (kb<2 → `smdO_first_curr`, kb≥2 → `smdO_second`), `k_row_rel = k_row & 31`
7. **End-of-qt (after Step H, before BARRIER t13)**: cp.async prefetch first-half dO[qt+1] → smdO_first_next; `cpa_commit()` **без wait** (waited в Step A qt+1)
8. Launcher `smem_bytes`: 41472 → 49664 (+8192 = smdO_first_B leg)

**bench_r2c_e2e.cu EXPECT**: 252 → 254 (обновлено осознанно с комментарием, потолок 256 ✓).

---

## §2. Гейт строго по порядку

### §2.a ptxas-факт (потолок 256r, spill=0, blocks=2)

```
ptxas info: Compiling entry function 'kernel_merged_v1' for 'sm_120a'
ptxas info: Function properties: 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info: Used 254 registers, used 1 barriers
```

**ЗЕЛЁНЫЙ**: 254r (+2r vs 252 base) — split-buffer address ALU + ping-pong pointers. Потолок 256r ✓ (люфт 2r). 0 spill/stack. SMEM 49664 → `floor(101376 / (49664+1024)) = 2` blocks ✓.

### §2.b Fingerprint gate

```
$ 037r_gate.sh (EXPECT=254)
bench_merged: FINGERPRINT kernel_merged_v1: numRegs=254, sharedSizeBytes=0, ...
GATE OK: numRegs=254 matches EXPECT=254
```

**ЗЕЛЁНЫЙ**. EXPECT обновлён осознанно (252→254 с комментарием "053: +2r vs 252 base — dO half-prefetch split readers + ping-pong").

### §2.c Корректность (bit-exact + racecheck)

- `r2c_merged_bit_exact`: **11/11 форм BIT-EXACT** (max_abs_diff=0.000e+00) ✓
- Chain **x3 runs**: 33/33 PASS ✓
- **INJECT_BITFLIP=1 control**: 0/11 (dS_nat mism=1 caught) ✓
- **compute-sanitizer memcheck**: 11/11 + `ERROR SUMMARY: 0 errors` ✓
- **compute-sanitizer racecheck** (правило 13, барьеры двинулись + новый cp.async group):
```
=== SUMMARY ===
  forms triple-bit-exact: 11 / 11
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```
**ЗЕЛЁНЫЙ**: 0 hazards на канон + canary + все 11 форм.

### §2.d Wall ABBA >= 8 пар — КРАСНЫЙ (правило-2/3 v2)

**Стенд**: 4 warmup + 16 timed (8 pairs), одна сессия, ABBA pattern `[BASE,CAND,CAND,BASE]×4`. temp 43-47°C.

| Pair | BASE ms | CAND ms | Δ (ms) | Δ% |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 48.445 | 49.366 | +0.921 | +1.902% |
| 2 | 48.455 | 49.367 | +0.912 | +1.882% |
| 3 | 48.462 | 49.372 | +0.910 | +1.878% |
| 4 | 48.464 | 49.518 | +1.054 | +2.175% |
| 5 | 48.472 | 49.522 | +1.050 | +2.166% |
| 6 | 48.478 | 49.512 | +1.034 | +2.133% |
| 7 | 48.461 | 49.372 | +0.911 | +1.880% |
| 8 | 48.458 | 49.387 | +0.929 | +1.917% |

**BASE median**: 48.4615 ms
**CAND median**: 49.3795 ms
**Δ median = +0.918 ms = +1.895%**

**Вердикт правило-2/3 v2**: CAND **МЕДЛЕННЕЕ** в **8/8 пар единогласно**, амплитуда ~+1.9% → **КРАСНЫЙ ← ОТКАТ (кандидат хуже)**.

### §2.e NCu-post — сверка предсказаний §0.d поименно

| Метрика | BASE | Прогноз | **CAND факт** | Δ | Вердикт |
|:--|:-:|:-:|:-:|:-:|:--|
| **wait** | 33.00% | ↓ 3-8 pp (25..30%) | **36.76%** | **+3.76 pp ↑** | **ОПРОВЕРГНУТ (↑ не ↓)** |
| **barrier** | 2.57% | +0..+0.5 pp (обход-1) | **2.92%** | **+0.35 pp** | **ОБХОД-1 РАБОТАЕТ** (vs 052 +2.29) ✓ |
| **ST conflicts** | 17.1M | +0..+30% (обход-2) | **18.1M** | **+5.6%** | **ОБХОД-2 РАБОТАЕТ** (vs 052 +123%) ✓ |
| LD conflicts | 132.1M | ≈ | 134.6M | +1.9% | ≈ ✓ |
| long_sb | 6.72% | -2..-3 pp | 4.20% | **-2.52 pp** | **prefetch mechanism работает** ✓ |
| mio | 8.86% | +0..+1 pp | 6.08% | **-2.78 pp** | ↓ противоположно (LSU distribution) |
| **short_sb** | 9.71% | ±0.5 pp | **10.98%** | **+1.27 pp** | **↑ split-buffer address ALU** |
| math_pipe | 13.66% | не предсказан | 11.47% | -2.19 pp | ↓ |
| **Wavefronts LSU** | 4.063B | не предсказан | **4.201B** | **+3.4%** (+138M) | extra LDS от split-buffer |
| DRAM | 9.85 GB | 9.79 ровно | 9.86 GB | ≈ | ✓ |
| Occupancy | 16.59% | 16.59 ровно | 16.59% | 0 | ✓ |
| regs | 252 | не предсказан | 254 | +2 | ✓ (в потолке 256) |

**Root cause (§e картину)**:

**Обходы TZ работают чисто**:
- Обход-1 (barrier): 052 был **+2.29pp**, 053 **+0.35pp** → **обход cpa_wait до Step A qt+1 работает** ✓
- Обход-2 (ST-conflicts): 052 был **+123%**, 053 **+5.6%** → **обход serialize prefetch после Step G STS работает** ✓
- LD conflicts +1.9% ≈ noise ✓

**Механизм prefetch работает механически**:
- long_sb **-2.52pp** (DRAM latency половины dO прячется под MMA)
- mio **-2.78pp** (LSU pipe distribution)
- math_pipe **-2.19pp** (меньше pressure)

**НО**: **split-buffer readers сами добавили НОВУЮ яму**:
- **short_sb +1.27pp** — address computation (`m_rel = m & 31`, `k_row_rel = k_row & 31`, choose `smdO_ptr`) добавляет short-scoreboard latency chain
- **Wavefronts LSU +3.4%** (+138M waves) — extra LDS wavefronts от split reader address dispatching
- **wait +3.76pp ↑** — комбинация LSU pressure от split + доп cp.async issues
- **+2 registers** — split logic + ping-pong pointer

**Куда делись спрятанные такты**:
- **Механически**: prefetch скрыл ~половину dO DRAM latency → расчётный выигрыш ~-0.5 ms
- **Реализовано**: split-buffer overhead + address ALU stall > выигрыша → net **+0.918 ms (+1.895%) ХУЖЕ**
- **Root cause**: `smdO_ptr = (wid<2) ? smdO_first_curr : smdO_second` (Step D) и `(kb<2) ? ... : ...` (Step H) добавляют предикатную выборку указателя перед каждой LDS/LDSM.

---

## §3. Вердикт-карта: 052 vs 053 (какая яма съела на этот раз)

| Компонент | 052 (Q ping-pong) | 053 (dO half-prefetch) | Комментарий |
|:--|:-:|:-:|:--|
| **Mechanism DRAM hidden** | long_sb -1.99pp | long_sb -2.52pp | 053 hides больше DRAM (half dO = 8KB > full Q = 8KB) |
| **Barrier cost** | **+2.29pp** ← яма-1 | +0.35pp | **053 обход-1 работает** |
| **ST conflict cost** | **+123%** ← яма-2 | +5.6% | **053 обход-2 работает** |
| **Split-buffer address ALU** | 0 (не split) | **short_sb +1.27pp** ← новая яма | 053 сам породил |
| **Wavefronts LSU** | +0.6% | **+3.4%** ← новая яма | split-buffer |
| **wait Δ** | +0.41pp | **+3.76pp** ← ↑ | 053 хуже |
| **Wall Δ** | -0.033% | **+1.895%** | 053 хуже |
| **Verdict** | statu quo/откат | **откат (красный)** |  |

**Ключевое отличие**: 052 обходов не имел → пал на barrier + ST conflicts (mechanism partial-work). 053 обходы TZ применил → **пал на split-buffer address overhead** (новая яма).

---

## §4. Стратегическая строка — WAR ЗА WAIT В MERGED ЗАКРЫВАЕТСЯ

TZ прямо предвидел этот исход:

> **«если и dO-prefetch съеден балансом — war за wait в merged закрывается ярлыком 'равновесное ядро, wait лечится только форматом dO (FP4-эпоха)', и это само по себе ценный вердикт для дорожной карты».**

**053 подтвердил** этот сценарий эмпирически:
- 052 (Q ping-pong): mechanism работает, но barrier + ST ямы → wall в шуме
- 053 (dO half-prefetch с обходами): обходы работают, но split-buffer сам новая яма → wall **хуже**
- **Общее правило**: **любая prefetch-схема в merged имеет свою яму**, стоимость которой эквивалентна выигрышу latency.

**Ярлык merged**:

> **«merged v40 — равновесное ядро в парето-фронте stall vs cost. wait 33% не движется вниз никакой prefetch-схемой (Q, dO, оба) при sm_120a fp8 в текущей раскладке. Далее — только формат: dO fp8→fp4 (FP4-эпоха) снимет 2× byte traffic → wait естественно ↓. Prefetch-скимы merge на 040 закрыты; кампания wait — стоп до FP4/новых форматов.»**

---

## §5. Правки production в 053

**После отката**: 0.

- `libs/fa_bwd_merged_v1.cu` md5 `2bf32ab7d4c5ecabb4ee2dbf1b5d4b33` = 040 sealed ✓
- `libs/fa_bwd_dk_new.cu` md5 `a9f0ded8261e53a143b521ffa647f458` = 033 sealed ✓
- `libs/fa_bwd_dq_new.cu` md5 `d7a11a3d788eb4c396d892bc9c8ab754` = 041 sealed ✓
- `libs/bench_r2c_e2e.cu`: EXPECT kernel_merged_v1 = **252** (restored, был 254 during cand).
- `runs/reports/037r_gate.sh`: EXPECT = **252** (restored).
- Diagnostic binaries: `runs/archive/053_pre/r2c_merged_wall_base` + `_cand` + `fa_bwd_merged_v1.cu.053_cand` сохранены для future retest.

---

## §6. Сиквенс (доклад — на столе)

**Merged wait-кампания закрыта** ярлыком "равновесное ядро". Далее:

1. **S2v4 (dk свизл писателя smQ + LDSM-читатель)** — открытая дверь по 052 §0.b (реестр). Апер 4-7% dk isolated → -1..-1.5% E2E. Пробивает 44.0 порог.
   - **Бумага не начата**. Триггер после 053.
   
2. **M5-монетка** (класс #5 merged smV fp8 B-op через LDSM) — 0.5-1.9% wall (043 §3.b). Не гарантирует 2/3 v2 KEEP. Осторожная монета.
   
3. **FP4-эпоха** (dO fp16→fp8/fp4) — стратегический пивот, вне scope 053. Требует новую бумагу когда/если FP4 стек созревает.

**Рекомендация ассистента для 054**: **S2v4 dk** (главная открытая дверь, механизм чист по liveness).

---

## §7. Итоги 053

1. **§0 SMEM-бумага пройдена**: (i) режется структурно, (iii) не даёт 16KB, **(ii) полу-нога 8KB выбрана** (49664B, 2 blk ✓).

2. **§0.b Обход ямы-1 (barrier)**: cp.async wait в Step A qt+1, не в t13 → **работает** (barrier +0.35pp vs 052 +2.29pp).

3. **§0.c Обход ямы-2 (ST-conflicts)**: prefetch issue после Step G STS → **работает** (ST +5.6% vs 052 +123%). CPU-судья ДВУХ фаз применён (адресная 4-way conflict prognosed; данная serialized).

4. **§гейт** — **§d КРАСНЫЙ**:
   - a. ptxas 254r (+2), 0 spill, 2 blk ✓
   - b. fingerprint EXPECT 252→254 обновлён осознанно
   - c. корректность: bit-exact 11/11 x3 + INJECT_BITFLIP catch + memcheck 0 + **racecheck 0 hazards** ✓
   - **d. ABBA 8 пар**: CAND медленнее единогласно 8/8, median Δ **+1.895%** ← **КРАСНЫЙ, ОТКАТ**
   - e. NCu-post: механизм работает (long_sb -2.52pp), но split-buffer address overhead > выигрыша

5. **Root cause + сравнение 052 vs 053**: обходы TZ работают, но split-buffer readers породили новую яму (short_sb +1.27pp + wavefronts +3.4% + wait +3.76pp).

6. **§5 Rollback**: prod merged восстановлен к 040 sealed, EXPECT восстановлен, bench_r2c_e2e пересобран, fingerprint 252/128/69 OK.

7. **§4 Стратегическая строка (war за wait закрывается)**: **«merged v40 = равновесное ядро; wait 33% лечится только форматом dO (FP4-эпоха)». Prefetch-схемы merged закрыты.**

8. **§6 Сиквенс**: **054 = S2v4** (dk свизл writer smQ + LDSM-читатель) — главная открытая дверь.

### Chain md5

- 051 `bd9ea399697ffe9f6ff206618c48a36d`
- 052 `274d9c70af8aebf66f1815779a72a7a1`
- **053 `cd9aae84a94fc02fd199f8e62bcb516e`**

### Файлы 053

- `runs/reports/053_dO_prefetch.md` (this report)
- `runs/reports/053_abba.sh` + `053_abba_data.txt` — §d ABBA
- `runs/reports/053_ncu.sh` + `053_ncu_data.txt` — §e NCu
- `runs/archive/053_pre/*` — diagnostic binaries + candidate source

---

**End 053. Обходы TZ работают, но split-buffer сам породил яму. Wall +1.895%. Откат. War за wait в merged **закрыт** ярлыком "равновесное ядро — FP4-эпоха". Сиквенс: 054 = S2v4.**
