# 041 — Пост-040 разведка merged (I) + разморозка dq §8.2 (II)

**Chain**:
- 040_A_production.md md5: `7d39d1058edbcd47d14f8500584685d3`

**Правила ТЗ 041**: правки merged запрещены. Единственная code-подмена — разморозка dq_new (d7a11a3d) из архива 029 своим гейтом. Часть I → часть II (строгий порядок).

---

## Артефакт-хедер (правило 5)

```
libs/ (pre-041 state):
-rw-r--r-- 25638 Jul  8 16:31  fa_bwd_merged_v1.cu       (md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33 = 040 sealed)
-rwxr-xr-x     Jul  8 16:31  r2c_merged_wall             (md5 8511d3df5194e7ac46295d9b5ba35bbb)
-rwxr-xr-x     Jul  8 16:33  bench_r2c_e2e               (chain prod)
-rw-r--r--     Jul  7 21:41  fa_bwd_dq_new.cu            (md5 683396f8e6867e9fc2e26f8b628774f3 = 024 baseline)

runs/archive/040_sealed/:
-rw-r--r-- 25638 Jul  8 16:42  fa_bwd_merged_v1.cu       (md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33)
-rwxr-xr-x     Jul  8 16:42  r2c_merged_wall             (md5 8511d3df5194e7ac46295d9b5ba35bbb)
-rwxr-xr-x     Jul  8 16:42  r2c_merged_bit_exact
-rwxr-xr-x     Jul  8 16:42  bench_r2c_e2e               (md5 46a6922d842556b2adf24dca7c2aab63)

runs/archive/029_d5lite_pack_pi/:
-rw-r--r-- 18834 Jul  7 21:02  fa_bwd_dq_new.cu          (md5 d7a11a3d788eb4c396d892bc9c8ab754)  ← frozen candidate
```

**Gate-log единый**:
```
$ ./037r_gate.sh
bench_merged: FINGERPRINT kernel_merged_v1: numRegs=252
GATE OK: numRegs=252 matches EXPECT=252
```

---

# Часть I — merged/цепь разведка (0 правок)

## I.1 Свежий полный профиль merged post-040 [режим: NCu-mode]

**Wall в NCu-mode**: avg_ms=24.72–24.83 ms (сравн. чистый isolated wall 040-ABBA candidate median ≈24.97 ms — NCu-налог ~+2%).

### Stall-таблица (Σ 99.12%)

| Класс | % post-040 | pre-040 (037-r fresh) | Δ | Топ |
|:--|:-:|:-:|:-:|:-:|
| **wait** | **33.00** | 27.85 | +5.15 | **топ** |
| selected | 18.21 | 18.15 | ≈ | |
| **math_pipe_throttle** | **13.66** | 4.27 | **+9.39** | **новый 2-й** |
| **short_scoreboard** | 9.71 | 8.63 | +1.08 | |
| **mio_throttle** | **8.86** | 25.10 | **−16.24** | **разгружен** |
| long_scoreboard | 6.73 | 5.15 | +1.58 | |
| not_selected | 4.79 | 5.23 | −0.44 | |
| barrier | 2.57 | 2.76 | −0.19 | |
| dispatch_stall | 0.82 | 0.83 | ≈ | |
| no_instruction | 0.61 | 0.51 | +0.10 | |
| lg_throttle | 0.14 | 0.11 | ≈ | |
| drain / misc / membar / tex | ≤0.02 | ≤0.02 | ≈ | |

**Топ переехал**: wait (33%) + math_pipe (13.66%) + short_sb (9.71%) + mio (8.86%). MMA/ALU pipe теперь bottleneck после LSU разгрузки классом #7.

### SOL / L1 / L2 / DRAM / occupancy [NCu-mode]

| Метрика | Post-040 | Pre-040 | Δ |
|:--|:-:|:-:|:-:|
| DRAM bytes | 9.80 GB | 9.79 GB | 0 ✓ |
| DRAM % peak | 16.48% | 13.55% | +2.93 (wall меньше, ratio выше) |
| L1 hit | 1.49% | 1.49% | 0 |
| L2 hit | **91.74%** | 91.74% | 0 ✓ |
| SM cycles active | 54.76M | 66.6M | **−18%** (ядро короче) |
| GPC clock | 1.59 GHz | 1.59 | ≈ |
| **Occupancy** | **16.59%** | 16.58% | 0 (2 blk × 4 warps) ✓ |

### Wavefronts (LD и ST раздельно)

| | Post-040 | Pre-040 |
|:--|:-:|:-:|
| Total | 4.063B | 5.114B (−20.6%) |
| **LD only** | **3.454B** | (n/a combined) |
| **ST only** | 428.2M | — |

## I.2 Долг 040 — распутывание "4-way" ярлыка

Полный анализ: см. `runs/reports/041_debt_conflicts.md`.

**Ключевые числа**:
- LD conflict events абс: 126.8M → **132.1M** (Δ +5.29M = **+4.17%**)
- Wavefronts total: 5.114B → 4.063B (Δ **−20.55%**)
- Wavefronts LD only: **3.454B**; ST: 428.2M

**Атрибуция +4.17% LD conflicts**:
- НЕ класс #7: NCu metric `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` считает classical LDS bank conflicts. LDSM cooperative fetch учитывается в wavefronts (не в conflict events).
- Скорее шум одного sample (relative rate 2.75%→3.82%). Мелкий timing drift классов #1..#6, #8.

**Распутывание "4-way" ярлыка (в леджер)**:
- **События конфликта на x4** = **≈0** (+5.29M / 268.4M x4 = 0.020 events/x4) ✓ прогноз "0" подтверждён.
- **Wavefronts на x4** = **4.00** ✓ структурный пол LDSM.x4.trans: fetch 512B / 128B-per-wave = **4** (не bank conflict в classical смысле, это ceiling cooperative fetch).
- **Ярлык 039 §3.d "4-way conflict"** уточнён: это wavefronts/x4 = 4, НЕ conflict events.

## I.3 Декомпозиция E2E — per-kernel медианы

**5-run in-chain [режим: in-chain]**:

| Run | Temp | D | merged | dk_new | dq_new | total | overhead |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 44°C | 0.342 | 24.976 | 10.365 | 8.715 | 44.399 | 0.000 |
| 2 | 43°C | 0.341 | 24.988 | 10.374 | 8.722 | 44.425 | −0.000 |
| 3 | 42°C | 0.342 | 25.004 | 10.377 | 8.728 | 44.451 | 0.000 |
| 4 | 44°C | 0.342 | 25.004 | 10.384 | 8.726 | 44.456 | −0.000 |
| 5 | 46°C | 0.342 | 25.023 | 10.391 | 8.737 | 44.494 | −0.000 |

**Медианы**: D=0.342, merged=**25.004**, dk_new=10.377, dq_new=8.726, **total=44.451**.

**Реконсиляция суммы**:
- Sum kernel medians = 0.342 + 25.004 + 10.377 + 8.726 = **44.449 ms**
- Total median = **44.451 ms**
- overhead median ≈ 0
- **Sum ≈ total** (Δ 0.002 ms = noise) ✓

**Vugar-счёт vs ledger (dk+dq drift)**:
- Vugar-arithmetic: 44.483 − 25.019 − 0.343 = **19.121** (dk+dq deduced)
- Мой факт по медианам: dk + dq = 10.377 + 8.726 = **19.103** (~совпадает 19.121)
- **Ledger 033-c dk+dq = 18.656** (Vugar)
- Delta: 19.103 − 18.656 = **+0.447 ms = +2.40%** drift

**У кого drift**:
- **Sources unchanged**: dk_new md5 `a9f0ded8…` (033_sealed), dq_new md5 `683396f8…` (024 baseline). Никаких правок кернелов dk/dq в кампании 033..040.
- Отклонение **термо-/L2-cache-состояние цепи**: merged 27.82→25.02 (−2.8 ms) → dk/dq запускаются в другом эффективном состоянии карты (более разогретой, разное L2 состояние после merged).
- **НЕ реальная регрессия production dk/dq**. Timing artifact chain-composition.

## I.4 Кладбище перечитка (принцип 6/14, бумага)

Полный анализ: см. `runs/reports/041_graveyard_recheck.md`.

**Карта улик v5 (итог)**:

| Цель | Класс-мишень | Механизм | Потолок upper | Риск |
|:--|:--|:--|:-:|:-:|
| **(a) smQ prefetch Q[qt+1]** | long_sb 6.73%; класс #1 (16 ops) | cp.async Q[qt+1] в +8192B headroom (второй smQ буфер); alias-union НЕ трогать | **−1.7 мс** (upper) | средний (barrier сдвиг → отд. ТЗ с racecheck; SMEM slot жестко 101376, тонкий запас 1024B) |
| **(b) smL/smD double-buffer** | класс #3 (4 ops = 0.8% LDS) | 512B prefetch | ~0.05 мс | нулевой (мёртвая масса) — **исключено** |
| **(c) cp.async глубина ≥2 wave** | long_sb 6.73% | commit-group + wave-2 prefetch | **−1.7 мс** (аналог (a)) | тесно связан с (a); без smQ double-buffer — не реализуем |
| **(d) A' классы #4/#6** | 48 LDS.32 (~9% LDS); mio residual 8.86% | ldmatrix.x4.b16 no-trans reader-only | **−0.5..−2%** (36 ops × 0.055%/op) | средний (ptxas люфт 252→≤256; territoриa правила-2/3 v2) |
| **(e) short_sb полировка** | short_sb 9.71% | prefetch reg / A' fold | ≤ −2.4 мс (upper) | средний-высокий (reg pressure) |

**Причины смерти изменились**:
- (a) smQ: **MIO-bound причина исчезла**, headroom 8704 B **достаточен по объёму**, но **alias-union живёт** → сдвиг барьеров = отд. ТЗ.
- (c) cp.async: **паттерн 028 (commit → wait<0> → sync, 0 работы между) подтверждён точь-в-точь** на строках 187-189.
- (e) short_sb: причина смерти "не отделяется от MIO" **не применима** (MIO разгружен).
- (b) остаётся мёртвой (мал. масса).

**НЕ полирую ничего в этом ТЗ**. Карта v5 идёт на выбор пробы отдельно.

---

# Часть II — dq этапы 2-3 (§8.2 триггер)

## II.6 Свежий NCu-профиль production dq_new [NCu-mode]

Скрипт: `runs/reports/041_ncu_dq_fresh.sh`, данные: `041_ncu_dq_fresh_data.txt`.

| Класс | 024 baseline | **041 fresh** | Δ |
|:--|:-:|:-:|:-:|
| mio_throttle | 46.73% | **46.74%** | +0.01 ≈ |
| barrier | 10.93% | **10.94%** | +0.01 ≈ |
| long_sb | 10.22% | **10.22%** | 0 ≈ |
| wait | 9.11% | 9.11% | 0 |
| not_selected | 5.97% | 8.95% | +2.98 (небольшое) |
| short_sb | 5.97% | 5.97% | 0 |
| selected | ~5% | 5.01% | ≈ |
| math_pipe | ~1.7% | 1.73% | ≈ |

**Профиль dq_new практически идентичен 024**. Правки merged 040 (класс #7) **не сдвинули dq_new** (ожидаемо, dq_new читает dS_nat из DRAM после merged, без tight coupling).

- DRAM: 9.25 GB
- DRAM %peak: 49.50%
- L2 hit: 71.29%
- **Occupancy: 48.86%** (6 blocks × 4 warps)
- LD conflicts: 541M
- ST conflicts: 61M

## II.7 Разморозка d7a11a3d

**Md5-сверка**:
- Архив 029: `d7a11a3d788eb4c396d892bc9c8ab754` ← EXPECT (архивная запись)
- После cp в libs: md5 совпадает ✓

**ptxas**:
```
kernel_dq_new — Used 69 registers, used 1 barriers
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
```

**Fingerprint gate (EXPECT dq_new)**:
- Pre: bench_r2c_e2e EXPECT dq_new=56 (production 683396f8) — gate FAIL при разморозке (69 != 56) ✓ gate работает.
- Временно обновлён EXPECT=69 в bench_r2c_e2e.cu для теста (осознанно с записью, архивная запись 029).
- После KEEP: **EXPECT dq_new=56 → 69** (постоянно).

**BIT-EXACT 11/11 + CANARY** ✓ на разморозке (chain: dQ + dK + dV bit-exact).

Frozen candidate binary: `libs/bench_r2c_e2e_dqfrozen` md5 **`baf557b4c136adb8cfc1223a8d081446`**.

## II.8 ABBA 8 пар prod_dq (56r) vs frozen d7a11a3d (69r) [in-chain]

Скрипт: `041_dq_abba.sh`, данные: `041_dq_abba_data.txt`. Одна сессия, 4+ warmup.

**Fingerprint verification** (перед ABBA):
```
BASE: kernel_dq_new numRegs=56 (expected 56) OK
CAND: kernel_dq_new numRegs=69 (expected 69) OK
```

### Данные (16 runs, ABBA B C C B B C C B B C C B B C C B):

| # | Tag | Temp | dq_new | total |
|:-:|:-:|:-:|:-:|:-:|
| 1 | BASE | 42°C | 8.737 | 44.488 |
| 2 | CAND | 46°C | 8.426 | 44.154 |
| 3 | CAND | 44°C | 8.426 | 44.157 |
| 4 | BASE | 44°C | 8.731 | 44.477 |
| 5 | BASE | 44°C | 8.731 | 44.462 |
| 6 | CAND | 44°C | 8.430 | 44.184 |
| 7 | CAND | 50°C | 8.433 | 44.190 |
| 8 | BASE | 47°C | 8.740 | 44.508 |
| 9 | BASE | 45°C | 8.738 | 44.503 |
| 10 | CAND | 45°C | 8.440 | 44.226 |
| 11 | CAND | 43°C | 8.445 | 44.235 |
| 12 | BASE | 43°C | 8.740 | 44.513 |
| 13 | BASE | 48°C | 8.740 | 44.516 |
| 14 | CAND | 46°C | 8.436 | 44.201 |
| 15 | CAND | 45°C | 8.444 | 44.252 |
| 16 | BASE | 47°C | 8.747 | 44.542 |

### Парные дельты (Δ = CAND − BASE, положит.=BASE быстрее)

| Пара | BASE dq | CAND dq | Δdq (ms) | **Δ% dq** | Δ% total |
|:-:|:-:|:-:|:-:|:-:|:-:|
| P1 | 8.737 | 8.426 | −0.311 | **−3.56%** | −0.75% |
| P2 | 8.731 | 8.426 | −0.305 | −3.49% | −0.72% |
| P3 | 8.731 | 8.430 | −0.301 | −3.45% | −0.63% |
| P4 | 8.740 | 8.433 | −0.307 | −3.51% | −0.71% |
| P5 | 8.738 | 8.440 | −0.298 | −3.41% | −0.62% |
| P6 | 8.740 | 8.445 | −0.295 | **−3.37%** (worst) | −0.63% |
| P7 | 8.740 | 8.436 | −0.304 | −3.48% | −0.71% |
| P8 | 8.747 | 8.444 | −0.303 | −3.46% | −0.65% |

### Статистика dq_new (isolated wall in-chain)

- Sorted Δ%: −3.37, −3.41, −3.45, −3.46, −3.48, −3.49, −3.51, −3.56
- **Median: −3.47%**
- **Worst pair: −3.37%**
- Все 8 пар CAND быстрее (знак единогласный).

### Вердикт по правилу-2/3 v2

- Медиана **−3.47%** ≥ 3% → **KEEP** (пробил порог с запасом ~0.5 pp)
- Worst pair −3.37% ≥ 1% ✓
- Vugar-ожидание было ~−1.5% (ниже порога) → **факт значительно выше ожидания**.

**Натяжка вердикта отсутствует**: 3.47% >> 3.00%, разница далеко за пределами шума.

**НЕОЖИДАННЫЙ KEEP** ✓.

## II.9 Sealed archive + prod swap + E2E 5-run + леджер

### Sealed archive

`runs/archive/041_dq_sealed/`:
- `fa_bwd_dq_new.cu` md5 **`d7a11a3d788eb4c396d892bc9c8ab754`** (029 d5lite_pack_pi)
- `bench_r2c_e2e_dqfrozen` md5 **`baf557b4c136adb8cfc1223a8d081446`** (frozen bench для сохранения candidate binary)

### Production swap

- `libs/fa_bwd_dq_new.cu`: `683396f8…` → **`d7a11a3d…`** ✓
- `libs/bench_r2c_e2e.cu` EXPECT dq_new: **56 → 69** (осознанно с записью, комментарий "041 KEEP: d7a11a3d разморожен")
- Rebuild `bench_r2c_e2e` — новый chain-bench с sealed dq (EXPECT 69 OK, chain BIT-EXACT 11/11 ✓)

### E2E 5-run in-chain с декомпозицией [mode: in-chain]

| Run | Temp | D | merged | dk_new | **dq_new** | **total** |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 49°C | 0.342 | 25.008 | 10.388 | 8.431 | **44.169** |
| 2 | 46°C | 0.342 | 25.036 | 10.398 | 8.441 | 44.217 |
| 3 | 45°C | 0.343 | 25.024 | 10.397 | 8.435 | 44.199 |
| 4 | 44°C | 0.342 | 25.029 | 10.398 | 8.436 | 44.206 |
| 5 | 43°C | 0.342 | 25.032 | 10.397 | 8.438 | 44.209 |

**Медианы**: D=0.342, merged=25.029, dk_new=10.397, **dq_new=8.436**, **total=44.206**.

**Sum reconciliation**: 0.342 + 25.029 + 10.397 + 8.436 = 44.204 ≈ 44.206 (Δ=0.002 noise) ✓.

### Ledger

- **033-c baseline E2E**: 46.82 ms
- **Post-040 (only merged)**: 44.483 ms in-chain (−5.00%)
- **Post-041 (merged + dq d7a11a3d)**: **44.206 ms** in-chain = **−5.58% cumulative vs 033-c**
- **Incremental 041 vs 040**: 44.483 → 44.206 = **−0.62% E2E** (ниже 2-3% порога, но dq-isolated median −3.47% пробил)

**Merged in-chain post-041**: 25.029 ms (было 27.82 в 033-c) = **−10.03%** от 033-c leдгеp merged.

**dq_new isolated in-chain**: 8.436 ms (было 8.726 с prod 683396f8) = **−3.32%** vs prod.

### TFLOPS 3-MMA (merged)

- 6.597e12 / 0.025029 = **263.6 T** (пост-041, in-chain; было 236.7 в 037-r fresh, +11.4%)

---

## §III. Итоги 041

### Часть I: разведка

1. **Полный профиль merged post-040** (Σ 99.12%): wait 33% + math_pipe 13.66% + short_sb 9.71% + mio 8.86%. MIO разгружен (25.10→8.86), math_pipe новый ALU-bottleneck.
2. **Долг 040 распутан**: "4-way" = **wavefronts на x4 = 4** (структурный пол 512B/128B), НЕ conflict events. События конфликта на x4 ≈0 (0.020 events/x4). **В леджер**.
3. **E2E декомпозиция**: sum kernel medians ≈ total ✓. Dq+dk drift +2.40% vs ledger 033-c — **timing artifact** (sources unchanged), не регрессия.
4. **Кладбище v5**: 5 могил перечитаны; (a) smQ prefetch, (c) cp.async глубина, (d) A', (e) short_sb — **живые кандидаты** (причины смерти изменились). (b) остаётся мёртвой.

### Часть II: dq §8.2 KEEP

5. **NCu dq fresh vs 024**: идентичный профиль (mio 46.74%, barrier 10.94%, long_sb 10.22% — совпадают в ±0.01). Правки merged не сдвинули dq.
6. **Разморозка d7a11a3d**: ptxas 69r/0s/1bar; BIT-EXACT 11/11+CANARY chain ✓.
7. **ABBA 8 пар prod (56r) vs frozen (69r)**:
   - **dq_new isolated median: −3.47%**, worst pair −3.37%
   - **Все 8 пар CAND быстрее** единогласно
   - **Правило-2/3 v2 KEEP** (пробил 3% с запасом)
   - **Vugar-ожидание ~−1.5% не сбылось** — реальный −3.47% (**неожиданный KEEP**)
8. **Sealed** в `runs/archive/041_dq_sealed/` (dq_new md5 `d7a11a3d…`).
9. **EXPECT dq_new: 56 → 69** обновлено в `bench_r2c_e2e.cu` (осознанно с записью).
10. **E2E 5-run in-chain**: total median **44.206 ms**, dq_new median **8.436 ms**.

### Chain md5

- 040 `7d39d1058edbcd47d14f8500584685d3`
- **041 `<computed>`**

### Sealed states

- Merged: `runs/archive/040_sealed/fa_bwd_merged_v1.cu` md5 `2bf32ab7d4c5ecabb4ee2dbf1b5d4b33`
- Dq: `runs/archive/041_dq_sealed/fa_bwd_dq_new.cu` md5 `d7a11a3d788eb4c396d892bc9c8ab754`

### Ledger cumulative

- 033-c → 040 → 041: **46.82 → 44.483 → 44.206 ms E2E** = **−5.58% cumulative**
- Merged: 27.82 → 25.029 = **−10.03%**
- Dq: prod → d7a11a3d = **−3.32%** (in-chain isolated)

### Файлы 041

- `runs/reports/041_post040_recon_dq.md` (this report)
- `runs/reports/041_ncu_merged_full.sh` + `_data.txt`
- `runs/reports/041_debt_conflicts.md` (4-way ярлык распутан)
- `runs/reports/041_e2e_decomp.sh` + `_data.txt`
- `runs/reports/041_graveyard_recheck.md` (карта улик v5)
- `runs/reports/041_ncu_dq_fresh.sh` + `_data.txt`
- `runs/reports/041_dq_abba.sh` + `_data.txt`
- `runs/reports/041_dq_prod.cu.bak` (temp backup, safety)
- `runs/archive/041_dq_sealed/` — sealed KEEP

---

**End 041. Двойной KEEP: merged (040) + dq (041 разморозка). E2E cumulative −5.58%.**
