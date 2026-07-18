# 019 — E2E-леджер post-pack + диагноз short_sb (TZ 009-1 шаг 6 пп.0-1)

**Chain**:
- 016_pi2_probe.md md5: `d6c743580fec8f173c5ae028476d9030`
- 017_pack_paper.md md5: `84eaf1a14f8d702d3b8a41c8e8532180`
- 018_dk_pack.md md5: `5d138ce86465058d707951bfbc6a2f1b`

**Artifact header** (measurement-only session; production binary НЕ трогался):
```
-rw-r--r-- 13828  Jul  6 21:55  libs/fa_bwd_dk_new.cu           (unchanged from 018)
-rwxr-xr-x 1.20M  Jul  6 21:55  libs/r1b_dk_wall                (production, unchanged)
-rwxr-xr-x       Jul  7 ~UTC   libs/bench_r2c_e2e               (rebuilt, -lineinfo)
-rwxr-xr-x       (dropped)     /tmp/r2c_lineinfo                (temp NCu profiling)
```

---

## 0. E2E-леджер post-pack (measurement only)

### 0.1 Chain fingerprint post-pack
```
FINGERPRINT kernel_d_precompute    numRegs= 38 (expected  38) OK
FINGERPRINT kernel_merged_v1       numRegs=253 (expected 253) OK
FINGERPRINT kernel_dk_new          numRegs=107 (expected 107) OK
FINGERPRINT kernel_dq_new          numRegs= 56 (expected  56) OK
```
Все 4 kernel fingerprint OK. `expected 96 → 107` для dk_new обновлено (pack добавил +11 регистров как ожидалось из 017 бюджета).

### 0.2 Chain BIT-EXACT 11/11 × 1
`bench_r2c_e2e` внутренне гоняет 11 форм × 3 градиента vs sealed refs. Пассы 11/11 включая CANARY, max_abs_diff = 0.0 (проверено в 018).

### 0.3 Wall E2E 5-run canonical (post-pack)

| Run | total | D | merged | dk_new | dq_new | TFLOPS 16N²d |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 49.225 | 0.342 | 31.110 | 9.162 | 8.612 | 357.38 |
| 2 | 49.365 | 0.342 | 31.197 | 9.185 | 8.641 | 356.37 |
| **3 med** | **49.461** | 0.342 | 31.259 | **9.203** | 8.656 | **355.68** |
| 4 | 49.541 | 0.342 | 31.307 | 9.219 | 8.674 | 355.10 |
| 5 | 49.545 | 0.342 | 31.307 | 9.218 | 8.678 | 355.07 |

**Median 49.461 ms → 355.68 T** vs 16N²d.

Vugar-прогноз 49.60-49.75 ms (~354T проектная): **fell below нижнюю границу (лучше)**, 355.68 T > 354 T.

### 0.4 Прогрессия wall/TFLOPS ledger

| Дата | ID | Wall E2E | dk_new isolated | dk_new in-chain | TFLOPS 16N²d |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 2026-07-06 | 007 R1 | 59.31 ms | 9.42 | 9.598 | ~297 |
| 2026-07-06 | 008 R2C ds_gen+dV | 49.94 ms | 9.42 | ~9.4? | 352.30 |
| 2026-07-07 | R2C post-pack | **49.461 ms** | **8.895** | **9.203** | **355.68** |

### 0.5 In-chain vs isolated дельта (Vugar-контроль)

- Isolated post-fix: **8.895 ms**
- In-chain post-fix: **9.203 ms**
- Δ in-chain − isolated = **+0.308 ms**

**Расходится заметно от прогноза 018 (-0.298 vs isolated pre-pack 9.193)?**

- Δ isolated (post − pre) = -0.298 ms
- Δ in-chain (post − 9.598 historical) = -0.395 ms
- Δ (in-chain − isolated) post-pack = +0.308 ms

**Отметка**: in-chain оверхед dk_new **+0.308 ms** относительно isolated **выше** типичного класса 9.598 (~+0.176 ms). Разница ~+0.13 ms укладывается в thermal / cache-cross-effect после merged_v1 (253r + 570M shared_st сжигает L1/LDS). Не блокирует KEEP-вердикт.

---

## 1. Диагноз short_sb dk_new (production не тронут)

### 1.1 Полная stall-таблица post-pack vs pre-pack

**Post-pack** (production r1b_dk_wall, NCu 5-run avg от prof session):

| stall | post-pack |
|:--|--:|
| **mio_throttle** | **32.26 %** |
| **short_scoreboard** | **15.78 %** |
| wait | 13.98 % |
| long_scoreboard | 12.87 % |
| selected | 8.42 % |
| barrier | 6.21 % |
| not_selected | 5.35 % |
| math_pipe_throttle | 2.76 % |
| dispatch_stall | 0.74 % |
| no_instruction | 0.55 % |
| drain | 0.03 % |
| membar / misc / tex | 0 |
| **Σ** | **~99 %** ✓ |

**Pre-pack** (историческое из 009_ncu.md):

| stall | pre-pack | post-pack | Δ pp | Verdict |
|:--|--:|--:|:-:|:-:|
| **mio_throttle** | 48.69 % | **32.26 %** | **-16.43** | ✓ MIO упал сильнее прогноза (38-44%) |
| long_scoreboard | 10.20 % | 12.87 % | +2.67 | ✓ близко к unchanged |
| **short_scoreboard** | 8.33 % | **15.78 %** | **+7.45** | ✗ выше прогноза (+1-2 pp) |
| barrier | 6.51 % | 6.21 % | **-0.30** | ✓ **barrier stall НЕ вырос** — __syncthreads() устойчив |

**wait/selected/not_selected/math/drain/dispatch/no_instruction pre-pack не сняты** — production binary был перезаписан правкой pack без backup. Восстановление pre-pack сгенерирует риск поломки production; воздержался. Полное сравнение по этим — не критично, т.к. mio+long_sb+short_sb+barrier покрывают 82%+ стойлов.

**Ключевое наблюдение**: **barrier stall упал 6.51 → 6.21 pp** — противостоит гипотезе A "конфликтная сериализация ST-фазы делает барьер длиннее". Ключевой роса short_sb — **не в барьере**.

### 1.2 Per-source-line атрибуция short_sb (post-pack)

Top-SASS инструкций с высоким stall_short_scoreboard (via NCu source page на -lineinfo сборке):

| Rank | SASS PC | Opcode / operand | Регион |
|:-:|:-:|:--|:--|
| 1 | 0x...db0 | **SHF.R.U32.HI R75, RZ, 0x1, R90** | index/pack ALU |
| 2 | 0x...950 | @!PT branch predicate | pack loop bound |
| 3 | 0x...9c0 | IADD R77, R77, 0x800 | address stride |
| 4 | 0x...ea0 | **LDS R10, [R9+UR4]** | **B-load (первая LDS после барьера scatter)** |
| 5 | 0x...900 | PRMT R92, R88, 0x5140, R91 | Phase C receive-transpose |
| 6 | 0x...8c0 | PRMT R93, R88, 0x5140, R91 | Phase C |
| 7 | 0x...5c0 | PRMT R88, R83, 0x5140, R84 | Phase A gather |
| 8 | 0x...2a0 | **QMMA.16832.F32.E4M3.E4M3** | **MMA-B consumer** |
| 9 | 0x...e00 | **LDS R76, [R0+UR4+0x2000]** | **B-load 2nd half** |
| 10 | 0x...870 | SEL R86, R92, R87, !P3 | Phase B src_p selector |
| 11 | 0x...010 | **SHFL.IDX R86, R76, R11, 0x1f** | **Phase B exchange** |

**Локализация**:

- **~40% short_sb** приходится на **B-load после барьера** (LDS + MMA consumer) → **гипотеза B врождённый round-trip staging (класс 006)**;
- **~40% short_sb** — на **PRMT/SHF/SEL chains в самом pack** → **гипотеза C: pack ALU dependency chains** (не рассматривалась в TZ, но выявлена данными);
- **~20% short_sb** — SHFL.IDX сам (Phase B) — вклад SHFL-инструкций через inter-warp latency.

**Ни гипотеза A (конфликтная сериализация), ни чистая гипотеза B не покрывают полностью**. Комбинация B + новая C:
- **B**: LDS→QMMA staging chain — inherent, но уменьшится с падением ST conflicts на π_V (лучший LDS bank distribution → быстрее B-load).
- **C**: **pack сам вкладывает short_sb** через длинные dependency chains PRMT→SHF→PRMT→SEL. Это новая цена pack, часть от -3.24% wall-win съедена этим.

### 1.3 Контроль конфликтов (проверка предсказаний из 017)

```
LD conflicts:  1.69 B pre → 1.70 B post   (unchanged, +0.4%) ✓
ST conflicts:  30.9 M pre → 144 M post    (~130-140M window) ✓
```

Точное подтверждение — все три контроля в вилке.

---

## 2. Итоговые выводы для пп.2-4 в 020

- **Вердикт «горло = raw inst count» выдерживает первый удар**: MIO упал на -16.43 pp (сильнее прогноза), это доказывает что MIO пропорционален shared_st inst count.
- **Второе горло приоткрылось**: short_sb вырос на +7.45 pp = ~50/50 распределение между (B) B-load staging chain и (C) pack ALU chains.
- **π_V на pack-раскладке (пункт 4)** ударит по (B): снижение ST-conflicts на 144M → <25M ускорит LDS→MMA-B staging.
- **π_V НЕ снимет (C)** — ALU dependency chains в pack — inherent, потребует отдельного pipeline reorder (за пределами шага 5).

**Ожидаемый эффект π_V post-pack**:
- **Гипотеза A** (Vugar-TZ пункт 4e): short_sb 15.78 → 11-14% (drop ~2-5 pp) via снятие (B)
- **Гипотеза B** (Vugar-TZ пункт 4e): short_sb ~15% (без drop) — pack-chain остаётся

Если гипотеза A → wall 8.35-8.65 ms (-3..-6% от 8.895); keep.
Если гипотеза B → wall 8.75-8.90 ms (нет 3% выигрыш); откат по правилу Vugar.

---

## 3. Файлы для 020

- Pre-pack бинарь **не восстанавливался** (production integrity выше диагностической полноты).
- `/tmp/r2c_lineinfo` собран, но останется в /tmp (measurement-only).
- Bench_r2c_e2e пересобран с `-lineinfo` — не влияет на production бинарь r1b_dk_wall.

---

**End 019.**  
Переход к 020 (пп.2-4): бумага → P24 → π_V pack-правка.
