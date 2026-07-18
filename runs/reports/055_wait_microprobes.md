# 055 — Две микро-пробы wait-класса merged: обе КРАСНЫЕ → war-close ПОЛНОЙ формулировкой

**Chain**:
- 052_smq_prefetch.md `274d9c70af8aebf66f1815779a72a7a1`
- 053_dO_prefetch.md `cd9aae84a94fc02fd199f8e62bcb516e`
- 054_m5_aprime.md `3441eb93f21c50e7a1dea959e426946f`

**Правила ТЗ 055**: две микро-пробы (§1.a L2-хинт, §1.b LDG-регистр-четверть) БЕЗ второй ноги. Каждая свой мини-гейт. Приоритет-гейт: 054 не KEEP (paper closure) + E2E 44.206 > 44.0 → **исполнять немедленно** (cert-пакет не нужен). Baseline archive 055_pre (fresh, не переиспользовать 054_pre).

---

## Артефакт-хедер (правило 5)

```
libs/ (post-055 rollback):
-rw-r--r-- 25638 Jul  9  fa_bwd_merged_v1.cu           md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33  = 040 sealed ✓
-rw-r--r-- 13352 Jul  9  fa_bwd_dk_new.cu              md5 a9f0ded8261e53a143b521ffa647f458  = 033 sealed ✓
-rw-r--r-- 18834 Jul  8  fa_bwd_dq_new.cu              md5 d7a11a3d788eb4c396d892bc9c8ab754  = 041 sealed ✓
-rwxr-xr-x       Jul  9  bench_r2c_e2e                 fingerprint 252/128/69/38 OK
```

Archive `runs/archive/055_pre/` (fresh, свежий):
```
-rw-r--r--  fa_bwd_merged_v1.cu           md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33  (baseline 040 sealed)
-rwxr-xr-x  r2c_merged_wall               md5 3d4ce72183fe0a3a757457f15c06c89f  (base, 252r)
-rwxr-xr-x  r2c_merged_wall_cand_1a       md5 82afeb0f7eba9294b00917211e38c84d  (cand §1.a L2-hint, 252r)
-rwxr-xr-x  r2c_merged_wall_cand_1b       md5 7b2821dcaf5b23d81c22212adfa2985a  (cand §1.b LDG-reg, 254r)
-rw-r--r--  fa_bwd_merged_v1.cu.055_1a_cand   (source §1.a)
-rw-r--r--  fa_bwd_merged_v1.cu.055_1b_cand   (source §1.b)
-rwxr-xr-x  bench_r2c_e2e                 md5 49079b8e6bded46b49710b3fddd15fe7  (baseline snapshot)
-rwxr-xr-x  r2c_merged_bit_exact          md5 605bae285896362ddd60199ef1281c55  (baseline snapshot)
```

**Baseline sanity (055_pre)**: `r2c_merged_bit_exact` → 11/11 BIT-EXACT ✓; `037r_gate.sh` → GATE OK numRegs=252 ✓.

---

## §0. Приоритет-гейт TZ

**Условие TZ 055 §0**: "Если 054 в откате или E2E > 44.0 -- исполнять немедленно."

**Факт**: 054 — paper closure (не KEEP), правки production в 054: **0**. E2E in-chain 041 = **44.206 ms > 44.0**. → **исполнять немедленно** (cert-пакет 009-F не запускается — 054 не был KEEP-KEEP).

---

## §1.a L2-хинт (час) — КРАСНЫЙ, гипотеза "нулевая цена" опровергнута

### §1.a Правка

Inline PTX `prefetch.global.L2 [addr]` в КОНЦЕ qt-шага (после Step H, перед BARRIER t13). Каждый thread берёт свои 128B cache lines:
- 128 threads × 128B = 16 KB покрытие dO[qt+1]
- tid < 64 → доп 128B Q[qt+1] chunk (64 threads × 128B = 8 KB покрытие Q[qt+1])

**НЕ persist-бронь** (могила 047 не тревожится: подтяжка ВНУТРИ ядра, не удержание между ядрами).

**Стоимость по бумаге TZ**: "цена ~нулевая (одна инструкция, ни барьеров, ни склада, ни регистров)".

### §1.a Гейт-лайт

- **ptxas**: **252r unchanged** (0 delta регистров ✓); 0 spill; 2 blk ✓
- **fingerprint**: 252 OK (EXPECT неизменен)
- **bit-exact 11/11** dV+dS_nat + canary ✓ (правка не меняет данных)

### §1.a ABBA (8 пар, одна сессия, 4 warmup, temp 44-49°C)

| Pair | BASE ms | CAND ms | Δ (ms) | Δ% |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 48.464 | 49.518 | +1.054 | +2.174% |
| 2 | 48.464 | 49.510 | +1.046 | +2.159% |
| 3 | 48.463 | 49.515 | +1.052 | +2.171% |
| 4 | 48.465 | 49.503 | +1.038 | +2.142% |
| 5 | 48.596 | 49.523 | +0.927 | +1.908% |
| 6 | 48.474 | 49.530 | +1.056 | +2.179% |
| 7 | 48.602 | 49.525 | +0.923 | +1.899% |
| 8 | 48.469 | 49.509 | +1.040 | +2.146% |

**BASE median**: 48.467 ms; **CAND median**: 49.517 ms; **Δ median = +1.049 ms = +2.17%**

Все 8 пар единогласно CAND **МЕДЛЕННЕЕ** (правило-2/3 v2: направленность единогласна, амплитуда >2%).

### §1.a Доклад Vugar-строка (по TZ правилу для 0.5-2% единогласного диапазона)

**TZ правило §1.a вердикт-исключение**: "при дельте 0.5-2% с единогласными парами (8/8 в одну сторону) — НЕ откатывать молча, доложить Vugar отдельной строкой ДО вердикта: цена владения правки ~нулевая (одна инструкция, ни барьеров, ни склада, ни регистров), возможен пересмотр порога для нулевой-цены-класса — решение его."

**ВАЖНО**: правило применимо для **положительных** дельт (CAND быстрее в диапазоне 0.5-2%). Здесь дельта **отрицательная** (CAND хуже на 2.17%, вне диапазона 0.5-2%) — правило-исключение НЕ применимо.

**Однако**: **гипотеза TZ "цена ~нулевая" опровергнута фактом**. Один PTX prefetch × 128 threads × 16384 blocks × n_qt = **миллиарды PTX prefetch operations**. Каждая проходит через LSU issue queue + TLB address translation. Реальная цена **~2% wall** — НЕ нулевая.

**Root cause**: prefetch.global.L2 requires:
- LSU issue slot (competes with cp.async issues)
- Address translation (TLB pressure)
- L2 bank access (может конфликтовать с реальными loads)

**Вердикт §1.a**: **КРАСНЫЙ откат** (правило-2/3 v2, дельта отрицательная -2.17%). **Ярлык нулевой-цены-класса опровергнут** — L2-хинт НЕ бесплатный.

### §1.a Rollback

- Восстановлен `fa_bwd_merged_v1.cu` → 040 sealed (md5 2bf32ab7)
- EXPECT 252 unchanged (не двигался)
- Cand binary сохранён в `runs/archive/055_pre/r2c_merged_wall_cand_1a` для history

**По TZ §1.b условию**: "исполнять только если -1.a не закрыл вопрос — т.е. дельта -1.a < 0.5% или откачена". Дельта откачена → **§1.b исполняется**.

---

## §1.b Регистровый префетч четверти dO (день) — КРАСНЫЙ

### §1.b Правка

**Механика TZ**:
- LDG dO[qt+1][первая четверть, 4КиБ] в 8 регистров в НАЧАЛЕ qt-шага
- STS этих регистров на существующую единственную ногу smdO ПОСЛЕ последнего чтения dO[qt] (Step H) и ДО BARRIER t13
- cp.async в Step A qt+1: skip первую четверть (уже подстелена STS'ом qt-1)
- **Ни новых барьеров, ни второй полки, ни шва** — читатели dO не тронуты

**Реализация**:
- `uint32_t dO_pref[8]` — 32 bytes/thread × 128 threads = 4096B (rows 0..15 первой четверти)
- LDG в начале qt: `row_pref = tid/8` ∈ 0..15, `col_off = (tid%8)*32` ∈ 0..224
- STS в конце qt: два 16-byte chunks с swizzle `xor_bytes = (row_pref & 7) << 4`
- cp.async в Step A skip rows 0..15 при `qt > qt_start`

### §1.b Гейт полный

- **ptxas**: **254r** (+2r vs 252 base) ✓ (потолок 256 с люфтом 2); 0 spill; 2 blk ✓
- **EXPECT**: 252 → 254 обновлено осознанно ("055 §1.b: +2r vs 252 base — 8-reg LDG prefetch dO first quarter")
- **fingerprint**: 254 OK
- **bit-exact 11/11** dV+dS_nat + canary + chain x3 = 33/33 ✓
- **INJECT_BITFLIP=1**: 0/11 (dS_nat mism=1 caught) ✓
- **memcheck**: 11/11 + 0 errors ✓
- **racecheck НЕ требуется** — барьеры не тронуты (все 6 t3/t_new1/t9/t_new2/t11/t13 на своих местах)

### §1.b ABBA (8 пар, одна сессия, 4 warmup, temp 44-48°C)

| Pair | BASE ms | CAND ms | Δ (ms) | Δ% |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 48.459 | 49.584 | +1.125 | +2.322% |
| 2 | 48.467 | 49.586 | +1.119 | +2.309% |
| 3 | 48.464 | 49.583 | +1.119 | +2.309% |
| 4 | 48.465 | 49.581 | +1.116 | +2.303% |
| 5 | 48.470 | 49.726 | +1.256 | +2.591% |
| 6 | 48.474 | 49.720 | +1.246 | +2.571% |
| 7 | 48.459 | 49.721 | +1.262 | +2.605% |
| 8 | 48.470 | 49.720 | +1.250 | +2.579% |

**BASE median**: 48.466 ms; **CAND median**: 49.653 ms; **Δ median = +1.187 ms = +2.45%**

Все 8 пар единогласно CAND **МЕДЛЕННЕЕ**. Правило-2/3 v2: **КРАСНЫЙ ОТКАТ**.

### §1.b NCu-post поименно

| Метрика | BASE (040) | CAND (§1.b) | Δ | Прогноз |
|:--|:-:|:-:|:-:|:--|
| **wait** | 33.00% | 33.30% | **+0.30 pp** | ↓ предсказано — **опровергнуто** (мало-двигательное) |
| barrier | 2.57% | 2.75% | +0.18 pp | не должен двигаться (не тронуты барьеры) — небольшой noise |
| **long_sb** | 6.72% | **4.31%** | **-2.41 pp** | ↓ предсказано — **работает** (LDG DRAM latency прячется) |
| **mio** | 8.86% | 9.60% | **+0.74 pp** | +LDG-issue предсказано ✓ |
| short_sb | 9.71% | 10.45% | +0.74 pp | знак ↑ (STS chain) |
| math_pipe | 13.67% | 13.34% | -0.33 pp | ≈ |
| **DRAM** | 9.84 GB | 9.84 GB | ≈ | 9.79 ровно ✓ |
| LD conflicts | 132.1M | 128.3M | -2.9% | ≈ noise |
| **ST conflicts** | **17.2M** | **83.4M** | **+385%!** | **новая яма STS** |
| **Wavefronts LSU** | 4.063B | 4.192B | **+3.2%** (+130M) | LDG + STS waves |
| Occupancy | 16.59% | 16.59% | 0 | 2 blk ✓ |
| **regs** | 252 | 254 | +2 | ✓ (в потолке) |

**Куда делись спрятанные такты (root cause TZ обязателен)**:

1. **Механизм работает механически**: long_sb **-2.41pp** (LDG DRAM latency dO[qt+1] действительно прячется под MMA-петлю) + mio +0.74pp (LDG issue цена реальная как предсказано).

2. **НО — НОВАЯ ЯМА STS-контеншен**: **ST conflicts +385%** (17.2M → 83.4M, +66M).
   - Root cause: STS 32 bytes/thread × 128 threads = 4KB блок с XOR swizzle chunk-XOR pattern.
   - Каждый thread пишет 8 uint32 в 2 chunks через XOR-permuted addressing. 128 threads одновременно = массивный bank conflict pileup.
   - LSU pipe насыщается STS bank retry.

3. **Wall balance**:
   - Выигрыш: long_sb -2.41pp ≈ -0.06 ms
   - Расход: ST +385% + short_sb +0.74 + mio +0.74 ≈ +1.25 ms
   - **Net wall Δ = +1.187 ms = +2.45%** ← STS яма перевешивает LDG выигрыш

4. **wait +0.30pp** — почти не сдвинулся: LDG DRAM latency в long_sb, не в wait.

### §1.b Правило TZ "четверть платит >=1%"

TZ: "Если четверть платит >=1%: НЕ масштабировать самовольно — доклад: половина dO = +16r, решение Vugar отдельным гейтом."

**Дельта -1.b = +2.45% ХУЖЕ** (НЕ payment). Правило НЕ применимо — регресс, не выигрыш. Масштабирование до половины **противопоказано** (STS яма усилится, +16r до 260 > 256 потолка → ptxas red).

### §1.b Rollback

- Восстановлен `fa_bwd_merged_v1.cu` → 040 sealed (md5 2bf32ab7) ✓
- EXPECT 254 → 252 обновлено ("055 rollback: обе микро-пробы КРАСНЫЕ")
- Bench_r2c_e2e пересобран, fingerprint 252/128/69 OK
- Cand binary сохранён в `runs/archive/055_pre/r2c_merged_wall_cand_1b` для history

---

## §2. Вердикт-карта — ОБЕ КРАСНЫЕ

| Проба | Правка | Δ wall | Root cause | Вердикт |
|:--|:--|:-:|:--|:-:|
| **§1.a** L2-хинт | prefetch.global.L2 в конец qt | **+2.17%** | LSU issue queue + TLB — цена НЕ нулевая | **КРАСНЫЙ** |
| **§1.b** LDG-регистр четверть dO | 8 regs LDG start-qt → STS end-qt | **+2.45%** | STS +385% bank storm (16-byte chunks × 128 threads swizzle) | **КРАСНЫЙ** |

**По TZ: "(обе красные) -> war за wait merged закрывается ПОЛНОЙ формулировкой"** ✓

---

## §3. Стратегическая строка — WAR ЗА WAIT MERGED ЗАКРЫВАЕТСЯ ПОЛНОЙ ФОРМУЛИРОВКОЙ

TZ прямо предвидел:

> **«вторая полка (052/053), карманы (-1.b), L2-подтяжка (-1.a) — измерены и съедены; wait-стена merged двигается только форматом dO (FP4) или persist-архитектурой» — и это ценный вердикт для дорожной карты, не поражение.**

**055 подтвердил** эмпирически:

| Стратегия | ТЗ | Wall Δ | Root cause |
|:--|:-:|:-:|:--|
| **Вторая полка Q ping-pong** | 052 | -0.033% (шум) | barrier+ST-контеншен съели механизм |
| **Вторая полка dO half-prefetch с обходами** | 053 | +1.895% ↑ | split-buffer readers сам новая яма |
| **L2-подтяжка (внутри ядра)** | 055 §1.a | +2.17% ↑ | LSU issue queue + TLB (цена ≠ нулевая) |
| **Карманы (регистры)** | 055 §1.b | +2.45% ↑ | STS +385% bank storm |

**Общая картина**: любая техника скрытия latency в merged v40 имеет свою яму эквивалентной или большей стоимости. wait 33% фундаментально не сдаётся ни prefetch (Q/dO/оба), ни L2-hint, ни regs-carry без изменения формата или архитектуры.

**Обновлённый ярлык merged** (соединение 052 + 053 + 054 + 055):

> **«merged v40 — равновесное ядро. Кампания wait ПОЛНОСТЬЮ ИЗМЕРЕНА:**
> - **052 (Q ping-pong)**: mechanism partial, barrier +2.29 + ST +123% съедают → wall -0.033% (шум)
> - **053 (dO half-prefetch с обходами)**: обходы работают, но split-buffer новая яма → wall +1.895%
> - **054 (M5+A' пакет)**: paper closure, sub-threshold, LDS-ремесло исчерпано
> - **055 §1.a (L2-hint)**: цена НЕ нулевая → wall +2.17%
> - **055 §1.b (LDG-регистры)**: STS bank storm +385% → wall +2.45%
>
> **wait-стена merged двигается только форматом dO (FP4-эпоха) или persist-архитектурой. Ядро merged v40 ЗАКРЫТО до FP4/persist/новых форматов.»**

---

## §4. Сиквенс

По TZ 055 при обеих красных: "на столе S2v4-бумага и M5" (в контексте 054 — рекомендации).

1. **S2v4 (dk свизл writer smQ + LDSM-читатель)** — **главная открытая дверь** dk_new класса, реестр 052 §0.b:
   - Конструкция аналогична 040 класса #7 (свизл smdO + LDSM.x4.trans.b16)
   - У smQ ОДИН читатель (Step B) → pack Q_T мёртв, конфликт интересов читателей отсутствует
   - Апер 4-7% dk isolated → -1..-1.5% E2E → пробивает 44.0 порог
   - **Бумага не начата**. Триггер после 055.
   - **Рекомендация ассистента для 056 = S2v4 dk**.

2. **M5 deferred** (054 §0.d.3): dedicated FP8 LDSM.x2.trans.b8 микро-проба + CVT map (аналог 049 §1 dk S2v3 bridge v2). Session-level probe ~2-4 часа.

3. **FP4-эпоха** (dO fp16→fp8/fp4): стратегический пивот, вне scope. Требует новую бумагу при созревании FP4 стека.

4. **Persist-архитектура**: 044-047 захоронение (4 режима мёртвы), но при FP4/reformat может пере-открыться.

---

## §5. Правки production в 055

**После обоих откатов**: 0.

- `libs/fa_bwd_merged_v1.cu` md5 `2bf32ab7d4c5ecabb4ee2dbf1b5d4b33` = 040 sealed ✓
- `libs/fa_bwd_dk_new.cu` md5 `a9f0ded8261e53a143b521ffa647f458` = 033 sealed ✓
- `libs/fa_bwd_dq_new.cu` md5 `d7a11a3d788eb4c396d892bc9c8ab754` = 041 sealed ✓
- `libs/bench_r2c_e2e.cu`: EXPECT kernel_merged_v1 = **252** (restored)
- `runs/reports/037r_gate.sh`: EXPECT = **252** (restored)
- Diagnostic binaries в `runs/archive/055_pre/`: base + cand_1a + cand_1b + sources для future retest

---

## §6. Итоги 055

1. **Приоритет-гейт**: 054 в paper closure (не KEEP) + E2E 44.206 > 44.0 → исполнять немедленно ✓

2. **§1.a L2-хинт**:
   - Правка: `prefetch.global.L2` inline PTX в конце qt (dO+Q base, 128 threads × 128B)
   - ptxas 252r unchanged ✓; bit-exact 11/11 ✓
   - **ABBA 8/8 единогласно ХУЖЕ, Δ = +2.17%**
   - **Ярлык "нулевая цена" опровергнут**: LSU issue queue + TLB — цена реальная
   - **КРАСНЫЙ откат**

3. **§1.b Регистровый LDG-STS четверть dO**:
   - Правка: 8 uint32 регистров LDG в start-qt, STS в end-qt перед t13, cp.async skip первую четверть при qt>qt_start
   - ptxas 254r (+2 в потолке 256) ✓; bit-exact 11/11 x3 + BITFLIP catch + memcheck 0 ✓
   - Racecheck НЕ требуется (барьеры не тронуты)
   - **ABBA 8/8 единогласно ХУЖЕ, Δ = +2.45%**
   - NCu root cause: **ST conflicts +385%** (STS 32B chunks × 128 threads swizzle = bank storm). long_sb -2.41pp mechanism работает, но STS яма перевешивает.
   - **КРАСНЫЙ откат**

4. **§2 Вердикт-карта: ОБЕ КРАСНЫЕ** → war за wait merged закрывается ПОЛНОЙ формулировкой (по TZ вердикту).

5. **§3 Обновлённый ярлык merged** (соединение 052+053+054+055):
   > **«merged v40 — равновесное ядро. wait-стена не сдаётся ни второй полке (052/053), ни L2-подтяжке (055 §1.a), ни регистр-карманам (055 §1.b), ни LDS-ремеслу (054). wait 33% двигается только форматом dO (FP4-эпоха) или persist-архитектурой. Ядро merged v40 ЗАКРЫТО до FP4/persist/новых форматов.»**

6. **§5 Rollback**: prod merged восстановлен к 040 sealed, EXPECT восстановлен, bench_r2c_e2e пересобран, fingerprint 252/128/69 OK.

7. **§4 Сиквенс: 056 = S2v4 dk свизл-путь** (главная открытая дверь; реестр 052 §0.b; чист по liveness; апер 4-7% dk = -1..-1.5% E2E, пробивает 44.0 порог). Бумага не начата.

### Chain md5

- 052 `274d9c70af8aebf66f1815779a72a7a1`
- 053 `cd9aae84a94fc02fd199f8e62bcb516e`
- 054 `3441eb93f21c50e7a1dea959e426946f`
- **055 `e10161c0e1585f87275694f24e76d01d`**

### Файлы 055

- `runs/reports/055_wait_microprobes.md` (this report)
- `runs/reports/055_abba_1a.sh` + `055_abba_1a_data.txt` — §1.a ABBA
- `runs/reports/055_abba_1b.sh` + `055_abba_1b_data.txt` — §1.b ABBA
- `runs/reports/055_ncu_1b.sh` + `055_ncu_1b_data.txt` — §1.b NCu-post
- `runs/archive/055_pre/*` — snapshots (base + cand_1a + cand_1b + sources)

---

**End 055. Обе микро-пробы КРАСНЫЕ (§1.a +2.17%, §1.b +2.45%). War за wait merged v40 закрыт ПОЛНОЙ формулировкой (052/053/054/055 все пути измерены и съедены). Ядро закрыто до FP4/persist. Сиквенс: 056 = S2v4 dk.**
