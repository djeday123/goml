# 028 — dq_new long_sb диагноз: cp.async single-stage, tail negligible, D5-lite feasible

**Chain**:
- 027_dq_pack_pi.md md5: `0ea2fda332db23b3368ae176c5ec575b`

**Artifact header** (production не тронут):
```
-rw-r--r-- 13352  Jul  5 17:51  libs/fa_bwd_dq_new.cu       (sealed pre-pack, md5 683396f8…)
-rwxr-xr-x         Jul  7        libs/r1c_dq_wall            (56r, rebuild -lineinfo для профиля)
-rw-r--r--         Jul  7        libs/Makefile.r1c_dq_wall   (добавлен -lineinfo для source-page NCu)
```

Правок production нет. Только measurement + paper.

---

## 0. Occupancy-разбор

### 0.1 Launch stats (NCu Occupancy + LaunchStats sections)

| Метрика | Значение |
|:--|--:|
| Grid Size | **16384 blocks** (bh=128 × n_qt=128) |
| Block Size | 128 threads |
| Threads total | 2 097 152 |
| **Waves Per SM** | **14.52** |
| Block Limit Warps | 12 |
| Theoretical Active Warps/SM | **24** |
| Theoretical Occupancy | 50% (SMEM-limited) |
| **Achieved Occupancy** | **48.86%** (98% от теоретического) |
| **Achieved Active Warps/SM** | **23.45** |

### 0.2 Число SM и хвост волн

- Waves Per SM = 14.52 → 14 полных wave + 0.52 tail wave  
- SMs = Grid / (waves × blocks/SM) = 16384 / (14.52 × 6) = **~188 SM** (Blackwell RTX PRO 6000)
- Tail wave: 0.52 × 188 × 6 = **587 блоков в хвостовой волне** vs 1128 max/wave
- Utilization tail wave: 587 / 1128 = **52%** SMs заняты; 48% idle

### 0.3 Wall loss to tail

Модель "wave-blocks": wall = 15 wave-durations на 14.52 wave-of-work → efficiency = **14.52/15 = 96.8%** → **3.2% wall loss to tail**.

Абсолютно: 8.51 ms × 0.032 ≈ **0.27 ms**, из них 4/5 в tail effect потерь = ~0.22 ms — сопоставимо с типичным keep-порогом 3%, **но неоптимизируемо** без изменения bh или sl.

Continuous-scheduling модель точнее: каждый SM обрабатывает 16384/188 = 87.1 блоков в среднем. Slowest SM = 88 блоков. Utilization = 87.1/88 = 99.0% → **потери ~1.0%**. Реалистичнее.

**Вывод по хвосту**: 1-3% потерь — существенно ниже long_sb 14.9%. **Хвост не главное горло**.

### 0.4 Не_selected / selected баланс

Из 024 pre-pack:
- `selected` = 5.01% (варп был выбран и issue успешно)
- `not_selected` = 8.94% (варп готов, но scheduler не выбрал)
- Ratio not/(sel+not) = **64%**

**Интерпретация**: 64% циклов, где варп доступен но не выбран → **resource contention или dependency chains**. Compiler-scheduled dependencies (LDS→MMA staging) занимают функциональные единицы.

### 0.5 Сколько варпов прячут латентность в устоявшемся режиме

- 23.45 warps/SM active × 4 schedulers = **5.86 warps/scheduler** (peak 12)
- long_sb 14.88% × 23.45 warps ≈ **3.5 warps waiting on LDG** per cycle
- Остальные ~20 warps в active-loop, но 8.94% not_selected + 9.11% wait → **effective throughput warps ≈ 20 × (1 − 0.18) = 16 warps** реально issue instructions per cycle per SM

**Ключ**: warp-headroom **есть**, но не спасает — cp.async wait это **block-level barrier** (`cpa_wait<0>` + `__syncthreads`), **все 24 warps простаивают одновременно**. Warp concurrency латентности не прячет для этого паттерна.

---

## 1. cp.async-конвейер dq_new (из кода)

### 1.1 Структура текущего конвейера (fa_bwd_dq_new.cu:105-169)

**Per kt-iteration (n_kt = 128 итераций)**:
```
cp.async K → smK_area     (8 KB, 512 chunks, 4 chunks/thread)     ← 4 LDGSTS/thread
cp.async dS_nat → smdS    (4 KB, 256 chunks, 2 chunks/thread)     ← 2 LDGSTS/thread
                                                                 (+ partial STS.U8 fallback CANARY)
cpa_commit();                                                    ← 1 commit group
cpa_wait<0>();                                                   ← wait ALL groups
__syncthreads();                                                 ← BARRIER #1 (K + dS ready)
```

**Число commit groups в flight между «заказал» и «жду» = 0**.  
**Глубина конвейера = 1 (single-stage)**.

### 1.2 SASS-attribution (из NCu source-page dump)

Всего 3 memory-fetching инструкции в kernel_dq_new SASS:
- **LDGSTS.E.BYPASS.128** @0x430 → cp.async K (major group, 8 KB traffic)
- **LDGSTS.E.BYPASS.128** @0x500 → cp.async dS_nat (minor group, 4 KB)
- **LDG.E.U8** @0x640 → partial CANARY fallback (predicated, rarely fires)

**long_sb концентрация**: Ожидаемо на **DEPBAR инструкции сразу после `cpa_wait<0>`**, ждущей завершения обеих LDGSTS. Плюс на первой LDS (Phase 1.5 read) — стойл до готовности данных.

**По объёму**:
- K load: **8 KB/kt × 128 kt = 1 MB/block** LDG traffic
- dS load: **4 KB/kt × 128 kt = 512 KB/block** LDG traffic
- Total per block: **1.5 MB**

Per-warp: 128 kt × 6 LDGSTS/kt / 32 = **24 LDGSTS/warp/qt** (полный tour через kt-loop).

### 1.3 Место для углубления БЕЗ +байтов?

**SMEM footprint текущий**: 13824 B/block = smK_area(8704) + smdS(5120).

**D5-lite гипотеза**: перестановка `commit`/`wait` без double-buffer, **без +SMEM**.

**Возможные варианты**:

#### (α) Разделить cp.async K и dS на **два commit_group** с работой между
```
cp.async K, commit_group;         // group_0
cp.async dS_nat, commit_group;    // group_1
cpa_wait<1>;                      // wait until 1 group left (K done, dS still in flight)
sync;                             // K ready
// Phase 1.5 K→K_T transpose (uses only smK_area, does NOT touch smdS)
cpa_wait<0>;                      // wait dS
sync;                             // dS ready
// MMA-C
```
**Overlap**: Phase 1.5 (~4 K CHUNK reads + write) выполняется параллельно с cp.async dS_nat.  
**Гейн**: время dS cp.async - время Phase 1.5. dS = 4 KB, Phase 1.5 = few LDS + PRMT + STS.U8 (сотни ns).  
**Стоимость**: **+1 барьер** (barrier stall может вырасти).

#### (β) Заказать K/dS для **kt+1** перед wait текущего kt
Требует **double buffer** SMEM (smK_area × 2 + smdS × 2 = 27648 B/block).  
→ **D5-full**, не подходит для D5-lite.

#### (γ) Prefetch K для kt+1 внутри MMA-C loop
- Нужен **отдельный smK_area buffer для K+1** = +8704 B/block = 22528 B/block
- 102400 / 22528 = 4.5 → **4 blocks/SM** (регресс 6→4)
- **Vugar правило: 6→4 автоматический регресс** ← НЕ ok

**Итог**: **D5-lite (α)** — единственный вариант без +SMEM.  
**D5-full (β/γ)** = регресс blocks/SM, требует отдельного решения Vugar.

---

## 2. Бумага-вилка (таблица, правок нет)

### D5-lite (α: two commit_group, работа между)

| Параметр | Значение |
|:--|:--|
| SMEM Δ | 0 (unchanged) |
| Blocks/SM | 6 (unchanged) |
| Barrier count | 4 → **5** (+1) |
| long_sb прогноз | 14.88 → **11-13** (drop 2-4 pp) |
| barrier прогноз | 9.52 → **10-11** (+0.5-1.5 pp)  |
| Wall прогноз | 8.51 → **8.32-8.44** ( **-1.0 до -2.2%** ) |
| Риск | Barrier рост сожрёт часть long_sb win |
| Register cost | 0 (структурная перестановка) |
| Bit-exact | preserved (только reorder, no math change) |

### D5-full (β/γ: double buffer prefetch)

| Параметр | Значение |
|:--|:--|
| SMEM Δ | +8704 или +13824 B/block |
| Blocks/SM | **4** (6→4 регресс, **автоматически stopped by Vugar правило**) |
| Achieved warps | 24 → 16 (-33% occupancy) |
| long_sb прогноз | 14.88 → **6-9** (drop 6-9 pp, лучше D5-lite) |
| Wall прогноз | 8.51 → **8.20-8.40** ( **-1.3 до -3.6%** ) |
| Риск | Occupancy drop может нейтрализовать пользу; блок-регресс = отдельное решение |
| Register cost | +2-4r (dual-buffer address bookkeeping) |
| Bit-exact | preserved |

### Software-pipelining вычислений поверх wait

| Параметр | Значение |
|:--|:--|
| SMEM Δ | 0 |
| Blocks/SM | 6 |
| Гейн | Можно предвычислить PI_V(n_d) для всех ni, k_j addresses и т.д. между `cpa_commit()` и `cpa_wait<0>()` |
| Проблема | Работы **очень мало** между текущими commit/wait (2-3 инструкций); почти всё зависит от загруженных данных |
| Wall прогноз | ~ **-0.3 до -0.7%** (маргинально) |
| Register cost | +2-5r |
| Риск | Marginal gain vs implementation complexity |

---

## 3. Рекомендация: **D5-lite (α)** первым

**Обоснование**:
- **Zero SMEM cost** → 6 blocks/SM сохраняется
- Прогноз wall **-1.0..-2.2%** — под порогом keep, **но в вилке ABBA (2-3%) при удачном исходе**
- Bit-exact сохраняется автоматически
- **Комбинирует с ранее замороженным pack+π** (архив `fe02e056…`) — если D5-lite даёт даже 1.5%, а pack+π ~1.5%, суммарно **связкой можно достичь ≥3%**
- Barrier налог (+1) — умеренный риск, monitorable

**Прогноз ДО замера** (регистрирую):
- D5-lite соло: **-1.5% median** (в вилке [-1.0, -2.2])
- D5-lite + pack+π (реанимация архива): **-2.5..-3.5%** (сумма минус 20% overlap) — **пересекает 3% keep**

**Не рекомендую**:
- **D5-full** — автоматический регресс blocks/SM, отдельное решение Vugar требуется
- **Software-pipelining** — маргинальный gain, высокая complexity

---

## 4. Файлы

- Occupancy dump: NCu section Occupancy + LaunchStats
- Source page dump: `/tmp/dq_source_lineinfo.ncu-rep` (temp, для source-line attribution)
- Скрипты: `runs/reports/028_occupancy.sh`, `028_long_sb_lines.sh`
- Production не тронут: fa_bwd_dq_new.cu md5 `683396f8…` unchanged.

Chain md5: 027 `0ea2fda3…` → **028 `<computed>`**

---

**End 028.**  
Ожидаю решение Vugar:
1. D5-lite (α) первый шаг — правка + гейты + wall + возможно ABBA
2. Связка D5-lite + реанимация pack+π (архив `fe02e056…`)
3. D5-full — обсуждение осознанного размена blocks/SM (6→4)
4. Смена ландшафта — переключение на другое ядро
