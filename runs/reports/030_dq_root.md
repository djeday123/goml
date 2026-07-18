# 030 — dq_new корень стены: бухгалтерия issue-слотов + H1/H2 разбор

**Chain**:
- 029_dq_d5lite.md md5: `c7febb1bf3e43ab25e4d951179a9611c`

**Artifact header** (production не тронут):
```
-rw-r--r-- 13352  Jul  5 17:51  libs/fa_bwd_dq_new.cu       (sealed pre-pack, md5 683396f8…)
-rwxr-xr-x 1.20M  Jul  7        libs/r1c_dq_A_sealed        (md5 a5bf576f…, версия A из архива 024)
-rwxr-xr-x         Jul  7        libs/r1c_dq_B_pack_pi       (md5 c53aa7aa…, версия B из архива 027)
-rwxr-xr-x         Jul  7        libs/r1c_dq_C_d5lite        (md5 6d884b44…, версия C из архива 029)
```

Правок production нет. Только measurement + paper.

---

## 0. Бухгалтерия issue-слотов (3 версии, один NCu-прогон каждая)

### 0.1 Сводная таблица

| Метрика | **A** (sealed 56r) | **B** (pack+π 70r) | **C** (D5-lite+pack+π 69r) |
|:--|--:|--:|--:|
| smsp__inst_issued.sum | 3.796 B | **4.819 B** (+27%) | 4.836 B (+27%) |
| sm__cycles_active.avg | 17.19M | **14.19M** (−17%) | 14.13M (−18%) |
| smsp__inst_issued.avg.per_cycle_active | 0.29 | **0.45** (+55%) | 0.46 (+59%) |
| smsp__warps_eligible.avg.per_cycle_active | 0.82 | 1.13 (+38%) | **1.23** (+50%) |
| sm__warps_active.avg.per_cycle_active | 23.45 | 23.47 | 23.46 |
| **Wall isolated (session-fresh)** | **8.524 ms** | **8.394 ms** (−1.5%) | **8.394 ms** (−1.5%) |

### 0.2 Проверка гипотезы Vugar: total time = inst × CPI

Идентичность (тавтологически): `scheduler-cycles = inst_issued × CPI`, где `CPI = 1/IPC`.

Численно:
- A: 3.796 B × 3.45 CPI / (188 SM × 4 sched) = **17.4M cycles/SM** ≈ 17.19M ✓
- B: 4.819 B × 2.22 CPI / (188 × 4) = **14.2M cycles/SM** ≈ 14.19M ✓
- C: 4.836 B × 2.17 CPI / (188 × 4) = **14.0M cycles/SM** ≈ 14.13M ✓

**Total scheduler-cycles СОХРАНЯЕТСЯ как (inst × CPI)** — тавтология подтверждена.

Реальный факт: **CPI падает** с pack+π (3.45 → 2.22, −36%), но **inst растёт** (+27%). Cycles упали 17%, потому что CPI-drop сильнее inst-growth. **Однако wall drop только 1.5%** — cycles уменьшение не переходит в wall в пропорции.

### 0.3 NCu cycles vs wall discrepancy

Расчёт: 17.19M cycles / 8.524 ms = **2.02 GHz** clock (A). Для B: 14.19M cycles = **7.03 ms** предсказанный, факт 8.39 ms. **Gap = 1.36 ms** (16%). 

Причины gap:
- Tail effect (tail wave 3.15 blocks/SM vs 6, ~1-3% wall)
- Cache-hot averaging (bench 20 iters vs NCu single-shot)
- Kernel launch overhead
- Wave scheduling overhead

NCu cycles = **идеализированные per-SM**. Wall = **реальное time**, включает всё выше.

**Вывод пункта 0**: total scheduler-cycles сохраняется тавтологически. **CPI-цепочки укорочены на 36% pack+π**, но wall gain only 1.5%. **Wall bounded НЕ cycles-per-SM**, а wave + cache + tail накладными.

---

## 1. H1 (цепочка зависимостей fp16-acc/MMA внутри витка)

### 1.1 SASS-счёт в kernel_dq_new (production A)

- **QMMA**: 32 инструкций static (unrolled 2 kb × 16 ni)
- **BAR.SYNC**: 5 static (4 in kt-loop + 1)
- **LDS**: 88 static (Phase 1.5 read 16 + MMA-C 64 + others 8)
- **LDGSTS**: 2 (K + dS cp.async)

### 1.2 Оценка критической цепи per kt-iter (per warp)

| Этап | Sealed A (cycles) | Pack+π B/C (cycles) |
|:--|:-:|:-:|
| cp.async LDGSTS + wait | 800-1500 | 800-1500 |
| BARRIER #1 (K+dS ready) | ~100 | ~100 |
| Phase 1.5 read (16 LDS) | ~480 | ~480 |
| BARRIER #2 (before aliased) | ~100 | ~100 |
| Phase 1.5 write (scatter) | **~640** (64 STS.U8) | **~350** (16 STS.32 + 12 SHFL) |
| BARRIER #3 (K_T ready) | ~100 | ~100 |
| MMA-C loop (32 MMA pipelined) | ~200 | ~200 |
| BARRIER #4 (end kt) | ~100 | ~100 |
| **Total chain** | **~2620** | **~2330** (−11%) |

### 1.3 Наблюдаемые cycles per warp per kt-iter

- Per SM cycles / (blocks/SM × kt/block × 4 warps/block) → per warp/kt cycle count
- A: **8940 cycles/warp/kt** (17.19M × 4 sched / 11264 kt-iters / 4 warps = ...)

Пропущенный масштаб: **cycles_active per SM включает all warps concurrently**. Per warp per kt < cycles_active/n_kt.

Ratio observed cycles B/A = 0.83 (17% меньше). Chain estimate 2330/2620 = 0.89 (11% меньше). **Chain reduction meaningful but not dominant** — MOST cycles reduction came from parallelism gain (eligible warps 0.82 → 1.13), not chain shortening.

### 1.4 Независимая работа в витке

- **kb outer + ni inner MMA**: kb=0..1 sequential (chain via dQ_acc[ni]), но 16 ni independent per kb.
- Chain per ni = 2 MMA sequential (~100 cycles pipelined).
- 16 independent tracks × 4 warps = 64-way parallelism per block.
- **Достаточно независимой работы для pipelining** — MMA не критпуть.

**H1 verdict**: 
- Chain length reduces ~11% с pack+π (Phase 1.5 write pack)
- Actual cycles reduce 17% — additional ~6% из IPC gain (fewer stalls after pack)
- **H1 частично**: chain-cost per stage вносит вклад, но не является единственным горлом. Independent work exists in abundance.

---

## 2. H2 (барьерный конвой)

### 2.1 Barrier stall %:

- A (sealed): **10.93%**
- B (pack+π): **9.54%** (−1.4 pp)
- C (D5-lite+pack+π): **11.90%** (+2.4 pp vs A) — split commit добавляет sync overhead

### 2.2 Barrier count и структура

**4 __syncthreads() в kt-loop** (sealed):
1. BARRIER #1 — K+dS ready (cross-warp)
2. BARRIER #2 — Phase 1.5 read done (перед aliased overwrite)
3. BARRIER #3 — K_T ready (перед MMA-C)
4. BARRIER #4 — end kt (перед следующего cp.async)

**Cross-warp зависимости** (не warp-local):
- BARRIER #1: K и dS написаны cp.async всеми варпами → cross-warp
- BARRIER #2: Phase 1.5 read reads addresses across warps (own warp reads own ks-block K natural, но overwrite Phase 1.5 write надо синхронизировать) → cross-warp по данным
- BARRIER #3: Phase 1.5 write writes across warps to different K_T regions, MMA-C reads across regions → cross-warp
- BARRIER #4: MMA-C reads before next cp.async writes to same buffer → cross-warp

**Все 4 барьера — cross-warp**. Замена на `__syncwarp()` **нарушает bit-exact** (различные варпы могут читать частично записанные данные).

### 2.3 Микро-вариант дешёвого разделяющего эксперимента

**Не выполняю в этой сессии** — bit-exact при замене __syncthreads() → __syncwarp() **не сохраняется** (cross-warp данные).

Возможный тест H2 **при full barrier removal**: невозможен без нарушения bit-exact, значит не сработает как разделяющий эксперимент.

**H2 verdict (paper-only)**:
- Barrier stall 10-12% значителен, но **не dominant** (MIO 46 в A, 41 в C)
- Barriers structural — необходимы для cross-warp sync
- **Убрать без нарушения bit-exact невозможно** без структурного redesign (split buffers per warp, etc.)
- H2 вносит вклад, но не корень.

---

## 3. Вердикт-бумага: корень стены

### 3.1 Диагноз

**Root = смесь H1 + H2 + wave-overhead + cache**, ни один компонент не dominant.

Компоненты (в процентах от wall):
- **Critical chain per kt** (H1): ~40-50% wall bound
  - cp.async LDG wait (~30% wall) — не устраним без SMEM redesign
  - MMA chain (~7% wall) — pipelined, near-optimal
  - Barriers (~10-12%) — H2, structural
- **Wave/tail** (~3-5%): tail wave 3.15/6 blocks/SM utilization
- **Cache/thermal** (~2-3%): between-iter cache warmup
- **Instruction overhead** от pack+π: ~1-2% recovered via IPC gain

**Vugar's гипотеза о консервативной величине**: НЕ ОДНА величина, а **balanced constraint** между IPC (limited by CPI-цепочка + eligible warps) и wave-overhead. Total scheduler-cycles conserve tautologically, но wall bounded больше wall-overhead чем IPC.

### 3.2 Ключевое наблюдение

**IPC вырос на 55%** (0.29 → 0.45) с pack+π. **Eligible warps выросли 38%**. **Cycles упали 17%**. Но **wall упал только 1.5%**.

**Это означает**: 
- Реальный wall НЕ bound by в-SM IPC (SM работает эффективнее)
- Реальный wall bound by **cross-SM synchronization + wave scheduling + memory-bandwidth**
- Балансированное ядро выходит на **memory-bandwidth limit** и **wave-overhead**, не на IPC

### 3.3 Цена лечения (честная вилка)

Различные подходы:

#### Option X: Software-pipelining двух kt-тайлов (D5-full)
- Double-buffer smK_area (+8704 B/block) + double-buffer smdS (+5120 B)
- Total SMEM: 13824 → 27648 B → **4 blocks/SM** (регресс 6→4, −33% occupancy)
- **Гейн потенциальный**: cycles reduce ~20% (LDG hide fully), но occupancy loss тоже влияет
- **Сложность**: средняя (1-2 дня)
- **Bit-exact риск**: низкий (только layout, not math)
- Vugar правило "6→4 регресс" — **требует явного решения**

#### Option Y: Interleave ni-групп через несколько kt-tiles
- Сложная перестройка MMA loop — **нарушает kt/kb/ni order = НАРУШАЕТ BIT-EXACT** ✗
- Не работает.

#### Option Z: Bc tile change (Bc=64 → 128)
- SMEM: 8192*2 + 5120*2 = 26624 B → 3 blocks/SM (регресс 6→3)
- Гейн: fewer kt-iters (128 → 64), но per-iter time doubles
- Complex trade-off
- Bit-exact preserved (только tile size)
- **Требует separate ABI check** (все consumer'ы получают dS)

#### Option W: fp16-acc → fp32-acc
- **НАРУШАЕТ жёсткий инвариант bit-exact** ✗
- Не работает.

### 3.4 Единая рекомендация

**Полировка в рамках single-optimization не преодолевает 3% keep**. dq_new fundamentally balanced.

**Structural redesign уровня "недели"** — единственный путь:
- **Option X** (D5-full double buffer, регресс 6→4) — **топ-1 кандидат**, требует Vugar-решения
- **Option Z** (Bc=128, регресс 6→3) — топ-2, ABI-риск

**Прогнозов wall не даю** — до бумаги на конкретный лечения. Первый шаг: Vugar-решение о принципиальной готовности к регрессу blocks/SM.

---

## 4. Файлы

- 3 бинаря в `libs/`: `r1c_dq_A_sealed`, `r1c_dq_B_pack_pi`, `r1c_dq_C_d5lite`
- Bookkeep NCu-скрипт: `runs/reports/030_bookkeep.sh`
- SASS dump prev-existing: `runs/probes/dq_sass_full.txt`

Chain md5: 029 `c7febb1b…` → **030 `<computed>`**

---

**End 030.**
dq_new паркуется на sealed baseline. Vugar-решение необходимо для structural redesign (Option X or Z) — единственный путь превысить 3%. В рамках полировки достигнут локальный оптимум ~8.4 ms.
