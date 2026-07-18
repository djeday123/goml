# B2 HANDOFF — FlashAttention FP8 Backward, sm_120a Blackwell consumer

**Дата:** 2026-06-12. Передача сессии перед началом B2 (первое backward-ядро Pass 2 dK+dV).

---

## 0. КАК ПОЛЬЗОВАТЬСЯ ЭТИМ ДОКУМЕНТОМ

Это handoff между сессиями. Вставляй первым сообщением в новую сессию — она поднимет полное состояние проекта без переспрашиваний. Не сокращай, не редактируй. Здесь есть всё: что построено, почему именно так, что закрыто (и под какими цифрами), какие калибровки железа, как валидировать, какие риски в B2.

**Базовый принцип проекта**: production-grade код, никаких экспериментов под видом продакшна. Каждое утверждение подкреплено цифрой/механизмом. Изменения без NCu/SASS/ptxas-подтверждения не идут.

Конец документа — одной строкой: *«Следующий шаг: B2, Pass 2 (dK+dV), старт с корректности.»*

---

## 1. ИДЕНТИФИКАЦИЯ ПРОЕКТА

### Что строим
**libfa_sm120** — production FP8 FlashAttention forward + backward для NVIDIA Blackwell consumer GPU (compute capability 12.0, sm_120a). Цель: production-quality kernel + публичная C-ABI/Go/Python binding library, не research-демо.

- **Forward** — закрыт на этапе SHIP-1: peak **652.40 T ± 0.87** (bh=128 sl=8192 hd=128 wnd=0), **647.14 T ± 0.64** (bh=64 sl=8192). Champion v121r.
- **Backward** — в работе. B1 фундамент (математика, CPU/FP64 эталоны, L-патч на forward) полностью готов. Стоим на пороге B2: первое реальное backward-ядро (Pass 2 = dK + dV).

### Железо
- **GPU**: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- **Compute capability**: 12.0 (sm_120a — "a" суффикс архитектурно-специфический code-gen)
- **SM count**: 188 SMs
- **FP8 e4m3 теоретический peak**: ~960 TFLOPS (m16n8k32 QMMA, ×2 per cycle FFMA-эквивалент)
- **HBM**: 96 GB
- **Driver**: 580.159.03 (на момент 30-run champion бенча)
- **Card temperatures**: 30-46°C под нагрузкой, sm_clock 2617-2677 MHz при ~250W

### Софт-стек
- **CUDA Toolkit**: 13.1 (`/usr/local/cuda-13.1/`)
- **NVCC**: `/usr/local/cuda-13.1/bin/nvcc`. Флаги: `-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xptxas=-v`
- **Nsight Compute**: `/usr/local/cuda-13.1/bin/ncu` (старый системный `/usr/bin/ncu` = 2021.3.1, **не использовать**: не знает sm_120a-метрики)
- **PyTorch**: для cross-check FP32/FP64
- **Go**: 1.22+ (для cgo binding `gofa`)
- **Python**: ctypes single-file binding (без torch на import)

### Репозиторий
- **GitHub**: https://github.com/djeday123/fa-blackwell-fp8
- **Тег v0.1.0** выпущен 2026-06-12 13:03 UTC. Title: "v0.1.0 — FlashAttention FP8 forward, peak 652.4 TFLOPS"
- **Asset релиза**: `champion_30x_v121r_bh64_bh128_sl8192.log` (30-run measurement, attached)
- **License**: Apache-2.0
- **Topics**: flash-attention, fp8, cuda, blackwell, sm120, nvidia, attention, transformer, gpu, inference, fp8-e4m3, cpp, go, python, ml-systems, gpu-kernels
- **Author commit identity**: Vugar Bakhshaliyev <39285558+djeday123@users.noreply.github.com>

### Все ключевые пути на gpuserver

| Путь | Что |
|---|---|
| `/root/repos/fa-blackwell-fp8/` | Опубликованный репо (production v121r, под тегом v0.1.0) |
| `/data/lib/podman-data/projects/goml/` | Research workspace (вся история v1→v122, эталоны, эксперименты) |
| `/data/lib/podman-data/projects/goml/libs/` | B1 артефакты (fa_bwd_cpu_reference*, fa_bwd_common.cuh) |
| `/data/lib/podman-data/projects/goml/libs/fa_sm120/` | Production library + train-вариант (НЕ закоммичен) |
| `/data/lib/podman-data/projects/goml/runs/` | Все NCu-репорты + champion-логи |
| `/data/lib/podman-data/projects/claude-dashboard/workspace/8ffe507afe53/` | Sandbox workspace (b1fix-checker, lpatch test/bench harness) |

---

## 2. ГДЕ МЫ В ПРОЦЕССЕ — ТЕКУЩЕЕ СОСТОЯНИЕ

```
Forward [DONE]
├── v121r production kernel (255 regs, 0 spill, 0 stack)
├── Dispatcher (8 ниш для hd=64/128, causal, window)
├── C-ABI library + Go cgo + Python ctypes
├── 29-case dispatcher unit test
├── 30-run same-thermal champion log
└── v0.1.0 published, Apache-2.0

B1 [DONE]
├── B1.1 Математика градиентов (Tri Dao Variant 3)
├── B1.2 CPU FP32 reference (изначально 93.3%, после B1-FIX = honest 100% above-floor)
├── B1.3 fa_bwd_common.cuh (10 кирпичей: cp.async, swizzles, transpose_v, mma, mbarrier...)
├── B1.4 Геометрия Pass 2 (главный вывод: атомики НЕ нужны, block-residency)
├── B1.5 Вердикт-вилка: GO
├── B1-FIX Strict checker (FP64 probe + hybrid tol → 100% above-floor)
├── B1-FIX Multi-seed (4 seeds) + Mutation (boundary 0.2%) + PyTorch FP32 cross-check
├── B1-FIX-EXTRA FP64 golden backward (max abs_err 2.8e-16 = 1 FP64 ULP vs PyTorch FP64)
└── L-патч v121r-train (Δregs=0, cost 0.244%, корректность 8/8 + long-form sl=8192)

B2 [NEXT — здесь стартует новая сессия]
├── Pass 2 = dK + dV
├── Монолит → ptxas → если spill=0 победа, иначе split fallback
├── M_TILES≥2 обязательно, аккумуляторы FP32 в регистрах блока
├── Корректность ПЕРВАЯ: vs FP64 golden, 8 форм + sl=300 wnd=96
└── Цель 320-380T (0.5-0.6× forward). <250T → NCu разбор прежде чем оптимизировать

B3 [планируется] Pass 1 = dQ (симметрично Pass 2, dispatch по Q-tile вместо K-tile)
B4 [планируется] End-to-end correctness Pass1+Pass2 на 8 формах
B5 [планируется] 30-run same-thermal bench backward, формализация в C-ABI
```

---

## 3. FORWARD — ЧТО ПОСТРОЕНО И ПОЧЕМУ ИМЕННО ТАК

### Финальный диспетчер (8 ядер, табличный)

Правило `grid = bh × ceil(sl / Br)` где Br = 128 (для hd=128 семейства) и 64 (hd=64 семейство). Каждое ядро занимает свою нишу по `(bh, sl, hd, causal, window)`.

| hd | Условие | Кернел | Почему |
|---|---|---|---|
| 128 | `wnd=0`, peak grid (bh≥32 ∨ sl≥4096) | **v121r** | Sr→half2 softmax pipeline, **+2-6% над v121** |
| 128 | `wnd>0` (sliding/causal window) | **v121** | window champion, address-arith hoisted |
| 128 | `bh=4, sl≤2048, wnd=0` (wave-tail) | **v122** | Br=64 / M_TILES=1, equalises partial waves |
| 128 | `bh ∈ {4..16}, sl≤4096, wnd=0` (mid-grid) | **v118** | 1-producer + 3-consumer warp-specialised |
| 128 | `bh=4, sl=8192, wnd=1024` (sliding niche) | **v117b** | partial top-sync + localfix smK |
| 128 | narrow boundary configs | **v96b** | universal baseline, smK localfix |
| 64 | peak grid | **v89** | P-in-registers, shfl-based gather |
| 64 | wave-tail | **v80b** | Br=64 wave-tail, V cp.async overlap |

Cells «narrow boundary configs» = {bh=16 sl=4096, bh=32 sl=2048, bh=128 sl=2048, bh=64 sl=4096} — где peak v121r внезапно проседает (узкий регион). Зафиксированы 29-case unit test'ом в `test/test_dispatch.c`.

### Пять работавших ходов

1. **FP8 MMA (m16n8k32 e4m3.e4m3.f16)** — базовый кирпич, +2× throughput над FP16.
2. **localfix** (`smK[2]` array → arithmetic stride `smK_base + s * K_STRIDE_BYTES`) — устранил LDL.64 на dynamic index в hot loop, **+5% peak v96 → v96b**. LDL = 8× хуже LDS, catastrophic. Это пример **work-reduction** (убрали инструкцию с критического пути), не redistribution.
3. **Address arithmetic hoisting v121** — вынесли swizzle-инварианты (`gid_lane_base`, `gid3`, `gid7`, `swz_byte`-частные) в **prefix-блок батча**. ADDR-регистров 268 → 258. **+3-5% peak**.
4. **Sr→half2 softmax conveyor (v121r)** — Sr хранится как `__half2[8][2][2]` вместо `float[8][2][4]`. Все softmax operations (rescale, add nm, ex2.approx) идут f16x2 в нативе. Half-precision arithmetic vs scalar f32 даёт ~2× throughput на этих микро-операциях.
5. **Dispatcher с 8 нишами** — правильное ядро под форму, не одно «компромиссное».

### Закон критического пути (ГЛАВНЫЙ ПРИНЦИП)

> Каждая инструкция в hot-loop должна быть либо **MMA** (полезная), либо **неустранимо нужна** (transpose-tax, sync). Любая случайная арифметика на критическом пути SASS = регрессия. Перед добавлением кода в hot-loop спрашивать: **уйдёт ли это в неблокирующий префикс через hoisting** (v121 stage1)? Если нет — это уже не оптимизация.

Этот закон родился из **v94/v95 span-compression** — компилятор уже на оптимуме liveness span, ручные правки → −2-4%. Из **v91 KV combined preload** — force-pattern сократил compiler scheduling flexibility → "wall of loads + wall of MMAs" sub-optimal. Из **v98 K-preload regression −0.68%** — explicit b0_arr[8] форсит rigid op order вместо compiler interleaving, mio_throttle +0.52pp.

### Производственный профиль v121r (NCu MemoryWorkloadAnalysis_Tables)

NCu подтвердил production-state на bh=4096 sl=128 (grid 4096 × 1 thread = bh=128 sl=8192 equivalent):

| Metric | Value | Что говорит |
|---|---:|---|
| DRAM Throughput | **2.88%** | Не HBM-bound, ОГРОМНЫЙ resource margin |
| L1TEX Hit Rate Global Loads | 0% | Q/K/V через `cp.async.cg` (cache-global bypass) |
| L1TEX Hit Rate Global Stores | 93.08% | Output writeback через L1 |
| **L2 Hit Rate Loads** | **97.57%** | KV slabs reuse near-perfect |
| L2 Hit Rate Stores | 93.22% | Output stays in L2 |
| Avg Bytes/Sector Global Stores | 8/32 (**25%**) | Главный open lever (FP16 output coalescing) |
| **Local memory (all metrics, 18 каналов)** | **0** | ptxas-гарантия + NCu runtime подтверждены |
| Shared bank conflicts (load) | 33.36% | Structural FP8 layout floor |
| Shared bank conflicts (store) | 62.60% | transpose_v + smP write structural |

**Stall composition** (NCu sm_120a):
- wait 37.77% (top stall — inherent latency hiding limit at 8 warps/SM)
- math_pipe_throttle 8.86%
- short_scoreboard 6.63%
- mio 4.48%
- barrier 8.61%
- lg_throttle 3.31%
- dispatch_stall 2.92%

Этот профиль = **архитектурный fixed-point**. Все 4+ source-level levers (v98 K-preload, v102 3-blocks, v106 Br=96, v107 persistent, v115 SHFL transpose, v117 partial sync) дали NEUTRAL/-1% потому что redistribute stalls, не reduce work.

### Кладбище закрытых направлений (НЕ предлагать заново)

| Направление | Результат | Механизм отказа |
|---|---|---|
| warpspec 1P+3C (v111) | **−14.2%** PEAK (484T vs 568T) | Структурно 25% меньше MMA throughput (3 compute warps вместо 4). Niche: small-batch sliding window +20%. |
| 3 blocks/SM via SMEM redesign (v102) | **−2.7%** PEAK | Occupancy DID load (8→12 warps), но math_pipe +9pp + short_scb +4pp + mio +3pp = Eligible -6.60pp. Architecture lacks wgmma scaling. |
| Br=96 / 6-warp (v106) | **−15.6%** PEAK | 50% больше warps не помогает MMA-floor'у; mio+barrier+math_pipe выросли +20pp |
| ldmatrix.trans / SHFL transpose (v115) | **−28%** PEAK (correctness OK) | Eligible почти как baseline, но больше инструкций per output (LDS+2SHFL+permute+STS) → scheduler can't hide |
| setmaxnreg.{dec,inc}.sync (sm_120a) | **silent no-op** | Компилируется, SASS пустой. Reg redistribution between warpgroups недоступен. FA3-style 1P+3C reg budget split физически невозможен. |
| persistent kernel (v107) | **−2%** PEAK | HW scheduler уже handles partial waves efficiently. cycles_active 87% UNCHANGED. |
| TMA point-replacement (v82 Q, v83 K) | оба neutral/-1.5% | TMA не помогает на reg-bound kernel'е с уже-оптимальным cp.async pattern. SHFL dispatch single-lane vs 128-thread parallel cp.async |
| ILP softmax (v85, v86) | оба neutral | Softmax не был source stall'а (proven через NCu pmsampling) |
| K-preload v98 hd=128 | **−0.68%** (10-run statistically significant) | mio_throttle +0.52pp dominated short_scb savings, "wall of LDS then wall of MMAs" pattern interferes |
| KV combined preload (v91) | **−0.2..-2.2%** | Force-pattern reduces compiler scheduling flexibility |
| 4 blocks/SM via cool-temp extraction (v93) | falsified | Cool temps real but absorbed into HOT-schedule gaps at LB=3 budget |
| Span compression (v94/v95) | both neutral | Compiler at optimum for liveness span AND packing |
| f32 accumulator in QK MMA | irrelevant | f16 accum probed both old path AND sm_120a kind::f8f6f4 — works |
| FP4 m16n8k64 backdoor on sm_120 consumer | **doesn't exist** | INVALID SHAPE-field bits trap as illegal. No hidden FP8 m16n8k64. |
| Span compression (v94/v95) — точечная reg реg-savings | both neutral | Span compression direction exists (v87 −8 proved), но v89 уже absorbed all such gains |

---

## 4. КАЛИБРОВКИ КАРТЫ (КРИТИЧНО ДЛЯ B2 ГЕОМЕТРИИ)

### Регистровый бюджет
- **Потолок**: 255 регистров per thread на sm_120a.
- **Стоимость регистра у потолка**: ≈**0.5-1% wall** (v96c доказал: +8 регистров → -2% wall).
- **Правило Δregs ≤ +2** при правках hot-loop production-ядра.
- **Spill = катастрофа**: LDL ~250 cycles (как L2), STL аналогично. v102 spill 404 B → math_pipe + short_scb стали top stalls.

### Util формула
**`tensor_util ≈ M_TILES / (M_TILES + 0.6)`** — линейная функция числа M-tiles на варп.
- M_TILES=2 → 2/2.6 ≈ **0.77** (production v121r, v96b)
- M_TILES=1 → 1/1.6 ≈ **0.625** = **−37% peak** (v122 wave-tail, OK для маленьких grid)
- M_TILES=3 — не пробовали (SMEM upper bound)

**M_TILES≥2 ОБЯЗАТЕЛЬНО** для production-ядер. Это калибровка v122 ровно показала.

### Латентности
| Уровень | Cycles |
|---|---:|
| Register port | 0 |
| SMEM (LDS) | ~30 |
| L2 | ~250 |
| DRAM (HBM) | ~600 |

### SMEM cap на sm_120a
- **Opt-in maximum**: **100 KB / SM** (на sm_120a через `cudaFuncAttributeMaxDynamicSharedMemorySize`).
- Production v121r: 48.5 KB → 2 blocks/SM (LB=2).
- 3 blocks/SM требует ≤33 KB/block — для backward ОЧЕНЬ tight.

### f16x2 непрерывность священна
- Все softmax operations должны держать данные в `__half2`-packed.
- `ex2.approx.f16x2` = ~2× throughput vs `ex2.f32` (v79 lever 4).
- `cvt.rn.satfinite.e4m3x2.f16x2` — нативный quantize 2 fp16 → 2 fp8 одной инструкцией.

### QMMA m16n8k32 fp8 e4m3 → f16 acc (микробенч verified)
- **Latency** L = **29 cycles**
- **Throughput** T = **16.8 cycles** (per warp on 1 SMSP)
- За warp на 1 SMSP: ~467 GFLOPS FP8.
- Card peak ≈ 4× (4 SMSPs/SM × 188 SMs × cycle-balance) = ~960 TFLOPS theoretical.
- v96b peak 595T = 62% of card ceiling. **Tensor util 45.4% explained by phase structure** (softmax/transpose/barriers ~55% non-MMA time), не chain depth.

### FP4 микробенч
- e2m1 m16n8k64: L=29, T=17.3 → T_fp4/T_fp8 = 1.03 (FP4 НЕ throttled).
- ×1.95 TFLOPS per SMSP.
- **BUT**: FP4 f32 acc = 4 regs/MMA vs FP8 f16 acc = 2 regs/MMA → **doubles acc register footprint**. Может сломать 2 blocks/SM при 256-reg cap. Probe reg budget ПЕРЕД любой FP4-rewrite.

### mbarrier на sm_120a
- **Parity convention**: `{0, 0}` для свежего barrier. `SYNCS.PHASECHK` реализует "pass if current_parity != arg". Свежий bar parity=0, первый wait с arg=0.
- `expected_phase = {1,1}` = детерминированный hang на kv_max≥2 (M7 task доказал).
- State в **HW SYNCS unit**, `ld.shared.b64` от bar даёт 0 (нет читаемого backing).
- Для диагностики: atomicAdd-индексируемые лог-записи в pinned-mapped GMEM + SASS-проверка SYNCS.ARRIVE/PHASECHK.
- Квалификаторы `.shared::cta` / `.release.cta` / `.acquire.cta` и `test_wait` vs `try_wait` — **no-op** на ptxas 13.1/sm_120a (SASS-дифф 0 байт). Оставляем для совместимости, не тратим время на PTX-перебор.

### ldmatrix на sm_120a — медленный
- v115 SHFL cooperative transpose показал: даже при Eligible почти same, perf **−28%**. scheduler не может hide SHFL/ldmatrix dependency chain. Используется только в narrow cases (E1 probe C: ldmatrix.x2.trans vs 2×LDS scheduler cost — оба neutral, скорее 2×LDS лучше).

### TMA HW available, но scheduling-shape-rejected
- Probe 2026-06-08: `cp.async.bulk` + `bulk.tensor.2d` compile+run PASS на sm_120.
- v67 TMA conveyor: −10% vs v66 cp.async.cg uniformly. **Single-lane dispatch beat by 128-thread parallel cp.async.** Address-arith path open, но не для drop-in replacement в production kernel'е.

---

## 5. B1 BACKWARD-ФУНДАМЕНТ — ПОЛНАЯ ДЕТАЛИЗАЦИЯ

### 5.1 Математика (Tri Dao Variant 3)

**Forward** сохраняет на Q-row:
- O_i = (1/l_i) Σ_j exp(s_ij - m_i) V_j, где s_ij = (Q_i · K_j) × scale
- L_i = m_i + log(l_i) — log-sum-exp, **ONE float per Q-row**

**Backward** (нужен L для recompute):
```
Precompute (Pass 0 ИЛИ внутри Pass 2 per Q-tile):
  D_i = Σ_d dO_id × O_id

For each (i, j):
  S_ij = (Q_i · K_j) × scale
  P_ij = exp(S_ij - L_i)            ← НЕ хранится, recompute из Q, K, L
  dP_ij = Σ_d dO_id × V_jd
  dS_ij = P_ij × (dP_ij - D_i)
  dV_jd += P_ij × dO_id              ← Pass 2 (по K-tile)
  dK_jd += dS_ij × Q_id × scale      ← Pass 2 (по K-tile)
  dQ_id += dS_ij × K_jd × scale      ← Pass 1 (по Q-tile)
```

### 5.2 Карта транспонирований

В backward нужны **два транспонирования сверх forward**:

| Операция | Что транспонируем | Зачем |
|---|---|---|
| P^T · dO → dV partial | P [Br, Bc] → P^T [Bc, Br] | A-операнд в MMA m16n8k32 |
| dS^T · Q → dK partial | dS [Br, Bc] → dS^T [Bc, Br] | A-операнд в MMA |
| dO · V^T → dP partial | V [Bc, hd] → V^T [hd, Bc] | B-операнд в MMA |
| Q · K^T → S | K [Bc, hd] → K^T [hd, Bc] | как в forward |

`transpose_v` уже в `fa_bwd_common.cuh` (PRMT 4×4 ситуатив). Можно использовать ту же логику для smK_T (производство transpose_K).

Альтернативно: **ldmatrix.trans** доступен, но **медленный** на sm_120a (v115 показал). Если используем — только когда альтернатива хуже.

### 5.3 P recompute, НЕ хранение

Critical decision: P не хранится между forward и backward. Backward сам recomputes:
- `S = Q · K^T × scale` (QK MMA)
- `P = exp(S - L_i)` (с использованием L который пишет L-патч)

**Цена recompute** = 1 QK MMA + 1 softmax (exp). На бумаге: при `kv_iters = sl/Bc = 128`, recompute стоит ~50% от forward compute, но memory bandwidth saved (не нужно хранить P [sl, sl] в HBM).

Это **mandatory tradeoff** для backward на длинных последовательностях (sl=8192 хранение P = 256 MB per (bh, head) — невозможно).

### 5.4 Сага валидации эталона — путь 93.3% → 100%

**ЗАЧЕМ ПРОЧИТАТЬ**: это foundation для **ВСЕЙ корректности** в B2-B5. Если методология валидации шумит — все скоростные оптимизации ниже её noise floor невидимы.

#### Стадия 1: B1.2 initial — 93.3% PASS
Первая версия finite-diff проверялки. Допуск **5% relative**, 20 random позиций на тензор × 3 = 60 checks. 4/60 fail = pass rate **93.3%**.

**Это казалось «почти готово»**, но пользователь правильно отказался идти в B2 на 93.3% — backward не должен валидироваться на эталоне, который сам сходится только на 93%.

#### Стадия 2: B1-FIX strict — открытие трещины
Усилили проверялку (rel_tol → 1e-3, all 4096 positions). Результат: **31.7% agreement at <0.1%**. 

Top discrepancies показали:
- dQ (0,22,56): ana=−1.17e-4, num=−2.51e-3 (rel_err 20×)
- dV (0,3,49): ana=+3.16e-3, num=−3.4e-5 (sign flip!)
- max rel_err: dQ 20.43, dK 38.60, dV 2.59

**Cluster scattered** — но magnitudes слишком большие чтобы быть просто FP32 noise.

#### Стадия 3: Корневой диагноз
**Трещина была в проверялке, не в backward.** Четыре слоя проблемы:

1. **FP32 probe-forward** даёт шум ~1e-7 relative в O. При finite-diff на малых grads (|grad|<1e-3) шум амплифицируется в `loss_plus − loss_minus` → rel_err 100%+. **FP64 loss-accumulator НЕ ЛЕЧИТ** — шум живёт в самих O[i] из FP32 forward.
2. **Фиксированный eps=5e-3** — не адаптирован под локальные |x|. Мелкие x теряют сигнал в truncation.
3. **FP32 loss-аккумулятор** — catastrophic cancellation между двумя близкими большими суммами по 4096 элементов.
4. **20 random positions** — sampling bias на 4096-элементном тензоре.

#### Стадия 4: B1-FIX v2 — FP64 probe + hybrid tol
- **`fa_fwd_fp64`** — отдельный double-precision forward только для probes. FP32 backward остался тем что проверяется.
- **Hybrid pass criterion**: `|num − ana| < abs_tol + rel_tol × |ana|`, где `abs_tol = 1e-4`, `rel_tol = 1e-3`. abs-член (1e-4) поглощает FP32-backward intrinsic noise floor. rel-член (1e-3) держит 0.1% rel на значимых grads.
- **Adaptive h** = `1e-3 × max(|x|, 1e-2)` per position.
- **Все 4096 элементов dQ/dK/dV** проверяются (не 20). top-20 по abs_err с индексами (b,n,d) + кластер-анализ.

**Результат**: **100.0000% above-floor agreement** (10371/10371 позиций at seed=42). max abs_err dQ 5.39e-6, dK 7.10e-6, dV 1.81e-5. max rel_err на |ana|>1e-2: всё 0.0001 (0.01%).

#### Стадия 5: Sanity-довески (4 проверки)

**Multi-seed** {42, 7, 123, 2024} — все **100.0000%** above-floor. Кластер d=36 у dV @ seed=42 переезжает на другой dim/тензор при других seeds → **артефакт sampling-сортировки top-by-abs_err**, не bug.

**Mutation tests** — sensitivity boundary:
| Мутация | rel_err | FP32 verdict |
|---|---:|---|
| dQ ×1.001 (0.1%) | 1.00e-3 | **undetected** (honest boundary, на noise-floor) |
| **dQ ×1.002 (0.2%)** | 2.00e-3 | **just DETECTED** (99.43%) ← sensitivity edge |
| dQ ×1.005 (0.5%) | 4.98e-3 | DETECTED (79.45%) |
| dV ×1.01 (1%) | 9.90e-3 | DETECTED (63.33%) |
| dQ sign-flip | 2.00 | DETECTED (68.28%) |
| dK без ×scale (×√128) | 9.12e-1 | DETECTED (63.88%) |

Граница 0.2% ровно матчит наш rel_tol=1e-3. Меньшие мутации индистинктибельны от FP32-шума **по дизайну**.

**PyTorch FP32 cross-check** — 100.0000% above-floor agreement, max abs_err 1.2e-7. Three implementations (CPU ref + FP64 finite-diff + torch.autograd FP32) converged.

#### Стадия 6: B1-FIX-EXTRA — FP64 GOLDEN

Чтобы ИСКЛЮЧИТЬ слепоту проверялки на FP32-floor, написали **FP64 backward** как полностью независимую реализацию. Mechanical port FP32 → double, математика identical.

Tolerance: `abs_tol=1e-10`, `rel_tol=1e-6`. Finite-diff h = `1e-5 × max(|x|,1)` — оптимум central diff для double (`h ≈ ε^(1/3)`).

**Результаты** (seed=42):

| Метрика | dQ | dK | dV |
|---|---:|---:|---:|
| max abs_err | 3.54e-10 | 1.99e-9 | 3.07e-10 |
| max rel_err above-floor | 2.94e-8 | 1.24e-7 | 2.28e-8 |
| above-floor pass | 100% | 100% | 100% |

**Mutation boundary провалилась с 0.2% (FP32) до 0.0001% (FP64)** — ровно 3 порядка, точно как ожидалось (FP64 ULP / FP32 ULP ≈ 1e9, ε^(1/3) ratio ≈ 1e-3).

**PyTorch float64 cross-check** — max abs_err **2.78e-16 = 1 FP64 ULP**, max rel above-floor 1.09e-14. Это **bit-identical с точностью до FMA reordering**. Three FP64 implementations сошлись на fundamental FP64 floor.

**Verdict**: формулы математически верны **не "в пределах FP32-шума", а на FP64 ULP уровне**. Backward correctness ironclad.

### 5.5 Двухуровневая валидация для B2/B4

Это **обязательный паттерн** для любого GPU backward kernel'я.

| Уровень | Тест | Tolerance |
|---|---|---|
| **CI быстрый** | GPU FP32 output ↔ FP32 reference | hybrid `abs 1e-4 + rel 1e-3` |
| **Debug точный** | GPU output upcast → FP64 ↔ FP64 golden | rel `1e-5..1e-6` (под FP16/FP8 quantum, но выше FP64 floor) |

Для B2 (GPU dV/dK в FP32 output):
- CI: max abs ~1e-3 ожидаемо (FP16 PV MMA quantum), max rel ~1e-3 above-floor
- Debug: upcast GPU FP32 → double, compare to FP64 golden. Ожидаем rel ~1e-5 above-floor (FP8 input quantum + FP16 intermediate)

Если GPU sustained показывает на Debug >1e-4 rel above-floor — **значит реальный bug в kernel'е**, не numerical noise. Это **detection capability** двухуровневой схемы.

### 5.6 L-патч на forward (необходимое условие backward)

#### Зачем
Tri Dao Variant 3 backward требует `L_i = m_i + log(l_i)` для recompute `P_ij = exp(S_ij - L_i)`. Forward v121r не пишет L (нет потребителя). L-патч добавляет L-write в эпилог.

#### Реализация
- **Отдельный файл**: `_v121r_train_kernel.cu` (`fa96b_train_kernel` symbol). Production `_v121r_kernel.cu` (под тегом v0.1.0) **не тронут**.
- **Signature**: `+ float* __restrict__ L_out`. `nullptr` → дort production-compatible call (но эта ветка не используется в train mode — train kernel всегда с L).
- **Эпилог**: L-write **ПЕРЕД** O-write loop'ом. Это критично для performance — log2f issue'ятся в shadow O-store LSU latency.
- **Гейт**: `tid == 0` (только 1 thread из 4-col-группы пишет; rmax/rsexp одинаковы у всех 4 col-threads после shfl-reduce в softmax-loop).

#### Math (log2-space → natural-base conversion)

Forward kernel умножает scores на `log2(e) = 1.4426` (line ~422: `fs = scale × qk_descale × 1.4426`) чтобы использовать `exp2.approx.f16x2` вместо `expf`. Поэтому в эпилоге:
- `rmax[mi][r]` = m_i_natural × log2(e) (log2-space)
- `rsexp[mi][r]` = Σ exp(s_natural − m_natural) = l_i_natural (base-инвариант, оба numerator и denominator под общим основанием)
- `L_natural = m_natural + log(l_natural) = (rmax_log2 + log2(rsexp)) × loge(2)`

Cost per Q-row: 1× `lg2.approx.f32` + 1× FFMA + 1× ST.E.32. ~12 SASS инструкций epilogue, 1× tid==0 гейт.

#### Результаты
- **ptxas**: 255 registers (Δ=0 vs production), 0 spill, 0 stack, 1 distinct barrier ID. Budget +2-3 не использован вообще — L-вычисление уместилось в существующие live registers.
- **Correctness** (bh=4 sl=256 hd=128, seed=42):
  - O train vs CPU ref: max abs **4.66e-3**, max rel 1.27e-2 (FP16 quantum)
  - L train vs CPU ref: max abs **5.46e-4** (на L_ref=7.67 → rel ~7e-5 = 0.007%)
  - L finite: 1024/1024
- **Long-form correctness** (bh=8 sl=8192 hd=128, **65536 L-позиций**):
  - max abs **6.66e-4** (idx=28612, ref=10.49, gpu=10.50)
  - max rel 6.3e-5
  - 0/65536 positions exceed 1e-3 — long sequences не накапливают ошибку
- **30-run bench** (bh=128 sl=8192 hd=128 wnd=0, extended warmup 40 alternating, then Round 1 + Round 2 reverse):
  - Round 2 truest steady-state: prod **520.58 T**, train **520.64 T** → train **+0.06 T БЫСТРЕЕ** within σ
  - Aggregated: delta −1.27 T = **−0.244%**, |t|=1.78, σ pooled 0.71T
  - **VERDICT: PASS** — cost 0.244% under 0.5% budget

#### Production не тронут, train локален
- `_v121r_kernel.cu` в `/root/repos/fa-blackwell-fp8/src/` под тегом v0.1.0 **не тронут**.
- `_v121r_train_kernel.cu` живёт в `/data/lib/podman-data/projects/goml/libs/fa_sm120/src/` локально.
- **В репо НЕ коммитим** до готовности всего backward (B2-B5). Тогда выйдет одним релизом v0.2.0 со всей backward-инфраструктурой.

---

## 6. B1.4 — ГЕОМЕТРИЯ Pass 2 (КРИТИЧНО ДЛЯ B2)

### Главный результат: атомики НЕ нужны

**Block-residency proven**:
- Один CUDA block обрабатывает **одну пару (bh, K-tile)**.
- Block владеет Bc=64 K-rows × hd=128 elements = 8192 элементов dK[K_tile, hd] и dV[K_tile, hd].
- Block iterates over **ВСЕ Q-tiles** в seq_len (sl/Br = 128 для sl=8192 Br=64).
- dK/dV accumulators живут в **регистрах block'а** через ВСЕ Q-tiles.
- Single block writes its dK/dV slice to global ONCE в эпилоге.

**Никаких атомиков** не требуется — каждый K-row dV/dK уникально owned ONE block'ом.

Симметрично для Pass 1 (dQ): block владеет Q-tile, итерирует по K-tiles.

### Регистровый риск (B1.4 estimate)

**Монолит** (dK+dV в одном kernel'е):
- dV_acc[Bc=64, hd=128] FP32 = 32 KB → too big for registers целиком
- Per-warp: 16 K-rows × 128 hd / 32 threads = 64 floats/thread = 64 regs **just for dV_acc**
- + dK_acc per-thread = 64 regs
- + S_partial / P_recompute / dS / dP / D[Br] / L[Br] live state
- **Estimated total: 270-280 regs** при потолке 255 → **spill вероятен**

**Fallback split** (dV-only + dK-only отдельные kernels):
- Each ~200 regs, 0 spill
- Double-read Q/dO/V/K из L2 (97% hit) → cost minimal
- **B1.4 верд** ict: split не поражение, **возможно оптимум**

### SMEM budget
- smQ (Br × hd FP8) = 64 × 128 = 8 KB
- smK (Bc × hd FP8) = 64 × 128 = 8 KB (fixed для всего kernel'я)
- smV (Bc × hd FP8) = 8 KB (для dP MMA)
- smdO (Br × hd FP16) = 16 KB (FP16 — это input от autograd)
- smP (Br × Bc FP8) = 4 KB (квантованный P для dV-MMA A-operand)
- smdS (Br × Bc FP8) = 4 KB (квантованный dS для dK-MMA A-operand)
- smL (Br × FP32) = 256 B
- **Total** ≈ 48 KB → 2 blocks/SM viable (как в forward).

### M_TILES≥2 обязательно
v122 калибровка показала: M_TILES=2 → 1 = util × 0.625 = **−37%**. Не уменьшать.

Для backward Br/Bc = 64 даёт M_TILES = Br/16 = 4 (для QK MMA) и M_TILES_DV = Bc/16 = 4 (для dV-MMA). Это **выше** forward'а M_TILES=2. Так что util должен быть **лучше** или сопоставим.

### Цель TFLOPS
- Forward v121r peak 652T
- Backward target **320-380T** = 0.5-0.6× forward
- Это норма для FA backward — больше compute (5 MMAs per tile: QK, PV, dPV, dKQ, ну и квантизации), но dispatch density лучше из-за M_TILES=4 vs forward 2

**<250T → разбор NCu прежде чем оптимизировать**. Это диагностика регрессии, не выходная оптимизация.

---

## 7. ПЛАН B2-B5

### B2: Pass 2 (dK + dV), первое backward-ядро

**Стратегия**:
1. **Монолит сначала**: попытка одним kernel'ём вычислить и dK, и dV. ptxas check.
2. **Если spill=0 при ≤255 regs → победа**. Используем монолит.
3. **Если spill>0 → fallback split**: отдельные dV-only и dK-only kernel'ы (~200 regs each). Это **возможно оптимум**, не failure.
4. **Если совсем плохо**: каскад — сначала dV-only kernel валидировать, потом отдельно dK-only. Поэтапная валидация.

**Implementation principles**:
1. **Корректность ПЕРВАЯ, скорость ВТОРАЯ**. Не оптимизировать на этом этапе.
2. Можно начать с **slow baseline** (плотные FP32 циклы без MMA) — даст 10-50T, но 100% correct → база для перехода на MMA.
3. Использовать кирпичи из `fa_bwd_common.cuh` (parity {0,0}, swizzle hoisting, half2 patterns, mma_fp8_f16, ex2_approx_f16x2).
4. P recompute из Q, K, L (не хранится).
5. dO precision — fp16 input от autograd. Варианты:
   - Quantize dO → e4m3 per Q-tile (max abs scaling, FlashAttention-3 style) → use FP8 MMA
   - Use FP16 m16n8k16 MMA напрямую (половинный throughput но без quantize loss)
   - MVP: quantize dO в FP8, потом если нужна точность — switch на FP16 MMA
6. dK/dV accumulators **FP32 в регистрах** блока.
7. **Quantize precision**: при quantize FP16 dO → FP8 e4m3, сохраняй per-Q-tile `dO_scale = max(|dO|) / 448` чтобы восстановить magnitude в accumulate stage.

**Correctness validation**:
1. CI: vs FP32 CPU reference, 8 standard forms (bh ∈ {1,4,8,16,64,128}, sl ∈ {256,1024,4096,8192}, hd=128), hybrid tol abs 1e-4 + rel 1e-3
2. Debug: upcast GPU FP32 → double, vs FP64 golden, rel 1e-5..1e-6
3. **Canary**: sl=300 wnd=96 (нестандартная форма с window) — должна тоже работать
4. **Causal**: бенчи + correctness с causal=1
5. **Long-form**: bh=8 sl=8192 — minimum (как L-патч валидировали)

**Bench**:
1. После корректности — 30-run same-thermal bench на peak форме (bh=128 sl=8192 hd=128 wnd=0)
2. Target 320-380T total (dK + dV).
3. <250T → NCu разбор. Какие stalls? Eligible? Регистры?

### B3: Pass 1 (dQ)
Симметрично Pass 2 но dispatch по Q-tile:
- Один block обрабатывает (bh, Q-tile) пару
- Iterate over all K-tiles
- dQ_acc[Br, hd] в регистрах
- Аккумулятор block-resident, **атомики не нужны**
- Same FP8 MMA infrastructure, just transposed

### B4: end-to-end correctness Pass1+Pass2
- Запустить полный backward: dQ from Pass 1, dK+dV from Pass 2
- Сравнить с FP64 golden на 8 формах + canary sl=300 wnd=96
- Если ВСЁ зелёное (rel < 1e-4 above-floor) → backward correctness CLOSED

### B5: 30-run same-thermal bench backward
- bh=128 sl=8192 hd=128 wnd=0 (как forward champion config)
- Target ~300T sustained (forward × 0.5)
- Если ≥320T → отлично
- Bench log для v0.2.0 release notes

### После B5: v0.2.0 release
- C-ABI расширение: `fa_backward(ctx, ...)`
- Go cgo extension
- Python ctypes extension
- Train kernel + backward kernels commit'ятся в репо
- 30-run backward champion log как Release asset

---

## 8. АРТЕФАКТЫ — ВСЕ ПУТИ

### B1 эталоны (постоянные)
| Файл | Путь | Что |
|---|---|---|
| `fa_bwd_cpu_reference.cu` | `/data/lib/podman-data/projects/goml/libs/` | Original B1.2 (с старым 93.3% check). Не удалять — историческая запись. |
| `fa_bwd_cpu_reference_b1fix.cu` | там же | Strict FP32 checker (B1-FIX) — current main reference |
| `fa_bwd_cpu_reference_fp64_golden.cu` | там же | **GOLDEN FP64 backward**, использовать для двухуровневой валидации |
| `fa_bwd_common.cuh` | там же | 10 кирпичей: cp.async обёртки, swz_byte/_bc/_smvt, load_tile_fp8, transpose_v, mma_fp8_f16, fp16x2_to_e4m3x2, ex2_approx_f16x2, mbarrier helpers (parity {0,0}), e4m3 host roundtrip, RangeIter |

### Forward production (под тегом v0.1.0, НЕ ТРОГАТЬ)
| Файл | Путь | Что |
|---|---|---|
| `_v121r_kernel.cu` | `/data/lib/podman-data/projects/goml/libs/fa_sm120/src/` | Production champion kernel |
| `_v121r_launcher.cuh` | там же | Launcher (concatenated by Makefile) |
| `fa_dispatch.cpp` | там же | Pure dispatcher function (8 ниш) |
| `fa_ctx.cu` | там же | Context lifecycle, arch probe, fa_forward entry |
| `include/fa_sm120.h` | `/data/lib/podman-data/projects/goml/libs/fa_sm120/include/` | C-ABI header |
| `Makefile` | `/data/lib/podman-data/projects/goml/libs/fa_sm120/` | `make lib`, `make test`, `make test_readme` |

То же зеркало в `/root/repos/fa-blackwell-fp8/` — это **публичный репо**, v0.1.0 released.

### L-патч (локально, НЕ закоммичено)
| Файл | Путь | Что |
|---|---|---|
| `_v121r_train_kernel.cu` | `/data/lib/podman-data/projects/goml/libs/fa_sm120/src/` | Train variant с LSE output |
| `_v121r_train_launcher.cuh` | там же | Train launcher (signature `+ float* L_out`) |

### Sandbox workspace (B1-FIX validation infrastructure)
Все в `/data/lib/podman-data/projects/claude-dashboard/workspace/8ffe507afe53/`:
| Файл | Что |
|---|---|
| `b1fix_check.cpp` | Strict FP32 checker (multi-seed, mutation, dump modes) |
| `b1fix_fp64_check.cpp` | FP64 strict checker (mutation boundary 1e-6) |
| `torch_cross_check.py` | PyTorch FP32 cross-check |
| `torch_cross_check_fp64.py` | PyTorch FP64 cross-check |
| `test_lpatch.cu` | L-patch correctness (bh=4 sl=256) |
| `test_lpatch_long.cu` | L-patch long-form (bh=8 sl=8192) |
| `bench_lpatch.cu` | L-patch 30-run cost bench (alternating warmup) |
| `Makefile.b1fix` | Build/run targets for B1-FIX checks |
| `Makefile.lpatch` | Build/run targets for L-patch tests |

### Champion log (forward) и NCu reports
- **Champion log**: `/data/lib/podman-data/projects/goml/champion_30x_20260611_134533.log` (gpuserver-tagged original) + sanitized в workspace (для GitHub release)
- **NCu reports** (binary `.ncu-rep`):
  - `/data/lib/podman-data/projects/goml/runs/ncu_v96b_baseline.ncu-rep`
  - `/data/lib/podman-data/projects/goml/runs/ncu_v78_full.ncu-rep`
  - + множество других (~30 файлов)

### Repository
- **Published**: https://github.com/djeday123/fa-blackwell-fp8
- **Tag**: v0.1.0
- **Assets**: champion_30x_v121r_bh64_bh128_sl8192.log
- **Owner**: djeday123 (личный GitHub аккаунт)
- **Author identity (git commits)**: Vugar Bakhshaliyev <39285558+djeday123@users.noreply.github.com>
- **License**: Apache-2.0 + NOTICE
- **Topics set**: flash-attention fp8 cuda blackwell sm120 nvidia attention transformer gpu inference fp8-e4m3 cpp go python ml-systems gpu-kernels

---

## 9. ПРОЦЕСС И ПРАВИЛА (МЕТОДОЛОГИЯ)

### Трёхзвенная модель
1. **Vugar решает** — задачи, направления, что не делать
2. **Агент исполняет** — реализация, измерения, ptxas/NCu
3. **Скептик-ревью верифицирует** — независимая проверка результатов, contrarian analysis

Каждое утверждение пройти через все три звена.

### Одно изменение — одно измерение
- НЕ bundling нескольких изменений в один бенч
- Каждое source-level изменение → ptxas check → bench (30-run если production) → NCu (если значимая дельта)
- v98 lesson: 3-run "+0.16%" был outlier artifact, 30-run показал −0.68% (t=−8.87)

### 30-run same-thermal для production-кернелов
- 10 warmup launches (clock stabilization)
- 30 main runs, каждый = median-of-5 launches
- Same thermal block (нет cooldown между runs)
- nvidia-smi sampling sm_clock + temp per run для thermal diagnostic
- **For comparisons**: extended warmup 40+ alternating launches между двумя kernel'ями (L-patch bench technique), потом round 1 + round 2 reverse для thermal asymmetry detection

### ptxas-контроль после каждого изменения
Ожидаемые поля:
- `Used X registers, used Y barriers`
- `Z bytes stack frame, A bytes spill stores, B bytes spill loads`
- Δregs > +3 → НЕ принимать без обоснования
- Spill > 0 → принципиально пересмотреть

### NCu = судья для любых +X% claims
- NCu Estimated Speedup = **локально-вакуумная эвристика**, обычно overestimates ~3× когда kernel scheduler-bound
- v77 lesson: "+23% NCu estimate" → wall-clock **−17%**
- Всегда: проверять GLOBAL bottleneck (wait? math_pipe? mio?) ПЕРЕД верить per-issue speedup

### Корректность 8/8 перед скоростью
- 8 стандартных форм: bh ∈ {1,4,8,16,64,128} × sl ∈ {256,1024,4096,8192}, hd=128
- + canary sl=300 wnd=96 (нестандартная sliding window, causal=1)
- + causal=1 на всех 8
- Всё PASS перед началом optimization сессии

### Two classes of source-level optimization (КРИТИЧЕСКИЙ УРОК)
На fixed-point kernel'е (v96b/v121r):
- **REDISTRIBUTION** (v98 K-pre, v113/v115/v116/v117) — redistribute stalls at SAME work. Wall-clock NEUTRAL, никогда не конвертируется.
- **WORK-REDUCTION** (v96b/v118 localfix) — remove instructions from critical path. **REAL wall-clock gain**.

**Селекционный критерий для любых будущих оптимизаций**: только WORK-REDUCTION конвертируется at fixed point.

### FP32 floor реален
- Tolerance ≥0.2% honest (FP32 backward intrinsic noise floor)
- Меньшие мутации indistinguishable from noise **by design**
- Pure relative tol = ложные fails на малых grads. **Hybrid abs+rel** обязательно.

### Probe должен быть на ступень точнее проверяемой штуки
- Проверяем FP32 backward → probe в FP64
- Проверяем GPU FP16 output → probe в FP32 reference
- Проверяем GPU bwd → probe в FP64 golden

### Денoминатор для %-of-peak claims
- ВСЕГДА measured peak для этой карты
- Лжавал когда выдумывал 640T (vs measured 1099T peak) → 36% превратилось в 62% — completely changed framing

---

## 10. ОТКРЫТЫЕ ВОПРОСЫ И РИСКИ ДЛЯ B2

### Risk 1: Spill при монолите
**Высокая вероятность**. B1.4 estimate 270-280 regs vs 255 budget.
- **Mitigation**: split fallback (dV-only + dK-only) готова.
- **Detection**: ptxas check **в first compile**. Если spill > 0 → не пытаться "compress" монолит, сразу split.
- **Принцип**: split не поражение, возможно оптимум. Не bullshit себя оптимизацией монолита.

### Risk 2: Transpose-налог
В backward нужны **два транспонирования сверх forward** (P^T и dS^T для dK-MMA), плюс смещенные access patterns для V^T в dPV.
- **v114 lesson**: посчитать SMEM reads **до написания кода**. Если transpose vызывает 2× LDS volume vs FP8 quantum savings — overhead.
- **Estimate**: P [Br=64, Bc=64] FP8 = 4 KB read 2× (раз для dV-MMA P^T, ещё раз потенциально для dKK если кешируем) = 8 KB LDS additional.
- transpose_v кирпич в common.cuh готов (PRMT 4×4), можно reuse для smK_T (но trade-off: store-conflict 32-way на smK_T если stride=64, нужен SMV_T_STRIDE=68 padding — уже в common.cuh).

### Risk 3: LDS / MMA balance backward'а
- Forward имеет ~2 GEMMs per kv-iter (QK + PV)
- **Backward Pass 2 имеет ~3 GEMMs** per Q-iter (QK + dV-partial + dK-partial)
- **Backward Pass 1 имеет ~3 GEMMs** per K-iter (QK + PV-recompute для D? + dQ-partial)
- Util может быть **ВЫШЕ** forward, потому что плотнее MMA-per-LDS
- **Calibration**: NCu wait stall first, если <30% → util-friendly architecture для backward

### Risk 4: dO precision (FP16 input)
- dO приходит из autograd как FP16 (PyTorch default)
- Quantize dO → FP8 e4m3 каждую Q-iter? Или FP16 MMA?
- **FP8 quantize approach (FA-3 style)**:
  - per-Q-tile compute `dO_scale = max(|dO|) / 448`
  - quantize `dO_e4m3 = round((dO / dO_scale) * 1)` → `cvt.rn.satfinite.e4m3x2.f16x2`
  - MMA outputs scaled, apply `dO_scale` in FP32 accumulate
- **FP16 MMA approach**: `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` — exists на sm_120a. Half throughput vs FP8 m16n8k32 но без quantize loss.
- **MVP recommendation**: FP8 quantize. Если correctness нет → switch на FP16 MMA как diagnostic.

### Risk 5: D_i precompute vs on-the-fly
- D_i = Σ_d dO_id × O_id — per Q-row, 1 float
- Option A: **Precompute Pass 0** — отдельный быстрый kernel, читает O+dO, пишет D[bh, sl]. +1 GB read+write на bh=128 sl=8192.
- Option B: **On-the-fly per Q-tile** — kernel читает O_i вместе с dO_i, вычисляет D_i на месте.
- **Recommendation**: B (on-the-fly), avoids extra global memory pass. SMEM bandwidth дешевле HBM.

### Risk 6: FP4 reserve и FP6 storage (НЕ ДЛЯ B2)
- FP4 m16n8k64 e2m1 verified working (×1.95 TFLOPS)
- **НО**: f32 acc = 4 regs/MMA vs FP8 f16 acc = 2 regs/MMA → doubles acc footprint
- May break 2 blocks/SM at 256-reg cap
- **Вторая очередь**, не сейчас. Сначала FP8 baseline.

### Risk 7: Block-residency может потребовать grid-stride
- При bh × n_k_tiles превышает SM count много раз → wave-tail. На bh=128 sl=8192 Bc=64 → 128 × 128 = 16384 blocks, при 188 SMs × 2 LB = 376 active = **44 wave**. Wave-tail не критичен для backward (не точечная оптимизация).
- При маленьких bh = большие partial waves. Решение: либо partial wave OK, либо persistent kernel (v107 forward = neutral, для backward возможно другая story).

---

## 11. СТАТЬЯ — ПАРАЛЛЕЛЬНЫЙ ТРЕК (НЕ БЛОКЕР B2)

### Хабр-лонгрид (готов или near-ready)
- Тон: production-grade, технический, без bs
- Аудитория: русскоязычные GPU-разработчики, ML-engineer'ы
- Содержание: путь forward (с числами), что закрыто и почему, mecha calibration, дispatcher rationale
- Author: Vugar Bakhshaliyev

### LinkedIn / Medium — английский «путь»
- Story 93.3% → 100% (наука валидации)
- Громкий тон, аудитория международная (ML researchers, GPU enthusiasts)
- Hook: «How a 93.3% validation looked like success — and why I refused to ship the kernel on it»

### Параллельность
- **B2 имплементация НЕ блокируется статьёй**. Статья пишется одновременно, не последовательно.
- README в репо уже содержит `Technical write-up: coming soon` — обновим коммитом в день публикации статьи.

---

## 12. ШПАРГАЛКА ДЛЯ НОВОЙ СЕССИИ — ПЕРВЫЕ ШАГИ B2

### Шаг 0: Поднять контекст
- Прочитать этот документ (B2_HANDOFF.md) целиком
- Прочитать `fa_bwd_common.cuh` (10 кирпичей, что доступно)
- Прочитать `_v121r_train_kernel.cu` (как L пишется, для понимания backward input layout)
- Прочитать `fa_bwd_cpu_reference_fp64_golden.cu` (FP64 backward math, validation ground truth)

### Шаг 1: Дизайн (бумага → файл)
- Block geometry: (bh × n_kt) blocks, 128 threads, 4 warps
- Each warp owns Bc/4 = 16 K-rows
- Br=64, Bc=64, hd=128, M_TILES_QK=4, M_TILES_DV=4
- SMEM ~48 KB → 2 blocks/SM
- dV_acc[N_TILES=16][per_thread_share] FP32 в регистрах
- dK_acc[N_TILES=16][per_thread_share] FP32 в регистрах
- (если split → один из двух per kernel)

### Шаг 2: MVP — Correct, Slow
- Можно начать с **FP32 CUDA-core baseline** (плотные циклы, без MMA)
- 10-30T speed, но 100% correctness
- Validate vs FP32 reference на маленькой форме (bh=4 sl=256)
- Validate vs FP64 golden (upcast GPU FP32 → double)

### Шаг 3: FP8 MMA enable
- Заменить циклы на m16n8k32 e4m3 → f16 acc
- QK MMA + softmax recompute + dV MMA + dK MMA
- Re-validate correctness (FP16 quantum tolerance loosens 1e-4 → 1e-3)
- ptxas check (regs, spill)

### Шаг 4: Hot-loop optimizations (только если MVP корректен)
- Address hoisting (v121 stage1 паттерн)
- ks-batched MMA reorder (v87 паттерн)
- f16x2 softmax conveyor (v121r паттерн)
- transpose_v reuse для smK_T
- **Не оптимизировать без NCu measurement** (NCu Estimated Speedup mirage!)

### Шаг 5: Correctness 8 forms + canary
- Запустить на 8 стандартных формах + sl=300 wnd=96
- vs FP64 golden, two-level validation
- Если ВСЁ PASS → bench

### Шаг 6: 30-run same-thermal bench
- bh=128 sl=8192 hd=128 wnd=0
- Target 320-380T (dK+dV)
- <250T → NCu разбор

### Шаг 7: Запись в memory + handoff к B3
- Сохранить B2 outcome (TFLOPS, ptxas, spill, ключевые decisions)
- B3 = Pass 1 dQ, симметрично

---

## 13. КРАТКИЙ КОНСПЕКТ КЛЮЧЕВЫХ ЦИФР

| Метрика | Значение | Источник |
|---|---:|---|
| Forward production peak | **652.40 ± 0.87 TFLOPS** | 30-run champion log bh=128 sl=8192 |
| Forward steady mean | 522 T (post-warmup) | L-patch bench Round 2 |
| sm_120a card FP8 peak | ~960 TFLOPS theoretical | QMMA T=16.8 cycles × 4 SMSPs × 188 SMs |
| v121r registers | 255, 0 spill, 0 stack | ptxas |
| v121r SMEM | 48.5 KB | макс через cudaFuncAttribute |
| v121r barrier IDs | 1 | ptxas |
| FP32 floor | ~0.1% rel error | Mutation boundary |
| FP64 floor | ~0.0001% rel error | Mutation boundary FP64 |
| L-patch cost | 0.244% (within σ) | 30-run bench |
| L-patch reg growth | **+0 (Δ=0)** | ptxas |
| Backward target | 320-380 TFLOPS | B1.4 estimate, 0.5-0.6× forward |
| Backward correctness | hybrid abs 1e-4 + rel 1e-3 | B1-FIX standard |
| FP64 golden tol | abs 1e-10 + rel 1e-6 | B1-FIX-EXTRA |
| Card SM count | 188 | hardware |
| Card mem | 96 GB HBM | hardware |
| sm_clock under load | 2617-2677 MHz | bench |
| Card temp under load | 30-46°C | bench |
| Card power under load | ~250 W | bench |
| HBM throughput at peak | 2.88% | NCu DRAM Throughput |
| L2 hit rate loads | 97.57% | NCu |
| Local memory all metrics | 0 | NCu 18-channel verified |

---

## 14. AHA-МОМЕНТЫ И ТРАВМЫ ПРОЕКТА (КОРОТКО)

- **f16 accumulator работает в QMMA где docs показывают только f32** (sm_89 old path AND sm_120a kind::f8f6f4). Probe both paths — есть undocumented optimization.
- **NCu Estimated Speedup — local vacuum** (overestimates ~3× когда kernel scheduler-bound, может быть **negative** если fix adds инструкции). v77 lesson: +23% estimate → -17% wall.
- **Sub-1% comparisons требуют 20+ sequential runs** at same thermal state. 3-run sampling unreliable. v98 lesson.
- **Tensor util mirage**: 45% не означает «есть 55% headroom». 55% — это softmax/transpose/barriers phase structure, не MMA chain depth.
- **«100% correctness» бывает разной**. 100% above-floor != 100% everywhere — small grads intrinsically noisy в FP32. Honest framing критично.
- **mbarrier на sm_120a parity {0,0}** — все остальные конвенции (включая Hopper FlashInfer pattern {1,1}) дают детерминированный hang.
- **ldmatrix медленный** — даже когда Eligible same, perf падает на ~28%. Используется только когда альтернативы хуже.
- **Любая «random» арифметика в hot-loop = регрессия**. Compiler уже на оптимуме liveness span and packing — ручные правки force-pattern уменьшают scheduling flexibility.
- **В backward атомики не нужны** — block-residency per K-tile (Pass 2) или Q-tile (Pass 1). Доказано B1.4.
- **L = m + log(l) пишется одной log2f + одной FFMA** в эпилоге. Δregs=0, cost 0.244% wall-clock. log2-space scores (×1.4426) делают L_natural простой conversion.

---

## 15. ПОСЛЕДНЯЯ СТРОКА

**Следующий шаг: B2, Pass 2 (dK+dV), старт с корректности.**
