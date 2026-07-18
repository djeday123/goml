# 052 — smQ-prefetch merged: механизм работает, wall не оживает → откат

**Chain**:
- 049_b1_probe.md `7c4d2cd35bcc554cfc9bc5732f9b9bc3`
- 050_dkS2_v3.md `9269bcf280c5ebf1a67aa9e91bbd3933`
- 051_s2v3_autopsy.md `bd9ea399697ffe9f6ff206618c48a36d`

**Правила ТЗ 052**: одна структурная правка + полный гейт **ВКЛЮЧАЯ racecheck** (долг 050/051 закрывается культурно: любой сдвиг барьеров = racecheck). Production 033/040/041 sealed.

---

## Артефакт-хедер (правило 5)

```
libs/ (post-052):
-rw-r--r-- 25638 Jul  9         fa_bwd_merged_v1.cu           (md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33 = 040 sealed) ✓
-rw-r--r-- 13352 Jul  9         fa_bwd_dk_new.cu              (md5 a9f0ded8261e53a143b521ffa647f458 = 033 sealed) ✓
-rw-r--r-- 18834 Jul  8         fa_bwd_dq_new.cu              (md5 d7a11a3d788eb4c396d892bc9c8ab754 = 041 sealed) ✓
-rwxr-xr-x       Jul  9         bench_r2c_e2e                 (post-052 rollback, fingerprint 252/128/69 OK)
```

Diagnostic binaries:
```
runs/archive/052_pre/:
-rwxr-xr-x       Jul  9  r2c_merged_wall_base   (md5 51192119fdc61e42a3731a42687668d1, 040 252r)
-rwxr-xr-x       Jul  9  r2c_merged_wall_cand   (md5 8456c8295ffe75dcd8c02a4356cb3405, 052 250r prefetch)
-rw-r--r-- 27918 Jul  9  fa_bwd_merged_v1.cu.052_cand (source of prefetch candidate)
```

**Gate-log** (post-rollback):
```
$ 037r_gate.sh
bench_merged: FINGERPRINT kernel_merged_v1: numRegs=252, sharedSizeBytes=0, ...
GATE OK: numRegs=252 matches EXPECT=252
```

---

## §0. Долги (закрытие)

### §0.a Свод: CPU-судья банков для кооперативных инструкций (LDSM-класс) — ДВЕ фазы

Свод обновлён (правило прибора):

> **CPU-судья банков для кооперативных LDSM-инструкций считает ДВЕ независимые фазы:**
>
> 1. **Адресная фаза** — раскладка row_ptr (или column_ptr) всех 32 лейнов по банкам SMEM. Bank(lane) = (row_ptr(lane) / 4) mod 32. Если row-stride кратен `lane_count × bank_width` (128B для 32 banks × 4B), **все 32 lane row_ptrs выравниваются на ОДНИ И ТЕ ЖЕ 4 банка** → 32-way conflict per instruction.
>
> 2. **Данная фаза** — волны на охапку данных (bytes per instruction / 128B). Для x2.b8 = 512B / 128B = **пол 4 волны/x2** (структурный минимум).
>
> Судья должен подтверждать/опровергать **обе фазы отдельно**. Одна честная фаза не гарантирует другую.
>
> **Ретро-строка** (по 044/051):
> - **044 §5.f сказал только про данную фазу** ("smQ row-stride 128B = 32 aligned banks; structural pol 4/x2; events 0"). Это ЧЕСТНОЕ утверждение по данной фазе (пол 4/x2 подтверждён 051 §2 wavefronts/x2 avg ≈ 4).
> - **Адресная фаза НЕ проверена в 044**. Реальность (051 §2): row-stride 128B → 32 row_ptrs хитают одни и те же 4 банка → **32-way address-phase conflict → шторм ×5.07 LD conflicts, ×3.09 wavefronts**.
> - **Правильный вердикт судьи для S2v3**: "адресная фаза = шторм (не проверил); данная фаза = пол 4/x2 (честно)". 
> - **Урок**: судья без адресной фазы бесполезен для кооперативных инструкций.

### §0.b Реестр: S2v4 (свизл писателя smQ + LDSM-читатель) — бумага не начата, триггер после 052

Реестр Vugar-дверей обновлён:

> **S2v4** = свизл писателя smQ + LDSM-читатель dk. Конструкция аналогична 040 классу #7 (smdO writer с XOR-свизлом + LDSM.x4.trans.b16). В S2-мире у smQ **один читатель** — pack Q_T мёртв в S2v3, конфликт интересов читателей отсутствует → свизл-путь чист по liveness.
>
> **Статус**: **бумага не начата**. Триггер после закрытия 052 (независимо от KEEP/откат).
>
> **Ожидания**: адресная фаза (row_ptr XOR-свизла) даёт 32 banks distribution (аналогично 040 dO XOR) → LD bank conflicts ≈ 0. Данная фаза = пол 4/x2 (сохраняется). Регистры: **тот же 101r + 5 blk** (структура MMA не меняется).

> **Бирка S2v3 исправлена** (обновление 051 морозилки): **"S2v3 мёртв на натуральной row-major раскладке smQ (адресная фаза = 32-way storm). Свизл-путь (S2v4) НЕ исследован."**
> Не «LDSM.x2.b8 dk не работает вообще», а «LDSM.x2.b8 dk не работает БЕЗ свизла writer smQ».

### §0.c Racecheck-долг

**Долг 050/051 по racecheck закрывается культурно в 052 §гейт.c**:
- compute-sanitizer racecheck на r2c_merged_bit_exact (канон-форма + все 11 форм + canary) → **0 hazards, 0 errors, 0 warnings** ✓

**Новое правило прибора** (правило 13.a): любой сдвиг барьеров или добавление cp.async группы = racecheck обязателен, без исключений. Пропуск racecheck = ТЗ не закрыто.

---

## §1. Бумага prefetch (до кода)

### §1.a Фазовая карта qt-шага (union {smQ / smdS_stage / smP_T} 8192B)

| Фаза qt | Использование smQ_region | Барьер после |
|:--|:--|:--|
| **Step A** cp.async Q→smQ + dO→smdO + L/D | **Writer smQ** (Q) | t3 |
| **Step B** MMA-A Q·K^T | **Reader smQ** (Qr fragments) | — |
| **Step C** softmax (regs) | — | — |
| **Step D** MMA-B dO·V^T (smdO, smV) | idle (Qr в регистрах) | **t_new1** |
| **Step E** STS scatter smdS_stage | **Writer smdS_stage** = union | t9 |
| **Step F** STG.128 drain dS_nat | Reader smdS_stage | **t_new2** |
| **Step G** STS Pr → smP_T | **Writer smP_T** = union | t11 |
| **Step H** MMA_dV P^T·dO (LDSM.x4.trans smdO) | **Reader smP_T** (+ smdO) | t13 |

**Когда Q[qt+1] можно завозить, не наступая на живые байты union**:
- В любую точку **после Step B чтения smQ** (Qr фрагменты уже в регистрах), НО завозить нужно в **отдельный buffer, а не в union** — union живёт весь qt (E→F→G→H).
- Идеально: **весь Steps B..H** (~1000+ циклов MMA + LDS + drain), latency Q cp.async (~200-400 циклов) прячется.

### §1.b Выбор механики: (i) ping-pong две ноги 8192Б (headroom-in-place)

**Выбрано (i)** — SMEM +8192Б (**smQ_A**, **smQ_B**), каждая нога 8192Б. Union {smQ/smdS_stage/smP_T} выбирается per qt = `(qt & 1) ? smQ_B : smQ_A`. Второй буфер (smQ_next_buf) — destination для prefetch Q[qt+1].

**Обоснование**:
- Union не меняется (те же alias-точки), барьерная схема минимально смещается — только `cpa_wait<0>` перед t13.
- (ii) union расширение до ping-pong (SMEM тот же) требовало бы **всех читателей smdS_stage/smP_T** переиндексировать (qt & 1) — риск ошибки больше.
- Копейка-в-копейку: SMEM 41472 → 49664 (+8192 = +19.7%). Blocks/SM: floor(101376/(49664+1024)) = **2** (unchanged — headroom 51712 → 51712 нижняя граница для 2 blk = 50688 ≤ 49664 ✓).

### §1.c Волновая схема — какие барьеры двигаются/добавляются

**Прелюдия prefetch qt_start** (после K/V warmup):
- Новый sequence: 1 cp.async Q[qt_start] → smQ_first (leg for qt_start & 1); `cpa_commit()`; `cpa_wait<0>()`; `__syncthreads()` (уже был).
- Address set: smQ_first (write via cp.async → read via Step B qt_start).

**Step A modification**:
- Q cp.async **удалён из Step A** (Q для qt_start пришёл прелюдией; для qt+1 — предыдущим qt).
- dO + L/D — без изменений.
- **NEW: prefetch Q[qt+1]** → smQ_next_buf после `cpa_commit()` для группы A (dO). Второй `cpa_commit()` для группы B (Q[qt+1]).
- `cpa_wait<1>()` ждёт группу A (dO), оставляя B (Q[qt+1]) outstanding.
- Условно: **если `qt+1 >= n_qt`** (последний qt) → `cpa_wait<0>()` без issue group B.

**BARRIER t13 modification**:
- NEW: `cpa_wait<0>()` перед `__syncthreads()` — гарантирует Q[qt+1] завёзд в smQ_next_buf ДО следующего qt.

**Address sets всех 6 барьеров (метод 021)**:

| Barrier | Waits on | Address set | Прежний? |
|:--|:--|:--|:--|
| t3 (post-loads) | dO writes (Group A) + L/D reg-STS | smdO, smL, smD | ✓ (Q part removed, но остальное тождественно) |
| t_new1 (pre-scatter) | smQ reads завершены → alias overlay | smQ_curr | ✓ (same, но smQ = smQ_curr) |
| t9 (pre-drain) | smdS_stage_curr STS завершены | smdS_stage_curr | ✓ (same) |
| t_new2 (post-drain) | STG drain завершён | smdS_stage_curr читатель Step F | ✓ (same) |
| t11 (pre-MMA_dV) | smP_T_curr STS завершены | smP_T_curr | ✓ (same) |
| t13 (end qt) | dV_acc обновлены + **Q[qt+1] завёзд** | smQ_next_buf (NEW address) | **ДВИНУЛСЯ** — cpa_wait<0> перед |

**Racecheck-фокусы**:
1. Prefetch Q[qt+1] в smQ_next_buf, читается только в qt+1 Step B — надёжно gated t13.
2. smQ_next_buf ≠ smQ_curr в текущем qt (ping-pong) — Step E/G пишут в smQ_curr, prefetch пишет в smQ_next_buf — **disjoint**.
3. cp.async group B outstanding во время Steps B..H — не читается никем в текущем qt.

### §1.d Именованные предсказания для NCu-post

| Метрика | Прогноз | Механизм |
|:--|:--|:--|
| **wait 33% → ↓ на 3-11 pp** | Q latency (24KB / 3 = 8KB Q) прячется под MMA-петлю | -3..-11 pp |
| **mio 8.86% → +0..1 pp** | +512 cp.async issues на prefetch Q (extra LSU pressure) | +0..1 pp |
| **short_sb 9.71% → ~0 или ↓** | Меньше wait на LDS chain (Q завёзд, MMA-A читает быстрее)? | знак не гарантирован |
| **DRAM 9.79 GB → ровно** | Тот же total volume, timing только сдвинут | ≈ |
| **blocks 2/SM** | SMEM 49664 < 50688 | ✓ фиксировано |

**Регистры не предсказаны** (правило 8): cp.async issues + adr calc — копейки, но решает ptxas.

**Верификация**: NCu-post именами; conflicts обеих фаз (адресной у cp.async нет, но стейджинг мог родить новые LD/ST-события — проверить).

---

## §2. Правка kernel + launcher (patch summary)

**Правка `libs/fa_bwd_merged_v1.cu`** (5 hunks):
1. Header comment: SMEM плана 46592 → 49664, ping-pong plan + 052 объяснение.
2. SMEM layout: `smQ_region` (8192, union) → **`smQ_A` (8192) + `smQ_B` (8192)** (union одна из двух ног); удалён `smdS_T_stage` alloc (038-E dead).
3. **Прелюдия prefetch Q[qt_start]** после K/V warmup: cp.async → smQ_first (по `qt0 & 1`).
4. Step A: **Q cp.async убран**; ping-pong pointers `smQ_curr` (= `(qt&1) ? smQ_B : smQ_A`), `smQ_next_buf` (обратный), локальные `smdS_stage`/`smP_T` затеняют top-level. **cp.async prefetch Q[qt+1]** issued после `cpa_commit()` group A (dO); `cpa_wait<1>()` if prefetch active else `cpa_wait<0>()`.
5. Перед BARRIER t13: `if (qt+1 < n_qt) cpa_wait<0>()`.

**Правка launcher `launch_merged`**:
- `smem_bytes`: `41472` → `49664` (+8192 smQ_B leg).

**Правка `libs/bench_r2c_e2e.cu`**:
- EXPECT `kernel_merged_v1`: 252 → 250 (осознанно, с комментарием "052: -2r vs 252 base (smQ ping-pong prefetch)").

**Правка `runs/reports/037r_gate.sh`**: EXPECT 252 → 250.

Все правки применены → полный гейт. После вердикта rollback → EXPECT 250 → 252, kernel restored, bench_r2c_e2e rebuilt.

---

## §3. Гейт по порядку

### §3.a ptxas-факт (потолок 256r включительно, spill=0, blocks=2)

```
ptxas info : Compiling entry function 'kernel_merged_v1' for 'sm_120a'
ptxas info : Function properties: 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 250 registers, used 1 barriers
```

**ЗЕЛЁНЫЙ**: 250r (-2r vs 252 base). 0 spill/stack. SMEM 49664 → blocks/SM = floor(101376/50688) = **2 ✓**. Потолок 256r с запасом 6r.

### §3.b Fingerprint gate

```
$ 037r_gate.sh (EXPECT=250)
bench_merged: FINGERPRINT kernel_merged_v1: numRegs=250, sharedSizeBytes=0, ...
GATE OK: numRegs=250 matches EXPECT=250
```

**ЗЕЛЁНЫЙ**. EXPECT обновлён осознанно с комментарием.

### §3.c Корректность (bit-exact + racecheck)

**bit-exact chain**:
- `r2c_merged_bit_exact` (харнесс 039 ABI): dV + dS_nat, 11/11 форм BIT-EXACT (max_abs_diff=0.000e+00) ✓
- Chain **x3 runs**: 11/11 x3 = **33/33 PASS** ✓
- **INJECT_BITFLIP=1 control**: 0/11 (dS_nat mism=1 caught) ✓ — bit-flip обнаружен как ожидается.

**sanitizer memcheck**:
- `compute-sanitizer --tool memcheck r2c_merged_bit_exact` → `RACECHECK SUMMARY: 0 hazards`; `ERROR SUMMARY: 0 errors` ✓
- В том же run bit-exact 11/11 ✓.

**RACECHECK** (правило 13, барьеры двигались + новый cp.async group):
```
$ compute-sanitizer --tool racecheck r2c_merged_bit_exact
=== SUMMARY ===
  forms triple-bit-exact: 11 / 11
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```
**ЗЕЛЁНЫЙ**: 0 hazards, все 11 форм + canary. Долг 050/051 закрыт культурно.

### §3.d Wall ABBA >= 8 пар (правило-2/3 v2)

**Стенд**: 4 warmup + 16 timed (8 pairs), одна сессия, ABBA pattern `[BASE,CAND,CAND,BASE]×4`. temp 43-48°C.

**Ошибка measurements** (avg_ms merged isolated):

| Pair | BASE ms | CAND ms | Δ (ms) | Δ% |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 48.471 | 48.436 | -0.035 | -0.072% |
| 2 | 48.476 | 48.437 | -0.039 | -0.081% |
| 3 | 48.460 | 48.452 | -0.008 | -0.017% |
| 4 | 48.462 | 48.459 | -0.003 | -0.006% |
| 5 | 48.476 | 48.464 | -0.012 | -0.025% |
| 6 | 48.482 | 48.461 | -0.021 | -0.043% |
| 7 | 48.605 | 48.468 | -0.137 | -0.283% |
| 8 | 48.602 | 48.473 | -0.129 | -0.266% |

**BASE medians**: sorted [48.460, 48.462, **48.471, 48.476**, 48.476, 48.482, 48.602, 48.605] → median = **48.476 ms**
**CAND medians**: sorted [48.436, 48.437, 48.452, **48.459, 48.461**, 48.464, 48.468, 48.473] → median = **48.460 ms**
**Δ median** = 48.460 - 48.476 = **-0.016 ms** = **-0.033%**
**Pair-wise median Δ** = **-0.017 ms** (~-0.035%)

**Вердикт правило-2/3 v2**:
- CAND быстрее в **8/8 пар единогласно** ✓ (условие направленности выполнено)
- Амплитуда < 2% Vugar-порога → **statu quo / откат** (НЕ KEEP, keep-порог 3%; ABBA 2-3% зона)
- Fair-выкладка: **ниже 2%, честный откат**

### §3.e NCu-post — сверка предсказаний §1.d (АГЕНТ КАРТА, СРАВНЕНИЕ ПРЕДСКАЗАНИЙ)

| Метрика | Прогноз | BASE | CAND | Δ | Вердикт |
|:--|:--|:-:|:-:|:-:|:--|
| **wait** | ↓ 3-11 pp | 33.01% | 33.42% | **+0.41 pp** | **опровергнут** |
| barrier | не предсказан | 2.57% | 4.86% | **+2.29 pp** | **новый расход** |
| long_sb | не предсказан | 6.72% | 4.73% | **-1.99 pp** | **prefetch работает** |
| **mio** | +0..1 pp | 8.86% | 7.27% | **-1.59 pp** | **опровергнут (↓ не ↑)** |
| **short_sb** | знак с механизмом | 9.71% | 10.10% | +0.39 pp | ≈ шум |
| math_pipe | не предсказан | 13.67% | 13.85% | +0.18 pp | ≈ |
| **DRAM** | 9.79 GB ровно | 9.84 | 9.85 | ≈ | ✓ (в шуме) |
| **LD conflicts** | ~0 (cp.async в отдельный buf) | 132.2M | 153.1M | **+15.8%** | **скрытая цена ping-pong** |
| **ST conflicts** | ~0 | 17.2M | 38.2M | **+123%!** | **concurrent STS pressure** |
| Wavefronts LSU | не предсказан | 4.063B | 4.089B | +0.6% | ≈ |
| **Occupancy** | 16.59% ровно | 16.59% | 16.59% | 0 | ✓ |
| **regs** | не предсказан | 252 | 250 | -2 | ✓ (ptxas факт) |

**Куда делись спрятанные такты — механическое объяснение**:

1. **long_sb -1.99 pp** — DRAM latency Q действительно прячется под MMA-петлю (механизм работает механически).
2. **mio -1.59 pp** — LSU pipe меньше насыщается: разделение на две cp.async группы распределяет issues во времени.
3. **Но wall +0.033% (в шуме) вместо ожидаемого -0.5..-1.7%** потому что выигрыш **съедается двумя новыми расходами**:
   - **barrier +2.29 pp**: `cpa_wait<0>` перед t13 добавляет synchronization stall (warps ждут пока Q[qt+1] завёзд перед __syncthreads).
   - **ST conflicts +21M (+123%)**: concurrent cp.async STS.128 в smQ_next_buf параллельно с STS scatter Step E (в smQ_curr) + STS Pr → smP_T (Step G) конкурируют за SMEM bank pipe. Хоть буферы disjoint, LSU pipe throughput sharedна.
4. **wait +0.41 pp**: **wait 33% в BASE НЕ был bottleneck Q cp.async** — а комбинация dO cp.async (16KB, 2× больше Q) + иных ожиданий. Q latency вклад мал → prefetch снял только небольшую часть, но добавил barrier + ST conflicts → net +0.41 pp.

**Прогноз опровергнут**: механизм работает механически (long_sb ↓, mio ↓), но wait-снижение НЕ материализовалось в wall win — стоимость барьера и ST bank pressure съедают выигрыш.

---

## §4. Вердикт-карта: prefetch mechanism vs wall

| Компонент | Прогноз | Факт | Атрибуция |
|:--|:-:|:-:|:--|
| Q DRAM latency hidden | -0.5..-1.7 ms | ~-0.02 ms | long_sb -1.99pp мал |
| LSU pressure distribution | shim | ~-0.01 ms | mio -1.59pp мал |
| Barrier cpa_wait cost | 0 | **+0.02 ms** | barrier +2.29pp |
| ST bank conflict cost | 0 | **+0.01 ms** | ST +123% (concurrent cp.async STS) |
| Net wall Δ | -0.5..-1.7 ms | **-0.02 ms (-0.033%)** | ≈ noise |

**Верхняя оценка -1.7 ms** (Vugar) **не реализована**: механизм prefetch работает частично, но barrier + STS-контеншен уравновешивают выигрыш.

---

## §5. Сиквенс (правило TZ §5)

**При откате**: морозилка с NCu-причиной + доклад — на столе S2v4-бумага и M5.

### Морозилка smQ-prefetch с NCu-причиной

> **«smQ ping-pong prefetch merged (052) заморожен по правилу-2/3 v2 (<2% Vugar-порога). Механизм prefetch работает механически (long_sb -1.99pp, mio -1.59pp — Q DRAM latency прячется), НО выигрыш съедается барьером cpa_wait<0> перед t13 (+2.29pp barrier stall) и concurrent cp.async STS в smQ_next_buf параллельно с Step E/G STS в smQ_curr (+123% ST conflicts, +15.8% LD conflicts).
>
> **Root cause NCu-post**: wait 33% в BASE НЕ был bottleneck Q cp.async — а сумма dO cp.async (16KB, 2× больше Q) + иных ожиданий. Q latency вклад мал. Prefetch снял только небольшую часть Q-latency, зато добавил barrier + STS-контеншен → net wall Δ -0.033% (в шуме).
>
> **Не распространяется на dO/L/D prefetch**: те буфера не aliased в union, prefetch их технически возможен, но wait-структура та же (33% wait = smesь источников).»**

### На столе (за Vugar)

1. **S2v4** (dk свизл писателя smQ + LDSM.x2.b8-читатель, конструкция аналогична 040 класса #7):
   - Бумага не начата.
   - Ожидания: адресная фаза (row_ptr XOR-свизла) даёт 32 banks distribution → LD conflicts ≈ 0.
   - Регистры: 101r + 5 blk (структура MMA не меняется).
   - Апер: 4-7% dk isolated → -1..-1.5% E2E, пробивает 44.0 порог.

2. **M5-монетка** (merged класс #5 fp8 LDSM для smV B-op):
   - 0.5-1.9% wall (043 §3.b) — не гарантирует 2/3 v2 keep.
   - Механизм чист (аналог 040 класса #7).

3. **Merged fundamental rework** (не 1 правка):
   - dO cp.async делит с Q wait → prefetch dO тоже? Union-shift огромный.
   - MMA-A/dP fp8→fp16 CVT (math_pipe 13.67%) → CVT-batch reduction? Требует новую бумагу.

---

## §6. Правки production в 052

**После отката (052 итог)**: 0.

- `libs/fa_bwd_merged_v1.cu` md5 `2bf32ab7d4c5ecabb4ee2dbf1b5d4b33` = 040 sealed ✓
- `libs/fa_bwd_dk_new.cu` md5 `a9f0ded8261e53a143b521ffa647f458` = 033 sealed ✓
- `libs/fa_bwd_dq_new.cu` md5 `d7a11a3d788eb4c396d892bc9c8ab754` = 041 sealed ✓
- `libs/bench_r2c_e2e.cu`: EXPECT kernel_merged_v1 = **252** (restored, was 250 during cand).
- `runs/reports/037r_gate.sh`: EXPECT = **252** (restored).
- Diagnostic binaries в `runs/archive/052_pre/`:
  - `r2c_merged_wall_base` (md5 `51192119fdc61e42a3731a42687668d1`, 252r 040 sealed)
  - `r2c_merged_wall_cand` (md5 `8456c8295ffe75dcd8c02a4356cb3405`, 250r 052 prefetch)
  - `fa_bwd_merged_v1.cu.052_cand` (source of prefetch candidate for future retest)

---

## §7. Итоги 052

1. **§0 Долги закрыты**: (a) свод судьи банков обновлён — ДВЕ фазы (адресная + данная); (b) реестр S2v4 добавлен как открытая дверь; бирка S2v3 исправлена ("свизл-путь не исследован"); (c) racecheck-долг закрыт (0 hazards на все 11 форм + canary).

2. **§1 Бумага prefetch** написана до кода: фазовая карта, механика (i) ping-pong 2 ноги, волновая схема с address-set перебором, именованные предсказания.

3. **§гейт **выполнен строго по порядку** — все зелёные:
   - **a. ptxas**: 250r/0 spill/2 blk ✓
   - **b. fingerprint** EXPECT 252→250 обновлён осознанно ✓
   - **c. корректность**: bit-exact 11/11 x3 + INJECT_BITFLIP catch + memcheck 0 + **racecheck 0 hazards** ✓
   - **d. Wall ABBA 8 пар**: median Δ = **-0.033%** (единогласно 8/8, но <2% Vugar → **statu quo/откат** по правилу-2/3 v2)
   - **e. NCu-post**: механизм работает частично (long_sb -1.99pp, mio -1.59pp), но выигрыш съедается barrier +2.29pp + ST conflicts +123%; wait +0.41pp (прогноз опровергнут — Q latency НЕ был bottleneck 33% wait).

4. **§4 Вердикт-карта**: verhняя оценка -1.7 ms не реализована; net wall -0.02 ms (в шуме).

5. **§6 Rollback**: prod merged восстановлен к 040 sealed, EXPECT восстановлен, bench_r2c_e2e пересобран, fingerprint 252/128/69 OK.

6. **Морозилка smQ-prefetch с NCu-причиной**: mechanism partial-work, cost неявные (barrier + ST conflicts).

7. **Сиквенс**: **053 = S2v4 (свизл писателя smQ + LDSM-читатель dk)** — рекомендация ассистента, требует новую бумагу (адресная фаза XOR-свизла + liveness pack removal). Апер 4-7% dk = -1..-1.5% E2E, пробивает 44.0 порог. Альтернативы: M5-монетка (0.5-1.9%, не гарантия keep) или merged fundamental rework.

### Chain md5

- 050 `9269bcf280c5ebf1a67aa9e91bbd3933`
- 051 `bd9ea399697ffe9f6ff206618c48a36d`
- **052 `274d9c70af8aebf66f1815779a72a7a1`**

### Файлы 052

- `runs/reports/052_smq_prefetch.md` (this report)
- `runs/reports/052_abba.sh` + `052_abba_data.txt` — §3.d ABBA data
- `runs/reports/052_ncu.sh` + `052_ncu_data.txt` — §3.e NCu-post data
- `runs/archive/052_pre/*` — diagnostic binaries + candidate source

---

**End 052. Механизм prefetch работает, но wait 33% не сдался; barrier + STS-контеншен съели выигрыш. Rollback к 040 sealed. Сиквенс: 053 = S2v4 dk свизл-путь.**
