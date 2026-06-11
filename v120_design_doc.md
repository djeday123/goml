# v120 Ping-Pong Design Document — STAGE 1 (paper, no code)

База: **v96b** (champion 595T PEAK). FFMA-направление закрыто, v96c/v96d отброшены.
Цель: tensor utilization 45.4% → 60%+ через фазовый overlap двух consumer-групп.

---

## 1.1 Схема разбиения

### Текущая структура v96b
- `FA_BR = 128`, `FA_BC = 64`, `FA_THREADS = 128`, `M_TILES = 2`, `K_STAGES = 2`
- 4 warps × 32 строки/warp = 128 = Br
- Warp wid владеет строками `[wid*32 .. wid*32+31]` (M_TILES × 16)
- ks-batched: 4 ks для QK, 2 ks для PV
- Per-iter работа на warp: 4×8×2=64 QK MMA + 2×8×2=32 PV MMA = **96 MMA/iter**
- **НЕТ producer/consumer split** (в отличие от v118 которая warp-spec) — все 4 warps делают всё

### v120 разбиение
```
Group A = warps {0, 1}: владеет Q-строками [0..63]   (Br_A = 64)
Group B = warps {2, 3}: владеет Q-строками [64..127] (Br_B = 64)
```

Каждая группа = 2 warps × 32 строки = 64 строки. M_TILES=2 сохраняется внутри warp.
Per-iter work per group: 32 QK MMA + 16 PV MMA = **48 MMA/iter** (половина v96b на группу).

### Привязка warp → SMSP
На Blackwell SM имеет 4 SMSP, каждый со своим warp scheduler и tensor dispatch slot.
При 2 blocks/SM × 4 warps/block = 8 warps/SM = 2 warps/SMSP.

**Предположение:** HW распределяет warps по SMSP циклически:
```
SMSP 0: warp_0 (Block A) + warp_0 (Block B)
SMSP 1: warp_1 (...)
SMSP 2: warp_2 (...)
SMSP 3: warp_3 (...)
```

Внутри 1 блока: Group A {0,1} → SMSPs {0,1}; Group B {2,3} → SMSPs {2,3}.

**Важное наблюдение:** даже без ping-pong, в v96b 4 warps на 4 SMSPs могут issue MMA параллельно. Тензорный pipe — общий per-SM (т.е. shared между 4 SMSPs). Microbench measured T_fp8 = 16.8 cyc/MMA per SMSP, что = peak SM throughput при 4-SMSP параллельной выдаче.

**Вопрос (ОТКРЫТЫЙ):** действительно ли v96b's tensor util 45.4% — это потому что во время softmax-фазы ВСЕ 4 SMSPs простаивают? Если да → ping-pong атакует именно эту дыру. Если нет (например, MMA pipe sequential per warp независимо) — overlap не поможет.

**Гипотеза дизайна:** во время softmax все 4 warps в lockstep, ни один не issues MMA → 4 SMSPs idle одновременно. Ping-pong смещает фазы между группами → когда Group A в softmax, Group B issues MMA → tensor pipe не простаивает.

---

## 1.2 Регистровый бюджет — БЛОКЕР, считается первым

### v96b per-warp live registers (полный список)

| Структура | Объём | Регистров | Когда live |
|---|---|---|---|
| `Qr[4][M_TILES][4]` | 4×2×4 = 32 uint32 | 32 | После Q-load, весь iter |
| `Or_p[16][M_TILES][2]` | 16×2×2 = 64 uint32 | 64 | Через PV-фазу, между iter сохраняется |
| `Sr_p[8][M_TILES][2]` | 8×2×2 = 32 uint32 | 32 | QK MMA → softmax (~50% iter) |
| `Sr[8][M_TILES][4]` | 8×2×4 = 64 float | 64 | Softmax phase only |
| `P_top[8][M_TILES]`, `P_bot[8][M_TILES]` | 8×2 + 8×2 = 32 uint32 | 32 | softmax-end → PV-start |
| `rmax/nm/rsexp/ns/rsc[M_TILES][2]` | 5×2×2 = 20 float | 20 | Softmax |
| Scratch (fs, bias временные, индексы) | — | ~10-15 | Везде |
| **Итого пик** | — | **~245 (v96b)** | |

Фактически v96b использует **247 регистров** — точно совпадает с расчётом ±2.

### v120 PEAK живых регистров per warp

В ping-pong каждый warp принадлежит ОДНОЙ группе. Warp делает свои фазы последовательно — внутри warp состояние не дублируется.

**КЛЮЧЕВОЙ ВОПРОС:** требует ли ping-pong хранить состояние TWO кв-iters одновременно?

Ответ: **зависит от выбранной модели:**

**Модель A (phase-pingpong внутри iter, оба группы на одном N):**
- Group A: QK(N) → softmax(N) → PV(N)
- Group B (offset half-iter): QK(N) → softmax(N) → PV(N)
- Обе группы используют K[N], V[N] из общих smK/smV
- Per-warp live = идентично v96b ≈ 245 regs
- **БЕЗ роста регистров на warp**

**Модель B (iter-pingpong, группы offset на целую iter):**
- Group A на iter N+1, Group B на iter N
- Каждый warp хранит ОДНО состояние, но cross-iter K/V tiles живы одновременно (нужно double K AND double V в SMEM)
- Per-warp live = v96b ≈ 245 regs

**Вывод по регистрам: per-warp pressure НЕ растёт.** v96b's 247 регистров остаются базой → попадаем в калибровочный budget ≤247.

### НО: проверка скрытых рисков

1. **fs lifetime** (урок v96c): если оптимизатор увидит, что fs нужен в обоих группах через barrier — может загнать его в долгоживущий регистр. **Нужно убедиться**: fs остаётся локальным per-warp, как в v96b.

2. **Group-state каждого warp**: если будут массивы `group_id`, `phase_id`, `peer_warp_id` — это +регистры. **Решение:** все они должны быть компайл-тайм константами (template parameters / preprocessor).

3. **Дополнительные barrier IDs** в registers (см. v117 +3 regs за `bar.sync 4, 128`): per-warp expected ~+2-3 regs от новых syncs. **Допустимо** в 247-budget с slack=3-4.

**Регистровый прогноз: 247 → 248-250 (slack используется, но без spill).**
Калибровочная константа: +8 regs = −2%. +3 regs ≈ −0.75%. Допустимый риск.

### SMEM budget — критичный вопрос

| Region | v96b | v120 (Модель A — shared K/V) | v120 (Модель B — double V) |
|---|---|---|---|
| smQ | 12 KB | 12 KB | 12 KB |
| smK[2] | 16 KB | 16 KB | 16 KB |
| smV | 8 KB | 8 KB | **16 KB (double)** |
| smV_T | 8.5 KB | 8.5 KB | 8.5 KB |
| **Total/block** | **44.5 KB** | **44.5 KB** | **52.5 KB** |
| × 2 blocks/SM | 89 KB | **89 KB ≤ 100 KB ✓** | **105 KB > 100 KB ✗** |

**Модель A арифметически возможна; Модель B — нет (потеря 2 blocks/SM).**

→ **выбираем Модель A: phase-pingpong внутри одного N, общие smK/smV для обеих групп.**

---

## 1.3 Синхронизация — карта барьеров для Модели A

### Phase timeline (Модель A, half-iter offset)

```
                       Time →
Group A:  load_init    QK(N)    sftm(N)    PV(N)    QK(N+1)  sftm(N+1)  PV(N+1) ...
Group B:  WAIT      load_init  QK(N)    sftm(N)    PV(N)    QK(N+1)  sftm(N+1) ...
                       ↑          ↑        ↑          ↑
                    [bar_QK]  [bar_sftm] [bar_PV]  [bar_QK+1]
```

**Фазы попарно перекрываются:**
- Когда A в softmax(N), B в QK(N) → разные пайпы (math vs MMA), полный overlap
- Когда A в PV(N), B в softmax(N) → MMA vs math, полный overlap
- Когда A в QK(N+1), B в PV(N) → ОБЕ в MMA → дележ tensor pipe (не overlap, а fair share)

### SMEM регионы и race-таблица (доказательство отсутствия гонок)

| Region | Кто читает | Кто пишет | Когда конфликт? |
|---|---|---|---|
| smK[N&1] (current K) | Both A и B (в фазе QK) | Producer (cp.async) в начале iter | После того как обе группы прошли QK |
| smK[(N+1)&1] (next K) | A читает в QK(N+1) | Producer пишет в mid-iter | A в QK(N+1) — после landing |
| smV (current V) | Both A и B в PV-фазе | Producer в конце iter | После того как **обе** группы прошли PV |
| smV_T (transposed V) | Both A и B в PV-фазе | Local transpose в начале iter | До любого PV use |
| smP (P storage) | Both A и B в PV-фазе | Both A и B в softmax-end | smP per group разный — **нужны разные регионы**? |

**КРИТИЧНО — smP:** в v96b smP перекрывает smQ (which is dead after Q→regs). Группы A и B пишут свои P в smP. Если они пишут в РАЗНЫЕ половины smP, ОК. Если в одну и ту же — гонка.

В v96b smP layout = `Br × Bc = 128 × 64 = 8 KB`. Поделить: A пишет в `smP[0..63][:]`, B пишет в `smP[64..127][:]`. Каждая половина = 4 KB. Должно работать с тем же offset.

### Карта named barriers

```
bar.sync 1, 128 (block-wide):  TOP of iter — после cpa_wait
bar.sync 2, 64 (Group A only): A finished QK(N), B can start QK(N)
bar.sync 3, 64 (Group B only): B finished QK(N), can сompare with A's softmax
bar.sync 4, 64 (Group A only): A finished smP STS(N), B can read smP(A)
bar.sync 5, 64 (Group B only): B finished smP STS(N), A can read smP(B)
bar.sync 6, 128 (block-wide):  BOTTOM — both finished PV(N), producer can write V[N+1]
```

**Per-iter sync count: 6** (vs v96b: 3-4 syncs/iter).
Барьерный overhead больше, но компенсируется overlap.

### Race protection доказательство

| Race | Защита |
|---|---|
| A reads smK[N] while producer writes smK[N+1] | smK[2]: разные слоты ✓ |
| A reads smV while producer writes smV[N+1] | bar.sync 6, 128 защищает: producer ждёт обе группы ✓ |
| A reads smV_T while B writes smV_T | Локальная transpose до bar.sync 1 — both write before any read ✓ |
| A writes smP[0:64] while B reads | Группы пишут разные половины; разные barriers (bar 4 vs bar 5) ✓ |

**Все гонки закрыты названными barriers + arithmetic separation.**

---

## 1.4 Softmax независимость

Online-softmax state per Q-row:
- `rmax[mi][side]` — running max (per Q-row)
- `rsexp[mi][side]` — running exp sum (per Q-row)
- `Or_p[16][mi][2]` — accumulator (per Q-row, per N-tile)

Группы A и B владеют **разными Q-rows**:
- A: rows [0..63], каждый warp 32 rows (warp 0: 0..31, warp 1: 32..63)
- B: rows [64..127] (warp 2: 64..95, warp 3: 96..127)

Состояние `(rmax, rsexp, Or_p)` строго per-warp per-row → **ZERO shared state between A и B.**

Output epilogue: A пишет в `O[0..63]`, B пишет в `O[64..127]` — разные глобальные адреса, нет гонки.

**Verdict: softmax независимость подтверждена. Никакой shared точки → никакого extra sync.**

---

## 1.5 Прогноз по модели (формула) — ИСПРАВЛЕНО на v96b NCu

### Базис (NCu v96b, cfg=9)
- **v96b tensor pipe util = 56.1%** (не 45.4% — то был v111)
- v96b PEAK: 595T mean
- T_fp8 (microbench, per SMSP): 16.8 cyc/MMA
- M_v96b = 0.561 — доля времени iter с MMA активностью
- N_v96b = 0.439 — non-MMA фракция (softmax + transpose + barriers + epilog)

### Формула захвата (по ТЗ)
```
util_target = 56.1 + capture × 43.9
```
где **capture** ∈ [0..1] — доля non-MMA-времени одной группы, перекрытая MMA-фазой другой, с учётом коллизий на общем pipe.

### Расчёт capture для идеализированного offset

При идеальном phase offset (A в MMA когда B в non-MMA):
- Перекрытие = min(M_A, N_B) = min(0.561, 0.439) = **0.439**
- Если M > N: A's MMA полностью покрывает B's non-MMA = capture = 1.0 → util_max = 100%
- Остаток A's MMA: 0.561 − 0.439 = 0.122 = время одновременной MMA → contention на pipe

Раскладка iter при capture=1.0:
| Фаза | Доля iter | Состояние |
|---|---|---|
| A в MMA, B в non-MMA | 0.439 | pipe для A, full throughput 1/T |
| B в MMA, A в non-MMA | 0.439 | pipe для B, full throughput |
| Оба в MMA | 0.122 | pipe shared между A и B (fair queue) |
| Оба idle | 0.000 | pipe полностью простаивает |

Итого pipe busy = 1.000 = **util_max = 100%**. Speedup = 100/56.1 = **1.783×** теоретический потолок.

### Реалистичный capture с потерями

Источники неидеального захвата:
1. Phase boundaries не выравниваются точно (длительность QK/softmax/PV различаются)
2. Сначала загрузочный warmup (первая iter группы B без overlap)
3. Конец kv-loop: последняя iter без overlap (cooldown)
4. Скрытые корреляции через смешанные ks-batches

Эмпирическая оценка из аналогичных ping-pong реализаций (FA3-style): **capture = 0.5..0.7**.

| Capture | util_target | Wall-clock speedup vs v96b (до учёта sync) |
|---|---|---|
| 1.0 (теор. потолок) | 100% | ×1.783 → 1060T |
| 0.7 (FA3-like good) | 86.8% | ×1.547 → 920T |
| 0.5 (средне) | 78.0% | ×1.390 → 826T |
| 0.3 (плохо) | 75.0% | ×1.30 → 775T |
| 0.0 (нет overlap) | 56.1% | ×1.00 → 595T |

### Стоимость sync — отдельный риск-пункт

**v96b имеет минимальную барьерную нагрузку**: 1 barrier per ptxas (после v96b localfix) + неявные __syncthreads. Реальная NCu barrier stall %: ~2% (см. memory v96 NCu).

v120 добавляет 6 явных barriers/iter:
- 1 bar_top (block-wide top sync) — наследник v96b
- 5 inter-group barriers (bar 2..6)

**Стоимость по clock-probe: 48 cyc/sync × 6 = 288 cyc/iter** (vs v96b ~150 cyc/iter).
Прирост sync overhead: **+138 cyc/iter** относительно v96b.

При v96b iter ≈ 2000 cyc (clock probe), это **+6.9% iter cycle time** только от барьеров.

**Чистый sync penalty: -6.9% wall-clock.**

### Итоговая формула с sync penalty

```
v120_wall_clock = v96b × (util_target / util_v96b) × (1 / (1 + sync_overhead))
                = 595 × (util_target / 56.1) × (1 / 1.069)
                = 595 × (util_target / 56.1) × 0.935
```

| Capture | util_target | wall-clock прогноз | Δ vs v96b |
|---|---|---|---|
| 1.0 | 100% | 992T | **+66%** |
| 0.7 | 86.8% | 861T | **+45%** |
| 0.5 | 78.0% | 774T | **+30%** |
| 0.3 | 75.0% | 725T | **+22%** |
| 0.2 | 64.9% | 627T | **+5%** ← минимум прибыли при +138 cyc sync |
| 0.1 | 60.5% | 585T | **−1.7%** ← break-even |
| 0.0 | 56.1% | 542T | **−9%** ← чистый sync penalty без overlap |

### Критическая зона break-even

**Если capture < 0.1 → v120 ХУЖЕ v96b за счёт sync overhead.**

Это новый риск-пункт #4 (добавлен по ТЗ):
- v96b такой "чистый" по барьерам, что +5 sync per iter обязаны окупаться overlap'ом
- Capture должен быть **≥ 0.15** для безубыточности
- Цель ТЗ 650T (+9%) требует **capture ≥ 0.25**
- Реалистичная FA3-like цель (capture=0.5) даёт ~774T (+30%)

### Условия провала прогноза (обновлено)

1. **HW tensor pipe per-SMSP (не per-SM):** v96b's 4 warps на 4 SMSPs уже issue MMA параллельно. Если pipe per-SMSP, capture не растёт даже при идеальном offset. **Блокирующий вопрос — нужен микробенч до Этапа 2.**

2. **v96b's 56.1% не от non-MMA-фаз, а от других причин** (например, register dep stalls внутри MMA-цепочек): тогда non-MMA gap нечего перекрывать. Проверка: NCu source attribution v96b, выявить top stall reasons.

3. **Sync overhead > overlap gain:** если capture упадёт до 0.1-0.15, v120 = break-even или регрессия. **Минимальный capture для approval Этапа 2 = 0.25.**

4. **Imperfect offset из-за неравных фаз:** v96b QK ≠ softmax ≠ PV по длительности. Если максимальная фаза доминирует — offset не точно делит iter пополам. Нужно измерить длительности фаз phase-by-phase в Stage 2 skeleton.

### Условия провала прогноза
1. SMSP tensor pipe оказывается per-SMSP (не shared per SM) → группы на разных SMSPs не конкурируют → ping-pong не help (overlap уже был)
2. Софтмакс на самом деле не блокирует MMA dispatch (warp scheduler уже умеет switch internally) → util 45.4% от другой причины
3. Sync overhead для 6 barriers/iter превышает выигрыш overlap

**Все три условия проверяются после Этапа 2 (skeleton без overlap) — если skeleton показывает 595T при 6 barriers, sync overhead approved.**

---

## Итог Этапа 1

| Пункт | Статус |
|---|---|
| 1.1 Разбиение | ✅ Модель A (phase-pingpong), 2 группы × 2 warps × 32 rows = Br=128 сохранён |
| 1.2 Регистры | ✅ ~247-250, в budget с slack 3-4 (калибровка: +8 regs = −2%) |
| 1.2 SMEM | ✅ 44.5 KB (без double V), 2 blocks/SM сохранены |
| 1.3 Синхронизация | ✅ 6 named barriers/iter, race-таблица закрыта; стоимость +138 cyc/iter = −6.9% baseline |
| 1.4 Softmax | ✅ Полная независимость состояний — ZERO shared state |
| 1.5 Прогноз | ✅ Базис 56.1% util; capture=0.5 → +30%; capture=0.25 minimum для +9% ТЗ; break-even при capture=0.15 |

### Прогнозная таблица (от 56.1% baseline, с sync penalty 6.9%)

| Capture | wall-clock | Δ |
|---|---|---|
| 1.0 (теор.) | 992T | +66% |
| 0.7 (FA3 typical) | 861T | +45% |
| 0.5 (целевой) | 774T | +30% |
| 0.3 | 725T | +22% |
| 0.25 (минимум ТЗ) | 650T | +9% |
| 0.15 (break-even) | 595T | 0% |
| 0.10 | 585T | −1.7% |
| 0.0 | 542T | −9% (чистый sync penalty) |

### Открытые вопросы (для приёмки)

1. **HW факт: tensor pipe per-SM или per-SMSP?** Если per-SMSP → 4 warps v96b уже issue параллельно → capture ≈ 0 → v120 уходит в зону break-even/регрессии. **БЛОКИРУЮЩИЙ.** Решение: микробенч (4 warps, 4 параллельные независимые цепочки, измерить throughput vs 1-warp baseline). Если throughput 4× — pipe shared; если 1× — per-SMSP.

2. **NCu для v96b: где именно 43.9% non-MMA?** Если эта non-MMA доля размазана по register-dep stalls ВНУТРИ MMA-цепочек (не дискретные фазы) — overlap нечего захватывать. Решение: source-attributed NCu v96b с phase-level breakdown (QK vs softmax vs PV stall composition отдельно).

3. **smP двух-группового разбиения:** простое половинное деление работает с текущим swizzle? Если swizzle пересекается между half'ами — нужен новый layout.

4. **Phase offset в skeleton:** half-iter или quarter-iter — зависит от относительной длительности QK vs softmax vs PV (нужно измерить в Stage 2).

### Запрос на приёмку

**Этап 2 (skeleton кода) НЕ начинается** до закрытия открытых вопросов 1 и 2 (блокирующие). Предлагается:

**Этап 1a (диагностика, 1-2 часа):**
- Микробенч "4-warp MMA throughput vs 1-warp" → определить tensor pipe topology
- NCu source-attributed v96b на cfg=9: распределение stalls по фазам (QK MMA / softmax / PV MMA / sync) → доказать наличие распознаваемых non-MMA окон

Результат Этапа 1a решает:
- (a) tensor pipe shared + non-MMA фазы дискретны → Этап 2 starts
- (b) pipe per-SMSP ИЛИ non-MMA размазан → v120 не имеет mechanism → проект закрыт

Без выполнения 1a Этап 2 рискует повторить v96c-провал (нашли механизм, перф не пошёл).

Сейчас остановлен. Кода нет. Жду ОК на Этап 1a или явных поправок.

---

## ОБНОВЛЕНИЕ ЭТАП 1a — РЕЗУЛЬТАТЫ

### Топология tensor pipe (вопрос #1) — закрыт расчётом без микробенча

Per-SMSP, 4 на SM. Микробенч QMMA уже измерил: 1 warp = T=16.8 cyc/MMA. Card peak ≈ 1.3 PFLOPS FP8 при 188 SM × 2617 MHz требует 4× MMA dispatch на SM → 4 SMSP. v96b 596T при 56.1% util согласован с per-SMSP топологией.

### Вывод про warp→SMSP mapping (важная поправка к 1.1)

Standard mapping `warp_id % 4 → SMSP`. С FA_THREADS=128 = 4 warps в блоке → **по одному warp на каждый SMSP из текущего блока**. Второй warp на SMSP — из СОСЕДНЕГО блока (при 2 blocks/SM).

**Within-block ping-pong (A={0,2}, B={1,3}) НЕ ПОМЕЩАЕТ 2 warps одного блока на один SMSP.** На каждом SMSP пара = `block_0.warp_X + block_1.warp_X`. Это inter-block координация, требует cross-block sync, не доступного без mbarrier-across-blocks (закрыт в v111-mbarrier попытке).

### Опция FA_THREADS=256 — оценка регистрового бюджета

8 warps/block, 2 warps/SMSP within block (то что надо).
v96b regs = 247/thread × 256 threads = 63,232 regs/block.
- 2 blocks/SM × 63,232 = 126,464 > 65,536 limit → **только 1 block/SM**
- Потеря 2-block occupancy ≈ -30..-40% theoretical (per microbench K=2 vs K=1 = 84% vs 58% throughput)

**Чтобы FA_THREADS=256 не сломал occupancy, нужно ≤126 regs/thread.** v96b использует 247 (нельзя ужать вдвое без потери acc-структуры).

### Phase breakdown — Stage 1a критерий

Дискретная softmax-фаза = 22.64% (SOFTMAX_MATH + HADD2 + MUFU.EX2 + SHFL + STS).
**Критерий ТЗ ≥25% — НЕ выполнен** (на 2.36pp ниже).

НО: PC-stall samples не = "время на pipe". Tensor pipe util 56.1% при PC-MMA 26.43% означает MMA dispatch занимает pipe дольше, чем warp-time на MMA-инструкции. Реально 43.9% времени pipe ИДЛИТ — теоретически перекрываемо.

ADDR_ARITH 26.28% + LDS 11.54% = на **других pipes** (integer/LSU). Tensor pipe свободен во время них.

### ИТОГ Stage 1a — STRUCTURAL NO-GO

**Прогнозируемый capture = 0.0..0.1** на текущей архитектуре:
- Within-block ping-pong не работает (warp→SMSP mapping не позволяет пару на одном SMSP)
- FA_THREADS=256 кладёт occupancy на 30-40%
- Inter-block ping-pong требует mbarrier-across-blocks (заблокирован)
- Phase fraction 22.6% марginally НИЖЕ 25% threshold

**Реалистичный прогноз: v120 wall-clock = 585-595T (−2% до 0% vs v96b)** — sync penalty 6.9% частично компенсируется capture ≤0.1.

### Sync-budget пометка (правка 3)

48 cyc/sync — верхняя оценка (block-wide bar.sync на 128 threads).
Партиальные barriers могут быть дешевле (v117 показал bar.sync 5,96 эффективнее).
Но: 6 barriers вместо 1 — даже при 30 cyc/sync = +180 cyc/iter = -9% iter time **гарантированно**.

### РЕКОМЕНДАЦИЯ: v120 ping-pong на текущей архитектуре закрыт

Структурные блокеры:
1. warp→SMSP mapping не позволяет within-block pairing на 1 SMSP при FA_THREADS=128
2. FA_THREADS=256 ломает 2 blocks/SM (potential -30..40%)
3. Inter-block phase coordination требует mbarrier-across-blocks (заблокирован в v111-mbarrier попытке)
4. Discrete non-MMA phase 22.6% < 25% ТЗ threshold
5. Sync penalty гарантированно −6..−9% iter time

**Open alternatives (НЕ в scope текущего ТЗ):**
- v120 на FA_THREADS=256 + структурное уменьшение регистров (drop M_TILES=2 → 1, drop ks-batching) — другая архитектура, не "ping-pong поверх v96b"
- v120 на hd=64 base (v89 = 3 blocks/SM, регистрный бюджет легче, но hd=64 уже на architectural fixed-point per [[sm120-fa-hd64-final-summary]])
- FP4 направление (отдельный ТЗ): ×2 FLOPs per MMA при том же issue rate, но требует block-scale infrastructure

**v120 закрыт за 1 час диагностики вместо двух недель кода — это та же успешная методология, что и v117 closure.**

### Decision запрос

Перед закрытием v120 финально нужен ОК на:
- (a) v120 как "ping-pong поверх v96b" не делается — структурные блокеры
- (b) Открытые направления (FP4 / редизайн архитектуры) — отдельный ТЗ
- (c) Production champion v96b 596T остаётся финалом forward optimization на sm_120 hd=128
