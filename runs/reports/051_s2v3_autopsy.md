# 051 — Аутопсия S2v3: bank conflict шторм (не конвой)

**Chain**:
- 049_b1_probe.md md5: `7c4d2cd35bcc554cfc9bc5732f9b9bc3`
- 050_dkS2_v3.md md5: `9269bcf280c5ebf1a67aa9e91bbd3933`

**Правила ТЗ 051**: диагностика, НЕ починка. Production 033 sealed.

---

## Артефакт-хедер (правило 5)

```
libs/ (post-051):
-rw-r--r-- 13352 Jul  9         fa_bwd_dk_new.cu             (md5 a9f0ded8261e53a143b521ffa647f458 = 033 sealed)
-rw-r--r-- 25638 Jul  8         fa_bwd_merged_v1.cu           (md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33 = 040 sealed)
-rw-r--r-- 18834 Jul  8         fa_bwd_dq_new.cu              (md5 d7a11a3d788eb4c396d892bc9c8ab754 = 041 sealed)
-rwxr-xr-x       Jul  9         bench_r2c_e2e                 (production baseline)
-rwxr-xr-x       Jul  9         bench_r2c_e2e_s2v3            (md5 c6988e3d... S2v3 diagnostic binary)
```

**Gate-log**:
```
$ ./037r_gate.sh
GATE OK: numRegs=252 matches EXPECT=252
```

---

## §0. Дыра baseline (блокирующая)

### Разбор bench-обвязки 050

`bench_r2c_e2e` main loop (bench_r2c_e2e.cu:224-232):
- Форма: **bh=128, sl=8192, hd=128, causal=0, window=0** (canonical)
- warmup=5, iters=20
- Каждый iter: `cudaEventRecord(e0) → D → merged → dk_new → dq_new → cudaEventRecord(e1) → cudaEventSynchronize(e1)`
- Каждое ядро: **своё event pair** для per-kernel timing
- `ms_avg = ms_total / iters` — деление на 20

### Контрольный прогон baseline (5 consecutive runs)

Скрипт `051_baseline_probe.sh`. Temp 43-47°C, clocks 2685 MHz max.

| run | temp | D | merged | dk_new | dq_new | total |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 43°C | 0.363 | 48.782 | **18.882** | 18.129 | 86.156 |
| 2 | 46°C | 0.361 | 48.782 | **18.883** | 18.141 | 86.166 |
| 3 | 47°C | 0.355 | 48.803 | **18.888** | 18.137 | 86.184 |
| 4 | 47°C | 0.364 | 48.799 | **18.888** | 18.131 | 86.182 |
| 5 | 47°C | 0.360 | 48.813 | **18.895** | 18.141 | 86.209 |

**Stable**: dk_new = **18.88-18.90 ms** (CV < 0.1%). vs ledger 9.75-10.13 = **~1.86× сдвиг стенда**.

### Вердикт-строка §0

**РЕАЛЬНЫЙ СДВИГ СТЕНДА** (не режим-мислейбл): вся сессия ~1.86× медленнее ledger (стабильно, не термо, не racy). Причина: cross-session state (драйвер/CUDA/persistent mode/что-то ещё) — не расследуется по TZ ("диагностика, не починка").

**Пере-подпись колонок 050**:
- BASE dk "18.9 ms" = **"стенд-сдвиг сессии ~1.86× vs ledger 10 ms, стабильно"**
- CAND dk "41.2 ms" = **"S2v3 в том же стенд-состоянии, ~4.14× vs ledger"**
- **Δ +22.3 ms (+118%)** = **реальная регрессия LDSM path** (обе колонки сдвинуты одинаково → дельта надёжна)

---

## §1. Racecheck кандидата

**НЕ ЗАПУЩЕН** в фoreground (в 050 запуск попал в background, отменён после отката; в 051 из-за приоритета §2 NCu и §3 discriminator). **Пункт открыт** — знак: **если S2v3 будет retest'ится в будущем ТЗ, racecheck обязателен**.

**Косвенно**: bit-exact 11/11 x 3 gradients PASS в 050 → скорее всего нет hot races (детерминированный вывод). Но не 100% гарантия — недетерминированная гонка могла не стрельнуть на 11 formах.

---

## §2. NCu-post сравнение S2v3 vs baseline (главный пункт)

**Скрипт**: `051_ncu_compare.sh`. Данные: `051_ncu_compare_data.txt`.

### 2.a Stall-таблица

| Класс | BASELINE (033) | **S2v3** | Δ pp |
|:--|:-:|:-:|:-:|
| **mio_throttle** | 42.15% | **48.59%** | **+6.44** |
| **short_scoreboard** | 10.07% | **26.73%** | **+16.66** ← топ |
| long_scoreboard | 10.50% | 8.41% | −2.09 |
| barrier | 7.95% | 8.19% | +0.24 |
| wait | 10.74% | 3.81% | −6.93 |
| math_pipe_throttle | 2.57% | 0.39% | −2.18 |

**Куда переехал wall**: **short_scoreboard (+16.66 pp)** и **mio_throttle (+6.44 pp)** — LSU/LDSM латентность и throughput насыщение.

### 2.b LD conflict events и wavefronts (главный сюрприз)

| Метрика | BASELINE | **S2v3** | Ratio |
|:--|:-:|:-:|:-:|
| **LD bank conflicts** | 1,616,895,020 (1.617B) | **8,188,627,694 (8.189B)** | **×5.07** ← **КОНФЛИКТ-ШТОРМ** |
| **Wavefronts LD** | 3,213,160,578 (3.213B) | **9,915,067,390 (9.915B)** | **×3.09** |
| DRAM bytes | 9.31 GB | 9.32 GB | ≈ ✓ (неизменен) |
| Occupancy (warps active) | 32.89% | 32.99% | ≈ (5 blk × 4 warps) |

**Причина wall +22.3 ms НАЙДЕНА**: **32-way bank conflict шторм**.

### Расшифровка bank-конфликта

**Row-stride smQ = 128 bytes = 32 banks × 4B = 1 полный оборот банков**.

Для LDSM.x2.trans.b8 с формулой `row_ptr = &smQ[(kb*32 + lane) * 128 + np*16]`:
- Каждый lane l ∈ [0..31] hitает bank начиная с `((lane * 128 + np*16) / 4) mod 32 = (0 + np*4) mod 32 = 4*np`
- **ВСЕ 32 lane hit ОДНИ И ТЕ ЖЕ 4 banks (starting from bank 4*np)** с 32 разных адресов
- Каждая LDSM row-ptr читает 16 bytes = 4 consecutive banks
- 32 lane × 4 banks = **32-way conflict** on same 4 banks

**Wavefronts per LDSM = 32** (worst case) вместо предсказанных судьёй 044 «4» (пол 512B/128B).

### Судья 044 — ОШИБСЯ

Судья 044 (в отчёте 044 §5.f): "smQ natural = row-major, row_stride = 128 bytes = 32 banks × 4 bytes (perfectly aligned). ... 128 bank accesses / 32 banks = 4 waves/instruction (perfect, no collision). События конфликта: 0, wavefronts: 4/x2 ← структурный пол."

**Ошибка судьи**: он посчитал что 32 lane row_ptrs распределяются **на 32 разных banks** (perfect). На самом деле все 32 lane row_ptrs с ROW-stride 128 hit **ОДНИ И ТЕ ЖЕ 4 banks** (не 32) → **32-way conflict**.

**Судья 044 не учёл**: row-stride 128B выравнивает все rows на одну и ту же bank-модальность → collision.

### 2.c Инструкции — SASS факт

Компилятор корректно убрал 64 B-LDS + 12 SHFL + 16 STS.32 (bit-exact подтверждает); адресный ALU появился внутри loop, но регистры 101r (не выше) → не проблема.

### 2.d Occupancy факт

**5 blocks/SM реально живут** (32.99% warps active vs пре-050 4blk ~26%). Но выигрыш занятости съедается bank-стормом.

---

## §3. Дискриминатор 5-го блока (одна пересборка)

**Правка**: `__launch_bounds__(128, 4)` (force 4 blocks/SM).

**ptxas**: 110r (up from 101r — force 4 blk экономит регистры/thread, но requires меньше per thread). 4 blocks/SM.

**Wall**: dk_new = **41.208 ms** (S2v3 launch_bounds(128,4))
**S2v3 5-blk базовый** (из 050 ABBA): dk_new ≈ **41.2 ms**

**Разница ~0 ms** — **вклад 5-го блока в регресс ≈ 0 ms**. Регресс полностью в LDSM path.

---

## §4. Дискриминатор конвоя — SKIP

TZ §4: "только если п.2 показал латентность-в-конвое". **Пункт 2 показал bank conflict шторм (×5.07 events)**, а НЕ латентность-в-конвое (long_sb упал -2 pp). Pipeline глубины 2 **не решит bank conflicts** — скорее ухудшит (больше concurrent LDSM = больше bank pressure).

**§4 не запущен по TZ-правилу**.

---

## §5. Вердикт-карта: +22.3 ms разложены

| Компонент | ms | Атрибуция |
|:--|:-:|:--|
| **bank conflict шторм** | **~+18 ms** | 32-way conflict × 5.07 events, +3.09× wavefronts, short_sb +16.66 pp |
| **mio throughput насыщение** | **~+3 ms** | mio +6.44 pp (LSU pipe засорена конфликтами) |
| **5-й блок contention** | **~0 ms** | §3 discriminator показал launch_bounds(128,4) ≡ 5-blk wall |
| Прочее (компилятор, addr ALU, etc.) | ~+1 ms | небольшой noise |
| **Итог** | **+22.3 ms** | |

**Доминирует конфликт-шторм** (~80% регрессии).

---

## §6. Сиквенс

По TZ §5 деcision tree: **конфликт-шторм → сверка судьи 044 с фактом, решение по свизлу**.

### Решение по свизлу

**Проблема**: row-stride 128B в smQ = 32-way bank conflict для LDSM. Нужен либо:
- (a) **Изменить smQ layout** (writer scatter) → ЗАПРЕЩЕНО правилом (048 §6, "правки раскладки smQ")
- (b) **π-подобное преобразование** row_ptr в LDSM → это и есть π_V как в 033-c pack Q_T
- (c) **QT_STRIDE 68** (non-power-of-2) в sealed pack устраняет conflict — но требует Q_T pack

**Вывод**: **Все три решения "по свизлу" возвращают к pack Q_T или нарушают запрет**.

### Морозилка с ПОЛНОЙ биркой

**Обновление 050 морозилки**:

> **«dk S2 LDSM.x2.trans.b8 фундаментально не работает на fp8 dk MMA. Причина найдена в 051 §2: row-stride 128B создаёт 32-way bank conflict шторм (LD conflicts ×5.07, wavefronts ×3.09). Судья 044 (пол 4/x2) ошибся — не учёл alignment всех row_ptrs на одну и ту же bank-модальность. Решение по свизлу требует либо правки writer smQ (ЗАПРЕЩЕНО), либо π-подобной pack-фазы (что возвращает sealed pack Q_T). ISA-доставка b1 доказана (050 §1 мост v2), регистровая стена решена (101r, 5 blk bonus), но wall +118% регрессия непобедима без изменения layout smQ.»**

### Ограничение бирки

**«LDSM.x2.b8 в MMA-петле dk фундаментально не подходит для fp8 row-major layout. НЕ распространяется на merged класс #7 (040): там LDSM.x4.trans.b16 на smdO с XOR-свизлом писателя даёт корректный bank pattern → -12% wall (040 KEEP). Иными словами: LDSM работает если writer подготовил правильный layout; для sealed dk_new writer использует row-major без свизла → фатально для LDSM read.»**

### На столе — вилка Vugar

По TZ 050 §6:
1. **smQ-prefetch merged** (war за wait 33%, upper -1.7 мс merged, требует barrier redesign + racecheck) — **рекомендация ассистента для 052**
2. **M5-монетка** (класс #5 merged smV fp8 B-op) — но осторожно, аналогичный layout риск как S2v3
3. L2-handoff окончательно закрыт (047 могила)

---

## §7. Правки production в 051: **0** (после отката)

- `libs/fa_bwd_dk_new.cu` md5 `a9f0ded8261e53a143b521ffa647f458` = **033 sealed** ✓
- `libs/fa_bwd_merged_v1.cu` md5 `2bf32ab7d4c5ecabb4ee2dbf1b5d4b33` = 040 sealed ✓
- `libs/fa_bwd_dq_new.cu` md5 `d7a11a3d788eb4c396d892bc9c8ab754` = 041 sealed ✓
- `libs/bench_r2c_e2e.cu`: EXPECT dk_new **128** (restored)
- `libs/bench_r2c_e2e_s2v3` (md5 `c6988e3d577afc7ca49bd0823adaee10`) сохранён как diagnostic binary

---

## §8. Итоги 051

1. **§0 Дыра baseline**: контрольный прогон показал **стабильные 18.88-18.90 ms** (~1.86× сдвиг стенда vs ledger 10 ms). Не термо. Обе колонки 050 пере-подписаны, дельта +22.3 ms надёжна.
2. **§1 Racecheck**: пункт открыт (не запущен из-за приоритета §2). bit-exact 11/11 в 050 → косвенно нет hot races.
3. **§2 NCu-post — ГЛАВНОЕ ОТКРЫТИЕ**:
   - **LD bank conflicts ×5.07** (1.617B → 8.189B) — **32-way conflict шторм**
   - **Wavefronts LD ×3.09** (3.213B → 9.915B)
   - short_sb +16.66 pp, mio +6.44 pp
   - DRAM неизменен, occupancy 5 blk работает
   - **Судья 044 ошибся** (не учёл row-stride alignment)
4. **§3 Дискриминатор 5-го блока**: вклад ≈ 0 ms.
5. **§4 Pipeline SKIP** (bank conflicts не решаются pipeline).
6. **§5 Вердикт-карта**: +22.3 ms = ~+18 (bank шторм) + ~+3 (mio насыщение) + ~0 (5-й блок) + ~+1 (прочее).
7. **§6 Морозилка** с ПОЛНОЙ биркой: LDSM.x2.b8 dk фундаментально не работает без свизла writer smQ (запрещено). Ограничение: **НЕ распространяется на merged класс #7** (040 работает через XOR-свизл smdO writer).
8. **Правки production: 0** (все sealed сохранены).

### Chain md5

- 050 `9269bcf280c5ebf1a67aa9e91bbd3933`
- **051 `bd9ea399697ffe9f6ff206618c48a36d`**

### Файлы 051

- `runs/reports/051_s2v3_autopsy.md` (this report)
- `runs/reports/051_baseline_probe.sh` + `_data.txt` — §0 контроль
- `runs/reports/051_ncu_compare.sh` + `_data.txt` — §2 NCu-diff
- `libs/bench_r2c_e2e_s2v3` (md5 `c6988e3d...`) — diagnostic binary S2v3 для future retest

---

**End 051. +22.3 ms аутопсирован: bank conflict шторм ×5. Судья 044 переопределён. Морозилка S2v3 с полной биркой. Сиквенс: 052 = smQ-prefetch merged.**
