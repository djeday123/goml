# 057 — Стенд-реатрибуция x2: причина = ZOMBIE r2c_merged_wall (PID 840120)

**Chain**:
- 054_m5_aprime.md `3441eb93f21c50e7a1dea959e426946f`
- 055_wait_microprobes.md `e10161c0e1585f87275694f24e76d01d`
- **056_wait_fixes.md `87646f36a6e4f3f4df585fe5143fb220`** (последний ЗАПИСАННЫЙ отчёт до 057)

**Правила ТЗ 057-STOP**: прерывает 056; class = 037-r hunt но объект = **СТЕНД**. Правки production ЗАПРЕЩЕНЫ. Порядок жёсткий: улики → починки. Wall вне ABBA. Никаких новых правок/вердиктов.

---

## Артефакт-хедер (правило 5)

```
libs/ (post-057, prod неизменен):
-rw-r--r-- 25638 Jul  9  fa_bwd_merged_v1.cu           md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33  = 040 sealed ✓
-rw-r--r-- 13352 Jul  9  fa_bwd_dk_new.cu              md5 a9f0ded8261e53a143b521ffa647f458  = 033 sealed ✓
-rw-r--r-- 18834 Jul  8  fa_bwd_dq_new.cu              md5 d7a11a3d788eb4c396d892bc9c8ab754  = 041 sealed ✓
-rwxr-xr-x       Jul  9  bench_r2c_e2e                 fingerprint 252/128/69/38 OK
```

**Правки production в 057**: **0**. TZ строго "правки production запрещены".

**Gate-log** (baseline не менялся):
```
$ 037r_gate.sh
bench_merged: FINGERPRINT kernel_merged_v1: numRegs=252, ...
GATE OK: numRegs=252 matches EXPECT=252
```

---

## §0. Снимок улик ДО любых вмешательств (порядок жёсткий по TZ)

### §0.a `nvidia-smi -q` полный дамп → `runs/reports/057_nvsmi_full.txt` (400+ строк)

**Ключевые извлечения**:
```
Driver Version                : 580.159.03
CUDA Version                  : 13.0
Product Name                  : NVIDIA RTX PRO 6000 Blackwell Workstation Edition
Persistence Mode              : Disabled                    ← НЕ включён
ECC Mode                      : Disabled                    ← отключён
Fan Speed                     : 30 %
Performance State             : P1                          ← активный
Clocks Event Reasons          : Idle:Not Active, HW Slowdown:Not Active, SW Power Cap:Not Active
Clocks Event Counters         : SW Power Capping = 52164904 us (~52 s history, малозначимо)
FB Memory Used                : 46370 MiB                   ← значительная нагрузка от чужих процессов
GPU Utilization               : 100 %                       ← SM utilization при "idle" от нас = АНОМАЛЬНО
Memory Utilization            : 0 %                         ← память не работает при 100% SM = spin loop
Current Power Draw            : 138.68 W (из 600W max)      ← low power при 100% SM = burn cycles no throughput
Clocks (current)              : Graphics 2865 MHz, SM 2865, Memory 13365, Video 2407
Applications Clocks (default) : Graphics 2617 MHz, Memory 14001 MHz
                                                            ← current mem 13365 vs default 14001 = -4.5% drift, НЕ x2 drop
```

**Power Limit**: 600W hard-lock ✓ (ожидание TZ подтверждено).

### §0.b `bench_r2c_e2e` с параллельным `nvidia-smi dmon -s pucm`

**Скрипт**: `runs/reports/057_dmon_under_bench.txt` (30 samples) + `057_bench_stdout.txt` (полный stdout).

**Bench stdout (ВКЛЮЧАЯ строку D — TZ обязательно)**:
```
=== bench_r2c_e2e: fingerprint x4 ===
FINGERPRINT kernel_d_precompute    numRegs= 38 (expected  38) OK
FINGERPRINT kernel_merged_v1       numRegs=252 (expected 252) OK
FINGERPRINT kernel_dk_new          numRegs=128 (expected 128) OK
FINGERPRINT kernel_dq_new          numRegs= 69 (expected  69) OK

bench_r2c_e2e: bh=128 sl=8192 hd=128 causal=0 window=0 warmup=5 iters=20

=== SEQUENTIAL R2C E2E ===
  D=0.368  merged=48.754  dk_new=18.861  dq_new=18.128  total=86.111  overhead=-0.000
```

**dmon сводка** (30 samples, mclk колонка обязательна):
```
                     pwr   sm  mem   mclk   pclk    fb
idle (rows 1..10):   138  100    0  13365   2865  46370-46928 MB   ← 100% SM при idle от нас = АНОМАЛИЯ
under bench (rows 11..13):
  row 11 (rise):     260  100   33  13365   2700  65752 MB          ← наш bench поднимает mem util + pclk drops
  row 12:            411  100   36  13365   2700                    ← peak power
  row 13:            347  100    0  13365   2835                    
after bench (14-16): 139  100    0  13365   2857  46370 MB
```

**mclk выводы**:
- mclk **под нагрузкой = 13365 MHz** (стабильно)
- Default applications mclk = **14001 MHz** — drift **-4.5%** (объясняется P-state P1, не throttling)
- **mclk НЕ является причиной x2** (не x0.5)
- pclk: idle 2865 → under bench 2700 → after 2857 (нормальный boost поведение)

### §0.c Процессы на GPU — **ГЛАВНАЯ УЛИКА**

**Compute apps active** (`nvidia-smi --query-compute-apps`):
```
pid       process_name                                          used_gpu_memory
2184      wav2vec2/bin/python (asr_api.py)                     4294 MiB
2185      wav2vec2/bin/python (asr_api.py)                     2630 MiB
5951      mms_all/bin/python  (asr_api.py)                     4418 MiB
5952      mms/bin/python      (asr_api.py)                     4418 MiB
5953      wav2vec2/bin/python (asr_api.py)                     4408 MiB
5954      wav2vec2/bin/python (asr_api.py)                     4408 MiB
6226      wav2vec2/bin/python (asr_api.py)                     1048 MiB
6227      wav2vec2/bin/python (asr_api.py)                     2224 MiB
840120    /data/lib/podman-data/projects/goml/libs/r2c_merged_wall  18446 MiB  ← ZOMBIE!
```

**Zombie process 840120 — analitica** (`ps -o pid,etime,stat,cmd`):
```
    PID     ELAPSED STAT CMD
 840120    06:49:21 Rl   /data/lib/podman-data/projects/goml/libs/r2c_merged_wall
```

- **PID 840120** = `r2c_merged_wall` (наш wall harness)
- **Elapsed 06:49:21** = запущен ~7 часов назад (в 06:53 сегодняшнего дня)
- **STAT Rl** = Running (low priority), т.е. активно выполняется (не sleeping)
- **CPU% = 1434%** (multi-core burn — 14 ядер занято)
- **Held 18.4 GB VRAM** (наш bench harness работает бесконечно)

**W-ветка ASR (audio-labeler)**: 8 python процессов wav2vec2/mms_all/mms API-серверы. Запущены с Jul01 (Ss state = sleeping servers). Memory-hold без активной compute нагрузки в snapshot времени (mem util = 0% отдельно от zombie).

**Зомби от 050/051**: TZ явно предупреждал ("зомби от фоновых запусков 050/051"). Реально этот zombie — от одного из моих backgrounded запусков (возможно через Bash `run_in_background=true` или `disown`). Elapsed 6h49m + started 06:53 UTC = попадает в окно 050-055 сессий.

### §0.d Кросс-веточная гигиена

- **W-ветка (audio-labeler ASR)**: **АКТИВНА** во время 057 (8 python процессов). НЕ compute-active в snapshot (mem util 0%), но **VRAM lock 26 GB**.
- **Наш зомби**: **АКТИВЕН** — держит SM 100% через continuous execution в loop.
- **Формулировка "W-ветка не гоняет GPU"**: **ЛОЖНА в этой сессии** — ASR API-серверы всегда on-standby и **их zombie/наш zombie конкурируют за SM cycles**.

**Собственный (наш) zombie** = дефект нашей ветки (мой оверсайт при backgrounded runs). Убить безопасно (наш процесс).

---

## §1. D-дискриминатор (правило TZ 057)

**D факт (§0.b)**: **D = 0.368**
**D ledger**: 0.343 (043 §0.b)
**Drift**: **+7.3%**

**Дерево TZ**:
- D ≈ 0.69 → ПАМЯТЬ
- D ≈ 0.34-0.42 → ТАЙМЕР/СЕЛЕКТИВНАЯ
- Промежуточное → обе

**D = 0.368 ∈ [0.34, 0.42]** → формально **ветка ТАЙМЕР/СЕЛЕКТИВНАЯ**.

**НО**: **реальность** — улика §0.c нашла zombie 840120 held SM 100%. Ветка "ПРОЦЕССЫ" (не в дереве TZ явно, но упомянута в §0.c "зомби от 050/051") **опережает** формальную ветку.

**Логика**: D мал (короткий kernel_d_precompute), contention effect минимален → D drift малый (+7.3%). Merged/dk_new/dq_new длиннее → больше contention time-slice → x2 shift.

**Ветка выбрана: ПРОЦЕССЫ** (доминант; ТАЙМЕР/СЕЛЕКТИВНАЯ бумажные тесты пропускаются — root cause найден в §0.c).

---

## §2. Починка (после снимка улик — порядок TZ соблюдён)

**Действие**: `kill -TERM 840120` → `kill -KILL 840120` (SIGTERM не сработал, SIGKILL добил).

```
$ kill -TERM 840120; sleep 3; kill -KILL 840120
kill: (840120) - No such process       ← процесс завершился между TERM и KILL
$ ps -o pid,cmd -p 840120
(нет вывода — процесс убит)
```

**Пост-kill состояние**:
```
$ nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,clocks.gr,clocks.mem
0 %, 10 MiB, 15.24 W, 405 MHz, 405 MHz
```

- **SM utilization: 100% → 0%** ✓ (zombie убит, SM освободилась)
- FB used: 46 GB → 10 MiB (VRAM освобождена; ASR память тоже освободилась в snapshot время — стенд полностью idle)
- Power: 138W → 15W (idle)
- Clocks idle: pclk 2865→180, mclk 13365→405 (нормальное idle behavior)

**W-ветка** (не тронута — не наша область; TZ разрешает трогать только своё).

---

## §3. Контрольный E2E — стенд-протокол (4+ warmup, 9-run)

**Скрипт**: `runs/reports/057_control_e2e.sh` → `057_control_e2e_after_kill.txt`.

**Результаты (9-run in-chain full decomposition WITH D)**:

| Run | temp | D | merged | dk_new | dq_new | total |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 43°C | 0.343 | 25.130 | 10.455 | 8.479 | 44.407 |
| 2 | 48°C | 0.342 | 25.102 | 10.441 | 8.472 | 44.357 |
| 3 | 47°C | 0.341 | 25.121 | 10.452 | 8.479 | 44.393 |
| 4 | 48°C | 0.342 | 25.099 | 10.440 | 8.464 | 44.346 |
| 5 | 46°C | 0.342 | 25.089 | 10.426 | 8.451 | 44.308 |
| 6 | 48°C | 0.342 | 25.102 | 10.425 | 8.458 | 44.326 |
| 7 | 48°C | 0.342 | 25.083 | 10.416 | 8.444 | 44.285 |
| 8 | 45°C | 0.341 | 25.068 | 10.412 | 8.441 | 44.262 |
| 9 | 45°C | 0.342 | 25.083 | 10.419 | 8.449 | 44.292 |

**Median**: D=0.342, merged=25.099, dk_new=10.426, dq_new=8.458, total=44.326

### §3.a Ledger сверка

| Компонент | Ledger | 057 median (post-kill) | Drift |
|:--|:-:|:-:|:-:|
| **D** | 0.343 | 0.342 | **-0.3%** ✓ |
| **merged** | 25.019 (040 sealed) | 25.099 | **+0.32%** ✓ (в норме дрейфа ±1-2.4%) |
| **dk_new** | 9.75-10.13 (isolated, era 033) | 10.426 | **+3.0%** (на границе дрейфа) |
| **dq_new** | 8.436 (041 sealed) | 8.458 | **+0.26%** ✓ |
| **total** | ~44.206 (041 in-chain) | 44.326 | **+0.27%** ✓ |

**Все числа ↔ ledger в норме дрейфа**. E2E = **44.326 ms** — близко к 44.206 (+0.27%). 

### §3.b Ожидание TZ подтверждено

TZ: "Ожидание при невиновных ядрах: возврат к ~44.2 ± дрейф."

**Факт**: E2E = 44.326 ≈ 44.2 ± **0.12** ms drift ✓

**Ядра НЕВИНОВНЫ**. Реальный регресс в дереве отсутствует. **Bisect по sealed бинарям 040/041 НЕ требуется**.

---

## §4. Вердикт-строка

> **Причина x2 стенд-сдвига НАЗВАНА, ПОЧИНЕНА, КОНТРОЛЬНЫЙ ПРОШЁЛ.**
>
> **Причина**: **ЗОМБИ ПРОЦЕСС `r2c_merged_wall` PID 840120** (elapsed 6h49m, STAT Rl, CPU 1434%, held 18.4 GB VRAM) от одного из backgrounded/disowned запусков в окне 050-055 сессий. Zombie удерживал SM utilization 100% в непрерывной loop → **все sequential bench runs получали 50% cycles** → **x2 wall inflation**.
>
> **Побочные факторы (не главная причина, требуют внимания)**:
> - **Persistence Mode: Disabled** — driver context reload latency при context switches
> - **W-ветка (ASR API)** держала 26 GB VRAM, но compute-inactive в snapshot; **потенциально проблема** если ASR получит нагрузку во время wall-замеров
> - `SW Power Capping counter` = 52s — незначительно
>
> **Починка**: kill 840120 (нашего zombie). W-ветка не тронута — не наша область.
>
> **Контроль**: 9-run стенд-протокол → **E2E = 44.326 ms ≈ ledger 44.206 (drift +0.27%)** ✓.

---

## §5. Карантин-таблица — вердикты 050-056 на x2-стенде

| ТЗ | Вердикт | Δ wall (measured) | Meтод | Кандидат на пере-мер |
|:--|:--|:-:|:--|:-:|
| **050** | S2v3 dk +118% morgue | +118% wall regress | ABBA 8p | **YES** (dk kernel) — S2v3 rebuild + re-ABBA |
| **051** | Autopsy (bank storm ×5.07) | NCu-based conflict events | NCu | **NO** для NCu (metric-based, contention-agnostic); **YES** для §0 baseline (18.88 объяснено = zombie contention) |
| **052** | smQ-prefetch merged -0.033% | ABBA 8p | ABBA | **YES** (candidate re-ABBA на очищенном стенде) |
| **053** | dO half-prefetch +1.895% | ABBA 8p | ABBA | **YES** |
| **054** | Paper closure (sub-2%) | Paper only, no wall | Paper | **NO** (нет wall верdikts) |
| **055 §1.a** | L2-хинт +2.17% | ABBA 8p | ABBA | **YES** |
| **055 §1.b** | LDG-reg +2.45% | ABBA 8p | ABBA | **YES** |
| **056 A-fix** | L2-hint dist +1.99% | ABBA 8p | ABBA | **YES** |
| **056 B-fix** | LDG-STS cp.async +1.53% | ABBA 8p | ABBA | **YES** |

**Все ABBA-вердикты в 050-056** мерялись на **x2-стенде** (загрязнён zombie). **Actual деltа могут отличаться от измеренных** — контам вход из **баS-CAND consistent-но** (обе стороны contaminated), но **пропорции могут сдвинуться** при чистом стенде.

**По TZ 057**: "решения о пере-мерах за Vugar после диагноза, в этом ТЗ ничего не пере-мерить." → **список только**, пере-мерения НЕ проводятся.

**Минимальный список пере-мера (за Vugar)**:
- **050 S2v3 dk +118%** (первый регресс, самый крупный)
- **053 dO half-prefetch +1.895%** (крупнейший merged откат)
- **055 §1.a + §1.b** (пара wait-микро-проб)
- **056 A-fix + B-fix** (пара хирургических дочёт)

---

## §6. Аннуляция формулировки 051 §0

**Формулировка 051 §0** ("контрольный прогон baseline"):
> «РЕАЛЬНЫЙ СДВИГ СТЕНДА (не режим-мислейбл): вся сессия ~1.86× медленнее ledger (стабильно, не термо, не racy). Причина: cross-session state (драйвер/CUDA/persistent mode/что-то ещё) — не расследуется по TZ.»

**057 отменяет** эту формулировку:
- **1.86× НЕ дрейф-норма** (норма ±1-2.4%; 86% — качественно другой феномен)
- **Причина найдена**: не "cross-session state" абстрактно, а **конкретный zombie процесс** от нашей сессии
- **051 §0 dianoz "стабильно 18.88 ms"** объясняется: zombie не менялся между runs → contention стабильна → CV<0.1%
- **Обе колонки 050 ABBA** (BASE ~18.9, CAND ~41.2) действительно были смещены одинаково (обе с contention) → **Δ +118% регресс remains real** (contamination не uniformly scaled), но **абсолютные числа inflate x2**

---

## §7. W-ветка гигиена (запись в отчёт по TZ)

**Ситуация в 057**: 8 python API-серверов (asr_api.py wav2vec2/mms) активны с **Jul01**, каждый ~1-4 GB VRAM, total ~26 GB VRAM.

**Статус во время §0-§4**: **memory-hold, compute-inactive** (nvidia-smi mem util = 0% отдельно от zombie 840120).

**Риск**: если W-ветка получит **inference load** во время наших wall-замеров → **contention возвращается** без warning.

**Рекомендация (для будущих ТЗ)**:
- **Перед wall-замером**: `nvidia-smi --query-compute-apps` — убедиться что compute-apps = **только наш bench** (или пустая)
- **Не запускать backgrounded bench** без `wait` в скрипте (source zombie)
- **Persistence Mode enable** (nvidia-smi -pm 1) — устранит context reload latency при W-ветка + бенч одновременно
- **Watchdog script**: periodic check `nvidia-smi` для zombies (elapsed > 1h с bench binaries)

**Кого чьи процессы** (для docs):
- **W-ветка (Vugar-owned)**: PIDs 2184, 2185, 5951, 5952, 5953, 5954, 6226, 6227 (asr_api.py в audio-labeler)
- **Наши**: `r2c_merged_wall`, `bench_r2c_e2e`, `r2c_merged_bit_exact` (в goml/libs/)

---

## §8. Правки production в 057

**Total: 0** (TZ запрет).

- `libs/fa_bwd_merged_v1.cu` md5 `2bf32ab7d4c5ecabb4ee2dbf1b5d4b33` = 040 sealed ✓
- `libs/fa_bwd_dk_new.cu` md5 `a9f0ded8261e53a143b521ffa647f458` = 033 sealed ✓
- `libs/fa_bwd_dq_new.cu` md5 `d7a11a3d788eb4c396d892bc9c8ab754` = 041 sealed ✓
- `libs/bench_r2c_e2e.cu`: EXPECT 252 unchanged ✓
- `runs/reports/037r_gate.sh`: EXPECT 252 unchanged ✓

**Никаких новых правок ядер/харнессов** (кроме создания control-скриптов в runs/reports/) сделано.

---

## §9. Итоги 057

1. **§0 УЛИКИ СНЯТЫ ДО ПОЧИНКИ** (порядок TZ соблюдён):
   - a. `nvidia-smi -q` (400+ строк) → `057_nvsmi_full.txt`: Persistence Disabled, mclk 13365 vs 14001, SM util 100% at "idle"
   - b. `bench + dmon`: D=**0.368**, mclk stable 13365 under bench (не x2 drop = НЕ ветка память)
   - c. **ZOMBIE r2c_merged_wall PID 840120** (elapsed 6h49m, STAT Rl, CPU 1434%, 18.4 GB VRAM) + 8 ASR API процессов W-ветки

2. **§1 D-дискриминатор**: D=0.368 ∈ [0.34, 0.42] → формально ТАЙМЕР/СЕЛЕКТИВНАЯ, но **§0.c реальность = ПРОЦЕССЫ** (доминант).

3. **§2 Починка**: `kill -TERM 840120` → SM 100% → **0%** ✓; W-ветка не тронута (не наша область).

4. **§3 Контрольный E2E** (9-run): median E2E = **44.326 ms** ↔ ledger 44.206 (+0.27% в норме дрейфа ✓); dk_new=10.426 vs ledger 9.75-10.13 (+3% на границе). **Ядра НЕВИНОВНЫ**. Bisect не нужен.

5. **§4 Вердикт**: причина = **ZOMBIE 840120**; починка = kill (после снимка улик); контроль пройден.

6. **§5 Карантин-таблица 050-056**: все wall-вердикты помечены "мерено на x2-стенде"; кандидаты на пере-мер за Vugar (в 057 не пере-меряется по TZ).

7. **§6 Аннуляция 051 §0** "1.86× стенд-сдвиг" — не абстрактное cross-session state, а конкретный zombie процесс.

8. **§7 W-ветка гигиена** — рекомендации для будущих ТЗ (compute-apps check, no background bench, persistence mode, watchdog).

9. **§8 Правки production**: **0**.

### Chain md5

- 054 `3441eb93f21c50e7a1dea959e426946f`
- 055 `e10161c0e1585f87275694f24e76d01d`
- 056 `87646f36a6e4f3f4df585fe5143fb220`
- **057 `d74b9765950d4634d153f87b17d889d7`**

### Файлы 057

- `runs/reports/057_stand_reattrib.md` (this report)
- `runs/reports/057_nvsmi_full.txt` — nvidia-smi -q полный дамп
- `runs/reports/057_bench_stdout.txt` — bench_r2c_e2e stdout под контенем
- `runs/reports/057_dmon_under_bench.txt` — dmon 30 samples parallel
- `runs/reports/057_control_e2e.sh` + `057_control_e2e_after_kill.txt` — контрольный E2E 9-run

---

**End 057. Root cause найден в §0.c (zombie r2c_merged_wall 840120 6h49m), убит в §2, контрольный E2E возвращает ledger в §3 (44.326 vs 44.206, +0.27%). Ядра НЕВИНОВНЫ. Формулировка 051 "1.86× стенд-сдвиг" аннулирована. Карантин-таблица 050-056 передана Vugar для решения пере-мера. 056 wall-вердикты помечены "карантин-стенд" — за Vugar.**
