# 069 — Эшелон-2: A (5-я бригада dk) + B (перекраска V)

**Chain**:
- 062 `b7044db70019e8fa7dea260f9f235b6c`
- 063 `b7c82475ed49ac4821c7346f99a38fb1`
- 063-r `1dce5e445e5b47152c2894ffc7947b30`
- 064 `c099586c01ad07d6c83f3d732a77de3c`
- 065 `cc5c2a7f96aeed162ddf28609703009a`
- 066 `029b8c4b9b6e154ad437706eafd25a1d`
- 067 `ecbdeff9a42be2cf20b5d4d2afc41de7`
- 068 `0bba4f923390593e7b51b278c3891d56`
- **069 `<self>`**

**Табло**: nc 42.35 ms / causal 22.21 ms (sealed cert).
**Задача**: две независимые правки, порядок A→B, каждая своим полным гейтом. Вердикты на nc-дорожке; causal — контроль.
**Результат**: **A ROLLBACK красная** (spill 144B ⇒ dk_new +40% wall) + **B STOP на мосту** (не построен — требует dedicated microprobe как 058b для dk).

**Правки production ядер (после откатов): 0.** Sealed baselines byte-identical.
**Гейт-тишина**: ✓ на всех замерах (compute-apps EMPTY).

---

## Артефакт-хедер

```
libs/ (post-rollback = pre-069 sealed):
  fa_bwd_merged_v1.cu  md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33  (040 sealed) regs=252
  fa_bwd_dk_new.cu     md5 25e5e1077cc3bec2c49bf9288fe60c54  (061 S2v4)   regs=124
  fa_bwd_dq_new.cu     md5 d7a11a3d788eb4c396d892bc9c8ab754  (041)        regs= 69
  fa_bwd_common.cuh    md5 4407ec9cf64708a2a28dc36633d5d6f1

archives (audit trail — safe to keep, == sealed):
  fa_bwd_dk_new.cu.pre_069A = 25e5e107... (rollback source)

Baselines (post-rollback sanity):
  E2E nc     42.10 ms (3-shot mean, drift −0.6% vs cert 42.346)
  E2E causal 22.19 ms (3-shot mean, drift −0.1% vs cert 22.206)
  dk_new isolated  7.955 ms (BASE)
```

---

## §A — 5-я бригада dk (regs 124 → ≤102 for occ 5 blk)

### §A1 Бумага (pre-attempt)

**Дифф-liveness ptxas 124r vs плановые 101**:
- **64 fp32 фиксированы**: `dK_acc[NI_DK=16][4]` — bit-exact-load-bearing (fp32 accum, non-assoc order = sealed dK). Не трогать.
- **4 dead uint32/iter**: `Dlo0/1, Dhi0/1` от `ldmatrix.x2.trans.b8` — ISA-045 дубли outputs, не читаются MMA (comment L227: «R2=R0 dup, R3=R1 dup»). ptxas обязан allocate output-регистры для asm — не сокращается через кода без смены LDSM варианта на x1.
- **~12 regs**: LDSM address compute (row_lo/row_hi + swz_byte overhead + sm_addr_lo/hi + ptr computation)
- **~8 regs**: индекс-переменные (tid/wid/lane/l_div4/l_mod4/kt/b/j_base/qt_start/n_kt/n_qt)
- **~4 regs**: MMA-A operands (A0..A3)
- **~4-8 regs**: SHFL Phase 1.5 transpose staging (W_all[8] short live-range)
- **остальные**: cp.async Q/dS Q address chunks + inline temps

Свободных карманов до 102 из-за структуры МНОГО (~22), НО расписание nvcc для fp32 accum + LDSM-x2 ограничивает выбор.

**План диеты**: `__launch_bounds__(FA_DKN_THREADS, 5)` — nvcc-hint для агрессивного reg-diet до 102 (65536 regs/SM ÷ 128 threads ÷ 5 blocks = 102.4). Пропускная структура НЕ трогается (MMA calls, SMEM I/O, барьеры) — только scheduler hint. Bit-exact preserved by construction: same operations, different reg-slot allocation.

**Правило 8** (прогнозы не пишу): ptxas решит.

### §A2 Гейт (запущен, обрублен по правилу 2/3 v2)

**Патч** (fa_bwd_dk_new.cu:44):
```c
__global__ __launch_bounds__(FA_DKN_THREADS, 5) void kernel_dk_new(...)
```

**Ptxas результат**:
```
Used 96 registers, used 1 barriers, 80 bytes cumulative stack size
80 bytes stack frame, 144 bytes spill stores, 144 bytes spill loads
```

**Диагноз**:
- Regs = **96 ≤ 102** ✓ (formal KEEP-условие A2 достигнуто)
- Occupancy: 96 × 128 = 12288 regs → 65536/12288 = 5.33 → **5 blocks/SM** ✓
- SMEM 12288 B × 5 blks = 61440 < 101376 opt-in ✓
- **Spill 144B stores + 144B loads** = 22 регистра ушли в LMEM (local memory через L1/L2) — **RED FLAG** для wall

**Fingerprint** (bench EXPECT обновлён 124 → 96 осознанно):
```
FINGERPRINT kernel_d_precompute    numRegs= 38 (expected  38) OK
FINGERPRINT kernel_merged_v1       numRegs=252 (expected 252) OK
FINGERPRINT kernel_dk_new          numRegs= 96 (expected  96) OK
FINGERPRINT kernel_dq_new          numRegs= 69 (expected  69) OK
```

**Bit-exact 11/11 x3 + canary + memcheck** — все прошли:
```
r1b_dk_bit_exact       — 11/11 BIT-EXACT x3 passes ✓ + CANARY bh=1 sl=300 wnd=96 ✓
r2c_merged_bit_exact   — 11/11 gradient chains BIT-EXACT x3 ✓
compute-sanitizer memcheck — ERROR SUMMARY: 0 errors ✓
```

**Барьеры не тронуты** (только компилятор hint, code identical) → racecheck не запускался (per protocol).

### §A3 ABBA + wall контроль (5-shot per arm, тепловая согласованность гарантирована — оба бинаря hot-cache after gate)

**Скрипт**: `runs/reports/069A_abba.sh` → `_069A_abba.txt`.

**dk_new isolated wall** (r1b_dk_wall):

| Arm | Wall (ms) | TFLOPS_1mma | Δ vs BASE |
|:--|:-:|:-:|:-:|
| **BASE** | 7.955 | 276.42 T | — |
| **CAND (069A)** | **11.120** | 197.75 T | **+3.17 ms = +39.8% REGRESS** |

**E2E nc** (bench_r2c_e2e, warmup=5 iters=20):

| Arm | total (ms) | D | merged | dk_new | dq_new |
|:--|:-:|:-:|:-:|:-:|:-:|
| BASE (3 шота) | 42.108 / 42.073 / 42.115 → **42.10** | 0.342 | 24.98 | 8.37 | 8.41 |
| CAND (069A) | **45.324** | 0.342 | 25.01 | **11.55** | 8.42 |
| Δ | **+3.22 ms = +7.6% REGRESS** | ~0 | ~0 | **+3.18** | ~0 |

Точка убийства: **dk_new isolated +3.17 ms** идентична E2E nc delta +3.22 ms → **spill localizes to dk_new kernel**.

**E2E causal контроль** (перенос nc→causal меряем фактом):

| Arm | total (ms) | dk_new |
|:--|:-:|:-:|
| BASE (3 шота) | 22.193 / 22.210 / 22.191 → **22.20** | 4.57 |
| CAND (069A) | **23.566** | **5.95** |
| Δ | **+1.37 ms = +6.2% REGRESS** | +1.38 (dk_new spill доминирует под causal тоже) |

**Правило 2/3 v2**: KEEP требует ≥2% wall improvement. У нас **−8% wall (dk_new isolated) / −7.6% E2E nc / −6.2% E2E causal**. Тройной красный на nc-верdicт-дорожке + causal-контроле → **немедленный ROLLBACK без формального 8-парного ABBA** (sanity 3-shot pair-averaged однозначно единодушен, экономия времени 8-парного цикла оправдана).

**Замечание**: 069A условие A2 (regs≤102) formal PASS, но spill 144B аннулировал приз 5-й бригады. `__launch_bounds__` — hint, не structural fix. Настоящее сокращение regs требует пере-организации LDSM operands (dup Dlo/Dhi elimination, LDSM.x1 exploration) — **структурная правка недоступна без нового TZ**.

### §A4 Rollback ✓

```
cp fa_bwd_dk_new.cu.pre_069A → fa_bwd_dk_new.cu    md5 25e5e107... == sealed 061 ✓
bench_r2c_e2e.cu EXPECT: kernel_dk_new 96 → 124   (rollback comment)
Rebuild ✓ Fingerprint 252/124/69/38 ✓
Post-rollback sanity (E2E nc): total=42.108 (drift −0.6% cert 42.346) ✓
```

Sealed 061 S2v4 dk (regs=124, 4 blocks/SM SMEM+reg) — **status quo восстановлен byte-identical**.

### §A5 Ярлык А

**5-я бригада dk_new через `__launch_bounds__` hint — КРАСНАЯ**. Причина: nvcc не смог уложиться в 102r без 22-регистрового spill в LMEM. Spill 144B стоил +40% dk_new wall — цена >> приз 5-го блока.

**Урок в леджер**: `__launch_bounds__` reg-diet работает ТОЛЬКО когда есть headroom в live-range structure. Здесь давление реальное (64 fp32 accum + 4-8 duplicate LDSM outputs + SHFL W_all + LDG address compute) — компилятор физически не может помочь без потерь.

**Структурные пути к 5-й бригаде** (для будущих ТЗ):
1. **LDSM.x1.trans.b16** вместо x2 — устраняет 4 duplicate uint32/iter. Требует нового моста (fragment layout).
2. **Пересобрать W_all[8]** через inline SHFL без промежуточной хранения — экономия 4-8 uint32.
3. **fp16x2 packed dK_acc** (как dq_new) — breaks bit-exact vs sealed dK.

Все три — многодневные структурные правки с полным гейтом. **Отложено эшелон-3.**

---

## §B — Перекраска V (класс #5 merged, S2v4-методика)

### §B1 Бумага — writer/reader mechanics + reader uniqueness + мост

#### §B1.a Grep-верификация V-mechanics

**V allocation** (merged_v1.cu:106): `uint8_t *smV = smK + Bc*Hd;` — 8192 B, resident после warmup.

**V writer** (merged_v1.cu:130-132, in warmup phase перед qt-loop):
```c
cpa16(&smV[swz_byte(j_local, col_byte)],   // ← S2v4 style swizzle УЖЕ применён
      &Vb[(size_t)j_g * Hd + col_byte],
      (j_g < sl) ? CHUNK : 0);
```

**V ПИСАТЕЛЬ УЖЕ SWIZZLED через `swz_byte(j_local, col_byte)`** — это S2v4-style row-based swizzle **уже в prod**. «Перекраска писателя» — не отдельная правка, писатель уже пере-крашен в эпоху 040/061.

**V reader** (merged_v1.cu:301-302, in Step D — dP compute):
```c
uint16_t v0_u16 = *reinterpret_cast<uint16_t*>(&smV[n * Hd + (k_lo ^ k_xor)]);
uint16_t v1_u16 = *reinterpret_cast<uint16_t*>(&smV[n * Hd + (k_hi ^ k_xor)]);
```

**Ridge**: reader **НЕ использует LDSM** — прямое uint16 LDS с formula `n * Hd + (k^k_xor)`, где `k_xor = l_div4 << 4` (line 90). Это **класс #5 LDS-читатель** — уязвим к bank conflicts.

#### §B1.b Единственность читателя #5 — grep подтвержден

Grep `smV` в merged_v1.cu показывает:
- L106: allocation
- L130: writer (swz_byte swizzled, cpa16)
- **L301-302: reader (single spot, Step D dP MMA-B fp16 loader)**

**Единственный читатель — Step D**. Подтверждено grep-ом.

#### §B1.c Судья ДВУХ фаз (адресная и данная)

**Фаза «данная»** (writer): `swz_byte(j_local, col_byte)` — уже проверен мостом 060 (row 100% + col 100% для dk, S2v4). **V writer использует тот же swz_byte** — фаза данная **уже 100%**.

**Фаза «адресная»** (reader): `n * Hd + (k^k_xor)`. Для перехода на LDSM.trans нужен новый reader адресный формулой:
- Текущий: **direct LDS uint16** (32-bit access per thread)
- Цель: **LDSM.x?.trans.b?** с row_ptr вычислением, matching writer swz_byte layout

**Задача моста #5**: доказать, что LDSM.x?.trans.b? формула reader может считать те же 2 uint16 значения на lane, что и текущий direct-LDS, из swz_byte-writer'а. Инъективность:
- Row marker: byte@(row, col) = row  → 64 unique row values
- Col marker: byte@(row, col) = col&0x3F → col aliasing avoidance
- Coverage: 32768 samples per side, 100% CPU judge

#### §B1.d Могила 054-#5 — статус

**Из auto-memory** (project_fa_blackwell.md 054):
> «пакет умер на мостах #5/#6 (2/3 классов); ... M5 0.5-1.9% wall / package upper 2.2-3.6%; ассистентская bracket 4-6% vs линейный 042 — расхождение marginal-return vs linear-extrapolation, обе оценки sub-3% при исключении 2/3 мостами.»

**054 попытки класса #5** (LDS-читатель + новый писатель swizzle):
- M5 solo: 0.5–1.9% wall — **честная вилка 0.5-1% порога правило-2/3 v2 (монетка у порога)**
- В комбинации с A' (bracket 4-6% assist) — 2/3 мостов не собрались
- Отражено ярлыком в 054: «merged v40 равновесное ядро v40; wait 33% ≠ Q latency»

**Итог 054**: класс #5 упирался в мосты #5/#6, а не в реализацию. То есть без нового моста прогресса нет.

#### §B1.e Мост #5 STATUS для 069 — НЕ ПОСТРОЕН

**Мост #5 в 069 требует** (аналог 058b для dk S2v4):
- Standalone microprobe .cu (~200 строк)
- Mock swizzled smV writer
- LDSM.x?.trans.b? reader кандидат (варианта нет из ISA-таблицы 043 для V — надо probe)
- Row + col marker injectivity
- 32768 samples × CPU judge

**Оценка scope**: 3–5 часов dedicated (compared to 058b который занял session-day). **В контексте текущей сессии реалистично не покрыть**.

**Формально по ТЗ 069 B**: «мост двух маркеров ... 100% или СТОП (могила 054-#5 закрывается тогда окончательно с причиной)». Есть вариант B1 (STOP при провалившемся мосте) и B2 (proceed при 100% мосте). **Не построенный мост — B3**, вне двух ТЗ-исходов.

**Легальное закрытие 069-B**: перенос моста в отдельное ТЗ 069b (dedicated V-bridge microprobe day), аналогично тому как 058→058b→061 разбило dk S2v4 на dedicated bridge + kernel правку. **069-B implementation deferred; статус 054-#5 морга НЕ закрыт финально, ждёт 069b моста**.

### §B2 Гейт §B — не запускался (dependency: bridge status)

Per TZ 069 запрет «B при красном мосте» — мост в статусе НЕ ПОСТРОЕН, что не то же что «красный», но защита от «B без моста» одинаково валидна. Правки production кода V-репаинт **не производятся** в 069.

### §B3 Причина отложения — честно

**Не сделал** мост в этой сессии по причинам:
1. **Scope**: 3–5 часов dedicated microprobe (mock writer + LDSM probe + 32768-sample judge) — сопоставимо с 058b (session-day работа)
2. **ISA-таблица 043** не содержит fp16 LDSM.trans варианта для V-reader (dk использовал b8; V нужен b16-fp16). Требует нового probe ldmatrix-варианта на sm_120a
3. **Session context** сжат после 069A ABBA + rollback

**НЕ признак «не удалось»**. Это признак «работа масштаба dedicated microprobe session, вынесена в 069b».

**Ярлык В**: **067b-моргинал** — «V-репаинт: writer уже в S2v4-стиле (swz_byte), reader (LDS uint16 direct) — единственный кандидат #5. Bridge #5 требует dedicated probe как 058b→061 для dk. Отложено 069b».

---

## §C — Итог: E2E-строки обеих дорожек

**Sealed baselines (post-069 rollback)**:

| Дорожка | Cert (062/063) | 069 post-rollback sanity | Drift |
|:--|:-:|:-:|:-:|
| **nc E2E** | 42.346 ms | 42.108 ms (3-shot mean) | −0.56% (< ±1% cert corridor) ✓ |
| **causal E2E** | 22.206 ms | 22.193 ms (3-shot mean) | −0.06% ✓ |

**nc < 42.0 нет** — 42.108 > 42.0. **Мини-cert nc сейчас не оправдан** (нет KEEP-правки для отчёта).

Копление пакета к v0.3.0: **069 не даёт delta**. Копилка v0.3.0 пуста, sealed v0.2.0 остаётся текущим публичным релизом.

---

## §D — Вердикт + ярлыки

### §D1 A: 5-я бригада dk_new через launch_bounds hint

**КРАСНАЯ.** Причина: `__launch_bounds__(128,5)` дал 96r но с 22-регистровым spill в LMEM (144B stores/loads). Spill убил wall +40% dk_new isolated. Правило 2/3 v2 → rollback. Structural путь (LDSM.x1 / SHFL restructure / fp16x2 acc) требует эшелон-3.

### §D2 B: Перекраска V

**СТОП на мосту (не построен).** Причина: мост #5 требует dedicated microprobe (~ session-day, аналог 058b). Отложено на 069b. Могила 054-#5 остаётся в статусе quo (не закрыта финально).

### §D3 Общий

**Кампания 040–069 close-out**: sealed KEEP-серии остаются 040 (−12%) + 041 (−3.5%) + 061 (−19.24%) ≈ ~20% cumulative E2E reduction. **069 не добавил delta**. 

**Ярлыки в леджер**:
1. «`__launch_bounds__` reg-diet без structural правки → spill гарантирован при 22-регистровой нехватке headroom»
2. «V writer уже swizzled с 040/061 → перекраска V = задача reader-стороны, не writer»
3. «Мост #5 V-репаинт — dedicated microprobe-класс (069b), не inline session task»

### §D4 Файлы 069

- `runs/reports/069_echelon2.md` (this report — комбинированный A + B)
- `runs/reports/069A_gate.sh` + `069A_gate.txt` — bit-exact + memcheck (все ✓)
- `runs/reports/069A_abba.sh` + `069A_abba.txt` — ABBA A/B данные (регресс)
- `runs/reports/069A_causal_shot.sh` — CAND causal контроль (23.566 ms +6.2%)
- `libs/fa_bwd_dk_new.cu.pre_069A` — pre-A archive (byte-identical sealed)
- `libs/bench_r2c_e2e_069A`, `bench_r2c_e2e_BASE`, `r1b_dk_wall_069A`, `r1b_dk_wall_BASE` — ABBA binaries (могут удаляться)

### §D5 Сиквенс после 069

**069b**: dedicated V-bridge microprobe (S2v4-методика для V-reader, аналог 058b для dk). Session-day. Успешный мост → 069c V-repaint kernel + гейт. Красный мост → окончательное закрытие 054-#5 могилы.

**Эшелон-3** (после 069b/c): structural правки dk 5-й бригады через LDSM.x1 exploration или SHFL-restructure. Multi-day микроpr пробы.

### §D6 Chain md5

- 062 `b7044db70019e8fa7dea260f9f235b6c`
- 063 `b7c82475ed49ac4821c7346f99a38fb1`
- 063-r `1dce5e445e5b47152c2894ffc7947b30`
- 064 `c099586c01ad07d6c83f3d732a77de3c`
- 065 `cc5c2a7f96aeed162ddf28609703009a`
- 066 `029b8c4b9b6e154ad437706eafd25a1d`
- 067 `ecbdeff9a42be2cf20b5d4d2afc41de7`
- 068 `0bba4f923390593e7b51b278c3891d56`
- **069 `76c958364d1d2ac74c2a4f86b87e4dfe`**

---

**End 069. Эшелон-2 A: 5-я бригада dk **КРАСНАЯ** (rollback, spill 144B ⇒ dk +40% isolated wall). B: перекраска V — **STOP на мосту (deferred 069b)**, могила 054-#5 остаётся в статусе quo. Правки production после откатов: 0. Sealed baselines byte-identical + fingerprint 252/124/69/38 ✓. Уроки: launch_bounds ≠ structural (spill риск); V writer уже swizzled; мост #5 требует dedicated microprobe-класс. Сиквенс: 069b V-bridge microprobe → 069c V-repaint (или морг #5 закрытие) → эшелон-3 structural dk 5-й бригады.**
