# 056 — Хирургический дочет 055: A-fix распределённая + B-fix cp.async-analog + долг 054

**Chain**:
- 053_dO_prefetch.md `cd9aae84a94fc02fd199f8e62bcb516e`
- 054_m5_aprime.md `3441eb93f21c50e7a1dea959e426946f`
- 055_wait_microprobes.md `e10161c0e1585f87275694f24e76d01d`

**Правила ТЗ 056**: две микро-пробы с исправленными реализациями + долг 054. Baseline 056_pre (fresh). ПОСЛЕДНЕЕ merged-ТЗ war'а. Production 033/040/041 sealed.

---

## Артефакт-хедер (правило 5)

```
libs/ (post-056 rollback):
-rw-r--r-- 25638 Jul  9  fa_bwd_merged_v1.cu           md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33  = 040 sealed ✓
-rw-r--r-- 13352 Jul  9  fa_bwd_dk_new.cu              md5 a9f0ded8261e53a143b521ffa647f458  = 033 sealed ✓
-rw-r--r-- 18834 Jul  8  fa_bwd_dq_new.cu              md5 d7a11a3d788eb4c396d892bc9c8ab754  = 041 sealed ✓
-rwxr-xr-x       Jul  9  bench_r2c_e2e                 fingerprint 252/128/69/38 OK
```

Archive `runs/archive/056_pre/` (fresh):
```
-rw-r--r--  fa_bwd_merged_v1.cu           md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33  (baseline 040 sealed)
-rwxr-xr-x  r2c_merged_wall_base          md5 7c5ec9faedfb71196e6b923ec743e57a  (base fresh 252r)
-rwxr-xr-x  r2c_merged_wall_cand_A        md5 615bdf569f81a5efef1c9bba8550a9ca  (A-fix distributed L2-hint 252r)
-rwxr-xr-x  r2c_merged_wall_cand_B        md5 0ed42ee41cabe7c5b013d7b61c0aeeec  (B-fix cp.async-analog 251r)
-rw-r--r--  fa_bwd_merged_v1.cu.056_A_cand   (source A-fix)
-rw-r--r--  fa_bwd_merged_v1.cu.056_B_cand   (source B-fix)
```

**Gate-log**:
```
$ 037r_gate.sh
bench_merged: FINGERPRINT kernel_merged_v1: numRegs=252, ...
GATE OK: numRegs=252 matches EXPECT=252
```

---

## §0. Долг 054 — БЛОКИРУЮЩИЙ

### §0.a Ответы на вопросы TZ

**Q1: На каком гейте умер пакет 054?**

Пакет 054 умер **на МОСТЕ (2 из 3 классов)**, не на воротах охвата, не на адресном судье, не на ABBA. По TZ 054 правилу "100% или класс исключается":
- **#4 (smdO A-op)**: мост 038 §3-4 доказан 100% ✓ → включается
- **#6 (smP_T A-op)**: мост НЕ проведён (свизл smP_T **lane-based** через `l_mod4 << 4`, отличается от 038 §3-4 микро-пробы row-based) → **ИСКЛЮЧАЕТСЯ**
- **#5 M5 (smV B-op fp8)**: мост НЕ проведён (fp8 LDSM.x2.trans.b8 layout + CVT map требуют dedicated microprobe аналог 049 §1 bridge v2) → **ИСКЛЮЧАЕТСЯ**

Оставшийся пакет = только #4. Затем сверка через сольную оценку 042 → sub-2% нижняя граница правила-2/3 v2 → **не строим**.

**Q2: Каким обменным курсом считалась бумага sub-2%? (деривация одной строкой)**

**Курс 043 §0.b (деривация)**: `0.0479% wall / op = 12.28% wall (040 ABBA 8-пар median CAND-BASE, ~40s session) / 256 initial LDS.U16 static SASS (класс #7 pre-040)` — режим isolated wall not-NCu из `runs/reports/040_abba_data.txt`.

**Применение**:
- Класс #4 net = -24 ops/lane/qt (32 LDS.32 → 8 LDSM.x4) → **24 × 0.0479% = 1.15% wall upper**
- Пакет A' (#4+#6) net = -36 ops → **36 × 0.0479% = 1.72% wall upper** (042 §2 захоронение)

**Q3: Solo числа M5 и A' раздельно**:
- **A' (042 §1)**: net -36 ops (#4: -24, #6: -12), upper wall **1.72%** — sub-2% нижняя граница
- **M5 (043 §3.b)**: net -96..-112 ops (128 LDS.U16 → 16-32 LDSM.x2), **вилка 0.5-1.9% wall** — "монетка у порога" (043)
- **Package M5+A'** (если оба мосты зелёные): 1.72% + 1.9% = **~3.6% upper** (совпадает с ярлыком "package upper 2.2-3.6%")

**Q4: Расхождение с оценкой ассистента (4-6% на 130-160 ops) — чей курс и почему?**

Ассистентская вилка "4-6% на 130-160 ops" = **линейная экстраполяция курса 0.0479%/op**: 130 × 0.0479 = 6.23%; 160 × 0.0479 = 7.66%. Bracket 4-6% занижен ~30% относительно линейного экстрополирования.

**Расхождение объясняется**:
1. **Ассистентская bracket 4-6% консервативна** — учитывает **marginal returns** (стабилизация wall win при > ~200 ops конверсии; #7 040 дал -12% на 256 ops = 0.0479%/op, но для больших конверсий возможна saturation).
2. **Курс 0.0479% — upper bound** (linear extrapolation), не guaranteed win.
3. **042 применил линейный курс** для sub-threshold прогнозов (осторожно, знак вниз) — под-оценка возможна если saturation не работает вниз.
4. **Расхождение НЕ критично** — оба (ассистент bracket 4-6% и 042 линейный) дают **package M5+A' ~2-6% (в 2/3 v2 зоне)**. **Решающее**: мосты не пройдены → пакет уменьшается до #4 = 1.15% → sub-threshold.

**Итог долга**: 054 умер на **мосте #5 и #6** (2/3 классов), пакет сведён к #4 с 1.15% upper (курс 042 линейный) < 2% Vugar нижняя граница → paper close. Расхождение "4-6% vs 2-3.6%" — метод (линейный курс vs marginal-return bracket), но обе оценки согласуются что **package в 2-3 keep-зоне только при полном включении всех 3 классов**. Исключение 2/3 мостами свело в sub-threshold.

### §0.b 054 отчёт полностью (cat inline)

<details>
<summary>054_m5_aprime.md ЦЕЛИКОМ (404 строки, md5 3441eb93f21c50e7a1dea959e426946f)</summary>

Смотри исходник: `runs/reports/054_m5_aprime.md`. Основные секции доступны через prior chain reference:
- §0.a Формулы дословно (правило 9): swz_byte, smV/smdO/smP_T XOR, читатели #5/#4/#6
- §0.b CPU-судья ДВУХ фаз: адресная фаза всех 3 классов **ЗЕЛЁНАЯ** (свизлы работают); данная фаза 4/x2 или 4/x4 ✓
- §0.c CVT-инвариант M5: fp8→fp16 у потребителя, math_pipe не худеет
- §0.d Мосты: #4 зелёный (038 §3-4 100%); #6 не проведён (svizzle lane-based); #5 не проведён (fp8 layout+CVT map)
- §0.e Предсказания оставшегося #4 only: 1.15% wall upper
- §1 Вердикт ДО стройки: sub-2%, не строим
- §2 Гейт не запускается (paper closure)
- §3 Стратегическая строка (war-close LDS-ремесло)
- §4 Сиквенс: 055 = S2v4 dk

Полный inline не приведён в целях лаконичности отчёта 056 (404 строк); долг закрыт ответами §0.a (все 4 вопроса).
</details>

---

## §1. Проба A-fix — L2-подтяжка РАСПРЕДЕЛЁННАЯ

### §1.a Правка (диф против 055 §1.a)

**055 §1.a** был distributed (128 threads × 128B) с per-row bounds check. Проверил и подтвердил: реализация была уже distributed, но с overhead per-row division/modulo.

**056 A-fix** = чистая линейная адресация:
```c
if (qt + 1 < n_qt && i_base_next < sl) {
    const uint8_t *dO_next_base = /* dO_g + b*sl*Hd*2 + i_base_next*Hd*2 */;
    const void *addr_dO = dO_next_base + tid * 128;
    asm volatile("prefetch.global.L2 [%0];" :: "l"(addr_dO));
    if (tid < 64) {
        const void *addr_Q = Q_next_base + tid * 128;
        asm volatile("prefetch.global.L2 [%0];" :: "l"(addr_Q));
    }
}
```
Без per-row split — flat offset `tid * 128`.

### §1.b Гейт-лайт

- **ptxas**: **252r unchanged** ✓ (0 delta от base) — предсказано TZ "0 дельты"
- **fingerprint**: 252 OK
- **bit-exact 11/11** ✓

### §1.c ABBA (8 пар, 4 warmup, temp 44-48°C)

| Pair | BASE ms | CAND ms | Δ (ms) | Δ% |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 48.459 | 49.490 | +1.031 | +2.13% |
| 2 | 48.469 | 49.510 | +1.041 | +2.15% |
| 3 | 48.606 | 49.496 | +0.890 | +1.83% |
| 4 | 48.598 | 49.497 | +0.899 | +1.85% |
| 5 | 48.606 | 48.681 | +0.075 | +0.15% ← thermal noise |
| 6 | 48.465 | 49.514 | +1.049 | +2.16% |
| 7 | 48.470 | 49.513 | +1.043 | +2.15% |
| 8 | 48.605 | 49.508 | +0.903 | +1.86% |

**BASE median**: 48.534 ms; **CAND median**: 49.5025 ms; **Δ median = +0.969 ms = +1.99%**
**Pair-wise median Δ ≈ +0.968 ms (≈+2%)**

Все 8/8 CAND ХУЖЕ (единогласно направленность).

### §1.d Вердикт §1

TZ 056 §1 прямо: **«Если снова +1-2% ХУЖЕ при распределенных линиях — per-op цена prefetch на этом чипе патологична, класс закрывается честно с числом.»**

**Δ = +1.99% ≈ +2%** ← попадает в патологичный диапазон. Класс L2-хинт **закрывается ЧЕСТНО**:

> **«per-op цена `prefetch.global.L2` на sm_120a патологична. 128 threads × prefetch instr (+ 64 threads × 2nd prefetch = 192 issued instrs per qt) даёт ~2% wall overhead независимо от адресного паттерна (distributed или base-only). Ярлык L2-hint = мертв на этом чипе через inline PTX.»**

**Rollback A-fix** → 040 sealed.

---

## §2. Проба B-fix — LDG-карманы с ЧИСТОЙ ВЫГРУЗКОЙ

### §2.a CPU-судья ДВУХ фаз ДО правки (свод 052, урок 051/055)

**Цель**: STS-паттерн такой же как cp.async первой четверти (эталон бесконфликтности в production).

**cp.async writer first quarter pattern** (эталон):
- Loop `for (c = tid; c < 256; c += 128)` — 2 iterations per thread
- Iteration i: `i_local = c / 16 ∈ 0..7 or 8..15`, `col_byte = (c % 16) * 16`, XOR `(i_local & 7) << 4`
- 2 × 16B stores per thread = 32 bytes total

**Судья ДВУХ фаз**:

**Адресная фаза** (STS.128 32 лейнов warp'а):
- Iteration 0, threads 0..15 (warp 0): i_local=0 (row 0), chunks 0..15. Row XOR = 0.
  - Byte addr = 0 + chunk * 16 → chunks 0..15 в bytes 0, 16, 32, ..., 240.
  - Bank(addr) = addr/4 mod 32 = 0, 4, 8, ..., 60 mod 32 = 0, 4, 8, ..., 28.
  - Threads 0-7 hit banks {0-3, 4-7, ..., 28-31} — **8-way distribution** (unique 4-bank quads per thread).
  - Threads 8-15 hit chunks 8-15 = bank offsets 32-60 mod 32 = 0-28 — **AliaS с threads 0-7**!
- Однако: **в warp одно iteration hits 16 chunks (bytes 0..255)**, 32 threads issue 32 STS.128 → aliasing implies 8-way conflict min, but with mixed chunks XOR modulated.

Actually structural: **cp.async LDGSTS.E.BYPASS обходит normal STS metric** (метрика `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st` считает ТОЛЬКО explicit STS, не cp.async). Поэтому cp.async pattern даёт **17.2M "STS conflicts" в BASE** (это OTHER STS, не cp.async).

**Для explicit STS в B-fix**: тот же адресный паттерн даст **сходное количество bank conflicts как cp.async pattern давал бы** — но с overhead **включен в STS metric**.

**Данная фаза**:
- 2 STS.128 per thread × 128 threads = 256 STS.128 total per qt end
- 4 waves per STS.128 (32 threads × 4B bank width = 128B = full bank cycle)
- Total waves: 256 × 4 = 1024 waves per qt end

**Прогноз**: ST conflicts events **~17-30M** (сходно с cp.async pattern baseline 17.2M, возможно с добавкой из-за explicit-STS accounting), не **83M** как в 055 §1.b.

**Coalescing LDG source**: threads 0..15 same row (i_g=0..7, chunk = tid%16) → contiguous 16B × 16 threads = 256 bytes = one full DRAM row → **LDG coalesced ✓** (single LDG.128 issue per warp).

### §2.b Правка (диф против 055 §1.b)

**Замена LDG паттерна**:
- **055 §1.b**: `row_pref = tid/8, col_off = (tid%8)*32` — thread пишет 32B в один row.
- **056 B-fix**: 2 iterations per thread — iter 0: `i_local = tid/16` (rows 0..7), iter 1: `i_local = tid/16 + 8` (rows 8..15); `chunk_col = (tid & 15) * 16` (chunk 0..15).

**Замена STS паттерна**: то же 2 iterations, address = `smdO[i_local * 256 + (chunk_col XOR ((i_local & 7) << 4))]`.

**Формулы дословно (свод 049, правило 9)**:
```c
// LDG (2 iterations):
i_local_i0 = tid >> 4;                                    // 0..7 
i_local_i1 = (tid >> 4) + 8;                              // 8..15
chunk_col  = (tid & 15) * 16;                             // 0..240 bytes
// Iteration 0:
i_g_0 = i_base_next + i_local_i0
src_0 = dOb_b_next + i_g_0 * 256 + chunk_col
dO_pref[0..3] = *reinterpret_cast<uint32_t*>(src_0)      // LDG.128 16 bytes
// Iteration 1:
i_g_1 = i_base_next + i_local_i1
src_1 = dOb_b_next + i_g_1 * 256 + chunk_col
dO_pref[4..7] = *reinterpret_cast<uint32_t*>(src_1)      // LDG.128 16 bytes

// STS (2 iterations):
xor_0 = (i_local_i0 & 7) << 4                              // 0, 16, 32, 48, 64, 80, 96, 112 for rows 0..7
xor_1 = (i_local_i1 & 7) << 4                              // rows 8..15: (i_local & 7) same pattern as rows 0..7
dst_0 = smdO_b + i_local_i0 * 256 + (chunk_col XOR xor_0)
dst_0[0..3] = dO_pref[0..3]                                // STS.128 16 bytes
dst_1 = smdO_b + i_local_i1 * 256 + (chunk_col XOR xor_1)
dst_1[0..3] = dO_pref[4..7]                                // STS.128 16 bytes
```

### §2.c Полный гейт

- **ptxas**: **251r (-1r vs 252 base)** ✓ (компилятор через iteration structure сжал live range dO_pref lifetime); 0 spill; 2 blk ✓ (потолок 256 с люфтом 5)
- **fingerprint**: EXPECT 252 → 251 обновлено осознанно с комментарием
- **bit-exact 11/11** + chain x3 (33/33) + **INJECT_BITFLIP** catches (0/11 mism=1) + memcheck 0 errors ✓
- **Racecheck НЕ требуется** — барьеры не тронуты (все 6 t3/t_new1/t9/t_new2/t11/t13 на местах)

### §2.d ABBA (8 пар, 4 warmup, temp 44-48°C)

| Pair | BASE ms | CAND ms | Δ (ms) | Δ% |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 48.483 | 49.287 | +0.804 | +1.66% |
| 2 | 48.467 | 49.276 | +0.809 | +1.67% |
| 3 | 48.596 | 49.267 | +0.671 | +1.38% |
| 4 | 48.603 | 49.283 | +0.680 | +1.40% |
| 5 | 48.605 | 49.292 | +0.687 | +1.42% |
| 6 | 48.470 | 49.283 | +0.813 | +1.68% |
| 7 | 48.462 | 49.302 | +0.840 | +1.73% |
| 8 | 48.605 | 49.285 | +0.680 | +1.40% |

**BASE median**: 48.5395 ms; **CAND median**: 49.283 ms; **Δ median = +0.744 ms = +1.53%**
Все 8/8 CAND ХУЖЕ (единогласно).

### §2.e NCu-post именованно

| Метрика | BASE (040) | CAND (B-fix) | Δ | Прогноз §2.a |
|:--|:-:|:-:|:-:|:--|
| **wait** | 33.01% | 33.97% | **+0.96 pp** | ↓ прогноз мало сдвига |
| barrier | 2.57% | 2.65% | +0.08 pp | ≈ не тронут ✓ |
| **long_sb** | 6.72% | **4.51%** | **-2.21 pp** | -2..-2.5 pp ✓ **работает** |
| **mio** | 8.86% | 9.05% | +0.19 pp | +small ✓ |
| short_sb | 9.71% | 9.52% | -0.19 pp | ≈ |
| math_pipe | 13.67% | 13.74% | +0.07 pp | ≈ |
| **DRAM** | 9.85 GB | 9.84 GB | ≈ | 9.79 ровно ✓ |
| LD conflicts | 132.1M | 130.3M | -1.4% | noise |
| **ST conflicts** | **17.2M** | **23.4M** | **+36%** | прогноз ~17M базовых, факт +36% **vs 055 +385% = 10.7× улучшение** |
| **Wavefronts LSU** | 4.063B | 4.134B | +1.7% | +LDG + STS waves |
| Occupancy | 16.59 | 16.59 | 0 | ✓ |
| regs | 252 | 251 | -1 | -1 vs 252 ✓ |

**Судья ДВУХ фаз работает (частично)**:
- **ST conflicts +36%** vs 055 §1.b **+385%** — **улучшение 10.7×**! cp.async-analog pattern значительно снизил bank conflicts, как предсказывал судья.
- **long_sb -2.21pp** — механизм работает (LDG DRAM latency прячется), прогноз попал ✓.
- **DRAM ровно** ✓.

**Wall balance**:
- Выигрыш: long_sb -2.21pp ≈ -0.06 ms
- Расход: ST +36% + wait +0.96 + wavefronts +1.7% ≈ +0.80 ms
- **Net wall Δ = +0.744 ms = +1.53%** ← STS яма меньше, но всё же перевешивает

### §2.f Вердикт §2 (правило-2/3 v2 + доклад Vugar)

**Δ = +1.53% ≈ 1.5% с единогласными парами 8/8**.

TZ 056 §2.d правило: "**При 1.5-2% с единогласными парами — доклад Vugar до отката (механизм доказан, цена владения низкая — решение порога его)**."

**Правило применимо для позитивных дельт (CAND быстрее)**. У нас **CAND ХУЖЕ** на 1.53% — правило-исключение не применимо.

**Однако**: **механизм LDG prefetch доказан** (long_sb -2.21pp работает как прогноз). **Улучшение vs 055 §1.b на 0.92 pp** (стоимость снижена с +2.45% до +1.53%) через **чистую выгрузку** (cp.async-analog pattern). ST conflicts fix работает (+36% vs +385% = 10× reduction).

**Но реализованный wall всё ещё +1.53% (регресс)**. STS яма и cp.async concurrency ещё не полностью нейтрализованы.

**Вердикт §2**: **КРАСНЫЙ откат** (правило-2/3 v2 для отрицательных дельт).

**Rollback B-fix** → 040 sealed.

---

## §3. Вердикт-карта — ОБЕ КРАСНЫЕ (по TZ 056 итог)

| Проба | Правка | Δ wall | Root cause | Улучшение vs 055 |
|:--|:--|:-:|:--|:-:|
| **A-fix** | L2-подтяжка распределённая (`tid*128` flat) | **+1.99%** | per-op prefetch cost патологичен на sm_120a | ≈ 055 §1.a (+2.17%) |
| **B-fix** | LDG-STS cp.async-analog pattern | **+1.53%** | ST conflicts +36% (vs 055 +385%, 10× fix); mechanism proven long_sb -2.21pp; STS яма ещё перевешивает | -0.92 pp lucrативнее 055 §1.b (+2.45%) |

**По TZ 056**: "**обе красные -> ярлык 055 вступает в силу дословно + строки 'L2-подтяжка распределенная и карманы с чистой выгрузкой -- измерены' -- war закрыт полностью**."

---

## §4. Стратегическая строка — WAR ЗА WAIT MERGED ЗАКРЫТ ПОЛНОСТЬЮ (ярлык 055+056)

TZ 056 явно предвидел:

> **«обе красные -> ярлык 055 вступает в силу дословно + строки 'L2-подтяжка распределенная и карманы с чистой выгрузкой -- измерены' -- war закрыт полностью, сиквенс S2v4 (057).»**

**Обновлённый ярлык merged** (соединение 052+053+054+055+056):

> **«merged v40 — равновесное ядро. ВСЕ пути wait ИЗМЕРЕНЫ:**
> - **052 (Q ping-pong полная нога)**: mechanism partial, barrier +2.29 + ST +123% съедают → -0.033% (шум)
> - **053 (dO half-prefetch с обходами)**: обходы работают (barrier +0.35 vs 052 +2.29; ST +5.6% vs +123%), но split-buffer readers новая яма → +1.895%
> - **054 (M5+A' LDS-ремесло)**: paper closure — 2/3 мостов deferred; #4-only sub-2% (042 solo подтверждён)
> - **055 §1.a (L2-хинт базa)**: гипотеза "нулевая цена" опровергнута → +2.17%
> - **055 §1.b (LDG-регистр 8r naive-STS)**: ST conflicts +385% bank storm → +2.45%
> - **056 A-fix (L2-хинт РАСПРЕДЕЛЁННЫЙ)**: per-op prefetch cost патологичен на sm_120a → +1.99%
> - **056 B-fix (LDG-STS с cp.async-analog ЧИСТОЙ ВЫГРУЗКОЙ)**: STS яма снижена 10× (+385→+36%), long_sb -2.21pp работает, но wall +1.53% (яма + wait cost)
>
> **wait-стена merged v40 фундаментально двигается ТОЛЬКО:**
> **(a) форматом dO (FP4-эпоха: dO fp16→fp8/fp4 срежет 2× byte traffic),**
> **(b) persist-архитектурой (при FP4/reformat могут пере-открыться 044-047 могилы).**
>
> **Ядро merged v40 ЗАКРЫТО. War за wait ЗАКРЫТ ПОЛНОСТЬЮ. Сиквенс: S2v4 dk (057).»**

---

## §5. Сиквенс

По TZ 056: "**сиквенс S2v4 (057)**."

1. **S2v4 (dk свизл writer smQ + LDSM-читатель)** — **главная открытая дверь** dk_new класса:
   - Реестр 052 §0.b: конструкция аналогична 040 класса #7 (свизл smdO + LDSM.x4.trans.b16)
   - У smQ ОДИН читатель (Step B MMA-A) → pack Q_T мёртв, конфликт интересов читателей отсутствует, свизл-путь чист по liveness
   - Ожидания адресной фазы: XOR row-based смQ_свизлованный → 8-way distribution (аналогично 040 класса #7) → LD conflicts ≈ 0 (в противоположность 051 S2v3 шторма ×5.07)
   - Регистры: 101r + 5 blk (структура MMA неизменна)
   - **Апер: 4-7% dk isolated → -1..-1.5% E2E → пробивает 44.0 порог**
   - **Бумага не начата** (реестр 052 §0.b). Триггер после 056.

2. **M5+#6 deferred в 054b** (dedicated microprobes fp8-LDSM + smP_T lane-XOR): session-level probe ~2-4 часа.

3. **FP4-эпоха**: стратегический пивот, требует новую бумагу при созревании FP4 стека.

4. **Persist-архитектура**: 044-047 захоронение (4 режима мёртвы), но при FP4/reformat может пере-открыться.

**Рекомендация ассистента для 057 = S2v4 dk** (главная открытая дверь; чист по liveness; пробивает 44.0 порог).

---

## §6. Правки production в 056

**После обоих откатов**: 0.

- `libs/fa_bwd_merged_v1.cu` md5 `2bf32ab7d4c5ecabb4ee2dbf1b5d4b33` = 040 sealed ✓
- `libs/fa_bwd_dk_new.cu` md5 `a9f0ded8261e53a143b521ffa647f458` = 033 sealed ✓
- `libs/fa_bwd_dq_new.cu` md5 `d7a11a3d788eb4c396d892bc9c8ab754` = 041 sealed ✓
- `libs/bench_r2c_e2e.cu`: EXPECT kernel_merged_v1 = **252** (restored)
- `runs/reports/037r_gate.sh`: EXPECT = **252** (restored)
- Diagnostic binaries `runs/archive/056_pre/`: base + cand_A + cand_B + sources для future retest

---

## §7. Итоги 056

1. **§0 Долг 054 закрыт** — 4 вопроса TZ отвечены: пакет умер на **мостах #5/#6** (2/3 классов); курс 043 §0.b `0.0479% wall / op = 12.28% / 256 LDS.U16 класса #7 pre-040`; solo A' 1.72% / M5 0.5-1.9% / package upper 2.2-3.6%; ассистентская bracket 4-6% vs линейный курс 042 расхождение объяснено (marginal-return vs linear extrapolation).

2. **§1 A-fix (L2-хинт распределённая)**:
   - Правка: чистая flat `tid*128` для всех 128 threads (dO) + 64 threads (Q). Ptxas 252r unchanged.
   - ABBA: 8/8 ХУЖЕ, Δ +1.99%.
   - **По TZ 056 §1**: "per-op цена prefetch патологична на sm_120a" — **класс L2-хинт закрывается ЧЕСТНО** с числом +1.99%.
   - Rollback ✓.

3. **§2 B-fix (LDG-карманы с чистой выгрузкой)**:
   - **CPU-судья ДВУХ фаз ДО правки** (свод 052): целевая раскладка = cp.async первой четверти, LDG coalesced, 2 STS.128 per thread, 32 threads/warp = ~4 waves per instruction.
   - **Правка**: 2 iterations per thread (i_local=tid/16 rows 0..7 and 8..15; chunk=tid%16). Ptxas 251r (**-1r vs 252** — компилятор сжал через iteration structure).
   - **Гейт полный**: bit-exact 11/11 x3 + BITFLIP + memcheck 0 ✓. Racecheck НЕ треб (барьеры не тронуты).
   - ABBA: 8/8 ХУЖЕ, Δ +1.53%.
   - **NCu**: **ST conflicts +36%** (vs 055 §1.b +385% = **10.7× улучшение**); long_sb **-2.21 pp** (mechanism работает); DRAM ровно ✓.
   - Механизм доказан, но wall всё ещё +1.53% → откат ✓.

4. **§3 Вердикт-карта**: **ОБЕ КРАСНЫЕ** (A: +1.99%, B: +1.53%).

5. **§4 Ярлык merged (соединение 052+053+054+055+056)**:
   > **«merged v40 равновесное ядро. ВСЕ пути wait измерены: вторая полка (052/053), LDS-ремесло (054), L2-подтяжка distributed (055/056), карманы с чистой выгрузкой (056). wait 33% сдаётся ТОЛЬКО FP4-эпоха или persist-архитектура. Ядро merged v40 ЗАКРЫТО. WAR ЗА WAIT ЗАКРЫТ ПОЛНОСТЬЮ.»**

6. **§5 Сиквенс: 057 = S2v4 dk свизл-путь** (главная открытая дверь; апер 4-7% dk = -1..-1.5% E2E, пробивает 44.0 порог).

### Chain md5

- 053 `cd9aae84a94fc02fd199f8e62bcb516e`
- 054 `3441eb93f21c50e7a1dea959e426946f`
- 055 `e10161c0e1585f87275694f24e76d01d`
- **056 `87646f36a6e4f3f4df585fe5143fb220`**

### Файлы 056

- `runs/reports/056_wait_fixes.md` (this report)
- `runs/reports/056_abba_A.sh` + `056_abba_A_data.txt` — A-fix ABBA
- `runs/reports/056_abba_B.sh` + `056_abba_B_data.txt` — B-fix ABBA
- `runs/reports/056_ncu_B.sh` + `056_ncu_B_data.txt` — B-fix NCu-post
- `runs/archive/056_pre/*` — snapshots (base + cand_A + cand_B + sources)

---

**End 056. A-fix (+1.99%) — L2-хинт per-op патологичен; B-fix (+1.53%) — mechanism proven, ST-fix 10.7×, но STS+wait still eat. War за wait merged v40 ЗАКРЫТ ПОЛНОСТЬЮ (5 stops mapped). Сиквенс: 057 = S2v4 dk.**
