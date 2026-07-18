# 017 — Pack Q_T-scatter: paper + unit-test skeleton (2026-07-06, add r1 2026-07-06)

## Статус: paper ДО кода. Production не тронут. **Правки r1: (i) SHFL внутри MIO — ярлычная фикс; (ii) PRMT-бюджет ≤90 ALU (факт даст SASS); (iii) полная карта → замещена Vugar spec-хореографией; (iv) baseline-ledger +9.18 (004)**.

## ARTIFACT-HEADER

### Кросс-верификация
```
md5 016_pi2_probe.md:  d6c743580fec8f173c5ae028476d9030
md5 015_pi_probe.md:   e0f4c0e7ba4222b803ff7356e102875d
```

## 017-a. Exchange-сеть ИЗ КОДА (не из модели)

### Recap census 015-c (Q feeder → Qr registers)
`fa_bwd_dk_new.cu:141-152`:
```c
uint32_t Qr[KS_QK][4];  // KS_QK = 4
for (int ks = 0; ks < 4; ++ks) {
    int m_lo = wid * 16 + l_div4;
    int m_hi = m_lo + 8;
    int k_lo = ks * 32 + l_mod4 * 4;
    int k_hi = k_lo + 16;
    Qr[ks][0] = *(uint32_t*)&smQ[m_lo * Hd + k_lo];   // 4 bytes: Q[m_lo, k_lo..k_lo+3]
    Qr[ks][1] = *(uint32_t*)&smQ[m_hi * Hd + k_lo];   // Q[m_hi, k_lo..k_lo+3]
    Qr[ks][2] = *(uint32_t*)&smQ[m_lo * Hd + k_hi];   // Q[m_lo, k_hi..k_hi+3]
    Qr[ks][3] = *(uint32_t*)&smQ[m_hi * Hd + k_hi];   // Q[m_hi, k_hi..k_hi+3]
}
```

Per lane (c=l_mod4, d=l_div4, wid=fixed): 16 uint32 (4 ks × 4 slots) = 64 bytes Q data.

### Scatter formula (154-169, ЦЕЛЬ для pack)
```c
for (ks in 0..4) for (bt in 0..4) {
    smQ_T[(ks*32+c*4+bt)     * QT_STRIDE + wid*16+d]     = (Qr[ks][0] >> (bt*8)) & 0xFF;
    smQ_T[(ks*32+c*4+bt)     * QT_STRIDE + wid*16+d + 8] = (Qr[ks][1] >> (bt*8)) & 0xFF;
    smQ_T[(ks*32+c*4+bt +16) * QT_STRIDE + wid*16+d]     = (Qr[ks][2] >> (bt*8)) & 0xFF;
    smQ_T[(ks*32+c*4+bt +16) * QT_STRIDE + wid*16+d + 8] = (Qr[ks][3] >> (bt*8)) & 0xFF;
}
```
Total per lane: 4 ks × 4 bt × 4 = **64 STS.U8/qt/lane** ✓

### Хореография pack — Vugar spec verbatim (r1)

Обозначения: лейн l, **c = l&3**, **d = l>>2**, **p = d&3** (индекс в кваде), **h = d>>2** (0 = низкий квад / кол-слова +0; 1 = высокий / +4). Квад: лейны `c + 4p′ + 16h`, p′ = 0..3. Слоты **s = 0..3**: s&1 = m-половина, s>>1 = k-половина.

**Фаза A — gather вдоль ks (на слот s, 8 PRMT, селекторы фиксированные)**:
```
t01_lo = prmt(Qr[0][s], Qr[1][s], 0x5140)   // [Q0.b0, Q1.b0, Q0.b1, Q1.b1]
t01_hi = prmt(Qr[0][s], Qr[1][s], 0x7362)   // [Q0.b2, Q1.b2, Q0.b3, Q1.b3]
t23_lo = prmt(Qr[2][s], Qr[3][s], 0x5140)
t23_hi = prmt(Qr[2][s], Qr[3][s], 0x7362)
G0 = prmt(t01_lo, t23_lo, 0x5410)   // [Q0.b0, Q1.b0, Q2.b0, Q3.b0]  → адресату p′=0
G1 = prmt(t01_lo, t23_lo, 0x7632)   //  байт-позиция внутри G = ks
G2 = prmt(t01_hi, t23_hi, 0x5410)
G3 = prmt(t01_hi, t23_hi, 0x7632)
```

**Фаза B — обмен (на слот, 3 SHFL, раунды r = 1..3)**:
```
src(r)   = c + 4*((p − r) & 3) + 16h          // источник
expose(r)= G[(p + r) & 3]                     // что выставляю
V[(p − r) & 3] = shfl.sync.idx(expose(r), src(r), 0xffffffff)
W_in[p]  = G[p]                               // своё остаётся
// корректность: источник i выставляет G[(i_p+r)&3] = G[p_получателя] ✓
```

**Фаза C — приёмное транспонирование (на слот, 8 PRMT, то же дерево, селекторы = Фаза A)**:
```
входы W_in[0..3] (байт-позиция = ks) → OUT[ks] (байт-позиция = p′)
```

**Фаза D — стор (на слот × ks, 1 STS.32)**:
```
row     = ks*32 + 16*(s>>1) + 4c + p
colbase = wid*16 + 8*(s&1) + 4h               // ≡ 0 mod 4, alignment гарантирован
STS.32 smQ_T[row*QT_STRIDE + colbase] = OUT[ks]
```

### Счёт per qt / lane
- Фаза A: 8 PRMT × 4 slots = **32 PRMT**
- Фаза B: 3 SHFL × 4 slots = **12 SHFL** ✓
- Фаза C: 8 PRMT × 4 slots = **32 PRMT**
- Фаза D: 4 slots × 4 ks = **16 STS.32** ✓
- **Итого PRMT (без runtime-selector): 64 + возможный SEL/PRMT overhead в expose(r)**

**Спецификация вариантов runtime-selector expose(r)** (единственная тонкость):
- **(A) fixed-tree + SEL**: G-массив как 4 регистра + двухуровневый SEL по r → 64 PRMT + ~36 SEL
- **(B) direct-gather**: 3 PRMT/раунд с регистровыми precomputed константами от p (12 констант вне qt-петли) → 80 PRMT + 0 SEL
- **Локальный массив G[] с runtime-index ЗАПРЕЩЁН** (LDL/STL → sass-detector)

### Верификация полноты (Vugar-проверка)
- Биекция покрытия: для фиксированных (s, ks) 32 lanes пишут 32 разных слова = 16 rows × 2 col-words per h
- По slots/ks: все rows 0..127 × все col-words покрыты, без дыр/дубликатов
- Байтовые адреса **побитно совпадают** со старыми STS.U8 → инвариант 017-d держится
- Bank pattern инструкции = P16 measured **1.00 events/store**

### PRMT-счёт (детализация)
- **Sender-side gather** (перед SHFL): выбор нужного байта в исходном uint32 → PRMT-выбор × slots × ks. Оценка: **36 PRMT** (per lane per qt).
- **Recipient-side scatter** (после SHFL): вставка полученных байтов в правильные позиции packed uint32 → PRMT-scatter. Оценка: **48 PRMT** (больше, т.к. 4 target positions × 16 STS.32).
- **Итого PRMT: ~84/qt/lane**.
- **Пиковые temp регистры: +8..14** (per-shfl temp uint32 + intermediate packed states).

## 017-b. Inst-census до/после (полный)

### До pack (baseline)
| Op | Count per qt/lane | Пояснение |
|----|:-----------------:|-----------|
| LDS.32 feeder (Q → Qr) | 16 | 4 ks × 4 slots |
| LDS.32 B-load (Q_T reads в MMA) | 64 | KB_DK=2 × NI_DK=16 × 2 |
| LDS.32 A-load (dS_T) | 8 | KB_DK=2 × 4 |
| STS.U8 scatter (Q_T writes) | **64** | 4 ks × 4 bt × 4 |
| **Total MIO-ops** | **152** | |

### После pack (target) — ярлыки исправлены r1
| Op | Count per qt/lane | Пояснение | Δ MIO |
|----|:-----------------:|-----------|:-----:|
| LDS.32 feeder | 16 | unchanged | 0 |
| LDS.32 B-load | 64 | unchanged | 0 |
| LDS.32 A-load | 8 | unchanged | 0 |
| **STS.32 scatter** | **16** | pack 4×4 bytes → uint32 | **-48 vs STS.U8** |
| **SHFL (MIO/shared pipe)** | **+12** | **исполняется через MIO — тот же порт выдачи** | **+12** |
| **Total MIO-ops** | **116** | **152 − 48 + 12 = 116, нетто −36 (−23.7%)** | |

**Правка r1 (SHFL внутри MIO)**: SHFL исполняется через MIO/shared-пайп (тот же порт выдачи). Если бы SHFL был вне MIO, прогноз был бы −31.6% трафика, но mio_throttle-прогноз считан от нетто −36 (−23.7%) и остаётся в силе.

### ALU-цена (правка r1: бюджет vs факт)
- **SHFL: 12/qt/lane** (MIO pipe, не ALU)
- **PRMT/SEL бюджет: ≤ 90 ALU ops/qt/lane** (регистрирую как бюджет; факт даст SASS в юнит-тесте)
  - Вариант (A): 64 PRMT + ~36 SEL (fixed-tree runtime-selector через SEL)
  - Вариант (B): 80 PRMT + 0 SEL (direct-gather с precomputed константами)
  - **Local memory array с runtime-indexing ЗАПРЕЩЁН** (LDL/STL = утечка → sass-detector в юнит-тесте)
- Peak temp регистры: +8..14 (ptxas прогноз)
- **MIO 48.7% → ALU pipe свободен** (не конкурируют)

## 017-c. Банковая бумага STS.32 @68 (без новых проб)

Ссылка на **P16 measured** (014_probe_ext.md):
- packed-STS.32 @68 no π = **1.00 events/store**
- Причина: 16 rows × 2 word-cols per row = 32 targets, при stride 17 sawtooth bank 0 collides с bank 0 (row 0 col_word=0 + row 15 col_word=1, 17*15+1=256 mod 32 = 0)

Прогноз для pack STS.32:
- 16 STS.32/qt/lane × 32 lanes × 4 warps = 2048 STS.32 warp inst / launch × n_qt=128 × grid=16384 blocks / warp = ...
- Actually: warp-level = 16 STS.32/warp/qt = 16 × 128 qt × 65536 warps = **134 M warp-inst**
- × 1.00 events/inst = **~134 M ST conflicts new** (vs current 30.9M baseline)

**Прогноз ST conflicts post-pack: ~130-140 M** (было 30.9M pre-pack). Это ожидаемое повышение (STS.32 packs больше данных за warp inst → 2-way conflict × wide inst).

Но wavefronts: 16 STS.32 wavefronts vs 64 STS.U8 wavefronts (assuming coalesce factor same) — вдвое меньше суммарно. Compute-throughput MIO падает.

## 017-d. Инвариант байтов (проверка по коду)

Проверка smQ_T writers/readers в kernel_dk_new:
```
Writer (единственный): lines 164-167 (STS.U8 scatter, target для правки)
Reader (единственный): lines 190-191 (MMA-B, LDS.32 read at [n_d * QT_STRIDE + k_i])
```

**Нет других писателей / читателей smQ_T**. Feeder читает smQ (не smQ_T). A-load читает smdS_T (не smQ_T). Эпилог пишет dK (не smQ_T).

**π_V НЕ включаем** (морозилка per TZ, откачена в 016). Pack сохраняет ТЕ ЖЕ byte-адреса в smQ_T:
- До: STS.U8 records byte at address `(k * QT_STRIDE + m)` для каждого (k, m) target
- После: STS.32 records 4 bytes at aligned address `(k * QT_STRIDE + m0)` для (k, m0..m0+3) targets
- Byte content at (k, m) identical

**Bit-exact сохраняется автоматически** (layout не меняется, ширина STS меняется).

## 017-e. Предсказания (регистрируются ДО замера)

### ptxas
- **≤118 регистров / 0 spill / 4 blocks/SM** (жёсткий потолок 128 — регресс blocks/SM = откат)
- Peak temp regs: +8..14 vs baseline 96r → **~104-110r** ожидание

### Fingerprint
- **SMEM unchanged**: 20992 dyn / 22016 total (layout не менялся, только STS ширина)
- **4 blocks/SM** (SMEM-limited, unchanged)

### NCu post-pack
| Metric | Baseline | Predicted post-pack | Rationale |
|--------|:--------:|:-------------------:|:----------|
| **shared_ld inst** | 889 M | **unchanged 889 M** | reads не меняются |
| **shared_st inst** | 570 M | **~150 M (-72%)** | 64 STS.U8 → 16 STS.32 (×4 wider each) |
| **LD conflicts** | 1.69 B | **unchanged ~1.69 B** | reads без π |
| **ST conflicts** | 30.9 M | **~130-140 M** | STS.32 pattern P16=1.00 × new inst count |
| **mio_throttle** | 48.69 % | **38-44 %** | -36 ops/qt/lane → ~-24% MIO cycles |
| **long_scoreboard** | 10.20 % | ~10 % | unchanged |
| **short_scoreboard** | 8.33 % | ~8 % | unchanged |
| **Wall dk_new isolated** | 9.340 ms | **7.7-8.5 ms** (центр 8.1) | issue-bound floor ~7.1 |

### Named остаток unattributed 949M
- Прогноз: **unchanged** (был не π-sensitive; если LDGSTS-internals, тоже pack-insensitive)
- **Если 949M → значимо меньше** = новая информация в named-аудит (не blocker)

## 017-f. Дерево (предавторизовано, финальное)

- **wall-выигрыш ≥3% И bit-exact** → keep. NCu-разбор:
  - Conflicts вышли на критпуть? → π-реанимация (LD-сторона π_V переносится, ST — новый вывод под pack)
  - MIO всё ещё топ при упавших instructions? → перенос на merged (staging = те же STS)
- **wall <3%** → **откат + большой стоп**: опровержение вердикта 016 «горло = raw inst count». Ре-атрибуция mio до любых следующих правок.
- **bit-exact fail** → стоп-разбор до зелёного bit-exact в unit-тесте (не подгонять индексы под gpu).

## 017-g. Unit-test план (микро-ядро runs/probes/, production не трогать)

### Дизайн
`runs/probes/pack_qt_unit_test.cu`:
- 4 warps × 32 threads = 128 threads, 1 block
- SMEM: smQ[64*128] = 8192 B init + smQ_T[128*68] = 8704 B pack target
- Init smQ с уникальными маркер-байтами: `smQ[m][k] = (m * 128 + k) & 0xFF` (или `(m*Hd+k) mod 251` для уникальности)
- Q feeder → Qr (mirror production)
- **Pack**: SHFL.bfly + PRMT → 16 uint32 per lane → STS.32 × 16 to smQ_T
- __syncthreads()
- CPU-эталон: enumerate all (k, m) → expected byte
- GPU output smQ_T dumped to host
- **Assert: 8192/8192 valid bytes match** (ignoring padding stride bytes)
- SASS check: exactly 12 SHFL + 16 STS.32 + 0 STS.U8

### Что unit-test валидирует
- Правильность SHFL-выбора маски (mask=1, 2, 3)
- Правильность PRMT-паттернов (gather + scatter индексы)
- Байтовая эквивалентность к STS.U8 scatter

### Что не тестирует unit-test
- Bank conflicts (P16 = 1.00 known, паттерн предсказуем)
- Wall (только production integration покажет)

## 017-h. Порядок работы

1. ✅ **Paper (017)** — этот отчёт
2. ⏳ **Unit-test** (`runs/probes/pack_qt_unit_test.cu`) — byte-assert 256/256 per warp
3. ⏳ **SASS-check** (12 SHFL / 16 STS.32 / 0 STS.U8)
4. ⏳ **Production правка** `libs/fa_bwd_dk_new.cu:154-169` — только scatter, feeder/B-load/A-load не трогать
5. ⏳ **Гейты** (ptxas → fingerprint → tripled bit-exact 11/11 + CANARY + sanitizer + wall + NCu)
6. ⏳ **Отчёт 018_dk_pack.md** с гейтами + NCu-сверка предсказаний

## 017-i. Расшифровка baseline-чисел dk_new (для 018, r1 +9.18)

| Wall | Источник | Контекст |
|------|:--------:|----------|
| **9.18 ms** | 004 isolated до canary-фикса 005a | 5-run pre-ABI padded stride_ds; **cold-cache** |
| **9.42 ms** | 005a canonical wall isolated (Vugar-cert re-cert) | 5-run isolated, после ABI-fix stride_ds |
| **9.598 ms** | 007 R1-E2E cert (in-chain) | Sequential E2E chain thermal-loaded; contention от соседей |
| **9.340 ms** | 016 session-fresh (pre-fix, this session) | 5-run pre π_V |
| **9.193 ms** | 016 π_V post-fix (rolled back) | 1.57% выигрыш, откат |

Baseline для 018 = **свежий 5-run в той же сессии** (правило после ноты 016: не переносить wall между сессиями).

## Резюме 017

- ✅ **Exchange-сеть выведена из кода** (census 015-c)
- ✅ **Sender→recipient→byte-position карта** для представителя (ks=0, bt=0, slot=0)
- ✅ **Счёт 12 SHFL/qt/lane derived** (3 SHFL per ks × 4 ks) — совпадает с TZ
- ✅ **Inst-census 152 → 116 (-23.7%)** verified
- ✅ **PRMT-счёт ~84, temp regs +8..14** (регистр-предсказание ≤118)
- ✅ **Bank paper STS.32 = 1.00** (P16 measured, ST conflicts ~130-140 M ожидаемо)
- ✅ **Байтовая инвариантность** (writers/readers smQ_T проверены — 1 writer @ scatter target, 1 reader @ MMA-B)
- ✅ **π_V морозится** (морозилка per TZ)
- ✅ **Predictions зарегистрированы**: ptxas ≤118r/0s, wall 7.7-8.5 ms (центр 8.1), mio 38-44%, ST conflicts 130-140 M
- ⏳ Unit-test + production edit + гейты — следующим шагом

**Готов к unit-test implementation.** Paper прошёл self-check "count ≤12 SHFL" — 12 совпало.
