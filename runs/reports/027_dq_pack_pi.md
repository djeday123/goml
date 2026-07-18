# 027 — dq_new pack+π составная правка: конфликты убиты, но wall bound на LDG → **откат штатный** (правило-2/3 v2)

**Chain**:
- 024_dq_census.md md5: `8462480a45d88dfcbcaabac3599dd2ff`
- 025_dq_pack_paper.md md5: `23425012f92d20f5f3da35267accbc06`
- 026_dq_pack.md md5: `666732460fbc612c6b5bc46e1e3f35fe`

**Artifact header** (после отката):
```
-rw-r--r-- 13352  Jul  5 17:51  libs/fa_bwd_dq_new.cu       (sealed pre-pack, md5 683396f8…)
-rwxr-xr-x         Jul  7        libs/r1c_dq_wall            (56r restored)
-rwxr-xr-x         Jul  7        libs/bench_r2c_e2e          (fingerprint expected 56 restored)
-rw-r--r--         Jul  7        runs/archive/027_post_pack_pi_dq/fa_bwd_dq_new.cu  (md5 fe02e056…)
-rw-r--r--         Jul  7        runs/archive/025_post_pack_dq/                    (md5 8811e45d…)
-rw-r--r--         Jul  7        runs/archive/024_pre_pack_dq/                     (md5 683396f8…)
```

---

## 2a. π-бумага под пост-pack раскладку

### 2a.1 Bank-паттерн новых 16 STS.32 (объяснение +90M конфликтов из 026)

Post-pack Phase D (из 025-b) пишет:
- `base_row = ks*32 + 16*slot_half + 4c + p`
- `col_word = 2*(ni_base+j) + h`

**Битовая карта base_row**: `bit[1:0]=p, bit[3:2]=c, bit[4]=slot_half, bit[6:5]=ks`.

Bank = `(base_row × 17 + col_word) mod 32`. Lane-часть (за fixed wid, slot, j) =
`f(c, p, h) = 4c + 17p + h mod 32` — **точно тот же паттерн, что pre-π dk_new (020)**:
- Bank 0 удваивается на (c=0,p=0,h=0) и (c=3,p=3,h=1)
- Bank 16 не появляется

Это **объясняет +90M новых ST-конфликтов** в 026 (150 vs 60.9).

### 2a.2 Кандидат π: **PI_V** (та же формула, что 023 dk_new KEEP)

```
PI_V(r) = ((r&7)<<2) | (((r>>3)&1)<<1) | ((r>>4)&1) | (r & 0x60)
Bit-perm: r0→2, r1→3, r2→4, r3→1, r4→0, r5,r6 сохраняются
```

Применённая к base_row: физ.строка = `32*ks + 16*c0 + 4p + 2*c1 + slot_half`.

Bank после PI_V (lane-часть) = `16*c0 + 4p + 2*c1 + h mod 32` — **биекция {0..31}**.

## 2b. CPU-судья (метод 021) — все ЗЕЛЁНЫЕ

```
=== ASSERT 1: ST-side bijection (post-pack + PI_V) ===
  PASS: 4·4·4 = 64 STS.32 групп × 32 lanes → 32 distinct banks each ✓

=== ASSERT 2: LD-side bijection (B-load + PI_V) ===
  PASS: KB×NI×2 = 2·16·2 = 64 LDS.32 групп × 32 lanes → 32 distinct banks each ✓

=== ASSERT 3: PI_V row bijection [0..127] ===
  PASS: 128 rows → 128 unique phys rows ✓

=== ASSERT 4: побайтовое покрытие 8192 (invariant of PI_V permutation) ===
  PASS: обе разметки покрывают one and the same 8192 bytes ✓
```

## 2c. P-серии probes

**Пропущены** — PI_V bank-паттерн для dq **bit-identical** к dk_new pack+π_V (та же base_row bit-map, та же f(c,p,h) формула, та же PI_V перестановка). P24/P25 в 023 (dk ST) и P18-P21 в 016 (LD) уже подтвердили **0.00/0.00** для этого паттерна NCu-замером.

Дополнительное GPU-probing без added information → пропуск с явным примечанием.

## 2d. Production правка + гейты

### Правка (одним диффом)

- Восстановлен pack scatter из `runs/archive/025_post_pack_dq/` (md5 сверка `8811e45d…` ✓)
- Добавлен PI_V к **обеим** сторонам одной правкой: Phase D write + B-load read
- Feeder, A-load, epilogue, cp.async — не тронуты

Post-source md5: **`fe02e0567b2341d81126e854cc208a69`** (архив `027_post_pack_pi_dq/`)

### (a) ptxas

- **70r** / 0 spill / 0 stack / 1 barrier
- Прогноз 75-82r (с налогом тесноты 1.5-2× → до 85r) — факт **70r**, **лучше нижней границы прогноза** на 5r
- Потолок 85r для 6 blocks — **под потолком +15r запас**
- **π_V-налог** vs pack-only (72r): **-2r** (compiler compact PI_V macro эффективнее чем ожидалось; урок 022 нижняя граница)

### (b) fingerprint

```
FINGERPRINT kernel_d_precompute    numRegs= 38 (expected  38) OK
FINGERPRINT kernel_merged_v1       numRegs=253 (expected 253) OK
FINGERPRINT kernel_dk_new          numRegs=124 (expected 124) OK    ← unchanged (соседи)
FINGERPRINT kernel_dq_new          numRegs= 70 (expected  70) OK    ← 56→70 обновлено
```
SMEM 13824 B unchanged, 6 blocks/SM ✓. dk_new+merged соседи не тронуты.

### (c) triple bit-exact + CANARY + sanitizer

- **BIT-EXACT 11/11 + CANARY (dK+dV+dQ)** ✓
- **fp16-acc order kt→kb→ni preserved** (правка не тронула MMA loop) ✓
- **Sanitizer 0 errors** ✓

### (d) Wall session-pair 5+5

**Baseline (pre-pack, md5 683396f8…)**:
```
Runs: 8.501 8.507 8.510 8.509 8.513
Median: 8.509 ms (CV 0.05%)
```

**Post pack+π (70r)**:
```
Runs: 8.389 8.384 8.388 8.384 8.384
Sorted:      8.384 8.384 8.384 8.388 8.389
Median: 8.384 ms (CV 0.03%)
```

**Δ vs baseline: -0.125 ms = -1.47%**

**Правило-2/3 v2**:
- ≥3% → 8.253 — не пройден
- 2-3% ABBA → 8.254-8.339 — не в вилке (8.384 > 8.339)
- **<2% → откат** — **8.384 → откат штатный**

### (e) NCu-сверка предсказаний

| Метрика | pre (024) | pack-only (026) | pack+π (027) | Прогноз | Verdict |
|:--|--:|--:|--:|:-:|:-:|
| **LD conflicts** | 541 M | 543 M | **1.95 M** | ~150 M | ✓✓ **побит на 99%** |
| **ST conflicts** | 60.9 M | 150 M | **30.3 M** | <30 M | ✓ **на границе** |
| shared_st inst | 537 M | 134 M | 134 M | unchanged | ✓ |
| **mio_throttle** | 46.73% | 36.38% | **35.47%** | 36-38 | ✓ **в вилке** |
| short_sb | 5.97% | 2.94% | 3.26% | — | ~ |
| barrier | 10.93% | 9.52% | 9.54% | — | ~ |
| **long_sb** | 10.22% | 14.87% | **14.88%** | **~10-11** | ✗ **не вернулся** |
| wait | 9.11% | 11.05% | 10.91% | — | ~ |
| not_selected | 8.94% | 11.28% | 11.52% | — | ~ |

**Прогнозы попали или превышены**: LD, ST, MIO. **long_sb НЕ вернулся** — прогноз мимо.

---

## 3. Диагноз: dq_new НЕ conflict-bound

Ключевое наблюдение:
- LD conflicts **-99.6%** (543M → 1.95M) — **драматично**
- ST conflicts **-80%** (150M → 30M) — сильно
- **Wall drop только -1.47%** — **непропорционально мало** для такого падения конфликтов

**Следствие**: **конфликты не были критическим горлом dq_new**. long_sb 14.88% (LDG cp.async wait) остался, потому что это независимый класс стойлов.

Сравнение с dk_new (для контекста):
- dk_new (022 pre-ABBA): π_V применён, ST conflicts 144M → 17M, wall -2.39% → ABBA → KEEP
- dq_new (027): π_V ту же победу дал, но wall выигрыш втрое меньше

**Новый факт для техлога**: **dq_new — LDG-bound, не MIO+conflict-bound**. Pack+π снимает MIO и conflicts, но **long_sb (LDG cp.async wait) остаётся** — cp.async K/dS сама по себе dominates.

---

## 4. Vugar decision tree (после 027)

- **Не π-цикл больше** (сегодня применён, максимум достижимости)
- **Не pack соло** (026 показал ту же цену)
- **Направление D5 (cp.async двойной буфер)** — было в 024 бумаге, отложено; теперь становится основным кандидатом:
  - long_sb 14.88% — таргет
  - Cost: SMEM 6→4 blocks (Vugar правило "6→4 автоматически регресс") — **требует обсуждения**
- **D4 (barrier reduction)** — барьер 9.52% (не главный, но заметный)
- Альтернатива: **переключение на другое ядро** (dv_mma_p1, merged_v1)

---

## 5. Откат штатный

- `libs/fa_bwd_dq_new.cu` ← `runs/archive/024_pre_pack_dq/` (md5 `683396f8…`)
- `libs/r1c_dq_wall` rebuilt: 56r ✓
- `libs/bench_r2c_e2e` rebuilt: fingerprint 56 restored, chain OK
- **Оба архива сохранены**:
  - `runs/archive/025_post_pack_dq/` (pack-only, md5 `8811e45d…`)
  - `runs/archive/027_post_pack_pi_dq/` (pack+π, md5 `fe02e056…`)
- **Оба варианта — в морозилку**: эффект реален (LD -99.6%, ST -80%), но не преодолевает keep-порог

---

## 6. Вердикт в техлог

- **Pack+π инструктивно работает лучше прогнозов** (LD -99.6% vs -72% прогноз; ptxas 70r vs 75-82 прогноз)
- **Но wall не окупает — dq_new не conflict-bound**
- **Приоритет для следующего шага**: **long_sb (cp.async LDG) — критическое горло** dq_new
- **Rule-2/3 v2 действует**: <2% откат без спорa. Морозилка сохраняет готовые бинари для будущего LDG-переработки.

---

## 7. Файлы

- Prod dq_new: sealed pre-pack (56r)
- Prod dk_new: sealed π_V (124r, 023 KEEP)
- Archive dq: `024_pre_pack_dq/`, `025_post_pack_dq/`, `027_post_pack_pi_dq/`
- CPU-судья: `runs/probes/probe_dq_pi_pack.py`

---

# Приложение — 025-c: фазовый аудит регистров dk_new (0 GPU, paper-only)

## Вопрос
124r по ptxas — интегральный пик. В какой фазе qt-витка он сидит? Если pack-фаза, то MMA-фаза свободнее и retention-B fits.

## Оценка register lifetime по фазам (из fa_bwd_dk_new.cu структуры)

**Sticky (whole kernel)**:
- `dK_acc[NI_DK=16][4]` = **64 fp32** (16 × 4 f32)
- Loop counters + grid indexers + base pointers ≈ 10-15r
- **Sticky base ≈ 75-80r**

**Phase Feeder+Pack peak** (feeder Q load + Phase A/B/C/D + PI_V write):
- Sticky (~78)
- `Qr[4][4]` = 16 uint32 регистров, live до конца Phase D
- Pack temps: t01/t23 (4) + G0-G3 (4) + V0-V3 (4) + u01/u23 (4) + OUT0-3 (4) = 20 регистров peak (при `#pragma unroll` не переиспользуются)
- PI_V compute: 2-3 регистра
- Address compute: 2-3r
- **Feeder+Pack peak ≈ 78 + 16 + 20 + 3 + 3 ≈ 120r** (близко к наблюдаемым 124)

**Phase MMA peak** (kb=0..1 × ni=0..NI_DK):
- Sticky (~78)
- A0-A3 = 4 uint32 (переиспользуется через ni-loop, alive per kb)
- B0/B1 = 2 uint32 (alive per ni)
- PI_V(n_d) compute temp = 2-3r
- Loop iters + addr compute = 3-5r
- **MMA peak ≈ 78 + 4 + 2 + 3 + 5 ≈ 92r** ← **≤ 112 порог**

**Epilogue peak** (unpack fp16 → fp32 + store):
- Sticky (~78) + unpack temps + store addr = ~85-88r

## Вердикт (paper-only)

**MMA-фаза ≈ 92r ≤ 112 → окно retention-B доступно** (по эстимации).

Retention-B skeleton (удержание 4 ni-групп × 2 halves = 8 uint32 B-фрагментов через kb-петлю):
- Добавляет 8 регистров к MMA-фазе: 92 + 8 = **100r**
- Whole-kernel peak = max(pack, MMA_ret) = max(124, 100) = **124r** (unchanged)
- Прогноз: **regs 124r maintained, spill 0, LDL/STL 0, 4 blocks** — окно **живое по эстимации**

**Прогноз соло-эффекта**: 
- Retention снимает ~24 LDS.32 per lane per kt (4 ni × 2 halves × 3 fetches saved) — LDG (cp.async) не меняется
- Wall прогноз: **-1.4..-1.8%** (соло, под порогом keep 3%)
- **Кандидат связки** (retention + другое лекарство) или **смены ландшафта** (после dq_new + другое)

**GPU-подтверждение не запускалось** (paper-only по TZ). Строка в реестр опционов:

> **retention-B dk_new**: окно живое (по эстимации ~100r в MMA vs 112 порог), прогноз соло −1.4..−1.8% (под порогом), статус — **кандидат связки или смены ландшафта**.

---

**End 027.**  
Ожидаю решение Vugar: (i) D5 cp.async двойной буфер, (ii) D4 barrier, (iii) смена ядра, (iv) активация retention-B на dk_new в связке.
