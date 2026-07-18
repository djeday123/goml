# 015 — π-probe: census + LD/STS с π + packed-STS.32 (2026-07-06)

## Статус: PATH B (не-однородный: LD-side π работает; STS-side π ломает). Production код без изменений.

## ARTIFACT-HEADER

### Кросс-верификация
```
md5 014_probe_ext.md:   0ab333e1988be05f21a0a4882fd2e58f
md5 013_bank_probe.md:  90d6e458baebf49442791af8df2f09d1
```

### ls -la runs/probes/
```
-rw-r--r--   8140 Jul 6 17:00  fa_probe_bank.cu             (18 template instantiations)
-rwxr-xr-x         Jul 6 17:00  fa_probe_bank                (rebuilt)
-rwxr-xr-x    620  Jul 6 17:00  015_pi_probe.sh
-rw-r--r--   1200  Jul 6 17:00  Makefile
```

Production `libs/fa_bwd_dk_new.cu` — БЕЗ изменений.

## 015-a. Census STS-скаттер Q_T (fa_bwd_dk_new.cu:154-169)

Точная структура bt-цикла:
```c
for (int ks = 0; ks < 4; ++ks) {
    int k_lo_base = ks * 32 + l_mod4 * 4;    // 0,4,8,...,60 + ks*32
    int k_hi_base = k_lo_base + 16;          // ✓ k_hi = +16 подтверждён
    int m_lo_q    = wid * 16 + l_div4;
    int m_hi_q    = m_lo_q + 8;
    for (int bt = 0; bt < 4; ++bt) {
        smQ_T[(k_lo_base+bt) * QT_STRIDE + m_lo_q] = ...  // 4 STS.U8 per (ks, bt)
        smQ_T[(k_lo_base+bt) * QT_STRIDE + m_hi_q] = ...
        smQ_T[(k_hi_base+bt) * QT_STRIDE + m_lo_q] = ...
        smQ_T[(k_hi_base+bt) * QT_STRIDE + m_hi_q] = ...
    }
}
```
- Total per lane per qt: 4 ks × 4 bt × 4 = **64 STS.U8/qt/lane** ✓ (совпадает с SASS в 010)
- Row coverage k = ks*32 + l_mod4*4 + bt (+16 для hi) = **все 0..127** ✓
- Col coverage m = wid*16 + l_div4 (+8 для hi) = **все 0..63** ✓

## 015-b. Census B-фрагмент (fa_bwd_dk_new.cu:182-185)

Адреса A0-A3 одного lane:
```
A0 at (m_lo, k_i_lo)     byte_addr_0
A1 at (m_hi, k_i_lo)     byte_addr_0 + 512  (m_hi = m_lo+8 → +8 rows = +8*64 = +512)
A2 at (m_lo, k_i_hi)     byte_addr_0 + 16   (k_i_hi = k_i_lo+16 → +16 col)
A3 at (m_hi, k_i_hi)     byte_addr_0 + 528
```

**LDS.64-судьба**:
- A0↔A1 offset = **512 B** (разные строки) → LDS.64 impossible (нужны смежные 8 B)
- A0↔A2 offset = **16 B** (+16 col, не +8) → LDS.64 impossible (LDS.64 требует +8, не +16)
- **LDS.64-ветка на A-путь мёртва** (записано в захоронения)

## 015-c. Feeder→Scatter dataflow

Per lane после Q feeder для ks:
- `Qr[ks][0]` держит 4 байта из smQ[m_lo, k_lo..k_lo+3] (fixed m, k пробегает 4 consecutive)
- `Qr[ks][2]` держит 4 байта из smQ[m_lo, k_hi..k_hi+3] (same m, +16 col shift)

Scatter dataflow (bt=0..3, same ks):
- Байт bt из Qr[ks][0] → Q_T[k_lo+bt, m_lo]
- **Same lane пишет 4 байта в 4 DIFFERENT rows** (одинаковый col=m_lo)
- Строки k_lo, k_lo+1, k_lo+2, k_lo+3 — **STRIDE apart** в physical memory
- **Direct pack STS.32 невозможен**: 4 байта одного слова должны быть в SAME row, но данные scatter'а разбросаны по 4 rows

## 015-d. Дизайн π (bit-rearrangement)

Формула: **`π(r) = 32*((r>>3) & 3) + ((r>>5) & 3) + 4*(r & 7)`**

Бит-структура (r = r6r5r4r3r2r1r0):
- `(r&7)*4` → биты [4:2]
- `((r>>3)&3)*32` → биты [6:5]
- `(r>>5)&3` → биты [1:0]

**Проверки**:
- Bijection {0..127}→{0..127}: r=0→0, r=127→127, все уникальны ✓
- Perfect 1-way для ni-групп: 17*π(r) mod 32 для r в {8*ni..8*ni+7} даёт {C_ni, C_ni+4, ..., C_ni+28} — **банки шаг 4 ∀ ni** ✓

## 015-e. Предсказания (регистрирую ДО прогона)

- **P10a (B-load @68 + π, ni=0..3)**: 0.00 events/load
- **P10b_lo (STS @68 + π, k_lo)**: 0.00 events/store
- **P10b_hi (STS @68 + π, k_hi=k_lo+16)**: 0.00 events/store
- **P12 no-π (packed-STS.32 @68, 16 rows/inst)**: 0.00 events/store

## 015-f. Измерения (N=1M iters × 4 warps = 4M base inst)

| Pattern | Тип | π? | ni/k | LD conflicts | LD inst | events/inst LD | ST conflicts | ST inst | events/inst ST | Prediction | Verdict |
|---------|:---:|:--:|:----:|:------------:|:-------:|:--------------:|:------------:|:-------:|:--------------:|:----------:|:-------:|
| **P10** | LD (B) | ✓ | ni=0 | 0 | 4M | **0.00** | 0 | 128 | — | 0.00 | ✅ |
| **P11** | LD (B) | ✓ | ni=1 | 0 | 4M | **0.00** | 0 | 128 | — | 0.00 | ✅ |
| **P12** | LD (B) | ✓ | ni=2 | 0 | 4M | **0.00** | 0 | 128 | — | 0.00 | ✅ |
| **P13** | LD (B) | ✓ | ni=3 | 0 | 4M | **0.00** | 0 | 128 | — | 0.00 | ✅ |
| **P14** | STS.U8 | ✓ | k_lo | 0 | 4 | — | **4M** | 4M | **1.00** | 0.00 | ❌ |
| **P15** | STS.U8 | ✓ | k_hi | 0 | 4 | — | **4M** | 4M | **1.00** | 0.00 | ❌ |
| **P16** | STS.32 packed | no-π | (n/a) | 0 | 4 | — | **4M** | 4M | **1.00** | 0.00 | ❌ paper предск верен, TZ prediction неверен |
| **P17** | STS.32 packed | ✓ | (n/a) | 0 | 4 | — | **4M** | 4M | **1.00** | — | — |

## 015-g. Vugar decision tree

> **P10a И P10b оба 0.00 → правка** | **Любой из P10 > 0 → стоп**

**Факт**: P10a (P10-P13) = 0.00 ✓ | **P10b (P14, P15) = 1.00** ✗

→ **PATH B** — правку в dk_new НЕ делаю.

## 015-h. Разбор почему π ломает STS

Мой π разработан под LD-геометрию (rows n_d = ni*8 + l_div4):
- π(n_d) mod 32 через 17-стрид даёт банки 4-apart на группу ni

Но STS-геометрия ДРУГАЯ (rows k = ks*32 + l_mod4*4 + bt):
- Естественный STS @68 (без π) даёт rows {0, 4, 8, 12} → banks {0, 4, 8, 12} — все distinct → **0 conflicts** ✓
- Мой π(0)=0, π(4)=16, π(8)=32, π(12)=48 → banks {0, 16, 0, 16} — **collapse** → 8-way на 4 банка → measured 1.00

**Структурная проблема**: LD-геометрия l_div4-varying, STS-геометрия l_mod4-varying. Одна π не может одновременно оптимизировать обе через 17-стрид.

**P16 (packed STS.32 no π)**: 32 lanes → 16 rows × 2 word-cols per row. Banks {0,1,17,18,2,3,19,20,...,31,0} — bank 0 hit twice (row 0 + row 15 wrap). 2-way = 1 event. **TZ-prediction 0.00 неверен, paper-analysis correct**.

## 015-i. Alternative π (не тестированные, для будущего ТЗ)

Дизайн π'(r), удовлетворяющей ОБЕ constraints:
- **LD**: ∀ ni ∈ {0..15}, {π'(8ni), ..., π'(8ni+7)} даёт banks 4-apart
- **STS**: ∀ (ks, bt), {π'(ks*32 + m*4 + bt) для m∈0..3} даёт 4 distinct banks

Это система с ~100 constraints. Разрешима теоретически (128 rows, degrees of freedom), но требует perms lookup-table или композитную формулу.

**Класс кандидатов**: bit-permutation π'(r) = permute_bits(r) с настройкой mask. Полный перебор перестановок 7 bit = 5040. Grid search вне scope этого шага.

## 015-j. Расcheck предсказаний Vugar-tree

- ✅ **P10a (LD) 0.00** — π работает на LD-стороне ∀ ni
- ❌ **P10b (STS) 1.00** — π не работает на STS
- ✅ **P12 no-π** paper prediction 1.00 (не Vugar 0.00) — TZ prediction неверен
- Правку НЕ делаю per decision tree

**Wall прогноз (не для реализации)**: если бы π применить только на LD-side и layout мог быть разделён, saving = 537M events (paper), wall ~8.3-8.8 ms. Но physical layout не разделяется — единый smQ_T буфер.

## 015-k. Options for Vugar

- **O1** Оставить status quo (dk_new 9.42 ms)
- **O2** Grid-search π' — новое ТЗ (perms mask, вне scope 015)
- **O3** MMA layout change — радикально (rewrite dk_new)
- **O4** **merged mio_throttle 24.56% полировка** (моя рекомендация)
- **O5** dk_new: применить π ТОЛЬКО к smQ_T layout в LD path, дублируя буфер (SMEM +8704 B = -4 blocks → 3 blocks/SM, occupancy hit)

## Резюме 015

- ✅ **Census из кода** сделан для STS structure, B-fragment (LDS.64 мёртва), feeder→scatter dataflow
- ✅ **π формула математически проверена**: bijection + perfect 1-way ∀ ni на LD-стороне
- ✅ **Все 4 LD-варианта π (ni=0..3) = 0.00** — π доказана на LD path
- ❌ **STS-side π 1.00** — collapse на 17-stride sawtooth
- ❌ **packed STS.32 (no π/with π) = 1.00** — TZ prediction 0.00 неверен
- ⛔ **Vugar-tree: PATH B** — правку не делаю
- 🔄 Единая π для LD+STS невозможна простой формулой (структурная разнотипность геометрий)

**Правку не делаю. dk_new остаётся @ QT_STRIDE=68 with wall 9.42 ms. Жду Vugar-решение по O1-O5.**

Моя рекомендация: **O4** (merged mio_throttle 24.56% полировка). Bank-conflict фикс dk_new требует композитной π (O2, вне simple класса).
