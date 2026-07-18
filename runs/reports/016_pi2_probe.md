# 016 — π_V probe: LD+STS одновременно OK, но conflicts вне критпути dk_new (2026-07-06)

## Статус: КАЛИБРОВКА π_V УСПЕШНА (LD 0.00 + STS 0.00). Правка сделана → wall выигрыш 1.57% < 3% порог → ОТКАТ.

## ARTIFACT-HEADER

### Кросс-верификация
```
md5 015_pi_probe.md:   e0f4c0e7ba4222b803ff7356e102875d  (9382 bytes)
md5 014_probe_ext.md:  0ab333e1988be05f21a0a4882fd2e58f
```

### ls -la runs/probes/ + reports/
```
Probes:
-rw-r--r--  runs/probes/fa_probe_bank.cu             (24 template patterns теперь, P1-P23)
-rw-r--r--  runs/probes/016_pi_v_probe.sh
-rwxr-xr-x  runs/probes/fa_probe_bank                 (rebuilt with P18-P23)

Reports:
-rw-r--r--  runs/reports/016_pi2_probe.md
-rw-r--r--  runs/reports/016_ncu_dk_post_fix.sh
```

Production `libs/fa_bwd_dk_new.cu` — **правка сделана и откатана** (git-clean baseline восстановлен, 11/11 bit-exact ре-верифицирован после отката).

## 016-a. Формула π_V (из TZ, дословно)

```
π_V(r) = ((r & 7) << 2) | (((r >> 3) & 1) << 1) | ((r >> 4) & 1) | (r & 0x60)
```

### Бит-перестановка
- r = r6 r5 r4 r3 r2 r1 r0 (7 бит)
- π_V(r) = r6 r5 r2 r1 r0 r3 r4

### CPU-assert биекции (/tmp/pi_v_assert.py)
```
pi_V bijection {0..127} -> {0..127}: OK (128 unique values, range 0..127)
```

## 016-b. Бумага ДО прогона (paper предсказания)

### LD side: bank starts group ni (для 17*π_V(8*ni+l) mod 32, sorted)
| ni | Sorted starts | Min diff | Verdict |
|:--:|:--|:-:|:-:|
| 0 | {0, 4, 8, 12, 16, 20, 24, 28} | 4 | ✓ perfect 1-way |
| 1 | {2, 6, 10, 14, 18, 22, 26, 30} | 4 | ✓ |
| 2 | {1, 5, 9, 13, 17, 21, 25, 29} | 4 | ✓ |
| 3 | {3, 7, 11, 15, 19, 23, 27, 31} | 4 | ✓ |
| 4-15 | цикл mod 4 (0,1,2,3) | 4 | ✓ ∀ ni |

### STS side (для 2 репрезентативных вариантов)
| Group | Occupied banks | Verdict |
|-------|:--------------|:-:|
| (ks=0, bt=0, hi=0) | {0, 1, 2, 3, 16, 17, 18, 19} | ✓ 8 distinct |
| (ks=0, bt=0, hi=1) | {1, 2, 3, 4, 17, 18, 19, 20} | ✓ 8 distinct |

**Числа совпали с TZ-ожиданием {шаг 4} и {+0..3, +16..19}** → прогон согласно предавторизации.

## 016-c. Измерения P18-P23 (N=1M iters × 4 warps = 4M base inst)

| Pattern | Тип | ni/hi | LD conflicts | ST conflicts | LD inst | ST inst | events/inst | Prediction | Verdict |
|---------|:---:|:-----:|:------------:|:------------:|:-------:|:-------:|:-----------:|:----------:|:-------:|
| **P18** | LD B @68 + π_V | ni=0 | 0 | 0 | 4M | 128 | **0.00 LD** | 0.00 | ✅ |
| **P19** | LD B @68 + π_V | ni=1 | 0 | 0 | 4M | 128 | **0.00 LD** | 0.00 | ✅ |
| **P20** | LD B @68 + π_V | ni=2 | 0 | 0 | 4M | 128 | **0.00 LD** | 0.00 | ✅ |
| **P21** | LD B @68 + π_V | ni=3 | 0 | 0 | 4M | 128 | **0.00 LD** | 0.00 | ✅ |
| **P22** | STS.U8 @68 + π_V | k_lo, m_lo | 0 | 0 | 4 | 4M+128 | **0.00 ST** | 0.00 | ✅ |
| **P23** | STS.U8 @68 + π_V | k_hi, m_hi | 0 | 0 | 4 | 4M+128 | **0.00 ST** | 0.00 | ✅ |

**Все 6 паттернов дали 0.00. Vugar-tree ветка "все 0.00 → правка" активирована.**

## 016-d. Production-правка (сделана и откатана)

### Применённые изменения (в libs/fa_bwd_dk_new.cu)
```c
#define PI_V(r) ((((r) & 7) << 2) | ((((r) >> 3) & 1) << 1) | (((r) >> 4) & 1) | ((r) & 0x60))

// Q_T scatter (writes):
smQ_T[PI_V(k_lo_base + bt) * QT_STRIDE + m_lo_q] = ...   // было (k_lo_base + bt)
smQ_T[PI_V(k_lo_base + bt) * QT_STRIDE + m_hi_q] = ...
smQ_T[PI_V(k_hi_base + bt) * QT_STRIDE + m_lo_q] = ...
smQ_T[PI_V(k_hi_base + bt) * QT_STRIDE + m_hi_q] = ...

// MMA B-load (reads):
uint32_t B0 = *(uint32_t*)&smQ_T[PI_V(n_d) * QT_STRIDE + k_i_lo];   // было n_d *
uint32_t B1 = *(uint32_t*)&smQ_T[PI_V(n_d) * QT_STRIDE + k_i_hi];
```

Обе стороны одной правкой. Layout-only (byte content preserved).

### Гейты (все PASS до отката)
| Gate | Result |
|------|:-------|
| ptxas | **96r/0s/1 barrier** ✓ (unchanged from baseline) |
| Fingerprint | numRegs=96, sharedSizeBytes=0 (all dynamic), maxThreadsPerBlock=640 ✓ |
| SMEM | 20992 B (unchanged, layout-only) ✓ |
| BIT-EXACT dk_new | **11/11 включая CANARY**, max_abs_diff=0.000e+00 ✓ |
| BIT-EXACT dq_new (consumer) | **11/11 включая CANARY** ✓ |
| Sanitizer | **0 errors** (canary_sanitizer.sh) ✓ |

## 016-e. Wall (a)-baseline vs post-fix (одна сессия)

Baseline pre-fix (5-run):
```
9.327 / 9.336 / 9.340 / 9.352 / 9.359 → median 9.340 ms  CV: 0.14%
```

Post-fix π_V (5-run):
```
9.182 / 9.193 / 9.188 / 9.209 / 9.213 → median 9.193 ms  CV: 0.14%
```

**Δ = -0.147 ms = -1.57%** (Vugar-window 8.4-9.0 ms не достигнут; 9.19 чуть выше вилки, но keep-порог wall ≤ 9.3 ✓)

## 016-f. NCu post-fix (детализация конфликтов)

| Metric | Pre-fix baseline | Post-fix π_V | Δ | Predict |
|--------|:----------------:|:------------:|:-:|:-------:|
| **LD conflicts** | 1,694,738,730 | **1,150,885,338** | **-543.8M (-32.1%)** | 1.10-1.25B ✓ **попадание** |
| ST conflicts | 30,883,293 | 32,736,464 | +1.85M (+6%) | ~30M unchanged ✓ |
| LD wavefronts | 2,432,936,234 | 1,889,082,842 | -543.9M | — |
| ST wavefronts | 567,754,205 | 569,607,376 | ~unchanged | — |
| shared_ld inst | 889,192,448 | 889,192,448 | **unchanged** (same code path) | — |
| shared_st inst | 570,425,344 | 570,425,344 | unchanged | — |
| **mio_throttle** | 48.69% | **48.10%** | -0.59 pp | expected drop |
| long_scoreboard | 10.20% | 10.37% | +0.17 pp | — |
| short_scoreboard | 8.33% | 8.31% | -0.02 pp | — |

### Атрибуция: what worked, what didn't
- **B-класс устранён точно**: paper 011 сказал 537M events из B-load; measured drop 543.8M (~совпало)
- **A-класс не тронут**: paper 201M events (4-way на dS_T @Br=64) — π_V не влияет на A path
- **Unattributed 951M остался unchanged** (1.15B post - 201M A ≈ 949M — pre-fix значение) → **unattributed НЕ π-sensitive** (paper prediction "< 1.05B если pi-чувствителен" не пробит; 1.15B > 1.05B)
- **ST conflicts не выросли** — контраст с stride-80 fix (там бы добавилось 537M ST)
- **mio_throttle почти unchanged** (48.69 → 48.10, -0.6 pp) → **MIO = raw instruction count, не conflicts**

## 016-g. Vugar keep-порог → ОТКАТ

```
Keep-порог: wall <= 9.3 И bit-exact; выигрыш < 3% -> откат с формулировкой
"конфликты вне критпути dk_new, mio = raw inst count" -- это не поражение,
а перенацеливание на pack (минус инструкции, не минус конфликты).
```

- Wall 9.193 <= 9.3 ✓
- BIT-EXACT ✓
- **Выигрыш 1.57% < 3% → ОТКАТ**

**Правка откатана**, dk_new вернулся в baseline (QT_STRIDE=68 без π_V, 11/11 bit-exact re-verified post-rollback).

## 016-h. Формулировка вывода (для техлога)

**Конфликты B-класса были вне критического пути dk_new**. π_V сработала математически (paper 011 подтверждён на числах: 537M events устранены точно), но wall не пошёл соответствующе, потому что **MIO throttle сидит на raw instruction count (shared_ld inst 889M unchanged)**, не на конфликтах-wavefronts.

**Перенацеливание на pack**: вместо снижения conflicts (multiplier per inst), нужно снижать **число LDS-инструкций per unit of work** (pack multiple ops in one wider LDS). Это следующая цель после O4 (merged mio).

## 016-i. Named остаток (уточнение для будущего аудита)

Post-fix analysis:
- Paper B-класс = 537M events — устранён точно ✓
- Paper A-класс = 201M events (4-way на dS_T) — не тронут π_V
- Unattributed pre-fix = 951M events
- Post-fix remaining = 1,150M
- Unattributed post-fix = 1,150M - 201M (A) ≈ **949M** ≈ pre-fix 951M

**Unattributed 949M практически invariant vs π_V** → это НЕ B-class variants, а **другой класс событий** (кандидаты: LDGSTS.E.BYPASS.128 внутренние LDS счётчики, compiler ILP loop-body duplication, warmup phase).

## Резюме 016

- ✅ **π_V формула верифицирована**: LD (P18-P21 = 0.00 ∀ ni∈{0..3}) + STS (P22-P23 = 0.00 обе стороны)
- ✅ **Production правка сделана**: ptxas 96r/0s, bit-exact 11/11 × 2 (dk_new + dq_new), sanitizer 0
- ✅ **NCu post-fix**: LD conflicts -32.1% (paper prediction попадание), ST не выросли (контраст с stride-80)
- ⛔ **Wall выигрыш 1.57% < 3% keep-порог** → **ОТКАТ per TZ**
- ✅ **Диагностика подтверждена**: conflicts вне критпути, MIO = raw inst count
- ✅ **Правка чисто откатана**, dk_new baseline восстановлен, 11/11 bit-exact re-verified
- 🎯 **Перенацеливание на pack** (снижение raw LDS inst count вместо conflicts multiplier)

**dk_new остаётся @ QT_STRIDE=68 без π_V. Wall 9.34 ms. R2C E2E остаётся 49.94 ms / 352.30 T.**

Жду Vugar-решение по следующему направлению:
- **Pack** (снижение LDS inst count) — новое ТЗ по O2/O3 в pack-направлении
- **O4 merged mio_throttle 24.56%** — моя изначальная рекомендация
- Обе цели теперь имеют общий урок: минус инструкции, не минус конфликты
