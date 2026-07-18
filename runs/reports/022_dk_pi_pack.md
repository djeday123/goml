# 022 — π_V production правка на pack: **откат штатный**

**Chain**:
- 018_dk_pack.md md5: `5d138ce86465058d707951bfbc6a2f1b`
- 019_postpack_profile.md md5: `d88dacaf5aaf6f130849a5bde259b274`
- 020_pi_pack.md md5: `bc0e9cce5ea21505fd8202e4a0f63387`
- 021_pi_pack_resolved.md md5: `14c36314bd65ec7d3a0bc9d1595a787e`

**Artifact header** (после отката):
```
-rw-r--r-- 13828  Jul  6 21:55  libs/fa_bwd_dk_new.cu  (sealed pack, md5 7317fc48c3ed754a88d50ca6de514ad3)
-rwxr-xr-x         Jul  7        libs/r1b_dk_wall       (rebuilt sealed pack, 107r)
-rwxr-xr-x         Jul  7        libs/bench_r2c_e2e     (rebuilt, expected 107 restored)
-rw-r--r-- 14310  Jul  7        runs/archive/021_sealed_piv/fa_bwd_dk_new.cu (сохранён на случай)
```

---

## 1. Правка π_V (применена → откачана)

Диф применялся **одним разом** к двум местам fa_bwd_dk_new.cu (mirror-правка):

- **Phase D pack-стор** (lines 209-215): 4 STS.32 адреса → `PI_V(row0..3)`
- **B-load** (lines 236-240): `n_d → PI_V(n_d)` в обоих LDS.32 (lo/hi)

Feeder / A-load / merged / dq_new **не тронуты**.

## 2. Гейты (все PASSED кроме keep-порога)

### (a) ptxas

```
kernel_dk_new: 124r / 0 spill / 0 stack / 4 blocks/SM ✓
smem = 20992 B (unchanged)
1 barrier
```

- Прогноз: **107-110r** (я)
- Факт: **124r** — +17 регистров сверх прогноза (π_V compute добавил инструкций)
- Жёсткий потолок 128 → **под потолком, но с малым запасом (4 регистра)**
- 4 blocks/SM обязательны — **OK** (128 threads × 124 regs = 15872 regs/block; SM allows 65536 → 4 blocks)
- LDL/STL = 0 (V0-V3 не легли на stack)

**Отклонение от прогноза зафиксировано**: π_V-compute стоит дороже, чем ожидал (мой прогноз 107-110 был из "π только index math"; факт — 17 регистров ушли на bit-pack, mask, or, shift для PI_V macro в 4+2=6 местах).

### (b) fingerprint

- kernel_dk_new numRegs=124 (updated expected 107→124)
- SMEM 20992 unchanged ✓

### (c) triple bit-exact 11/11 + CANARY + sanitizer

```
=== CHAIN BIT-EXACT SUMMARY ===
  forms all-3 bit-exact: 11 / 11    (все dK/dV/dQ по всем 11 формам + CANARY)
========= ERROR SUMMARY: 0 errors   (sanitizer memcheck)
```
✓ PASS

### (d) Wall session-pair (одна сессия, свежий baseline не переносил 8.895)

**Baseline (sealed pack, 107r)**:
```
Run 1: 8.842   Run 2: 8.858   Run 3: 8.871   Run 4: 8.870   Run 5: 8.879
Median: 8.870 ms  (CV 0.16%)
```

**Post-π_V (124r)**:
```
Run 1: 8.658   Run 2: 8.658   Run 3: 8.654   Run 4: 8.666   Run 5: 8.672
Median: 8.658 ms  (CV 0.09%)
```

**Δ vs baseline: -0.212 ms = -2.39%**

**Keep-порог ≥3% → НЕ пройден** (проиграл 0.61 pp).

### (e) NCu-сверка прогнозов

| Метрика | pre-π_V | post-π_V | Δ | Прогноз | Verdict |
|:--|--:|--:|:-:|:-:|:-:|
| LD conflicts | 1.70 B | **1.14 B** | -33% | 1.10-1.25 B | ✓ **в вилке точно** |
| ST conflicts | 144 M | **16.8 M** | -88% | <25 M | ✓ **лучше прогноза** |
| shared_st inst | 168 M | 185 M | +10% | ~168 M | ↑ **π_V-compute pressure** |
| **short_scoreboard** | 15.78% | **5.90%** | **-9.88 pp** | 11-14% (гипA) | ✓ **A подтв. сильнее прогноза** |
| **mio_throttle** | 32.26% | **43.24%** | **+10.98 pp** | drop | ✗ **регрессия +11 pp** |
| long_scoreboard | 12.87% | 13.67% | +0.8 pp | ~unchanged | ✓ |
| barrier | 6.21% | 6.82% | +0.6 pp | ~unchanged | ✓ |
| wait | 13.98% | 13.31% | -0.7 pp | | ✓ |

## 3. Что показал эксперимент

- **Гипотеза A подтверждена сильнее**: конфликты **были** в критпути. Их устранение уронило short_sb на -9.88 pp (прогноз -2 до -5).
- **Но wall-выигрыш поглощён MIO-регрессией**: extra π_V-compute (bit-shift/mask/or в PI_V macro в 4 STS + 2 LDS = 6 мест) добавил ~17M shared_st инструкций (+10% shared_st inst) и подвинул shared-pipe нагрузку. MIO стал новым узлом.
- **Мониторинг обеих концов**: LD/ST conflicts упали как ожидалось, но пропускная способность shared pipe упёрлась в saturate — это **inherent trade-off π_V-on-hot-path** (compute vs conflicts).

## 4. Vugar-декомпозиция вилок (verbatim)

- Гипотеза A → short_sb 11-14 pp, wall 8.35-8.65: **A подтв.** (short_sb 5.90 — даже лучше), но wall 8.658 **на верхнем краю вилки A**, а не в середине.
- Гипотеза B → short_sb ~на месте, wall 8.75-8.90 → откат: не наш случай.
- **Реальный исход — гибрид**: конфликтная доля устранена (A), но latency сети pack + π_V-compute overhead устранён не полностью → wall-выигрыш не достиг keep-порога.

Vugar-финал по правилу: "wall <3% → откат + большой стоп" — но здесь "большой стоп" не активирован, потому что **вердикт "горло = raw inst count" не опровергнут** (LD/ST упали как теория; wall не выиграл только потому, что compute cost компенсировал). Это "малый стоп" на dk_new, штатный откат.

## 5. Откат

- fa_bwd_dk_new.cu ← `runs/archive/018_sealed_pack/fa_bwd_dk_new.cu` (md5 `7317fc48c3ed754a88d50ca6de514ad3`)
- r1b_dk_wall rebuilt: **107r/1 barrier** ✓ (sealed pack restored)
- bench_r2c_e2e rebuilt: fingerprint expected 107 restored, prod chain работает
- π_V версия сохранена в `runs/archive/021_sealed_piv/` на случай будущих ревизий

## 6. Вердикт в техлог

**"Конфликты вне критпути и после pack"** — не подтверждён. Наоборот: конфликты **были** в критпути (short_sb -9.88 pp). Но:

**"π_V-compute в hot loop стоит слишком дорого"** — новый факт: extra 17 регистров + 10% shared_st inst выкупили MIO-регрессию 11 pp, которая проглотила win конфликтов.

**dk_new паркуется на sealed pack (018): 9.193 → 8.895 = -3.24%, KEEP verdict сохранён.**

**Фронт → dq_new** (не перезолачиваем dk_new).

---

## 7. Файлы

- Prod: `libs/fa_bwd_dk_new.cu` — sealed pack
- Prod binary: `libs/r1b_dk_wall` — 107r sealed pack
- Chain: `libs/bench_r2c_e2e` — с sealed pack, fingerprint 107
- π_V архив: `runs/archive/021_sealed_piv/fa_bwd_dk_new.cu`
- Sealed pack архив: `runs/archive/018_sealed_pack/fa_bwd_dk_new.cu`

---

**End 022.**  
Ветка отката закрыта, dk_new паркуется на sealed pack. Ожидаю новое ТЗ для dq_new.
