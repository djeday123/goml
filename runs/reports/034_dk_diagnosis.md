# 034 — Доводка dk-цены после 033: диагноз (a/b/c) + рекомендация пробы

**Chain**:
- 033_c_full.md md5: `1d23f27d178a54b827d8205cb6679610`

**Artifact header** (production не тронут):
```
-rw-r--r-- 14667  Jul  7 20:00  libs/fa_bwd_dk_new.cu    (033_sealed, md5 a9f0ded8…)
-rw-r--r-- 21584  Jul  7        libs/fa_bwd_merged_v1.cu (033_sealed, md5 deb3a0e1…)
-rwxr-xr-x         Jul  8        libs/r1b_dk_126          (эксперимент maxrregcount=126)
-rwxr-xr-x         Jul  8        libs/r1b_dk_124          (эксперимент maxrregcount=124→122)
```

Только measurement + paper. Морозилки, архивы, merged — не тронуты.

---

## 1. Контекст

**dk-цена 033 (out of вилки 032-b):**
- **isolated**: 8.842 → 9.750 = **+0.91 ms (+10.3%)**, прогноз был +0.3..+0.8 ms → **выше вилки**
- **in-chain**: 8.842 → 10.126 = **+1.28 ms (+14.5%)**
- **Разрыв in-chain − isolated**: **+0.38 ms** (было ~0.03 в pre-033) — **новый эффект**

Цель: вернуть 0.4-0.8 мс адресной доводкой без отката структуры.

---

## 2. (a) NCu полная stall-таблица dk isolated: post-033 vs pre-033 (023 sealed π_V)

| Метрика | pre-033 (023 sealed π_V) | post-033 (W2) | Δ pp |
|:--|--:|--:|:-:|
| barrier | 6.21% | **7.94%** | **+1.73** (5-й барьер W2) |
| **mio_throttle** | 32.26% | **42.20%** | **+9.94** ← главное |
| long_scoreboard | 12.87% | 10.45% | -2.42 |
| **short_scoreboard** | 15.79% | 10.07% | -5.72 |
| wait | 13.98% | 10.74% | -3.24 |
| not_selected | 5.35% | 6.20% | +0.85 |
| selected | 8.42% | 7.78% | -0.64 |
| math_pipe_throttle | 2.76% | 2.57% | -0.19 |

**Балансный анализ**:
- **MIO +9.94 pp** — доминирующий рост, +≈ 0.5-0.7 мс wall (пропорционально)
- **short_sb -5.72 pp** — освобождение от pack Q_T эффекта (но это меньший класс)
- **long_sb -2.42 pp** — cp.async pattern change (dS_nat load дешевле dS_T load?)
- **barrier +1.73 pp** — цена 5-го __syncthreads() W2, ~0.13-0.17 мс

### 2.1 Instruction breakdown

| Метрика | pre-033 | post-033 | Δ |
|:--|--:|--:|:-:|
| smsp__inst_executed_op_shared_ld | 889 M | **956 M** | **+67 M (+7.5%)** |
| smsp__inst_executed_op_shared_st | 168 M | **235 M** | **+67 M (+40%)** |
| **Total shared ops** | 1057 M | **1191 M** | **+134 M (+12.7%)** |
| LD conflicts | 1.70 B | 1.62 B | -80 M |
| ST conflicts | 144 M | **226 M** | +82 M |

### 2.2 Атрибуция +0.91 ms isolated:

- **Transit MIO** (+134 M shared ops): **≈ +0.6-0.75 ms** ← главная причина  
  - 8 LDS.32 (W2 read) + 8 STS.32 (W2 aliased write) per lane per qt
  - Всего 16 доп shared ops per lane per qt × 128 qt × 32 lanes × 4 warps × 16384 blocks = ≈ 4.3 GOps — pipeline pressure
- **5-й барьер** (+1.73 pp barrier): ≈ +0.13-0.17 ms
- **Остальное** (small stall shifts): ≈ +0.05 ms

**Сумма ≈ 0.8-0.95 мс** — сходится с наблюдением +0.91 мс ✓.

**Диагноз**: **не 128r-фон, не барьер (13-17% цены), а сам transit MIO (65-80% цены)**.

---

## 3. (b) Разрыв in-chain − isolated +0.38 мс

**Из NCu isolated**: L2 hit rate = **66.65%**

**Гипотеза числом**: 
- Isolated: cache-cold cycles от K/dS re-loads → некоторые L2 misses идут в DRAM, добавляют latency
- In-chain: **merged только что нагрузил L2 своим dS_nat write** (~9.79 GB) → часть dS_nat уже в L2 когда dk идёт, но L2 evictions от merged perturb dk's cache lines Q/K
- **Разрыв +0.38 мс = L2 competition + thermal/scheduler drift**

Точная числовая атрибуция требует NCu in-chain (не isolated), но она сложна:  
- E2E-профиль NCu кастомизирует kernel launch order и меняет L2 behavior → replay overhead ≠ real
- Это второстепенный класс (+0.38 ms при основном приросте +0.91 ms) — оставляем как **thermal + L2-competition** без глубокой доводки

---

## 4. (c) ptxas-эксперимент нулевой цены — **закрыт**

Собрал dk с `-maxrregcount 126` и `-maxrregcount 124`:

| max regs | Actual | Spill | LDL/STL | Blocks | Wall 3-run median |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 128 (baseline) | 128 | 0 | 0 | 4 | **9.750** |
| 126 | 126 | 0 | 0 | 4 | 9.804 (+0.054, +0.5%) |
| 124→122 | 122 | 0 | 0 | 4 | 9.849 (+0.099, +1.0%) |

**Блоков не прибавилось** (все ≥4 blocks/SM by SMEM: 20992 B). **Wall хуже** при снижении регистров — компилятор reorders / spilling ranges.

**Люфт планировщику от 128→126→124 меньше цены пересчёта.** Ступень (0/ii) лестницы — **закрыта, не идёт в правку**.

---

## 5. Приоритет улик (по величине цены)

| Класс | Цена | Приоритет |
|:--|:-:|:-:|
| **(iii) transit MIO** | **+0.6-0.75 ms (65-80%)** | **топ-1** |
| (i) 5-й барьер | +0.13-0.17 ms (14-19%) | 2 |
| (ii) 128r-фон | ≈ 0 (эксп. (c) closed) | ✗ |
| L2/thermal in-chain | +0.38 ms | 3 (второстепенно) |

---

## 6. Рекомендация — приоритетная проба **(iii) SASS-аудит W2 transit**

По TZ 034: "если перекладочный MIO: SASS-аудит 6 SHFL+8 STS — прецедент V[]-класса".

**Проверить в SASS dk_new (post-033)**:
1. **V[]-класс**: есть ли runtime-indexed local массивы, вызывающие LDL/STL? (В коде я использую V0-V3 named vars — но компилятор мог создать shadow spill)
2. **Extra LDS/STS**: точный счёт LDS.32/STS.32 в pack region vs ожидаемые 8+8 = 16
3. **Unrolled ops**: `#pragma unroll` in Phase transpose expanded correctly?
4. **PRMT count**: 32 PRMT (16 × 2 slots) — не больше?
5. **SHFL synchronized cost**: реальное 6 или больше?

**Ожидание**: LDL/STL = 0 (проверено в 033-b), но может быть **implicit LDS overhead** от компилятора (загрузка constants, etc.) в pack region.

**Если найдётся V[]-класс** (LDL/STL > 0 в transit region) — правка **-4..-8r + wall win 0.3-0.5 ms** (аналог 018 fix класса).

**Если аудит чистый** — переход к пробе (i) барьерная тень (сместить W2 transit к времени когда варпы всё равно ждут).

---

## 7. Пробы в очереди (после доводки dk)

- **π_V перезамер на новом ландшафте** — по TZ **в очередь после (i)-(iii)**. Новая цена (mio 42, barrier 8) может изменить экономику π_V (снятие или сохранение).

---

## 8. Файлы

- NCu stall + L2 script: `runs/reports/034_stall_l2.sh`
- maxrregcount experiment script: `runs/reports/034_maxreg.sh`
- Binaries эксперимента: `libs/r1b_dk_126`, `libs/r1b_dk_124`

Chain md5: 033-c `1d23f27d…` → **034 `<computed>`**

---

**End 034.**

**Диагноз**: главная цена = transit MIO (+134M shared ops = +65-80% cost). Барьер вторичен (14-19%), 128r-фон незначителен. maxrregcount эксперимент закрыт (люфт не даёт выигрыш).

**Приоритет проб**: **(iii) SASS-аудит W2 transit** первым. Ожидаемый gain 0.3-0.5 мс при V[]-классе; чистый SASS → переход к (i) barrier shadow.
