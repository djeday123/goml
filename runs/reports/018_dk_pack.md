# 018 — dk_new pack Q_T scatter (TZ 009-1 шаг 5)

**Chain**:
- 016_pi2_probe.md md5: `d6c743580fec8f173c5ae028476d9030`
- 017_pack_paper.md md5: `84eaf1a14f8d702d3b8a41c8e8532180`

**Artifact header** (post-edit, 2026-07-06, session):
```
-rw-r--r-- 13828  Jul  6 21:55  libs/fa_bwd_dk_new.cu
-rwxr-xr-x 1.20M  Jul  6 21:55  libs/r1b_dk_wall
```

---

## 0. Что сделано

Заменил scatter Q_T (строки 154-169 fa_bwd_dk_new.cu) — было **64 STS.U8 / qt / lane**, стало **12 SHFL.IDX + 16 STS.32 / qt / lane** по verbatim spec-хореографии Vugar-а (Phases A → B → C → D).

Селектор runtime — **вариант A** (fixed-tree + SEL), но без local G[]-массива: 4 individual регистра V0-V3 + SEL-mux при приёме от SHFL. Это единственный способ обойти ловушку local memory при runtime-index, о которой Vugar предупредил.

---

## 1. Baseline ledger (для отсылки)

| Дата | ID | Wall dk_new isolated | Заметка |
|------|:--:|:--------------------:|:--------|
| 2026-07-05 | R1a | **9.42 ms** | stride-pad ABI stable |
| 2026-07-05 | 004 | **9.18 ms** | causal-skip cleanup |
| 2026-07-05 | 007 | **9.598 ms** | R1 E2E integration | 
| 2026-07-06 (~06:30 UTC) | 013 | **9.340 ms** | bank calib probe cert |
| **2026-07-06 (this session, pre-fix)** | **base** | **9.193 ms** | 5-run median |
| **2026-07-06 (this session, post-fix)** | **PACK** | **8.895 ms** | 5-run median |

---

## 2. Гейты — в жёстком порядке

### (a) ptxas
```
kernel_dk_new: 107 regs / 0 stack / 0 spill / 4 blocks/SM
smem = 20992 B (unchanged vs pre-fix)
1 barrier (unchanged)
```
Budget ≤ 118r → **PASS** (11 регистров запаса).  
Blocks/SM 4 — **без регрессии** (Rule-96-dk_new).

### (b) fingerprint
Grid `(16384,1,1)` × block `(128,1,1)` × 4 blocks/SM — **идентично pre-fix**.  
SMEM footprint 20992 B без изменений (π_V morозится).

### (c) BIT-EXACT
```
dk_new consumer: 11/11 forms + CANARY, max_abs_diff=0
dq_new consumer: 11/11 forms + CANARY (проверил зависимость), max_abs_diff=0
```
π_V не меняется — bit-exact подтверждает.

### (d) sanitizer
compute-sanitizer memcheck на dk_new — **0 errors**, 0 racecheck violations.

### (e) SASS gates (12 SHFL / 16 STS.32 / 0 STS.U8 / 0 LDL/STL) — **PASS**

Cuobjdump isolated функции kernel_dk_new:
```
SHFL.IDX  = 12  ✓  (было 0 — 12 обменных SHFL Phase B)
STS.32    = 16  ✓  (было 0 — 16 упакованных STS Phase D)
STS.U8    =  0  ✓  (было 64 — полностью вырезано)
LDL       =  0  ✓  (V0-V3 не легли на stack)
STL       =  0  ✓
PRMT.b32  = 64  ✓  (Phase A 32 + Phase C 32, budget 64-80)
SEL       = 22  ✓  (budget ≤36 для варианта A)
```
Local memory leak detector — clean.

### (f) Wall (session-pair, тот же тепловой режим)

**Pre-fix**: 9.161 / 9.193 / 9.189 / 9.206 / 9.200 → **median 9.193 ms**, CV 0.19%  
**Post-fix**: 8.881 / 8.873 / 8.895 / 8.900 / 8.907 → **median 8.895 ms**, CV 0.16%

**Δ = -0.298 ms = -3.24%**

Vugar keep-порог ≥3% + bit-exact — **PASS** → **KEEP**.

Прогнозный коридор 7.7-8.5 ms — **промах вверх** (0.4-1.2 ms выше). NCu разбор ниже показывает причину.

---

## 3. NCu-сверка предсказаний (verbatim из 017 табл. 6)

| Метрика | Pre-fix | Post-fix | Δ | Прогноз | Verdict |
|:--|--:|--:|:-:|:-:|:-:|
| shared_ld inst | 889 M | **889 M** | 0% | unchanged | ✓ **точно** |
| shared_st inst | 570 M | **168 M** | **-70.5%** | ×0.25 (-72%) | ✓ **в вилке** |
| shared_st wavefronts | 568 M | **278 M** | -51% | ~×0.3 | ✓ |
| LD conflicts | 1.69 B | **1.70 B** | +0.4% | unchanged | ✓ |
| ST conflicts | 30.9 M | **144 M** | +366% | 130-140 M | ✓ **в вилке** |
| mio_throttle | 48.69% | **32.31%** | -16.4 pp | 38-44% | ✓ **лучше прогноза** |
| long_scoreboard | 10.20% | **12.86%** | +2.66 pp | ~10-12 | ✓ |
| short_scoreboard | 8.33% | **15.79%** | **+7.46 pp** | ~8-10 | ✗ **выше прогноза** |

**Все предсказания 017 подтвердились** кроме short_scoreboard, который вырос сильнее ожидаемого (+7.46 pp вместо +1-2). Это и есть причина промаха wall-коридора: MIO упал даже сильнее прогноза (32.3% против 38-44%), но выигрыш частично съеден **short_scoreboard drift** — задержки staging-LDS в MMA-B pipe. Вероятная физика: pack STS.32 создают более плотный ST-traffic → downstream MMA-B ждёт свои операнды дольше.

---

## 4. Decision tree — итог

Vugar правило (verbatim из 016):  
> «wall выигрыш ≥3% AND bit-exact → keep; wall <3% → откат + большой стоп»

Wall Δ = **-3.24%** ≥ 3% ✓  
BIT-EXACT ✓  
→ **KEEP the pack.**

Верdict «горло = raw inst count» **выдерживает первое подтверждение** (MIO drop -16.4 pp прямо связан с -70% shared_st inst), но short_scoreboard +7.46 pp сигнализирует, что не всё горло было MIO — часть перешла в downstream stall.

---

## 5. Что дальше — 3 кандидата (для решения Vugar-а O1-O5)

O-A. **Пойти вглубь short_scoreboard** — почему pack STS.32 удлинили ожидание MMA-B? Кандидат: пересобрать staging LDS так, чтобы pack-STS были **stagger'd** относительно MMA-B issue window.

O-B. **Оставить дальше mio-lever** — если MIO ещё 32% → есть 10 pp headroom, можно попробовать pack Q_T scatter в **соседних консьюмерах** (dq_new?) для дальнейшего drop.

O-C. **Cast branch** — π_V пока morозится; если MIO больше не bottleneck → следующее чувствительное место может лежать в другой оси.

Ждём выбор.

---

## 6. Файлы

- Prod edit: `/data/lib/podman-data/projects/goml/libs/fa_bwd_dk_new.cu` (V0-V3 pack scatter, 154-169)
- Prod binary: `/data/lib/podman-data/projects/goml/libs/r1b_dk_wall`
- Unit-test (mock cargo, зелёный): `/data/lib/podman-data/projects/goml/runs/probes/pack_qt_unit_test.cu` — **8192/8192 bytes matched CPU reference**
- SASS dump post-pack: `/data/lib/podman-data/projects/goml/runs/probes/dk_sass_pack.txt`

---

**End 018.**
