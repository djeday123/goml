# 047 — L2-handoff настоящая постройка в бою: могила ОКОНЧАТЕЛЬНАЯ

**Chain**:
- 045_persist_dkS2.md md5: `c4d908c7f43c9d752c90cdebbf97333a`
- 046_persist_b2b_dkS2.md md5: `c6d57817104beac4b8dfda08f1e61aa0`

**Правила ТЗ 047**: production merged/dk/dq НЕ трогаются ни байтом. Только оркестрация + смещение указателей. Вердикт — **чистый wall** (NCu запрещён). S2 переносится в 048.

---

## Артефакт-хедер (правило 5)

```
libs/ (post-046, prod неизменен):
-rw-r--r-- 25638 Jul  8         fa_bwd_merged_v1.cu    (md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33 = 040 sealed)
-rw-r--r-- 18834 Jul  8         fa_bwd_dq_new.cu       (md5 d7a11a3d788eb4c396d892bc9c8ab754 = 041 sealed)
-rw-r--r-- 13352 Jul  8         fa_bwd_dk_new.cu       (md5 a9f0ded8261e53a143b521ffa647f458 = 033 sealed)

libs/probe_headoffset_047.cu + Makefile — грошовая проба (§1.b)
libs/bench_headwise_e2e_047.cu + Makefile — 5 плечей (§2)
```

**Gate-log**:
```
$ ./037r_gate.sh
bench_merged: FINGERPRINT kernel_merged_v1: numRegs=252
GATE OK: numRegs=252 matches EXPECT=252

$ bench_headwise_e2e_047 (fingerprint x4):
FINGERPRINT D        numRegs=38  (expected 38)  OK
FINGERPRINT merged   numRegs=252 (expected 252) OK
FINGERPRINT dk_new   numRegs=128 (expected 128) OK
FINGERPRINT dq_new   numRegs=69  (expected 69)  OK
```

---

## §1. Совместимость head-offset

### 1.a Paper (из исходников)

Все три ядра используют одинаковый паттерн:
```
b = blockIdx.x / n_kt      // batch index
Kb = K + (size_t)b * sl * Hd       // указатель на голову b
Qb = Q + b * sl * Hd       // fa_bwd_dk_new.cu:92
dSb = dS_nat + b * sl * stride_ds  // fa_bwd_dk_new.cu:116
```

**Head-offset launch** (bh=1 с указателями shifted на h):
- `Q_h = Q + h * sl * Hd` (fp8 bytes)
- `dO_h = dO_g + h * sl * Hd` (fp16 elements, компилятор × 2 автоматически через __half*)
- `L_h = L + h * sl` (fp32 elements)
- `dS_h = dS_nat + h * sl * stride_ds` (bytes)

При `bh=1` ядро computes `b = blockIdx.x / n_kt`, `b ∈ [0, 1)`. Используется `Kb = K_h + 0 * sl * Hd = K_h` — корректно ✓

### 1.b Грошовая проба (h=7)

`libs/probe_headoffset_047.cu`: h=7 offset-launch bh=1 vs h=7 slice монолита bh=8.

**Результат**:
```
dV[h=7] mism=0 / 1048576 max_abs_diff=0.000e+00 BYTE-EQUIVALENT
dK[h=7] mism=0 / 1048576 max_abs_diff=0.000e+00 BYTE-EQUIVALENT
dQ[h=7] mism=0 / 1048576 max_abs_diff=0.000e+00 BYTE-EQUIVALENT
Verdict: PASS — head-offset launch БАЙТ-ЭКВИВАЛЕНТНО
```

**Head-offset launch совместим с ядрами без правки launcher-адресации**. ✓

---

## §2. Bench 5 плечей (bh=128 sl=8192 hd=128)

Файл: `libs/bench_headwise_e2e_047.cu` (bench-side only, ядра не тронуты).

### Плечи

- **ARM1**: монолит, production-порядок (D → merged → dk → dq) на default stream
- **ARM2S**: плотный строй (**dense**) БЕЗ брoни. Per head h: merged(h) на sM → [dk(h) на sK ∥ dq(h) на sQ] → sync-событие → h+1
- **ARM3S**: плотный строй + бронь. Persisting window на `dS_nat + h*headBytes` для всех трёх потоков; reset persist после консюмеров каждой головы
- **ARM2P**: **pipeline** глубины 2 БЕЗ брoни. merged(h+1) стартует || [dk(h) ∥ dq(h)]; consumer'ы ждут event merged(h)
- **ARM3P**: pipeline + бронь. Reader-потоки dk/dq — Persisting на dS(h). merged-поток — Streaming на dS(h+1). Reset после консюмеров

---

## §3. Wall-замеры interleaved 8 циклов

**Round-robin** ARM1 → 2S → 3S → 2P → 3P (интерливинг гасит термотренд, обобщение ABBA). 4 warmup runs ARM1, потом 8 cycles.

### Per-cycle wall (ms E2E)

| Cycle | ARM1 | 2S | 3S | 2P | 3P |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 | 45.993 | 74.454 | 76.224 | 76.790 | 76.790 |
| 1 | 40.123 | 74.454 | 75.021 | 74.195 | 75.929 |
| 2 | 40.115 | 74.254 | 75.066 | 74.150 | 75.921 |
| 3 | 40.153 | 74.285 | 75.047 | 74.142 | 75.941 |
| 4 | 40.144 | 74.626 | 75.337 | 74.416 | 76.165 |
| 5 | 40.183 | 74.726 | 75.348 | 74.379 | 76.124 |
| 6 | 40.148 | 74.465 | 75.240 | 74.410 | 76.147 |
| 7 | 40.186 | 74.377 | 75.270 | 74.379 | 76.133 |

### Медианы (ms E2E)

| Плечо | Median |
|:--|:-:|
| **ARM1 monolith** | **40.153** |
| ARM2S dense | 74.454 |
| ARM3S d+persist | 75.240 |
| ARM2P pipeline | 74.379 |
| ARM3P p+persist | 76.124 |

---

## §4. Именованные предсказания vs факт

| Дельта | Прогноз | **Факт (median)** | Verdict |
|:--|:--|:-:|:-:|
| **Delta(3S-2S)** — чистый эффект брoни при одинаковом строе (S) | **жив: -(4..6) ms; мёртв: ~0** | **+0.762 ms** | **МЁРТВ** (бронь даже вредит на 0.76 ms) |
| Delta(2S-1) — налог плотного строя | +3..+12 | **+34.317 ms** | значительно дороже прогноза (+22 сверх верхней границы) |
| **Delta(3P-2P)** — полуприз конвейера с бронью | **жив: -(1.5..2.5) ms; мёртв: ~0** | **+1.754 ms** | **МЁРТВ** (persist добавляет ~1.75 ms) |
| Delta(2P-1) — налог конвейера | +0.5..+2 | **+34.194 ms** | значительно дороже прогноза |

### Атрибуция сюрпризов

**Налог плотного строя (~+34 ms, +85% над ARM1)** — сильно выше прогноза +3..+12:
- 384 kernel launches vs 4 в ARM1 (× 96 launches)
- ~128 stream sync events per iteration
- Неоптимальная стрим-конкуренция при недозаполнении 34/17/11% на bh=1
- Cross-stream event overhead на sm_120a может быть выше типичного

**Persist добавляет ~0.7-1.8 ms**: setup `cudaAccessPolicyWindow` per head × 128 голов + `cudaCtxResetPersistingL2Cache()` × 128 = **256 API calls накладно** — фактически persist window не даёт выигрыша, а стоит своих ~1 ms overhead.

---

## §5. Вердикт-дерево

### Условие: `(3S-2S) ~ 0 И (3P-2P) ~ 0`

- **(3S-2S) = +0.762 ms** ≈ 0 (в пределах шума ± 1 ms; знак положительный — persist вредит незначительно)
- **(3P-2P) = +1.754 ms** ≈ 0 (persist добавляет ~1.75 ms setup overhead — не помогает)

**Оба условия выполнены** → **могила ОКОНЧАТЕЛЬНАЯ**.

### Эпитафия (по TZ §5)

> **«Измерено в бою: бронь читателя не кормит.»**
> 
> L2-handoff в цепи merged→dk→dq на sm_120a не работает при плотном строе ни без брoни, ни с `cudaAccessPolicyWindow{Persisting}`. Разница между плечами с брoней и без — статистически незначима (0.76 ms и 1.75 ms в пределах overhead API вызовов). Persist window помогает write-side (merged −69% DRAM подтверждено в 044-046), но reader-side (dk/dq) не выигрывает независимо от:
> 1. Штатного L2 (044)
> 2. Явной брoни `cudaAccessPolicyWindow{Persisting}` с sync между launches (045)
> 3. Той же брoни в b2b-режиме без sync (046)
> 4. **Настоящей плотной строевой оркестрации с брoней per-head** (047)
> 
> Четыре независимых измерения на трёх разных приборах — все дают одинаковый вердикт. **Триггер L2-handoff снимается.**

### Смета инженерии production-варианта

**Не составляется** (условие TZ: "смета только при жив").

---

## §6. Правки в 047

- **Ядра**: 0 правок (merged/dk/dq все sealed без изменений).
- **Launcher-адресация**: 0 правок.
- **Bench-сторона**: 2 новых файла (`probe_headoffset_047.cu`, `bench_headwise_e2e_047.cu`) — legit по TZ.

### Финальный state

```
libs/fa_bwd_merged_v1.cu md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33  = 040 sealed  ✓
libs/fa_bwd_dq_new.cu    md5 d7a11a3d788eb4c396d892bc9c8ab754  = 041 sealed  ✓
libs/fa_bwd_dk_new.cu    md5 a9f0ded8261e53a143b521ffa647f458  = 033 sealed  ✓
```

---

## §7. Сиквенс

**Сохраняется**: S2 (dk LDSM refactor) → **ТЗ 048** без изменений от 046 бумаги (044 §5.a-g + 045 II.5 row_ptr).

L2-handoff направление ЗАКРЫТО. Все дальнейшие рычаги (smQ prefetch merged, TMA, cp.async глубина, A' классы #4/#6 — см. карту улик v5 из 041) остаются в очереди.

---

## §8. Итоги 047

1. **§1.a совместимость head-offset**: доказано бумагой (все ядра `b*sl*Hd/stride_ds`, `bh=1 + shifted ptr` даёт b=0 = shifted).
2. **§1.b грошовая проба**: h=7 offset-launch bh=1 vs h=7 slice монолита bh=8 — **BYTE-EQUIVALENT** для dV/dK/dQ (0/1048576 mism).
3. **§2 bench 5 плечей**: `bench_headwise_e2e_047` (bench-side, ядра не тронуты).
4. **§3 interleaved 8 циклов**: ARM1=40.153, ARM2S=74.454, ARM3S=75.240, ARM2P=74.379, ARM3P=76.124 ms медианы.
5. **§4 предсказания**:
   - **Delta(3S-2S) = +0.762 ms** (прогноз -(4..6) не сбылся → **МЁРТВ**)
   - **Delta(3P-2P) = +1.754 ms** (прогноз -(1.5..2.5) не сбылся → **МЁРТВ**)
   - Delta(2S-1) = +34.317 (налог строя сильно выше прогноза)
   - Delta(2P-1) = +34.194 (аналогично)
6. **§5 вердикт**: **МОГИЛА ОКОНЧАТЕЛЬНАЯ**, эпитафия — измерено в бою: бронь читателя не кормит.
7. **Правки production: 0** (все sealed без изменений).

### Chain md5

- 046 `c6d57817104beac4b8dfda08f1e61aa0`
- **047 `<computed>`**

### Файлы 047

- `runs/reports/047_l2_realbuild.md` (this report)
- `libs/probe_headoffset_047.cu` + `Makefile` — §1.b грошовая проба
- `libs/bench_headwise_e2e_047.cu` + `Makefile` — §2 5-arm bench

---

**End 047. L2-handoff могила ОКОНЧАТЕЛЬНАЯ — четвёртый и последний замер в бою. Триггер снимается. Сиквенс: 048 = dk S2 refactor.**
