# HANDOFF: fa-blackwell-fp8 campaign post-071b → new Fable session

**Дата**: 2026-07-10
**Передача**: Opus 4.7 (session 062-071b) → Fable 5 (следующая сессия)
**Пользователь**: Vugar (vugar.bakhshaliev@smartsolutions.az)

Этот документ — **исчерпывающий handoff** для восстановления полного контекста кампании оптимизации FA-Blackwell-fp8 backward на sm_120a RTX PRO 6000 Blackwell. Читать целиком перед первым замером.

---

## §0 COLD-START CHECKLIST (первые 60 секунд новой сессии)

```bash
# 1. Working dir
cd /data/lib/podman-data/projects/goml

# 2. Gate-silence
nvidia-smi --query-compute-apps=pid,process_name --format=csv
# ожидание: только header, EMPTY (иначе — kill процессы или разбираться)

# 3. Sealed md5 проверка
md5sum libs/fa_bwd_dk_new.cu libs/fa_bwd_merged_v1.cu libs/fa_bwd_dq_new.cu libs/fa_bwd_common.cuh
# ожидание ТОЧНО:
#   25e5e1077cc3bec2c49bf9288fe60c54  libs/fa_bwd_dk_new.cu     (S2v4 061 KEEP)
#   2bf32ab7d4c5ecabb4ee2dbf1b5d4b33  libs/fa_bwd_merged_v1.cu  (040 KEEP + LDSM.x4.trans #7)
#   d7a11a3d788eb4c396d892bc9c8ab754  libs/fa_bwd_dq_new.cu     (041 KEEP разморозка)
#   4407ec9cf64708a2a28dc36633d5d6f1  libs/fa_bwd_common.cuh

# 4. Fingerprint gate первым shot'ом
libs/bench_r2c_e2e | head -6
# ожидание:
#   FINGERPRINT kernel_d_precompute    numRegs= 38 (expected  38) OK
#   FINGERPRINT kernel_merged_v1       numRegs=252 (expected 252) OK
#   FINGERPRINT kernel_dk_new          numRegs=124 (expected 124) OK
#   FINGERPRINT kernel_dq_new          numRegs= 69 (expected  69) OK

# 5. Sanity wall (для thermal state)
libs/bench_r2c_e2e 2>&1 | grep total=
# nc ожидание: total ~= 42.1-42.4 ms (cert 42.346, drift ±1%)

# Если что-то не совпадает — СТОП и разбираться (не начинать новую правку!)
```

---

## §1 ПРОЕКТ: identity + аппаратура

**Задача**: production-grade FP8 e4m3 FlashAttention **backward pass** kernel для NVIDIA RTX PRO 6000 Blackwell WS Edition (`sm_120a`) / consumer Blackwell.

**Хардвар**:
- 188 Streaming Multiprocessors
- 65536 registers / SM
- 101376 B opt-in shared memory / SM (по умолчанию 49152, увеличивается через `cudaFuncSetAttribute`)
- 32 threads / warp, 128 threads / block (все prod kernels)
- 96 GB VRAM (97239 MB total)
- L2 = 131072 KB (128 MB)
- CC 12.0 (`sm_120a`)
- Driver 580.159.03 или новее

**Canonical form** для всех замеров:
- `bh=128, sl=8192, hd=128, seed=42`
- `Br = 64` (rows Q tile), `Bc = 64` (K/V tile) для всех trio kernels
- `warmup=5 iters=20` в bench_r2c_e2e

**Компиляция**:
- Компилятор: `/usr/local/cuda-13.1/bin/nvcc` (НЕ default `/usr/bin/gcc-cuda`)
- Флаги: `-O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas=-v`
- Cache dir: `HOME=/tmp` (иначе ncu warnings про Documents/NVIDIA Nsight)

---

## §2 SEALED PRODUCTION СОСТОЯНИЕ (byte-identical неизменны)

### Основные kernels (правки в 072+ ЗАПРЕЩЕНЫ без TZ)

| Файл | md5 | Regs | Occ blk/SM | Sealed эпоха | SMEM B |
|:--|:--|:-:|:-:|:-:|:-:|
| `libs/fa_bwd_merged_v1.cu` | `2bf32ab7d4c5ecabb4ee2dbf1b5d4b33` | 252 | 2 (SMEM-lim) | 040 LDSM.x4.trans #7 (−12%) | 41472 |
| `libs/fa_bwd_dk_new.cu` | `25e5e1077cc3bec2c49bf9288fe60c54` | 124 | 4 (reg-lim) | 061 S2v4 swz+LDSM.x2 (−19.24%) | 12288 |
| `libs/fa_bwd_dq_new.cu` | `d7a11a3d788eb4c396d892bc9c8ab754` | 69 | 6 (SMEM-lim) | 041 разморозка (−3.5%) | 13824 |
| `libs/fa_bwd_common.cuh` | `4407ec9cf64708a2a28dc36633d5d6f1` | — | — | swz_byte + cpa16 helpers | — |

### Reference kernels (bit-exact сравнение)

| Файл | md5 | Regs | Роль |
|:--|:--|:-:|:--|
| `libs/fa_bwd_ds_gen.cu` | `665a350d3da8ae90b816ccd6b55db346` | 130 | R1 reference dS_nat + dS_T generator |
| `libs/fa_bwd_dk.cu` | `068d6a4fdf5ae04816ebca199b9293cc` | 248 (dk sealed) + 38 (d_precompute) | sealed dK reference + D-precompute |
| `libs/fa_bwd_dv_mma_p1.cu` | — | — | sealed dV_p1 reference |
| `libs/fa_bwd_dq.cu` | — | — | sealed AA1 dQ reference |

### Fingerprint EXPECT (в `libs/bench_r2c_e2e.cu:69-72`)

```cpp
{"kernel_d_precompute", (const void*)fa_bwd_dk::kernel_d_precompute,       38},
{"kernel_merged_v1",    (const void*)fa_bwd_merged_v1::kernel_merged_v1,  252},
{"kernel_dk_new",       (const void*)fa_bwd_dk_new::kernel_dk_new,        124},
{"kernel_dq_new",       (const void*)fa_bwd_dq_new::kernel_dq_new,         69},
```

**Правило**: изменение любого EXPECT-числа = осознанная правка в комментарии + reason + новая эпоха.

### Bench patch state (070 KEEP)

`libs/bench_r2c_e2e.cu` — единственная non-sealed правка post-062:
- **Line 126** (chain): `CKR(cudaMalloc(&dS_nat,dsz)); dS_T = nullptr;   // 070: dS_T dead-alloc removed`
- **Line 208** (wall): `CKR(cudaMalloc(&dS_nat,dsz)); dS_T = nullptr;   // 070: dS_T dead-alloc removed`
- **Archive**: `libs/bench_r2c_e2e.cu.pre_070` (audit trail, восстановление тривиально `cp`)

---

## §3 CERT NUMBERS (публичные, sealed) + производственный wall

### Non-causal E2E

- **Wall**: **42.346 ms** cert (062 30-run median), CV **0.098%**
- **TFLOPS proj (16N²d)**: **415.44 T** (Tri Dao V3 reference, sealed 8-MMA)
- **TFLOPS fused (10N²d)**: **259.65 T** (R2C actual = 5-MMA fused-min)
- **Sequential vs sealed 285.44 T**: **+46.1% cumulative** (за 040+041+061)

### Causal E2E

- **Wall**: **22.206 ms** cert (063 30-run median), CV **0.074%**
- **Live/full formula**: 8256/16384 = **0.5039** (upper-triangle including diagonal)
- **NCu insts ratio**: 0.505-0.506 (063-r) — совпал с paper до 0.2%

### Per-kernel breakdown (066/063 reference)

| Kernel | NC (ms) | CAUSAL (ms) | CAUSAL/NC | NCu max/avg (causal) |
|:--|:-:|:-:|:-:|:-:|
| D-precompute | 0.342 | 0.340 | 0.994 (control) | 1.00 |
| merged_v1 | 25.125 | 12.733 | 0.5068 | 1.021 |
| dk_new | 8.423 | 4.572 | 0.5428 | 1.031 |
| dq_new | 8.462 | 4.586 | 0.5420 | **1.045** (max skew) |
| **Total** | **42.352** | **22.231** | 0.5250 | — |

### FP8 precision floors (inherent to e4m3, not bugs)

| Gradient | NC max_abs_diff | Causal max_abs_diff |
|:--|:-:|:-:|
| dK | ~4.65e-3 | ~3.1e-2 |
| dV | ~4.75e-3 | ~3.2e-2 |
| dQ | ~4.87e-3 | ~3.3e-2 |

---

## §4 PUBLIC v0.2.0 (Apache-2.0, sealed 064-065)

### Git state

- **Directory**: `/data/lib/podman-data/projects/goml/release_v0.2.0/`
- **Commit**: `506b69f v0.2.0: initial public release`
- **Tag**: `v0.2.0` (annotated, локально)
- **Remotes**: **НЕТ** (Vugar настраивает `git remote add origin <URL>`)
- **License**: Apache License 2.0 + SPDX headers во всех 11 sources

### Post-SPDX md5 (в теге v0.2.0)

```
src/fa_bwd_dk_new.cu       eb492e0729ef643280591b8c8dd8a29d
src/fa_bwd_merged_v1.cu    720774c28807d01214adff16c9003221
src/fa_bwd_dq_new.cu       7660bd960cc39c799d588c573bb47c5d
src/fa_bwd_common.cuh      5a948c2e8005f569424f0b4e8c25928e
src/fa_bwd_dk.cu           d839118d09dbb8c974eab29e77e4e566
src/fa_bwd_dq.cu           abd23f297e19987e8cd5233526cac821
src/fa_bwd_ds_gen.cu       dfeaf3fdb825f10201bd086c9c10bc6c
src/fa_bwd_dv_mma_p1.cu    20402df15838314270456dc7b84ec1dc
tests/bench_r2c_e2e.cu     a2177f774b6a339d2f774ccc2c14c970
tests/r1b_dk_bit_exact.cu  890ca510d5541d6e336b6fe3a9f2e2fa
tests/r2c_merged_bit_exact.cu 9b83c4bbf312ea61b939ef2f50d4ebb5
```

**Разница** vs libs/ md5: только 2 строки SPDX header (`// SPDX-License-Identifier: Apache-2.0\n// Copyright (c) 2026 Vugar and the FA-Blackwell-fp8 authors\n`). Bytes ядер unchanged.

### GH Release draft (готов для gh release create)

- **Draft body**: `runs/reports/065_release_body.md` md5 **`f1ca49c7ed811ce847c2892bbc5ec9a3`**
- **Grep clean**: 0 совпадений на `x2.8` / `2.8x` / `H200` (проверено 065)
- **H100 упомянут** строго по карте дуэлей: FA3 быстрее абсолютно, мы ×2 per-dollar, empty-chair приглашение на прямой матч

### Команды для Vugar (после настройки remote)

```bash
cd /data/lib/podman-data/projects/goml/release_v0.2.0
git remote add origin git@github.com:<USER>/fa-blackwell-fp8.git
git push -u origin main
git push origin v0.2.0

gh release create v0.2.0 \
  --title "FP8 FlashAttention backward, world-first -- 42.35 ms nc (415 proj / 260 fused) | 22.21 ms causal (495 eff fused), 30-run cert, sm_120a" \
  --notes-file /data/lib/podman-data/projects/goml/runs/reports/065_release_body.md
```

### Публичный URL (после push)

`https://github.com/<USER>/fa-blackwell-fp8/releases/tag/v0.2.0`

Команда внешнего человека для верификации:
```bash
git clone <URL> && cd fa-blackwell-fp8 && git checkout v0.2.0 && ./verify.sh
```

---

## §5 CHAIN MD5 всех отчётов 062-071b

```
062  b7044db70019e8fa7dea260f9f235b6c   cert package 400 T KEEP
063  b7c82475ed49ac4821c7346f99a38fb1   causal release 22.206 ms 30-run
063-r 1dce5e445e5b47152c2894ffc7947b30  work-detector NCu + chrono cross-check +0.53%
064  c099586c01ad07d6c83f3d732a77de3c   🚀 v0.2.0 SEALED (Apache-2.0 + SPDX + clean-clone verify)
065  cc5c2a7f96aeed162ddf28609703009a   GH release draft (release_body f1ca49c7)
066  029b8c4b9b6e154ad437706eafd25a1d   causal recon (skew histogram + DRAM attribution)
067  ecbdeff9a42be2cf20b5d4d2afc41de7   🔴 dead-dS-skip = уже реализовано (qt_start=kt эпохи 018/026)
068  0bba4f923390593e7b51b278c3891d56   🔴 wave-remap +11.5% wall REGRESS (L2-locality loss)
069  76c958364d1d2ac74c2a4f86b87e4dfe   🔴A dk +40% spill / 🟠B V-repaint STOP на мосту
070  83fc3f2c3c2817a2660defb3f246330e   🟢 KEEP dS_T dead-alloc снят (−8.59 GB VRAM)
        (текущий диск b4b0ba63... — hook consolidation изменил файл после моего Edit)
071  42126d3ff26848dacc1fc660a06dcdd1   🟠 x1-probe STOP на мосту (гибрид: dup + −11r)
071b 19e3a3818ef0456c070fc03b49d0c773   🪦 МОГИЛА b0_b bridge (3 независимые причины)
```

**Verdicts summary**:
- 🟢 KEEP: 062 (cert), 063 (causal), 064 (release), 065 (draft), 070 (VRAM)
- 🟠 STOP на мосту: 069B (V), 071 (x1 гибрид)
- 🔴 ROLLBACK: 067 (dead-write skip уже был), 068 (remap), 069A (spill)
- 🪦 МОГИЛА: 071b (b0_b bridge не собирается)

---

## §6 РЕЕСТР B (пост-071b) — что заморожено / могила / активно

### B(i) 5-я бригада dk_new

| Путь | Приз (r) | Кумулятив | Статус | Дата закрытия |
|:--|:-:|:-:|:-:|:--|
| **Цель 5 blk/SM** | −22 | — | цель (65536/128/102=5.01) | — |
| launch_bounds hint (069A) | −28 formal | +22 spill LMEM | 🔴 069A red | 2026-07-10 |
| LDSM.x1 single-path (071/071b) | −11 при зелёном мосте | −11 | **🪦 МОГИЛА** | 2026-07-10 |
| SHFL restructure W_all[8] | est. −4..−8 | −15..−19 | ⚫ не пробовалось | эшелон-3 |
| fp16x2 packed dK_acc | est. −32 | breaks bit-exact | ⚫ не пробовалось + sealed re-baseline | эшелон-3 |
| cp.async pattern change | est. TBD | — | ⚫ не пробовалось | эшелон-3 |

**LDSM.x1 могила — 3 независимые причины**:
1. b0_b физически at col=8 (доказано col-marker probe: x2.R1 bytes = 0x08) — **misaligned для LDSM.x1.b8** (требует 16-byte alignment)
2. m16n16.x1.trans.b8 R1 = HW dup R0 (071 подтвердил в 32/32 lanes warp 0) — R1 НЕ carries b0_b
3. m16n8.trans **не поддерживается на sm_120a**: `ptxas fatal "Illegal matrix shape для ldmatrix + .trans not allowed for m16n8"`

**Приз −11r structurally exists (ptxas discriminator: 96r → 113r с 4 x1 vs 2 x2), но NOT semantically achievable** — kernel собирается с guess addr, но bit-exact заведомо красный.

### B(ii) V-reader-LDSM (класс #5 merged)

| Аспект | Состояние |
|:--|:--|
| V writer | Уже swz_byte swizzled с 040/061 (`merged_v1.cu:130`) |
| V reader | Единственный (Step D dP MMA-B, `merged_v1.cu:301-302`), direct LDS uint16 |
| Мост #5 (LDSM.x?.trans.b16 fragment mapping) | **НЕ ПОСТРОЕН** (dedicated microprobe ≈ 058b session-day) |
| Могила 054-#5 | Не закрыта финально (status quo) |
| Боезапас | 0.5-1.9% wall (054 evidence) — «монетка у порога» правило 2/3 v2 |
| **Триггер разморозки** | FP4-эпоха утолщает очередь merged / попутный мост из другого ТЗ / dedicated 069b probe day |

### B(iii) Ремап-v2 intra-bh

- **Гипотеза**: сортировка kt внутри одной bh (bh outer сохранён — сохраняет bh-major L2-locality в отличие от 068 heavy-first LPT который её сломал)
- **Потолок**: ≤0.3% causal E2E — низкий (частичная балансировка, только intra-bh кластер)
- **Триггер**: дешёвый rescue-приз если другой ТЗ даст микро-регресс

### B(iv) Правило 12 (spill/LDL=0)

**АКТИВНО постоянно, независимо от текста ТЗ**:
> «Если ptxas говорит `stack frame > 0 bytes` OR `spill_stores > 0` OR `spill_loads > 0` — правка автоматически КРАСНАЯ, гейт останавливается, независимо от того что regs/occ формально попали в KEEP-условие ТЗ. Артефакт `.ptxas.log` обязан заголовком включать эту тройку строкой первой».

**Обоснование** (069A): `__launch_bounds__(128, 5)` дал formally 96r ≤102 + 5 blks/SM = formal KEEP по условию ТЗ. НО stack=80, spill_store=144, spill_load=144. ABBA показал −40% dk isolated wall. Правило автоматизирует раннее уловление таких «formal PASS + hidden spill FAIL» перед ABBA.

---

## §7 АКТИВНЫЕ ПРАВИЛА LEDGER

### Правило 2/3 v2 (KEEP-порог)

- KEEP требует ≥ **2% wall improvement** на вердикт-дорожке ТЗ (nc или causal — что задано)
- Плюс bit-exact 11/11 на 3 gradients + memcheck 0 errors
- Плюс canary (r1b_dk_bit_exact --inject BITFLIP catch)
- Плюс fingerprint неизменён ИЛИ осознанно обновлён
- ABBA ≥ 8 pairs (default protocol 059/060)
- Единогласный вердикт: 8/8 CAND быстрее (или медленнее)
- **Правило 2/3 v2 отличие от v1**: v2 требует триаду {statistical significance + bit-exact + fingerprint} — v1 был только wall

### Правило 8 (прогнозы регистров не пишу)

> «ptxas решит»

Не спекулировать сколько регистров будет ДО прогона. Только измерять по факту.

### Правило 12 (spill/LDL=0)

См. §6 B(iv).

### ABBA protocol (default ≥8 пар)

Alternating A/B/B/A/A/B/B/A × 2 = 16 shots. Тепловая усреднённость — half order forward, half reverse.

Реализация: два binaries (BASE и CAND) построенных ЗАРАНЕЕ, затем alternating exec без пересборок.

**Short-circuit**: если 3-shot single-arm показал ≥5% регресс однозначно (068/069A), rollback без формального 8-парного цикла (экономия thermal budget).

### Гейт-тишина

**ПЕРЕД каждым замером**:
```bash
nvidia-smi --query-compute-apps=pid,process_name --format=csv
```

Ожидание: только header. Любой foreign PID → **STOP + kill/wait** (см. 057 zombie 840120 = x2 slowdown).

### Artifact-header первой строкой отчёта

Обязательные компоненты:
- md5 всех sealed sources (dk_new, merged, dq_new, common)
- Fingerprint EXPECT vs actual
- Гейт-тишина статус ✓/✗
- Правки production ядер: **N** (обычно 0 в разведках / probes)

### cudaMemGetInfo > nvidia-smi memory.used

**Урок 070**: nvidia-smi `memory.used` показывает только COMMIT'ed pages (touched). Логически allocated но untouched memory ((например dS_T без записи) НЕ отображается. Использовать `cudaMemGetInfo(&free, &total)` для точной alloc-дельты.

---

## §8 УРОКИ КАМПАНИИ (в леджер)

### 068: grid-order = скрытая L2-locality

- Heavy-first LPT wave-remap (kt-outer / bh-inner под causal, dq reverse) дал **+11.5% wall REGRESS**
- Причина: разрушение bh-major FIFO-локальности → K/V L2 thrash (128 разных bh за волну = 2 MB одновременно активных K/V-регионов не помещаются в L2)
- + dS_nat write-hotspot на одной kt-column (128 блоков пишут concurrently)
- L2 pattern loss +2.60 ms >> skew-tax приз-верх 0.61 ms = **×4.3 overhead**

**Уроки**:
1. Grid-order определяет L2-locality даже когда outputs disjoint
2. Блочная independence по output ≠ по DRAM pattern
3. Wave-tail skew — приз-верх, не реалистичный (max/avg=1.045 → скрытая цена реорганизации > потенциальный выигрыш)

### 067: hypothesis-testing ТЗ = красная строка

- ТЗ 067 предполагало «dead-dS-writes существуют» → приз −0.8..−1.2 ms wall + 4.6 GB транзиент
- Grep-верификация: `qt_start = causal ? kt : 0` в merged:146 + Step F line 379-395 ВНУТРИ qt-loop = **skip уже реализован** с эпохи 018/026
- Balance-check: если бы dead писались, DRAM ratio был бы ~0.95, наблюдаемое 0.56 → доказательство
- **Урок**: перед implementation проверить grep + существующий код на предмет «уже сделано»

### 069A: __launch_bounds__ БЕЗ structural правки → spill гарантирован

- `__launch_bounds__(FA_DKN_THREADS, 5)` дал formal PASS: regs 96 ≤ 102, occ 5 blk/SM, spill fields — но 22 регистра ушли в LMEM
- Приз reg-diet съеден spill traffic: dk isolated 7.955→11.120 = **+40% wall**
- **Урок**: hint без structural правки → spill при 22-регистровой нехватке headroom
- Дал жизнь **Правилу 12** (spill/LDL=0 автоматически КРАСНЫЙ)

### 069B: V writer уже swizzled → «перекраска V» = задача reader

- V writer в merged (line 130) УЖЕ использует `swz_byte(j_local, col_byte)` с эпохи 040/061
- Единственная точка class #5 LDS-конфликта — reader (line 301-302, direct LDS uint16)
- Требует dedicated microprobe для LDSM.trans.b16 fragment layout под m16n8k16.f16 MMA-B (ISA-таблица 043 не содержит)

### 070: nvidia-smi memory.used lazy commit

- Naive `nvidia-smi --query-gpu=memory.used` показал 568 MB peak для bench, ожидалось 18 GB
- Причина: nvidia-smi видит только commit'ed (touched) pages. Untouched cudaMalloc'ed memory не отображается
- **Решение**: `cudaMemGetInfo(&free, &total)` inline probe → 18824 MB with_dST → 10632 MB no_dST → **точный Δ 8192 MB = 8.59 GB**

### 071b: LDSM.x1.b8 sm_120a — HW quirks (3 независимых)

1. R1 = dup R0 (32/32 lanes) — hardware duplicate на b8 variant (не устраняется компилятором)
2. m16n8.trans НЕ поддерживается (ptxas fatal)
3. b0_b at col+8 misaligned для 16-byte-aligned LDSM

**Диагноз откуда −11r в ptxas discriminator**: раздробление live-ranges (4 x1 asm-blocks × 2 outputs = 4 «точки разрыва») даёт nvcc scheduler больше свободы для reg-slot reuse dup outputs между инструкциями, чем 1 x2 asm-block с 4 outputs (все живы одновременно). **НЕ «мертвые карманы освобождены»**.

### Общий урок campaign

- **Sealed KEEP-серии**: 040 (LDSM.x4.trans #7 −12%) + 041 (dq разморозка −3.5%) + 061 (S2v4 dk −19.24%) = **~20% cumulative E2E wall reduction**
- **Cert 400 TFLOPS взят с запасом** (415 proj / 260 fused nc)
- **Публичный v0.2.0 запечатан** (Apache-2.0 + SPDX + clean-clone verify)
- **Causal-дожим (066-071b) ИСЧЕРПАН** без structural K/V-pipeline changes ИЛИ MMA acc precision change
- Все дальнейшие пути = эшелон-3 (multi-day dedicated microprobe sessions)

---

## §9 ДОРОЖНАЯ КАРТА эшелон-3 (не активно, ждёт триггера)

### dk 5-я бригада (SINGLE-PATH могила через LDSM.x1)

Единственные оставшиеся пути:

#### Путь A: SHFL restructure W_all[8]

- **Цель**: устранить промежуточное regfile хранение через inline SHFL exchange
- **Est. приз**: −4..−8r
- **Требует**: новую ISA пробу SHFL patterns для 128-thread intra-warp reorder
- **Риск**: bit-exact preserved если exchange semantically equivalent; ptxas решит

#### Путь B: fp16x2 packed dK_acc

- **Гипотеза**: mirror dq_new pack (16 uint32 vs 64 fp32 = −32r одним махом)
- **Est. приз**: −32r → 92r target, 5 blk/SM с запасом
- **BLOCKER**: **BREAKS bit-exact vs sealed dK** (fp16 acc vs fp32 acc)
- **Требует**: полный **sealed re-baseline** (30-run cert новых чисел + пересеализация эпохи, новый sealed dK для references, новые FP64 floors)
- **Стоимость**: 5-7 dev-days + пересборка epoch

#### Путь C: cp.async pattern change

- **Гипотеза**: новая load path (SMEM→register через direct LDS вместо LDSM.trans)
- **Est. приз**: TBD (bridge probe нужен)
- **Требует**: полностью новый bridge probe

**Комбо-путь**: любые 2-3 из A/B/C для достижения target −22r → 5 blk/SM.

### V-repaint (класс #5 merged) — эшелон-3 candidate

- **Blocker**: мост #5 LDSM.x?.trans.b16 fragment layout под m16n8k16.f16 не построен
- **Session cost**: ~ session-day dedicated microprobe (analog 058b для dk S2v4)
- **Триггер**: FP4-эпоха утолщает очередь merged (0.5-1.9% → 1-4% при удвоенной throughput)

### Ремап-v2 intra-bh — эшелон-3 candidate

- **Дизайн**: сохраняет bh-major FIFO-локальность (в отличие 068)
- **Потолок**: ≤0.3% wall — низкий
- **Стоимость**: 1-2 dev-days
- **Триггер**: rescue-приз если основные пути дают микро-регресс

### W-эшелон (память, не wall)

- **Compact tri-band dS_nat**: 8 GB → 4 GB per layer memory savings
- **Требует правки индексации в 3 ядрах** (merged writer + dk/dq readers)
- Стоимость: 5-7 dev-days + 3 × 30-run cert + пересеализация
- **Триггер**: W-серия упирается в память

---

## §10 ФАЙЛОВАЯ СТРУКТУРА (важные пути)

### Working dir

```
/data/lib/podman-data/projects/goml/
├── libs/                           # Production ядра, references, benches, probes
│   ├── fa_bwd_merged_v1.cu         # 040 SEALED (main merged kernel)
│   ├── fa_bwd_dk_new.cu            # 061 SEALED (S2v4 dk)
│   ├── fa_bwd_dq_new.cu            # 041 SEALED (dq new)
│   ├── fa_bwd_common.cuh           # swz_byte, cpa16 helpers, FA_BWD_STRIDE=128
│   ├── fa_bwd_ds_gen.cu            # R1 reference dS generator
│   ├── fa_bwd_dk.cu                # sealed dK reference + D-precompute
│   ├── fa_bwd_dv_mma_p1.cu         # sealed dV_p1 reference
│   ├── fa_bwd_dq.cu                # sealed AA1 dQ reference
│   ├── bench_r2c_e2e.cu            # Production wall bench (dS_T = nullptr post-070)
│   ├── bench_r2c_e2e.cu.pre_070    # Pre-070 archive (audit trail)
│   ├── bench_r2c_e2e               # Compiled binary (fingerprint + wall + bit-exact chain)
│   ├── r2c_merged_bit_exact.cu     # Bit-exact test for merged (dV + dS_nat)
│   ├── r1b_dk_bit_exact.cu         # Bit-exact test for dk_new + canary
│   ├── r1b_dk_wall.cu              # dk isolated wall bench (ABBA)
│   ├── r1b_dk_wall                 # dk isolated wall binary
│   ├── r2c_merged_wall.cu          # merged isolated wall
│   ├── r1c_dq_wall.cu              # dq isolated wall
│   ├── Makefile.bench_r2c_e2e      # Main bench Makefile
│   ├── Makefile.r2c_merged_bit_exact
│   ├── Makefile.r1b_dk_bit_exact
│   ├── Makefile.r1b_dk_wall
│   ├── x1_probe_071.cu             # 071 probe: LDSM.x1.trans.b8 hw quirk detection
│   ├── x1_bridge_071b.cu           # 071b bridge v1 (col+16, row+16 candidates)
│   ├── x1_bridge_071b_v2.cu        # 071b bridge v2 (m16n8 — ptxas fatal)
│   ├── fa_bwd_dk_new_x1.cu         # 071 ptxas discriminator (namespace fa_bwd_dk_new_x1)
│   ├── S2v4_bridge_probe_060.cu    # 060 dk S2v4 bridge microprobe (historical)
│   ├── S2v4_col_probe_061.cu       # 061 col-marker probe (historical)
│   ├── ldmatrix_isa_probe_043.cu   # 043 ISA-taблица LDSM variants
│   ├── ldmatrix_dkS2_probe_045.cu  # 045 ISA-045 quirk discovery
│   ├── probe_dkS2v2_bridge_048.cu  # 048 dkS2v2 bridge (historical dead-end)
│   ├── query_caps.cu               # Device capability probe (188 SMs, 65536 regs)
│   ├── quiet_gpu.sh                # Optional: kill foreign processes (059 helper)
│   └── [dozens of historical variants]
├── runs/reports/                   # Все отчёты 001-071b
│   ├── 062_cert400.md
│   ├── 063_causal_release.md
│   ├── 063r_causal_release_fix.md
│   ├── 064_release_final.md
│   ├── 065_release_notes.md + 065_release_body.md
│   ├── 066_causal_recon.md + 066_ncu_skew.sh + 066_ncu_skew.txt
│   ├── 067_dead_ds_skip.md
│   ├── 068_wave_remap.md + 068_build.sh + 068_sanity_causal.sh
│   ├── 069_echelon2.md + 069A_gate.sh/txt + 069A_abba.sh/txt + 069A_causal_shot.sh
│   ├── 070_dst_registry.md + 070_build.sh/txt + 070_gate.sh/txt + 070_vram_probe.cu/.sh/.txt
│   ├── 071_x1_probe.md + 071_build_probe.sh + 071_probe.txt + 071_run_probe.sh
│   ├── 071b_x1_bridge.md + 071b_bridge.txt
│   └── HANDOFF_071b_to_Fable.md    # ← этот файл
└── release_v0.2.0/                 # Public release tree
    ├── .git/                       # Local repo, commit 506b69f, tag v0.2.0
    ├── LICENSE                     # Apache License 2.0 полный текст
    ├── README.md                   # Public README (Apache-2.0, cert numbers)
    ├── Makefile
    ├── verify.sh                   # 5-stage verify: build → fingerprint → bit-exact merged → bit-exact dk → wall 5-run
    ├── src/                        # 8 sources (with SPDX headers)
    ├── tests/                      # 3 test sources
    ├── bin/                        # (empty until build)
    └── docs/cert/cert_summary.md
```

### Auto-memory

```
~/.claude/projects/-data-lib-podman-data-projects-claude-dashboard-workspace-538111eee755/memory/
├── MEMORY.md                                # Индекс (загружается автоматически в новую сессию)
├── project_fa_blackwell.md                  # Основной проектный memo (кампания 001-071b)
├── project_session_071b_snapshot.md         # Session snapshot post-071b (только что сохранён)
├── project_r_series_snapshot.md             # Session continuation R-series (эпоха 060-)
├── reference_handoffs.md                    # Handoff docs pointer
└── feedback_working_rules.md                # Rule-253/Rule-96/Vugar-rule + artifact-header
```

---

## §11 SESSION-FLOW правила (для новой Fable сессии)

### Начало сессии

1. **Cold-start checklist** (см. §0 — 60 секунд)
2. **Read auto-memory** (harness автоматически загрузил MEMORY.md через SessionStart:resume hook)
3. **Прочитать HANDOFF** (этот файл) если контекст нужен
4. **Прочитать project_session_071b_snapshot.md** для быстрой ориентации

### При получении ТЗ

1. Проверить не противоречит ли реестру B (§6) — «уже пробовалось / морозилка / могила»
2. Определить какая правка: production / bench-side / probe / paper-only
3. **Если production** → архив `.pre_XXX` + full gate mandatory
4. **Если probe** → отдельная директория / Makefile, вне production namespaces
5. Планировать якорь-строку отчёта (verdict + числа) ДО начала работы

### При завершении ТЗ

1. **Верификация sealed** (md5 dk_new/merged/dq_new = sealed)
2. **Финальная fingerprint** (252/124/69/38)
3. **Отчёт** в `runs/reports/NNN_*.md` с artifact-header
4. **Chain md5 обновление** (в конце отчёта)
5. **Auto-memory update**: MEMORY.md строка + project_fa_blackwell.md строка
6. **`md5sum` отчёта** для следующей ссылки в chain

### Правила красных строк

- 🔴 **ROLLBACK** = откат к sealed + строка в реестр «путь X не работает потому что Y»
- 🪦 **МОГИЛА** = окончательное закрытие пути (в отличие от 🧊 морозилки — не разморозится)
- 🧊 **Морозилка** = временное закрытие с триггером разморозки
- 🟠 **STOP на мосту** = probe/gate обнаружил blocker, production не тронут

---

## §12 ROLLBACK PROTOCOLS

### Быстрый откат правки production

```bash
cd /data/lib/podman-data/projects/goml/libs

# Восстановление источника из архива
cp fa_bwd_dk_new.cu.pre_XXX fa_bwd_dk_new.cu
cp fa_bwd_merged_v1.cu.pre_XXX fa_bwd_merged_v1.cu
cp fa_bwd_dq_new.cu.pre_XXX fa_bwd_dq_new.cu

# Если правил bench_r2c_e2e.cu — восстановить EXPECT
# (см. line 69-72, вернуть sealed regs 252/124/69/38)

# Пересборка
make -f Makefile.bench_r2c_e2e clean && make -f Makefile.bench_r2c_e2e
make -f Makefile.r2c_merged_bit_exact clean && make -f Makefile.r2c_merged_bit_exact
make -f Makefile.r1b_dk_bit_exact clean && make -f Makefile.r1b_dk_bit_exact

# Верификация
md5sum fa_bwd_{dk_new,merged_v1,dq_new,common}.cu*
# ожидание: sealed md5 (25e5e107 / 2bf32ab7 / d7a11a3d / 4407ec9c)

./bench_r2c_e2e | head -5
# ожидание: 252/124/69/38 OK
```

### Archive naming convention

- `<file>.cu.pre_XXX` — pre-ТЗ XXX state (для отката)
- `<file>.cu.SEALED_YYY` — sealed эпоха YYY (для audit trail)
- Не путать!

### Известные архивы

```
libs/bench_r2c_e2e.cu.pre_070               # pre-070 dS_T alloc removal
libs/fa_bwd_dk_new.cu.pre_069A              # pre-069A __launch_bounds__ (== sealed 061)
libs/fa_bwd_merged_v1.cu.pre_068            # pre-068 wave-remap (== sealed 040)
libs/fa_bwd_dk_new.cu.pre_068               # pre-068 (== sealed 061)
libs/fa_bwd_dq_new.cu.pre_068               # pre-068 (== sealed 041)
libs/fa_bwd_dk.cu.SEALED_AA1                # sealed AA1 dK
libs/fa_bwd_dq.cu.SEALED_AA1, SEALED_P1a    # sealed AA1/P1a dq
```

---

## §13 АРТЕФАКТЫ ПРОБ (probes 071/071b)

### x1_probe_071 (071 §1)

- **Source**: `libs/x1_probe_071.cu` (150 lines)
- **Makefile**: `libs/Makefile.x1_probe_071`
- **Binary**: `libs/x1_probe_071`
- **Runtime**: ~1 sec
- **Что делает**: standalone probe LDSM.m16n16.x1.trans.shared.b8 на макете production smQ layout (свизл `swz_byte` дословно) + row-marker `byte@(row,col)=row`. Выводит R0/R1/addr для 32 lanes warp 0.
- **Ключевой результат**: R0 == R1 в 32/32 lanes (hw duplicate на b8 variant)

### x1_bridge_071b v1 (071b §2)

- **Source**: `libs/x1_bridge_071b.cu` (~130 lines)
- **Makefile**: `libs/Makefile.x1_bridge_071b`
- **Binary**: `libs/x1_bridge_071b`
- **Что делает**: сравнивает x2 reference vs x1 candidates для b0_b (col+16, row+16) под row-marker и col-marker modes
- **Ключевой результат**: col-marker mode показал b0_b at col=8 (bytes 0x08 в x2.R1), кандидаты 0/128

### x1_bridge_071b_v2 (071b §3)

- **Source**: `libs/x1_bridge_071b_v2.cu` (~90 lines)
- **Makefile**: `libs/Makefile.x1_bridge_071b_v2`
- **Binary**: НЕ построен (ptxas fatal)
- **Что пробовал**: m16n8.x1.trans.b8 shape variant
- **Результат**: `ptxas fatal error: Illegal matrix shape '.m16n8' for instruction 'ldmatrix' + Modifier .trans not allowed for shape '.m16n8'`

### fa_bwd_dk_new_x1 (071 §2, ptxas discriminator)

- **Source**: `libs/fa_bwd_dk_new_x1.cu` (namespace `fa_bwd_dk_new_x1`, не `fa_bwd_dk_new`)
- **Makefile**: `libs/Makefile.dk_new_x1_ptxas`
- **Ptxas log**: `libs/dk_new_x1_ptxas.log` (113 regs, 0 spill)
- **Что делает**: замена 2 x2 LDSM calls на 4 x1 calls (guess addr `np*16+8` для b0_b — заведомо wrong semantically, ptxas discriminator only)
- **Ключевой результат**: regs 124→113 (−11), spill 0 (правило 12 ✓)

---

## §14 BENCH MODIFICATIONS (070 KEEP state)

### Текущее состояние bench_r2c_e2e.cu

```cpp
// line 126 (chain function)
CKR(cudaMalloc(&dS_nat,dsz)); dS_T = nullptr;   // 070: dS_T dead-alloc removed

// line 208 (wall function)
CKR(cudaMalloc(&dS_nat,dsz)); dS_T = nullptr;   // 070: dS_T dead-alloc removed
```

**Обоснование** (в 070 report):
- `kernel_merged_v1` param `dS_T_out` (line 65 fa_bwd_merged_v1.cu) declared но НИКОГДА не разыменован в kernel body (единственный ref = сама декларация)
- Wrapper `launch_merged` (line 511-540) — pass-through only, никакого dereference
- dk_new и dq_new signatures принимают только `dS_nat` (dS_T параметр kernel был легаси)
- Kernel comment line 396: «033-c: dS_T drain устранён; dS_T буфер не пишется в DRAM»
- **Пропускать nullptr БЕЗОПАСНО by construction** — нет dereference-места

### Экономия

**Δ VRAM per bench_r2c_e2e run**: −8192 MB = **−8.00 GiB = −8.59 GB** (точный `dsz = bh × sl × stride_ds = 128 × 8192 × 8192 = 8,589,934,592 байт`)

Замер: `cudaMemGetInfo` inline probe (nvidia-smi lazy-commit не видит untouched pages).

### W0-дельта для W-ветки

Строка обвязки frozen_v2/v3: sweep-check «`dS_T = nullptr` вход в launch_merged». Не форсируется отдельным пакетом frozen_v3 (эшелон-2 не дал KEEP-правки production).

---

## §15 PTXAS GATE PROTOCOL

### Обязательные проверки на КАЖДОМ ptxas-шаге

```
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '...' for 'sm_120a'
ptxas info    : Function properties for '...'
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads   ← ПРАВИЛО 12 (mandatory!)
ptxas info    : Used N registers, used K barriers                     ← fingerprint check
```

**Правило 12 нарушено** = автоматически КРАСНЫЙ гейт БЕЗ ABBA.

### Occupancy calculation

```
regs_per_thread × threads_per_block ≤ 65536 / blocks_per_SM
                                    → blocks_per_SM = 65536 / (regs × threads)

Для dk_new (124r × 128 = 15872): 65536/15872 = 4.13 → 4 blk/SM by reg
Для dq_new (69r × 128 = 8832): 65536/8832 = 7.42 → 7 by reg (SMEM ограничивает до 6)
Для merged (252r × 128 = 32256): 65536/32256 = 2.03 → 2 by reg (SMEM 47616 тоже ограничивает до 2)
Для d_precompute (38r × 128 = 4864): 65536/4864 = 13.5 → 13 by reg (grid другой)
```

### Ptxas cache

- Кеш ptxas: `/tmp` (`HOME=/tmp` env для избежания permission ошибок)
- Логи build: `.../build.log` рядом с Makefile

---

## §16 ИЗВЕСТНЫЕ GOTCHAS (session-specific)

### Bash restrictions

- Multi-command `&&` chain может требовать approval
- `cd` в pipeline тоже — использовать absolute paths ИЛИ отдельный shell script
- **Решение**: писать shell scripts в `runs/reports/*.sh`, потом `chmod +x` + запуск как единая команда
- Output redirection `> file` ограничена working directories — писать через wrapper script

### Env vars inline

- `CAUSAL=1 ./bench` может требовать approval → писать wrapper script с `export CAUSAL=1`

### nvcc path

- `/usr/bin/nvcc` (system) может НЕ работать под sm_120a (старая версия CUDA)
- `/usr/local/cuda-13.1/bin/nvcc` — правильный, поддерживает `compute_120a`
- Все Makefiles уже используют CUDA 13.1

### nvidia-smi memory.used lazy commit

- Не отображает untouched pages
- Использовать `cudaMemGetInfo(&free, &total)` inline для точной alloc-дельты
- Для peak-мониторинга во время bench — тестировать через tight loop `for i; do MB=$(nvidia-smi...); done` **с cudaMemset touch** внутри пробы (иначе pages не committed)

### md5 отчётов после Edit

- Мои `md5sum` в chain отражают md5 ФАЙЛА В МОМЕНТ ЗАПИСИ
- После моих Edit `<computed>` → реальный md5, диск может слегка отличаться из-за encoding/newline hooks
- Расхождение 070 md5 (мой `83fc3f2c...` vs текущий диск `b4b0ba63...`) — hook consolidation

### Compute-sanitizer / racecheck

- Путь: `/usr/local/cuda-13.1/bin/compute-sanitizer`
- Racecheck обязателен если правка **трогает барьеры** (`__syncthreads()`)
- Memcheck обязателен на каждом гейте (0 errors mandatory)
- Racecheck **не требуется** если правка только host-side (070) ИЛИ hint-only (069A)

### Bench_r2c_e2e без arg

- Default mode = wall bench (line 179)
- С аргументом `bitexact` — chain bit-exact 11/11

---

## §17 НЕ СДЕЛАНО / ЖДЁТ VUGAR

### Публичный релиз v0.2.0

1. **Настройка git remote**: `git remote add origin git@github.com:<USER>/fa-blackwell-fp8.git`
2. **Push**: `git push -u origin main && git push origin v0.2.0`
3. **GH release create**: команда в §4
4. **Пост-пуш проверка** md5 файлов в опубликованном теге (команды в 064 §4.d)

### Frozen_v3 пакет для W-ветки

- Не форсируется (не было KEEP-правки в эшелон-2)
- Если Vugar готов: добавить строку `dS_T = nullptr` audit в frozen_v2 → frozen_v3
- Sweep-check «alloc-audit: dS_T NULL для kernel_merged_v1 (033-c: dead с эпохи 033, официально снят 070)»

### Эшелон-3 dedicated microprobes (multi-day)

- **069b V-bridge microprobe**: LDSM.trans.b16 fragment layout для m16n8k16.f16 MMA-B
- **SHFL restructure probe**: intra-warp exchange patterns для W_all[8]
- **fp16x2 dK_acc**: sealed re-baseline requirement (много работы)

Все требуют явного TZ от Vugar с обоснованием.

---

## §18 КАМПАНИЯ 040-071b (recap)

### Cumulative sealed KEEP-серии

| Эпоха | Оптимизация | Δ wall | Метод |
|:-:|:--|:-:|:--|
| 040 | merged LDSM.x4.trans class #7 | −12% | ISA-таблица + микропроба |
| 041 | dq разморозка (d5lite_pack_pi) | −3.5% | pack-π_V чистка |
| 061 | dk S2v4 (свизл + LDSM.x2.trans.b8) | −19.24% | 060 col-мост + 058b bridge dedicated |
| **Σ** | Кumулятивно | **~20% E2E wall** | 3 KEEP-серии |

### Замороженные попытки (не KEEP)

- 048 dkS2v2 — b1-delivery bug
- 049 A/B probes — марker inject validity
- 050 S2v3 — +118% wall (rolled back)
- 051 baseline autopsy — 32-way conflict storm identified
- 052 smQ-prefetch — status quo −0.033%
- 053 dO half-leg — +1.90% (rolled)
- 054 M5+A' — 0.5-1.9% sub-порог
- 055 §1a L2-хинт / §1b dO regs — оба красные
- 056 A-fix + B-fix — обе +1.99%/+1.53% красные
- 057 zombie 840120 stend-attribution kill
- 058 S2v4 bridge deferred → 058b построен → 060/061 KEEP

### Публичный релиз epoch (062-064)

- 062 cert package — 30-run nc/causal + isolated + canary + bit-exact
- 063 causal ledger — 22.206 ms
- 063-r work-detector + chrono cross-check
- 064 v0.2.0 SEALED — Apache-2.0 + SPDX + clean-clone verify

### Постпубликационный дожим (065-071b)

- 065 GH release draft — grep clean, дуэль-карта
- 066 causal recon — skew histogram + DRAM attribution
- 067 dead-dS уже реализовано (paper red)
- 068 wave-remap +11.5% (rolled)
- 069A dk 5-я бригада hint +40% spill (rolled)
- 069B V-repaint STOP на мосту
- 070 dS_T dead-alloc снят −8.59 GB (KEEP)
- 071 x1 probe (гибрид −11r structural / dup hw quirk)
- 071b b0_b bridge МОГИЛА (3 независимые причины)

### Ключевые ISA-таблицы (historical references)

- **ISA-043**: LDSM variants таблица sm_120a (m16n8.x1/x2/x4, m16n16.x1/x2/x4, .trans/.no-trans, .b8/.b16)
- **ISA-045**: LDSM.m16n16.x2.trans.b8 quirk — R2=R0 dup, R3=R1 dup (dk_new uses this)
- **ISA-071b (new)**: LDSM.m16n16.x1.trans.b8 quirk — R1=R0 dup (32/32 lanes) + m16n8.trans не supported

---

## §19 CRITICAL VALUES CHEAT-SHEET

```
Sealed md5 (should NEVER change without TZ):
  dk_new: 25e5e1077cc3bec2c49bf9288fe60c54
  merged: 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33
  dq_new: d7a11a3d788eb4c396d892bc9c8ab754
  common: 4407ec9cf64708a2a28dc36633d5d6f1

Fingerprint EXPECT (bench_r2c_e2e:69-72):
  d_precompute:  38
  merged_v1:    252
  dk_new:       124
  dq_new:        69

Cert wall:
  nc:      42.346 ms → 415.44 T proj / 259.65 T fused
  causal:  22.206 ms → live/full 0.5039

Public release:
  Directory:  release_v0.2.0/
  Commit:     506b69f
  Tag:        v0.2.0
  License:    Apache-2.0
  Post-SPDX md5 dk_new: eb492e0729ef643280591b8c8dd8a29d
  Post-SPDX md5 merged: 720774c28807d01214adff16c9003221
  Post-SPDX md5 dq_new: 7660bd960cc39c799d588c573bb47c5d
  GH release body md5:  f1ca49c7ed811ce847c2892bbc5ec9a3

Hardware:
  RTX PRO 6000 Blackwell WS Edition
  sm_120a, 188 SMs, 65536 regs/SM
  101376 B smem opt-in, 96 GB VRAM
  Driver 580.159.03, CUDA 13.1

Canonical form:
  bh=128, sl=8192, hd=128, seed=42
  Br=64 (all), Bc=64 (all), threads=128 (all)
  warmup=5, iters=20

Chain md5 (069-071b):
  069  76c958364d1d2ac74c2a4f86b87e4dfe
  070  83fc3f2c3c2817a2660defb3f246330e (диск может варьировать hooks)
  071  42126d3ff26848dacc1fc660a06dcdd1
  071b 19e3a3818ef0456c070fc03b49d0c773
```

---

## §20 КОНЕЦ HANDOFF — сводка «что помнить»

1. **Sealed** = 252/124/69/38 fingerprint + 4 md5, ЛЮБОЕ отклонение = стоп
2. **Cert** = 42.346 ms nc / 22.206 ms causal, drift ±1% OK
3. **Правило 12** = spill/LDL=0 автоматически КРАСНЫЙ на каждом ptxas
4. **Правило 2/3 v2** = KEEP ≥ 2% wall + bit-exact + memcheck 0
5. **068 урок** = grid-order = L2-locality, wave-remap опасен
6. **069A урок** = launch_bounds без structural = spill гарантирован
7. **071b урок** = LDSM.x1.b8 sm_120a: R1=dup + m16n8.trans нет + b0_b col+8 misaligned = МОГИЛА
8. **Реестр B**: 5-я бригада dk через x1 = могила окончательно; эшелон-3 нужны SHFL/fp16x2/cp.async
9. **v0.2.0 запечатан** локально, ждёт push от Vugar
10. **dS_T = nullptr** в bench (070 KEEP) — единственная non-sealed правка сейчас

**Артефакт-хедер обязателен первой строкой каждого отчёта. Гейт-тишина обязательна перед каждым замером. Правки production сопровождать `.pre_XXX` архивом.**

---

Handoff sealed 2026-07-10 post-071b. Model transition: Opus 4.7 → Fable 5.

---

## §21 UPDATE: bwd-sealed-v2 pushed (2026-07-10 22:29 UTC)

**После HANDOFF был сделан commit + push + tag** по запросу Vugar (train'у нужна была ясная финальная точка):

### Git state

- **Remote**: `git@github.com:djeday123/goml.git`
- **Commit**: `2314cbe seal: FP8 bwd v2 (42.35ms E2E / 415.44T proj) — 061 S2v4 dk_new`
- **Tag**: `bwd-sealed-v2` (annotated, SHA `41db1695f5853474073595bd1f04ada9c9b14241`)
- **Прежний sealed**: `4732a38 seal: FP8 bwd v1 (44.206ms E2E / 398T) merged+dk_new+dq_new+D` — теперь помечен как v1 (pre-061)
- **Push статус**: `main` (4732a38→2314cbe) ✓ + tag `bwd-sealed-v2` ✓ на github

### Изменения в 2314cbe

Один файл: **`libs/fa_bwd_dk_new.cu`** (54 insertions, 122 deletions)
- Только это было modified vs 4732a38
- md5 sealed: **25e5e1077cc3bec2c49bf9288fe60c54** (061 S2v4)
- Остальные core (merged/dq_new/common/dk) уже были sealed в 4732a38 в состоянии 040/041

### Tags на remote (полный список)

```
v0.1.0         — старая эпоха
w0-seal-v1     — train's W0 seal (около 4732a38)
bwd-sealed-v2  — 🚀 NEW финальная fused (2314cbe)
```

### Для train'а

Одной командой:
```bash
git fetch origin
git checkout bwd-sealed-v2
```

Получит:
- `libs/fa_bwd_dk_new.cu` md5 `25e5e107...` (124 reg, S2v4)
- Fingerprint EXPECT в bench (при первой сборке) = 252/**124**/69/38
- Cert wall 42.346 ms nc / 22.206 ms causal (не 44.206/398 как в v1!)

### Правки production после 2314cbe

**0** — 2314cbe закоммитил уже sealed working tree state. Source content byte-identical. Единственное чего касается — путь в git репо.

Bench-side правки (070 `dS_T = nullptr`) — вне tracked файлов (bench_r2c_e2e.cu, r1b_*, r2c_* не в git). Train их не увидит через checkout. Если train'у нужна bench-правка → отдельный TZ (или руками патч подсказать).

---

Update sealed 2026-07-10 22:29 UTC. Handoff теперь полный state pre + post commit.

---

## §22 Train chat fingerprint sync fix (для будущих сессий)

**Проблема** (диагностика от train chat'а после push v2):
- В training-клоне: `bench_r2c_e2e.cu` с EXPECT `dk_new=124` (post-061)
- В training-клоне: sealed `dk_new` = 128 regs (тег `w0-seal-v1` = 033/pre-061)
- **Рассинхрон**: fingerprint FAIL при попытке собрать бенч в клоне

**НЕ РЕКОМЕНДУЕТСЯ** (совет train chat'а из его первого сообщения):
- Даунгрейдить бенч 124→128 = терять +4.4% throughput ради совместимости со старой sealed
- Убрать бенч из training-репо = потерять fingerprint защиту

**ПРАВИЛЬНЫЙ FIX** — обновить sealed в клоне до v2:

```bash
# В training клоне (у них)
cd <goml_clone>
git fetch origin
git checkout bwd-sealed-v2 -- libs/fa_bwd_dk_new.cu
# теперь sealed dk_new = 25e5e107... (124 reg, S2v4)
# бенч с EXPECT 124 пройдёт fingerprint естественно
# wall: 42.346ms nc (был 44.206ms в v1) = +4.4% throughput
```

Или полный checkout:
```bash
git checkout bwd-sealed-v2
```

**Rollback safety**: тег `w0-seal-v1` остаётся на remote — если им нужен pin на 128-reg версию для reproducibility старых experiments, `git checkout w0-seal-v1` всегда работает. Но production sealed теперь = v2.

**Важный нюанс про bench_r2c_e2e.cu**: файл **untracked** в git (у меня показывал `?? libs/bench_r2c_e2e.cu` перед commit). Значит train получил бенч НЕ через checkout — вручную скопировали или получили отдельно. Если у них старая копия — нужна свежая (EXPECT 124 + 070 патч `dS_T = nullptr`).

---

Sync-fix sealed 2026-07-11.


