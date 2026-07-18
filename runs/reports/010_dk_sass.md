# 010 — SASS-аудит dk_new + dq_new (нулевая стоимость, measure-only) (2026-07-06)

## ARTIFACT-HEADER — cross-verification

### md5 009_ncu.md (проверка существования и содержания)
```
62977f103b01436c877c56d4b7a48a4a  runs/reports/009_ncu.md   (11748 bytes)
```

### ls -la runs/reports/ (полный)
```
total 2944 (state 2026-07-06 09:27)
-rw-r--r--   6916 Jul 5 11:17  001_R1a.md
-rw-r--r--   1061 Jul 5 11:13  001_r1a_bit_exact.log
-rw-r--r--   6821 Jul 5 11:16  001_r1a_ncu.txt
-rw-r--r--   1235 Jul 5 11:15  001_r1a_wall.log
-rw-r--r--   5016 Jul 5 11:24  002_R1a_reconcile.md
-rw-r--r--   1235 Jul 5 11:23  002_r1a_canonical_wall.log
-rw-r--r--   7820 Jul 5 14:12  003_R1b.md
-rw-r--r--   1444 Jul 5 14:05  003_r1b_bit_exact.log
-rw-r--r--   1088 Jul 5 14:09  003_r1b_dk_bit_exact.log
-rw-r--r--   4521 Jul 5 14:07  003_r1b_dram_ncu.csv
-rw-r--r--   1235 Jul 5 14:06  003_r1b_wall_dual.log
-rw-r--r--   6672 Jul 5 15:12  004_R1b_fix.md
-rw-r--r--   1054 Jul 5 15:08  004_r1b_fix_bit_exact.log
-rw-r--r--   6716 Jul 5 15:11  004_r1b_fix_dk_ncu.txt
-rw-r--r--   1115 Jul 5 15:10  004_r1b_fix_dk_wall.log
-rw-r--r--   3720 Jul 5 15:58  005_F1_hygiene.md
-rw-r--r--   5731 Jul 5 16:16  005a_canary_fix.md
-rw-r--r--   8602 Jul 5 17:58  005b_R1c.md
-rw-r--r--   1652 Jul 5 17:57  005b_R1c_dq_ncu.txt
-rw-r--r--   7141 Jul 5 21:11  006_baseline.md
-rw-r--r--   8169 Jul 5 22:35  006_dsgen_fix.md
-rw-r--r--   2298 Jul 5 22:34  006_II_ncu.log
-rw-r--r--   8324 Jul 5 23:30  007_R1_e2e.md
-rw-r--r--   9320 Jul 6 07:21  008_R2C.md
-rw-r--r--  11748 Jul 6 09:26  009_ncu.md                       ← Vugar-верификация
-rwxr-xr-x    938 Jul 6 08:26  009_dram_dedup.sh
-rwxr-xr-x    387 Jul 6 08:44  009_dram_split.sh
-rwxr-xr-x   1432 Jul 6 08:27  009_ncu_dk_dq.sh
-rwxr-xr-x    946 Jul 6 08:25  009_ncu_merged_detail.sh
-rwxr-xr-x    241 Jul 6 08:24  009_ncu_merged.sh
-rw-r--r--  1226067 Jul 6 09:27  010_dk_new_sass_full.txt        (bench_dk_wall дамп)
-rw-r--r--  1258723 Jul 6 09:27  010_dq_new_sass_full.txt        (bench_dq_wall дамп)
-rwxr-xr-x    624 Jul 6 09:27  010_sass_full.sh                  (cuobjdump 13.1)
-rwxr-xr-x   2120 Jul 6 09:26  010_sass_probe.sh
… (более ранние shell-скрипты и логи)
```

## Tools версия
```
cuobjdump: CUDA 13.1 (2005-2025 NVIDIA)
nvdisasm: CUDA 13.1 (2005-2025 NVIDIA)
```

## 010.1 — dk_new SASS-таблица (kernel isolated, строки 8999–10251 dk_full)

### Инструкционный профиль (per-invocation, unrolled qt-body)
| Op | Count | Значение |
|----|:-----:|----------|
| **QMMA.16832.F32.E4M3.E4M3** | **32** | main MMA (m16n8k32 fp8→fp32) = KB_DK=2 × NI_DK=16 fully unrolled per qt |
| **LDS** (32-bit default) | **94** (из них 6 dead `@!PT LDS RZ`) → **88 real** | LDS.32 4-byte reads |
| **LDS.U16** | **0** | 16-bit reads НЕ используются |
| **LDS.64/.128** | **0** | wider LDS не эмитятся компилятором |
| **LDSM** (ldmatrix) | **0** | **ldmatrix НЕ используется** — прямые LDS.32 |
| STS.U8 | 64 | per-byte STS для Q_T scatter (byte-level transpose) |
| STS.128 | 2 | 16-byte STS |
| LDGSTS.E.BYPASS.128 | 2 | cp.async 16-byte (Q + dS_T) |
| STG.E | 64 | dK epilogue fp32 stores |
| STG.E.U8 | 0 | нет byte-writes в dk_new |
| FMUL | 64 | dK_acc × scale эпилог |
| CS2R (spill workspace) | 32 | нет spill но CS2R = clock/state to reg |

### Ключевые числа
- **LDS.32 / QMMA ratio = 88 / 32 = 2.75** (внутри qt-body)
- **Distribution 88 LDS.32 per qt-iter**:
  - 16 LDS.32 = Q into Qr regs (для Q_T transpose STS)
  - 8 LDS.32 = A operand reads (dS_T, 4 A × KB_DK=2)
  - 64 LDS.32 = B operand reads (Q_T, 16 ni × 2 B × KB_DK=2)
- Runtime × n_qt=128 iters × 128 blocks/SM (from 8192 blocks / n_kt=128) = 128 * 88 = **11264 real LDS.32 per block launch**
- Grid = 16384 blocks × 4 warps × 11264 LDS.32 = **738M LDS.32 warp-instructions per launch** → factually LSU wavefronts 3.28B (SASS не считает all instances, только static)

### Wavefronts excessive (из 009 NCu)
- LSU wavefronts: **3.28 B**
- Bank conflicts LD: **1.69 B**
- **Excessive rate LD: 1.69 / 3.28 = 51.5 %** — **КАЖДЫЙ ВТОРОЙ wavefront получает конфликт**
- Это НЕ ноль (Vugar-предположение о ds_gen 006 =2% на dk_new **не переносится**)

## 010.2 — dq_new SASS-таблица (kernel isolated, строки 4082–5553)

### Инструкционный профиль
| Op | Count | Значение |
|----|:-----:|----------|
| **QMMA.16832.F16.E4M3.E4M3** | **32** | main MMA (fp8→fp16-acc, AA1-style, packed dQ_acc) |
| LDS (32-bit) | 97 (−6 dead) → 91 real | LDS.32 4-byte reads |
| LDS.U16 | 0 | не используется |
| LDS.64/.128 | 0 | не эмитятся |
| LDSM | **0** | ldmatrix НЕ используется |
| STS.U8 | 65 | K_T byte-scatter transpose |
| STG.E | 64 | dQ epilogue |
| FMUL | 64 | dQ_acc × scale |
| HADD2.F32 | 64 | dQ_acc packed fp16 → fp32 unpack |

### Wavefronts excessive (из 009 NCu)
- LSU wavefronts: 2.10 B
- Bank conflicts LD: 541 M
- **Excessive rate LD: 25.8 %** — половина от dk_new, но всё ещё существенная

## 010.3 — Retention/wider-LDS analysis (paper)

### Что можно retention'ить в dk_new
Loop-структура dk_new:
```
for qt: [Q, dS_T меняются]
  cp.async Q → smQ; cp.async dS_T → smdS_T
  LDS Q → Qr regs; STS.U8 Q_T scatter → smQ_T
  for kb (=2): [dS_T k_j strip]
    LDS A0..A3 (dS_T)
    for ni (=16): [n_d strip Q_T]
      LDS B0, B1 (Q_T)
      QMMA(...)
```

- **Q cross-qt retention**: **невозможно** (Q per-qt меняется)
- **Q_T cross-kb retention within qt**: **потенциал −32 LDS.32/qt** (полная передача Q_T в 64 uint32 per lane — killers регистров, отпадает)
- **Q_T per-ni retention**: current loop уже так — ni inner reads B per ni, но B шейрится по kb → нет
- **A cross-ni retention**: **уже так** (A loaded per kb outer, reused 16 ni) — оптимально
- **Loop reorder (ni outer, kb inner)**: сокращает B (32→16 ni=1), но множит A (8→128) — **регрессия, отпадает**

### Ldmatrix (без .trans) — единственный вектор
- Компилятор НЕ генерирует LDSM для fp8-layouts даже на sm_120a (LDSM.M16N16 доступен для 8-bit, но по-разному для конкретных layout'ов).
- **Q_T layout после STS.U8-transpose**: natural row-major [n_d][k_i] с stride QT_STRIDE=68. Для LDSM.M16N16.b8 требуется 16-aligned row start — **stride 68 не подходит** (68 mod 16 = 4).
- **dS_T layout**: [j_local][i_local] stride Br=64 = 16-aligned ✓ (в 006 003a: stride_ds на global, но smdS_T использует Br). LDSM.M16N16 для fp8 A operand возможен, если layout соответствует.
- **PTX доступность**: `ldmatrix.sync.aligned.m8n16.x4.shared.b8` — Blackwell sm_120 поддерживает (документация pt xas 13.x — надо проверить).

### Bank-conflicts source (диагноз)
- Bank conflicts LD 51.5 % dk_new — **B3 padding пересмотр НЕ мёртв** (Vugar killed на ds_gen numbers, где 7.5%; в dk_new 51.5% — совсем другая история).
- Гипотеза: **stride QT_STRIDE=68 даёт частые конфликты при MMA-B чтении**. Проверка: `n_d = ni*8 + l_div4` (lane-разброс по 8), `k_i_lo/hi = kb*32 + l_mod4*4` (lane-разброс по 16). Адрес = n_d * 68 + k_i. Lane-адреса — надо считать bank hits.

## 010.4 — Правки в план 009-1

### Убито/reopened
- **B3 padding пересмотр** — **REOPENED** (для dk_new): 51.5% conflicts factually. Требует bank-pattern-анализа для QT_STRIDE.
- **B1 ldmatrix.trans** — остаётся мёртвым (fp8 layouts + .trans на sm_120a не работает).
- **Ldmatrix без .trans** — жив: `ldmatrix.sync.m8n16.b8` для A operand (dS_T), если stride позволяет. Требует PTX-inline проверки.
- **Q retention** — мёртв (Q per-qt, LSU-нейтрально).

### Три бесплатных probing-candidate (в порядке предпочтения)
1. **B3 padding** для smQ_T (QT_STRIDE=68 → tested 72/76/80). Bank-pattern-анализ на бумаге + ptxas-first. Регресс blocks/SM = откат. **Стоимость правки: 1 строка** (`#define FA_DKN_QT_STRIDE 72`).
2. **Ldmatrix без .trans** для A operand (dS_T). PTX-inline вариант mma-A load. Bit-exact-риск: layout меняется, но байты at (M, K) те же.
3. **B3 padding** для smdS_T (Br=64 → 72/80). Br=64 already 16-aligned, но bank-pattern-tuning.

Direction (2) даёт наибольший потенциал (LDSM = 1 warp inst против 6 LDS.32 = ×6 reduction на A path).

## 010.5 — Answer to Vugar's спецификация step-0

| Vugar-запрос | Факт |
|--------------|------|
| **SASS dk_new счёт LDS.32/.64/.128, LDSM** | LDS: 88 real; LDS.64/.128: 0; LDSM: 0 |
| ldmatrix присутствует | **НЕТ** (0 LDSM in both kernels) |
| memory_l1_wavefronts_shared_excessive | **51.5% dk_new / 25.8% dq_new** (SIGNIFICANT — B3 REOPEN) |
| retention-решение (что перегружается) | Q_T (64 LDS.32/qt = 72% of MMA-B reads); полное retention отпадает (64 uint32/lane убьёт регистры); B3-fix для reducing conflict rate — **проще** |
| SASS dq_new тот же дамп | done — 32 QMMA.16832.F16 (fp16-acc), 91 LDS real, 0 LDSM |
| Симметричная картина MIO | dq_new тот же паттерн (LDS.32 only, no LDSM), но conflicts вдвое меньше |

## Резюме 010

- ✅ **SASS-аудит dk_new+dq_new (нулевая стоимость)** сделан, artifact-header + md5 внешне-верифицируем 009_ncu.md
- ✅ **ldmatrix НЕ используется** — компилятор дефолтит к LDS.32
- ✅ **Bank conflicts LD в dk_new — 51.5%** (excessive); в dq_new — 25.8% (moderate)
- ⚠️ **B3 padding пересмотр — reopened** для dk_new (Vugar killed на ds_gen 7.5%; dk_new 51.5% = другая картина)
- ⚠️ **Ldmatrix (без .trans) для A operand (dS_T)** — жив как probing-candidate; PTX-inline нужен
- **Q_T layout stride=68** — вероятный источник bank conflicts (n_d * 68 + k_i address pattern)

### Рекомендация к 009-1 (обновлённая)
**Шаг 1a**: **B3 padding probe для QT_STRIDE в dk_new** — самая дешёвая (1 строка изменения), самая безопасная (никакой PTX), самый вероятный wall-эффект через уменьшение excessive wavefronts.
Гейт: ptxas regs+blocks ДО замера. Регресс blocks (4→3) = откат.

Если QT_STRIDE не бьёт conflicts достаточно — **шаг 1b: PTX-inline ldmatrix.m8n16.b8** для A operand.

Жду решения: **B3-QT_STRIDE-probe** ИЛИ **ldmatrix-b8-probe** — одно, не оба сразу.
