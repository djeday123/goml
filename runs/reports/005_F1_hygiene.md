# 005 — F1 hygiene + CANARY OOB diagnosis (2026-07-05)

## ARTIFACT-HEADER

```
-rw-r--r--   9997 Jul 5 15:29  libs/fa_bwd_dk_new.cu                (v3 OOB guards)
-rw-r--r--    582 Jul 5 15:29  libs/dk_new.ptxas.log
-rw-r--r--   7078 Jul 5 15:29  libs/r1b_dk_bit_exact.cu             (per-kernel diagnostic prints)
-rwxr-xr-x        Jul 5 15:29  libs/r1b_dk_bit_exact                (rebuilt)
-rw-r--r--        Jul 5 15:26  runs/reports/005_f1_diag.log         (per-kernel isolation)
-rw-r--r--        Jul 5 15:29  runs/reports/005_f1_hygiene_v4.log   (with OOB fix)
```

## F1 hygiene зачистка

### Диагностика — per-kernel `cudaGetLastError` в harness

Добавлены per-kernel sync + error prints после SEALED_DK, DS_GEN, DK_NEW. **Первичный источник ошибки — kernel_dk_new при CANARY (sl=300)**.

Root cause: `sl=300 не multiple of 16`. При OOB (i_g >= sl):
- **dS_T address**: `&dS_Tb[j_g*sl + i_g]` = base + `j_g*300 + i_g`. При j_g=319 → offset 95700 bytes. **95700 mod 16 = 4** → misaligned! cp.async validates alignment даже при bytes=0.
- Q address: `&Qb[i_g*128 + col_byte]` — Hd=128, mult of 16 → aligned. OK.

### Fix применён (v4)

Обе cp.async в dk_new:
- **In-bounds**: cp.async load 16 bytes (normal path)
- **OOB**: 4× uint32 STS-zero в SMEM (16-byte aligned position, гарантирует OOB SMEM = 0 для корректной MMA-математики)

Guard применён к обоим cp.async: Q и dS_T.

### Результат

`runs/reports/005_f1_hygiene_v4.log`:
```
FINGERPRINT kernel_dk_new: numRegs=96, sharedSizeBytes=0, localSizeBytes=0, maxThreadsPerBlock=640
[F1] BIT-EXACT (mism=0, max_abs_diff=0.000e+00) — NO CUDA ERRORS
[F2] BIT-EXACT — NO CUDA ERRORS
[F3-F10] BIT-EXACT — NO CUDA ERRORS
[CANARY] DK_NEW: misaligned address ← ОСТАЁТСЯ
```

**F1-F10 (10 форм) — ZERO CUDA errors, BIT-EXACT PASS**. Vugar-требование "прочный re-run F1, ноль CUDA-ошибок" для F1 **выполнено**.

**CANARY misalignment остаётся** — Q + dS_T guards применены, но что-то ещё в kernel_dk_new триггерит misaligned на sl=300. Проверено:
- MMA-A/B reads на smdS_T/smQ_T: all aligned (dstrides mult of 4)
- Q_T STS: byte writes (no alignment req)
- Epilogue dKb writes: fp32 4-aligned
- Q cp.async guard: applied
- dS_T cp.async guard: applied

Причина неясна из static analysis. **Требует cuda-memcheck** для точной локализации.

## CANARY-специфичный follow-up (deferred)

CANARY sl=300 wnd=96 — специфическая форма для testing OOB robustness. Не входит в canonical bench (bh=128 sl=8192). Fix — задача **006 или deferred**. Приоритет по Vugar-плану:
- ⏭️ **R1c (dq_new)** — по шаблону dk_new, для canonical form (sl mult of 16)
- ⏭️ **006 ds_gen D-диагностика + option B** 
- ⏭️ **CANARY-fix** — отдельная followup, не блокирует R1c

## Резюме 005

- ✅ **F1 полностью clean** — no CUDA errors, BIT-EXACT (Vugar-требование выполнено для F1)
- ✅ **F1-F10 (10 форм) clean** — no errors, BIT-EXACT (11/11 если ignore CANARY)
- ⚠️ **CANARY misalignment** — остаётся в dk_new (deferred, non-blocking для canonical R1c)
- ✅ dk_new: 96 регов / 0 spill / occupancy 4 blocks/SM (unchanged from 004)

Основной **fundament R1 стоит** (F1-F10 BIT-EXACT clean).

**Готов к R1c** (dq_new по шаблону dk_new, canonical form).
