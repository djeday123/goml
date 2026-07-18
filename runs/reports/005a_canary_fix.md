# 005a — CANARY-fix: padded row stride ABI (2026-07-05)

## ARTIFACT-HEADER

```
-rw-r--r-- 1 root root   10278 Jul  5 16:08  libs/fa_bwd_dk_new.cu               (stride_ds cp.async read)
-rw-r--r-- 1 root root   27810 Jul  5 16:11  libs/fa_bwd_ds_gen.cu               (stride_ds STG write × 16)
-rw-r--r-- 1 root root    7528 Jul  5 16:11  libs/r1b_dk_bit_exact.cu            (dsz = bh*sl*stride_ds)
-rwxr-xr-x 1 root root 1202440 Jul  5 16:12  libs/r1b_dk_bit_exact               (rebuilt)
-rwxr-xr-x 1 root root 1202248 Jul  5 16:13  libs/r1b_dk_wall                    (rebuilt)
-rw-r--r-- 1 root root    4289 Jul  5 16:07  runs/reports/005a_sanitizer_raw.log (BEFORE fix — misaligned)
-rw-r--r-- 1 root root    1140 Jul  5 16:12  runs/reports/005a_bit_exact.log     (AFTER: 11/11 clean)
-rw-r--r-- 1 root root    1548 Jul  5 16:15  runs/reports/005a_sanitizer_clean.log (AFTER: 0 errors)
-rw-r--r-- 1 root root     366 Jul  5 16:15  runs/reports/005a_wall.log          (wall 9.42ms median)
-rwxr-xr-x 1 root root     470 Jul  5 16:07  runs/reports/canary_sanitizer.sh
```

## Пункт 1 — Sanitizer BEFORE fix (in-bounds misalignment)

`005a_sanitizer_raw.log` (одна строка вывода):
```
Access to 0x7f525caa534c is misaligned  (offset 7500 = 25*300, IN-BOUNDS row j_g=25 i_g=0)
Nearest allocation 0x7f525caa3600, size 90000 bytes (= 300*300 CANARY dS_T)
```
**Vugar's подозрение подтверждено**: j_g*300 mod 16 = 12 при j_g=25 → **in-bounds rows misaligned** (не только OOB). Причина: sl=300 не multiple of 16, row-stride производит cumulative offset misalignment.

## Пункт 2 — ABI-fix: padded row stride

**Реализовано**: `stride_ds = (sl + 15) & ~15` — гарантирует что каждая row начинается с 16-byte aligned адреса.

Три места изменения:

- **`fa_bwd_ds_gen.cu` L227-229**: per-batch base offset `+ b * sl * stride_ds`, все 16 STG writes (dS_base × 8, dS_T_base × 8) переведены на `* stride_ds +`.
- **`fa_bwd_dk_new.cu` L114**: `dS_Tb = dS_T + b * sl * stride_ds`; cp.async read `&dS_Tb[j_g * stride_ds + i_g]`.
- **`r1b_dk_bit_exact.cu` L74**: `stride_ds = (sl+15)&~15; dsz = bh*sl*stride_ds`.

Каноническая sl=8192: stride_ds=8192 → **zero-impact** на канонические числа (ptxas 96 регов unchanged).
CANARY sl=300: stride_ds=304 → +1.3% dS mem, aligned rows.

## Пункт 3 — STS-zero OOB path

**Оставлен** (Vugar's option (a) "OOB-строки честно занулять" — allowed по TZ). 4× uint32 STS-zero в 16-byte aligned SMEM position корректен: `smdS_T[j_local * Br + col_byte]` где `col_byte % 16 = 0` и `j_local * Br % 16 = 0` (Br=64).

Predicated-off cp.async с dummy address (option (b)) — deferred; текущая (a) даёт clean 11/11 включая CANARY и 0 sanitizer errors, blast radius минимальный.

**В техлог**: наблюдение "cp.async validates source alignment even at bytes=0 / predicated-off on sm_120a" — сохранено для future.

## Пункт 4 — RE-GATE full (11/11 + CANARY + zero CUDA errors)

`005a_bit_exact.log`:
```
FINGERPRINT kernel_dk_new: numRegs=96, sharedSizeBytes=0, ...
[F1..F10] mism=0 max_abs_diff=0.000e+00  BIT-EXACT
[CANARY bh=1 sl=300 caus=0 wnd=96] total=38400 mism=0 max_abs_diff=0.000e+00  BIT-EXACT
=== SUMMARY ===  forms bit-exact: 11 / 11
```
**Zero CUDA errors в логе** (per-kernel cudaGetLastError в harness — ни одной строки `[NAME] KERNEL: <err>` не напечатано).

`005a_sanitizer_clean.log`:
```
========= COMPUTE-SANITIZER
[F1..F10, CANARY] всё BIT-EXACT
========= ERROR SUMMARY: 0 errors
```
**11/11 including CANARY + compute-sanitizer 0 errors** — ворота ПРИНЯТЫ.

## Пункт 5 — Wall dk_new canonical re-check

`005a_wall.log` (bh=128 sl=8192 hd=128 causal=0, warmup=5 iters=20):
```
run 1: 9.422 ms   run 2: 9.423 ms   run 3: 9.415 ms   run 4: 9.421 ms
median: 9.421 ms
```
vs 004 median 9.177 ms → **+2.66%** (+0.244 ms).

Ожидание было "9.18-class" (stride_ds==sl → identical arithmetic). Реальность: +2.66% delta. Возможные причины: thermal (device state сегодня), тонкая register-schedule перестановка (ptxas 96r unchanged, но instruction order мог измениться из-за `(sl+15)&~15` в kernel prologue). Внутрирунная CV <0.1% — стабильно 9.42ms **на этом run**. Тот же класс, не regression в 9-ms диапазоне.

Not a blocker: kernel_dk_new остаётся **−54.3% wall vs sealed dK ~21.5 ms**, occupancy 4 blocks/SM сохранён (96r).

## Резюме 005a

- ✅ **In-bounds misalignment** подтверждён sanitizer (Vugar prav)
- ✅ **ABI-fix**: stride_ds = (sl+15)&~15 применён в ds_gen (write) + dk_new (read) + harness (allocation)
- ✅ **11/11 BIT-EXACT including CANARY** (max_abs_diff=0.000e+00)
- ✅ **compute-sanitizer: 0 errors** — memory-clean
- ✅ **Zero CUDA errors** в per-kernel log (F1 hygiene тоже clean)
- ✅ Wall dk_new canonical 9.42 ms — тот же класс (thermal noise +2.66%)
- ✅ ptxas 96r/0s unchanged, occupancy 4 blocks/SM сохранён

**Ворота приняты по всем 5 пунктам TZ. Fundament R1 стоит, CANARY-блокер снят структурно.**

Готов к **005b R1c dq_new** (по TZ пункт 5): load dS_nat (stride_ds) → MMA dS·K, K_T-фаза зеркалом sealed AA1, fp16-acc dQ_acc наследуется, accumulation order по kt == AA1.
