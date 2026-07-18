# AB2 dK FP64-golden floor reference (2026-07-04)

Sealed kernel (probe): `libs/fa_bwd_dk.cu.AB2` (128 regs / 0 spill / 0 stack, fp16-acc MMA #2).  
Arithmetic: FP16-accumulator MMA #2 `mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16`.

Judge: FP64-golden (принцип 7). Bit-exact vs НОВЫХ эталонов ОБЯЗАН совпасть в AB3 (schedule-only).

Source log: `runs/r2_seal/ab2_v2_biteq.log` (seed=42).

| Form | bh | sl | causal | wnd | max_abs | max_rel_af | verdict |
|------|:--:|:---:|:------:|:---:|:-------:|:----------:|:-------:|
| F1   | 1  | 128  | 0 | 0   | **4.336e-03** | 3.237e-01 | PASS |
| F2   | 1  | 128  | 1 | 0   | **2.878e-02** | 1.295e+00 | PASS |
| F3   | 2  | 256  | 0 | 0   | **3.683e-03** | 2.964e-01 | PASS |
| F4   | 2  | 256  | 1 | 0   | **2.869e-02** | 1.307e+00 | PASS |
| F5   | 4  | 384  | 0 | 0   | **3.638e-03** | 2.638e-01 | PASS |
| F6   | 4  | 384  | 1 | 0   | **3.333e-02** | 1.608e+00 | PASS |
| F7   | 1  | 512  | 0 | 128 | **4.932e-03** | 3.038e-01 | PASS |
| F8   | 1  | 512  | 1 | 128 | **2.878e-02** | 1.295e+00 | PASS |
| F9   | 1  | 2048 | 0 | 0   | **6.711e-03** | 5.193e-01 | PASS |
| F10  | 1  | 2048 | 1 | 0   | **2.808e-02** | 1.362e+00 | PASS |
| CANARY | 1 | 300 | 0 | 96 | **4.957e-03** | 3.546e-01 | PASS |

Thresholds (Vugar-locked):
- **nc (F1/F3/F5/F7/F9/CANARY)**: ≤1e-2 — passed 2×+ запаса
- **causal (F2/F4/F6/F8/F10)**: ≤5e-2 — passed 1.5×+ запаса

**AB3 schedule-only jam-x2 MUST bit-exact match**: max_abs каждой формы должно СОВПАСТЬ с этой таблицей до последней значащей цифры. Отклонение = арифметика тронута, AB3 неверна.
