# AA1 dQ FP64-golden floor reference (2026-07-03)

Sealed kernel: `libs/fa_bwd_dq.cu` == `.SEALED_AA1` (161 regs / 0 spill / KT_STRIDE=68 / no LB).  
Arithmetic: FP16-accumulator MMA-C variant (`mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16`).

Judge: **FP64-golden** (принцип 7). Old fp32-acc regression references superseded.

Source log: `runs/r2_seal/aa1_biteq.log` (seed=42, tol abs 5e-2 + rel 5e-1, sig 1e-2).

## Reference table (regression baseline for any AB* build)

| Form | bh | sl | causal | wnd | max_abs | max_rel_af | verdict |
|------|:--:|:---:|:------:|:---:|:-------:|:----------:|:-------:|
| F1   | 1  | 128  | 0 | 0   | **4.653e-03** | 3.800e-01 | PASS |
| F2   | 1  | 128  | 1 | 0   | **2.180e-02** | 7.452e-01 | PASS |
| F3   | 2  | 256  | 0 | 0   | **3.781e-03** | 2.822e-01 | PASS |
| F4   | 2  | 256  | 1 | 0   | **2.614e-02** | 7.837e-01 | PASS |
| F5   | 4  | 384  | 0 | 0   | **3.896e-03** | 2.741e-01 | PASS |
| F6   | 4  | 384  | 1 | 0   | **3.144e-02** | 9.621e-01 | PASS |
| F7   | 1  | 512  | 0 | 128 | **4.531e-03** | 3.035e-01 | PASS |
| F8   | 1  | 512  | 1 | 128 | **2.180e-02** | 7.452e-01 | PASS |
| F9   | 1  | 2048 | 0 | 0   | **6.836e-03** | 5.428e-01 | PASS |
| F10  | 1  | 2048 | 1 | 0   | **2.180e-02** | 7.452e-01 | PASS |
| CANARY | 1 | 300 | 0 | 96 | **4.870e-03** | 3.166e-01 | PASS |

## Thresholds (Vugar-locked)
- **Non-causal (F1/F3/F5/F7/F9/CANARY)**: max_abs ≤ **1e-2** — passed with ≥2× запаса.
- **Causal (F2/F4/F6/F8/F10)**: max_abs ≤ **5e-2** — passed with ≥1.6× запаса.
- **K_T verifier**: bit-exact 8192/8192 cells match (transpose formula independent of MMA-C acc precision).

## Use for future regression
Any dQ modification must produce max_abs ≤ AA1 value (±10% tolerance for scheduling variance) на всех 11 formах.  
Breakage → доложить + attribute (compiler schedule vs arithmetic change).

## Related
- Previous fp32-acc floor references (P1a): archived in `runs/j_seal_dq_iso*.json` reference-set; superseded.
- dK reference (from AB2): TODO — снять после AB2 build.
- dV reference (unchanged from Yarus-1): existing files valid.
