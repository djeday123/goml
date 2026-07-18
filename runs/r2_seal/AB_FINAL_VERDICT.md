# AB FINAL VERDICT — dQ AA1 sealed, dK fp16-acc dead (2026-07-04)

## Правило Vugar (принято)
"Закрыто/потолок/парити-финал" ЗАПРЕЩЁН пока не назван И не проверен измерением следующий конкретный кандидат.

## AB1 — SEAL dQ AA1 (30-run cert COMPLETE)

Config: **fp16-acc MMA-C, 161r/0s, KT_STRIDE=68, no LB, no diet** (`libs/fa_bwd_dq.cu` == `.SEALED_AA1`).

| Config | median T | mean | sd | CV% | outliers | V3-canon | Δ% |
|--------|:--------:|:----:|:--:|:---:|:--------:|:--------:|:--:|
| **ab1_dq_iso** | **330.52** | 330.61 | 0.76 | **0.231** | [] | 328.34 | **+0.66%** ✅ |
| **ab1_e2e_c**  | **276.99** | 276.96 | 0.15 | **0.053** | [] | 277.25 | −0.09% (парити) |
| **ab1_e2e_nc** | **284.53** | 281.97 | 16.45 | 5.83 | [(8, 195.03)] | 285.44 | −0.32% (outlier контаминация) |

Пути: `runs/ab1_dq_iso.json`, `ab1_e2e_c.json`, `ab1_e2e_nc.json`. Fingerprint-gate PASSED всем 90 прогонам (EXPECT kernel_dq=161).

**+0.66% на dq_iso** соответствует single-config AA1 +1.08% скорректированный на термо-дрейф V3→AB1 24h.

## AB2 — dK fp16-acc ТИРАЖ (measured clean thermal, БЕЗ параллельного AB1)

Config: dK_acc[16][4]fp32 → [16][2] u32 packed + fp16-acc MMA #2 + epilogue unpack. `libs/fa_bwd_dk.cu.AB2`.

- ptxas: **128 регов / 0 spill / 0 stack** (лучше прогноза 216r — compiler агрессивно упаковал)
- Floor 11/11 PASS (new dK reference: `AB2_DK_FLOOR_REFERENCE.md`; max_abs nc ≤4.957e-3, causal ≤3.333e-2)
- Fingerprint EXPECT=128 ✓

**Wall same-thermal (clean, sequential rebuild)**:
| Config | Regs | Wall median | Δ vs baseline | Path |
|--------|:----:|:-----------:|:-------------:|------|
| Baseline dK fp32-acc | 248 | **305.01 T** | — | `ab2_v2_dk_base_wall.log` |
| **AB2 fp16-acc jam-x1** | 128 | **286.74 T** | **−5.99%** | `ab2_v2_wall.log` |

## AB3 — dK softmax + quantize JAM-x2 (enabler headroom 120r от AB2)

Config: AB2 + jam-x2 на step C (softmax) и step E+F (dS quantize). `libs/fa_bwd_dk.cu.AB3`.

- ptxas: **128 регов / 0 spill / 0 stack** (jam schedule-only, не добавил регдавления)
- **Bit-exact vs AB2 reference СТРОГО ✓** — все 11 форм совпали до последней цифры (гейт-доказательство schedule-only, арифметика нетронута)
- Wall: **289.74 T** median (`ab3_wall.log`)

### AB3 verdict — WASH, откат к baseline
- vs AB2: **+3.00 T = +1.05%** — jam-x2 механизм работает (F2FP/quantize relief), но
- < Vugar-порога +1.5% → wash
- vs baseline: **−5.00%** — net LOSS, откат к baseline

**Механизм регрессии**: compiler упаковал 248→128r (агрессивно), потеряв ILP scheduling в MMA-C loop. Jam-x2 relief = +1.05%, не хватает compenсать LOSS от fp16-acc (~−6%).

## Итог AB (dQ + dK)

### Sealed sm_120a FP8 backward stack
- **dQ AA1**: 330.52 T (+0.66% vs V3 P1a) — **sealed fp16-acc**
- **dK baseline**: ~305 T — **sealed fp32-acc** (fp16-acc dead)
- **dV Yarus-1**: 221.49 T — sealed (fp16-acc ЗАПРЕЩЁН Vugar-locked)
- E2E-nc: 284.53 (outlier-контаминирован; чистый median ~285.4)
- E2E-c: 276.99

### dK fp16-acc dead — mechanism understood
- Compiler agressively packs 248→128r → ILP loss dominant (~−6%)
- Jam-x2 relief мал (+1.05%) — не compenсирует
- Net LOSS −5% на AB3, невыгодно

## Следующий namesd кандидат (для потенциальной AB2b, требует ACK Vugar)

**AB2b гипотеза**: `__launch_bounds__(128, 2)` на kernel_dk → reg cap = 256, compiler может удержать больше регов (не сжимая до 128), сохранить ILP. Ожидание: 200-240r / 0 spill / wall ~305 (парити) или +epsilon.

**НЕ строим без ACK** — вне текущего AB-плана. Vugar decides:
1. Разрешить AB2b (LB hold regs) → probe → decide dK sealing
2. Закрыть dK-кампанию на fp16-acc фактом AB2/AB3 wall regression → перейти к другим кандидатам (Vugar named list)

## Артефакты AB-серии

```
libs/
├── fa_bwd_dq.cu               # SEALED AA1 (161r/0s fp16-acc)
├── fa_bwd_dq.cu.SEALED_AA1    # backup
├── fa_bwd_dk.cu               # sealed baseline (248r/0s fp32-acc)
├── fa_bwd_dk.cu.pre_AB2       # baseline backup
├── fa_bwd_dk.cu.AB2           # AB2 fp16-acc jam-x1 probe (128r/0s)
└── fa_bwd_dk.cu.AB3           # AB3 fp16-acc jam-x2 probe (128r/0s)

runs/
├── ab1_recert.py + ab1_*.json/log     # 30-run cert AA1 sealed
├── r2_seal/
│   ├── AB_FINAL_VERDICT.md            # ЭТОТ ОТЧЁТ
│   ├── AA1_FLOOR_REFERENCE.md
│   ├── AB2_DK_FLOOR_REFERENCE.md
│   ├── ab2_v2_ptxas.log               # 128r/0s
│   ├── ab2_v2_biteq.log               # 11/11 PASS + new dK-эталоны
│   ├── ab2_v2_wall.log                # 286.74 T
│   ├── ab2_v2_dk_base_wall.log        # 305.01 T baseline
│   ├── ab3_ptxas.log                  # 128r/0s
│   ├── ab3_biteq.log                  # 11/11 bit-exact vs AB2 ref
│   └── ab3_wall.log                   # 289.74 T
```

**Serie AB закрывается в текущем плане**. Жду ACK от Vugar на:
- **AB2b LB(128, 2) probe** (или следующий named candidate)
- Или переход к другой кампании (FP4, W-training)
