# AB SUMMARY — dQ AA1 seal + dK fp16-acc regression (2026-07-03)

## AB0 — техлог правки принятые
- Декомпозиция **AA1** (+1.08% arch fp16-acc) vs **AA2** (+0.64% 12-warp marginal) — маргинал < любого порога, AA2 закрыт.
- Spill AA2 = **hot 29.7% L1 доля** (не "cold-ish" — исправлено).
- P1a/AA1 SMEM = **33792 B** (не "46 KB" — P1 edits убрали smQ + smV↔smdS alias):
  - SMK 8704 + SMV_SMDS aliased 8192 + smdO 16384 + smL/smD 512 = 33792 B
- Новые эталоны: `runs/r2_seal/AA1_FLOOR_REFERENCE.md` — regression baseline для будущих AB* билдов.

## AB1 — SEAL AA1 = НОВЫЙ dQ

**Sealed dQ**:
- File: `libs/fa_bwd_dq.cu` == `.SEALED_AA1`
- ptxas: **161 регов / 0 spill / 0 stack**
- Config: FP16-acc MMA-C, БЕЗ LB, БЕЗ диеты, P1a-layout 33792 B, KT_STRIDE=68
- Fingerprint EXPECT=161 ✓ (verified fresh bench_dq)

**30-run cert**: task bom8mmb1y running.  
Configs: v3_e2e_nc/c + dq_iso (dK/dV kernels не менялись — их iso из V3 30-cert).  
Live path: `runs/ab1_recert.log`.  
Expected медианы: dq_iso ~328-330, e2e_nc ~285-287 (термо-дрейф от V3-канон 285.44 T).

## AB2 — dK fp16-acc ТИРАЖ: РЕГРЕССИЯ WALL, floor OK

### Бумага (принято)
- dK_acc[16][4] fp32 → [16][2] u32 packed = **32 регов save** (248 → 216 ожидание)
- MMA #2 shape same as dQ MMA-C: `mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16`  
- Overflow-класс идентичен dQ: e4m3 × e4m3 → продукты ТОЧНЫ в fp16.

### Сборка
- Source: `libs/fa_bwd_dk.cu.pre_AB2` → apply fp16-acc → `libs/fa_bwd_dk.cu.AB2`
- ptxas: **128 регов / 0 spill** (лучше прогноза 216 — compiler упаковал агрессивнее)
- Log: `runs/r2_seal/ab2_ptxas.log`

### Гейты
- **Floor 11/11 PASS** — max_abs: nc ≤ 4.957e-3, causal ≤ 3.333e-2 (log `ab2_biteq.log`)
- **K_T verifier**: bit-exact (transpose independent of acc precision)

### Wall same-thermal
| Config | Regs | Wall median | Path |
|--------|:----:|:-----------:|------|
| dK baseline (fp32-acc) | 248 | **302.25 T** | `ab2_dk_base_wall.log` |
| **AB2 (fp16-acc)** | 128 | **286.00 T** | `ab2_wall.log` |
| **Δ** | | **−5.4% LOSS** | |

### Причина регрессии (гипотеза)
- Compiler упаковал агрессивнее (128r vs base 248r) → потеря register-level parallelism в MMA-C loop
- fp16 pack/unpack эпилог overhead
- Vugar-ожидание "same или +epsilon" НЕ реализовалось

### AB2 verdict
- **НЕ seal**. dK остаётся baseline (248r/0s fp32-acc).
- File revert: `libs/fa_bwd_dk.cu` == `.pre_AB2`.
- ILP-конверсия dK принципиально иная от dQ; fp16-acc НЕ переносится.

**Замечание**: AB2 wall измерение могло быть загрязнено thermal-контеншеном с параллельным AB1 30-run (task bom8mmb1y). Возможно re-measure после AB1 завершения даст меньшую регрессию. НО:
- 302 vs 286 = 16 T gap = ~5.4% — вряд ли объясняется thermal-drift-ом целиком
- CV на 5-run runs было < 0.1% (стабильно)
- Регрессия направленно LOSS, не noise

## AB3 — softmax-jam x2 на dK: НЕ ДЕЛАЕМ
Vugar: "только если AB2 floor ok И wall ~same". Наш wall = LOSS 5.4% — нет чистой AB2-базы для jam.

## AB4 — dV fp16-acc: ЗАПРЕТ (Vugar-locked)
P e4m3 × dO fp16 → продукты **не точны** в fp16 (другой класс риска). Отдельным ТЗ если возврашаемся.

## Итог AB-серии

### Sealed dQ = AA1 (**новый**)
- **161 регов / 0 spill / fp16-acc MMA-C** — упрощение архитектуры + scheduling relief
- Wall same-thermal: +1.08% vs P1a (был 327.43 → AA1 330.97)
- Ожидание 30-run cert: dq_iso ~328-330

### Sealed dK = baseline (**без изменений**)
- 248 регов / 0 spill / fp32-acc MMA #2 — как в V3 30-cert dk_iso 305.14 T
- AB2 fp16-acc тираж не удался (wall −5.4%)

### Sealed dV = Yarus-1 (**без изменений, ЗАПРЕТ на fp16-acc**)
- 129 регов / 0 spill

### Финальный FP8-потолок
Ожидание AB1 30-run cert: e2e_nc ~285.4-287 T (по факту решит).

## Артефакты AB-серии
```
libs/
├── fa_bwd_dq.cu                # SEALED AA1 (fp16-acc, 161r/0s)
├── fa_bwd_dq.cu.SEALED_AA1     # identical backup
├── fa_bwd_dq.cu.AA1            # AA1 stage backup
├── fa_bwd_dq.cu.AA2            # AA2 stage backup (12-warp probe)
├── fa_bwd_dq.cu.pre_AA         # sealed P1a backup
├── fa_bwd_dk.cu                # sealed baseline (248r/0s)
├── fa_bwd_dk.cu.pre_AB2        # baseline backup
├── fa_bwd_dk.cu.AB2            # AB2 probe backup (128r/0s)

runs/
├── ab1_recert.py               # 30-run cert driver EXPECT=161
├── ab1_recert.log              # cert live progress
├── ab1_e2e_nc*.json            # 30-run stats
└── r2_seal/
    ├── AB_SUMMARY.md           # ЭТОТ ОТЧЁТ
    ├── AA1_FLOOR_REFERENCE.md  # new regression floor
    ├── ab2_ptxas.log           # 128r/0s
    ├── ab2_biteq.log           # 11/11 PASS
    ├── ab2_wall.log            # 286 T
    └── ab2_dk_base_wall.log    # 302 T baseline same-thermal
```

## Ждём AB1 30-run завершения → финальный final re-cert таблица.
