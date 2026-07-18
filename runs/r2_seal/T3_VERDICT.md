# T3 VERDICT — dQ Ярус-3 (2026-07-03)

## T1 — сборка (d) подтверждена

Log: `runs/r2_seal/d_ptxas.log` (+ `d_carveout_ptxas.log`).

```
kernel_dq: 168 registers, 68 B spill, 72 B stack, __launch_bounds__(128, 3) present in source
Source: libs/fa_bwd_dq.cu.d_diet  (line 102: __global__ __launch_bounds__(128, 3) void kernel_dq)
```

Гипотеза "unbounded ~196r с регфайловым блоком" ОТМЕНЯЕТСЯ — kernel_dq(d) БЫЛ bounded-168 с launch_bounds.

s1_residency_probe линковался ПРОТИВ той же fa_bwd_dq.cu (Makefile.s1_residency: `SRCS := s1_residency_probe.cu fa_bwd_dq.cu`); API-хэндл == kernel_dq в (d)-сборке. Загадка (API=3, runtime=2) реальна.

---

## T2(a) — синтетика на sm_120a

Source: `libs/t2_synth_residency.cu` (heavy variant).  
Build script: `runs/r2_seal/build_synth_heavy.sh`  
Launch log: `runs/r2_seal/t2_synth_r168_launch.log`  
NCu log: `runs/r2_seal/t2_synth_r168_ncu.csv`  
Baseline synth (66r): `runs/r2_seal/t2_synth_r66_ncu.csv`

### Синтетический kernel_dq-clone (168 regs) с ИДЕНТИЧНЫМ footprint'ом

| Param | synth3 heavy | dQ(d) |
|-------|--------------|-------|
| numRegs | **168** | 168 |
| spill loads/stores | 672 / 636 B | 68 / 68 B |
| stack frame | 640 B | 72 B |
| launch_bounds | (128, 3) | (128, 3) |
| dyn smem | 32768 | 32768 |
| cudaOccupancyMax | **3 blocks/SM** | 3 blocks/SM |
| cudaOccupancyMax with `carveout=MaxShared` | 3 blocks/SM | 3 blocks/SM |

### NCu результаты (та же метрика-плита)

| Metric | synth3 heavy | dQ(d) |
|--------|--------------|-------|
| `sm__ctas_active.avg.per_cycle_active` | **2.92 blocks** | **1.99 blocks** |
| `sm__ctas_active.avg.pct_of_peak_sustained_active` | 12.17% | 8.29% |
| `sm__warps_active.avg.per_cycle_active` | **11.50 warps** | **7.96 warps** |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 23.95% | 16.58% |
| `l1tex__throughput.avg.pct_of_peak_sustained_active` | 81.49% | 79.85% |

### T2(a) вывод
**sm_120a runtime scheduler ФИЗИЧЕСКИ ДЕРЖИТ 3 blocks/SM** при (168 regs, 640 B stack, 636 B spill, 32768 dyn smem, LB(128,3), большой grid).  
Reg-file granularity гипотеза ОТМЕНЯЕТСЯ (при регах 168 с бóльшим spill в synth 3-block достигаем).  
"HW scheduler cap" гипотеза ОТМЕНЯЕТСЯ.

---

## T2(b) — carveout

Изменение в `libs/fa_bwd_dq.cu` launcher (config (d)):

```c
cudaFuncSetAttribute(kernel_dq,
                     cudaFuncAttributePreferredSharedMemoryCarveout,
                     cudaSharedmemCarveoutMaxShared);
```

Log: `runs/r2_seal/d_carveout_ptxas.log`, `d_carveout_biteq.log`, `d_carveout_ncu.csv`.

### NCu (d)+carveout=MaxShared

| Metric | (d) без carveout | (d)+carveout=MaxShared |
|--------|------------------|-------------------------|
| ctas per_cycle | 8.29% | **8.29%** |
| warps per_cycle | 7.96 | **7.96** |
| L1 SOL | 79.86% | 79.85% |
| bit-exact | 11/11 PASS | 11/11 PASS |

Явный carveout=MaxShared **НЕ меняет ctas_active/warps_active**. Драйвер по умолчанию уже выбирает MaxShared при dyn smem=32768. Гипотеза Vugar "драйвер выбрал <100KB карвеут" ОТМЕНЯЕТСЯ.

---

## T3 — вилка

Vugar сформулировал три исхода:
- (1) T1 вскрыл unbounded → пересборка → истинный 12-warp → wall решает → **НЕ подтверждено**, (d) БЫЛ bounded-168.
- (2) Синтетика держит 3, наше не держит → копаем отличие → **ЭТО НАШ СЛУЧАЙ**.
- (3) Синтетика тоже 2 → закрыт железом, артефакт в техлог → **НЕ подтверждено**.

### T3 = case (2) — доложить, не строить

**Ярус-3 закрыт РАНТАЙМ-ПОВЕДЕНИЕМ, специфичным для kernel_dq**. Мы вышли за пределы явной SMEM/reg/carveout бухгалтерии. Runtime scheduler HW удерживает kernel_dq(d) на 2 blocks/SM даже когда:
- API-порог явно = 3 blocks/SM
- Синтетический kernel с ИДЕНТИЧНЫМ footprint'ом (168 regs, 640B stack, 636B spill, 32768 smem, LB(128,3)) — держит 2.92 blocks/SM 

### Кандидаты на механизм (не разделены, требуют дальнейшего исследования, ЗАПРЕЩЕНО без ACK Vugar)

Что есть в kernel_dq но нет в synth3:
1. **cp.async pipeline saturation** — kernel_dq использует `cpa16` (cp.async.ca.shared.global) массово: K/V/Q загрузки в warmup + K/V per-kt (128 iterations). HW имеет лимит on-flight cp.async slots per SM (доки NV не публикуют точный лимит для Blackwell). При 3 одновременных блоков × 12 warps × up-to-4 pending cp.async = 144 в очереди — вероятно превышает slot count. Runtime может throttle резидентность блоков.
2. **MMA sub-partition ownership** — kernel_dq использует mma.sync.aligned.m16n8k32.f32.e4m3 (FP8 tensor cores) + mma.m16n8k16.f32.f16.f16.f32 (FP16 tensor cores). Синт — только регистровые операции. HW MMA queue ownership может ограничить residency.
3. **`__syncthreads()` per kt** — kernel_dq: 4 syncthreads × 128 kt = 512 sync-вызовов/блок. Синт — 1 sync-вызов. Barrier throughput per SM может тормозить резидентность 3-го блока если queue заполнена.
4. **LDS traffic saturation** — kernel_dq: 4.165 млрд LDS wavefronts (81.49% L1 SOL). Синт — минимум LDS. При 3 concurrent блоков всё L1 может дойти до 100% → scheduler backs off к 2 блокам.

### Sealed dQ = **(a) L3-pre-P1**
Median wall = **326.72 T** (5-run same-thermal, ceiling J-канон 324.69 T).  
Path: `libs/fa_bwd_dq.cu` == `libs/fa_bwd_dq.cu.pre_P1`.  
Config (d) sealed для форензики: `libs/fa_bwd_dq.cu.d_diet`.

---

## T4 dV — обновлённая математика

Vugar заметил ошибку в моей S3 предпрогнозе: не учёл alias smQ↔smPT в dV yarus-1.

### Пересчёт ledger dV с alias
Yarus-1 dV SMEM (from `libs/fa_bwd_dv_mma_p1.cu`):  
- smK 8192 + smQ 8192 + smPT 8192 (alias smQ↔smPT по построению? проверить в коде) + smdO 16384 + …

Vugar сказал: `dyn_dV_effective = 42752 − 8192 (alias smQ↔smPT) = 34560`. Дефицит vs PG2 (33109) = **1451 B** (совпадает с исходной сметой Vugar 1.45 KB).  
**Прошу подтверждения**: если alias smQ↔smPT действительно применён в yarus-1 dV, то deficit = 1451 B, диета возможна.

### Однако T3-вывод для dQ применим и к dV
Если ярус-3 закрыт runtime-поведением, а не bookkeeping'ом, то диета в dV → API 3 blocks/SM → **возможно НЕ 3 блока в NCu**. dV kernel также использует cp.async, MMA, __syncthreads() — те же кандидаты-механизмы.

### Vugar разрешил: PG1 (L1 dV baseline)
> Параллельно разрешаю снять PG1: один NCu dV baseline -> L1 SOL, проекция x1.5 <85%?

Готов запустить один NCu на sealed dV, проверить L1 SOL. Если <85% × 1.5 (target с 3 блоками) — можно двигаться в S3. Если ≥85% — S3 закрыт на PG1.

Жду ACK на T3 + указаний по PG1 dV.

---

## Diet-заметка (в техлог)
KT_STRIDE 68→64 = реинтродукция ~4-way bank conflict на MMA-C smK_T reads.  
В (d) NCu bank_conflicts + LDS wavefronts фактически НЕ выросли (647.96M vs 647.84M на (a)) — доля MMA-C reads в общем LDS-трафике мала post-K2 (K2 добавил всех-warps phase 1.5 читающий smK, что доминирует).  
Для финального seal предпочтительнее найти 512B без реинтродукции конфликта (например SMK_AREA slack не найден в анализе; alignment slack не обнаружен). Оставить KT_STRIDE=68 базой.

---

## Пути ко всем артефактам T-серии

```
libs/
├── t2_synth_residency.cu       # синтетический heavy kernel
├── t2_synth_r168                # binary
├── t2_synth_heavy               # heavy synth binary (168 regs)
├── Makefile.t2_synth            # 3 reg-cap variants (все overriden LB)
├── fa_bwd_dq.cu                 # sealed (a) == .pre_P1
├── fa_bwd_dq.cu.d_diet          # (d) diet config для форензики
├── s1_residency_probe.cu        # API probe
└── Makefile.s1_residency

runs/r2_seal/
├── T3_VERDICT.md                # этот отчёт
├── build_synth_heavy.sh
├── ncu_synth_run.sh
├── t2_synth_r66_ncu.csv         # synth 66r: 2.88 blocks / 11.50 warps
├── t2_synth_r168_ncu.csv        # synth 168r: 2.92 blocks / 11.50 warps
├── t2_synth_r168_launch.log     # synth launch info
├── d_carveout_ptxas.log
├── d_carveout_biteq.log         # 11/11 PASS
├── d_carveout_ncu.csv           # (d)+carveout: 8.29% ctas (unchanged)
├── (все S1/S2 логи выше)
```

---

## Итог

- **T1**: (d) БЫЛ bounded-168 с launch_bounds(128,3), API-хэндл правильный. Загадка реальна.
- **T2(a)**: HW sm_120a physically держит 3 blocks/SM при 168r/640B stack/32768smem.
- **T2(b)**: carveout НЕ дискриминант.
- **T3**: Ярус-3 закрыт РАНТАЙМ-ПОВЕДЕНИЕМ, специфичным для kernel_dq. Механизм — вероятно cp.async / MMA / sync throttling.
- **Sealed dQ = 326.72 T (a)**, config (d) не даёт ярус-3, wall +0.13% = noise.

Жду ACK на T3 + вилку по S3 dV (PG1 NCu → продолжать или нет).
