# S2 REPORT — dQ Ярус-3 ФИЗИЧЕСКОЕ ТЕСТИРОВАНИЕ (2026-07-03)

## S0 — долги отчёта

### 1. J-канон dQ isolated (30-run, same-thermal)
- **median = 324.69 T**
- mean = 324.643 T, sd = 0.346, CV = 0.107%
- t vs baseline (171.9) = 1708.35 → +88.86%
- **path: `/data/lib/podman-data/projects/goml/runs/j_seal_dq_iso.json`**
- outliers: [(1, 323.38)] — first run отбросить

### 2. Техлог-фикс
- **Свизл ≠ K3.** Свизл = ярус-1 (Tier-1 XOR-swizzle campaign — smK/smV/smQ/smdO манипуляции, sealed 2026-07-02).  
  K3 = dK Qr-revert probe = WASH (только -4 регов, -0.9% wall).
- Обновлённая цепочка sealed dQ: **свизл (ярус-1) → K2 (all-warps phase 1.5 revert) → L3 (softmax/quantize jam-x2)**.
- N2 (jam-x4) = WASH −0.28%, откачен, НЕ в цепи.

---

## S1 — бухгалтерия резидентности (API-вердикт)

Пробa: `libs/s1_residency_probe.cu`, Makefile: `libs/Makefile.s1_residency`.  
Логи: `runs/r2_seal/s1_probe_{a,c,d}.txt`.

### Device attrs (RTX PRO 6000 Blackwell, sm_120a)
- **`cudaDevAttrReservedSharedMemoryPerBlock` (X) = 1024 bytes**
- `cudaDevAttrMaxSharedMemoryPerBlockOptin` = 101376 bytes
- `cudaDevAttrMaxSharedMemoryPerMultiprocessor` = **102400 bytes**
- `cudaDevAttrMaxBlocksPerMultiprocessor` = 24
- `cudaDevAttrMaxThreadsPerMultiProcessor` = 1536 (48 warps peak)

### Порог 3 blocks/SM
- Наивный: dyn ≤ **(102400/3) − X = 33109 B**
- P1-layout (b/c/33792 B): **deficit 683 B** — ровно как Vugar предсказал
- S2-diet (d/32768 B): **margin 341 B**

### cudaOccupancyMaxActiveBlocksPerMultiprocessor (block=128 threads)
| Kernel | dyn smem | Regs | Spill | API вердикт |
|--------|----------|------|-------|-------------|
| (a) no launch_bounds | 46336 B (L3) | 163 | 0 | **2 blocks/SM** |
| (a) no launch_bounds | 33792 B (P1) | 163 | 0 | **2 blocks/SM** |
| (c) `__launch_bounds__(128,3)` | 46336 B | 168 | 40B stack/44B spill | **2 blocks/SM** |
| (c) `__launch_bounds__(128,3)` | 33792 B | 168 | ... | **2 blocks/SM** |
| **(d) diet + LB(128,3)** | **32768 B** | 168 | 72B stack/68B spill | **3 blocks/SM ✓** |

### Верифицированные механизмы
- **launch_bounds ≠ force**: даже с (128,3) при dyn=33792 API даёт 2 (per_block=34816 > 34133).
- **Diet работает по API**: (d) dyn=32768 → per_block=33792 → floor(102400/33792) = 3 ✓.

---

## S2 — диета и физический прогон

### Diet plan (config d)
1. **smQ → Qr[KS_QK][4] registers** (direct-LDG в warmup) — save 8192 B
2. **smV↔smdS aliased** (BARRIER после MMA-B V-reads, before smdS writes) — save 8192 B  
3. **smL/smD → direct-LDG в L_lo/L_hi/D_lo/D_hi registers** (`L[b*sl + i_g]` per-thread) — save 512 B
4. **KT_STRIDE 68 → 64** — save 512 B (smK_area с K-natural = max(8192, 8192) = 8192); принимаем ~4-way bank conflict в MMA-C smK_T reads.
5. `__launch_bounds__(128, 3)` для reg budget ≤170.

### Итоговый SMEM (d)
- smK_area (K/K_T aliased) 8192 + smV/smdS aliased 8192 + smdO 16384 = **32768 B dynamic**
- 168 regs / 68 B spill loads/stores / 72 B stack frame
- Log: `runs/r2_seal/d_ptxas.log`

### Гейт 1 — API residency ✓
```
smem=32768 B, block=128 -> 3 blocks/SM  [S2-diet (d)]
```
Log: `runs/r2_seal/s1_probe_d.txt`

### Гейт 2 — bit-exact 11/11 ✓
`runs/r2_seal/d_biteq.log` — dQ pass 100.0000% на всех формах, K_T bit-exact.

### Wall — ОДНА термо-сессия A/B  
(fresh baseline after (d) sealed в тех же секундах)

| Config | 5 runs (ms) | median T | median wall | log |
|--------|-------------|----------|-------------|-----|
| **(d)** S2-diet 168r/68Bspill/32768B | 20.157/20.146/20.164/20.176/20.168 | **327.16 T** | 20.164 | `runs/r2_seal/d_wall.log` |
| **(a)** L3-pre-P1 163r/0B/46336B (same session) | 20.188/20.189/20.192/20.198/20.200 | **326.72 T** | 20.192 | `runs/r2_seal/a_v2_wall.log` |
| **Δ** | | **+0.44 T = +0.13%** | −0.028 ms | — |

CV < 0.1% в обоих. **Дельта на уровне термо-шума**, НЕ +2%+.

### Гейт 3 — NCu РЕАЛЬНОСТЬ (шок!)  
NCu на (d) vs NCu на (a). Скрипт: `runs/r2_seal/ncu_d_run.sh`.  
Логи: `runs/r2_seal/a_ncu.csv`, `runs/r2_seal/d_ncu.csv`.

| Metric | (a) L3-pre-P1 46336B | (d) S2-diet 32768B | Δ |
|--------|---------------------|---------------------|---|
| `sm__ctas_active` (per_cycle) | **8.29 %** | **8.29 %** | **0** |
| `sm__warps_active.avg.per_cycle_active` | **7.96** | **7.96** | **0** |
| `sm__warps_active.pct_peak` | 16.58 % | 16.58 % | 0 |
| `l1tex__throughput` (L1 SOL) | 79.85 % | 79.86 % | ≈0 |
| `sm__pipe_tensor_cycles_active` | 47.11 % | 47.10 % | ≈0 |
| `l1tex__t_sectors_pipe_lsu_mem_local_op_ld/st` | 0 / 0 | **0 / 0** | 0 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld` | 1,086,586,880 | 1,086,586,880 | 0 |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared` | 4,165,317,717 | 4,165,434,120 | +0.003% |
| `l1tex__data_bank_conflicts_pipe_lsu` | 647,841,777 | 647,963,892 | +0.02% |
| stall `wait` | 1.24 | 1.24 | 0 |
| stall `mio_throttle` | 0.89 | 0.89 | 0 |
| stall `short_scoreboard` | 0.35 | 0.35 | 0 |
| stall `long_scoreboard` | 0.27 | 0.27 | 0 |
| stall `barrier` | 0.13 | 0.13 | 0 |

**warps_active per_cycle = 7.96** — НЕ 12! Даже когда API-порог достигнут (3 blocks/SM разрешено бухгалтерией), runtime scheduler держит те же ~2.65 blocks/SM average.

---

## S2 — верификация вилки Vugar

Vugar формулировал 3 исхода:
- **+2%+ → ярус-3 ОТКРЫТ, seal** → **НЕТ** (только +0.13%).
- **wash при ФАКТИЧЕСКИХ 12 варпах → ярус-3 закрыт истинно** → **НЕДОСТИЖИМО** (12 варпов НЕ достигнуты фактом NCu).
- **диета не находится / API-порог недостижим → закрыт бухгалтерией железа** → **НЕТ** (диета найдена, API-порог достигнут).

### НОВЫЙ 4-й исход — необнаруженный ранее
**API-порог достигнут, но runtime не активирует 3-й block**. Механизм за пределами явной SMEM/reg бухгалтерии, доступной через `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.

Гипотезы механизма (не разделены):
1. **Register file granularity** на sm_120a — если гранулярность regs/warp = 16 (не 8), 168 → 176 → 176×128×3 = 67584 > 65536 = не помещаются 3 блока по регам. Но тогда API должно бы сказать 2. Возможно API устаревшая.
2. **Warp scheduler tail effect** — grid=16384 blocks × 176 SMs = 93 waves × 2 blocks/wave. С 3 blocks: 31 waves. Тail avg drops ctas_active.
3. **Spill в L1 cache** — 68B spill + 72B stack per thread × 128 threads × 3 blocks = 53.7KB нужно для стека/spill. Возможно L1 cache pressure ограничивает residency.
4. **Named barriers HW-cap** — 3 blocks × N barriers каждый. HW-лимит на named barriers/SM.

### Ярус-3 ВЕРДИКТ (обновлённый)
**Ярус-3 ЗАКРЫТ железом sm_120a — но НЕ по SMEM-бюджету** (диета решила). Механизм HW-scheduler-side, deeper than SMEM/reg accounting визуализированный в cudaOccupancyMax API.

**Sealed dQ остаётся (a): median 326.72 T (5-run same-thermal, ceiling J-canon 324.69 T).**

---

## Пути ко всем артефактам S2

```
/data/lib/podman-data/projects/goml/libs/
├── fa_bwd_dq.cu               # sealed = (a) L3-pre-P1
├── fa_bwd_dq.cu.pre_P1        # identical
├── fa_bwd_dq.cu.d_diet        # S2 (d) diet build (для форензики)
├── fa_bwd_dq.cu.pre_diet      # backup (c) state (LB + no diet)
├── s1_residency_probe.cu      # API probe source
└── Makefile.s1_residency

/data/lib/podman-data/projects/goml/runs/
├── j_seal_dq_iso.json                    # J-canon dq_iso 324.69T (30 runs)
└── r2_seal/
    ├── R3_SEAL.md              # первичный R3 seal
    ├── S2_REPORT.md            # этот отчёт
    ├── s1_probe_a.txt          # API config (a): 2 blocks/SM
    ├── s1_probe_c.txt          # API config (c) LB: 2 blocks/SM
    ├── s1_probe_d.txt          # API config (d) diet: 3 blocks/SM ✓
    ├── d_ptxas.log             # (d) 168r/68B spill
    ├── d_biteq.log             # (d) 11/11 PASS
    ├── d_wall.log              # (d) 327.16 T median
    ├── a_v2_wall.log           # (a) same-thermal 326.72 T median
    ├── a_v2_biteq.log
    ├── a_v2_ptxas.log
    ├── a_ncu.csv               # NCu (a): 8.29% ctas / 7.96 warps
    ├── d_ncu.csv               # NCu (d): 8.29% ctas / 7.96 warps ← IDENTICAL
    ├── ncu_d_run.sh
    └── (все R2 логи выше)
```

---

## S3 — что дальше

Vugar ТЗ:
> S3 -- P2 dV: порог PG2 обновить = 34133 − X (при X=1024 -> 33109, дефицит dV ~1.45KB не 427B).  
> PG1 (L1 dV, проекция x1.5 <85%?) решает первым. По исходному ТЗ с этой поправкой -- ПОСЛЕ S2-вилки.

### Обновлённая dV-математика
- `X = 1024 B` подтверждён из S1
- **PG2 порог 3 blocks/SM для dV: dyn ≤ (102400/3) − 1024 = 33109 B**  
- dV Yarus-1 seals SMEM 41.75 KB → deficit vs PG2 = 41.75×1024 − 33109 = **8 651 B** (Vugar oценил 1.45KB — расхождение, проверю)
- Ждать вердикт Vugar по S3 запуску.

### Важное открытие для dV S3
Даже если для dV диета сожмёт SMEM до PG2-порога, **runtime может отказать в 3-м блоке по HW-scheduler-механизму (как в dQ S2 (d))**. Нужно сначала проверить на dV, работает ли ярус-3 физически, ДО инвестиций в диету. 

Возможные пути проверки для dV:
- **Baseline NCu на dV**: измерить `sm__ctas_active` и `warps_active per_cycle` при текущем 1-block sealed. Если warp-active/blocks_available ≈ 1.0, то ceiling не в scheduler, а во что-то ещё. Если <1.0, то scheduler-cap уже действует.
- Если dV scheduler-cap = уже действует → ярус-3 не открыть даже диетой.

Жду ACK/дальнейших указаний от Vugar перед S3.
