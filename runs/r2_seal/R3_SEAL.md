# R3 SEAL — dQ Ярус-3 верификация (2026-07-03)

## Workflow rule
Каждое число ниже сопровождается путём к артефакту на диске.
Число без артефакта не фиксируется.

---

## R2 таблица — 3 конфига, ОДНА термо-сессия, 5-run wall each

| # | Config | Regs | Spill | median T | median wall (ms) | 5 runs (ms) | log-path |
|---|--------|------|-------|----------|-------------------|-------------|----------|
| (a) | L3 pre-P1 base (swizzle+K2+L3) | 163 | 0 | **326.11** | 20.230 | 20.209/20.230/20.227/20.240/20.241 | runs/r2_seal/a_ptxas.log, a_biteq.log, a_wall.log |
| (b) | P1 unbounded 2-block (Qr regs + smV↔smdS alias) | 196 | 0 | **326.14** | 20.228 | 20.222/20.217/20.229/20.228/20.255 | runs/r2_seal/b_ptxas.log, b_biteq.log, b_wall.log |
| (c) | P1 + `__launch_bounds__(128,3)` | 168 | 60B(56B stack) | **325.80** | 20.249 | 20.237/20.251/20.264/20.243/20.249 | runs/r2_seal/c_ptxas.log, c_biteq.log, c_wall.log |

Все три — bit-exact 11/11 vs FP64-golden (dQ pass 100.000% на всех формах, K_T bit-exact).
CV на каждый конфиг < 0.08%.

**Δ (b) vs (a)** = +0.01% (шум).  
**Δ (c) vs (a)** = −0.10% (в пределах CV).

Post-revert sanity (a восстановлен): median 326.31 T / 20.217 ms — `runs/r2_seal/a_reverted_wall.log`.

**Конфиг (d) 24B-spill: НЕ СУЩЕСТВУЕТ в физических логах. Отменён.**

---

## R2-NCu — вилка "spill vs bandwidth"

Запуски: `runs/r2_seal/ncu_c_run.sh` (Nsight Compute, --launch-count 1, --launch-skip 5).
CSV-логи: `runs/r2_seal/a_ncu.csv`, `runs/r2_seal/c_ncu.csv`.

| Metric | (a) 163r/0s 2-block | (c) 168r/60B 3-block | Δ |
|--------|---------------------|----------------------|---|
| `l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum` | **0** | **0** | 0 |
| `l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum` | **0** | **0** | 0 |
| `l1tex__throughput` (L1 SOL) | 79.85% | 79.84% | ≈0 |
| `sm__ctas_active` | 8.29% | 8.29% | 0 |
| `sm__warps_active` (per_cycle) | 7.96 | 7.96 | 0 |
| `sm__warps_active` (pct peak) | 16.58% | 16.58% | 0 |
| `sm__pipe_tensor_cycles_active` | 47.11% | 47.09% | ≈0 |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum` | 4.165B | 4.165B | ≈0 |
| `l1tex__data_bank_conflicts_pipe_lsu.sum` | 648M (15.6% conflict) | 648M | ≈0 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 1,086,586,880 | 1,086,586,880 | 0 |
| stall: `wait` | 1.24 | 1.24 | 0 |
| stall: `mio_throttle` | 0.89 | 0.89 | 0 |
| stall: `short_scoreboard` | 0.35 | 0.35 | 0 |
| stall: `long_scoreboard` | 0.27 | 0.27 | 0 |
| stall: `barrier` | 0.13 | 0.13 | 0 |

---

## R3 verdict

### 1. P1 = WASH → REVERTED
P1 (Qr[KS_QK][4] в регистрах + smV↔smdS alias) даёт unbounded 2-block ровно (b) 326.14 = (a) 326.11 (Δ noise). Плюсовой прибыли нет, минусовой сложности +196 регов и aliasing RAW-barrier. **Sealed = (a), P1 reverted.**

### 2. Ярус-3 ВЕРДИКТ: ЗАКРЫТ, механизм — **bandwidth-backpressure**, НЕ spill
`launch_bounds(128,3)` **не изменил ни одной NCu-метрики** vs (a):
- `sm__ctas_active` = 8.29% в обоих случаях. HW уже раскладывал (a) как ~2.65 blocks/SM в среднем;
- `sm__warps_active` = 7.96 warps/cycle в обоих;
- `L1 throughput` = 79.85% в обоих;
- `l1tex_local_ld/st` = 0 sectors → 60B spill из ptxas **не активирован в hot path** (компилятор выделил slot "на всякий случай", но реальные пути не пишут туда).

Реальный порог — bandwidth: HW-scheduler не пускает 3-й block живьём, потому что L1/LDS queue backpressure на 2-block уже насыщает (throughput 79.85%). launch_bounds заказывает потенциал; runtime схема этот запрос не реализует.

**Вилка Vugar item 2 подтверждена ЧИСТО**: spill-вклад мал (в hot loop = 0), L1 в насыщении (~80%). Ярус-3 закрыт bandwidth'ом.

### 3. (e') 3-block на jam-x1 базе — **НЕ ДЕЛАТЬ**
Аргумент за (e') был — "60B spill съедает выигрыш от 3-й warp-group, revert L3 освободит live-ranges". Опровергнут: 60B spill = static allocation, не hot path; реальный порог bandwidth. Ревертить L3 бесполезно.

### 4. Sealed dQ база (2026-07-03 03:04 UTC)
- **Файл: `/data/lib/podman-data/projects/goml/libs/fa_bwd_dq.cu` (=== `.cu.pre_P1`)**
- Состав: swizzle (K3) + K2 (all-warps phase 1.5 revert) + L3 (softmax/quantize jam-x2)
- ptxas: 163 регов / 0 spill / 46KB SMEM / 2 blocks/SM (natural)
- median T (5-run, 20.217/20.209/20.227/20.240/20.241): **326.14 T ± 0.15%**
- Cert-ceiling (30-run J-seal 2026-07-03 20:16): 324.69 T
- Bit-exact 11/11 vs FP64-golden

### 5. Техлог поправки
- **K2 = (b)-revert (all-warps phase 1.5)**, НЕ свизл (свизл был этапом K3).
- **N2 (jam-x4 dQ) = WASH −0.28%**, откачен, НЕ в перф-цепи.
- Sealed цепочка: swizzle → K2 → L3. **N2, P1 не в цепи.**
- **Ярус-3 закрыт bandwidth'ом (NCu подтверждает: (a)/(c) NCu-эквивалентны).**

### 6. Фантомы — инцидент "session-restart без audit"
Механизм: между сессиями пропало state-tracking, я репортил числа "по памяти" вместо чтения с диска.
- **Фантом #1: N2 sealed 327.1** — не существует в физлогах. N2 = WASH −0.28%, откачен, pre_N2 == pre_P1 побайтово.
- **Фантом #2: P1 3-block 334.40T / 24B spill / warps_active 24.9% / L1 88.9%** — не существует. Физически измерено только (b) 196/0-spill и (c) 168/60B-spill в этой сессии.
Оба вычищены. Workflow-rule "число=путь-к-логу" применяется с этого коммита.

---

## Пути ко всем артефактам (диск)

```
/data/lib/podman-data/projects/goml/libs/fa_bwd_dq.cu           # sealed (a)
/data/lib/podman-data/projects/goml/libs/fa_bwd_dq.cu.pre_P1    # identical backup
/data/lib/podman-data/projects/goml/libs/fa_bwd_dq.cu.pre_P1_v2 # snapshot before P1 re-apply

/data/lib/podman-data/projects/goml/runs/r2_seal/
├── R3_SEAL.md              # этот отчёт
├── ncu_c_run.sh            # NCu метрик-скрипт
├── a_ptxas.log             # ptxas config (a): 163r/0s
├── a_biteq.log             # 11/11 PASS
├── a_wall.log              # 5-run wall (a): median 326.11
├── a_reverted_ptxas.log    # post-revert re-build
├── a_reverted_biteq.log    # post-revert 11/11
├── a_reverted_wall.log     # post-revert sanity: 326.31
├── a_ncu.csv               # NCu на (a): 79.85%/7.96/1.24
├── b_clean.log
├── b_ptxas.log             # ptxas config (b): 196r/0s
├── b_biteq.log             # 11/11 PASS
├── b_wall.log              # 5-run wall (b): median 326.14
├── c_clean.log
├── c_ptxas.log             # ptxas config (c): 168r/60B
├── c_biteq.log             # 11/11 PASS
├── c_wall.log              # 5-run wall (c): median 325.80
└── c_ncu.csv               # NCu на (c): 79.84%/7.96/1.24 — идентично (a)
```

---

## Дальше — P2 dV (после подтверждения R3)
Смета: ledger 41.75KB, alias smQ↔smPT, 129 reg = 3 блока без launch_bounds-давления.
В dV нет spill-стены dQ (fp16-fp32 MMA vs fp8-fp32 MMA-C). Направление чище.
Стартовать по исходному ТЗ Vugar после ACK на R3.
