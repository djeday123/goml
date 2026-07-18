# U VERDICT — dQ ярус-3 РЕАЛЬНОЕ ЗАКРЫТИЕ + PG1 dV (2026-07-03)

## U1 — Launch-форензика вскрыла ЛАУНЧЕР-БАГ

### До пересборки bench_dq (моя старая "T3-мистика"):

NCu на bench_dq (built Jul 3 02:49, до всех S1/S2/T-edits в fa_bwd_dq.cu):
```
Registers Per Thread             168        ← "правильное" из ptxas теку­щего source
                                            ← НА САМОМ ДЕЛЕ: LaunchStats показал 163
Dynamic Shared Memory Per Block  46.34 KB   ← НЕ 32.77 KB
Static Shared Memory Per Block   0 byte
Block Limit Shared Mem           2 blocks   ← ЛИМИТЕР: SMEM
Theoretical Occupancy            16.67%     ← 2 blocks/SM
Achieved Occupancy               16.58%     
Achieved Active Warps Per SM     7.96
```
Log: `runs/r2_seal/d_launch_occupancy.txt` (ARCHIVE OF THE BUG).

### Диагноз
- `bench_dq` binary имеет **свой отдельный Makefile** (`libs/Makefile.bench_dq`) — линкует `fa_bwd_dq.cu` при своей сборке.
- **Я НИКОГДА не пересобирал bench_dq** после первичной сборки Jul 3 02:49 (=R2 (a) time).
- Пересборка `make -f Makefile.fa_bwd_dq` перестраивает только `fa_bwd_dq_test` — bench_dq остаётся с оригинальным source.
- Все "config (b)/(c)/(d) wall" замеры в S2/R2 через bench_dq = **ИЗМЕРЯЛИ ТУ ЖЕ (a) BUILD** (163r/46336B smem).
- Все "NCu (c)/(d)" тоже профилировали (a).
- Разница +0.13% (d vs a) = **thermal drift**, не diff между конфигами.

### После пересборки bench_dq с текущим (d) source:

```
Registers Per Thread             168        ← правильно
Dynamic Shared Memory Per Block  32.77 KB   ← 32768 B ✓
Static Shared Memory Per Block   0 byte
Block Limit Shared Mem           3 blocks   ✓
Block Limit Registers            3          ✓
Block Limit Warps                12         ✓
Theoretical Occupancy            25%        ✓ 3 blocks/SM
Achieved Occupancy               24.78%     ✓
Achieved Active Warps Per SM     11.89      ← ФАКТ 12 WARP ✓✓✓
```
Log: `runs/r2_seal/d_launch_occupancy_REBUILT.txt`.

**Прозаика-2 подтверждена: launcher-баг. Мистика T3 отменяется.**

Все прежние "T3-нахождения" ("runtime scheduler-специфичное поведение kernel_dq" и pipeline/MMA/sync гипотезы) — **ОТКЛОНЯЮТСЯ**. То что profил ir'ovalos как "(d) build" на самом деле было (a) build.

---

## U1 — ПЕРВЫЙ ЧЕСТНЫЙ 12-WARP FACT-TEST WALL

Same-thermal сессия, свежие пересборки:

| Config | Regs | Smem dyn | Occ | Block Limit | Achieved Warps | Wall median | Path |
|--------|------|----------|-----|-------------|----------------|-------------|------|
| (a) 2-block | 163 | 46336 B | 16.67% | SMEM=2 | 7.96 | **327.37 T** (20.152 ms) | `a_REAL_wall.log`, `a_REAL_bench_ptxas.log` |
| (d) 3-block | 168 | 32768 B | 25% | SMEM=3 | **11.89** | **308.86 T** (21.359 ms) | `d_REAL_wall.log`, `d_bench_ptxas.log`, `d_launch_occupancy_REBUILT.txt` |
| **Δ** | | | | | | **−5.65% LOSS** | |

CV < 0.05% в обоих. bit-exact 11/11 обоих.

### U1 verdict = "wash при факт-12 = истинное закрытие конверсией" ветвь Vugar-вилки

**Ярус-3 dQ ИСТИННО ЗАКРЫТ КОНВЕРСИЕЙ** — не bandwidth ceiling, не HW-cap, не launcher-bug (эти отменены). Кон­версия 12-warp → wall НЕГАТИВНАЯ:
- L1 уже насыщен при 2 блоках (79.85%)
- 3-й блок добавляет L1 contention без throughput gain
- KT_STRIDE=64 → 4-way bank conflict реинтродукция (мала пост-K2, но +)
- 68B spill в hot path → L1 traffic +
- Occupancy 25% vs 16.67% — но HW не превратил во wall

**Sealed dQ = (a) 327.37 T** (5-run median в этой сессии; J-канон 30-run = 324.69 T).  
Path: `libs/fa_bwd_dq.cu` == `libs/fa_bwd_dq.cu.pre_P1`.

---

## Техлог обновление

### Перепишу вилку исходов R2/S2/T
1. Прошлый sealed dQ 326.72 T (R3_SEAL.md) — это тот же (a) через thermal-drift'ed bench_dq. Обновляю до **327.37 T** (свежая пересборка, sealed).
2. All my "T3-загадка runtime-специфичного поведения kernel_dq" — **ARTIFACT ЛАУНЧЕР-БАГА**. Отменяется. `runs/r2_seal/T3_VERDICT.md` содержит incorrect verdict.
3. **Sealed чистое закрытие ярус-3 dQ**: bandwidth-конверсия, wall −5.65% при факт-12 warp.

### Diet KT_STRIDE 68→64 → техлог
Реинтродукция 4-way bank conflict на MMA-C smK_T reads. В (d) real-test contributed to wall loss наряду с bandwidth конверсией. Для финального seal STRIDE=68 сохранён.

### Прошлые фантомы (уже задокументированы)
- N2-327.1 — не измерялся, вычищен  
- P1-3block 334.40T/24B — не измерялся, вычищен
- **Новый**: T3 "runtime-специфичность kernel_dq" — оказалась launcher-багом, а НЕ реальным свойством ядра. Задокументировать как incident-3.

---

## U2 — PG1 dV baseline

Rebuild bench_dv, NCu LaunchStats+Occupancy+Memory.  
Log: `runs/r2_seal/dv_baseline_launch_occ.txt`.

### dV baseline (Yarus-1 sealed)
```
Registers Per Thread             129
Dynamic Shared Memory Per Block  41.22 KB   (= 42212 B)
Static Shared Memory Per Block   0 byte
Block Limit Shared Mem           2 blocks   ← ЛИМИТЕР
Block Limit Registers            3
Block Limit Warps                12
Theoretical Occupancy            16.67%     (2 blocks/SM)
Achieved Occupancy               16.58%
Achieved Active Warps Per SM     7.96

Memory Workload:
  Memory Throughput              33.51 GB/s
  Mem Busy                       37.24 %
  L1/TEX Hit Rate                1.75 %      (pure streaming, no reuse)
  Mem Pipes Busy                 75.24 %     ← ключевая метрика
  Local Memory Spilling          0
```

### PG1 projection ×1.5 (12 warp target)
- **Mem Pipes Busy 75.24% × 1.5 = 112.8%** → превышает 100%, HW throttles
- Mem Busy 37.24% × 1.5 = 55.9% (< 85%, было бы OK)
- L1/TEX Hit Rate 1.75% — потоковое чтение, кэша нет; +50% warps не улучшат hit rate

### PG1 verdict = FAIL
**Ярус-3 dV закрыт bandwidth-конверсией ДО-tesтa**. Mem Pipes Busy 75.24% при 2 блоков — потолок близко. При 12 warp target projection превышает 100% pipes → wall-конверсия негативная (аналогично dQ (d) −5.65%).

**S3 dV ярус-3: НЕ СТРОИТЬ.** Диета до 33109B (deficit 1451B post-alias) — теоретически возможна, но wall-конверсия закрыта bandwidth pipes.

---

## Пути ко всем артефактам U-серии

```
libs/
├── fa_bwd_dq.cu               # sealed (a) = .pre_P1
├── fa_bwd_dq.cu.d_diet        # (d) config для форензики
├── fa_bwd_dq.cu.pre_P1        # sealed backup
├── s1_residency_probe.cu      # API probe
├── t2_synth_residency.cu      # HW-capacity synthetic
├── Makefile.bench_dq          # !!! отдельный от Makefile.fa_bwd_dq !!!
└── Makefile.bench_dv          # !!! то же !!!

runs/
├── j_seal_dq_iso.json         # J-канон 324.69 T
└── r2_seal/
    ├── U_VERDICT.md           # ЭТОТ ОТЧЁТ (текущий вердикт)
    ├── T3_VERDICT.md          # СТАРЫЙ, содержит invalid T3 conclusion
    ├── S2_REPORT.md           # СТАРЫЙ, содержит invalid metric comparisons
    ├── R3_SEAL.md             # СТАРЫЙ initial R3
    
    ├── a_REAL_wall.log        # (a) 5-run FRESH bench_dq: 327.37 T
    ├── a_REAL_bench_ptxas.log # (a) 163r/0s (from bench_dq build)
    ├── a_REAL_biteq.log       # 11/11 PASS
    
    ├── d_REAL_wall.log        # (d) 5-run FRESH bench_dq: 308.86 T
    ├── d_bench_ptxas.log      # (d) 168r/68B (from bench_dq build)
    ├── d_launch_occupancy.txt          # PRE-rebuild (a) launcher-баг ARTIFACT
    ├── d_launch_occupancy_REBUILT.txt  # POST-rebuild (d) правильно 3 block/12 warp
    ├── ncu_d_launchstats.sh
    
    ├── dv_baseline_launch_occ.txt   # dV baseline NCu PG1
    ├── dv_bench_ptxas.log           # dV baseline build (129r/0s)
    ├── ncu_dv_baseline.sh
    
    ├── s1_probe_{a,c,d}.txt   # API probe (все правильные)
    ├── t2_synth_r{66,168}_ncu.csv  # synth NCu (правильные)
    └── ... (все прошлые)
```

---

## ИТОГ

1. **T3 "мистика" — АРТЕФАКТ launcher-бага.** Мистика отменяется.
2. **Ярус-3 dQ ИСТИННО ЗАКРЫТ КОНВЕРСИЕЙ**: bandwidth-конверсия, factual 12-warp wall = 308.86 T = **−5.65% vs (a) 327.37 T**.
3. **Sealed dQ = 327.37 T** (свежий 5-run same-session; J-канон 324.69 T).
4. **PG1 dV FAIL**: Mem Pipes Busy 75.24% × 1.5 = 112.8% > 100%. Ярус-3 dV также закрыт bandwidth.
5. **Workflow rule wired**: bench_* binaries имеют СВОИ Makefiles отдельно от `*_test`. При изменении kernel source пересобирать ОБЕ Makefile — иначе тесты и бенчмарки могут работать на разных сборках.

**Перф-глава dQ ЗАКРЫТА.** Sealed: **327.37 T** dQ.

Жду ACK + указаний по следующему направлению (E2E-full, dK deeper, или другое).
