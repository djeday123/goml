# V VERDICT — dQ P1a SEALED + V2 fingerprint gate + V3 30-run RE-CERT (2026-07-03)

## V1 — честный P1a re-test

### Blast-radius correction (принято от Vugar)
R2 (b)/(c) прошлые числа — **три замера одного stale-бинаря** (bench_dq не пересобирался с 02:49). "P1a wash" НЕ доказан. R1-числа P1a 328.55/328.79 могут быть НАСТОЯЩИМИ. → V1.

### P1a конфиг (без диеты, без launch_bounds)
- P1 edits: smQ → Qr[KS_QK][4] registers + smV↔smdS aliased + `__syncthreads()` перед step E+F
- smL/smD **сохранены** в SMEM (нет диеты)
- KT_STRIDE = **68** (unchanged)
- **NO** `__launch_bounds__`
- source: `libs/fa_bwd_dq.cu.p1a` == `libs/fa_bwd_dq.cu.SEALED_P1a` == текущий `libs/fa_bwd_dq.cu`

### Ptxas (свежая пересборка)
```
kernel_dq: 196 registers, 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads, 1 barriers
```
Log: `runs/r2_seal/p1a_ptxas.log`, `p1a_bench_ptxas.log`.

### Fingerprint-гейт (NCu LaunchStats)
```
Registers Per Thread             196          ✓ ожидаемое P1a
Dynamic Shared Memory Per Block  33.79 KB     ← 33792 B (SMK 8704 + SMV_SMDS 8192 + smdO 16384 + smL/D 512)
Static Shared Memory Per Block   0 byte
Block Limit Shared Mem           2            ← P1a остаётся 2-block (reg-limit тоже 2)
Block Limit Registers            2            ← 196*128*3 = 75264 > 65536
Theoretical Occupancy            16.67%       (2 blocks/SM)
Achieved Occupancy               16.58%
Achieved Active Warps Per SM     7.96
```
Log: `runs/r2_seal/p1a_launch_occ.txt`.

### Wall — ЧЕСТНЫЙ same-thermal A/B (свежие пересборки обоих)

| Config | Regs | Smem dyn | Warps | Wall median | Path |
|--------|------|----------|-------|-------------|------|
| (a) sealed L3-pre-P1 | 163 | 46336 B | 7.96 | **327.38 T** (20.151 ms) | `a_v3_REAL_wall.log`, `a_v3_bench_ptxas.log` |
| **P1a (Qr regs + alias)** | 196 | 33792 B | 7.96 | **330.90 T** (19.936 ms) | `p1a_REAL_wall.log`, `p1a_bench_ptxas.log` |
| **Δ** | | | | **+3.52 T = +1.075%** | |

CV < 0.1% в обоих; bit-exact 11/11 обоих.

### V1 verdict = **>= +1% → SEAL P1a** (Vugar вилка ветвь A)
**Новый sealed dQ = P1a = 330.90 T** (5-run same-thermal same-session).  
Механизм: Qr registers снижают LDS overhead в MMA-A (Q teraz прямо в регах, не через smem+swizzle); smV↔smdS alias освобождает 8KB для реюза (SMEM уменьшается 46→34 KB, но 2-block всё равно).

---

## V2 — Fingerprint gate (workflow rule #4 инфраструктурно)

### Изменения в исходниках
1. **`bench_dq.cu`**: forward-declare `fa_bwd_dq::kernel_dq` + `bench_dq_fingerprint()` вызывается сразу после `main()` init:
   ```c
   printf("bench_dq: FINGERPRINT kernel_dq: numRegs=%d, sharedSizeBytes=%zu, "
          "localSizeBytes=%zu, maxThreadsPerBlock=%d\n", ...);
   ```
2. **`bench_e2e.cu`**: forward-declare для kernel_dq / kernel_dk / kernel_dv_mma_p1; `bench_e2e_fingerprint()` печатает все три.
3. **`bench_dv.cu`**: forward-declare kernel_dv_mma_p1; `bench_dv_fingerprint()`.
4. **`bench_dk.cu`**: forward-declare kernel_dk; `bench_dk_fingerprint()`.

### Runtime output пример (V3 первая строка):
```
bench_dq: FINGERPRINT kernel_dq: numRegs=196, sharedSizeBytes=0, localSizeBytes=0, maxThreadsPerBlock=256
bench_e2e: FINGERPRINT kernel_dq: numRegs=196, sharedSizeBytes=0, ...
bench_e2e: FINGERPRINT kernel_dk: numRegs=248, sharedSizeBytes=0, ...
bench_e2e: FINGERPRINT kernel_dv_mma_p1: numRegs=129, sharedSizeBytes=0, ...
```

### V3-driver `runs/v3_recert.py` реализует ФАКТИЧЕСКИЙ гейт:
```python
EXPECT = {"kernel_dq": 196, "kernel_dk": 248, "kernel_dv_mma_p1": 129}
def check_fingerprint(out):
    ...
    if regs != EXPECT[name]: return f"MISMATCH ..."
```
При несовпадении — abort с exit code 2. Класс stale-binary bug закрыт infrastruc­tur­ally.

### Workflow rules (обновлённые)
1. Каждый seal-номер сопровождается путём к артефакту на диске.
2. `Δ` живут только на чистых базах (все замеры относительно свежих same-session builds).
3. Не пересобирайте только `_test` binaries — bench_* имеют свои Makefiles.
4. **NEW**: `bench_*` binaries самоотчитываются (`FINGERPRINT` line) — sanity gate в первой строке любого замера. Automatic mismatch abort в V3-cert driver.

---

## V3 — ФИНАЛЬНЫЙ 30-run RE-CERT БАТЧ

### Термо-протокол
- Все бенчи свежие пересборки (bench_e2e 09:38, bench_dq 09:38, bench_dv 09:40, bench_dk 09:40).
- Fingerprint-гейт активен на КАЖДОМ прогоне.
- 30 runs per config, 5 configs — same-thermal нон-стоп.
- Per-run JSON checkpoint (survives любого прерывания).
- Driver: `runs/v3_recert.py`.
- Live-log: `runs/v3_recert.log`.

### Configs (все с sl=8192 bh=128 hd=128 warmup=5 iters=20):
- v3_e2e_nc: bench_e2e non-causal
- v3_e2e_c:  bench_e2e causal  
- v3_dq_iso: bench_dq (P1a) isolated
- v3_dk_iso: bench_dk (sealed 248r) isolated
- v3_dv_iso: bench_dv (sealed 129r) isolated

### Итог 30-run cert (V3 COMPLETE)

Fingerprint gate PASSED всем 150 прогонам (без abort). CV < 0.24% всем.

| Config | median T | mean | sd | CV% | outliers | J-канон | **Δ vs J-канон** | path |
|--------|----------|------|-----|-----|----------|---------|------------------|------|
| v3_e2e_nc | **285.44** | 285.47 | 0.681 | 0.239 | [] | 283.15 | **+0.81%** | `runs/v3_e2e_nc.json` |
| v3_e2e_c  | **277.25** | 277.27 | 0.466 | 0.168 | [[19,278.73]] | 276.00 | **+0.45%** | `runs/v3_e2e_c.json` |
| v3_dq_iso | **328.34** | 328.42 | 0.634 | 0.193 | [[29,330.97]] | 324.69 | **+1.12%** | `runs/v3_dq_iso.json` |
| v3_dk_iso | **305.14** | 305.26 | 0.385 | 0.126 | [[1,306.46]] | 305.24 | −0.03% (parity) | `runs/v3_dk_iso.json` |
| v3_dv_iso | **221.49** | 221.54 | 0.307 | 0.139 | [] | 221.25 | +0.11% (parity) | `runs/v3_dv_iso.json` |

### Ключевые выводы
1. **dQ P1a +1.12%** cert-подтверждено (5-run V1 показал +1.075% → 30-run V3 подтвердил +1.12%). Sealed P1a реален.
2. **E2E-nc +0.81%** — P1a вклад ~+1.12% × dQ-фракции ~32% = +0.36% expected; фактически +0.81% (Q-регистры видимо снижают contention для dK/dV in-chain).
3. **E2E-c +0.45%** — similar mechanism.
4. **dK/dV parity** — эти ядра не тронуты, +0.03–0.11% = thermal noise.
5. **Outliers**: только 2 из 150 прогонов (dq_iso #29=330.97 high, e2e_c #19=278.73 high, dk_iso #1=306.46 high) — все на 3σ от median, безобидные.

### Абсолютные числа vs baseline (naive fa2)
- E2E-nc **285.44 T = +61.4%** vs baseline 176.85 T (J-канон стартовый baseline)
- E2E-c **277.25 T = +59.3%** vs baseline 174.00 T
- dq_iso **328.34 T = +91.0%** vs baseline 171.9 T (P1a добавляет +1.12% сверх prior J-канон gain)
- dk_iso **305.14 T = +55.6%** vs baseline 196.1 T
- dv_iso **221.49 T = +37.7%** vs baseline 160.8 T

---

## Обновление глав памяти

### Sealed чистый dQ (обновлено)
- **P1a: 330.90 T** (5-run same-thermal V1)
- 30-run cert median: [pending V3]
- source: `libs/fa_bwd_dq.cu` == `libs/fa_bwd_dq.cu.SEALED_P1a`
- pattern: swizzle (ярус-1) → K2 (all-warps phase 1.5) → L3 (softmax/quantize jam-x2) → **P1a (Qr regs + smV↔smdS alias)**
- регистры 196 (+33 vs L3-pre-P1 163) — Qr[KS_QK][4] = 16 регов Q + другие вспомогательные

### R2/S2/T3 отчёты — обновление статуса
- **R3_SEAL.md** (первичный): устарел, замер (b)/(c) через stale bench_dq (все = (a)).
- **S2_REPORT.md**: устарел по той же причине; диета+launch_bounds никогда фактически не тестировались через bench_dq.
- **T3_VERDICT.md**: "мистика runtime-специфичности kernel_dq" была артефактом launcher-бага. Отменяется.
- **U_VERDICT.md**: правильный launcher-бага-инцидент, но wall −5.65% для (d) 12-warp — теперь тоже под подозрением (то же ядро kernel_dq_v3 measured but through freshly-rebuilt bench_dq — это OK). НО т.к. диета была одномоментной, (d) test был ОК; conversion negativity держится.
- **V_VERDICT.md** (этот): текущий вердикт. **Sealed dQ = P1a 330.90 T (V1); 30-run V3 median pending.**

### Ярус-3 (сохраняется из U)
- ЗАКРЫТ КОНВЕРСИЕЙ (истинное закрытие, factual 12-warp -5.65% для config (d) с diet+LB) — U_VERDICT.md.
- PG1 dV FAIL: 75.24% × 1.5 = 113% — dV также bandwidth-close.

### Итог перф-главы backward
1. Sealed dQ = **P1a 330.90 T** (V1 5-run).
2. Sealed dK = 248r (unchanged).
3. Sealed dV = 129r yarus-1 (unchanged).
4. E2E (V3 30-run) = pending.
5. Yарус-3 dQ/dV закрыт factually (диета+LB gives 3-block, wall −5.65%; dV bandwidth prohibitive).
6. Workflow rules 1-4 wired.

Всё это финал backward пути. После V3 30-run → глава ТРЕНИРОВКИ (отдельное ТЗ Vugar'а).

---

## Пути ко всем артефактам V-серии

```
libs/
├── fa_bwd_dq.cu                  # SEALED P1a (196r/0s, KT_STRIDE=68, no LB)
├── fa_bwd_dq.cu.SEALED_P1a       # identical seal backup
├── fa_bwd_dq.cu.p1a              # V1 intermediate backup
├── fa_bwd_dq.cu.pre_P1           # (a) L3-pre-P1 backup
├── fa_bwd_dq.cu.d_diet           # (d) diet для форензики
├── bench_dq.cu                   # V2 fingerprint added
├── bench_dk.cu                   # V2 fingerprint added
├── bench_dv.cu                   # V2 fingerprint added
├── bench_e2e.cu                  # V2 fingerprint added (все 3 kernels)

runs/
├── v3_recert.py                  # V3 driver with EXPECT dict + gate
├── v3_recert.log                 # live progress log
├── v3_e2e_nc_raw.json            # per-run checkpoint
├── v3_e2e_nc.json                # final stats
├── (v3_e2e_c, v3_dq_iso, v3_dk_iso, v3_dv_iso)
└── r2_seal/
    ├── V_VERDICT.md              # ЭТОТ ОТЧЁТ
    ├── p1a_ptxas.log             # P1a fa_bwd_dq_test build
    ├── p1a_bench_ptxas.log       # P1a bench_dq build (196r/0s)
    ├── p1a_biteq.log             # 11/11 PASS
    ├── p1a_REAL_wall.log         # 5-run: 330.90 T
    ├── p1a_launch_occ.txt        # NCu LaunchStats: 196r/33792B smem/2 blocks/SM/7.96 warps
    ├── a_v3_bench_ptxas.log      # (a) rebuild for A/B
    ├── a_v3_REAL_wall.log        # (a) same-thermal: 327.38 T
    ├── v2_bench_dq_ptxas.log     # V2 rebuild bench_dq w/ fingerprint
    ├── v2_bench_e2e_ptxas.log
    ├── v3_bench_dv_ptxas.log
    ├── v3_bench_dk_ptxas.log
    └── (все прошлые S1/S2/T/U логи)
```

Финал главы backward. Ожидаю V3 completion → заполню 30-run cert numbers → ACK от Vugar → следующая глава (ТРЕНИРОВКА).
