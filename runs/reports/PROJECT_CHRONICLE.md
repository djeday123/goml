# PROJECT CHRONICLE: goml + gotorch/v6

**Источник:** только `git log` обоих репо + `runs/reports/*.md` + durable-правила из memory.
**Реконструкция по диску, не по памяти.** Формулировки вердиктов — как в отчётах, без улучшайзинга.

**HEAD-hash на момент фиксации:**
- goml `9e1e3f0` (2026-08-02)
- gotorch/v6 `9cf4bc4` (2026-07-26)

---

## Эпоха I — CPU-goml (до 2026-05-19)

**Ранняя история из корней:** `daa0095` (2026-03-03) "Flash atention 153 T" — первый FA-fwd (goml). Далее CPU-first разработка nn/autograd/gputrain — до 4090-эпохи GPU.

**Вердикт эпохи:** framework foundation готов, GPU-путь начинается.

---

## Эпоха II — 4090-эпоха (2026-05-19 → 2026-06-11)

**Задача:** оптимизация FA + FP8 GEMM на sm_89 (RTX 4090).

**Артефакты (goml, chronological):**
- `28284e8` 2026-05-19 fix: critical gradient computation bugs (autograd, nn/norm, attention, rnn)
- `9104a68` 2026-05-24 fix: MatMul/BatchMatMul preserve Float32 dtype
- `f3b070c` 2026-05-24 perf: Welford variance, heap TopK, ConvTranspose2d matmul, pool
- `babc5dc` 2026-06-02 FA v54 backward — minimal viable CUDA kernel
- `4529821` 2026-06-03 FA v55 backward — Tri Dao two-pass tensor-core
- `f6a3418` 2026-06-03 FA v57 backward — 3-stage cp.async pipeline + SMEM recycle
- `91e43eb` 2026-06-03 FA v58 backward — **FP16 accumulator on S, dP MMAs +9-10%**
- `62dd9df` 2026-06-04 FA v55 forward — FP16 accumulator on Or
- `734ec15` 2026-06-04 FP8 forward v61 PoC — kind::f8f6f4 MMA path verified
- `82be57f` 2026-06-04 FP8 forward v62 — full pipeline end-to-end correctness
- `49614eb` 2026-06-04 FP8 v63 — pre-transpose V + HW FP8 cvt (2.2× over v62)
- `dea1f2b` 2026-06-04 FP8 v64 — double-buffered K,V loads with SMEM overlap
- `b56ab51` 2026-06-04 FP8 v65 — bigger tiles (Br=128, 256 threads) +11% large
- `d21d065` 2026-06-04 FP8 v66 — M_TILES=2 per warp +3% over v65
- `e9f913e` 2026-06-07 FP8 v68 — smV_T stride 64→68 padding +15-17%
- `665e9fd` 2026-06-07 FP8 v69 — single-buffer V → 2 blocks/SM **+51%**
- `fbdb868` 2026-06-10 full sm_120a Blackwell FP8 forward journey v68→v117
- `fd2a536` 2026-06-11 sm_120a hd=128 production stack — v118/v121/v122 + M6/M7/M8 mbarrier
- `34d6eee` 2026-06-11 chore(build): commit pre-built sm_120a binaries + libs + DISPATCHER

**Числа-вехи эпохи:** FP8 v66 **587T** (goml/reports/005b_R1c.md), FA-fwd v20 **151T** (в MEMORY.md core), FA-bwd stack up to Blackwell v0.1.0.

**Вердикт эпохи:** FP8 GEMM собственный (587T), FA sm_89→sm_120a перенесён.

---

## Эпоха III — FA-backward сага (2026-07-09 → 2026-07-18)

**Задача:** FA-bwd на sm_120a (Blackwell RTX 6000).

**Артефакты:**
- `4732a38` 2026-07-09 seal: **FP8 bwd v1** (44.206ms E2E / 398T) — merged+dk_new+dq_new+D + L-fwd + W0
- `2314cbe` 2026-07-10 seal: **FP8 bwd v2** (42.35ms E2E / **415.44T** proj) — 061 S2v4 dk_new
- Reports: `runs/reports/001_R1a.md` (первый R1a canonical), `runs/reports/003_R1b.md` (R1b dk), `runs/reports/005_F1_hygiene.md`, `runs/reports/005b_R1c.md` — R-серия (R1a/R1b/R1c) FA-bwd оптимизация.
- `af92ee9` 2026-07-18 state sync before gpu4 deployment: R03a interop artifacts

**Артефакты gotorch/v6:**
- `fc3adda` 2026-07-18 R02b purego backend + R03a interop artifacts

**Числа-вехи:** FA-bwd v0.2.0 **260T** (в MEMORY.md core-memories), первый FP8 FA-bwd 42.35ms, `399→415T` progression via S2v4 techniques.

**Вердикт эпохи:** FP8 FA-bwd работает на Blackwell, готов к встройке в тренировку.

**Durable-правило (образованное):** [[feedback-fa-buffers-zero-init]] — OFP16/LGPU zero-init перед ForwardTrain обязательно; иначе 0x7fff qNaN pattern (класс: контракт на выходные буферы).

---

## Эпоха IV — Мост R03 goml↔gotorch (2026-07-18 → 2026-07-23)

**Задача:** свести goml (backend) и gotorch/v6 (тензорная библиотека) через adapter.

**Артефакты:**
- `af92ee9` 2026-07-18 R03a interop artifacts era (goml)
- `fc3adda` 2026-07-18 R02b purego backend + R03a interop (gotorch)
- `950d2c7` 2026-07-13 docs: README v6.0.0 (gotorch)
- `20d1c5e` 2026-06-01 re-version v6.0.0 first properly working release (gotorch)
- `2567f40` 2026-07-20 ARCHITECTURE.md target stack picture (gotorch)
- `b7e8538` 2026-07-20 ARCHITECTURE.md goml role in stack
- `8db633d` 2026-07-20 R03b-impl-2 SetStream + R03a-89/R03b_design/impl reports (gotorch)
- `5733eb4` 2026-07-20 R03b-impl-1 Fix A + impl-2 adapter package (goml)
- `82a8a1c` 2026-07-23 R03b impl-3..5c: adapter direct + Linear A/B + F64 judge (goml)
- `b0adecb` 2026-07-23 R03b impl-3..5c: MatMulF32_TF32 per-call mode + F64 judge foundation (gotorch)

**Reports:** `runs/reports/R03a_89.md`, `runs/reports/R03b_design.md`, `runs/reports/R02b_final.md` (упомянуты в memory).

**Числа-вехи:** 6/6 PASS R03a interop (Blackwell+MPS sm_120), TF32-adapter **3648×** (в MEMORY.md).

**Вердикт эпохи:** adapter arrow работает, cgo vs purego = 99.95% parity, gotorch v6.0.0 shipped.

**Durable-правила:**
- [[gotorch-r03b-impl4-final]] — TF32-adapter 3648× (MatMulF32_TF32 per-call mode)
- [[gotorch-r03a-uva-works]] — UVA прозрачен на cross-ctx для sm_120a/CUDA 13.0

---

## Эпоха V — Порты P1-P5 (2026-07-23)

**Задача:** портировать RMSNorm, Embedding, RoPE, batched-MatMul из legacy goml в gotorch/v6 (delegate→direct).

**Артефакты (единый день, оба репо):**
- `045af64` (goml) / `6525de8` (gotorch) 2026-07-23 P1-ABJ 10-step exam via gputrain path
- `cd73dfe` / `2dac5a9` 2026-07-23 **P2-RMS: RMSNormF32/F64 fwd+grad, 50→54 methods**
- `fd3891b` / `59cfbfa` 2026-07-23 **P3-EMB: EmbeddingF32/F64 fwd+grad, 54→58 methods**
- `f9ff5fa` / `8372311` 2026-07-23 **P4-ROPE: RoPE F32/F64 fwd+grad, 58→62 methods**
- `7c1385a` / `63e5e3b` 2026-07-23 **P5A: int32 canonical + int64 facade Embedding, 62→66 methods**
- `d5576ac`/`a6b56b8` (gotorch) 2026-07-23 P5B paper batched FP16/FP8 MatMul (design + reclassify)
- `998b434` 2026-07-23 P2_RMS: reframe orphan finding

**Reports:** `runs/reports/P2_RMS.md`, `P3_EMB.md`, `P4_ROPE.md`, `P5A.md` (упомянуты в memory).

**Числа:** 50→75 methods (F64 parity), 5/5 PASS.

**Вердикт эпохи:** legacy portage завершён.

**Durable:** [[feedback-atomicadd-drift-oscillation]] (P3-EMB), [[feedback-launcher-param-naming]] (P4-ROPE), [[feedback-purego-arg-limit]] (P5A struct-args), [[feedback-cublas-handle-per-dlopen]] (P5A wrapper).

---

## Эпоха VI — B-серия wrapper-парк (2026-07-23 → 2026-07-24)

**Задача:** batched MatMul через C wrapper (F32/F64/F16/F8) — legacy safety + mixed-precision.

**Артефакты:**
- `e384684` 2026-07-23 **legacy safety fix: OOB read in broadcast batched matmul — sole read-only exception**
- `cb1e253` / `e4b2e7c` 2026-07-23 **B-impl-1: F32/F64 strided batched MatMul, 66→68 methods**
- `d479209` / `09e4588` 2026-07-23 B-impl-1 доработка: native strided через C wrapper + broadcast UB fix
- `956505f` 2026-07-23 unskip 3.3/4.3 tests — оба пути честные
- `1a5bbc5` (gotorch) 2026-07-23 **B-impl-2: F16 mixed precision MatMul, 68→72 methods** (F16 essentially bit-exact vs F32)
- `b7250e8` (gotorch) 2026-07-23 **B-impl-3: FP8 E4M3 Quantize/Cast + cublasLt anchor**
- `29d3e4d` (goml) / `c1f8b71` (gotorch) 2026-07-23 **B-impl-4: mixed-precision Step + A/B/J exam (F32/F16/F8)**
- `5ea1dbc` / `6732114` 2026-07-24 **B2-BATTLE: battle-scale exam — F16 bit-exact, F8 diagnostic (per-tensor amax collapse), Amdahl 99.97% CPU-backward**

**Reports:** `runs/reports/B_impl_1.md`, `B_impl_2.md`, `B_impl_3.md`, `B_impl_4.md`, `B2_BATTLE.md`.

**Числа-вехи:**
- B-impl-2: F16 bit-exact 1.6e-6 (MEMORY [[gotorch-r03b-b2-battle-closed]])
- B-impl-4: F16 bit-exact vs F32; F8 diagnostic per-tensor amax недостаточен
- B2-BATTLE **Амдал 99.97% CPU-backward** (главный anchor далее — оптимизация bwd)

**Вердикт эпохи:** F16 mixed OK, F8 требует per-block scaling. **CPU-backward = следующая цель.**

**Durable:** [[feedback-amax-freeze-delayed-scaling]] (B2 F8 плато = per-tensor amax + live A), [[feedback-cv-gate-strict]] (>1% в 30-run без разбора).

---

## Эпоха VII — A-серия battle-цепочка (2026-07-24 → 2026-07-25)

**Задача:** снять CPU-backward Амдала (B2 99.97%) — GPU-bwd, fused CE, scratch cache, prod-Step.

**Артефакты:**
- `32f36b7` (goml) / `0010aae` (gotorch) 2026-07-24 **A-0: matmul-backward на GPU через wrapper — 35× speedup**
- `1c055a5` / `71f2f09` 2026-07-25 **A-1: fused cross_entropy_f32 GPU-kernel — 246.8× vs B2**, устраняет 81% wall'а A-0
- `9e7b53a` / `b2ef866` 2026-07-25 **A-2: BattleScratch buffer cache — bit-exact vs A-1, CV 3.90% → 1.85%**
- `0398142` / `a0ac444` 2026-07-25 **A-3: prod-Step no test-D2H — 772× vs B2, CV 0.27%** (первый честный) — battle-цепочка ЗАКРЫТА
- `55c566c` (gotorch) 2026-07-25 A-3: форензика inline вердикт, FA-canary single-corridor

**Reports:** `A_0.md`, `A_1.md`, `A_2.md`, `A_3.md`.

**Числа-вехи (все в MEMORY.md core):** **35.2× → 246.8× → 255.7× → 772.2×** (vs B2 baseline).
CV: 3.90% → 1.85% → 0.27% (<1% gate).
Wall canary re-anchor 654±2T; CE 88.4% wall.

**Вердикт эпохи (durable):** "battle-цепочка ЗАКРЫТА" — 772× vs B2, CV <1% gate. Три durable feedback [[gotorch-r03b-battle-closed]]:
- [[feedback-lockosthread-battle-scale]] — primary-ctx per-thread INVALID_CONTEXT на battle scale
- [[feedback-cpumap-anomaly-verification]] — блок >2× / >50% wall = верифицировать 2-м методом
- [[feedback-sqrt-vs-linear-accumulator]] — GPU O(sqrt(K)·eps), 4-е применение

---

## Эпоха VIII — A-LLM-1 (2026-07-25 → 2026-07-26)

**Задача:** полноценный nn.LLM трансформер + FA-fwd + временный F32 attention-bwd.

**Артефакты:**
- `234288b` (gotorch) 2026-07-25 A-LLM-1 Этап 1: разведка nn.LLM + BattleA config + FA sources
- `9a05310` / `19cdcc7` 2026-07-25 **A-LLM-1 G1: fa_forward_train (LSE-fwd) сборка + purego binding + L-correctness PASS**
- `03331b0` / `06a1078` 2026-07-25 **A-LLM-1 G2: libfa_bwd_sm120.so bwd chain + fingerprint gate PASS**
- `7865451` / `8c93bcb` 2026-07-26 **A-LLM-1 Stage 3: forward transformer BattleA — B=1 smoke + B=4 repack bit-exact PASS**
- `913470d` / `9cf4bc4` 2026-07-26 **A-LLM-1 Stage 4: F32-attention-reconstruct BWD + grad-consistency CERT PASS 4/4 + FA-out-zero discovery**

**Reports:** `A_LLM1.md` (обновлён 4 раза).

**Числа-вехи:** Stage 3 loss `10.47` (B=1 smoke), B=4 repack `2.15e-6 = amax cross-batch coupling`. Stage 4 CERT 4/4 `1e-8` precision.

**Открытие (discovery):** `fa_forward_train` ALL ZEROS во всех Stages — **fresh disaster** [[gotorch-r03b-allm1-stage4-closed]].

**Вердикт эпохи:** трансформер дышит, F32 attention-recon как эталон для BWD, FA-bwd путь **отложен пока FA-fwd не вернёт non-zero** для training.

---

## Эпоха IX — A-LLM-2 охота за эталоном (2026-07-26 → 2026-08-02)

**Задача:** сделать корректный F32 backward reference (эталон) → потом A/B встройка FA-bwd.

### v1/v2 attempts + phantoms (07-26 → 07-29)
- `65755ad` 2026-07-26 A-LLM-2 v1: bwdBattleA F32-recon + trainStep + sign-of-life PASS
  **Вердикт из отчёта:** PASS отозван (caveat 1 = мат. ошибка); фантом номер 1.
- `f4d97cc` 2026-07-29 A-LLM-2 v2 attempt: F32-only stack — partial
- `a2ddc24` 2026-07-29 v2 hypothesis triage: bug HD-specific + MatMulF32Ex zero output
- `3f936b9` 2026-07-29 v2 GEMM zero hunt: async race softmax_bwd_f32 + MatMulF32Ex
- `cec42fc` 2026-07-29 v2 streams verified same + sign flip systemic

### Б-0 протокол установлен (07-29)
Race #1-3 диагностированы. Sync-fix accepted as workaround, но **на одном stream race невозможен физически** — durable [[feedback-ptx-cublas-sync]] переформулирован.

### v2-фазы Б-1..Б-3 + Ходы 1-2 (07-30)
- `50a7e23` 2026-07-30 Б-1 root cause: cublas GemmEx first-call ZERO per (m,n,k,transA,transB)
- `b19c764` 2026-07-30 Б-1+3: warmup гипотеза ОТКЛОНЕНА, dWq stuck dead
- `cf027f3` 2026-07-30 **Ход-1/2 BREAKTHROUGH: MatMulF32Ex flush-to-zero локализован** — dQ=dS@K exact zero → plain b.MatMul fix
- `b7566a0` 2026-07-30 Ход-2 full replacement: attention chain восстановлен (все 4 attn_recon matmul), W1 systematic 50%

### Ходы A.1/F (07-30)
- `23530fe` 2026-07-30 Ход A.1: grep-mishen empty (double-count не найден), nonlinearity ruled out
- `7f41ada` 2026-07-30 **Ход F: точный bisect — Wout CLEAN 35ppm, matmul-wrapper isolated OK** (7 grad points таблица)

### Ход D BREAKTHROUGH (07-30)
- `efd833c` 2026-07-30 **Ход-D BREAKTHROUGH: эталон certified 5/7 sub-1e-3, F32 floor achieved**
  Raw числа:
  ```
  Wout=3.57e-05, Wo=3.21e-04, W2=1.12e-03, W1=2.25e-04, Wv=3.26e-04 — TIGHT
  Wq relDiff=2.11e-02 (absDiff=4e-6 F32-floor), Embed relDiff=1.07e-02 (7e-6 F32-floor)
  Sign-of-life: 3.4932 → 2.6154 Δ=+0.878
  ```

### Ход P + wrapper state-dep (07-31 → 08-02)
- `256db7d` 2026-07-31 **Ход-P: sign-of-life LIVE + mixed-floor formula, L=4 deferred, wrapper state-dep**
  Обнаружено: cert numbers **нон-детерминистично меняются между runs** на identical code+seed.
- `6516059` 2026-07-31 де-wrapper attempt: transpose logic bug, reverted; standalone RMSNormGrad probe canonicalized (RMSNormGradF32 kernel CLEAN maxRel 1.6e-5)
- `9e1e3f0` 2026-08-02 **де-wrapper v2: 10 MatMulF32Ex → plain, unit-tested; determinism-gate STILL FAIL**
  ```
  matmulPlainT unit-test PASS 17/17 (helper математически correct)
  cert: 5/7 tight на run-1, но dWq max|Δ|=3.362e-03, dW1=6.57e-02, dEmbed=9.42e-02 (run-1 vs run-2)
  DETERMINISM FAIL — non-det ГЛУБЖЕ wrapper
  ```

**Reports/Артефакты:** `runs/reports/HANDOFF_ref_dewrapper.md` (соседний файл), commit messages в `runs/commit_msg_*.txt`.

**Фантомы (методологическая ценность):**
1. **v1 sign-of-life PASS** — отозван (caveat 1)
2. **"Cert PASS L=1 tight + L=4 first-time"** — user's 5-strike phantom при межсессионном hand-off; все 5 pushback-cycles методологически ценны
3. **Race #4 "measurement instrument in cert-читалке"** — прогноз пользователя ОТКЛОНЁН по grep (Sync at line 125-126 уже был; cert Wq relDiff=1.000 exact подтвердил)

**Мусор=мусор случаи (методология: input тень маскирует bug):**
1. Wq dead → маскировал W1/Embed 50% systematic (Ход A.1)
2. dNormedTop подозревался (F-PROBE) → matmul CLEAN, bug ниже (Ход D)
3. cublasGemmEx flush-to-zero гипотеза → magnitude standalone OK (Ход-B), context-dependent

**Числа-вехи (все под соответствующим commit-hash):**
- **Ход-1/2:** dS@K exact zero → alive 4.6e-4; Wq relDiff 1.0 → 0.635 (single-line change)
- **Ход-2:** attn chain restored, W1 50% systematic remains
- **Ход-D:** 5/7 sub-1e-3 tight; F32-floor 4-7e-6 abs
- **Ход-P:** sign-of-life 3.4932 → 2.6154 Δ+0.878 (single run efd833c); wrapper state-dep confirmed
- **Ход de-wrap v2:** 5/7 tight run-1 подтверждено; determinism FAIL |Δ|=0.09 for Embed

**Вердикт эпохи (текущий):** математика эталона верна с первого дня; неполадки — инструментальная грязь (4 слоя: race/wrapper/context-dep/wrapper-state), пойманы и локализованы. **Non-determinism глубже gt_gemm_ex** — cublasSgemm сам подозревается + EmbeddingGradF32 atomicAdd. Открытый долг wrapper повышен до критического, FA-встройка блокирована determinism gate.

**Durable-правила (образованные эпохой IX):**
- [[feedback-atomicadd-drift-oscillation]] — RMSNormGrad dgamma atomic non-det (класс)
- [[feedback-sqrt-vs-linear-accumulator]] — 4-е применение (mixed-floor формула)

---

## Реестр durable-правил (birth+incident)

| Правило | Родилось | Инцидент | Применения |
|---------|----------|----------|-----------|
| [[feedback-denominator-hygiene]] | 2026-06 | %-of-peak MEASURED not GUESS | FA-fwd 62%→36% reframed |
| [[feedback-ncu-estimated-speedup-mirage]] | 2026-06 | NCu per-issue overestimates ~3× occupancy-bound | Все NCU FA analysis |
| [[feedback-peak-vs-real-workload]] | 2026-06 | peak-loop ≠ GEMM ≠ FA | 3 workload regimes |
| [[feedback-ptx-jit-log-diagnostic]] | 2026-07-19 (R02b) | logBuf catch за секунды | R02b + P2-RMS |
| [[feedback-atomicadd-drift-oscillation]] | 2026-07-23 (P3-EMB) | Drift ~O(eps·N) not sqrt(N)·eps | P3-EMB промах 44× |
| [[feedback-launcher-param-naming]] | 2026-07-23 (P4-ROPE) | maxRel=1.0 zero-pos identity | P4-ROPE |
| [[feedback-purego-arg-limit]] | 2026-07-23 (B-impl-1) | v0.9.1 panic на 18+ args | struct-args wrapper .so |
| [[feedback-cublas-handle-per-dlopen]] | 2026-07-23 (B-impl-1) | NOT_INITIALIZED | wrapper local handle |
| [[feedback-lockosthread-battle-scale]] | 2026-07-25 (A-3) | primary-ctx per-thread INVALID_CONTEXT | Lock+defer Unlock в Step |
| [[feedback-detached-long-runs]] | 2026-07-24 (A-0) | setsid+nohup+marker | Все >5min |
| [[feedback-sqrt-vs-linear-accumulator]] | 2026-06 | GPU O(sqrt(K)·eps), 4-е применение | A-0 durable, mixed-floor formula (2026-08-02) |
| [[feedback-cv-gate-strict]] | 2026-07-25 (A-2/A-3) | >1% в 30-run FA без разбора | A-3 CV 0.27% первый честный |
| [[feedback-cpumap-anomaly-verification]] | 2026-07-25 (A-3) | 2-й метод при >2× или >50% wall | форензика A-3 |
| [[feedback-f32-finite-diff-lies]] | 2026-07-26 (S4) | FP32 GPU accum + D2H = ~eps*scale | F64-arbiter recomm |
| [[feedback-rope-grad-layout]] | 2026-07-26 | RoPE bwd на том же layout reversible | Не inverse kernel |
| [[feedback-amax-freeze-delayed-scaling]] | 2026-07-24 (B2 F8) | per-tensor amax на живом A = quant noise | delayed scaling |
| [[feedback-fa-buffers-zero-init]] | 2026-07-26 | OFP16/LGPU zero-init contract | 4 bwd kernels |
| [[feedback-fa-fwd-scratch-alloc-instability]] | 2026-07-26 (A-LLM-1 v2) | allocation triggers NaN | blocks v2 caveat-1 |
| **Б-0 protocol** | 2026-07-29 | commit-hash + raw output | 8 sessions applied |
| One-change-at-a-time | 2026-07 | wall-clock only converts на WORK-REDUCTION | v96b/v118/v121 |
| PTX-ASCII catch | 2026-07-24 (A-1) | non-ASCII in PTX comments | fused CE probe |
| Race-семейство (4 случая) | 2026-07-29..07-30 | ptx→cublas, sync, streams, «измерительный прибор» | Ход-P (последний) |
| Мусор=мусор (3 случая) | 2026-07-30 | Wq dead → 50% W1 masked; F-PROBE mismatch; standalone ≠ contextual | Ход A.1, F, B |

**Правило-мета (living-doc — устанавливается ЭТОЙ хроникой):**
> В конец каждой сессии обновлять `HANDOFF_ref_dewrapper.md` и `PROJECT_CHRONICLE.md` — one paragraph per session, hash + numbers + verdict. Non-optional.

---

## Реестр чисел-вех (с источником-артефактом)

| Число | Что | Источник (commit / report) |
|-------|-----|---------------------------|
| **587T** | FP8 GEMM v66 own SGEMM sm_89 | `d21d065` 2026-06-04 + `runs/reports/005b_R1c.md` |
| **151T (or 153T)** | FA-fwd v20 sm_89 | `daa0095` 2026-03-03 (title "Flash atention 153 T"); memory core "v20 151T" |
| **415.44T** | FP8 bwd v2 42.35ms E2E | `2314cbe` 2026-07-10 |
| **260T** | FA-bwd v0.2.0 (Blackwell) | MEMORY.md core-memories |
| **652.40T** | FA-fwd hd=128 bh=128 v121r | [[sm120-fa-forward-final-champion-numbers]] |
| **647.14T** | FA-fwd hd=128 bh=64 sl=8192 v121r | [[sm120-fa-forward-final-champion-numbers]] |
| **466T** | FA-fwd hd=64 v89 P-in-regs | [[sm120-fa-v89-pinregs-positive]] |
| **413T** | FA-fwd hd=64 FINAL | [[sm120-fa-hd64-final-summary]] |
| **3648×** | R03b-impl-4 TF32-adapter | `82a8a1c` 2026-07-23 |
| **35.2×** | A-0 GPU-bwd vs B2 | `32f36b7` 2026-07-24 |
| **246.8×** | A-1 fused CE vs B2 | `1c055a5` 2026-07-25 |
| **255.7×** | A-2 BattleScratch vs B2 | `9e7b53a` 2026-07-25 |
| **772.2×** | A-3 prod-Step vs B2 | `0398142` 2026-07-25 |
| **99.97%** | B2 Amdahl CPU-bwd | `5ea1dbc` 2026-07-24 + `B2_BATTLE.md` |
| **33ms** | attention-bwd wall (в MEMORY memory) | User сообщение (938→33ms) — phantom-класс, не на диске cert |
| **9.42e-02** | dEmbed max|Δ| run-1 vs run-2 (determinism FAIL) | `9e1e3f0` 2026-08-02 |
| **3.4932 → 2.6154 (Δ+0.878)** | Sign-of-life single-run efd833c | `efd833c` 2026-07-30 |
| **3.4932 → 2.6892 (Δ+0.804)** | Sign-of-life de-wrap v2 | `9e1e3f0` 2026-08-02 |

---

## Реестр открытых долгов (с ссылками)

| Долг | Ссылка | Приоритет |
|------|--------|-----------|
| **GPU-bwd нон-детерминизм (экс "wrapper-следствие")** | `2d576c1` A/B N=5, raw_allm3 | ПЕРЕКЛАССИФИЦИРОВАН (П-6б, 2026-08-03): свойство пути, задокументирован числом (DWq 3.5e-3, DW2 5.8e-2, DEmbed 2.2e-2, DWo процессо-зависимо 0..1.2e-1). НЕ блокирует FA-встройку — эталон = CPU-F64 |
| **cublasSgemm determinism** | `2d576c1`: fwd bit-det 5/5, bwd плавает | ПЕРЕКЛАССИФИЦИРОВАН (П-6б, 2026-08-03): часть свойства пути; standalone probe снят с CRITICAL |
| **Causal=1 не сертифицирован** (ядра реализуют: fa_bwd_merged_v1.cu:146,242; все cert causal=0) | `2d576c1` П-5 raw, A_LLM3_f64ref.md | HIGH — сертификация causal=1 + causal-ветка F64-эталона перед автогрессивным LM |
| **FA-F16-вход (Q/K/V half вместо e4m3)** — обоснование: индустриальный warmup = сотни-тысячи шагов, 938ms F32-recon на шаг непозволительны как боевой путь (F32-recon легален только как тестовая проводка warmup-звена). Математика ядер сохраняется, меняется входной тракт (2x трафик/SMEM) + HMMA вместо QMMA; на sm_120a FP16-пик = FP8-пику. Прогноз скорости: между 127T (v58-класс) и 260T. Первый аргумент карточки — вилка скейлинга из A-LLM-4 п.2.4b. Возврат в fa-blackwell-fp8 с полной NCu-дисциплиной | ТЗ A-LLM-4 (ревью 2026-08-03) | MID — после warmup-теста, перед индустриализацией |
| **EmbeddingGradF32 atomicAdd → sort-scatter** | HANDOFF, ptx_kernels.go embedding kernel | HIGH |
| **6 MatMulF32Ex в BH>1 attention_recon** | attention_recon.go L115,121,221,224,238,241 | HIGH (при multi-head cert) |
| **warmup-звено A-LLM-3** | User strategy — F16 first N steps + FP8 by |grad|-median | HIGH (после встройки) |
| **лестница 2-3 ступени (после встройки)** | User strategy | MID |
| **fp8-рецепт полный (per-block scaling, delayed)** | [[feedback-amax-freeze-delayed-scaling]] | MID |
| **L=4-init GPT-2 sqrt(2·L)** | HANDOFF section "L=4 deferred" | MID (не блокирует встройку) |
| **CE warp-shuffle** | Task #103 completed, deferred в nn.LLM | LOW |
| **#79 P3-EMB floor review** | Tasks list | LOW |
| **multi-GPU** | Не начато | LOW |
| **CV-gate: 30-run FA-класс** | [[feedback-cv-gate-strict]] | ACTIVE (при бенче) |
| **F1 hygiene (canary_sanitizer)** | `005_F1_hygiene.md` | CLOSED |

---

## Хроника в one paragraph — правило living-doc

**Правило (2026-08-02, hash `9e1e3f0`):** каждая session-close добавляет один параграф в конец этого файла. Формат: `## 2026-XX-XX (`<hash>`) — <session-purpose>` + 2-4 строки: что делали / вердикт-строка с числом / артефакт (commit/report). Never edit past paragraphs — только append.

---

## 2026-08-02 (`9e1e3f0`) — де-wrapper v2 + determinism-gate

Full de-wrapper 10 MatMulF32Ex → plain b.MatMul в bwdBattleAF32 + attention_recon BH=1 fwd. Unit-test `TestMatmulPlainT_Unit` PASS 17/17 (helper математически correct). Cert run-1 воспроизвёл efd833c 5/7 tight; **determinism-gate FAIL** (dEmbed |Δ|=9.4e-02 между run-1 и run-2 в том же процессе). Sign-of-life LIVE Δ+0.804. Заключение: non-determinism ГЛУБЖЕ gt_gemm_ex — cublasSgemm или EmbeddingGradF32 atomicAdd. Артефакты: `HANDOFF_ref_dewrapper.md`, `PROJECT_CHRONICLE.md` (этот файл). Открытый долг wrapper повышен до CRITICAL, FA-встройка блокирована.

## 2026-08-02 16:20 (`8b3cf23`) — ревьюерская сессия: смена магистрали на CPU-F64

(Параграф добавлен задним числом 2026-08-03 по П-6а — living-doc правило было нарушено в день установления.) Ревьюер переопределил задание: GPU-недетерминизм = физика параллелизма, НЕ дефект; fallback (reference bwd на CPU-F64) повышен до магистрали. 6-шаговая программа: bwdBattleAF64 формулами → det-gate bit-exact → F64 finite-diff → переклассификация GPU-F32-recon в «первый измеряемый» → sign-of-life → иерархия эталонов. Артефакт: `8b3cf23` (HANDOFF, doc-only). Позже тем же днём: ревью-дополнение П-0..П-6 принято Вугаром, вписано в HANDOFF коммитом `3770056` (усиления: dgamma зеркалированием PTX; causal-контракт — обязательный пункт сессии).

## 2026-08-03 (`2d576c1`) — A-LLM-3: CPU-F64 эталон, det-gate PASS, GPU-F32 переклассифицирован

`battle_a_llm_f64ref.go`: полный fwd/bwd cert-формы в F64, ноль GPU, dgamma зеркалирован с PTX (формула совпала с учебником), без causal mask, без map-итераций. Шаг 2 PASS: все 11 тензоров bit-exact Δ=0.000e+00 в процессе + fresh subprocess (hash 7ca2a06774090ed6). Шаг 3: гейт ТЗ eps=1e-6 — прогноз-промах 8/10 (шумовой пол δ_L≈1 ulp; контр-прогноз исполнителя записан до прогона); исправлен инструмент (Richardson, eps²-модель подтверждена скейлингом 100.6×/декаду), **CERT v2 PASS 10/10 rel≤1e-8** (лучшие точки 1e-11). Санити: F64 ana == канон efd833c до 6-7 знаков. Шаг 4: A/B N=5 — fwd bit-det 5/5, bwd нон-детерминизм задокументирован (DWq 3.5e-3, DW2 5.8e-2, DEmbed 2.2e-2, DWo процессо-зависимо 0..1.2e-1, мёртвая cert-точка Wo в одном прогоне); formula-floor провален 9/10 (прогнозировано) — floor пути = worst|Δ| из raw. Sign-of-life 3.4932→2.7121 Δ=+0.7810. П-5: стек когерентен в causal=0 (raw grep). Отчёт: `A_LLM3_f64ref.md`, raw: `raw_allm3/`. Иерархия действует: CPU-F64 арбитр → GPU-F32-recon (первый измеряемый) → FA-FP8 (встройка = следующая сессия).
