# HANDOFF: reference-ветка де-wrapper-изация + determinism-gate

**Дата закрытия:** 2026-07-30 (last-session HEAD `9e1e3f0`)
**Автор:** Claude Opus 4.7 предыдущей сессии
**Читай ПЕРВЫМ:** этот файл + `git log -1` + повторный прогон последнего зелёного unit-test.

---

## Б-0 старт следующей сессии (обязательный протокол)

Первые команды дословно:

```bash
cd /data/lib/podman-data/projects/goml
git log -1
# ожидается: 9e1e3f0 A-LLM-2 де-wrapper v2: 10 MatMulF32Ex → plain, unit-tested; determinism-gate STILL FAIL

git status --short
# ожидается: только libs/S2v4_bridge_probe_060, libs/bench_* — untracked wilds (не мешают)

# Повторный прогон последнего зелёного теста:
env GOTORCH_LIBS_DIR=/data/lib/podman-data/projects/gotorch/v6/libs \
  go test -run 'TestMatmulPlainT_Unit$' -timeout 60s -v ./internal/abjexam/
# ожидается: PASS 17/17 (12 small + 5 prod shapes)
```

Если TestMatmulPlainT_Unit **не PASS** — стоп, разбирать до дальнейших шагов.

---

## Что зафиксировано на диске (commit-hash + факты)

### commit `9e1e3f0` (последний, HEAD)
- **matmul_plain_helper_test.go** создан: `TestMatmulPlainT_Unit` PASS 17/17. Helper математически корректен (small maxAbs 1e-7, prod maxAbs 1.3e-6).
- **battle_a_llm_f32recon.go** де-wrapper 7 transA=T calls:
  - line 422: `matmulPlainT(grads.DWout, sc.Normed, sc.GradL, D, V, M, true, false)`
  - line 548: `matmulPlainT(lg.DW2, sc.FFNSilu, bs.DFFNOut, FFN, D, M, true, false)`
  - line 565: `matmulPlainT(lg.DW1, bs.NormedRecomp, bs.DFFNHidden, D, FFN, M, true, false)`
  - line 593: `matmulPlainT(lg.DWo, sc.Q, bs.DAttnOut, D, D, M, true, false)`
  - line 735: `matmulPlainT(lg.DWq, bs.NormedRecomp, bs.DQ, D, D, M, true, false)`
  - line 741: `matmulPlainT(lg.DWk, bs.NormedRecomp, bs.DK, D, D, M, true, false)`
  - line 744: `matmulPlainT(lg.DWv, bs.NormedRecomp, bs.DV, D, D, M, true, false)`
- **attention_recon.go** BH=1 fwd (line 67–95) — plain matmul для `S=Q@K^T` (host-transpose K) + `O=P@V` (без trans).
- **battle_a_llm_f32recon_cert_test.go** — mixed-floor formula + determinism-gate добавлены.

### commit `efd833c` (canonical breakthrough — first 5/7 TIGHT)
Numbers на этом hash (single-run):
```
Wout(top)       ana=-4.972173e-02  num=-4.971996e-02  relDiff=3.566e-05  TIGHT
Wo[L=0]         ana=-1.053664e-02  num=-1.053326e-02  relDiff=3.207e-04  TIGHT
W2[L=0]         ana=-2.430252e-03  num=-2.432987e-03  relDiff=1.124e-03  TIGHT
W1[L=0]         ana=-7.551747e-03  num=-7.550046e-03  relDiff=2.254e-04  TIGHT
Wv[L=0]         ana=-7.561454e-03  num=-7.558987e-03  relDiff=3.264e-04  TIGHT
Wq[L=0]         ana=+1.867208e-04  num=+1.907349e-04  relDiff=2.105e-02  (F32-floor abs 4e-6)
Embed[i0,d]     ana=+6.328002e-04  num=+6.396323e-04  relDiff=1.068e-02  (F32-floor abs 7e-6)
Sign-of-life:   3.4932 → 2.6154 (Δ=+0.8778)
```

### commit `9e1e3f0` (последний cert прогон — determinism confirmed FAIL)
```
Run-1 cert (в процессе, ana сразу после первого bwd):
Wo ana=-1.054e-02  (matches efd833c)  но:
Run-2 (после zeroGrads+2nd bwd):
Wo ana=-1.329e-02  (DIFFERENT)
dWq max|Δ|=3.362e-03, dW1 max|Δ|=6.570e-02, dEmbed max|Δ|=9.417e-02
DETERMINISM FAIL
Sign-of-life STILL LIVE: 3.4932 → 2.6892 (Δ=+0.8040)
```

**Non-determinism ГЛУБЖЕ wrapper.**

---

## Оставшиеся `gtB.MatMulF32Ex` в reference-путях (не устранены)

По grep `internal/abjexam/attention_recon.go`:
- **line 115**: `MatMulF32Ex(qsb, kb, sb, S, S, HD, false, true)` — BH>1 fwd, S=Q@K^T per batch
- **line 121**: `MatMulF32Ex(pb, vb, ob, S, HD, S, false, false)` — BH>1 fwd, O=P@V per batch
- **line 221**: `MatMulF32Ex(pb, dob, dvb, S, HD, S, true, false)` — BH>1 bwd, dV=P^T@dO
- **line 224**: `MatMulF32Ex(dob, vb, dpb, S, S, HD, false, true)` — BH>1 bwd, dP=dO@V^T
- **line 238**: `MatMulF32Ex(dsb, kb, dqb, S, HD, S, false, false)` — BH>1 bwd, dQ=dS@K
- **line 241**: `MatMulF32Ex(dsb, qb, dkb, S, HD, S, true, false)` — BH>1 bwd, dK=dS^T@Q

**Все 6 в BH>1 loop path.** Cert L=1 config = BH=1 — они НЕ исполняются в cert. Но при переходе к BH>1 (например, L=4 с B*H > 1) — исполнятся.

По grep `battle_a_llm_f32recon.go` — только comments/print references (no active gtB.MatMulF32Ex).

---

## Non-determinism пути (не устранённые)

По grep `battle_a_llm_f32recon.go` активные `gtB.*` calls (PTX kernels):

| Line | Call | Атомарность | Влияние на det |
|------|------|-------------|----------------|
| 204/285/311 | `RMSNormF32(fwd)` | Non-atomic | OK детерм. |
| 237/240 | `RoPEF32` | PTX | OK детерм. |
| 480/569/748 | `RMSNormGradF32` | **dgamma atomic.add.f32** (line 2002 в gotorch/v6/cuda/ptx_kernels.go) | dNormOut/dNorm1/dNorm2 нон-детерм. — но НЕ влияет на dW* chain |
| 678/681 | `RoPEGradF32` | PTX | OK детерм. |
| 536/712 | `RMSNormF32(recompute)` | Non-atomic | OK детерм. |
| **756** | **`EmbeddingGradF32`** | **atomicAdd scatter** | **dEmbed нон-детерм.** — совпадает с наблюдаемым Δ=9.4e-02 |

Плюс через `b.MatMul` → cublasSgemm который **потенциально non-det на tensor-core paths с эвристикой autoselect**. Не проверено standalone-тестом.

---

## Дословные команды прогонов

**Unit-test helper (первое что надо запустить):**
```bash
env GOTORCH_LIBS_DIR=/data/lib/podman-data/projects/gotorch/v6/libs \
  go test -run 'TestMatmulPlainT_Unit$' -timeout 60s -v ./internal/abjexam/
```

**Cert + determinism-gate + sign-of-life (основной прогон):**
```bash
env GOTORCH_LIBS_DIR=/data/lib/podman-data/projects/gotorch/v6/libs \
  go test -run 'TestALLM_BwdCertF32_MultiLayer$' -timeout 180s -count=1 -v ./internal/abjexam/
```

**Standalone RMSNormGrad probe (kernel CLEAN):**
```bash
env GOTORCH_LIBS_DIR=/data/lib/podman-data/projects/gotorch/v6/libs \
  go test -run 'TestRMSNormGrad_CPUF64_Arbiter$' -timeout 60s -v ./internal/abjexam/
```

**Zero-magnitude probe (context-dep zero диагностика, справочно):**
```bash
env GOTORCH_LIBS_DIR=/data/lib/podman-data/projects/gotorch/v6/libs \
  go test -run 'TestALLM_MatMulF32Ex_SmallMagnitude$' -timeout 90s -v ./internal/abjexam/
```

---

## Состояние тестов (state matrix)

| Test | State | Notes |
|------|-------|-------|
| `TestMatmulPlainT_Unit` | **PASS** 17/17 | Helper корректен, F32 accum floor 1.3e-6 abs, small-shape bit-exact |
| `TestRMSNormGrad_CPUF64_Arbiter` | **PASS** | maxRel 1.6e-5 F32 territory; RMSNormGradF32 kernel CLEAN на D=128 M=32 |
| `TestALLM_BwdCertF32_MultiLayer` | **FAIL** | 5/7 tight (Wq/Embed F32-floor abs); determinism-gate FAIL run1 vs run2 |
| `TestALLM_MatMulF32Ex_SmallMagnitude` | PASS-справочно | Все magnitude 1e0–1e-5 ALIVE в standalone; magnitude-triggered hypothesis REJECTED |
| `TestALLM_MatMulF32Ex_ZeroRepro` | PASS-справочно | Pattern-777 на 5 shape combos; wrapper standalone работает |

**Cert определяется как FAIL** из-за determinism-gate. Сами 5/7 grad points на run-1 — TIGHT.

---

## Грабли контекста (обязательно знать)

1. **matmulPlainT helper** — inline в bwdBattleAF32 (battle_a_llm_f32recon.go line ~385).
   - `hostTr(h, rows, cols)` — inline рядом (line ~380).
   - **CERT-ONLY**: 3× D2H+CPU-transpose+H2D на каждый вызов. **НЕ для скоростных путей.**
   - Full-flag: (transA, transB) bool — покрывает все 4 combo.
   - Legacy `matmulPlainTB` (transB-only) остался как wrapper над matmulPlainT для совместимости с ранее написанным кодом.

2. **D-hop instrumentation** только для `l == 0` (в bwdBattleAF32 line 505-515 для DFFNOut/DFFNSiluSnap/DFFNHidSnap Copy). **Для L>1 D-hop-2 diff НЕ 0** — потому что snap на последней итерации loop, но dX модифицировался в предыдущих. Не bug — конструктивная особенность.

3. **Warmup в cert-тесте**: `skipWarmup = true` в battle_a_llm_f32recon_cert_test.go. Fresh init на L=1 стабилен. **L=4 требует warmup или GPT-2 init**: fresh init L=4 даёт `|DEmbed|_max=29.8` (blowup).

4. **BH=1 fast path в attention_recon.go** — все de-wrapper только для BH=1. BH>1 loop path (line ~110+) остался на MatMulF32Ex — 6 calls нужно заменить при переходе к multi-head.

5. **F-PROBE print** в battle_a_llm_f32recon.go line 448 — оставлен как canary диагностика (пишет "F-PROBE: dNormedTop computed via plain b.MatMul" при каждом bwd). Можно убрать позже.

6. **Determinism-gate** в cert-тесте (line ~132-172): после первого bwd → snapshot dWq/dW1/dEmbed → zeroGrads → 2nd fwd+bwd → snapshot → compare. **FAIL** если |Δ|>0.

7. **Mixed-floor формула** в cert-тесте (line ~440-470): `floor_abs = C·sqrt(N)·eps_F32·scale·amplif` с C=50, scaleAmpl=20, minFloor=1e-5 для nStages>=8. Pass если `absDiff ≤ floor_abs OR relDiff ≤ 1e-2`.

---

## Git state чистота

**Uncommitted на HEAD 9e1e3f0:** только `libs/S2v4_*_probe_*`, `libs/bench_bh1_*`, `libs/bench_dk`, `libs/bench_dv_new` — все untracked wild файлы FA-bwd benchmarks, **не мешают cert**.

Все reference-ветка изменения закоммичены.

---

## Заголовок задания следующей сессии (одна строка)

**Полная де-wrapper-изация reference-ветки (6 MatMulF32Ex в BH>1 attention_recon) + починка determinism (cublasSgemm probe / EmbeddingGradF32 sort-scatter) + канонический baseline cert (2× bit-exact) + sign-of-life re-run + L=4 опционально + СТОП.**

Стратегическая сводка (пять пунктов + wrapper-следствие) — у пользователя в предыдущем сообщении. Здесь — только tactical handoff.

---

## Living-документ

Правило: в конец каждого будущего закрытия сессии — добавлять абзац об изменениях этого HANDOFF (и PROJECT_CHRONICLE.md — соседний файл).
