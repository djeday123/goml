# HANDOFF: reference-ветка де-wrapper-изация + determinism-gate

**Дата закрытия:** 2026-08-02 (last-session code HEAD `9e1e3f0`; поверх него doc-only коммиты `18ff979`, `8b3cf23` и reviewer-addendum коммит с этой правкой)
**Автор:** Claude Opus 4.7 предыдущей сессии; секция "ДОПОЛНЕНИЕ К ТЗ — ФИНАЛ" в конце файла — ревьюер, 2026-08-03
**Читай ПЕРВЫМ:** этот файл ЦЕЛИКОМ (включая ДОПОЛНЕНИЕ в конце — оно имеет приоритет) + `git log -1` + повторный прогон последнего зелёного unit-test.

---

## Б-0 старт следующей сессии (обязательный протокол)

Первые команды дословно:

```bash
cd /data/lib/podman-data/projects/goml
git log -1
# ожидается subject: "HANDOFF: ТЗ A-LLM-4 FA-встройка (финал, ревью принято)"
# это doc-only коммит; цепочка вниз: f340ab9 (doc) -> 2d576c1 (КОДОВЫЙ, A-LLM-3
# F64-эталон) -> 3770056 -> 8b3cf23 -> 18ff979 -> 9e1e3f0
# проверка: git log -6 --oneline должен содержать f340ab9 и 2d576c1

git status --short
# ожидается: только untracked wilds (~475 файлов: libs/*, runs/archive/*, runs/_canary_5run_fwd.sh) — не мешают
# модифицированных tracked файлов быть НЕ должно (ноль строк без префикса ??)

# Повторный прогон последнего зелёного (СМЕНИЛСЯ после A-LLM-3):
env GOTORCH_LIBS_DIR=/data/lib/podman-data/projects/gotorch/v6/libs \
  go test -run 'TestALLM_BwdCertF64_MultiLayer$' -count=1 -timeout 300s -v ./internal/abjexam/
# ожидается: PASS — det-gate bit-exact (Δ=0.000e+00, subprocess hash 7ca2a06774090ed6),
# CERT v2 10/10 rel<=1e-8. Прогонять МОЖНО без GPU (путь чисто CPU; GPU нужен только
# для TestALLM_ABF32vsF64_N5).

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

## Заголовок задания следующей сессии (одна строка) — ПЕРЕОПРЕДЕЛЁН 2026-08-02

**Строим эталон = CPU-F64 bwdBattleAF64 (все грады формулой, ноль GPU в reference-пути), det-gate bit-exact, F64 finite-diff numerical, GPU-F32-recon переклассифицируется в «первый измеряемый» — A/B vs F64 арбитр.**

Ревьюер решил: GPU-недетерминизм = физика параллелизма, НЕ дефект. Fallback (rewrite reference bwd на purego CPU-F64) повышается до магистрали.

### 6-шаговая программа fresh-сессии

**Шаг 1. Полный bwdBattleAF64.**
Вся backward-цепочка cert-формы (L=1, малые размеры: V=32, D=128, HD=128, S=32, B=1, FFN=128) на CPU в float64:
- dLogits — CE-grad формулой (softmax(logits) − onehot(target))/M
- top-RMSNorm-grad (формула: dx_j = γ_j·dy_j/rms − x_j·S·rms⁻³/D, S=Σγ_i·x_i·dy_i)
- FFN-residual add
- FFN chain: dW2 → dFFNSilu = dFFNOut@W2^T → silu-grad формулой (dsilu/dh = σ + h·σ·(1−σ)) → dFFNHidden → dW1 → dNormed(FFN)
- RMSNorm2-grad → residual add → dAttnOut
- attention chain: dQ_buf = dAttnOut@Wo^T → inverse permute → dOF32
- attnReconstructBwd формулами: dV=P^T@dO, dP=dO@V^T, dS=P*(dP−Σ(P*dP)), dQ=dS@K·scale, dK=dS^T@Q·scale
- RoPE-bwd формулой + permute-inverse
- dNormed = dQ@Wq^T + dK@Wk^T + dV@Wv^T
- weight grads: dWq/dWk/dWv = NormedRecomp^T@dQ/dK/dV; dWo = Q_buf^T@dAttnOut; dW1/dW2 аналогично
- RMSNorm1-grad → dX_pre_attn
- dEmbed **последовательный цикл scatter** (детерминизм by construction, НЕ atomicAdd)
- dWout = sc.Normed^T@dLogits

**Строительный материал уже на диске:**
- D-hop F64-этажи в cert-test line ~140-280 (dX_ref, dFFNSilu_ref, dFFNHidden_ref)
- `TestRMSNormGrad_CPUF64_Arbiter` в rmsnorm_grad_arbiter_test.go — F64-формула проверена
- `TestMatmulPlainT_Unit` cpuRef в matmul_plain_helper_test.go — F64 matmul reference

Никаких GPU-вызовов в этом пути. Файл: `internal/abjexam/battle_a_llm_f64ref.go` (создать).

**Шаг 2. Det-gate на F64-пути.**
Cert-test:
- Запустить F64-bwd дважды в процессе + fresh subprocess
- Сравнить ВСЕ грады (7+ точек) bit-exact
- **Требование:** max|Δ| = 0.000e+00 exact (не «около»)
- ПРОГНОЗ: PASS тривиально (последовательный код)

**Шаг 3. F64 finite-diff numerical.**
- Двусторонний finite-diff на F64-forward: fwdBattleAF64 (тоже написать — используя формулы, ноль GPU)
- eps = 1e-6
- Cert F64-ana vs F64-num
- **ПРОГНОЗ:** relDiff 1e-8..1e-10 на всех 7+ точках
- Это и есть **настоящий tight-сертификат математики**, без mixed-floor костылей

**Шаг 4. Переклассификация GPU-F32-recon.**
Из «эталона» в «первый измеряемый»:
- A/B GPU-F32 (текущий `bwdBattleAF32` в battle_a_llm_f32recon.go) vs CPU-F64 (Шаг 1) по всем точкам
- Floor двухзонной формулой **записать ДО прогона** (sqrt-накопление, 5-е применение [[feedback-sqrt-vs-linear-accumulator]]):
  ```
  floor_abs = C·√N_stages·eps_F32·scale·amplif
  ```
- Результат = **документированный floor GPU-recon-пути**
- Недетерминизм между прогонами задокументировать числом (max|Δ| прогон-к-прогону) как **свойство пути**, не как баг

**Шаг 5. Sign-of-life остаётся на GPU-F32 (скорость).**
- Перегнать 10-step training через `trainStepBattleAF32`
- Число в raw: initial loss → after10 loss → Δ

**Шаг 6. Иерархия эталонов в отчёт.**
```
CPU-F64 арбитр (bit-det, correctness-only)
    ↓ A/B (задокументированный floor)
GPU-F32-recon (первый измеряемый, недетерминистичен свойство пути)
    ↓ A/B (двухзонный floor 5e-3 abs, FP8-зоны boundary)
FA-FP8 боевой (следующая сессия — встройка A/B primary vs F64 arbiter)
```

L=4 — **отдельная задача, НЕ блокирует** (стык контрактов покрыт L=1).

### Что дают на входе

Файлы reference-ветки (сейчас всё GPU-F32):
- `internal/abjexam/battle_a_llm_f32recon.go` — trainStepBattleAF32 / fwdBattleAF32 / bwdBattleAF32
- `internal/abjexam/attention_recon.go` — attnReconstructFwd / attnReconstructBwd BH=1
- `internal/abjexam/battle_a_llm_f32recon_cert_test.go` — cert test с D-hop F64 arbiter (переиспользовать формулы!)

Файлы CPU-F64 assets:
- `internal/abjexam/rmsnorm_grad_arbiter_test.go` — F64-формула RMSNormGrad (Шаг 1a)
- `internal/abjexam/matmul_plain_helper_test.go` — F64 matmul cpuRef с trans-flags (Шаг 1 attention matmuls)

### Гейт закрытия сессии

- `bwdBattleAF64` создан и unit-test PASS формула-vs-формула на trivial случаях.
- `TestALLM_BwdCertF64_MultiLayer` new test с det-gate 2× + fresh subprocess, все Δ=0 exact.
- F64-ana vs F64-num rel<1e-8 на 7+ точках.
- A/B GPU-F32-recon vs CPU-F64: floor задокументирован, числа в raw.
- Sign-of-life re-run: свежее число.
- Commit + push с hash+raw.
- HANDOFF + CHRONICLE обновлены поверх этого состояния.

**FA-встройка = следующей fresh-сессией после этой.**

---

## Living-документ

Правило: в конец каждого будущего закрытия сессии — добавлять абзац об изменениях этого HANDOFF (и PROJECT_CHRONICLE.md — соседний файл).

---

## ДОПОЛНЕНИЕ К ТЗ — ФИНАЛ (ревью принято 2026-08-02, усиления учтены; вписано 2026-08-03)

Эта секция имеет приоритет над остальным файлом при любом противоречии.

П-0. Б-0 шапка исправлена этим же коммитом (см. начало файла): ожидаемый HEAD = doc-only
reviewer-addendum коммит, опознаётся по subject-строке "HANDOFF: reviewer addendum П-0..П-6
(doc-only, Б-0 шапка синхронизирована)". Код не менялся с 9e1e3f0; 18ff979/8b3cf23 — doc-only.

П-1. Шаг 1, контракт зеркальности: fwdBattleAF64/bwdBattleAF64 зеркалят GPU-F32-recon КАК ЕСТЬ,
включая attention БЕЗ causal mask. Не "улучшать" математику по учебнику. Расхождение формул
с f32recon = отдельный пункт отчёта, не молчаливая правка.

П-2 (усилено). Шаг 1 дополнение: dgamma всех трёх норм (top/norm1/norm2) в F64-ref выводить
ЗЕРКАЛИРОВАНИЕМ фактического PTX rmsnorm_grad_f32 (gotorch/v6/cuda/ptx_kernels.go, район
line 2002): взять S-сумму и её раздачу как в ядре, не из учебника. Прецедент "мусор==мусор"
на этом классе был. Если PTX-формула покажется неверной — отдельный пункт отчёта, не молчаливая
правка. Cert расширить до 10 точек: 7 прежних + DNormOut/DNorm1/DNorm2 (по одной top-magnitude
координате).

П-3. Шаг 2 ограничение: в F64-пути запрещены итерации по map (range m) в любом вычислении —
только слайсы и явные индексы. Иначе bit-exact между процессами не гарантирован по построению
языка.

П-4. Шаг 4 уточнение: нон-детерминизм GPU-F32 документировать по N=5 прогонам, число = max
попарного |delta| на каждую точку. Floor A/B обязан держаться на всех 5 прогонах vs F64,
не только на первом.

П-5 (обязательный пункт ЭТОЙ сессии, факт-подтверждение). Разведка ревьюера дала:
  (а) боевой G1-вызов internal/abjexam/battle_a_llm.go:503 подаёт causal=0, window=0
      (литералы) — full attention;
  (б) bwd-биндинги backend/cuda/fa_backward.go:113/133/152 имеют параметр causal; ядра его
      реализуют: libs/fa_bwd_merged_v1.cu:146 (qt_start = causal ? kt : 0) и :242 (маска
      j_g > i_g), fa_bwd_dk.cu (7 вхождений), fa_bwd_dq_new.cu (5 вхождений); все cert-прогоны
      шли causal=0 (fa_forward_test.go:117,196; fa_backward_test.go:189).
  Задача: подтвердить эти координаты grep-командами в отчёте (raw-вывод), вердикт одной
  строкой: "стек когерентен в causal=0". Causal-ветку F64 НЕ строить. В реестр долгов записать:
  "causal=1 реализован ядрами, но не сертифицирован; сертификация causal=1 + causal-ветка
  эталона — перед переходом к автогрессивному LM". Решение о рамках встройки: non-causal
  режим ядер.

П-6. Гейт закрытия, два доппункта: (а) в PROJECT_CHRONICLE.md добавить недостающий параграф
за ревьюерскую сессию 2026-08-02 16:20 (hash 8b3cf23, смена магистрали на CPU-F64) отдельно
от параграфа своей сессии; (б) в реестре долгов хроники переклассифицировать "wrapper-следствие
CRITICAL" и "cublasSgemm determinism CRITICAL" в "свойство пути, документируется числом Шага 4".

Порядок сессии: F64-эталон -> детерминизм-гейт -> F64-num cert 10 точек -> переклассификация
GPU-F32 с N=5 -> sign-of-life -> HANDOFF+хроника поверх стабильного -> СТОП.

---

## ЗАКРЫТИЕ A-LLM-3 (2026-08-03, кодовый commit 2d576c1) — living-doc

Программа выше ВЫПОЛНЕНА ЦЕЛИКОМ (Шаги 1-6 + П-1..П-6). Полный отчёт с raw:
runs/reports/A_LLM3_f64ref.md; сырьё: runs/reports/raw_allm3/*.log.

Итог одной строкой: эталон = CPU-F64 (battle_a_llm_f64ref.go), det-gate bit-exact
PASS (Δ=0.000e+00 в процессе + fresh subprocess), CERT v2 PASS 10/10 rel<=1e-8
(Richardson-инструмент; гейт ТЗ eps=1e-6 — задокументированный прогноз-промах 8/10,
шумовой пол δ_L ~ 1 ulp), GPU-F32-recon переклассифицирован в "первый измеряемый"
(fwd bit-det, bwd нон-детерминизм задокументирован числом по N=5), sign-of-life
3.4932 -> 2.7121 (Δ=+0.7810), стек когерентен в causal=0 (П-5 raw grep).

Изменения этого HANDOFF: Б-0 шапка перенацелена на новое состояние (последний
зелёный = TestALLM_BwdCertF64_MultiLayer, работает без GPU); эта секция добавлена.
CHRONICLE: добавлены параграфы за 8b3cf23 (задним числом, П-6а) и за 2d576c1;
реестр долгов: две CRITICAL-строки переклассифицированы в "свойство пути" (П-6б),
добавлен долг "causal=1 не сертифицирован" (HIGH).

## ТЗ A-LLM-4: FA-ВСТРОЙКА — ФИНАЛ (скелет выше ЗАМЕЩЁН этим ТЗ; ревью принято 2026-08-03)

Разведка ревьюера (факты диска, легли в структуру):
- МИНА 1: bwdBattleA v1 (battle_a_llm_bwd.go:308+) математически некорректен по
  собственным TODO: dW1/dWq/dWk/dWv берут sc.Normed последнего звена, RMSNormGrad1/2
  берут sc.X пост-всех-слоёв. Неверно даже при L=1. Встройка в v1 = A/B умирает
  по вине обвязки.
- МИНА 2: снапшоты в общем BattleAScratch триггерили FA-instability
  ([[feedback-fa-fwd-scratch-alloc-instability]]) — "просто починить v1" заряжено.
- Зеро-init требуют ровно 3/4 ядер: dV/dK/dQ ("must be zero-init" в сигнатурах
  fa_backward.go:112/133/152); d_precompute пишет D полностью.
- dSnat/dST: FP8 padded stride_ds=(sl+15)&~15. Буферы DFA_*, DOFP16 уже в
  BattleABwdScratch. CastF32ToF16 в адаптере есть. Заглушка: battle_a_llm_bwd.go:451.
- Канарейка-скрипт существует: runs/_canary_5run_fwd.sh.

СКОУП: цепочка gt_fa_bwd_d_precompute -> merged -> dk_new -> dq_new
(fa_backward.go:93/112/133/152, контракт v0.2.0) в attention-bwd, non-causal
(causal=0, window=0 — П-5). GPU-F32-recon и CPU-F64 пути НЕ трогаются. ДВУХЭТАПНО.

=== ЭТАП 1. FA-блок в эталонной обвязке (сертификация цепочки) ===

1.1. Третий attention-путь в f32recon-стеке (флаг или AttnBwdPath). Fwd НЕ меняется
     (attnReconstructFwd, снапшоты безопасны). На bwd вместо attnReconstructBwd:
     (a) QPermSnap/KPermSnap/VPermSnap F32 -> FP8 (квантизатор боевого fwdBattleA;
         amax-скейлы в лог);
     (b) fa_forward_train(FP8 -> O_FP16, L_F32), L напрямую (G1-cert);
         zero-init OFP16/LGPU ДО вызова ([[feedback-fa-buffers-zero-init]]);
     (c) dO: DOF32 -> CastF32ToF16 -> DOFP16;
     (d) gt_fa_bwd_d_precompute(O_FP16, dO_FP16 -> D_F32);
     (e) zero-init DFA_dVF32/dKF32/dQF32 (3/4 контракт; факт "кто требует" в отчёт);
     (f) merged -> dSnat/dST(FP8 padded)+dV; dk_new -> dK; dq_new -> dQ;
         scale-конвенция как в канонической цепочке fa_backward_test.go:255-305;
     (g) repack dQ/dK/dV F32 [BH,S,HD] -> дальше цепочка как есть (RoPE-bwd ->
         inv-permute зеркалом форвардного).
1.2. Stream-дисциплина: все вызовы stream=0; ВСЕ A/B-чтения после Sync.
1.3. Форма A/B (решение ревью): B=1 H=4 HD=128 S=2048 D=512 FFN=2048 L=1, V=1024.
     Обоснование: V влияет только на CE-хвост и масштаб dLogits, контракты
     FA-цепочки от V не зависят. УСИЛЕНИЕ: Wout-точка в A/B-таблице играет роль
     канарейки согласованности форм — если V-урезка сломала обвязку, Wout покажет
     первым (дешёвый).
1.4. A/B шага-1, двухарбитровый, поэлементный по 10 точкам A-LLM-3 + full-tensor
     max|delta| по dQ/dK/dV блока:
     - PRIMARY vs CPU-F64 (bwdBattleAF64 на той же форме, seed-зеркало весов/токенов);
     - SECONDARY vs GPU-F32-recon (та же обвязка, путь recon) — санити с учётом
       задокументированной дрожи (raw_allm3).
     Двухзонный floor (записать ДО прогона, в тесте):
     - scale = amax/448 (per-tensor, из квантизации);
     - T_norm = scale * 2^-6 (наименьшее НОРМАЛЬНОЕ e4m3 относительно скейла);
     - T_subnorm = scale * 2^-9 (субнормальный пол);
     - ОБА числа в лог. Зона A (|grad_F64| >= T_norm): floor 5e-3 abs.
       Зона B (|grad_F64| < T_norm): "FA ноль-класс при живом арбитре =
       документированная граница cold-start", отдельные строки, не FAIL.
       Клетки между T_subnorm и T_norm — отдельная строка "субнормальная зона"
       (бесплатная детализация для будущего warmup-критерия: его порог
       переключения будет считаться той же формулой).
     ПРОГНОЗ (до прогона): зона A проходит с запасом класса B-impl-4 (1.4e-7 vs
     5e-3); dQ/dK частично в зоне B (тройное явление cold-start).

=== ЭТАП 2. Боевой путь ===

2.1. Довести bwdBattleA до зеркала bwdBattleAF32: per-layer снапшоты В ОТДЕЛЬНОМ
     scratch (НЕ в BattleAScratch — мина 2); убрать v1-упрощения. Реализовать
     case AttnBwdFA сертифицированным блоком Этапа 1.
2.2. Смоук: bwdBattleA(recon-путь) vs bwdBattleAF32 — 10 точек, inter-path floor
     записать до прогона.
2.3. trainStepBattleA E2E, боевая форма (V=32000, L=1): траектория 20 шагов
     FA-путь vs GPU-F32-recon-путь. Плато-паттерн на FA-пути в первых шагах =
     ПРАВИЛЬНОСТЬ (cold-start; warmup-звено следующей сессией). Ножницы
     траекторий числом: per-step delta-loss таблица в raw.
2.4. Скорость: (a) свежий шаг-до на GPU-F32-recon пути (свой hash, raw);
     (b) ПРОГНОЗ шага-после ДО прогона по карте блоков: grep Stage5 в
     gotorch/v6/runs/reports/A_LLM1.md (класс ~109ms) + канон FA-bwd 42.346ms
     bh=128/sl=8192 (A_LLM1.md:43). Масштабирование на bh=4/sl=2048 дать ВИЛКОЙ:
     [идеальный скейлинг ~bh*sl^2; скейлинг с SM-недогрузом — 4 блока bh-параллелизма
     на 128 SM, утилизация падает]. Факт покажет, где мы; это же число — первый
     аргумент карточки долга "FA-F16-вход".
     (c) 30-run, CV-gate <1%; два числа при промахе прогноза;
     (d) пятая карта блоков (+2-й метод при блоке >2x или >50% wall).
2.5. СТОП-ЛИНИЯ (усиленная форма, ревью): FA-instability при снапшот-scratch
     (NaN/zero O при живом Этапе 1) = СТОП + полный факт в отчёт (форма, момент,
     что в буферах) + сессия закрывается ШТАТНО с Этапом 1 как деливераблом;
     Этап 2 уезжает в следующую с отдельным разбором. Героизм в конце контекстного
     окна — источник половины фантомов.

КАНАРЕЙКА: runs/_canary_5run_fwd.sh до Этапа 1 и после Этапа 2; коридор [652,656] T.
Вне коридора — стоп до разбора.

РЕГРЕССИЯ ПЕРЕД ЗАКРЫТИЕМ (-count=1, все PASS): TestALLM_BwdCertF64_MultiLayer,
TestALLM_F64Ref_Unit, TestALLM_ABF32vsF64_N5, TestMatmulPlainT_Unit.

ЯВНО НЕ В ЭТОЙ СЕССИИ: warmup-звено (следующая; критерий переключения записать до
реализации); causal=1 сертификация (долг HIGH); wrapper-репро-охота (свойство
пути); L=4-init (sqrt(2L)); оптимизация скорости FA-блока (сначала корректность);
FA-F16-вход (карточка долга в хронике — после warmup-теста).

ГЕЙТ ЗАКРЫТИЯ: A/B Этапа 1 — числа в raw по обеим зонам + субнормальной + вердикт;
Этап 2 траектория + скорость в raw (или стоп-линия с фактом); канарейка 2x в raw;
отчёт runs/reports/A_LLM4_fa_integration.md; commit(ы) + push с hash+raw;
HANDOFF Б-0 шапка перенацелена + параграф хроники; СТОП.
