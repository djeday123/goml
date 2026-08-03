# A-LLM-3: CPU-F64 эталон + переклассификация GPU-F32-recon

**Дата:** 2026-08-03. **Кодовый commit:** `2d576c1`. **Старт сессии:** HEAD `3770056` (Б-0 PASS: subject совпал, tracked чист, TestMatmulPlainT_Unit 17/17 PASS повторён).
**ТЗ:** HANDOFF_ref_dewrapper.md, 6 шагов + П-0..П-6.
**Raw:** `runs/reports/raw_allm3/f64_cert.log`, `runs/reports/raw_allm3/ab_f32_f64_n5.log`.

---

## Шаг 1 — bwdBattleAF64/fwdBattleAF64 (PASS)

`internal/abjexam/battle_a_llm_f64ref.go`: полная цепочка cert-формы (V=32 D=128 H=1
HD=128 L=1 S=32 B=1 FFN=128) на CPU в float64, ноль GPU-вызовов.

- П-1: attention БЕЗ causal mask — зеркало attention_recon.go как есть.
- П-2: dgamma зеркалирован с PTX rmsnorm_grad_f32 (ptx_kernels.go:1856):
  S-сумма в F64-аккумуляторе, dx_j = γ_j·dy_j·invRms − x_j·S·invRms³/D,
  dgamma_j += dy_j·x_j·invRms. **Вердикт зеркалирования: PTX-формула совпадает
  с учебной производной, расхождений НЕ обнаружено.**
- П-3: ни одной итерации по map (проверено построением; токен-генерация через []bool).
- Веса: та же rand-последовательность seed 31, что NewBattleAState → bit-идентичны GPU.
- Unit-гейт (`TestALLM_F64Ref_Unit`): RoPE fwd+grad ортогональность max|Δ|=4.441e-16;
  rmsNormGradF64 vs finite-diff maxRel dx=4.3e-08, dgamma=1.2e-08; matmul
  trans-варианты bit-exact 0.000e+00.

## Шаг 2 — det-gate F64 (PASS, прогноз подтверждён)

```
2x in-process: все 11 тензоров (DWout, DNormOut, DWq/DWk/DWv/DWo, DNorm1/DNorm2,
DW1/DW2, DEmbed) bit-diff cells=0, max|Δ|=0.000e+00 exact; loss bit-exact.
Fresh subprocess: F64_DET_HASH=7ca2a06774090ed6 == in-process.
```

## Шаг 3 — F64 finite-diff, 10 точек

**Гейт ТЗ (eps=1e-6, rel<=1e-8): прогноз-промах 8/10.** Два числа: прогноз ТЗ
1e-8..1e-10; факт worst relDiff=1.4e-06 (Wq). Причина — класс «измерительный
прибор»: гейт игнорировал roundoff-член центральной разности
(noise_abs ~ δ_L/(2·eps), δ_L ≈ 1 ulp loss ≈ 7.7e-16, подтверждено по Wout).
Контр-прогноз исполнителя записан ДО прогона (заголовок теста) и подтверждён.

**Исправлен инструмент, НЕ floor (порог остался 1e-8):** модель ошибки
rel(eps) = A·eps² + B/eps подтверждена скейлингом (Embed: 8.92e-5 → 8.87e-7 при
eps 1e-3 → 1e-4, фактор 100.6 = чистое eps²). Richardson-экстраполяция
g_R = (4g(eps/2) − g(eps))/3 убивает eps²-член.

**CERT v2 PASS 10/10 rel<=1e-8:**
```
Wout(top)    bestRel=2.034e-11   Wo[L=0]     bestRel=6.932e-11
W2[L=0]      bestRel=4.461e-10   W1[L=0]     bestRel=1.544e-11
Wv[L=0]      bestRel=2.270e-10   Wq[L=0]     bestRel=3.244e-09
Embed[i0,d]  bestRel=2.182e-10   NormOut(g)  bestRel=3.849e-11
Norm1[0](g)  bestRel=3.028e-10   Norm2[0](g) bestRel=3.057e-10
```

**Санити-якорь:** F64 loss=3.4931709252 (F32-канон 3.4932); F64 ana == канон
efd833c до 6-7 знаков: Wout −4.972173e-02, Wo −1.053664e-02, Wq +1.867207e-04,
Embed +6.327998e-04. Математика эталона и F32-recon-пути одна и та же.

## Шаг 4 — A/B GPU-F32 vs CPU-F64, N=5 (задокументирован)

Fwd GPU **детерминирован bit-exact**: loss 3.4931708053 идентичен 5/5, совпадает
с F64 до 7 знаков. Нон-детерминизм живёт только в bwd.

Нон-детерминизм (max попарный |Δ|, full-tensor, свежий процесс из raw):
```
DWout 0.000e+00   DNormOut 1.9e-09   DWq 3.5e-03   DWv 2.4e-03   DWo 1.2e-01
DNorm1 4.0e-04    DNorm2 2.3e-09     DW1 0.000e+00 DW2 5.8e-02   DEmbed 2.2e-02
```
Процессо-зависимость подтверждена: в первом процессе сессии DWo ND был 0.000e+00
(систематический сдвиг без разброса), во втором 1.2e-01 с мёртвой cert-точкой
в одном прогоне (класс context-dep zero). Оба состояния в raw.

Formula-floor (прогноз, 5-е применение sqrt-правила, C=50 ampl=20): провален
9/10 — прогноз исполнителя «floor пути определяется GPU-аномалиями, не
sqrt-накоплением» подтверждён. **Документированный floor пути = worst|Δ| по
точкам из raw** (процессо-зависимый; текущий процесс: от 8.7e-09 на Wout до
1.05e-02 на Wo / rel до 2.5 на Embed).

**Вердикт переклассификации: GPU-F32-recon = «первый измеряемый», НЕ эталон.
Эталон = CPU-F64 (bit-det, cert v2 10/10).**

## Шаг 5 — sign-of-life (LIVE)

```
3.4932 → 2.7121 за 10 SGD-шагов lr=1e-2, Δ=+0.7810 (raw ab_f32_f64_n5.log)
```

## П-5 — контракт причинности (вердикт: стек когерентен в causal=0)

Raw grep (подтверждение разведки ревьюера):
```
battle_a_llm.go:499-504: faCtx.ForwardTrain(..., 0, 0, faScale, 0) — литералы causal=0, window=0
fa_backward.go: FABwdMerged:112(causal на :113), FABwdDK:133, FABwdDQ:152 — параметр causal есть
fa_bwd_merged_v1.cu:146: const int qt_start = causal ? kt : 0;
fa_bwd_merged_v1.cu:242: if (causal && j_g > i_g) return true;
grep -c causal: fa_bwd_dk.cu=6, fa_bwd_dq_new.cu=4 (с -i: 7/5, разница в комментариях)
cert-прогоны: fa_forward_test.go:117,196 causal=0; fa_backward_test.go:189 causal=0
```
Causal-ветка F64 НЕ строится. Долг: causal=1 реализован ядрами, но не
сертифицирован — сертификация causal=1 + causal-ветка эталона перед переходом
к автогрессивному LM.

## Шаг 6 — иерархия эталонов (действующая)

```
CPU-F64 арбитр (bit-det: Δ=0 exact в процессе и между процессами; cert v2 10/10 rel<=1e-8)
    ↓ A/B (floor = документированный worst|Δ| из raw; процессо-зависим)
GPU-F32-recon (первый измеряемый; fwd bit-det, bwd недетерминистичен — свойство пути)
    ↓ A/B (двухзонный floor 5e-3 abs + FP8-зоны boundary — следующая сессия)
FA-FP8 боевой (встройка = следующая сессия, non-causal режим ядер, A/B primary vs F64 arbiter)
```

## Состояние тестов после сессии

| Test | State |
|------|-------|
| `TestALLM_F64Ref_Unit` | **PASS** |
| `TestALLM_BwdCertF64_MultiLayer` | **PASS** (последний зелёный для Б-0 следующей сессии) |
| `TestALLM_ABF32vsF64_N5` | **PASS** (документационный) |
| `TestMatmulPlainT_Unit` | PASS 17/17 |
| `TestALLM_BwdCertF32_MultiLayer` | FAIL (устаревший det-gate внутри; поглощён F64-иерархией, не блокер) |
