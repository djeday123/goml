# goml — Архитектура и роль в стеке

**Статус:** 2026-07-18 (R03b era). Актуально до пересмотра.

---

## Что такое goml сейчас

**goml** — референс-приложение тренировки языковой модели: `nn.LLM` (transformer), `train.Trainer`, `optim.AdamW`, `tokenizer.ByteTokenizer`, датасет-инжест. Плюс исторически — боевой GPU-движок `backend/cuda`, который несёт весь current CUDA-путь (PTX-ядра, cuBLAS обёртки, LayerNorm, RoPE, softmax-по-осям, SDPA, FP8/FP16 mixed precision), и FA-ядра в `libs/` (fa_v121r champion 647 TFLOPS на sm_120a).

Оба слоя сейчас работают и используются в тренировочных прогонах.

## Куда goml идёт

**Целевая картина стека (см. также [gotorch/v6/ARCHITECTURE.md](https://github.com/djeday123/gotorch/blob/master/ARCHITECTURE.md)):**

- **gotorch** — единственный целевой источник истины для переиспользуемых вычислительных компонентов (тензоры, слои, оптимизаторы, GPU-backend).
- **fa-blackwell-fp8** — отдельная библиотека FA-ядер, C-ABI + Go/Python-биндинги.
- **goml** — референс-приложение тренировки. GPU-движок `backend/cuda` **не развивается** как переиспользуемая библиотека; его функциональность поэтапно портируется в gotorch, и goml-трейн переключается на gotorch-версии.

Статус движка в goml: **работает, остаётся на месте, замораживается как read-only эталон**. **НЕ развивается как библиотека** для внешних пользователей — если нужен CUDA-backend для другого приложения, брать gotorch.

## Правило переходного периода

**`goml/backend/cuda` — read-only эталон.** Фиксы и развитие вычислительной логики идут **только в gotorch**. Никаких «заодно поправим в goml» — иначе источников истины снова два и стек начнёт разбегаться.

Единственные разрешённые правки:
1. Тонкие публичные аксессоры, нужные мосту `goml/backend/gotorch/` (напр. `Backend.Stream()` в R03b-impl-2).
2. Регистрация новых зависимостей в `go.mod`.
3. Критические регрессии, блокирующие текущий тренинг.
4. Приложенческий слой (`nn/`, `train/`, `optim/`, `tokenizer/`, `data/`) — развивается как обычно.

**НЕ трогается:** PTX-ядра в `backend/cuda`, cuBLAS-обёртки, LayerNorm/RoPE/SDPA/Embedding — вся вычислительная логика внутри `backend/cuda`.

**Известная разница режимов (R03b-impl-4-final):** legacy `cublas handle` в `backend/cuda/cublas.go:216` — TF32 (`cublasSetMathMode(TF32)`), gotorch — pedantic FP32. Adapter direct-путь через gotorch **~3600× точнее legacy** (измерено impl-4-final Sверка 3.2 на [16×16×16]: adapter maxAbs 9.5e-7 vs fb maxAbs 3.5e-3). Выравнивание gotorch под legacy TF32 **не выполняется** — эрозия целевого источника истины отклонена (см. gotorch/v6/ARCHITECTURE.md). При полном переходе трейна на gotorch эта разница исчезает в пользу FP32. TF32-vs-FP32 переоткрывается при порте композитных ядер в gotorch (например, где 1e-3 достаточно ради 2× throughput). Точечный TF32 доступен через `gotorch.PuregoBackend.MatMulF32_TF32` — режим свойство метода, не состояние backend'а.

## Мост goml ↔ gotorch (R03b)

Пакет `goml/backend/gotorch/` — адаптер, реализующий `backend.Backend` через gotorch с fallback в `backend/cuda` для методов, которых в gotorch ещё нет. Включается явно:

```go
import _ "github.com/djeday123/goml/backend/gotorch"
// ...
if err := gotorch.Enable(); err != nil { log.Fatal(err) }
```

или через env var `GOML_BACKEND=gotorch` + `gotorch.EnableIfEnv()`.

**Дизайн моста:**
- Один CUDA-контекст: primary (обе стороны retain'ят через `cuDevicePrimaryCtxRetain`, R03b-impl-1 Fix A).
- Один stream: goml создаёт stream, adapter отдаёт его в gotorch через `gotorch.PuregoBackend.SetStream`. Оба мира работают на одной очереди — порядок операций гарантирован stream'ом, полные sync внутри тела методов запрещены.
- Покрытие (см. [gotorch/v6/runs/reports/R03b_design.md](https://github.com/djeday123/gotorch/blob/master/runs/reports/R03b_design.md)):
  - **47% direct** — тонкие обёртки через gotorch-методы.
  - **44% stays-in-goml** — делегируются в `backend/cuda` через embedded fallback. **Эта цифра — карта будущих портов в gotorch.**
  - **6% gap** — 2 недостающих PTX-ядра, добавляются в gotorch мелкими коммитами.

Референс-goml-код в `backend/cuda` при переносе служит **эталоном**: gotorch-версия должна match'иться с ним bit-exact или в документированном floor'e через A/B-тест в одном процессе (роскошь, которую даёт мост).

## Порядок портирования (после R03b-impl-5)

По одному куску за итерацию:

1. Порт с goml-эталона в gotorch (не изобретать заново — оптимизации sm_120a/sm_89 уже выстраданы).
2. Тесты корректности: **gotorch-версия vs goml-эталон в одном процессе** через мост.
3. Переключение goml-трейн-пути на gotorch-версию.
4. Метод переходит из stays-in-goml в direct.

Когда список stays-in-goml → 0, фаза уборки `goml/backend/cuda` становится безопасной. **До этого — не разбирать.**

## Что goml остаётся давать миру

- Готовый тренировочный контур для LLM (`nn.LLM` + `train.Trainer`).
- Референс-эталон CUDA-путей (пока переносится в gotorch).
- FA-стек в `libs/` (v121r 647T на sm_120a) — отдельная строна, готовится к выделению в `fa-blackwell-fp8`.

---

# ENGLISH SECTION

## What goml is right now

**goml** is a reference LLM training application: `nn.LLM` (transformer), `train.Trainer`, `optim.AdamW`, `tokenizer.ByteTokenizer`, dataset ingest. Plus historically a battle GPU engine in `backend/cuda` that carries the entire current CUDA path (PTX kernels, cuBLAS wrappers, LayerNorm, RoPE, softmax-along-axis, SDPA, FP8/FP16 mixed precision), plus FA kernels in `libs/` (fa_v121r champion 647 TFLOPS on sm_120a).

Both layers work today and power production training runs.

## Where goml is headed

**Target stack picture** (see also [gotorch/v6/ARCHITECTURE.md](https://github.com/djeday123/gotorch/blob/master/ARCHITECTURE.md)):

- **gotorch** — the single canonical source of truth for reusable compute (tensors, layers, optimizers, GPU backend).
- **fa-blackwell-fp8** — a separate FA kernels library, C ABI + Go/Python bindings.
- **goml** — the reference training application. Its GPU engine `backend/cuda` is **no longer developed** as a reusable library; its functionality is ported into gotorch piece by piece, and the goml trainer switches to the gotorch versions.

Status of the engine in goml: **it works, it stays in place, it is frozen as a read-only reference**. It is **not developed as a library** for outside users — if you need a CUDA backend for another application, take gotorch.

## Transition-period rule

**`goml/backend/cuda` is a read-only reference.** Fixes and development of compute logic land **only in gotorch**. No "let's patch goml while we're here" — otherwise you get two sources of truth again and the stack starts to drift.

Only these edits are allowed:
1. Tiny public accessors the bridge `goml/backend/gotorch/` needs (e.g. `Backend.Stream()` in R03b-impl-2).
2. New `go.mod` dependency entries.
3. Critical regressions that block current training.
4. The application layer (`nn/`, `train/`, `optim/`, `tokenizer/`, `data/`) — developed as usual.

**Untouched:** PTX kernels in `backend/cuda`, cuBLAS wrappers, LayerNorm/RoPE/SDPA/Embedding — all compute logic inside `backend/cuda`.

## The goml ↔ gotorch bridge (R03b)

Package `goml/backend/gotorch/` — an adapter implementing `backend.Backend` via gotorch with fallback into `backend/cuda` for methods not yet in gotorch. Turn it on explicitly:

```go
import _ "github.com/djeday123/goml/backend/gotorch"
// ...
if err := gotorch.Enable(); err != nil { log.Fatal(err) }
```

or via env var `GOML_BACKEND=gotorch` + `gotorch.EnableIfEnv()`.

**Bridge design:**
- One CUDA context: primary (both sides retain via `cuDevicePrimaryCtxRetain`, R03b-impl-1 Fix A).
- One stream: goml creates the stream, the adapter injects it into gotorch through `gotorch.PuregoBackend.SetStream`. Both worlds run on one queue — operation order is guaranteed by the stream, full syncs inside method bodies are forbidden.
- Coverage (see [gotorch/v6/runs/reports/R03b_design.md](https://github.com/djeday123/gotorch/blob/master/runs/reports/R03b_design.md)):
  - **47% direct** — thin wrappers over gotorch methods.
  - **44% stays-in-goml** — delegated into `backend/cuda` via embedded fallback. **This number is the map of future ports into gotorch.**
  - **6% gap** — 2 missing PTX kernels, added to gotorch in small commits.

The goml reference code in `backend/cuda` serves as the **reference** during porting: the gotorch version must match it bit-exact or within a documented floor via an A/B test in one process (a luxury the bridge provides).

## Porting order (after R03b-impl-5)

One piece per iteration:

1. Port from the goml reference into gotorch (do not reinvent — sm_120a/sm_89 optimizations are already battle-hardened).
2. Correctness tests: **gotorch version vs goml reference in one process** via the bridge.
3. Switch the goml trainer path to the gotorch version.
4. The method moves from stays-in-goml to direct.

When the stays-in-goml list hits 0, the `goml/backend/cuda` cleanup phase becomes safe. **Not before.**

## What goml keeps giving the world

- A ready LLM training loop (`nn.LLM` + `train.Trainer`).
- A reference implementation of the CUDA path (while it is being ported into gotorch).
- The FA stack in `libs/` (v121r 647T on sm_120a) — a separate concern, being spun out into `fa-blackwell-fp8`.
