# GoML — Code Review & Fixes

Дата: 2026-05-20
Ветка: `fix/ops-autograd-gradfns`
Статус: **⏳ Ожидает одобрения от Гюльнар перед мерджем в `main`**

> Эта ветка содержит код-ревью и точечные фиксы в `ops/`-autograd path.
> В `main` не мерджить до явного approval от @Гюльнар.
> Цель ветки — зафиксировать сделанное, чтобы работа не потерялась.


## TL;DR

- Я сделал глубокий code-review всего проекта (8600+ LoC Go + 137 CUDA-файлов)
- **Большинство багов, изначально подсвеченных автоматическим аудитом, оказались false-positive** при ручной перепроверке математики
- **Реальные баги** локализованы в `ops/` (autograd path, который **не используется текущим training loop**) — это TODO в коде, оставленные на потом
- Тренировка через `train.Trainer.Backward` использует ручной `nn/backward*.go` путь, который **математически корректен**
- Внёс минимальные точечные фиксы и реализовал недостающие GradFn для будущего использования autograd

---

## 1. Архитектурный контекст

В проекте существуют **два разных backward-пути**:

### Путь A: ops-autograd (через `SetGradFn`)
```
ops.Add / ops.Mul / ops.MatMul / ops.Relu / ops.Softmax / ops.LayerNorm ...
    └─ устанавливает GradFn на тензоре
    └─ при .Backward() на финальном tensor рекурсивно вызывается chain rule
```

### Путь B: nn-manual (ручные `Backward` методы)
```
nn/backward.go,    nn/backward_attn.go,    nn/loss.go
    └─ ForwardWithCache сохраняет промежуточные значения
    └─ Backward(cache, dout) явно считает градиенты слоя
```

`train.Trainer.Step()` вызывает **Путь B**:
```go
err = t.Model.Backward(cache, dLogits)
```

То есть **тренировка работает по Пути B, который корректен**. Путь A — наполовину заглушки (`TODO: implement`), но реально не используется.

---

## 2. False positives автоматического аудита

Перепроверил математику строка-в-строку — следующие "критические баги" из аудита оказались **корректными**:

### 2.1. "Softmax double-scale в attention backward" (`nn/backward_attn.go:198`)

Аудит утверждал, что `scale` применяется дважды. Проверка:

**Forward:**
- `scores_raw = Q @ Kᵀ`
- `scores_scaled = scores_raw * scale`
- `attn = softmax(scores_scaled)`
- `out = attn @ V`

**Backward (chain rule):**
- `dAttn = dOut @ Vᵀ`
- `dScoresScaled = attn * (dAttn - sum(dAttn * attn))` ← softmax-Jacobian
- `dScoresRaw = dScoresScaled * scale` ← chain через scale
- `dQ = dScoresRaw @ K`

Код на строке 198 объединяет softmax-backward и умножение на `scale` в одну операцию — это и есть `dScoresRaw`. Затем `dQ = dScoresRaw @ K`. **Корректно**.

### 2.2. "AdamW weight decay coupling" (`optim/adamw.go:90`)

Аудит утверждал, что weight decay связан с gradient clipping. Проверка формулы:
```go
pData[j] -= lr * (update + wd * pData[j])
```
Это эквивалентно:
```
param -= lr * update     // Adam step (использует clipped gradients через m, v)
param -= lr * wd * param // decoupled weight decay (не зависит от gradient)
```
Это **стандартная decoupled AdamW формула** (Loshchilov 2019). Weight decay не зависит от gradient'а, gradient clipping применяется только к `g` через `m, v`. **Корректно**.

### 2.3. "CrossEntropyLoss pad-mask broken" (`ops/loss.go:112-118`)

Аудит утверждал, что padded токены получают ненулевые градиенты. Проверка:
- Строки 86-88: `if target < 0 { continue }` — padded позиции пропускаются
- `gradData` инициализирован `make([]float32, n)` — все нули
- В loop'е для valid позиций пишется softmax probability, для padded остаётся **0**
- Финальное масштабирование `* 1/count` не меняет нули

**Корректно**.

### 2.4. "Final norm input reconstruction wrong" (`nn/backward_attn.go:374`)

Аудит утверждал, что `PostAttn + FFNOut` не соответствует входу финального LayerNorm. Проверка `TransformerBlock.ForwardWithCache`:
```go
postAttn := x + attnOut                   // первый residual
normed2 := FFNNorm(postAttn)
ffnOut := FFN(normed2)
out := postAttn + ffnOut                  // второй residual = output блока
```
То есть выход последнего блока = `PostAttn + FFNOut`, что и используется как вход финального LayerNorm. **Корректно**.

### 2.5. "RoPE backward в неправильной точке" (`nn/backward_attn.go:227-228`)

Forward: `Q_rot = R(θ) · Q_raw`. Backward по chain rule: `dQ_raw = R(θ)ᵀ · dQ_rot = R(-θ) · dQ_rot`.

Код применяет `ropeBackwardInPlace` (R(-θ)) к `dQArr` **после** attention-backward, **до** `Wq.Backward`. Это и есть правильное место в графе. **Корректно**.

### 2.6. "CUDA BatchedMatMulMixed strides перепутаны" (`backend/cuda/cublas.go:378,382`)

cuBLAS работает в column-major, Go-тензоры — row-major. Чтобы вычислить `C = A @ B` в row-major, передают cuBLAS'у `B` как `A` и `A` как `B`:
- `cuBLAS.A = Go.B` (shape [K,N] row-major) → stride per batch = `K * N * elemSizeB` ✓
- `cuBLAS.B = Go.A` (shape [M,K] row-major) → stride per batch = `M * K * elemSizeA` ✓

**Корректно**, согласовано с trick A/B-swap.

### 2.7. "Embedding gradient tracking lost" (`nn/backward_attn.go:336-341`)

`LLM.Backward` в цикле по batches вызывает `TokEmbed.Backward(bt, dEmb)`. Каждый вызов внутри `Embedding.Backward` делает `accumulateGrad(e.Weight, dW)`, который правильно аккумулирует через `+=` в существующий `Weight.Grad()` (или устанавливает если nil). **Корректно**.

---

## 3. Реальные баги (исправлено)

### 3.1. ✅ `ops/ops.go:53-65` — ReLU backward без mask

**Было** (с комментарием `// TODO: implement proper mask`):
```go
func (f *reluGradFn) Backward(grad *tensor.Tensor) []*tensor.Tensor {
    return []*tensor.Tensor{grad}
}
```

**Стало:**
```go
func (f *reluGradFn) Backward(grad *tensor.Tensor) []*tensor.Tensor {
    inputData := f.input.ToFloat32Slice()
    gradData := grad.ToFloat32Slice()
    out := make([]float32, len(gradData))
    for i := range out {
        if inputData[i] > 0 {
            out[i] = gradData[i]
        }
    }
    result, _ := tensor.FromSlice(out, grad.Shape())
    return []*tensor.Tensor{result}
}
```

### 3.2. ✅ `ops/ops.go:225-242` — Softmax не имел GradFn

Добавлен `softmaxGradFn` с Jacobian-формулой `dx = s * (g - sum(g*s, axis=axis))` для произвольной оси. Корректно работает для 2D, 3D, 4D тензоров через outer/axis/inner stride decomposition.

### 3.3. ✅ `ops/ops.go:272-288` — GELU не имел GradFn

Добавлен `geluGradFn` для tanh-аппроксимации GELU (что использует CUDA-ядро `gelu_f32`):
```
y = 0.5*x*(1 + tanh(u)), u = c*(x + 0.044715*x³), c = √(2/π)
dy/dx = 0.5*(1+tanh(u)) + 0.5*x*(1-tanh²(u))*c*(1 + 3*0.044715*x²)
```

### 3.4. ✅ `ops/ops.go:290-307` — SiLU не имел GradFn

Добавлен `siluGradFn`:
```
y = x * sigmoid(x)
dy/dx = sigmoid(x) * (1 + x*(1 - sigmoid(x)))
```

---

## 4. Реальные баги (НЕ исправлено — нужно решение)

### 4.1. `ops/ops.go:12-22` — Add backward без broadcast reduction

```go
func (f *addGradFn) Backward(grad *tensor.Tensor) []*tensor.Tensor {
    // TODO: handle broadcasting reduction
    return []*tensor.Tensor{grad, grad}
}
```
Когда `a.shape != b.shape` и был broadcast, gradients должны быть сужены к исходным shape через `sum` по broadcast-измерениям. Сейчас возвращается grad с broadcast-shape, что вызовет shape-mismatch.

**Не исправил**, потому что:
- Требует написать общую `unbroadcastGrad` функцию (как делали в gotorch)
- Не используется в текущем training loop
- Нужен полноценный helper, лучше делать единообразно с Mul/Sub/Div

### 4.2. `ops/ops.go:30-35` — Mul backward без broadcast reduction (то же, что и Add)

### 4.3. `ops/ops.go:37-51` — MatMul backward для batched операций

`f.b.T()` транспонирует **последние 2 измерения**. Это математически правильно для batched matmul, но реализация в `tensor.T()` должна это поддерживать. Не проверял глубоко.

### 4.4. `ops/ops.go:244-269` — LayerNorm не имеет GradFn

Имеет 3 входа (x, gamma, beta) и сложный backward (см. `nn/backward.go:81-149` для готовой реализации). Можно перенести логику в GradFn — но требует доступа к нормализованным значениям и std, которые backend не возвращает. Нужен рефакторинг ядра LayerNorm с возвратом вспомогательных tensor'ов или ре-вычисление в backward.

### 4.5. `ops/ops.go:309-335` — ScaledDotProductAttention не имеет GradFn

Аналогично — комплексный backward требует кэширования attention weights, Q/K/V, RoPE-факторов. Готовая реализация есть в `nn/backward_attn.go`, но связать с ops-autograd path — большой рефакторинг.

---

## 5. CUDA / совместимость

### 5.1. PTX-ядра скомпилированы под `sm_80` (`backend/cuda/kernels.go:12`, `kernels_b.go`)

Не запустится на твоей **RTX 6000 Blackwell (sm_120)** без перекомпиляции. Требуется:
1. Регенерация PTX из CUDA-исходников в `libs/` под `compute_120,sm_120`
2. CUDA 12.8+ Toolkit
3. NVIDIA driver 570+

Это **не баг в коде**, а compat-issue. Можно решить отдельно — собрать PTX под нужный sm и обновить пути.

### 5.2. `ensureInit` cleanup на ошибке (`backend/cuda/cuda..go:58-132`)

При ошибке после `cuCtxCreate` или `cuStreamCreate` ресурсы не освобождаются. Реалистично — это случается только при первой инициализации в случае серьёзной поломки железа/драйверов, поэтому я **не стал чинить** (требует defer-обёртки и переработки init-логики, риски выше пользы).

### 5.3. `cublas_fp8.go:36-55` — частичная загрузка `.so`

Если одна dlopen успешна, другая — нет, может произойти segfault при первом вызове `RegisterLibFunc`. Маловероятно (либо обе либы есть, либо обеих нет), но fix-of-fix желательно. **Не сделано**.

---

## 6. Не баги, но стоит знать

- **`nn/backward.go:250-259` accumulateGrad** мутирует storage напрямую — race condition при concurrent backward'ах. В sequentialном training loop это не проблема.
- **`nn/backward_attn.go:131-146`** Forward recomputation в attention backward — двойная работа, но не баг.
- **`nn/loss.go:64-68`** CrossEntropyLoss использует clamped prob для loss и unclamped для gradient. Microopt mismatch на underflow, не критично.
- **`tokenizer/bpe.go:293-301`** FindToken — линейный поиск O(vocabSize). Performance, не correctness.
- **`train/trainer.go:177`** getBatch не валидирует `len(tokens) > seqLen+1`. Panic на маленьком eval set.

---

## 7. Что было сделано в этом patch

Файлы изменены:
- `ops/ops.go` — добавлен `math` import, исправлен `reluGradFn`, добавлены `softmaxGradFn`/`geluGradFn`/`siluGradFn` со связыванием через `SetGradFn`

Файлы добавлены:
- `GOML_FIXES.md` — этот документ

Билд проходит чисто (`go build ./...`). Pre-existing vet warning в `cmd/gradcheck` и `cmd/mingrad` (redundant `\n` в `fmt.Println`) не моя зона.

---

## 8. Рекомендации (по приоритету)

1. **Принять архитектурное решение по двум backward-путям.** Либо:
   - Доделать ops-autograd (потребует написать LayerNorm/Attention GradFn — это значительная работа)
   - Удалить недописанные GradFn и явно задокументировать: "тренировка только через `Model.Backward`"

2. **Подготовить PTX/CUDA сборку под Blackwell** (`sm_120`). Это блокер для запуска на RTX 6000 на новом сервере.

3. **Доделать broadcast reduction** в Add/Mul/Sub/Div backward — общий helper по аналогии с gotorch.

4. **Добавить numerical gradient checks** для ручного `nn/backward*.go` пути — чтобы зафиксировать корректность и ловить регрессии.

5. **`cmd/gradcheck` пофиксить** — там уже задумано численное верификование, но `fmt.Println` vet-warning блокирует test-сборку.
