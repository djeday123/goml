package gotorch

// Delegate-методы: stays-in-goml по решению R03b_design.md таблица покрытия.
// impl-2 стубает все НЕ-direct методы через b.fb (goml.cuda) — devPtr()
// fallthrough branch (`uintptr(s.Ptr())`) корректно достаёт указатель из
// нашего *Storage без type-assert.
//
// В impl-3 direct-методы (Neg, Exp, Log, Tanh, Relu, Sigmoid, Sub, Mul, Div,
// MatMul-batch=1, Softmax-axis=-1) переносятся в отдельные файлы (по образцу
// add.go), а этот файл сжимается до stays-in-goml + gaps (Abs, Sqrt пока
// stays-in-goml до impl-6).

import (
	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
)

// --- Unary (все stays-in-goml в impl-2; часть станет direct в impl-3) ---

func (b *Backend) Neg(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.fb.Neg(dst, src, shape, dtype)
}
func (b *Backend) Abs(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.fb.Abs(dst, src, shape, dtype)
}
func (b *Backend) Exp(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.fb.Exp(dst, src, shape, dtype)
}
func (b *Backend) Log(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.fb.Log(dst, src, shape, dtype)
}
func (b *Backend) Sqrt(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.fb.Sqrt(dst, src, shape, dtype)
}
func (b *Backend) Tanh(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.fb.Tanh(dst, src, shape, dtype)
}
func (b *Backend) Relu(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.fb.Relu(dst, src, shape, dtype)
}
func (b *Backend) Gelu(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.fb.Gelu(dst, src, shape, dtype)
}
func (b *Backend) Sigmoid(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.fb.Sigmoid(dst, src, shape, dtype)
}
func (b *Backend) Silu(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.fb.Silu(dst, src, shape, dtype)
}

// --- Binary (Add — direct в add.go; Sub/Mul/Div — stays до impl-3) ---

func (b *Backend) Sub(dst, a, bs backend.Storage, shapeA, shapeB, shapeOut core.Shape, dtype core.DType) error {
	return b.fb.Sub(dst, a, bs, shapeA, shapeB, shapeOut, dtype)
}
func (b *Backend) Mul(dst, a, bs backend.Storage, shapeA, shapeB, shapeOut core.Shape, dtype core.DType) error {
	return b.fb.Mul(dst, a, bs, shapeA, shapeB, shapeOut, dtype)
}
func (b *Backend) Div(dst, a, bs backend.Storage, shapeA, shapeB, shapeOut core.Shape, dtype core.DType) error {
	return b.fb.Div(dst, a, bs, shapeA, shapeB, shapeOut, dtype)
}

// --- Reduce (все stays-in-goml permanent, table row 21-23) ---

func (b *Backend) Sum(dst, src backend.Storage, shape core.Shape, axes []int, keepDim bool, dtype core.DType) error {
	return b.fb.Sum(dst, src, shape, axes, keepDim, dtype)
}
func (b *Backend) Max(dst, src backend.Storage, shape core.Shape, axes []int, keepDim bool, dtype core.DType) error {
	return b.fb.Max(dst, src, shape, axes, keepDim, dtype)
}
func (b *Backend) Mean(dst, src backend.Storage, shape core.Shape, axes []int, keepDim bool, dtype core.DType) error {
	return b.fb.Mean(dst, src, shape, axes, keepDim, dtype)
}

// --- MatMul (batched — stays-in-goml permanent; batch=1 — direct в impl-3) ---

func (b *Backend) MatMul(dst, a, bs backend.Storage, shapeA, shapeB core.Shape, dtype core.DType) error {
	return b.fb.MatMul(dst, a, bs, shapeA, shapeB, dtype)
}

// --- Composite (все stays-in-goml permanent; Softmax axis=-1 direct в impl-3) ---

func (b *Backend) Softmax(dst, src backend.Storage, shape core.Shape, axis int, dtype core.DType) error {
	return b.fb.Softmax(dst, src, shape, axis, dtype)
}
func (b *Backend) LayerNorm(dst, src, gamma, beta backend.Storage, shape core.Shape, normAxis int, eps float64, dtype core.DType) error {
	return b.fb.LayerNorm(dst, src, gamma, beta, shape, normAxis, eps, dtype)
}
func (b *Backend) Embedding(dst, weight, indices backend.Storage, vocabSize, embedDim, seqLen int, dtype core.DType) error {
	return b.fb.Embedding(dst, weight, indices, vocabSize, embedDim, seqLen, dtype)
}
func (b *Backend) RoPE(dst, src backend.Storage, shape core.Shape, headDim int, base float64, dtype core.DType) error {
	return b.fb.RoPE(dst, src, shape, headDim, base, dtype)
}
func (b *Backend) ScaledDotProductAttention(dst, q, k, v backend.Storage, batchSize, numHeads, seqLen, headDim int, causal bool, dtype core.DType) error {
	return b.fb.ScaledDotProductAttention(dst, q, k, v, batchSize, numHeads, seqLen, headDim, causal, dtype)
}

// --- Fill/Compare (все stays-in-goml permanent) ---

func (b *Backend) Fill(dst backend.Storage, shape core.Shape, value float64, dtype core.DType) error {
	return b.fb.Fill(dst, shape, value, dtype)
}
func (b *Backend) Arange(dst backend.Storage, start, step float64, n int, dtype core.DType) error {
	return b.fb.Arange(dst, start, step, n, dtype)
}
func (b *Backend) Where(dst, cond, a, bs backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.fb.Where(dst, cond, a, bs, shape, dtype)
}
