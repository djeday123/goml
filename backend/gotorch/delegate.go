package gotorch

// Delegate-методы: stays-in-goml по решению R03b_design.md таблица покрытия.
// impl-3 перенёс 14 direct-методов в отдельные файлы (unary.go, binary.go,
// matmul_softmax.go, add.go). Здесь остаются:
//   - Abs, Sqrt — gap до impl-6 (простые PTX-ядра будут добавлены в gotorch).
//   - Gelu, Silu — permanent stays-in-goml (LLM-специфично, композиция ok).
//   - Sum/Max/Mean-axis — permanent stays-in-goml (goml есть per-row reduce
//     kernel, gotorch имеет только global reduce).
//   - LayerNorm/Embedding/RoPE/SDPA — permanent stays-in-goml (fused kernels).
//   - Fill/Arange/Where — permanent stays-in-goml (редко hot path).

import (
	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
)

// --- Unary gaps (Abs, Sqrt — станут direct в impl-6) + permanent stays ---

func (b *Backend) Abs(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.fb.Abs(dst, src, shape, dtype)
}
func (b *Backend) Sqrt(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.fb.Sqrt(dst, src, shape, dtype)
}
func (b *Backend) Gelu(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.fb.Gelu(dst, src, shape, dtype)
}
func (b *Backend) Silu(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.fb.Silu(dst, src, shape, dtype)
}

// --- Reduce (permanent stays-in-goml, table rows 21-23) ---

func (b *Backend) Sum(dst, src backend.Storage, shape core.Shape, axes []int, keepDim bool, dtype core.DType) error {
	return b.fb.Sum(dst, src, shape, axes, keepDim, dtype)
}
func (b *Backend) Max(dst, src backend.Storage, shape core.Shape, axes []int, keepDim bool, dtype core.DType) error {
	return b.fb.Max(dst, src, shape, axes, keepDim, dtype)
}
func (b *Backend) Mean(dst, src backend.Storage, shape core.Shape, axes []int, keepDim bool, dtype core.DType) error {
	return b.fb.Mean(dst, src, shape, axes, keepDim, dtype)
}

// --- Composite (permanent stays-in-goml) ---

func (b *Backend) LayerNorm(dst, src, gamma, beta backend.Storage, shape core.Shape, normAxis int, eps float64, dtype core.DType) error {
	return b.fb.LayerNorm(dst, src, gamma, beta, shape, normAxis, eps, dtype)
}
// Embedding перенесён в embedding.go — стрелка delegate -> direct через
// P5A-EMB-I64 фасад (принимает goml int64, внутри конвертит в int32-канон).
//
// RoPE перенесён в rope.go — стрелка delegate -> direct для F32-пути (bit-exact
// vs goml.cuda), F64 в fb.RoPE (goml.cuda даёт not-supported).

func (b *Backend) ScaledDotProductAttention(dst, q, k, v backend.Storage, batchSize, numHeads, seqLen, headDim int, causal bool, dtype core.DType) error {
	return b.fb.ScaledDotProductAttention(dst, q, k, v, batchSize, numHeads, seqLen, headDim, causal, dtype)
}

// --- Fill/Compare (permanent stays-in-goml) ---

func (b *Backend) Fill(dst backend.Storage, shape core.Shape, value float64, dtype core.DType) error {
	return b.fb.Fill(dst, shape, value, dtype)
}
func (b *Backend) Arange(dst backend.Storage, start, step float64, n int, dtype core.DType) error {
	return b.fb.Arange(dst, start, step, n, dtype)
}
func (b *Backend) Where(dst, cond, a, bs backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.fb.Where(dst, cond, a, bs, shape, dtype)
}
