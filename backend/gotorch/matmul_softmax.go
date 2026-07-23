package gotorch

// impl-3 MatMul (batch=1 direct) + Softmax (axis=-1 direct).
// Batched MatMul и Softmax по не-последней оси делегируются в fb (stays-in-goml
// по R03b_design.md таблица покрытия).

import (
	"fmt"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
)

// MatMul: batch=1 → gotorch direct; batched → gotorch strided-batched (B-impl-1).
// goml.cuda.MatMul сигнатура: A[M,K] × B[K,N] → C[M,N] (batched через shapeA[..-2] product).
// gotorch.MatMulStridedBatchedF32: loop cublasSgemm по batch (тот же паттерн что goml BatchedMatMulF32).
func (b *Backend) MatMul(dst, a, bs backend.Storage, shapeA, shapeB core.Shape, dtype core.DType) error {
	if err := requireF32("MatMul", dtype); err != nil {
		return err
	}
	ndimA := len(shapeA)
	ndimB := len(shapeB)
	if ndimA < 2 || ndimB < 2 {
		return fmt.Errorf("gotorch adapter MatMul: shapes must have ndim >= 2, got shapeA=%v shapeB=%v", shapeA, shapeB)
	}
	M := shapeA[ndimA-2]
	K := shapeA[ndimA-1]
	N := shapeB[ndimB-1]

	batchSize := 1
	for i := 0; i < ndimA-2; i++ {
		batchSize *= shapeA[i]
	}
	if batchSize > 1 {
		// B-impl-1: стрелка delegate -> direct через loop-batched (тот же
		// паттерн что goml.cuda.BatchedMatMulF32, cublas.go:264). Bit-exact vs
		// goml при одинаковом math mode (проверяется в A/B тесте).
		return b.gt.MatMulStridedBatchedF32(
			wrapForeign(a), wrapForeign(bs), wrapForeign(dst),
			batchSize, M, N, K,
			int64(M*K), int64(K*N), int64(M*N),
		)
	}

	return b.gt.MatMulF32(wrapForeign(a), wrapForeign(bs), wrapForeign(dst), M, N, K)
}

// Softmax: axis=-1 (или последняя ось) → gotorch direct через rows/cols pattern;
// иначе → fb (stays-in-goml).
func (b *Backend) Softmax(dst, src backend.Storage, shape core.Shape, axis int, dtype core.DType) error {
	if err := requireF32("Softmax", dtype); err != nil {
		return err
	}
	ndim := len(shape)
	if ndim == 0 {
		return fmt.Errorf("gotorch adapter Softmax: shape must be non-empty")
	}
	// Нормализуем негативную ось.
	if axis < 0 {
		axis = ndim + axis
	}
	if axis != ndim-1 {
		// Non-last-axis softmax stays-in-goml (goml есть общая реализация для любой оси).
		return b.fb.Softmax(dst, src, shape, axis, dtype)
	}

	// Last-axis: rows = prod(shape[:-1]), cols = shape[-1].
	cols := shape[ndim-1]
	rows := 1
	for i := 0; i < ndim-1; i++ {
		rows *= shape[i]
	}
	return b.gt.SoftmaxF32(wrapForeign(src), wrapForeign(dst), rows, cols)
}
