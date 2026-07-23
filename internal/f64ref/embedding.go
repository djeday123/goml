package f64ref

// EmbeddingF64 — эталонная embedding lookup в CPU-FP64.
//
// Forward: y[b, s, :] = weight[indices[b, s], :]
//   weight  : [VocabSize, EmbedDim]
//   indices : [batch, seqLen] int64
//   y       : [batch, seqLen, EmbedDim]
//
// Sparse gather: не задействует GPU, тривиально копируется row-slices.
//
// Backward: dW[i, :] += sum_over(b,s where indices[b,s]==i) dy[b, s, :]
//           dIndices не считается (int, не имеет градиента).

import "fmt"

type EmbeddingF64 struct {
	Weight    *F64Tensor // [VocabSize, EmbedDim]
	VocabSize int
	EmbedDim  int
}

func NewEmbeddingF64(vocabSize, embedDim int) *EmbeddingF64 {
	w := Zeros(vocabSize, embedDim)
	// Малая шкала для стабильности при численном градиенте.
	return &EmbeddingF64{
		Weight:    w,
		VocabSize: vocabSize,
		EmbedDim:  embedDim,
	}
}

// Forward — indices это []int64 форма [batch, seqLen] хранится плоско.
func (e *EmbeddingF64) Forward(indices []int64, batch, seqLen int) *F64Tensor {
	if len(indices) != batch*seqLen {
		panic(fmt.Sprintf("EmbeddingF64: indices len %d != batch*seqLen %d", len(indices), batch*seqLen))
	}
	out := Zeros(batch, seqLen, e.EmbedDim)
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			idx := int(indices[b*seqLen+s])
			if idx < 0 || idx >= e.VocabSize {
				panic(fmt.Sprintf("EmbeddingF64: index %d out of vocab %d", idx, e.VocabSize))
			}
			// copy embedding row.
			srcOff := idx * e.EmbedDim
			dstOff := (b*seqLen + s) * e.EmbedDim
			copy(out.Data[dstOff:dstOff+e.EmbedDim], e.Weight.Data[srcOff:srcOff+e.EmbedDim])
		}
	}
	return out
}

// Backward: аккумулирует dy в dW по строкам, соответствующим indices.
// Возвращает dW [VocabSize, EmbedDim].
func (e *EmbeddingF64) Backward(indices []int64, dy *F64Tensor, batch, seqLen int) *F64Tensor {
	if len(indices) != batch*seqLen {
		panic("EmbeddingF64.Backward: indices len mismatch")
	}
	if dy.Shape[0] != batch || dy.Shape[1] != seqLen || dy.Shape[2] != e.EmbedDim {
		panic(fmt.Sprintf("EmbeddingF64.Backward: dy shape %v mismatch [%d,%d,%d]",
			dy.Shape, batch, seqLen, e.EmbedDim))
	}
	dW := Zeros(e.VocabSize, e.EmbedDim)
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			idx := int(indices[b*seqLen+s])
			srcOff := (b*seqLen + s) * e.EmbedDim
			dstOff := idx * e.EmbedDim
			for i := 0; i < e.EmbedDim; i++ {
				dW.Data[dstOff+i] += dy.Data[srcOff+i]
			}
		}
	}
	return dW
}
