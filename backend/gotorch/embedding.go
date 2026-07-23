package gotorch

// P3-EMB + P5A-EMB-I64: adapter direct-methods для Embedding.
//
// Extension API (P3): EmbeddingF32/F64/GradF32/F64 -- int32 индексы.
// Backend interface (P5A): Embedding через int64-фасад (goml пода)т свои
// int64 tokens как есть; фасад gt.EmbeddingF32I64/F64I64 конвертирует в
// int32-канон через переиспользуемый scratch и вызывает канонический kernel.
//
// Стрелка delegate -> direct СХЛОПНУТА (P5A) для forward. Grad-путь backend
// interface goml не имеет (nn.Embedding.Backward -- CPU F32 в nn/backward.go);
// grad наш живёт как extension.
//
// Sync-контракт: без внутреннего Sync (injected stream). NoFullSync grep-guard clean.

import (
	"fmt"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
)

// EmbeddingF32 — gather rows F32 table by int32 indices. See gotorch cuda API doc-string.
func (b *Backend) EmbeddingF32(table, indices, out backend.Storage, vocab, hidden, n int) error {
	if vocab <= 0 || hidden <= 0 || n <= 0 {
		return fmt.Errorf("gotorch adapter EmbeddingF32: vocab/hidden/n must be > 0")
	}
	return b.gt.EmbeddingF32(wrapForeign(table), wrapForeign(indices), wrapForeign(out), vocab, hidden, n)
}

// EmbeddingF64 — F64 version.
func (b *Backend) EmbeddingF64(table, indices, out backend.Storage, vocab, hidden, n int) error {
	if vocab <= 0 || hidden <= 0 || n <= 0 {
		return fmt.Errorf("gotorch adapter EmbeddingF64: vocab/hidden/n must be > 0")
	}
	return b.gt.EmbeddingF64(wrapForeign(table), wrapForeign(indices), wrapForeign(out), vocab, hidden, n)
}

// EmbeddingGradF32 — scatter-accumulate atomicAdd. dtable zeroed внутри gotorch.
// Недетерминизм при коллизиях документирован в тестах atomic-reproducibility.
func (b *Backend) EmbeddingGradF32(indices, dout, dtable backend.Storage, vocab, hidden, n int) error {
	if vocab <= 0 || hidden <= 0 || n <= 0 {
		return fmt.Errorf("gotorch adapter EmbeddingGradF32: vocab/hidden/n must be > 0")
	}
	return b.gt.EmbeddingGradF32(wrapForeign(indices), wrapForeign(dout), wrapForeign(dtable), vocab, hidden, n)
}

// EmbeddingGradF64 — F64 backward.
func (b *Backend) EmbeddingGradF64(indices, dout, dtable backend.Storage, vocab, hidden, n int) error {
	if vocab <= 0 || hidden <= 0 || n <= 0 {
		return fmt.Errorf("gotorch adapter EmbeddingGradF64: vocab/hidden/n must be > 0")
	}
	return b.gt.EmbeddingGradF64(wrapForeign(indices), wrapForeign(dout), wrapForeign(dtable), vocab, hidden, n)
}

// Embedding (backend.Backend interface) — стрелка delegate -> direct через
// P5A-EMB-I64 фасад. Принимает goml сигнатуру с int64 indices, конвертит
// внутри gotorch (cvt_u64_to_u32 в scratch), затем канонический int32 kernel.
// F32 -> EmbeddingF32I64; F64 -> EmbeddingF64I64.
// Дуальный dispatcher: dtype в сигнатуре -> метод-суффикс, без state-режима.
func (b *Backend) Embedding(dst, weight, indices backend.Storage, vocabSize, embedDim, seqLen int, dtype core.DType) error {
	switch dtype {
	case core.Float32:
		return b.gt.EmbeddingF32I64(wrapForeign(weight), wrapForeign(indices), wrapForeign(dst),
			vocabSize, embedDim, seqLen)
	case core.Float64:
		return b.gt.EmbeddingF64I64(wrapForeign(weight), wrapForeign(indices), wrapForeign(dst),
			vocabSize, embedDim, seqLen)
	default:
		// Fallback -- goml.cuda.Embedding поддерживает только F32, вернёт ошибку.
		return b.fb.Embedding(dst, weight, indices, vocabSize, embedDim, seqLen, dtype)
	}
}
