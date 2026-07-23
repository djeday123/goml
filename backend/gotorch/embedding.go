package gotorch

// P3-EMB: adapter direct-methods для Embedding F32/F64 fwd+bwd.
//
// Дизайн: как в P2-RMS — методы на конкретном типе *gotorch.Backend
// (extension API через type-assertion), НЕ через backend.Backend interface.
//
// Причина не-drop-in для goml.Backend.Embedding:
//   goml сигнатура: Embedding(dst, weight, indices, vocab, embed, seqLen, dtype) с int64 indices.
//   Наша сигнатура: EmbeddingF32(table, indices, out, vocab, hidden, n) с int32 indices.
// Замена goml.cuda.Embedding в gputrain-пути (int64 tokens) сломается без
// конверсии int64->int32 на upload'е. Оставляем goml delegate в fb.Embedding
// для gputrain-совместимости; наш порт живёт как library method + подготовка
// адаптера к будущему nn.LLM/Trainer'у где мы контролируем dtype индексов.
//
// Sync-контракт: без внутреннего Sync (injected stream). NoFullSync grep-guard clean.

import (
	"fmt"

	"github.com/djeday123/goml/backend"
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
