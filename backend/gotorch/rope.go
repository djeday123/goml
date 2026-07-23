package gotorch

// P4-ROPE: adapter direct-methods для RoPE F32/F64 fwd+bwd.
//
// Дизайн: methods на конкретном типе *gotorch.Backend (extension API), НЕ через
// backend.Backend interface -- goml.Backend.RoPE имеет сигнатуру
// (dst, src, shape, headDim, base float64, dtype), а наш API разделяет F32/F64
// и требует cos/sin таблицы для F64 (host math). Drop-in замена goml.RoPE
// на adapter возможна для F32-пути (шаблон goml PTX скопирован bit-exact),
// но требует переработать вызов на стороне nn/attention.go под split API.
//
// Стрелка delegate -> direct для F32-пути:
//   можно сделать -- goml.Backend.RoPE(F32) -> gotorch.RoPEF32(...) через
//   выделение shape[0..2] = (batch, heads, seqLen). Реализуем как метод RoPE
//   в интерфейсе через switch dtype: F32 -> direct, F64 -> ошибка (F64 путь
//   требует cos/sin таблиц, вне сигнатуры backend.Backend.RoPE).
//
// Sync-контракт: без внутреннего Sync (injected stream). NoFullSync grep-guard clean.

import (
	"fmt"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
)

// RoPE (via backend.Backend interface). F32-путь через gotorch direct;
// F64 не поддерживается через этот интерфейс (нужны cos/sin таблицы).
func (b *Backend) RoPE(dst, src backend.Storage, shape core.Shape, headDim int, base float64, dtype core.DType) error {
	if dtype != core.Float32 {
		// Fallback в fb -- goml.cuda.RoPE поддерживает только F32, вернёт ошибку.
		return b.fb.RoPE(dst, src, shape, headDim, base, dtype)
	}
	if len(shape) < 3 {
		return fmt.Errorf("gotorch adapter RoPE: shape ndim >= 3 required, got %v", shape)
	}
	batch := shape[0]
	heads := shape[1]
	seqLen := shape[2]
	return b.gt.RoPEF32(wrapForeign(src), wrapForeign(dst), batch, heads, seqLen, headDim, float32(base))
}

// RoPEF32 -- extension API. Использует goml PTX через adapter (bit-exact vs goml.cuda).
func (b *Backend) RoPEF32(x, out backend.Storage, batch, heads, seqLen, headDim int, base float32) error {
	if batch <= 0 || heads <= 0 || seqLen <= 0 || headDim <= 0 || headDim%2 != 0 {
		return fmt.Errorf("gotorch adapter RoPEF32: dims must be > 0 and headDim even")
	}
	return b.gt.RoPEF32(wrapForeign(x), wrapForeign(out), batch, heads, seqLen, headDim, base)
}

// RoPEGradF32 -- extension API.
func (b *Backend) RoPEGradF32(dy, dx backend.Storage, batch, heads, seqLen, headDim int, base float32) error {
	if batch <= 0 || heads <= 0 || seqLen <= 0 || headDim <= 0 || headDim%2 != 0 {
		return fmt.Errorf("gotorch adapter RoPEGradF32: dims must be > 0 and headDim even")
	}
	return b.gt.RoPEGradF32(wrapForeign(dy), wrapForeign(dx), batch, heads, seqLen, headDim, base)
}

// RoPEF64 -- extension API. cos/sin таблицы генерируются пользователем host-side.
func (b *Backend) RoPEF64(x, cosTable, sinTable, out backend.Storage, batch, heads, seqLen, headDim int) error {
	if batch <= 0 || heads <= 0 || seqLen <= 0 || headDim <= 0 || headDim%2 != 0 {
		return fmt.Errorf("gotorch adapter RoPEF64: dims must be > 0 and headDim even")
	}
	return b.gt.RoPEF64(wrapForeign(x), wrapForeign(cosTable), wrapForeign(sinTable), wrapForeign(out),
		batch, heads, seqLen, headDim)
}

// RoPEGradF64 -- extension API.
func (b *Backend) RoPEGradF64(dy, cosTable, sinTable, dx backend.Storage, batch, heads, seqLen, headDim int) error {
	if batch <= 0 || heads <= 0 || seqLen <= 0 || headDim <= 0 || headDim%2 != 0 {
		return fmt.Errorf("gotorch adapter RoPEGradF64: dims must be > 0 and headDim even")
	}
	return b.gt.RoPEGradF64(wrapForeign(dy), wrapForeign(cosTable), wrapForeign(sinTable), wrapForeign(dx),
		batch, heads, seqLen, headDim)
}
