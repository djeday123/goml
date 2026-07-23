package gotorch

// P2-RMS: adapter direct-methods для RMSNorm F32/F64 fwd+bwd.
//
// Правила:
//  - goml.backend.Backend interface НЕ содержит RMSNorm — только LayerNorm.
//    Мы добавляем RMSNorm как extension API на *gotorch.Backend (конкретный
//    тип, доступный через type-assertion). Пользователь:
//      b, _ := backend.Get(backend.CUDA)
//      rb, ok := b.(*gotorch.Backend); if ok { rb.RMSNormF32(...) }
//  - goml.backend/cuda.RMSNorm — FP16 dlopen orphan (0 callers), не пересекается
//    с этими F32/F64 портами.
//  - Direct-only: без fb-делегации (fb не имеет F32/F64 RMSNorm вообще).
//  - Sync-контракт как у всех adapter-методов: без внутреннего Sync (оба мира
//    на одном injected stream'е).

import (
	"fmt"

	"github.com/djeday123/goml/backend"
)

// RMSNormF32 — y = gamma * x / sqrt(mean(x²)+eps).
// Все буферы row-major [rows, cols]; gamma [cols].
func (b *Backend) RMSNormF32(x, gamma, y backend.Storage, rows, cols int, eps float32) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("gotorch adapter RMSNormF32: rows/cols must be > 0, got rows=%d cols=%d", rows, cols)
	}
	return b.gt.RMSNormF32(wrapForeign(x), wrapForeign(gamma), wrapForeign(y), rows, cols, eps)
}

// RMSNormF64 — F64 version. Для F64-судейства и pedantic-точности.
func (b *Backend) RMSNormF64(x, gamma, y backend.Storage, rows, cols int, eps float64) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("gotorch adapter RMSNormF64: rows/cols must be > 0, got rows=%d cols=%d", rows, cols)
	}
	return b.gt.RMSNormF64(wrapForeign(x), wrapForeign(gamma), wrapForeign(y), rows, cols, eps)
}

// RMSNormGradF32 — backward: dx, dgamma. dgamma обнуляется внутри gotorch перед atomic-reduction.
func (b *Backend) RMSNormGradF32(x, gamma, dy, dx, dgamma backend.Storage, rows, cols int, eps float32) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("gotorch adapter RMSNormGradF32: rows/cols must be > 0")
	}
	return b.gt.RMSNormGradF32(
		wrapForeign(x), wrapForeign(gamma), wrapForeign(dy),
		wrapForeign(dx), wrapForeign(dgamma),
		rows, cols, eps,
	)
}

// RMSNormGradF64 — F64 backward.
func (b *Backend) RMSNormGradF64(x, gamma, dy, dx, dgamma backend.Storage, rows, cols int, eps float64) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("gotorch adapter RMSNormGradF64: rows/cols must be > 0")
	}
	return b.gt.RMSNormGradF64(
		wrapForeign(x), wrapForeign(gamma), wrapForeign(dy),
		wrapForeign(dx), wrapForeign(dgamma),
		rows, cols, eps,
	)
}
