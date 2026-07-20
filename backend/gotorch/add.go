package gotorch

// Adapter direct-method: Add через gotorch.AddF32.
// Первый end-to-end путь для impl-2. Все остальные ops в impl-2 делегируются
// в fb (goml.cuda) — постепенно переводятся в direct в impl-3.
//
// Sync-контракт (R03b_design.md вопрос 5, правка 1):
// НЕТ Sync внутри метода. Оба мира на одном stream'е (injected при Enable),
// порядок операций гарантирован stream'ом. Единственный sync — конец Step
// на стороне пользователя (goml.trainer).

import (
	"fmt"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
)

// Add — direct через gotorch.AddF32. Shape broadcasting не поддерживается
// (см. R03b_design.md таблица покрытия #17 — goml.cuda тоже без broadcast,
// TODO в ops.go:81).
func (b *Backend) Add(dst, a, bs backend.Storage, shapeA, shapeB, shapeOut core.Shape, dtype core.DType) error {
	if dtype != core.Float32 {
		return fmt.Errorf("gotorch adapter Add: only float32 supported, got %s", dtype)
	}
	if shapeA.NumElements() != shapeB.NumElements() || shapeA.NumElements() != shapeOut.NumElements() {
		return fmt.Errorf("gotorch adapter Add: broadcasting not supported yet (shapeA=%v shapeB=%v)", shapeA, shapeB)
	}
	n := shapeA.NumElements()
	aFor := wrapForeign(a)
	bFor := wrapForeign(bs)
	cFor := wrapForeign(dst)
	return b.gt.AddF32(aFor, bFor, cFor, n)
}
