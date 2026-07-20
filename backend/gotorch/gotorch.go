// Package gotorch — adapter реализация backend.Backend поверх gotorch/cuda
// (R03b-impl-2, 2026-07-18).
//
// Архитектура: тонкая обёртка, каждый adapter-метод обёртывает goml.Storage
// в gotorch.ForeignStorage через WrapDevicePtr, вызывает нужный
// gotorch-метод, возвращает результат. Композитные операции (LayerNorm,
// Embedding, RoPE, SDPA, Sum/Mean-axis, Max, Fill, Arange, Where, Gelu,
// Silu, batched MatMul) остаются на goml PTX-ядрах через embedded fallback
// (`fb *cuda.Backend`).
//
// КОНТРАКТ синхронизации (см. R03b_design.md вопрос 5):
//   - Adapter при инициализации инжектирует goml stream в gotorch через
//     gotorch.PuregoBackend.SetStream. После этого ОБА мира работают на
//     одной очереди — порядок операций гарантирован stream'ом.
//   - Полные Sync внутри тела adapter-методов ЗАПРЕЩЕНЫ (осушают конвейер).
//   - Единственный допустимый sync — конец Step (goml.trainer сам делает
//     это неявно через materialization loss'а).
//
// Регистрация: adapter НЕ регистрируется в backend registry автоматически
// (иначе конфликт с уже зарегистрированным cuda). Пользователь включает
// его явно через `gotorch.Enable()` перед первым использованием backend.CUDA,
// либо через env var GOML_BACKEND=gotorch (проверяется в Enable()).
package gotorch

import (
	"fmt"
	"os"
	"unsafe"

	"github.com/djeday123/goml/backend"
	gomlcuda "github.com/djeday123/goml/backend/cuda"

	gtcuda "github.com/djeday123/gotorch/cuda"
)

// Backend — adapter реализация goml.Backend через gotorch + goml.cuda fallback.
type Backend struct {
	gt gtcuda.Backend    // покрытие direct-методов (Add, MatMul, ...)
	fb *gomlcuda.Backend // fallback для stays-in-goml методов
}

// Assertion: Backend удовлетворяет goml.Backend.
var _ backend.Backend = (*Backend)(nil)

// Enable — явное включение adapter'а в backend registry.
//
// Использование:
//
//	import _ "github.com/djeday123/goml/backend/gotorch"
//	// ... затем в начале программы:
//	if err := gotorch.Enable(); err != nil { log.Fatal(err) }
//
// Либо через env var:
//
//	GOML_BACKEND=gotorch ./my-app
//
// Enable() зовётся один раз. Повторный вызов — no-op.
//
// Внутри: инициализирует goml.cuda backend (fallback), достаёт его stream,
// создаёт gotorch backend, инжектирует stream, регистрирует adapter в
// backend registry с ключом backend.CUDA (перекрывая существующий).
var enabled bool

func Enable() error {
	if enabled {
		return nil
	}

	// 1. Get fallback goml.cuda backend (must be initialized already for Stream()).
	fbGeneric, err := backend.Get(backend.CUDA)
	if err != nil {
		return fmt.Errorf("gotorch adapter: goml.cuda backend not registered: %w", err)
	}
	fb, ok := fbGeneric.(*gomlcuda.Backend)
	if !ok {
		return fmt.Errorf("gotorch adapter: goml CUDA backend has unexpected type %T", fbGeneric)
	}

	// Force lazy init of fallback via a probe Alloc — иначе Stream() = 0.
	probe, err := fb.Alloc(64)
	if err != nil {
		return fmt.Errorf("gotorch adapter: fallback probe Alloc: %w", err)
	}
	fb.Free(probe)

	// 2. Get goml stream (создан в ensureInit).
	gomlStream := fb.Stream()
	if gomlStream == 0 {
		return fmt.Errorf("gotorch adapter: goml Stream() returned 0 after Alloc — no stream created")
	}

	// 3. Create gotorch backend on the same primary context (Fix A already applied).
	gt, err := gtcuda.NewBackend(0)
	if err != nil {
		return fmt.Errorf("gotorch adapter: NewBackend(0): %w", err)
	}
	pgt, ok := gt.(*gtcuda.PuregoBackend)
	if !ok {
		return fmt.Errorf("gotorch adapter: NewBackend returned unexpected type %T", gt)
	}

	// 4. Inject stream — оба мира теперь на одной очереди.
	// gomlStream — uintptr (CUDA handle), передаём как unsafe.Pointer через
	// reinterpret-cast (тот же pattern что gotorch.UnsafeExtractDevicePtr).
	streamPtr := *(*unsafe.Pointer)(unsafe.Pointer(&gomlStream))
	if err := pgt.SetStream(streamPtr); err != nil {
		return fmt.Errorf("gotorch adapter: SetStream: %w", err)
	}

	// 5. Register adapter overriding the default CUDA backend.
	adapter := &Backend{gt: pgt, fb: fb}
	backend.Register(adapter)
	enabled = true
	return nil
}

// EnableIfEnv включает adapter если env GOML_BACKEND == "gotorch".
// Идиома: вызывать в main() до первого backend.Get.
func EnableIfEnv() error {
	if os.Getenv("GOML_BACKEND") != "gotorch" {
		return nil
	}
	return Enable()
}

func (b *Backend) Name() string                   { return "gotorch-adapter" }
func (b *Backend) DeviceType() backend.DeviceType { return backend.CUDA }
