package gotorch

// Storage — обёртка над gotorch.Storage, реализующая goml.backend.Storage
// интерфейс. Владеет памятью, аллоцированной через gotorch.Alloc.
//
// R03b-impl-2 discipline (см. R03b_design.md вопрос 4):
//   - Foreign-wrap живёт только внутри одного метода adapter'а.
//   - Adapter не смешивает свою память (через gotorch.Alloc) с goml.Pool
//     (тот используется fallback'ом *cuda.Backend внутри).

import (
	"unsafe"

	"github.com/djeday123/goml/backend"

	gtcuda "github.com/djeday123/gotorch/cuda"
)

// Storage — adapter-owned handle.
type Storage struct {
	gtStore gtcuda.Storage
	byteLen int
	device  backend.Device
}

func (s *Storage) Device() backend.Device { return s.device }
func (s *Storage) Ptr() unsafe.Pointer    { return gtcuda.UnsafeExtractDevicePtr(s.gtStore) }
func (s *Storage) Bytes() []byte          { return nil }
func (s *Storage) ByteLen() int           { return s.byteLen }

// Free — освобождает gtStore через gotorch.Free (симметрично Alloc).
// Adapter НЕ вызывает Free сам внутри своих операций (владение чужое,
// см. R03b_design.md правило 3). Free вызывается только пользователем.
func (s *Storage) Free() {
	if s.gtStore.SizeBytes() == 0 {
		return
	}
	// gotorch.Backend.Free принимает Storage по значению; получить owner
	// backend неявно нельзя, поэтому Adapter владеет собственным gotorch
	// backend'ом и делает Free через него. Реализация в gotorch.go (Backend.Free).
	//
	// Storage.Free() как метод *Storage без указателя на Backend не может
	// вызвать gotorch.Free напрямую — Free zeroes gt state в PuregoBackend
	// pool (если такой введён). Пока полагаемся на adapter.Backend.Free.
	_ = s // no-op via method; use adapter.Backend.Free(s) instead
}

// wrapForeign превращает goml.Storage в gotorch.ForeignStorage через дверь входа.
// Foreign-wrap НЕ владеет памятью — ForeignStorage.Free это no-op by design (api.go:60-68).
func wrapForeign(s backend.Storage) gtcuda.ForeignStorage {
	return gtcuda.WrapDevicePtr(s.Ptr(), s.ByteLen(), 0)
}
