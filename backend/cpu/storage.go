package cpu

import (
	"unsafe"

	"github.com/vugar/goml/backend"
)

// storage is a CPU memory buffer backed by a Go byte slice.
type storage struct {
	data []byte
}

func newStorage(byteLen int) *storage {
	return &storage{data: make([]byte, byteLen)}
}

func (s *storage) Device() backend.Device { return backend.CPU0 }

func (s *storage) Ptr() uintptr {
	if len(s.data) == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(&s.data[0]))
}

func (s *storage) ByteLen() int { return len(s.data) }

func (s *storage) Free() {
	s.data = nil
}
