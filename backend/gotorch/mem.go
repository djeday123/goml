package gotorch

// Adapter memory management: Alloc/Free/Copy/ToDevice.
// Alloc — прямая через gotorch.Alloc, возвращает наш *Storage (обёртка над
// gotorch.Storage). Free — прямая через gotorch.Free.
// Copy (D2D) — прямая через gotorch.CopyD2D.
// ToDevice — compose из Alloc + CopyH2D/D2H.
//
// Владение: adapter владеет своей памятью через gotorch.Alloc. goml.Pool
// НЕ вовлечён (тот принадлежит fb.cuda, используется только для stays-in-goml
// операций). Два раздельных менеджера памяти в одном процессе (R03b_design.md
// вопрос 4).

import (
	"fmt"
	"unsafe"

	"github.com/djeday123/goml/backend"

	gtcuda "github.com/djeday123/gotorch/cuda"
)

func (b *Backend) Alloc(byteLen int) (backend.Storage, error) {
	if byteLen <= 0 {
		return nil, fmt.Errorf("gotorch adapter Alloc: byteLen must be > 0")
	}
	s, err := b.gt.Alloc(byteLen)
	if err != nil {
		return nil, fmt.Errorf("gotorch adapter Alloc: %w", err)
	}
	return &Storage{
		gtStore: s,
		byteLen: byteLen,
		device:  backend.CUDADevice(0),
	}, nil
}

func (b *Backend) Free(s backend.Storage) {
	if s == nil {
		return
	}
	// Только Storage адаптера может быть освобождён — иначе это чужая
	// память (не через наш Alloc), не трогаем.
	as, ok := s.(*Storage)
	if !ok {
		return
	}
	if err := b.gt.Free(as.gtStore); err != nil {
		// gotorch.Free возвращает error, но goml.Backend.Free — void.
		// Логируем через nil-op; в production можно завести logger.
		_ = err
	}
	// Помечаем как освобождённый — предотвращает double-free.
	as.gtStore = gtcuda.Storage{}
	as.byteLen = 0
}

func (b *Backend) Copy(dst, src backend.Storage, byteLen int) error {
	if byteLen <= 0 {
		return fmt.Errorf("gotorch adapter Copy: byteLen must be > 0")
	}
	dFor := wrapForeign(dst)
	sFor := wrapForeign(src)
	return b.gt.CopyD2D(dFor, sFor, byteLen)
}

// ToDevice — CPU↔GPU перенос. Compose (см. таблица покрытия #6).
func (b *Backend) ToDevice(dst backend.Device, src backend.Storage) (backend.Storage, error) {
	if dst.Type == backend.CUDA && src.Device().Type == backend.CPU {
		// CPU → GPU: alloc + H2D.
		newStore, err := b.Alloc(src.ByteLen())
		if err != nil {
			return nil, err
		}
		hostBytes := src.Bytes()
		if hostBytes != nil {
			if err := b.gt.CopyH2D(wrapForeign(newStore), hostBytes); err != nil {
				b.Free(newStore)
				return nil, err
			}
		}
		return newStore, nil
	}
	if dst.Type == backend.CPU && src.Device().Type == backend.CUDA {
		// GPU → CPU. Fb.ToDevice не подходит — оно требует src to be *cuda.Storage
		// (наш *gotorch.Storage panic на type-assert). Собираем сами: alloc CPU
		// storage через CPU backend + gotorch.CopyD2H в его Bytes.
		cpuB, err := backend.Get(backend.CPU)
		if err != nil {
			return nil, fmt.Errorf("gotorch adapter ToDevice(GPU→CPU): get CPU backend: %w", err)
		}
		cpuStore, err := cpuB.Alloc(src.ByteLen())
		if err != nil {
			return nil, err
		}
		if err := b.gt.CopyD2H(cpuStore.Bytes(), wrapForeign(src)); err != nil {
			cpuB.Free(cpuStore)
			return nil, err
		}
		return cpuStore, nil
	}
	return nil, fmt.Errorf("gotorch adapter ToDevice: unsupported %s -> %s", src.Device(), dst)
}

// Ensure imports used (silence linter if wrapper unused in some builds).
var _ = unsafe.Sizeof(uintptr(0))
