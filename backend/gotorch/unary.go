package gotorch

// impl-3 direct-методы unary для F32.
// Обёртки над gotorch: N=shape.NumElements(), foreign-wrap для входа/выхода.
//
// Sync-контракт: без .Sync() внутри метода. Оба мира на одном stream'е
// (injected в Enable), порядок операций гарантирован stream'ом. TestAdapterNoFullSync
// продолжает контролировать статически.

import (
	"fmt"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
)

// requireF32 — общая guard-проверка для F32-only-методов.
func requireF32(name string, dtype core.DType) error {
	if dtype != core.Float32 {
		return fmt.Errorf("gotorch adapter %s: only float32 supported, got %s", name, dtype)
	}
	return nil
}

func (b *Backend) Neg(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	if err := requireF32("Neg", dtype); err != nil {
		return err
	}
	return b.gt.NegF32(wrapForeign(src), wrapForeign(dst), shape.NumElements())
}

func (b *Backend) Exp(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	if err := requireF32("Exp", dtype); err != nil {
		return err
	}
	return b.gt.ExpF32(wrapForeign(src), wrapForeign(dst), shape.NumElements())
}

func (b *Backend) Log(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	if err := requireF32("Log", dtype); err != nil {
		return err
	}
	return b.gt.LogF32(wrapForeign(src), wrapForeign(dst), shape.NumElements())
}

func (b *Backend) Tanh(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	if err := requireF32("Tanh", dtype); err != nil {
		return err
	}
	return b.gt.TanhF32(wrapForeign(src), wrapForeign(dst), shape.NumElements())
}

func (b *Backend) Relu(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	if err := requireF32("Relu", dtype); err != nil {
		return err
	}
	return b.gt.ReLUF32(wrapForeign(src), wrapForeign(dst), shape.NumElements())
}

func (b *Backend) Sigmoid(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	if err := requireF32("Sigmoid", dtype); err != nil {
		return err
	}
	return b.gt.SigmoidF32(wrapForeign(src), wrapForeign(dst), shape.NumElements())
}
