package gotorch

// impl-3 direct-методы бинарных F32. Broadcasting не поддерживается — тот же
// TODO что в goml.cuda (ops.go:81). Проверяем shape-равенство.

import (
	"fmt"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
)

// requireBinaryFlat — общая проверка совместимости shape для не-broadcast бинарных ops.
func requireBinaryFlat(name string, shapeA, shapeB, shapeOut core.Shape) (int, error) {
	n := shapeA.NumElements()
	if shapeB.NumElements() != n || shapeOut.NumElements() != n {
		return 0, fmt.Errorf("gotorch adapter %s: broadcasting not supported (shapeA=%v shapeB=%v shapeOut=%v)",
			name, shapeA, shapeB, shapeOut)
	}
	return n, nil
}

func (b *Backend) Sub(dst, a, bs backend.Storage, shapeA, shapeB, shapeOut core.Shape, dtype core.DType) error {
	if err := requireF32("Sub", dtype); err != nil {
		return err
	}
	n, err := requireBinaryFlat("Sub", shapeA, shapeB, shapeOut)
	if err != nil {
		return err
	}
	return b.gt.SubF32(wrapForeign(a), wrapForeign(bs), wrapForeign(dst), n)
}

func (b *Backend) Mul(dst, a, bs backend.Storage, shapeA, shapeB, shapeOut core.Shape, dtype core.DType) error {
	if err := requireF32("Mul", dtype); err != nil {
		return err
	}
	n, err := requireBinaryFlat("Mul", shapeA, shapeB, shapeOut)
	if err != nil {
		return err
	}
	return b.gt.MulF32(wrapForeign(a), wrapForeign(bs), wrapForeign(dst), n)
}

func (b *Backend) Div(dst, a, bs backend.Storage, shapeA, shapeB, shapeOut core.Shape, dtype core.DType) error {
	if err := requireF32("Div", dtype); err != nil {
		return err
	}
	n, err := requireBinaryFlat("Div", shapeA, shapeB, shapeOut)
	if err != nil {
		return err
	}
	return b.gt.DivF32(wrapForeign(a), wrapForeign(bs), wrapForeign(dst), n)
}
