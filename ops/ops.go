package ops

import (
	"fmt"

	"github.com/vugar/goml/backend"
	"github.com/vugar/goml/tensor"
)

// ---- Autograd function implementations ----

type addGradFn struct {
	a, b *tensor.Tensor
}

func (f *addGradFn) Name() string             { return "AddBackward" }
func (f *addGradFn) Inputs() []*tensor.Tensor  { return []*tensor.Tensor{f.a, f.b} }
func (f *addGradFn) Backward(grad *tensor.Tensor) []*tensor.Tensor {
	// d(a+b)/da = 1, d(a+b)/db = 1
	// TODO: handle broadcasting reduction
	return []*tensor.Tensor{grad, grad}
}

type mulGradFn struct {
	a, b *tensor.Tensor
}

func (f *mulGradFn) Name() string             { return "MulBackward" }
func (f *mulGradFn) Inputs() []*tensor.Tensor  { return []*tensor.Tensor{f.a, f.b} }
func (f *mulGradFn) Backward(grad *tensor.Tensor) []*tensor.Tensor {
	// d(a*b)/da = b, d(a*b)/db = a
	gradA, _ := Mul(grad, f.b)
	gradB, _ := Mul(grad, f.a)
	return []*tensor.Tensor{gradA, gradB}
}

type matmulGradFn struct {
	a, b *tensor.Tensor
}

func (f *matmulGradFn) Name() string             { return "MatMulBackward" }
func (f *matmulGradFn) Inputs() []*tensor.Tensor  { return []*tensor.Tensor{f.a, f.b} }
func (f *matmulGradFn) Backward(grad *tensor.Tensor) []*tensor.Tensor {
	// d(A@B)/dA = grad @ B^T
	// d(A@B)/dB = A^T @ grad
	bT, _ := f.b.T()
	aT, _ := f.a.T()
	gradA, _ := MatMul(grad, bT)
	gradB, _ := MatMul(aT, grad)
	return []*tensor.Tensor{gradA, gradB}
}

type reluGradFn struct {
	input *tensor.Tensor
}

func (f *reluGradFn) Name() string             { return "ReluBackward" }
func (f *reluGradFn) Inputs() []*tensor.Tensor  { return []*tensor.Tensor{f.input} }
func (f *reluGradFn) Backward(grad *tensor.Tensor) []*tensor.Tensor {
	// d(relu)/dx = 1 if x > 0 else 0
	// Approximated: relu'(x) = x > 0, so grad * (input > 0)
	// For now, we recompute relu mask
	// TODO: implement proper mask
	return []*tensor.Tensor{grad}
}

// ---- Public API ----

func getBackend(t *tensor.Tensor) (backend.Backend, error) {
	return backend.GetForDevice(t.Device())
}

func allocOutput(shape tensor.Shape, dtype tensor.DType, device backend.Device) (backend.Storage, error) {
	bk, err := backend.GetForDevice(device)
	if err != nil {
		return nil, err
	}
	return bk.Alloc(shape.NumElements() * int(dtype.Size()))
}

func needsGrad(tensors ...*tensor.Tensor) bool {
	for _, t := range tensors {
		if t.RequiresGrad() {
			return true
		}
	}
	return false
}

// Add performs element-wise addition.
func Add(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	bk, err := getBackend(a)
	if err != nil {
		return nil, err
	}

	outShape, err := tensor.BroadcastShapes(a.Shape(), b.Shape())
	if err != nil {
		return nil, err
	}

	store, err := allocOutput(outShape, a.DType(), a.Device())
	if err != nil {
		return nil, err
	}

	if err := bk.Add(store, a.Storage(), b.Storage(), a.Shape(), b.Shape(), outShape, a.DType()); err != nil {
		return nil, err
	}

	out := tensor.NewTensor(store, outShape, a.DType())
	if needsGrad(a, b) {
		out.SetRequiresGrad(true)
		out.SetGradFn(&addGradFn{a: a, b: b})
	}
	return out, nil
}

// Mul performs element-wise multiplication.
func Mul(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	bk, err := getBackend(a)
	if err != nil {
		return nil, err
	}

	outShape, err := tensor.BroadcastShapes(a.Shape(), b.Shape())
	if err != nil {
		return nil, err
	}

	store, err := allocOutput(outShape, a.DType(), a.Device())
	if err != nil {
		return nil, err
	}

	if err := bk.Mul(store, a.Storage(), b.Storage(), a.Shape(), b.Shape(), outShape, a.DType()); err != nil {
		return nil, err
	}

	out := tensor.NewTensor(store, outShape, a.DType())
	if needsGrad(a, b) {
		out.SetRequiresGrad(true)
		out.SetGradFn(&mulGradFn{a: a, b: b})
	}
	return out, nil
}

// MatMul performs matrix multiplication.
func MatMul(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	bk, err := getBackend(a)
	if err != nil {
		return nil, err
	}

	// Ensure contiguous layout before sending to backend
	origA, origB := a, b
	if !a.IsContiguous() {
		a, err = a.Contiguous()
		if err != nil {
			return nil, fmt.Errorf("matmul: contiguous A: %w", err)
		}
	}
	if !b.IsContiguous() {
		b, err = b.Contiguous()
		if err != nil {
			return nil, fmt.Errorf("matmul: contiguous B: %w", err)
		}
	}

	shapeA := a.Shape()
	shapeB := b.Shape()
	ndimA := len(shapeA)
	ndimB := len(shapeB)

	M := shapeA[ndimA-2]
	N := shapeB[ndimB-1]

	// Output shape: batch dims + [M, N]
	outShape := make(tensor.Shape, ndimA)
	copy(outShape, shapeA[:ndimA-2])
	outShape[ndimA-2] = M
	outShape[ndimA-1] = N

	store, err := allocOutput(outShape, a.DType(), a.Device())
	if err != nil {
		return nil, err
	}

	if err := bk.MatMul(store, a.Storage(), b.Storage(), shapeA, shapeB, a.DType()); err != nil {
		return nil, err
	}

	out := tensor.NewTensor(store, outShape, a.DType())
	if needsGrad(origA, origB) {
		out.SetRequiresGrad(true)
		out.SetGradFn(&matmulGradFn{a: origA, b: origB})
	}
	return out, nil
}

// Relu applies rectified linear unit.
func Relu(t *tensor.Tensor) (*tensor.Tensor, error) {
	bk, err := getBackend(t)
	if err != nil {
		return nil, err
	}

	store, err := allocOutput(t.Shape(), t.DType(), t.Device())
	if err != nil {
		return nil, err
	}

	if err := bk.Relu(store, t.Storage(), t.Shape(), t.DType()); err != nil {
		return nil, err
	}

	out := tensor.NewTensor(store, t.Shape(), t.DType())
	if needsGrad(t) {
		out.SetRequiresGrad(true)
		out.SetGradFn(&reluGradFn{input: t})
	}
	return out, nil
}

// Softmax applies softmax along the given axis.
func Softmax(t *tensor.Tensor, axis int) (*tensor.Tensor, error) {
	bk, err := getBackend(t)
	if err != nil {
		return nil, err
	}

	store, err := allocOutput(t.Shape(), t.DType(), t.Device())
	if err != nil {
		return nil, err
	}

	if err := bk.Softmax(store, t.Storage(), t.Shape(), axis, t.DType()); err != nil {
		return nil, err
	}

	return tensor.NewTensor(store, t.Shape(), t.DType()), nil
}

// LayerNorm applies layer normalization.
func LayerNorm(x, gamma, beta *tensor.Tensor, normAxis int, eps float64) (*tensor.Tensor, error) {
	bk, err := getBackend(x)
	if err != nil {
		return nil, err
	}

	store, err := allocOutput(x.Shape(), x.DType(), x.Device())
	if err != nil {
		return nil, err
	}

	var gs, bs backend.Storage
	if gamma != nil {
		gs = gamma.Storage()
	}
	if beta != nil {
		bs = beta.Storage()
	}

	if err := bk.LayerNorm(store, x.Storage(), gs, bs, x.Shape(), normAxis, eps, x.DType()); err != nil {
		return nil, err
	}

	return tensor.NewTensor(store, x.Shape(), x.DType()), nil
}

// Gelu applies GELU activation.
func Gelu(t *tensor.Tensor) (*tensor.Tensor, error) {
	bk, err := getBackend(t)
	if err != nil {
		return nil, err
	}

	store, err := allocOutput(t.Shape(), t.DType(), t.Device())
	if err != nil {
		return nil, err
	}

	if err := bk.Gelu(store, t.Storage(), t.Shape(), t.DType()); err != nil {
		return nil, err
	}

	return tensor.NewTensor(store, t.Shape(), t.DType()), nil
}

// Silu applies SiLU (Swish) activation.
func Silu(t *tensor.Tensor) (*tensor.Tensor, error) {
	bk, err := getBackend(t)
	if err != nil {
		return nil, err
	}

	store, err := allocOutput(t.Shape(), t.DType(), t.Device())
	if err != nil {
		return nil, err
	}

	if err := bk.Silu(store, t.Storage(), t.Shape(), t.DType()); err != nil {
		return nil, err
	}

	return tensor.NewTensor(store, t.Shape(), t.DType()), nil
}

// ScaledDotProductAttention computes multi-head attention.
func ScaledDotProductAttention(q, k, v *tensor.Tensor, numHeads int, causal bool) (*tensor.Tensor, error) {
	bk, err := getBackend(q)
	if err != nil {
		return nil, err
	}

	shape := q.Shape()
	batchSize := shape[0]
	seqLen := shape[2]
	headDim := shape[3]

	store, err := allocOutput(shape, q.DType(), q.Device())
	if err != nil {
		return nil, err
	}

	if err := bk.ScaledDotProductAttention(
		store, q.Storage(), k.Storage(), v.Storage(),
		batchSize, numHeads, seqLen, headDim,
		causal, q.DType(),
	); err != nil {
		return nil, err
	}

	return tensor.NewTensor(store, shape, q.DType()), nil
}
