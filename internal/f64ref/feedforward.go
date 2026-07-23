package f64ref

// FeedForwardF64 — эталонный FFN блок в F64.
//
// SwiGLU (goml TinyConfig UseSwiGLU=true): y = W2 @ (SiLU(W1 @ x) ⊙ (W3 @ x))
// где ⊙ — hadamard product; SiLU(z) = z * sigmoid(z).
//
// Параметры:
//   W1: [hiddenDim, dim]  — gate projection
//   W3: [hiddenDim, dim]  — up projection
//   W2: [dim, hiddenDim]  — down projection
//
// Backward — стандартная chain rule.

import "math"

type FeedForwardF64 struct {
	W1  *F64Tensor // gate
	W3  *F64Tensor // up
	W2  *F64Tensor // down
	Dim int
	Hid int
}

func NewFeedForwardF64(W1, W2, W3 *F64Tensor) *FeedForwardF64 {
	return &FeedForwardF64{
		W1: W1, W2: W2, W3: W3,
		Dim: W1.Shape[1], Hid: W1.Shape[0],
	}
}

type FFNCache struct {
	x        *F64Tensor // input
	gate     *F64Tensor // W1 @ x = pre-SiLU
	silu     *F64Tensor // SiLU(gate)
	up       *F64Tensor // W3 @ x
	hidden   *F64Tensor // silu * up
	sigmoid  *F64Tensor // sigmoid(gate), для быстрого backward
}

func (ff *FeedForwardF64) Forward(x *F64Tensor) (*F64Tensor, *FFNCache) {
	// linears x [..., dim] → [..., hid]
	gateL := NewLinearF64(ff.W1, nil)
	upL := NewLinearF64(ff.W3, nil)
	gate := gateL.Forward(x)
	up := upL.Forward(x)

	// SiLU(gate) + hidden = SiLU(gate) * up.
	silu := Zeros(gate.Shape...)
	sig := Zeros(gate.Shape...)
	for i := range gate.Data {
		s := 1.0 / (1.0 + math.Exp(-gate.Data[i]))
		sig.Data[i] = s
		silu.Data[i] = gate.Data[i] * s
	}
	hidden := Zeros(gate.Shape...)
	for i := range hidden.Data {
		hidden.Data[i] = silu.Data[i] * up.Data[i]
	}
	// W2 down: [..., hid] → [..., dim].
	downL := NewLinearF64(ff.W2, nil)
	out := downL.Forward(hidden)
	return out, &FFNCache{
		x: x.Clone(), gate: gate, silu: silu, up: up, hidden: hidden, sigmoid: sig,
	}
}

func (ff *FeedForwardF64) Backward(dOut *F64Tensor, cache *FFNCache) (dX, dW1, dW2, dW3 *F64Tensor) {
	// downL backward:
	downL := NewLinearF64(ff.W2, nil)
	dHidden, dW2Ret, _ := downL.Backward(cache.hidden, dOut)
	dW2 = dW2Ret

	// hidden = silu * up. d(silu)/d(hidden_i) = up_i; d(up)/d(hidden_i) = silu_i.
	dSilu := Zeros(cache.silu.Shape...)
	dUp := Zeros(cache.up.Shape...)
	for i := range dHidden.Data {
		dSilu.Data[i] = dHidden.Data[i] * cache.up.Data[i]
		dUp.Data[i] = dHidden.Data[i] * cache.silu.Data[i]
	}
	// SiLU(z) = z * sig(z); d/dz = sig + z * sig * (1 - sig).
	dGate := Zeros(cache.gate.Shape...)
	for i := range dGate.Data {
		s := cache.sigmoid.Data[i]
		dGate.Data[i] = dSilu.Data[i] * (s + cache.gate.Data[i]*s*(1.0-s))
	}
	// gateL backward: dGate → dX_from_gate, dW1.
	gateL := NewLinearF64(ff.W1, nil)
	dxGate, dW1Ret, _ := gateL.Backward(cache.x, dGate)
	dW1 = dW1Ret

	// upL backward: dUp → dX_from_up, dW3.
	upL := NewLinearF64(ff.W3, nil)
	dxUp, dW3Ret, _ := upL.Backward(cache.x, dUp)
	dW3 = dW3Ret

	// dX = dxGate + dxUp.
	dX = Zeros(cache.x.Shape...)
	for i := range dX.Data {
		dX.Data[i] = dxGate.Data[i] + dxUp.Data[i]
	}
	return
}
