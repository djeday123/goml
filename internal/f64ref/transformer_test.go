package f64ref

// Ворота 5b — numerical grad для RoPE/Attention/FFN + transformer block.
// Особое внимание attention: dQ/dK/dV покомпонентно (не только сумма — классика ошибок из gotorch аудита).

import (
	"math/rand"
	"testing"
)

// ================================================================
// RoPEF64
// ================================================================
func TestRoPEF64_GradCheck(t *testing.T) {
	r := rand.New(rand.NewSource(10))
	const B, H, S, D = 2, 2, 4, 8
	x := randTensor(r, B, H, S, D)
	rope := NewRoPEF64(D, 10000.0)
	dY := dotDy(r, B, H, S, D)
	dX := rope.Backward(dY)
	gradCheck(t, "RoPEF64.dX",
		dX, x,
		func() *F64Tensor { return rope.Forward(x) },
		dY, 20, 1e-7)
}

// ================================================================
// AttentionF64 — dQ, dK, dV покомпонентно
// ================================================================
func TestAttentionF64_GradCheck(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	const B, H, S, D = 1, 2, 4, 4 // маленькие формы для central-diff sweep
	Q := randTensor(r, B, H, S, D)
	K := randTensor(r, B, H, S, D)
	V := randTensor(r, B, H, S, D)
	attn := NewMultiHeadAttentionF64(true) // causal

	out, cache := attn.Forward(Q, K, V)
	dO := dotDy(r, B, H, S, D)
	dQ, dK, dV := attn.Backward(dO, cache)

	// Ключевая проверка: dQ, dK, dV раздельно (классика ошибок — одинаковый grad во все три).
	_ = out

	gradCheck(t, "AttentionF64.dQ",
		dQ, Q,
		func() *F64Tensor {
			y, _ := attn.Forward(Q, K, V)
			return y
		},
		dO, 20, 1e-6)

	gradCheck(t, "AttentionF64.dK",
		dK, K,
		func() *F64Tensor {
			y, _ := attn.Forward(Q, K, V)
			return y
		},
		dO, 20, 1e-6)

	gradCheck(t, "AttentionF64.dV",
		dV, V,
		func() *F64Tensor {
			y, _ := attn.Forward(Q, K, V)
			return y
		},
		dO, 20, 1e-6)
}

// ================================================================
// FeedForwardF64 SwiGLU
// ================================================================
func TestFeedForwardF64_GradCheck(t *testing.T) {
	r := rand.New(rand.NewSource(12))
	const dim, hid = 4, 6
	const M = 3
	x := randTensor(r, M, dim)
	W1 := randTensor(r, hid, dim)
	W2 := randTensor(r, dim, hid)
	W3 := randTensor(r, hid, dim)
	ff := NewFeedForwardF64(W1, W2, W3)

	dO := dotDy(r, M, dim)
	_, cache := ff.Forward(x)
	dX, dW1, dW2, dW3 := ff.Backward(dO, cache)

	gradCheck(t, "FeedForwardF64.dX",
		dX, x,
		func() *F64Tensor {
			y, _ := ff.Forward(x)
			return y
		},
		dO, 15, 1e-6)

	gradCheck(t, "FeedForwardF64.dW1",
		dW1, W1,
		func() *F64Tensor {
			y, _ := ff.Forward(x)
			return y
		},
		dO, 15, 1e-6)

	gradCheck(t, "FeedForwardF64.dW2",
		dW2, W2,
		func() *F64Tensor {
			y, _ := ff.Forward(x)
			return y
		},
		dO, 15, 1e-6)

	gradCheck(t, "FeedForwardF64.dW3",
		dW3, W3,
		func() *F64Tensor {
			y, _ := ff.Forward(x)
			return y
		},
		dO, 15, 1e-6)
}
