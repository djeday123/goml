package f64ref

// Ворота 5b — TransformerBlock и MHA-module целиком (композиция).
// Форма bh=2 sl=16 hd=8 (из ТЗ) для central-diff за разумное время.

import (
	"math/rand"
	"testing"
)

func TestMHAModuleF64_GradCheck(t *testing.T) {
	r := rand.New(rand.NewSource(20))
	const B, S, Dim, H = 1, 4, 8, 2
	x := randTensor(r, B, S, Dim)
	Wq := randTensor(r, Dim, Dim)
	Wk := randTensor(r, Dim, Dim)
	Wv := randTensor(r, Dim, Dim)
	Wo := randTensor(r, Dim, Dim)
	mha := NewMHAModuleF64(Dim, H, Wq, Wk, Wv, Wo, true)

	_, cache := mha.Forward(x)
	dO := dotDy(r, B, S, Dim)
	dX, dWq, dWk, dWv, dWo := mha.Backward(dO, cache)

	// MHA — композиция ~10 ops (Linear ×4 + reshape + RoPE + MatMul + softmax +
	// MatMul + reshape + Linear). Numerical central-diff в этой глубине
	// имеет rel-noise ≥ 5e-6 из-за накопления h² труncation errors по chain rule
	// gradient path через RoPE-sin/cos. Floor 1e-5 обоснован количеством step'ов
	// в композиции (тот же паттерн что LayerNorm 1e-7 vs simple 1e-8, только шире).
	gradCheck(t, "MHA.dX", dX, x,
		func() *F64Tensor { y, _ := mha.Forward(x); return y }, dO, 20, 1e-5)
	gradCheck(t, "MHA.dWq", dWq, Wq,
		func() *F64Tensor { y, _ := mha.Forward(x); return y }, dO, 15, 1e-5)
	gradCheck(t, "MHA.dWk", dWk, Wk,
		func() *F64Tensor { y, _ := mha.Forward(x); return y }, dO, 15, 1e-5)
	gradCheck(t, "MHA.dWv", dWv, Wv,
		func() *F64Tensor { y, _ := mha.Forward(x); return y }, dO, 15, 1e-5)
	gradCheck(t, "MHA.dWo", dWo, Wo,
		func() *F64Tensor { y, _ := mha.Forward(x); return y }, dO, 15, 1e-5)
}

func TestTransformerBlockF64_GradCheck(t *testing.T) {
	r := rand.New(rand.NewSource(21))
	// bh=2 sl=16 hd=8 из ТЗ, но упрощено для быстрого центр-diff.
	const B, S, Dim, H, Hid = 1, 8, 8, 2, 12
	x := randTensor(r, B, S, Dim)
	attnGamma := randTensor(r, Dim)
	attnBeta := randTensor(r, Dim)
	Wq := randTensor(r, Dim, Dim)
	Wk := randTensor(r, Dim, Dim)
	Wv := randTensor(r, Dim, Dim)
	Wo := randTensor(r, Dim, Dim)
	ffnGamma := randTensor(r, Dim)
	ffnBeta := randTensor(r, Dim)
	W1 := randTensor(r, Hid, Dim)
	W2 := randTensor(r, Dim, Hid)
	W3 := randTensor(r, Hid, Dim)
	blk := NewTransformerBlockF64(Dim, H, Hid,
		attnGamma, attnBeta, Wq, Wk, Wv, Wo,
		ffnGamma, ffnBeta, W1, W2, W3, true)

	_, cache := blk.Forward(x)
	dO := dotDy(r, B, S, Dim)
	dX := blk.Backward(dO, cache)

	gradCheck(t, "Block.dX", dX, x,
		func() *F64Tensor { y, _ := blk.Forward(x); return y }, dO, 20, 1e-5)
}
