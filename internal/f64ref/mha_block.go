package f64ref

// MultiHeadAttentionModuleF64 — полный MHA-модуль с Q/K/V/O проекциями и RoPE.
// Соответствует goml/nn/attention.go.
//
// TransformerBlockF64 — pre-norm LLaMA style:
//   x = x + Attn(LN1(x))
//   x = x + FFN(LN2(x))

// ─────────────── MHA full module ───────────────

type MultiHeadAttentionModuleF64 struct {
	Wq       *LinearF64
	Wk       *LinearF64
	Wv       *LinearF64
	Wo       *LinearF64
	RoPE     *RoPEF64
	Attn     *MultiHeadAttentionF64
	NumHeads int
	HeadDim  int
	Dim      int
}

func NewMHAModuleF64(dim, numHeads int, Wq, Wk, Wv, Wo *F64Tensor, causal bool) *MultiHeadAttentionModuleF64 {
	if dim%numHeads != 0 {
		panic("MHA: dim must be divisible by numHeads")
	}
	headDim := dim / numHeads
	return &MultiHeadAttentionModuleF64{
		Wq:       NewLinearF64(Wq, nil),
		Wk:       NewLinearF64(Wk, nil),
		Wv:       NewLinearF64(Wv, nil),
		Wo:       NewLinearF64(Wo, nil),
		RoPE:     NewRoPEF64(headDim, 10000.0),
		Attn:     NewMultiHeadAttentionF64(causal),
		NumHeads: numHeads,
		HeadDim:  headDim,
		Dim:      dim,
	}
}

// Forward: x [batch, seqLen, dim] → out [batch, seqLen, dim] + cache.
type MHACache struct {
	x            *F64Tensor
	qProj, kProj, vProj *F64Tensor // Wq(x), Wk(x), Wv(x) — [B, S, Dim]
	qRope        *F64Tensor // после RoPE, [B, H, S, HD]
	kRope        *F64Tensor
	qBHSD        *F64Tensor // до RoPE в форме [B, H, S, HD] (для backward RoPE Q)
	kBHSD        *F64Tensor // до RoPE
	vBHSD        *F64Tensor
	attnCache    *AttentionCache
	attnOut      *F64Tensor // до Wo, форма [B, S, Dim]
}

func (m *MultiHeadAttentionModuleF64) Forward(x *F64Tensor) (*F64Tensor, *MHACache) {
	B, S := x.Shape[0], x.Shape[1]
	Dim := m.Dim
	// Проекции.
	qProj := m.Wq.Forward(x) // [B, S, Dim]
	kProj := m.Wk.Forward(x)
	vProj := m.Wv.Forward(x)

	// Reshape [B, S, Dim] → [B, S, H, HD] → transpose → [B, H, S, HD].
	qBHSD := reshape4dSHtoHS(qProj, B, S, m.NumHeads, m.HeadDim)
	kBHSD := reshape4dSHtoHS(kProj, B, S, m.NumHeads, m.HeadDim)
	vBHSD := reshape4dSHtoHS(vProj, B, S, m.NumHeads, m.HeadDim)

	// RoPE к Q и K (не V).
	qRope := m.RoPE.Forward(qBHSD)
	kRope := m.RoPE.Forward(kBHSD)

	// Attention SDPA.
	attnOut, attnCache := m.Attn.Forward(qRope, kRope, vBHSD) // [B, H, S, HD]

	// Transpose обратно [B, H, S, HD] → [B, S, H, HD] → reshape [B, S, Dim].
	attnBSD := reshape4dHStoSH(attnOut, B, m.NumHeads, S, m.HeadDim)
	attnBSD.Shape = []int{B, S, Dim}

	// Output projection.
	out := m.Wo.Forward(attnBSD)
	return out, &MHACache{
		x: x.Clone(), qProj: qProj, kProj: kProj, vProj: vProj,
		qRope: qRope, kRope: kRope,
		qBHSD: qBHSD, kBHSD: kBHSD, vBHSD: vBHSD,
		attnCache: attnCache, attnOut: attnBSD,
	}
}

func (m *MultiHeadAttentionModuleF64) Backward(dOut *F64Tensor, cache *MHACache) (dX, dWq, dWk, dWv, dWo *F64Tensor) {
	B, S := cache.x.Shape[0], cache.x.Shape[1]
	Dim := m.Dim
	// output projection backward.
	dAttnBSD, dWoRet, _ := m.Wo.Backward(cache.attnOut, dOut)
	dWo = dWoRet

	// dAttnBSD [B, S, Dim] → [B, H, S, HD] через reshape+transpose.
	dAttnBHSD := reshape4dSHtoHS(&F64Tensor{Data: dAttnBSD.Data, Shape: []int{B, S, Dim}},
		B, S, m.NumHeads, m.HeadDim)

	// attention backward → dQrope, dKrope, dV.
	dQrope, dKrope, dV := m.Attn.Backward(dAttnBHSD, cache.attnCache)

	// RoPE backward.
	dQbhsd := m.RoPE.Backward(dQrope)
	dKbhsd := m.RoPE.Backward(dKrope)

	// dQbhsd [B, H, S, HD] → [B, S, Dim] (reverse transpose+reshape).
	dQproj := reshape4dHStoSH(dQbhsd, B, m.NumHeads, S, m.HeadDim)
	dQproj.Shape = []int{B, S, Dim}
	dKproj := reshape4dHStoSH(dKbhsd, B, m.NumHeads, S, m.HeadDim)
	dKproj.Shape = []int{B, S, Dim}
	dVproj := reshape4dHStoSH(dV, B, m.NumHeads, S, m.HeadDim)
	dVproj.Shape = []int{B, S, Dim}

	// Wq/Wk/Wv backward → dX_from_each + dW*.
	dxQ, dWqRet, _ := m.Wq.Backward(cache.x, dQproj)
	dWq = dWqRet
	dxK, dWkRet, _ := m.Wk.Backward(cache.x, dKproj)
	dWk = dWkRet
	dxV, dWvRet, _ := m.Wv.Backward(cache.x, dVproj)
	dWv = dWvRet

	// dX = сумма dxQ + dxK + dxV.
	dX = Zeros(cache.x.Shape...)
	for i := range dX.Data {
		dX.Data[i] = dxQ.Data[i] + dxK.Data[i] + dxV.Data[i]
	}
	return
}

// reshape4dSHtoHS: [B, S, H, HD] → [B, H, S, HD] (transpose 1↔2).
// Вход хранится в стиле «by S then H»: byte(b*S*H*HD + s*H*HD + h*HD + d).
// Выход в стиле «by H then S»: byte(b*H*S*HD + h*S*HD + s*HD + d).
func reshape4dSHtoHS(x *F64Tensor, B, S, H, HD int) *F64Tensor {
	out := Zeros(B, H, S, HD)
	for b := 0; b < B; b++ {
		for s := 0; s < S; s++ {
			for h := 0; h < H; h++ {
				srcOff := ((b*S+s)*H + h) * HD
				dstOff := ((b*H+h)*S + s) * HD
				copy(out.Data[dstOff:dstOff+HD], x.Data[srcOff:srcOff+HD])
			}
		}
	}
	return out
}

// reshape4dHStoSH: обратное преобразование.
func reshape4dHStoSH(x *F64Tensor, B, H, S, HD int) *F64Tensor {
	out := Zeros(B, S, H, HD)
	for b := 0; b < B; b++ {
		for h := 0; h < H; h++ {
			for s := 0; s < S; s++ {
				srcOff := ((b*H+h)*S + s) * HD
				dstOff := ((b*S+s)*H + h) * HD
				copy(out.Data[dstOff:dstOff+HD], x.Data[srcOff:srcOff+HD])
			}
		}
	}
	return out
}

// ─────────────── TransformerBlockF64 ───────────────

type TransformerBlockF64 struct {
	AttnNorm *LayerNormF64
	Attn     *MultiHeadAttentionModuleF64
	FFNNorm  *LayerNormF64
	FFN      *FeedForwardF64
}

func NewTransformerBlockF64(dim, numHeads, hiddenDim int,
	attnGamma, attnBeta *F64Tensor,
	wq, wk, wv, wo *F64Tensor,
	ffnGamma, ffnBeta *F64Tensor,
	w1, w2, w3 *F64Tensor, causal bool) *TransformerBlockF64 {
	ln1 := &LayerNormF64{Gamma: attnGamma, Beta: attnBeta, Eps: 1e-5}
	ln2 := &LayerNormF64{Gamma: ffnGamma, Beta: ffnBeta, Eps: 1e-5}
	return &TransformerBlockF64{
		AttnNorm: ln1,
		Attn:     NewMHAModuleF64(dim, numHeads, wq, wk, wv, wo, causal),
		FFNNorm:  ln2,
		FFN:      NewFeedForwardF64(w1, w2, w3),
	}
}

type BlockCache struct {
	x         *F64Tensor // block input
	normed1   *F64Tensor // AttnNorm(x)
	lnCache1  *LayerNormCache
	attnOut   *F64Tensor
	mhaCache  *MHACache
	xAfter1   *F64Tensor // x + attnOut
	normed2   *F64Tensor
	lnCache2  *LayerNormCache
	ffnOut    *F64Tensor
	ffnCache  *FFNCache
}

func (b *TransformerBlockF64) Forward(x *F64Tensor) (*F64Tensor, *BlockCache) {
	normed1, ln1Cache := b.AttnNorm.Forward(x)
	attnOut, mhaCache := b.Attn.Forward(normed1)
	xAfter1 := Zeros(x.Shape...)
	for i := range xAfter1.Data {
		xAfter1.Data[i] = x.Data[i] + attnOut.Data[i]
	}
	normed2, ln2Cache := b.FFNNorm.Forward(xAfter1)
	ffnOut, ffnCache := b.FFN.Forward(normed2)
	out := Zeros(x.Shape...)
	for i := range out.Data {
		out.Data[i] = xAfter1.Data[i] + ffnOut.Data[i]
	}
	return out, &BlockCache{
		x: x.Clone(), normed1: normed1, lnCache1: ln1Cache,
		attnOut: attnOut, mhaCache: mhaCache, xAfter1: xAfter1,
		normed2: normed2, lnCache2: ln2Cache,
		ffnOut: ffnOut, ffnCache: ffnCache,
	}
}

func (b *TransformerBlockF64) Backward(dOut *F64Tensor, cache *BlockCache) (dX *F64Tensor) {
	// residual 2: dxAfter1_from_res = dOut; dffnOut = dOut
	dxAfter1FromRes := dOut
	dffnOut := dOut
	// FFN backward.
	dNormed2, dW1, dW2, dW3 := b.FFN.Backward(dffnOut, cache.ffnCache)
	_ = dW1
	_ = dW2
	_ = dW3 // сохраним в структурах если нужно — здесь для теста dX
	// LN2 backward.
	dxAfter1FromNorm, dGamma2, dBeta2 := b.FFNNorm.Backward(dNormed2, cache.lnCache2)
	_ = dGamma2
	_ = dBeta2

	// dxAfter1 = dxAfter1FromRes + dxAfter1FromNorm.
	dxAfter1 := Zeros(cache.x.Shape...)
	for i := range dxAfter1.Data {
		dxAfter1.Data[i] = dxAfter1FromRes.Data[i] + dxAfter1FromNorm.Data[i]
	}
	// residual 1: dx_from_res = dxAfter1; dattnOut = dxAfter1.
	dxFromRes := dxAfter1
	dattnOut := dxAfter1
	// Attention module backward.
	dNormed1, dWq, dWk, dWv, dWo := b.Attn.Backward(dattnOut, cache.mhaCache)
	_ = dWq
	_ = dWk
	_ = dWv
	_ = dWo
	// LN1 backward.
	dxFromNorm, dGamma1, dBeta1 := b.AttnNorm.Backward(dNormed1, cache.lnCache1)
	_ = dGamma1
	_ = dBeta1

	// dX = dxFromRes + dxFromNorm.
	dX = Zeros(cache.x.Shape...)
	for i := range dX.Data {
		dX.Data[i] = dxFromRes.Data[i] + dxFromNorm.Data[i]
	}
	return
}
