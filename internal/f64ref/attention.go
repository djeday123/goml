package f64ref

// MultiHeadAttentionF64 — эталонная SDPA с causal mask в F64.
//
// Форма:
//   Q, K, V: [batch, numHeads, seqLen, headDim]
//   scale  = 1 / sqrt(headDim)
//   S      = Q @ K^T * scale                   [batch, numHeads, seqLen, seqLen]
//   S_mask = S + causal_mask (−∞ в верхнем треугольнике если causal)
//   P      = softmax(S_mask, axis=-1)          [batch, numHeads, seqLen, seqLen]
//   O      = P @ V                             [batch, numHeads, seqLen, headDim]
//
// MatMul через gotorch F64 (loop по [batch, numHeads] пар).
// Softmax через host FP64 (простой, точный, без TF32-примесей).
// Causal mask — host loop.
//
// Backward:
//   dV[b,h] = P[b,h]^T @ dO[b,h]
//   dP[b,h] = dO[b,h] @ V[b,h]^T
//   dS[b,h,i,:] = P[b,h,i,:] * (dP[b,h,i,:] - sum(dP[b,h,i,:] * P[b,h,i,:]))  (softmax jacobian)
//   dS_scaled = dS * scale
//   dQ[b,h] = dS_scaled @ K[b,h]
//   dK[b,h] = dS_scaled^T @ Q[b,h]

import "math"

type MultiHeadAttentionF64 struct {
	Causal bool
}

func NewMultiHeadAttentionF64(causal bool) *MultiHeadAttentionF64 {
	return &MultiHeadAttentionF64{Causal: causal}
}

// Forward + cache для backward.
type AttentionCache struct {
	Q, K, V *F64Tensor // input копии
	P       *F64Tensor // softmax outputs [B,H,S,S]
	Scale   float64
}

func (a *MultiHeadAttentionF64) Forward(Q, K, V *F64Tensor) (*F64Tensor, *AttentionCache) {
	if len(Q.Shape) != 4 {
		panic("AttentionF64: Q must be 4D [B, H, S, D]")
	}
	B, H, S, D := Q.Shape[0], Q.Shape[1], Q.Shape[2], Q.Shape[3]
	scale := 1.0 / math.Sqrt(float64(D))
	// scores S [B,H,S,S]
	scores := Zeros(B, H, S, S)
	for b := 0; b < B; b++ {
		for h := 0; h < H; h++ {
			qBH := sliceBH(Q, b, h, S, D)          // [S, D]
			kBH := sliceBH(K, b, h, S, D)          // [S, D]
			kT := transpose2D(kBH)                  // [D, S]
			s := MatMulF64GPU(qBH, kT, S, D, S)    // [S, S]
			// scale + causal mask.
			for i := 0; i < S; i++ {
				for j := 0; j < S; j++ {
					v := s.Data[i*S+j] * scale
					if a.Causal && j > i {
						v = math.Inf(-1)
					}
					s.Data[i*S+j] = v
				}
			}
			// copy to scores[b,h,:,:]
			offset := ((b*H)+h)*S*S
			copy(scores.Data[offset:offset+S*S], s.Data)
		}
	}
	// softmax по последней оси.
	P := softmaxLastAxisF64(scores)
	// out = P @ V per [b, h].
	out := Zeros(B, H, S, D)
	for b := 0; b < B; b++ {
		for h := 0; h < H; h++ {
			pBH := sliceBH(P, b, h, S, S)
			vBH := sliceBH(V, b, h, S, D)
			o := MatMulF64GPU(pBH, vBH, S, S, D)
			offset := ((b*H)+h)*S*D
			copy(out.Data[offset:offset+S*D], o.Data)
		}
	}
	return out, &AttentionCache{
		Q: Q.Clone(), K: K.Clone(), V: V.Clone(),
		P: P, Scale: scale,
	}
}

func (a *MultiHeadAttentionF64) Backward(dO *F64Tensor, cache *AttentionCache) (dQ, dK, dV *F64Tensor) {
	B, H, S, D := cache.Q.Shape[0], cache.Q.Shape[1], cache.Q.Shape[2], cache.Q.Shape[3]
	dQ = Zeros(B, H, S, D)
	dK = Zeros(B, H, S, D)
	dV = Zeros(B, H, S, D)
	for b := 0; b < B; b++ {
		for h := 0; h < H; h++ {
			doBH := sliceBH(dO, b, h, S, D)
			pBH := sliceBH(cache.P, b, h, S, S)
			qBH := sliceBH(cache.Q, b, h, S, D)
			kBH := sliceBH(cache.K, b, h, S, D)
			vBH := sliceBH(cache.V, b, h, S, D)

			// dV = P^T @ dO
			pT := transpose2D(pBH)
			dvBH := MatMulF64GPU(pT, doBH, S, S, D)
			writeBH(dV, dvBH, b, h, S, D)

			// dP = dO @ V^T
			vT := transpose2D(vBH)
			dpBH := MatMulF64GPU(doBH, vT, S, D, S)

			// dS[i,j] = P[i,j] * (dP[i,j] - sum_k dP[i,k] * P[i,k])
			dsBH := Zeros(S, S)
			for i := 0; i < S; i++ {
				var sumProd float64
				for k := 0; k < S; k++ {
					sumProd += dpBH.Data[i*S+k] * pBH.Data[i*S+k]
				}
				for j := 0; j < S; j++ {
					dsBH.Data[i*S+j] = pBH.Data[i*S+j] * (dpBH.Data[i*S+j] - sumProd)
				}
			}
			// scale.
			for i := range dsBH.Data {
				dsBH.Data[i] *= cache.Scale
			}
			// causal mask нулит dS в верхнем треугольнике (софтмакс дал 0 там,
			// backward сохраняет 0). Уже автоматически через P=0 * (…) = 0.

			// dQ = dS @ K
			dqBH := MatMulF64GPU(dsBH, kBH, S, S, D)
			writeBH(dQ, dqBH, b, h, S, D)

			// dK = dS^T @ Q
			dsT := transpose2D(dsBH)
			dkBH := MatMulF64GPU(dsT, qBH, S, S, D)
			writeBH(dK, dkBH, b, h, S, D)
		}
	}
	return
}

// sliceBH выделяет [b,h,:,:] slice из тензора shape [B,H,S1,S2] в 2D [S1,S2].
func sliceBH(t *F64Tensor, b, h, s1, s2 int) *F64Tensor {
	H := t.Shape[1]
	offset := ((b*H)+h)*s1*s2
	data := make([]float64, s1*s2)
	copy(data, t.Data[offset:offset+s1*s2])
	return &F64Tensor{Data: data, Shape: []int{s1, s2}}
}

// writeBH пишет 2D slice в [b,h,:,:] позицию 4D тензора.
func writeBH(dst *F64Tensor, src *F64Tensor, b, h, s1, s2 int) {
	H := dst.Shape[1]
	offset := ((b*H)+h)*s1*s2
	copy(dst.Data[offset:offset+s1*s2], src.Data)
}

// softmaxLastAxisF64 — численно-стабильный softmax по последней оси, host FP64.
func softmaxLastAxisF64(x *F64Tensor) *F64Tensor {
	D := x.Shape[len(x.Shape)-1]
	rows := x.NumElements() / D
	out := Zeros(x.Shape...)
	for r := 0; r < rows; r++ {
		// max для стабильности.
		mx := math.Inf(-1)
		for i := 0; i < D; i++ {
			v := x.Data[r*D+i]
			if v > mx {
				mx = v
			}
		}
		if math.IsInf(mx, -1) {
			// вся строка -inf (все masked) — оставляем 0.
			continue
		}
		// exp + sum.
		var sum float64
		for i := 0; i < D; i++ {
			e := math.Exp(x.Data[r*D+i] - mx)
			out.Data[r*D+i] = e
			sum += e
		}
		// normalize.
		inv := 1.0 / sum
		for i := 0; i < D; i++ {
			out.Data[r*D+i] *= inv
		}
	}
	return out
}
