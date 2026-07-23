package f64ref

// RoPEF64 — Rotary Positional Embedding в CPU-FP64.
//
// Для позиции p и headDim пары (2i, 2i+1) вращаем на угол θ_p_i = p * base^(-2i/headDim):
//   x'[2i]   = x[2i]   * cos(θ) - x[2i+1] * sin(θ)
//   x'[2i+1] = x[2i]   * sin(θ) + x[2i+1] * cos(θ)
//
// Backward: dY тоже rotation, но с обратным углом (rotation matrix orthogonal):
//   dx[2i]   = dy[2i]   * cos(θ) + dy[2i+1] * sin(θ)
//   dx[2i+1] = -dy[2i]  * sin(θ) + dy[2i+1] * cos(θ)
//
// Форма x: [batch, numHeads, seqLen, headDim] flat as [batch*numHeads*seqLen, headDim].

import "math"

// applyRoPEF64 применяет вращение in-place на копию x.
// x.Shape = [batch, numHeads, seqLen, headDim] или [total, headDim] flat.
// pos извлекается из индекса по последней перед headDim размерности.
type RoPEF64 struct {
	HeadDim int
	Base    float64 // обычно 10000
}

func NewRoPEF64(headDim int, base float64) *RoPEF64 {
	if headDim%2 != 0 {
		panic("RoPEF64: headDim must be even")
	}
	return &RoPEF64{HeadDim: headDim, Base: base}
}

// Forward: x.Shape = [batch, numHeads, seqLen, headDim]. Позиция берётся из индекса seqLen.
func (r *RoPEF64) Forward(x *F64Tensor) *F64Tensor {
	if len(x.Shape) != 4 {
		panic("RoPEF64.Forward: expected 4D [batch, numHeads, seqLen, headDim]")
	}
	batch, numHeads, seqLen, headDim := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	if headDim != r.HeadDim {
		panic("RoPEF64: headDim mismatch")
	}
	out := Zeros(x.Shape...)
	for b := 0; b < batch; b++ {
		for h := 0; h < numHeads; h++ {
			for s := 0; s < seqLen; s++ {
				base := ((b*numHeads+h)*seqLen + s) * headDim
				for i := 0; i < headDim/2; i++ {
					theta := float64(s) * math.Pow(r.Base, -2.0*float64(i)/float64(headDim))
					cos, sin := math.Cos(theta), math.Sin(theta)
					x0 := x.Data[base+2*i]
					x1 := x.Data[base+2*i+1]
					out.Data[base+2*i] = x0*cos - x1*sin
					out.Data[base+2*i+1] = x0*sin + x1*cos
				}
			}
		}
	}
	return out
}

// Backward: dx = R^T @ dy = обратное вращение (сos, -sin) в матрице rotation.
func (r *RoPEF64) Backward(dy *F64Tensor) *F64Tensor {
	batch, numHeads, seqLen, headDim := dy.Shape[0], dy.Shape[1], dy.Shape[2], dy.Shape[3]
	dx := Zeros(dy.Shape...)
	for b := 0; b < batch; b++ {
		for h := 0; h < numHeads; h++ {
			for s := 0; s < seqLen; s++ {
				base := ((b*numHeads+h)*seqLen + s) * headDim
				for i := 0; i < headDim/2; i++ {
					theta := float64(s) * math.Pow(r.Base, -2.0*float64(i)/float64(headDim))
					cos, sin := math.Cos(theta), math.Sin(theta)
					dy0 := dy.Data[base+2*i]
					dy1 := dy.Data[base+2*i+1]
					dx.Data[base+2*i] = dy0*cos + dy1*sin
					dx.Data[base+2*i+1] = -dy0*sin + dy1*cos
				}
			}
		}
	}
	return dx
}
