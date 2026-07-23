package f64ref

// LayerNormF64 — эталонная LayerNorm в CPU-FP64.
//
// y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
//
// Нормализация по последней оси (стандарт для трансформера). Всё считается
// на host — операция дешёвая (per-row loop), точность важнее скорости.
//
// gamma, beta: shape [D], где D = shape[-1] (последняя ось x).

import "math"

type LayerNormF64 struct {
	Gamma *F64Tensor // [D]
	Beta  *F64Tensor // [D]
	Eps   float64
}

func NewLayerNormF64(D int, eps float64) *LayerNormF64 {
	g := Zeros(D)
	b := Zeros(D)
	for i := range g.Data {
		g.Data[i] = 1.0
	}
	return &LayerNormF64{Gamma: g, Beta: b, Eps: eps}
}

// Forward — CPU per-row LayerNorm.
// x: [..., D] → y: [..., D]. Возвращает также cache для backward.
type LayerNormCache struct {
	x          *F64Tensor // input (сохранён для backward)
	mean       []float64  // per-row mean, len = rows
	invStd     []float64  // per-row 1/sqrt(var+eps)
	normalized []float64  // (x - mean) * invStd flat [rows*D]
}

func (ln *LayerNormF64) Forward(x *F64Tensor) (*F64Tensor, *LayerNormCache) {
	D := x.Shape[len(x.Shape)-1]
	if D != ln.Gamma.Shape[0] {
		panic("LayerNormF64: D mismatch with Gamma")
	}
	rows := x.NumElements() / D
	out := Zeros(x.Shape...)
	cache := &LayerNormCache{
		x:          x.Clone(),
		mean:       make([]float64, rows),
		invStd:     make([]float64, rows),
		normalized: make([]float64, rows*D),
	}
	for r := 0; r < rows; r++ {
		// mean.
		var sum float64
		for i := 0; i < D; i++ {
			sum += x.Data[r*D+i]
		}
		mean := sum / float64(D)
		cache.mean[r] = mean
		// var (E[(x-mean)^2] = sum((x-mean)^2) / D).
		var vs float64
		for i := 0; i < D; i++ {
			d := x.Data[r*D+i] - mean
			vs += d * d
		}
		varv := vs / float64(D)
		invStd := 1.0 / math.Sqrt(varv+ln.Eps)
		cache.invStd[r] = invStd
		// normalize + affine.
		for i := 0; i < D; i++ {
			n := (x.Data[r*D+i] - mean) * invStd
			cache.normalized[r*D+i] = n
			out.Data[r*D+i] = n*ln.Gamma.Data[i] + ln.Beta.Data[i]
		}
	}
	return out, cache
}

// Backward LayerNorm.
// dy: [..., D] (gradient wrt output y).
// Возвращает dx, dGamma, dBeta.
//
// Формулы (Kingma-Kiros style, per-row):
//   dGamma[i] = sum_r dy[r,i] * normalized[r,i]
//   dBeta[i]  = sum_r dy[r,i]
//   dnorm[r,i] = dy[r,i] * gamma[i]
//   dvar[r]  = -0.5 * invStd^3 * sum_i (x[r,i] - mean[r]) * dnorm[r,i]
//   dmean[r] = -invStd * sum_i dnorm[r,i]  +  dvar[r] * -2/D * sum_i (x[r,i]-mean[r])
//              (второй член = 0 для центрированной x, но оставляю для честности)
//   dx[r,i]  = dnorm[r,i] * invStd  +  dvar[r] * 2*(x[r,i]-mean[r])/D  +  dmean[r]/D
func (ln *LayerNormF64) Backward(dy *F64Tensor, cache *LayerNormCache) (dx, dGamma, dBeta *F64Tensor) {
	D := dy.Shape[len(dy.Shape)-1]
	rows := dy.NumElements() / D
	x := cache.x
	dx = Zeros(x.Shape...)
	dGamma = Zeros(D)
	dBeta = Zeros(D)

	for r := 0; r < rows; r++ {
		mean := cache.mean[r]
		invStd := cache.invStd[r]
		// dGamma, dBeta accumulate over batch.
		for i := 0; i < D; i++ {
			dGamma.Data[i] += dy.Data[r*D+i] * cache.normalized[r*D+i]
			dBeta.Data[i] += dy.Data[r*D+i]
		}
		// dnorm.
		dnorm := make([]float64, D)
		for i := 0; i < D; i++ {
			dnorm[i] = dy.Data[r*D+i] * ln.Gamma.Data[i]
		}
		// dvar.
		var dvar float64
		for i := 0; i < D; i++ {
			dvar += (x.Data[r*D+i] - mean) * dnorm[i]
		}
		dvar *= -0.5 * invStd * invStd * invStd
		// dmean.
		var sumDnorm, sumXminusMean float64
		for i := 0; i < D; i++ {
			sumDnorm += dnorm[i]
			sumXminusMean += x.Data[r*D+i] - mean
		}
		dmean := -invStd*sumDnorm + dvar*(-2.0/float64(D))*sumXminusMean
		// dx.
		for i := 0; i < D; i++ {
			dx.Data[r*D+i] = dnorm[i]*invStd + dvar*2.0*(x.Data[r*D+i]-mean)/float64(D) + dmean/float64(D)
		}
	}
	return
}
