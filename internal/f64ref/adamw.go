package f64ref

// AdamWF64 — эталонный AdamW optimizer на host FP64.
//
// Формула AdamW (Loshchilov & Hutter, 2019):
//   t = step
//   g = grad(w)
//   m = beta1 * m + (1 - beta1) * g
//   v = beta2 * v + (1 - beta2) * g^2
//   m_hat = m / (1 - beta1^t)
//   v_hat = v / (1 - beta2^t)
//   w = w - lr * (m_hat / (sqrt(v_hat) + eps) + weightDecay * w)
//
// «Decoupled» weight decay — применяется отдельно, не через L2 в grad.

import "math"

type AdamWF64 struct {
	Params        []*F64Tensor // reference-slice параметров (тензоры in-place обновляются)
	Grads         []*F64Tensor // соответствующие градиенты (для каждого шага сбрасываются caller'ом)
	LR            float64
	Beta1, Beta2  float64
	Eps           float64
	WeightDecay   float64
	m, v          [][]float64 // moments per-parameter
	step          int
}

func NewAdamWF64(params []*F64Tensor, lr float64) *AdamWF64 {
	m := make([][]float64, len(params))
	v := make([][]float64, len(params))
	grads := make([]*F64Tensor, len(params))
	for i, p := range params {
		m[i] = make([]float64, len(p.Data))
		v[i] = make([]float64, len(p.Data))
		grads[i] = Zeros(p.Shape...)
	}
	return &AdamWF64{
		Params:      params,
		Grads:       grads,
		LR:          lr,
		Beta1:       0.9,
		Beta2:       0.95,
		Eps:         1e-8,
		WeightDecay: 0.1,
		m:           m,
		v:           v,
	}
}

// ZeroGrad — обнулить все градиенты (в начале Step).
func (opt *AdamWF64) ZeroGrad() {
	for _, g := range opt.Grads {
		for i := range g.Data {
			g.Data[i] = 0
		}
	}
}

// Step — один шаг AdamW. Ожидает что Grads уже заполнены backward.
// In-place обновляет Params.
func (opt *AdamWF64) Step() {
	opt.step++
	bc1 := 1.0 - math.Pow(opt.Beta1, float64(opt.step))
	bc2 := 1.0 - math.Pow(opt.Beta2, float64(opt.step))
	for k, p := range opt.Params {
		gData := opt.Grads[k].Data
		mData := opt.m[k]
		vData := opt.v[k]
		pData := p.Data
		for i := 0; i < len(pData); i++ {
			g := gData[i]
			mData[i] = opt.Beta1*mData[i] + (1-opt.Beta1)*g
			vData[i] = opt.Beta2*vData[i] + (1-opt.Beta2)*g*g
			mHat := mData[i] / bc1
			vHat := vData[i] / bc2
			update := mHat/(math.Sqrt(vHat)+opt.Eps) + opt.WeightDecay*pData[i]
			pData[i] -= opt.LR * update
		}
	}
}

// SetLR — для scheduler'ов.
func (opt *AdamWF64) SetLR(lr float64) { opt.LR = lr }
