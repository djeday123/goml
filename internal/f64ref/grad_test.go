package f64ref

// Ворота 5a: численный градиент для каждого backward компонента.
// Правило: |analytical - numerical| / (|numerical| + 1e-30) < 1e-8.
// F64 позволяет жёстко — метод из саги 93.3 (FP64-probe).

import (
	"math"
	"math/rand"
	"testing"
)

// randTensor — детерминированный F64 tensor заданной формы.
func randTensor(r *rand.Rand, shape ...int) *F64Tensor {
	n := 1
	for _, d := range shape {
		n *= d
	}
	data := make([]float64, n)
	for i := range data {
		data[i] = r.NormFloat64() * 0.5
	}
	return NewF64Tensor(data, shape)
}

// dotDy — детерминированный dY того же shape что y (для функции loss = <y, dY>).
func dotDy(r *rand.Rand, shape ...int) *F64Tensor {
	return randTensor(r, shape...)
}

// scalarDot — <a, b>.
func scalarDot(a, b *F64Tensor) float64 {
	var s float64
	for i := range a.Data {
		s += a.Data[i] * b.Data[i]
	}
	return s
}

// gradCheck — сравнение аналитического градиента с central-difference
// numerical градиентом для функции loss(params) = scalarDot(f(params), dY).
// Проверяет только `n` случайных элементов param (dtype-выборка не-каждый).
func gradCheck(t *testing.T, tag string,
	analytical *F64Tensor, param *F64Tensor,
	forward func() *F64Tensor, dY *F64Tensor,
	nSamples int, relTol float64) {
	t.Helper()

	if len(analytical.Data) != len(param.Data) {
		t.Fatalf("%s: analytical.Data %d != param.Data %d", tag, len(analytical.Data), len(param.Data))
	}
	const h = 1e-6
	r := rand.New(rand.NewSource(int64(len(param.Data))))
	worstIdx := -1
	worstRel := 0.0
	for k := 0; k < nSamples && k < len(param.Data); k++ {
		i := r.Intn(len(param.Data))
		orig := param.Data[i]
		// f(+h).
		param.Data[i] = orig + h
		yPlus := forward()
		lossPlus := scalarDot(yPlus, dY)
		// f(-h).
		param.Data[i] = orig - h
		yMinus := forward()
		lossMinus := scalarDot(yMinus, dY)
		// restore.
		param.Data[i] = orig
		numGrad := (lossPlus - lossMinus) / (2 * h)
		anaGrad := analytical.Data[i]
		diff := math.Abs(numGrad - anaGrad)
		rel := diff / (math.Abs(numGrad) + 1e-30)
		if rel > worstRel {
			worstRel, worstIdx = rel, i
		}
	}
	t.Logf("%s: sampled=%d worst rel=%.3e at idx=%d (tol=%.0e)",
		tag, nSamples, worstRel, worstIdx, relTol)
	if worstRel > relTol {
		t.Errorf("%s: worstRel=%.3e > tol=%.0e — analytical grad diverges from numerical", tag, worstRel, relTol)
	}
}

// ================================================================
// LinearF64
// ================================================================
func TestLinearF64_GradCheck(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	const M, K, N = 4, 6, 5
	x := randTensor(r, M, K)
	W := randTensor(r, N, K)
	bias := randTensor(r, N)
	lin := NewLinearF64(W, bias)

	// forward + analytical backward.
	y := lin.Forward(x)
	dY := dotDy(r, M, N)
	dX, dW, dB := lin.Backward(x, dY)

	// numerical grad для dW (param = W).
	gradCheck(t, "LinearF64.dW",
		dW, W,
		func() *F64Tensor { return lin.Forward(x) },
		dY, 20, 1e-8)

	// numerical grad для dB (param = bias).
	gradCheck(t, "LinearF64.dB",
		dB, bias,
		func() *F64Tensor { return lin.Forward(x) },
		dY, 5, 1e-8)

	// numerical grad для dX (param = x).
	gradCheck(t, "LinearF64.dX",
		dX, x,
		func() *F64Tensor { return lin.Forward(x) },
		dY, 20, 1e-8)

	// Sanity: y shape check.
	if y.Shape[0] != M || y.Shape[1] != N {
		t.Errorf("y shape %v != [%d, %d]", y.Shape, M, N)
	}
}

// ================================================================
// LayerNormF64
// ================================================================
func TestLayerNormF64_GradCheck(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	const rows, D = 4, 8
	x := randTensor(r, rows, D)
	ln := NewLayerNormF64(D, 1e-5)
	// Инициализируем gamma/beta не-тривиально.
	for i := range ln.Gamma.Data {
		ln.Gamma.Data[i] = 0.5 + r.NormFloat64()*0.1
		ln.Beta.Data[i] = r.NormFloat64() * 0.1
	}

	_, cache := ln.Forward(x)
	dY := dotDy(r, rows, D)
	dX, dGamma, dBeta := ln.Backward(dY, cache)

	// dx.
	gradCheck(t, "LayerNormF64.dX",
		dX, x,
		func() *F64Tensor {
			y, _ := ln.Forward(x)
			return y
		},
		dY, 20, 1e-7)

	// dGamma.
	gradCheck(t, "LayerNormF64.dGamma",
		dGamma, ln.Gamma,
		func() *F64Tensor {
			y, _ := ln.Forward(x)
			return y
		},
		dY, 8, 1e-7)

	// dBeta.
	gradCheck(t, "LayerNormF64.dBeta",
		dBeta, ln.Beta,
		func() *F64Tensor {
			y, _ := ln.Forward(x)
			return y
		},
		dY, 8, 1e-8)
}

// ================================================================
// EmbeddingF64
// ================================================================
func TestEmbeddingF64_GradCheck(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	const V, D = 10, 6
	const batch, seqLen = 2, 4
	emb := NewEmbeddingF64(V, D)
	for i := range emb.Weight.Data {
		emb.Weight.Data[i] = r.NormFloat64() * 0.1
	}
	indices := make([]int64, batch*seqLen)
	for i := range indices {
		indices[i] = int64(r.Intn(V))
	}

	y := emb.Forward(indices, batch, seqLen)
	dY := dotDy(r, batch, seqLen, D)
	dW := emb.Backward(indices, dY, batch, seqLen)

	gradCheck(t, "EmbeddingF64.dW",
		dW, emb.Weight,
		func() *F64Tensor { return emb.Forward(indices, batch, seqLen) },
		dY, 20, 1e-8)

	// Sanity: y shape check.
	if y.Shape[0] != batch || y.Shape[1] != seqLen || y.Shape[2] != D {
		t.Errorf("y shape %v", y.Shape)
	}
}

// ================================================================
// AdamWF64 vs ручной расчёт
// ================================================================
func TestAdamWF64_ManualFormula(t *testing.T) {
	// Игрушечная задача: один параметр 2 элемента, известный градиент,
	// один шаг — сверить формулу.
	p := NewF64Tensor([]float64{1.0, 2.0}, []int{2})
	opt := NewAdamWF64([]*F64Tensor{p}, 0.1)
	opt.Beta1 = 0.9
	opt.Beta2 = 0.999
	opt.Eps = 1e-8
	opt.WeightDecay = 0.01

	// grad = [0.5, -1.0], один шаг.
	opt.Grads[0].Data[0] = 0.5
	opt.Grads[0].Data[1] = -1.0
	opt.Step()

	// Ручной расчёт t=1:
	// m = 0.9*0 + 0.1*g = 0.1*g = [0.05, -0.1]
	// v = 0.999*0 + 0.001*g^2 = 0.001*[0.25, 1.0] = [0.00025, 0.001]
	// bc1 = 1 - 0.9^1 = 0.1
	// bc2 = 1 - 0.999^1 = 0.001
	// mHat = [0.05/0.1, -0.1/0.1] = [0.5, -1.0]
	// vHat = [0.00025/0.001, 0.001/0.001] = [0.25, 1.0]
	// update = mHat/(sqrt(vHat)+eps) + wd*p
	//        p_before = [1.0, 2.0]
	//        update[0] = 0.5/(sqrt(0.25)+1e-8) + 0.01*1.0 = 0.5/0.500001 + 0.01 ≈ 0.99999998 + 0.01 ≈ 1.00999998
	//        update[1] = -1.0/(sqrt(1.0)+1e-8) + 0.01*2.0 = -0.999999990 + 0.02 ≈ -0.97999999
	// p = p - lr*update
	//   p[0] = 1.0 - 0.1*1.00999998 = 1.0 - 0.100999998 = 0.899000002
	//   p[1] = 2.0 - 0.1*(-0.97999999) = 2.0 + 0.097999999 = 2.097999999
	expected := []float64{0.899000002, 2.098000001}
	for i := 0; i < 2; i++ {
		diff := math.Abs(p.Data[i] - expected[i])
		if diff > 1e-8 {
			t.Errorf("AdamW manual formula: p[%d]=%.15f expected=%.15f diff=%.3e",
				i, p.Data[i], expected[i], diff)
		}
	}
	t.Logf("AdamW step 1 manual: p=[%.15f, %.15f] (expected [%.9f, %.9f])",
		p.Data[0], p.Data[1], expected[0], expected[1])
}
