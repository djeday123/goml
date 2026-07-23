package f64ref

// LinearF64 — эталонный Linear-слой в F64.
// y = x @ W^T + bias, где
//   x    : [..., InF]
//   W    : [OutF, InF]
//   bias : [OutF] или nil
//
// MatMul идёт на GPU F64 через gotorch, bias/reshape — на host.

type LinearF64 struct {
	Weight *F64Tensor // [OutF, InF]
	Bias   *F64Tensor // [OutF] или nil
	InF    int
	OutF   int
}

// NewLinearF64 — конструктор. Weight/Bias — уже готовые тензоры.
func NewLinearF64(W, Bias *F64Tensor) *LinearF64 {
	if len(W.Shape) != 2 {
		panic("LinearF64: W must be 2D")
	}
	OutF := W.Shape[0]
	InF := W.Shape[1]
	if Bias != nil && (len(Bias.Shape) != 1 || Bias.Shape[0] != OutF) {
		panic("LinearF64: Bias shape must be [OutF]")
	}
	return &LinearF64{Weight: W, Bias: Bias, InF: InF, OutF: OutF}
}

// Forward считает y = x @ W^T + bias.
// x.Shape = [..., InF] → y.Shape = [..., OutF].
//
// Реализация: reshape x → [M, InF] где M = prod(x.Shape[:-1]);
// MatMul на GPU F64 [M, InF] × [InF, OutF] → [M, OutF] (обёртка W^T через
// col-major swap trick уже в gotorch); bias добавляется на host loop.
func (l *LinearF64) Forward(x *F64Tensor) *F64Tensor {
	if x.Shape[len(x.Shape)-1] != l.InF {
		panic("LinearF64.Forward: last dim of x must equal InF")
	}
	M := x.NumElements() / l.InF
	// Форма для GPU: x[M, InF] × W^T[InF, OutF] = y[M, OutF].
	// gotorch.MatMulF64(a, b, c, m, n, k) считает C = A×B row-major, где
	// A [M×K], B [K×N], C [M×N]. Здесь A=x[M, InF], B=W^T[InF, OutF].
	// W хранится [OutF, InF] — то есть его "row-major" память НЕ равна W^T.
	// Формируем W^T на host явно перед upload (10 строк, F64 точно).
	Wt := transpose2D(l.Weight)
	y := MatMulF64GPU(&F64Tensor{Data: x.Data, Shape: []int{M, l.InF}}, Wt, M, l.InF, l.OutF)

	// bias broadcast [OutF] → [M, OutF], добавляем на host.
	if l.Bias != nil {
		for row := 0; row < M; row++ {
			for i := 0; i < l.OutF; i++ {
				y.Data[row*l.OutF+i] += l.Bias.Data[i]
			}
		}
	}
	// Восстанавливаем выходной shape: заменяем последнюю dim x на OutF.
	outShape := make([]int, len(x.Shape))
	copy(outShape, x.Shape)
	outShape[len(outShape)-1] = l.OutF
	y.Shape = outShape
	return y
}

// Backward: given dY (gradient of loss wrt output y), returns:
//   dX = dY @ W                 (shape same as x)
//   dW = dY^T @ x   as [OutF, InF]  ← summed over batch dims
//   dB = sum_over_batch dY      shape [OutF] (nil if no bias)
//
// Через reshape [M, OutF] × [OutF, InF] → [M, InF] и [OutF, M] × [M, InF] → [OutF, InF].
func (l *LinearF64) Backward(x, dY *F64Tensor) (dX, dW, dB *F64Tensor) {
	if x.Shape[len(x.Shape)-1] != l.InF {
		panic("LinearF64.Backward: x InF mismatch")
	}
	if dY.Shape[len(dY.Shape)-1] != l.OutF {
		panic("LinearF64.Backward: dY OutF mismatch")
	}
	M := x.NumElements() / l.InF
	// dX = dY @ W: [M, OutF] × [OutF, InF] = [M, InF]
	dXFlat := MatMulF64GPU(
		&F64Tensor{Data: dY.Data, Shape: []int{M, l.OutF}},
		l.Weight, // [OutF, InF]
		M, l.OutF, l.InF)
	dXFlat.Shape = append([]int{}, x.Shape...)
	dX = dXFlat

	// dW = dY^T @ x: [OutF, M] × [M, InF] = [OutF, InF]
	dYT := transpose2D(&F64Tensor{Data: dY.Data, Shape: []int{M, l.OutF}})
	xFlat := &F64Tensor{Data: x.Data, Shape: []int{M, l.InF}}
	dW = MatMulF64GPU(dYT, xFlat, l.OutF, M, l.InF)

	// dB = sum over M of dY[M, OutF]
	if l.Bias != nil {
		dB = Zeros(l.OutF)
		for row := 0; row < M; row++ {
			for i := 0; i < l.OutF; i++ {
				dB.Data[i] += dY.Data[row*l.OutF+i]
			}
		}
	}
	return
}

// transpose2D — host-side для 2D тензора [rows, cols] → [cols, rows].
func transpose2D(t *F64Tensor) *F64Tensor {
	if len(t.Shape) != 2 {
		panic("transpose2D: not 2D")
	}
	rows, cols := t.Shape[0], t.Shape[1]
	out := Zeros(cols, rows)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out.Data[c*rows+r] = t.Data[r*cols+c]
		}
	}
	return out
}
