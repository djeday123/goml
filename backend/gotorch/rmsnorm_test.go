package gotorch

// P2-RMS Stage 3: adapter direct RMSNormF32/F64 fwd+bwd против F64 CPU-судьи.
//
// Пути:
//   B  = adapter direct (наш новый порт, gotorch RMSNormF32/F64 через adapter).
//   J  = F64 CPU-судья (numerical grad-check закрыт в rmsnorm_test.go на gotorch стороне).
//
// A-путь отсутствует: goml.cuda.RMSNorm — FP16 dlopen orphan (0 callers), не
// F32/F64, не в backend.Backend interface. Никакой legacy F32/F64 RMSNorm в
// goml нет — port через adapter = единственный живой путь.
//
// LLM-shape: [16, 64] (batch*seq=16, embed=64) — тот же shape что gputrain'ish
// но для transformer-style Block'а. Pre-registered floor:
//   F32 fwd: hybrid (abs=1e-4 + rel=1e-5), fails=0 из 1024.
//   F64 fwd: rel≤1e-12, fails=0 из 1024.
//   F32 grad dx: hybrid (abs=1e-4 + rel=1e-4).
//   F32 grad dgamma: hybrid (abs=1e-3 + rel=1e-4).
// Ожидание: с запасом 10-100× — gotorch-cuda тесты дают maxRel ~1e-7 для F32.

import (
	"math"
	"math/rand"
	"testing"

	"github.com/djeday123/goml/backend"
)

// rmsNormJudgeF64 — CPU F64 forward reference (тот же алгоритм что в gotorch/v6/cuda tests).
func rmsNormJudgeF64(x, gamma []float64, rows, cols int, eps float64) []float64 {
	y := make([]float64, rows*cols)
	for r := 0; r < rows; r++ {
		var s2 float64
		for c := 0; c < cols; c++ {
			v := x[r*cols+c]
			s2 += v * v
		}
		inv := 1.0 / math.Sqrt(s2/float64(cols)+eps)
		for c := 0; c < cols; c++ {
			y[r*cols+c] = gamma[c] * x[r*cols+c] * inv
		}
	}
	return y
}

func rmsNormGradJudgeF64(x, gamma, dy []float64, rows, cols int, eps float64) (dx, dgamma []float64) {
	dx = make([]float64, rows*cols)
	dgamma = make([]float64, cols)
	for r := 0; r < rows; r++ {
		var s2, S float64
		for c := 0; c < cols; c++ {
			v := x[r*cols+c]
			s2 += v * v
			S += gamma[c] * v * dy[r*cols+c]
		}
		inv := 1.0 / math.Sqrt(s2/float64(cols)+eps)
		inv3 := inv * inv * inv / float64(cols)
		for c := 0; c < cols; c++ {
			t1 := gamma[c] * dy[r*cols+c] * inv
			t2 := x[r*cols+c] * S * inv3
			dx[r*cols+c] = t1 - t2
			dgamma[c] += dy[r*cols+c] * x[r*cols+c] * inv
		}
	}
	return
}

func TestAdapterRMSNormF32_BvsJ(t *testing.T) {
	// Pre-registered floor (ДО прогона):
	// F32 fwd hybrid abs=1e-4 + rel=1e-5. Ожидаем 0 fails из 1024.
	const absTol, relTol = 1e-4, 1e-5

	bAny := tryEnable(t)
	b, ok := bAny.(*Backend)
	if !ok {
		t.Fatalf("adapter Get(CUDA) returned %T, expected *Backend", bAny)
	}

	const rows, cols = 16, 64
	const eps float32 = 1e-6
	r := rand.New(rand.NewSource(42))
	x := make([]float32, rows*cols)
	gamma := make([]float32, cols)
	xF64 := make([]float64, rows*cols)
	gF64 := make([]float64, cols)
	for i := range x {
		x[i] = float32(r.NormFloat64())
		xF64[i] = float64(x[i])
	}
	for i := range gamma {
		gamma[i] = float32(1.0 + 0.1*r.NormFloat64())
		gF64[i] = float64(gamma[i])
	}
	refY := rmsNormJudgeF64(xF64, gF64, rows, cols, float64(eps))

	xS := allocFromSlice(t, b, x)
	defer b.Free(xS)
	gS := allocFromSlice(t, b, gamma)
	defer b.Free(gS)
	yS, err := b.Alloc(rows * cols * 4)
	if err != nil {
		t.Fatalf("Alloc y: %v", err)
	}
	defer b.Free(yS)

	if err := b.RMSNormF32(xS, gS, yS, rows, cols, eps); err != nil {
		t.Fatalf("adapter RMSNormF32: %v", err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	got := downloadF32(t, b, yS)

	var maxAbs, maxRel float64
	fails := 0
	for i := range got {
		d := math.Abs(float64(got[i]) - refY[i])
		rel := d / (math.Abs(refY[i]) + 1e-30)
		if d > maxAbs {
			maxAbs = d
		}
		if rel > maxRel {
			maxRel = rel
		}
		if d > absTol+relTol*math.Abs(refY[i]) {
			fails++
		}
	}
	t.Logf("B(adapter FP32) vs J(F64) fwd [16,64]: maxAbs=%.3e maxRel=%.3e fails=%d/%d (floor abs=%.0e+rel=%.0e·|ref|)",
		maxAbs, maxRel, fails, len(got), absTol, relTol)
	if fails > 0 {
		t.Errorf("adapter RMSNormF32 vs judge: %d fails", fails)
	}
}

func TestAdapterRMSNormF64_BvsJ(t *testing.T) {
	// Pre-registered floor: F64 rel≤1e-12, 0 fails.
	const relTol = 1e-12

	bAny := tryEnable(t)
	b, ok := bAny.(*Backend)
	if !ok {
		t.Fatalf("adapter Get(CUDA) returned %T, expected *Backend", bAny)
	}

	const rows, cols = 16, 64
	const eps = 1e-6
	r := rand.New(rand.NewSource(43))
	x := make([]float64, rows*cols)
	gamma := make([]float64, cols)
	for i := range x {
		x[i] = r.NormFloat64()
	}
	for i := range gamma {
		gamma[i] = 1.0 + 0.1*r.NormFloat64()
	}
	refY := rmsNormJudgeF64(x, gamma, rows, cols, eps)

	xS := allocFromSliceF64(t, b, x)
	defer b.Free(xS)
	gS := allocFromSliceF64(t, b, gamma)
	defer b.Free(gS)
	yS, err := b.Alloc(rows * cols * 8)
	if err != nil {
		t.Fatalf("Alloc y: %v", err)
	}
	defer b.Free(yS)

	if err := b.RMSNormF64(xS, gS, yS, rows, cols, eps); err != nil {
		t.Fatalf("adapter RMSNormF64: %v", err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	got := downloadF64(t, b, yS)

	var maxRel float64
	fails := 0
	for i := range got {
		rel := math.Abs(got[i]-refY[i]) / (math.Abs(refY[i]) + 1e-30)
		if rel > maxRel {
			maxRel = rel
		}
		if rel > relTol {
			fails++
		}
	}
	t.Logf("B(adapter F64) vs J(F64) fwd [16,64]: maxRel=%.3e fails=%d/%d (floor rel=%.0e)",
		maxRel, fails, len(got), relTol)
	if fails > 0 {
		t.Errorf("adapter RMSNormF64 vs judge: %d fails", fails)
	}
}

func TestAdapterRMSNormGradF32_BvsJ(t *testing.T) {
	// Pre-registered floor:
	// dx hybrid abs=1e-4 + rel=1e-4, dgamma hybrid abs=1e-3 + rel=1e-4.
	const absDx, relDx = 1e-4, 1e-4
	const absDg, relDg = 1e-3, 1e-4

	bAny := tryEnable(t)
	b, ok := bAny.(*Backend)
	if !ok {
		t.Fatalf("adapter Get(CUDA) returned %T", bAny)
	}

	const rows, cols = 16, 64
	const eps float32 = 1e-6
	r := rand.New(rand.NewSource(44))
	x := make([]float32, rows*cols)
	gamma := make([]float32, cols)
	dy := make([]float32, rows*cols)
	xF, gF, dyF := make([]float64, rows*cols), make([]float64, cols), make([]float64, rows*cols)
	for i := range x {
		x[i] = float32(r.NormFloat64())
		dy[i] = float32(r.NormFloat64())
		xF[i] = float64(x[i])
		dyF[i] = float64(dy[i])
	}
	for i := range gamma {
		gamma[i] = float32(1.0 + 0.1*r.NormFloat64())
		gF[i] = float64(gamma[i])
	}
	refDx, refDg := rmsNormGradJudgeF64(xF, gF, dyF, rows, cols, float64(eps))

	xS := allocFromSlice(t, b, x)
	defer b.Free(xS)
	gS := allocFromSlice(t, b, gamma)
	defer b.Free(gS)
	dyS := allocFromSlice(t, b, dy)
	defer b.Free(dyS)
	dxS, _ := b.Alloc(rows * cols * 4)
	defer b.Free(dxS)
	dgS, _ := b.Alloc(cols * 4)
	defer b.Free(dgS)

	if err := b.RMSNormGradF32(xS, gS, dyS, dxS, dgS, rows, cols, eps); err != nil {
		t.Fatalf("adapter RMSNormGradF32: %v", err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	gotDx := downloadF32(t, b, dxS)
	gotDg := downloadF32(t, b, dgS)

	var mxDxAbs, mxDxRel, mxDgAbs, mxDgRel float64
	failsDx, failsDg := 0, 0
	for i := range gotDx {
		d := math.Abs(float64(gotDx[i]) - refDx[i])
		rel := d / (math.Abs(refDx[i]) + 1e-30)
		if d > mxDxAbs {
			mxDxAbs = d
		}
		if rel > mxDxRel {
			mxDxRel = rel
		}
		if d > absDx+relDx*math.Abs(refDx[i]) {
			failsDx++
		}
	}
	for i := range gotDg {
		d := math.Abs(float64(gotDg[i]) - refDg[i])
		rel := d / (math.Abs(refDg[i]) + 1e-30)
		if d > mxDgAbs {
			mxDgAbs = d
		}
		if rel > mxDgRel {
			mxDgRel = rel
		}
		if d > absDg+relDg*math.Abs(refDg[i]) {
			failsDg++
		}
	}
	t.Logf("B(adapter FP32) vs J(F64) bwd [16,64]: dx maxAbs=%.3e maxRel=%.3e fails=%d/%d; dgamma maxAbs=%.3e maxRel=%.3e fails=%d/%d",
		mxDxAbs, mxDxRel, failsDx, len(gotDx), mxDgAbs, mxDgRel, failsDg, len(gotDg))
	if failsDx > 0 {
		t.Errorf("adapter RMSNormGradF32 dx: %d fails (floor abs=%.0e+rel=%.0e·|ref|)", failsDx, absDx, relDx)
	}
	if failsDg > 0 {
		t.Errorf("adapter RMSNormGradF32 dgamma: %d fails (floor abs=%.0e+rel=%.0e·|ref|)", failsDg, absDg, relDg)
	}
}

// ─────────── helpers для F64 ───────────

func allocFromSliceF64(t *testing.T, b backend.Backend, xs []float64) backend.Storage {
	t.Helper()
	cpuB, _ := backend.Get(backend.CPU)
	cpu, err := cpuB.Alloc(len(xs) * 8)
	if err != nil {
		t.Fatalf("cpu.Alloc: %v", err)
	}
	buf := cpu.Bytes()
	for i, v := range xs {
		bits := math.Float64bits(v)
		for k := 0; k < 8; k++ {
			buf[i*8+k] = byte(bits >> (8 * k))
		}
	}
	gpu, err := b.ToDevice(backend.CUDADevice(0), cpu)
	cpuB.Free(cpu)
	if err != nil {
		t.Fatalf("ToDevice F64: %v", err)
	}
	return gpu
}

func downloadF64(t *testing.T, b backend.Backend, s backend.Storage) []float64 {
	t.Helper()
	cpu, err := b.ToDevice(backend.CPU0, s)
	if err != nil {
		t.Fatalf("ToDevice back F64: %v", err)
	}
	defer func() { cpuB, _ := backend.Get(backend.CPU); cpuB.Free(cpu) }()
	buf := cpu.Bytes()
	n := len(buf) / 8
	xs := make([]float64, n)
	for i := 0; i < n; i++ {
		var bits uint64
		for k := 0; k < 8; k++ {
			bits |= uint64(buf[i*8+k]) << (8 * k)
		}
		xs[i] = math.Float64frombits(bits)
	}
	return xs
}
