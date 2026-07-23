package gotorch

// P4-ROPE Stage 3: adapter direct RoPE F32/F64 vs J-судья + A vs B (goml.cuda vs adapter).
//
// Прогнозы (pre-registered):
//   P1 A/B fwd F32: bit-exact (наш PTX -- дословная копия goml.cuda.rope_f32).
//   P2 B/J F32 fwd на LLM-tiny [b=2,h=4,sl=16,hd=64]: hybrid abs=1e-4+rel=1e-3.
//   P3 B/J F64 fwd (host tables): rel <= 1e-12.
//   P4 B/J F32 bwd hybrid.

import (
	"math"
	"math/rand"
	"testing"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
)

// ropeF64Ref -- CPU host RoPE F64 reference (используется для J-судьи).
func ropeF64Ref(src []float64, batch, heads, seqLen, headDim int, base float64) []float64 {
	half := headDim / 2
	out := make([]float64, len(src))
	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			for p := 0; p < seqLen; p++ {
				rowBase := ((b*heads+h)*seqLen + p) * headDim
				for i := 0; i < half; i++ {
					angle := float64(p) * math.Pow(base, -2.0*float64(i)/float64(headDim))
					c := math.Cos(angle)
					s := math.Sin(angle)
					x0 := src[rowBase+i]
					x1 := src[rowBase+i+half]
					out[rowBase+i] = x0*c - x1*s
					out[rowBase+i+half] = x0*s + x1*c
				}
			}
		}
	}
	return out
}

func buildRoPETablesTest(seqLen, headDim int, base float64) (cos, sin []float64) {
	half := headDim / 2
	cos = make([]float64, seqLen*half)
	sin = make([]float64, seqLen*half)
	for p := 0; p < seqLen; p++ {
		for i := 0; i < half; i++ {
			angle := float64(p) * math.Pow(base, -2.0*float64(i)/float64(headDim))
			cos[p*half+i] = math.Cos(angle)
			sin[p*half+i] = math.Sin(angle)
		}
	}
	return
}

// A vs B forward: goml.cuda vs adapter. Должно быть bit-exact -- наш PTX копирует goml.
// Порядок: A запускается ДО Enable(); B после.
func TestAdapterRoPE_AvsB_Forward(t *testing.T) {
	gomlB, err := backend.Get(backend.CUDA)
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	if s, err := gomlB.Alloc(4); err == nil {
		gomlB.Free(s)
	}
	if gomlB.Name() == "gotorch-adapter" {
		t.Skipf("adapter уже включён -- этот тест требует чистого goml.cuda (skip)")
	}

	const batch, heads, seqLen, headDim = 2, 4, 16, 64
	const base = 10000.0
	n := batch * heads * seqLen * headDim
	r := rand.New(rand.NewSource(4444))
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	shape := core.Shape{batch, heads, seqLen, headDim}

	// A: goml.cuda.RoPE
	xS_A := allocFromSlice(t, gomlB, x)
	defer gomlB.Free(xS_A)
	oS_A, _ := gomlB.Alloc(n * 4)
	defer gomlB.Free(oS_A)
	if err := gomlB.RoPE(oS_A, xS_A, shape, headDim, base, core.Float32); err != nil {
		t.Fatalf("goml.cuda.RoPE: %v", err)
	}
	if syncer, ok := gomlB.(interface{ Sync() error }); ok {
		syncer.Sync()
	}
	gotA := downloadF32(t, gomlB, oS_A)

	// B: adapter (тот же путь через backend interface RoPE -> gt.RoPEF32).
	if err := Enable(); err != nil {
		t.Fatalf("adapter Enable: %v", err)
	}
	adapterAny, _ := backend.Get(backend.CUDA)
	adapter, ok := adapterAny.(*Backend)
	if !ok {
		t.Fatalf("adapter type-assert: %T", adapterAny)
	}
	xS_B := allocFromSlice(t, adapter, x)
	defer adapter.Free(xS_B)
	oS_B, _ := adapter.Alloc(n * 4)
	defer adapter.Free(oS_B)
	if err := adapter.RoPEF32(xS_B, oS_B, batch, heads, seqLen, headDim, float32(base)); err != nil {
		t.Fatalf("adapter RoPEF32: %v", err)
	}
	adapter.Sync()
	gotB := downloadF32(t, adapter, oS_B)

	mismatches := 0
	for i := range gotA {
		if math.Float32bits(gotA[i]) != math.Float32bits(gotB[i]) {
			mismatches++
		}
	}
	t.Logf("A(goml.cuda) vs B(adapter) F32 fwd [b=%d h=%d sl=%d hd=%d]: bit-exact=%d/%d",
		batch, heads, seqLen, headDim, len(gotA)-mismatches, len(gotA))
	if mismatches > 0 {
		t.Errorf("A vs B F32 fwd: %d bit-mismatches (P1 прогноз bit-exact)", mismatches)
	}
}

func TestAdapterRoPEF32_BvsJ(t *testing.T) {
	// Pre-registered: hybrid abs=1e-4 + rel=1e-3.
	bAny := tryEnable(t)
	b, ok := bAny.(*Backend)
	if !ok {
		t.Fatalf("adapter Get(CUDA): %T", bAny)
	}

	const batch, heads, seqLen, headDim = 2, 4, 16, 64
	const base float32 = 10000
	n := batch * heads * seqLen * headDim
	r := rand.New(rand.NewSource(5555))
	x := make([]float32, n)
	xF64 := make([]float64, n)
	for i := range x {
		x[i] = float32(r.NormFloat64())
		xF64[i] = float64(x[i])
	}
	refOut := ropeF64Ref(xF64, batch, heads, seqLen, headDim, float64(base))

	xS := allocFromSlice(t, b, x)
	defer b.Free(xS)
	oS, _ := b.Alloc(n * 4)
	defer b.Free(oS)
	if err := b.RoPEF32(xS, oS, batch, heads, seqLen, headDim, base); err != nil {
		t.Fatalf("adapter RoPEF32: %v", err)
	}
	b.Sync()
	got := downloadF32(t, b, oS)

	var maxAbs, maxRel float64
	fails := 0
	const absTol, relTol = 1e-4, 1e-3
	for i := range got {
		d := math.Abs(float64(got[i]) - refOut[i])
		rel := d / (math.Abs(refOut[i]) + 1e-30)
		if d > maxAbs {
			maxAbs = d
		}
		if rel > maxRel {
			maxRel = rel
		}
		if d > absTol+relTol*math.Abs(refOut[i]) {
			fails++
		}
	}
	t.Logf("B(adapter F32) vs J(F64) fwd [b=%d h=%d sl=%d hd=%d]: maxAbs=%.3e maxRel=%.3e fails=%d/%d (floor abs=%.0e+rel=%.0e·|ref|)",
		batch, heads, seqLen, headDim, maxAbs, maxRel, fails, len(got), absTol, relTol)
	if fails > 0 {
		t.Errorf("adapter RoPEF32 vs judge: %d fails", fails)
	}
}

func TestAdapterRoPEF64_BvsJ(t *testing.T) {
	// Pre-registered: rel <= 1e-12.
	bAny := tryEnable(t)
	b, ok := bAny.(*Backend)
	if !ok {
		t.Fatalf("adapter Get(CUDA): %T", bAny)
	}

	const batch, heads, seqLen, headDim = 2, 4, 16, 64
	const base = 10000.0
	n := batch * heads * seqLen * headDim
	r := rand.New(rand.NewSource(6666))
	x := make([]float64, n)
	for i := range x {
		x[i] = r.NormFloat64()
	}
	refOut := ropeF64Ref(x, batch, heads, seqLen, headDim, base)
	cosT, sinT := buildRoPETablesTest(seqLen, headDim, base)

	xS := allocFromSliceF64(t, b, x)
	defer b.Free(xS)
	cS := allocFromSliceF64(t, b, cosT)
	defer b.Free(cS)
	sS := allocFromSliceF64(t, b, sinT)
	defer b.Free(sS)
	oS, _ := b.Alloc(n * 8)
	defer b.Free(oS)

	if err := b.RoPEF64(xS, cS, sS, oS, batch, heads, seqLen, headDim); err != nil {
		t.Fatalf("adapter RoPEF64: %v", err)
	}
	b.Sync()
	got := downloadF64(t, b, oS)

	var maxRel float64
	fails := 0
	const relTol = 1e-12
	for i := range got {
		rel := math.Abs(got[i]-refOut[i]) / (math.Abs(refOut[i]) + 1e-30)
		if rel > maxRel {
			maxRel = rel
		}
		if rel > relTol {
			fails++
		}
	}
	t.Logf("B(adapter F64) vs J(F64) fwd [b=%d h=%d sl=%d hd=%d] (host tables): maxRel=%.3e fails=%d/%d (floor rel=%.0e)",
		batch, heads, seqLen, headDim, maxRel, fails, len(got), relTol)
	if fails > 0 {
		t.Errorf("adapter RoPEF64 vs judge: %d fails", fails)
	}
}
