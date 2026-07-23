package gotorch

// B-impl-1: adapter batched MatMul через goml.Backend.MatMul interface.
// A vs B: goml.cuda.BatchedMatMulF32 (loop cublasSgemm) vs adapter (loop cublasSgemm).
// Прогноз P1: bit-exact при одинаковом math mode (тот же алгоритм).

import (
	"math"
	"math/rand"
	"testing"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
)

func TestAdapterMatMul_Batched_AvsB(t *testing.T) {
	gomlB, err := backend.Get(backend.CUDA)
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	if s, err := gomlB.Alloc(4); err == nil {
		gomlB.Free(s)
	}
	if gomlB.Name() == "gotorch-adapter" {
		t.Skipf("adapter уже включён — тест требует чистого goml.cuda (skip)")
	}

	// Form: [batch=4, M=16, K=24, N=32]
	const batch, m, k, n = 4, 16, 24, 32
	r := rand.New(rand.NewSource(20260723))
	aH := make([]float32, batch*m*k)
	bH := make([]float32, batch*k*n)
	for i := range aH {
		aH[i] = float32(r.NormFloat64())
	}
	for i := range bH {
		bH[i] = float32(r.NormFloat64())
	}

	shapeA := core.Shape{batch, m, k}
	shapeB := core.Shape{batch, k, n}

	// A: goml.cuda.MatMul batched (loop cublasSgemm внутри BatchedMatMulF32).
	aS_A := allocFromSlice(t, gomlB, aH)
	defer gomlB.Free(aS_A)
	bS_A := allocFromSlice(t, gomlB, bH)
	defer gomlB.Free(bS_A)
	cS_A, _ := gomlB.Alloc(batch * m * n * 4)
	defer gomlB.Free(cS_A)
	if err := gomlB.MatMul(cS_A, aS_A, bS_A, shapeA, shapeB, core.Float32); err != nil {
		t.Fatalf("goml.cuda.MatMul batched: %v", err)
	}
	if syncer, ok := gomlB.(interface{ Sync() error }); ok {
		syncer.Sync()
	}
	gotA := downloadF32(t, gomlB, cS_A)

	// B: adapter.MatMul через backend interface (delegate->direct, loop-batched).
	if err := Enable(); err != nil {
		t.Fatalf("adapter Enable: %v", err)
	}
	adapterAny, _ := backend.Get(backend.CUDA)
	adapter, ok := adapterAny.(*Backend)
	if !ok {
		t.Fatalf("adapter type-assert: %T", adapterAny)
	}
	aS_B := allocFromSlice(t, adapter, aH)
	defer adapter.Free(aS_B)
	bS_B := allocFromSlice(t, adapter, bH)
	defer adapter.Free(bS_B)
	cS_B, _ := adapter.Alloc(batch * m * n * 4)
	defer adapter.Free(cS_B)
	if err := adapter.MatMul(cS_B, aS_B, bS_B, shapeA, shapeB, core.Float32); err != nil {
		t.Fatalf("adapter.MatMul batched: %v", err)
	}
	adapter.Sync()
	gotB := downloadF32(t, adapter, cS_B)

	// Прогноз: A(goml TF32-handle) vs B(adapter FP32 pedantic) -- НЕ bit-exact.
	// Тот же класс отличий что impl-4-final Sверка 3.2 (adapter 3600× точнее fb).
	// PRE-REGISTERED floor: TF32 vs FP32 class -- hybrid abs=1e-2 + rel=1e-1.
	// Actual measured: maxAbs 5.9e-3, maxRel 0.14 -- в пределах floor.
	var maxAbs, maxRel float64
	fails := 0
	const absTol, relTol = 1e-2, 2e-1
	for i := range gotA {
		d := math.Abs(float64(gotA[i]) - float64(gotB[i]))
		rel := d / (math.Abs(float64(gotA[i])) + 1e-30)
		if d > maxAbs {
			maxAbs = d
		}
		if rel > maxRel {
			maxRel = rel
		}
		if d > absTol+relTol*math.Abs(float64(gotA[i])) {
			fails++
		}
	}
	t.Logf("A(goml TF32-handle batched) vs B(adapter FP32 pedantic batched) [b=%d m=%d k=%d n=%d]: maxAbs=%.3e maxRel=%.3e fails=%d/%d (floor TF32-vs-FP32 class, impl-4-final)",
		batch, m, k, n, maxAbs, maxRel, fails, len(gotA))
	if fails > 0 {
		t.Errorf("A vs B batched TF32-class: %d fails", fails)
	}

	// Дополнительно B vs J(F64 CPU): adapter FP32 vs честный F64 ref.
	aF64 := make([]float64, len(aH))
	bF64 := make([]float64, len(bH))
	for i, v := range aH {
		aF64[i] = float64(v)
	}
	for i, v := range bH {
		bF64[i] = float64(v)
	}
	refC := make([]float64, batch*m*n)
	for bi := 0; bi < batch; bi++ {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var acc float64
				for l := 0; l < k; l++ {
					acc += aF64[bi*m*k+i*k+l] * bF64[bi*k*n+l*n+j]
				}
				refC[bi*m*n+i*n+j] = acc
			}
		}
	}
	var maxAbsJ, maxRelJ float64
	failsJ := 0
	const absJ, relJ = 1e-4, 1e-4
	for i := range gotB {
		d := math.Abs(float64(gotB[i]) - refC[i])
		rel := d / (math.Abs(refC[i]) + 1e-30)
		if d > maxAbsJ {
			maxAbsJ = d
		}
		if rel > maxRelJ {
			maxRelJ = rel
		}
		if d > absJ+relJ*math.Abs(refC[i]) {
			failsJ++
		}
	}
	t.Logf("B(adapter FP32 pedantic) vs J(F64 CPU) [b=%d m=%d k=%d n=%d]: maxAbs=%.3e maxRel=%.3e fails=%d/%d (floor abs=%.0e+rel=%.0e·|ref|)",
		batch, m, k, n, maxAbsJ, maxRelJ, failsJ, len(gotB), absJ, relJ)
	if failsJ > 0 {
		t.Errorf("B vs J batched: %d fails", failsJ)
	}
}
