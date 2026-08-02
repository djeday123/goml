package abjexam

// Unit test для matmulPlainT helper - все 4 flag combinations, F64 reference.
// PASS before applying к production reference-path replacements.

import (
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/djeday123/goml/backend"
	_ "github.com/djeday123/goml/backend/cpu"
	_ "github.com/djeday123/goml/backend/cuda"
	adapter "github.com/djeday123/goml/backend/gotorch"
	"github.com/djeday123/goml/core"
)

// hostTransposeF32 - reference transpose function.
func hostTransposeF32(h []float32, rows, cols int) []float32 {
	t := make([]float32, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			t[j*rows+i] = h[i*cols+j]
		}
	}
	return t
}

// matmulPlainT reference helper (identical к planned production one).
func matmulPlainT(b backend.Backend, dst, aStor, bStor backend.Storage, m, n, k int, transA, transB bool) error {
	var aUse backend.Storage = aStor
	var bUse backend.Storage = bStor
	if transA {
		aH := gpuToHost(b, aStor, k*m)
		aT := hostTransposeF32(aH, k, m)
		tmp, err := b.Alloc(m * k * 4)
		if err != nil {
			return err
		}
		defer b.Free(tmp)
		if _, err := uploadInto(b, tmp, f32ToBytes(aT)); err != nil {
			return err
		}
		aUse = tmp
	}
	if transB {
		bH := gpuToHost(b, bStor, n*k)
		bT := hostTransposeF32(bH, n, k)
		tmp, err := b.Alloc(k * n * 4)
		if err != nil {
			return err
		}
		defer b.Free(tmp)
		if _, err := uploadInto(b, tmp, f32ToBytes(bT)); err != nil {
			return err
		}
		bUse = tmp
	}
	return b.MatMul(dst, aUse, bUse, core.Shape{m, k}, core.Shape{k, n}, core.Float32)
}

// TestMatmulPlainT_Unit - все 4 flag combinations x 3 shapes vs CPU-F64 ref.
func TestMatmulPlainT_Unit(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	gomlB, err := backend.Get(backend.CUDA)
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	if s, err := gomlB.Alloc(4); err == nil {
		gomlB.Free(s)
	}
	if err := adapter.Enable(); err != nil {
		t.Fatalf("adapter Enable: %v", err)
	}
	adB, _ := backend.Get(backend.CUDA)
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Compute CPU reference: op(A) @ op(B), where op is transpose or not.
	// A stored as [k, m] if transA else [m, k].
	// B stored as [n, k] if transB else [k, n].
	// C[m, n] = op(A) @ op(B).
	cpuRef := func(aH []float32, bH []float32, m, n, k int, transA, transB bool) []float64 {
		c := make([]float64, m*n)
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var acc float64
				for l := 0; l < k; l++ {
					var av, bv float32
					if transA {
						// A stored [k, m], aH[l*m + i]
						av = aH[l*m+i]
					} else {
						av = aH[i*k+l]
					}
					if transB {
						// B stored [n, k], bH[j*k + l]
						bv = bH[j*k+l]
					} else {
						bv = bH[l*n+j]
					}
					acc += float64(av) * float64(bv)
				}
				c[i*n+j] = acc
			}
		}
		return c
	}

	al := func(n int) backend.Storage {
		s, err := adB.Alloc(n * 4)
		if err != nil {
			t.Fatalf("alloc: %v", err)
		}
		return s
	}

	runCase := func(name string, m, n, k int, transA, transB bool) {
		r := rand.New(rand.NewSource(int64(m*100 + n*10 + k)))
		aSize := m * k
		bSize := k * n
		aH := make([]float32, aSize)
		bH := make([]float32, bSize)
		for i := range aH {
			aH[i] = float32(r.NormFloat64()) * 0.5
		}
		for i := range bH {
			bH[i] = float32(r.NormFloat64()) * 0.5
		}
		A := al(aSize)
		defer adB.Free(A)
		B := al(bSize)
		defer adB.Free(B)
		C := al(m * n)
		defer adB.Free(C)
		if _, err := uploadInto(adB, A, f32ToBytes(aH)); err != nil {
			t.Fatalf("%s upload A: %v", name, err)
		}
		if _, err := uploadInto(adB, B, f32ToBytes(bH)); err != nil {
			t.Fatalf("%s upload B: %v", name, err)
		}
		if err := matmulPlainT(adB, C, A, B, m, n, k, transA, transB); err != nil {
			t.Fatalf("%s matmulPlainT: %v", name, err)
		}
		if s, ok := adB.(interface{ Sync() error }); ok {
			s.Sync()
		}
		cGpu := gpuToHost(adB, C, m*n)
		cRef := cpuRef(aH, bH, m, n, k, transA, transB)
		var maxAbs, maxRel float64
		var arg int
		for i := range cGpu {
			d := math.Abs(float64(cGpu[i]) - cRef[i])
			if d > maxAbs {
				maxAbs = d
				arg = i
			}
			den := math.Abs(cRef[i])
			if den < 1e-8 {
				den = 1e-8
			}
			r := d / den
			if r > maxRel {
				maxRel = r
			}
		}
		verdict := "PASS"
		// Mixed floor: F32 precision limit ~ 2e-7 rel per cell (single matmul).
		// For prod shapes K=32 accumulator, effective floor ~ sqrt(32)*eps ≈ 6.7e-7 rel.
		// Absolute floor 2e-6 (F32 chain).
		if maxRel > 1e-4 && maxAbs > 2e-6 {
			verdict = "FAIL"
			t.Errorf("%s: maxAbs=%.3e maxRel=%.3e (GPU[%d]=%+.6e vs F64=%+.6e)", name, maxAbs, maxRel, arg, cGpu[arg], cRef[arg])
		} else {
			t.Logf("%s [m=%d n=%d k=%d tA=%v tB=%v]: maxAbs=%.3e maxRel=%.3e %s", name, m, n, k, transA, transB, maxAbs, maxRel, verdict)
		}
	}
	// small shapes  x  all 4 flag combos
	shapes := [][3]int{{3, 4, 2}, {5, 3, 7}, {8, 8, 8}}
	for _, sh := range shapes {
		m, n, k := sh[0], sh[1], sh[2]
		runCase("FF", m, n, k, false, false)
		runCase("TF", m, n, k, true, false)
		runCase("FT", m, n, k, false, true)
		runCase("TT", m, n, k, true, true)
	}
	// Production shapes от cert path
	prodShapes := []struct {
		name        string
		m, n, k     int
		transA, transB bool
	}{
		{"dWq(TF D,D,M)", 128, 128, 32, true, false},
		{"dW2(TF FFN,D,M)", 128, 128, 32, true, false},
		{"dW1(TF D,FFN,M)", 128, 128, 32, true, false},
		{"dWo(TF D,D,M)", 128, 128, 32, true, false},
		{"dWout(TF D,V,M)", 128, 32, 32, true, false},
	}
	for _, sh := range prodShapes {
		runCase(sh.name, sh.m, sh.n, sh.k, sh.transA, sh.transB)
	}
}
