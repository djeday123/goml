package abjexam

// A-LLM-2 GEMM zero repro (2026-07-27).
// Изоляция бага MatMulF32Ex(dS[32x32], K[32x128], dQ[32x128]) -> exact zero.
// Гипотеза 1 (row-major swap edge case) + 2 (паттерн-777 tell-tale).

import (
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/djeday123/goml/backend"
	_ "github.com/djeday123/goml/backend/cpu"
	_ "github.com/djeday123/goml/backend/cuda"
	adapter "github.com/djeday123/goml/backend/gotorch"
)

func TestALLM_MatMulF32Ex_ZeroRepro(t *testing.T) {
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
	gtB := adB.(*adapter.Backend)

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	al := func(nBytes int) backend.Storage {
		s, err := adB.Alloc(nBytes)
		if err != nil {
			t.Fatalf("alloc: %v", err)
		}
		return s
	}

	// Params exactly как в attnReconstructBwd's dQ = dS @ K matmul.
	// m = S = 32, n = HD = 128, k = S = 32.
	// A = dS [m, k] = [32, 32]; B = K [k, n] = [32, 128]; C = dQ [m, n] = [32, 128].
	// transA = false, transB = false. Row-major.

	testCase := func(name string, m, n, k int, transA, transB bool) {
		mkN := m * k
		nkN := k * n
		mnN := m * n

		A := al(mkN * 4)
		defer adB.Free(A)
		B := al(nkN * 4)
		defer adB.Free(B)
		C := al(mnN * 4)
		defer adB.Free(C)

		// Random входные A, B (небольшой scale чтобы не выйти за FP32).
		r := rand.New(rand.NewSource(7))
		aH := make([]float32, mkN)
		bH := make([]float32, nkN)
		for i := range aH {
			aH[i] = float32(r.NormFloat64()) * 0.1
		}
		for i := range bH {
			bH[i] = float32(r.NormFloat64()) * 0.1
		}
		if _, err := uploadInto(adB, A, f32ToBytes(aH)); err != nil {
			t.Fatalf("upload A: %v", err)
		}
		if _, err := uploadInto(adB, B, f32ToBytes(bH)); err != nil {
			t.Fatalf("upload B: %v", err)
		}

		// Pattern-777 в C ДО вызова.
		cH777 := make([]float32, mnN)
		for i := range cH777 {
			cH777[i] = 777.0
		}
		if _, err := uploadInto(adB, C, f32ToBytes(cH777)); err != nil {
			t.Fatalf("upload C777: %v", err)
		}

		// Verify roundtrip: read back C, should be 777.
		cVerify := gpuToHost(adB, C, mnN)
		if cVerify[0] != 777.0 {
			t.Fatalf("%s: pre-call C[0]=%v (expected 777), upload/read broken", name, cVerify[0])
		}

		// Call MatMulF32Ex.
		if err := gtB.MatMulF32Ex(A, B, C, m, n, k, transA, transB); err != nil {
			t.Fatalf("%s: MatMulF32Ex: %v", name, err)
		}
		if s, ok := adB.(interface{ Sync() error }); ok {
			s.Sync()
		}

		// Read C после вызова.
		cH := gpuToHost(adB, C, mnN)
		var maxAbs, minAbs float32 = 0, math.MaxFloat32
		var zeroCount, seven77Count, otherCount int
		for _, v := range cH {
			a := v
			if a < 0 {
				a = -a
			}
			if v == 0 {
				zeroCount++
			} else if v == 777.0 {
				seven77Count++
			} else {
				otherCount++
			}
			if a > maxAbs {
				maxAbs = a
			}
			if a < minAbs {
				minAbs = a
			}
		}

		// CPU reference computation.
		var cpuMax float32
		refCheck := 3
		for row := 0; row < refCheck && row < m; row++ {
			for col := 0; col < refCheck && col < n; col++ {
				var acc float64
				for kk := 0; kk < k; kk++ {
					var av, bv float32
					if transA {
						av = aH[kk*m+row]
					} else {
						av = aH[row*k+kk]
					}
					if transB {
						bv = bH[col*k+kk]
					} else {
						bv = bH[kk*n+col]
					}
					acc += float64(av) * float64(bv)
				}
				if a := float32(math.Abs(acc)); a > cpuMax {
					cpuMax = a
				}
			}
		}

		verdict := "UNKNOWN"
		if seven77Count == mnN {
			verdict = "DID NOT WRITE (777 preserved everywhere)"
		} else if zeroCount == mnN {
			verdict = "WROTE ZEROS (all 0)"
		} else if otherCount > 0 && zeroCount < mnN/2 {
			verdict = "WROTE VALUES (correct-ish)"
		} else if zeroCount > 0 && seven77Count > 0 {
			verdict = "PARTIAL WRITE"
		}

		t.Logf("%s: [m=%d n=%d k=%d transA=%v transB=%v]", name, m, n, k, transA, transB)
		t.Logf("  C after: max|.|=%.3e, zeros=%d, 777s=%d, others=%d (of %d)", maxAbs, zeroCount, seven77Count, otherCount, mnN)
		t.Logf("  CPU-ref sample max = %.3e", cpuMax)
		t.Logf("  VERDICT: %s", verdict)
	}

	// Test A: square 32x32x32 — если работает, layout OK на squares.
	testCase("A: square 32x32x32 notrans", 32, 32, 32, false, false)
	// Test B: rectangle 32x128x32 (the failing case) — same as dS @ K.
	testCase("B: rect 32x128x32 notrans", 32, 128, 32, false, false)
	// Test C: rectangle 128x32x32 (dim-swapped) — checks m/n symmetry.
	testCase("C: rect 128x32x32 notrans", 128, 32, 32, false, false)
	// Test D: rectangle 32x128x32 transA — dS^T @ K.
	testCase("D: rect 32x128x32 transA=true", 32, 128, 32, true, false)
	// Test E: rectangle 32x128x32 transB.
	testCase("E: rect 32x128x32 transB=true", 32, 128, 32, false, true)
}

// Ход-B изоляция класса бага: input scale=1e-3 (магнитуда живого softmax_bwd output).
// Если MatMulF32Ex(dS_like, K_like, dQ_like, 32, 128, 32, F, F) на scale 1e-3 → EXACT ZERO
// при том что на scale 0.1 (previous tests) даёт корректный ответ, magnitude-triggered
// flush-to-zero в gt_gemm_ex/cublasGemmEx на sm_120a изолирован standalone.
func TestALLM_MatMulF32Ex_SmallMagnitude(t *testing.T) {
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
	gtB := adB.(*adapter.Backend)
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	al := func(nBytes int) backend.Storage {
		s, err := adB.Alloc(nBytes)
		if err != nil {
			t.Fatalf("alloc: %v", err)
		}
		return s
	}
	probeMag := func(name string, scale float32) {
		const m, n, k = 32, 128, 32
		A := al(m * k * 4)
		defer adB.Free(A)
		B := al(k * n * 4)
		defer adB.Free(B)
		C := al(m * n * 4)
		defer adB.Free(C)
		r := rand.New(rand.NewSource(7))
		aH := make([]float32, m*k)
		bH := make([]float32, k*n)
		// A: sparse-like distribution — most cells near-zero, few outliers к max=scale
		// (имитирует softmax_bwd output: dS = P*(dP - row-mean(P*dP))).
		for i := range aH {
			v := float32(r.NormFloat64()) * 0.1
			// squeeze range: 90% cells at scale*1e-3, 10% at scale.
			if r.Float64() < 0.9 {
				v *= 0.001
			}
			aH[i] = v * scale * 10.0 // → 90% at scale*1e-2, 10% at scale*10
		}
		for i := range bH {
			bH[i] = float32(r.NormFloat64()) * 0.1 // K weights ~ 0.1
		}
		if _, err := uploadInto(adB, A, f32ToBytes(aH)); err != nil {
			t.Fatalf("%s upload A: %v", name, err)
		}
		if _, err := uploadInto(adB, B, f32ToBytes(bH)); err != nil {
			t.Fatalf("%s upload B: %v", name, err)
		}
		zeroC := make([]byte, m*n*4)
		if _, err := uploadInto(adB, C, zeroC); err != nil {
			t.Fatalf("%s zero C: %v", name, err)
		}

		if err := gtB.MatMulF32Ex(A, B, C, m, n, k, false, false); err != nil {
			t.Fatalf("%s MatMulF32Ex: %v", name, err)
		}
		if s, ok := adB.(interface{ Sync() error }); ok {
			s.Sync()
		}
		cH := gpuToHost(adB, C, m*n)
		var maxA, maxC float32
		for _, v := range aH {
			a := v
			if a < 0 {
				a = -a
			}
			if a > maxA {
				maxA = a
			}
		}
		for _, v := range cH {
			a := v
			if a < 0 {
				a = -a
			}
			if a > maxC {
				maxC = a
			}
		}
		// CPU reference (row 0).
		var cpuMax float64
		for col := 0; col < 4; col++ {
			var acc float64
			for kk := 0; kk < k; kk++ {
				acc += float64(aH[kk]) * float64(bH[kk*n+col])
			}
			if math.Abs(acc) > cpuMax {
				cpuMax = math.Abs(acc)
			}
		}
		verdict := "ALIVE"
		if maxC == 0 {
			verdict = "EXACT ZERO (FLUSH!)"
		}
		t.Logf("%s [scale=%.0e]: |A|max=%.3e |C_gpu|max=%.3e |C_cpu_ref_r0|=%.3e → %s",
			name, scale, maxA, maxC, cpuMax, verdict)
	}
	probeMag("scale=1e-0", 1e0)
	probeMag("scale=1e-1", 1e-1)
	probeMag("scale=1e-2", 1e-2)
	probeMag("scale=1e-3", 1e-3)
	probeMag("scale=1e-4", 1e-4)
	probeMag("scale=1e-5", 1e-5)
}

// FRESH tests: no pattern-777, C initialized to zero (like cudaMalloc default).
// Isolates "cublas doesn't work when C initial state is ZERO" hypothesis.
func TestALLM_MatMulF32Ex_FreshZeroInit(t *testing.T) {
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
	gtB := adB.(*adapter.Backend)

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	al := func(nBytes int) backend.Storage {
		s, err := adB.Alloc(nBytes)
		if err != nil {
			t.Fatalf("alloc: %v", err)
		}
		return s
	}

	freshCase := func(name string, m, n, k int) {
		mkN := m * k
		nkN := k * n
		mnN := m * n

		A := al(mkN * 4)
		defer adB.Free(A)
		B := al(nkN * 4)
		defer adB.Free(B)
		C := al(mnN * 4)
		defer adB.Free(C)

		r := rand.New(rand.NewSource(11))
		aH := make([]float32, mkN)
		bH := make([]float32, nkN)
		for i := range aH {
			aH[i] = float32(r.NormFloat64()) * 0.1
		}
		for i := range bH {
			bH[i] = float32(r.NormFloat64()) * 0.1
		}
		if _, err := uploadInto(adB, A, f32ToBytes(aH)); err != nil {
			t.Fatalf("upload A: %v", err)
		}
		if _, err := uploadInto(adB, B, f32ToBytes(bH)); err != nil {
			t.Fatalf("upload B: %v", err)
		}
		// EXPLICIT zero C init (uploadInto 0-bytes).
		zeroC := make([]byte, mnN*4)
		if _, err := uploadInto(adB, C, zeroC); err != nil {
			t.Fatalf("upload C zeros: %v", err)
		}

		// FIRST call.
		if err := gtB.MatMulF32Ex(A, B, C, m, n, k, false, false); err != nil {
			t.Fatalf("MatMulF32Ex 1st: %v", err)
		}
		if s, ok := adB.(interface{ Sync() error }); ok {
			s.Sync()
		}
		cH1 := gpuToHost(adB, C, mnN)
		var maxC1 float32
		for _, v := range cH1 {
			a := v
			if a < 0 {
				a = -a
			}
			if a > maxC1 {
				maxC1 = a
			}
		}

		// SECOND call (same inputs).
		if err := gtB.MatMulF32Ex(A, B, C, m, n, k, false, false); err != nil {
			t.Fatalf("MatMulF32Ex 2nd: %v", err)
		}
		if s, ok := adB.(interface{ Sync() error }); ok {
			s.Sync()
		}
		cH2 := gpuToHost(adB, C, mnN)
		var maxC2 float32
		for _, v := range cH2 {
			a := v
			if a < 0 {
				a = -a
			}
			if a > maxC2 {
				maxC2 = a
			}
		}

		verdict := "MATCH"
		if maxC1 == 0 && maxC2 > 0 {
			verdict = "FIRST-CALL-ZERO (WARMUP NEEDED!)"
		} else if maxC1 > 0 && maxC2 == 0 {
			verdict = "SECOND-CALL-ZERO (very weird)"
		} else if maxC1 == 0 && maxC2 == 0 {
			verdict = "BOTH-ZERO (matmul broken)"
		}

		t.Logf("%s [m=%d n=%d k=%d]: 1st |C|=%.3e, 2nd |C|=%.3e -- %s",
			name, m, n, k, maxC1, maxC2, verdict)
	}

	// Same shapes as pattern-777 but with ZERO C initial (like cudaMalloc default).
	freshCase("A: square 32x32x32 fresh-zero", 32, 32, 32)
	freshCase("B: rect 32x128x32 fresh-zero (the failing shape)", 32, 128, 32)
	freshCase("C: rect 128x32x32 fresh-zero", 128, 32, 32)
}
