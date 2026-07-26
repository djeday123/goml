package abjexam

// A-LLM-1 Stage 4 (2026-07-26): grad-consistency СЕРТИФИКАТ.
//
// Reference: CPU F64 attention forward+backward computed from the same math
// (Softmax + Matmul chain rule derivation). Chosen over finite-difference because:
//  1) FD accuracy limited by FP32 precision at small eps (~1e-4) and truncation
//     at large eps; CPU F64 gives 1-ULP reference at any scale.
//  2) FD requires many fwd calls per entry; CPU F64 is one-shot O(S^2·HD·BH).
// This is standard practice: compare implementation vs mathematical formula
// evaluated at higher precision (P5B / f64ref approach from R03b_impl5).
//
// Small form: BH=1, S=4, HD=8. Total 32 entries per tensor. Verifies both fwd
// (O) and bwd (dQ, dK, dV) match CPU F64 reference within FP32 accumulator floor.
//
// Floor: 1e-4 abs (sqrt(HD)·eps model for HD=8, ε=1.2e-7 gives ~3e-7 per-element
// GEMM noise; observed max drift ~ 6e-8 on tested seeds).

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

func TestALLM_AttnRecon_GradConsistency(t *testing.T) {
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

	const (
		BH    = 1
		S     = 4
		HD    = 8
		floor = 1e-4
	)
	scale := float32(1.0 / math.Sqrt(float64(HD)))
	sc64 := float64(scale)

	nQ := BH * S * HD
	nS := BH * S * S

	al := func(nElem int, sz int) backend.Storage {
		s, err := adB.Alloc(nElem * sz)
		if err != nil {
			t.Fatalf("alloc: %v", err)
		}
		return s
	}

	Q := al(nQ, 4)
	defer adB.Free(Q)
	K := al(nQ, 4)
	defer adB.Free(K)
	V := al(nQ, 4)
	defer adB.Free(V)
	O := al(nQ, 4)
	defer adB.Free(O)
	Sscratch := al(nS, 4)
	defer adB.Free(Sscratch)
	P := al(nS, 4)
	defer adB.Free(P)
	Qscaled := al(nQ, 4)
	defer adB.Free(Qscaled)
	dO := al(nQ, 4)
	defer adB.Free(dO)
	dQ := al(nQ, 4)
	defer adB.Free(dQ)
	dK := al(nQ, 4)
	defer adB.Free(dK)
	dV := al(nQ, 4)
	defer adB.Free(dV)
	dPtemp := al(nS, 4)
	defer adB.Free(dPtemp)
	dStemp := al(nS, 4)
	defer adB.Free(dStemp)

	r := rand.New(rand.NewSource(7))
	qH := make([]float32, nQ)
	kH := make([]float32, nQ)
	vH := make([]float32, nQ)
	doH := make([]float32, nQ)
	for i := 0; i < nQ; i++ {
		qH[i] = float32(r.NormFloat64()) * 0.3
		kH[i] = float32(r.NormFloat64()) * 0.3
		vH[i] = float32(r.NormFloat64()) * 0.3
		doH[i] = float32(r.NormFloat64()) * 0.3
	}

	upload := func(dst backend.Storage, host []float32) {
		if _, err := uploadInto(adB, dst, f32ToBytes(host)); err != nil {
			t.Fatalf("upload: %v", err)
		}
	}
	upload(Q, qH)
	upload(K, kH)
	upload(V, vH)
	upload(dO, doH)

	// -- GPU F32 fwd + bwd --
	if err := attnReconstructFwd(adB, gtB, Q, K, V, O, Sscratch, P, Qscaled, BH, S, HD, scale); err != nil {
		t.Fatalf("recon fwd: %v", err)
	}
	if err := attnReconstructBwd(adB, gtB, Q, K, V, P, dO, dQ, dK, dV, dPtemp, dStemp, BH, S, HD, scale); err != nil {
		t.Fatalf("recon bwd: %v", err)
	}
	if s, ok := adB.(interface{ Sync() error }); ok {
		s.Sync()
	}
	oGPU := gpuToHost(adB, O, nQ)
	dQGPU := gpuToHost(adB, dQ, nQ)
	dKGPU := gpuToHost(adB, dK, nQ)
	dVGPU := gpuToHost(adB, dV, nQ)

	// -- CPU F64 reference (same math, F64 precision) --
	SP := make([]float64, S*S)
	Ocpu := make([]float64, nQ)
	dVcpu := make([]float64, nQ)
	dPcpu := make([]float64, S*S)
	dScpu := make([]float64, S*S)
	dQcpu := make([]float64, nQ)
	dKcpu := make([]float64, nQ)
	// S = Q @ K^T * scale
	for i := 0; i < S; i++ {
		for j := 0; j < S; j++ {
			var acc float64
			for k := 0; k < HD; k++ {
				acc += float64(qH[i*HD+k]) * float64(kH[j*HD+k])
			}
			SP[i*S+j] = acc * sc64
		}
	}
	// P = softmax(S)
	for i := 0; i < S; i++ {
		maxV := SP[i*S]
		for j := 1; j < S; j++ {
			if SP[i*S+j] > maxV {
				maxV = SP[i*S+j]
			}
		}
		var sum float64
		for j := 0; j < S; j++ {
			SP[i*S+j] = math.Exp(SP[i*S+j] - maxV)
			sum += SP[i*S+j]
		}
		for j := 0; j < S; j++ {
			SP[i*S+j] /= sum
		}
	}
	// O = P @ V
	for i := 0; i < S; i++ {
		for d := 0; d < HD; d++ {
			var acc float64
			for j := 0; j < S; j++ {
				acc += SP[i*S+j] * float64(vH[j*HD+d])
			}
			Ocpu[i*HD+d] = acc
		}
	}
	// dV[j,d] = sum_i P[i,j] * dO[i,d]
	for j := 0; j < S; j++ {
		for d := 0; d < HD; d++ {
			var acc float64
			for i := 0; i < S; i++ {
				acc += SP[i*S+j] * float64(doH[i*HD+d])
			}
			dVcpu[j*HD+d] = acc
		}
	}
	// dP = dO @ V^T
	for i := 0; i < S; i++ {
		for j := 0; j < S; j++ {
			var acc float64
			for d := 0; d < HD; d++ {
				acc += float64(doH[i*HD+d]) * float64(vH[j*HD+d])
			}
			dPcpu[i*S+j] = acc
		}
	}
	// dS = P * (dP - rowsum(P*dP))
	for i := 0; i < S; i++ {
		var rs float64
		for j := 0; j < S; j++ {
			rs += SP[i*S+j] * dPcpu[i*S+j]
		}
		for j := 0; j < S; j++ {
			dScpu[i*S+j] = SP[i*S+j] * (dPcpu[i*S+j] - rs)
		}
	}
	// dQ = dS @ K * scale
	for i := 0; i < S; i++ {
		for k := 0; k < HD; k++ {
			var acc float64
			for j := 0; j < S; j++ {
				acc += dScpu[i*S+j] * float64(kH[j*HD+k])
			}
			dQcpu[i*HD+k] = acc * sc64
		}
	}
	// dK = dS^T @ Q * scale
	for j := 0; j < S; j++ {
		for k := 0; k < HD; k++ {
			var acc float64
			for i := 0; i < S; i++ {
				acc += dScpu[i*S+j] * float64(qH[i*HD+k])
			}
			dKcpu[j*HD+k] = acc * sc64
		}
	}

	// -- Compare GPU F32 vs CPU F64 --
	check := func(name string, gpu []float32, cpu []float64) {
		var maxAbs float32
		var maxRel float32
		var badIdx int = -1
		for i := 0; i < len(gpu); i++ {
			d := gpu[i] - float32(cpu[i])
			if d < 0 {
				d = -d
			}
			mag := float32(cpu[i])
			if mag < 0 {
				mag = -mag
			}
			if mag < 1e-6 {
				mag = 1e-6
			}
			rel := d / mag
			if d > maxAbs {
				maxAbs = d
				badIdx = i
			}
			if rel > maxRel {
				maxRel = rel
			}
		}
		if maxAbs > floor {
			t.Errorf("%s CERT FAIL: maxAbs=%.3e (floor %.1e), maxRel=%.3e, bad idx=%d gpu=%.6e cpu=%.6e",
				name, maxAbs, float32(floor), maxRel, badIdx, gpu[badIdx], cpu[badIdx])
		} else {
			t.Logf("%s CERT PASS: maxAbs=%.3e (floor %.1e), maxRel=%.3e",
				name, maxAbs, float32(floor), maxRel)
		}
	}
	check("O   (fwd)", oGPU, Ocpu)
	check("dV  (bwd)", dVGPU, dVcpu)
	check("dQ  (bwd)", dQGPU, dQcpu)
	check("dK  (bwd)", dKGPU, dKcpu)
}
