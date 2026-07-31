package abjexam

// Ход-D standalone RMSNormGradF32 probe: CPU-F64 reference vs GPU kernel.
// Форма D=128 M=32 - точно как в cert BwdCertF32_MultiLayer.
// F32-precision floor 1e-6 rel. Расхождение = kernel bug на форме.
// Совпадение = зло в chain wiring bs.DX ниже.

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

func TestRMSNormGrad_CPUF64_Arbiter(t *testing.T) {
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

	const M = 32
	const D = 128
	const eps32 float32 = 1e-5

	r := rand.New(rand.NewSource(31))
	xH := make([]float32, M*D)
	gammaH := make([]float32, D)
	dyH := make([]float32, M*D)
	// Match the scale seen in real bwd: |x|~1-3 (post-residual sum),
	// |gamma|~0.02 (fresh init), |dy|~2e-3.
	for i := range xH {
		xH[i] = float32(r.NormFloat64()) * 1.5
	}
	for i := range gammaH {
		gammaH[i] = float32(r.NormFloat64()) * 0.02
	}
	for i := range dyH {
		dyH[i] = float32(r.NormFloat64()) * 2e-3
	}

	al := func(nBytes int) backend.Storage {
		s, err := adB.Alloc(nBytes)
		if err != nil {
			t.Fatalf("alloc: %v", err)
		}
		return s
	}
	X := al(M * D * 4)
	defer adB.Free(X)
	Gamma := al(D * 4)
	defer adB.Free(Gamma)
	DY := al(M * D * 4)
	defer adB.Free(DY)
	DX := al(M * D * 4)
	defer adB.Free(DX)
	DGamma := al(D * 4)
	defer adB.Free(DGamma)

	if _, err := uploadInto(adB, X, f32ToBytes(xH)); err != nil {
		t.Fatalf("upload X: %v", err)
	}
	if _, err := uploadInto(adB, Gamma, f32ToBytes(gammaH)); err != nil {
		t.Fatalf("upload Gamma: %v", err)
	}
	if _, err := uploadInto(adB, DY, f32ToBytes(dyH)); err != nil {
		t.Fatalf("upload DY: %v", err)
	}
	// dgamma должен быть pre-zeroed (atomicAdd across rows).
	zeroD := make([]byte, D*4)
	if _, err := uploadInto(adB, DGamma, zeroD); err != nil {
		t.Fatalf("upload DGamma zero: %v", err)
	}

	if err := gtB.RMSNormGradF32(X, Gamma, DY, DX, DGamma, M, D, eps32); err != nil {
		t.Fatalf("RMSNormGradF32: %v", err)
	}
	if s, ok := adB.(interface{ Sync() error }); ok {
		s.Sync()
	}
	dxGPU := gpuToHost(adB, DX, M*D)
	dgammaGPU := gpuToHost(adB, DGamma, D)

	// CPU F64 reference. Formula:
	//   ms = sum(x^2)/D + eps
	//   rms = sqrt(ms)
	//   S_row = sum_i(gamma_i * x_i * dy_i)
	//   dx_j = gamma_j * dy_j / rms - x_j * S_row / (D * rms^3)
	//   dgamma_i += dy_i * x_i / rms  (accumulate across rows)
	dxRef := make([]float64, M*D)
	dgammaRef := make([]float64, D)
	for row := 0; row < M; row++ {
		var sumX2 float64
		for j := 0; j < D; j++ {
			x := float64(xH[row*D+j])
			sumX2 += x * x
		}
		ms := sumX2/float64(D) + float64(eps32)
		rms := math.Sqrt(ms)
		var Srow float64
		for j := 0; j < D; j++ {
			Srow += float64(gammaH[j]) * float64(xH[row*D+j]) * float64(dyH[row*D+j])
		}
		invRms := 1.0 / rms
		invRms3ByD := invRms * invRms * invRms / float64(D)
		for j := 0; j < D; j++ {
			g := float64(gammaH[j])
			x := float64(xH[row*D+j])
			dy := float64(dyH[row*D+j])
			dxRef[row*D+j] = g*dy*invRms - x*Srow*invRms3ByD
			dgammaRef[j] += dy * x * invRms
		}
	}

	// Compare.
	var maxAbsDx, maxRelDx float64
	var argMaxDx int
	for i := range dxRef {
		gpu := float64(dxGPU[i])
		ref := dxRef[i]
		diff := math.Abs(gpu - ref)
		den := math.Abs(ref)
		if den < 1e-8 {
			den = 1e-8
		}
		rel := diff / den
		if diff > maxAbsDx {
			maxAbsDx = diff
			argMaxDx = i
		}
		if rel > maxRelDx {
			maxRelDx = rel
		}
	}
	var maxAbsDg, maxRelDg float64
	for i := range dgammaRef {
		gpu := float64(dgammaGPU[i])
		ref := dgammaRef[i]
		diff := math.Abs(gpu - ref)
		den := math.Abs(ref)
		if den < 1e-8 {
			den = 1e-8
		}
		rel := diff / den
		if diff > maxAbsDg {
			maxAbsDg = diff
		}
		if rel > maxRelDg {
			maxRelDg = rel
		}
	}
	var xMax, gMax, dyMax float32
	for _, v := range xH {
		a := v
		if a < 0 {
			a = -a
		}
		if a > xMax {
			xMax = a
		}
	}
	for _, v := range gammaH {
		a := v
		if a < 0 {
			a = -a
		}
		if a > gMax {
			gMax = a
		}
	}
	for _, v := range dyH {
		a := v
		if a < 0 {
			a = -a
		}
		if a > dyMax {
			dyMax = a
		}
	}
	t.Logf("Inputs: |x|max=%.3e |gamma|max=%.3e |dy|max=%.3e", xMax, gMax, dyMax)
	t.Logf("dX max cell: idx=%d GPU=%+.6e CPU_F64=%+.6e diff=%.3e rel=%.3e",
		argMaxDx, dxGPU[argMaxDx], dxRef[argMaxDx], maxAbsDx, maxRelDx)
	t.Logf("dGamma summary: absMax diff=%.3e relMax=%.3e", maxAbsDg, maxRelDg)

	const floorRel = 1e-4 // F32-класс с rsqrt.approx
	if maxRelDx > floorRel {
		t.Errorf("dX GPU vs CPU-F64: max rel diff=%.3e > floor %.0e", maxRelDx, floorRel)
	}
	if maxRelDg > floorRel {
		t.Errorf("dGamma GPU vs CPU-F64: max rel diff=%.3e > floor %.0e", maxRelDg, floorRel)
	}
	if maxRelDx <= floorRel && maxRelDg <= floorRel {
		t.Logf("VERDICT: RMSNormGradF32 kernel CLEAN на форме D=%d M=%d", D, M)
	}
}
