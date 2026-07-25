package cuda

// A-LLM-1 G2 (2026-07-25): backward .so smoke + fingerprint gate + D-сверка.
//
// НЕ встраиваем в trainStep (это следующее звено). Только доказать:
//   1) .so собран корректно
//   2) fingerprints kernel'ов совпадают с v0.2.0 cert reference
//   3) contracts работают: D-precompute даёт правильный D на малой форме
//   4) canonical chain (bh=128, sl=8192, hd=128) запускается без ошибок с L от fa_forward_train

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/djeday123/goml/backend"
)

// TestFABwd_Fingerprints — cert reference numRegs check.
// v0.2.0 tests/bench_r2c_e2e.cu:67-73:
//   d_precompute=38, merged_v1=252, dk_new=124, dq_new=69
// Расхождение хоть на единицу = STOP.
func TestFABwd_Fingerprints(t *testing.T) {
	if err := FABwdLoad(); err != nil {
		t.Skipf("libfa_bwd_sm120.so unavailable: %v", err)
	}
	// Ensure CUDA primary context initialized.
	gomlB, err := backend.Get(backend.CUDA)
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	if s, err := gomlB.Alloc(4); err == nil {
		gomlB.Free(s)
	}

	kernels := []struct {
		id       int
		name     string
		expected int
	}{
		{0, "kernel_d_precompute", 38},
		{1, "kernel_merged_v1", 252},
		{2, "kernel_dk_new", 124},
		{3, "kernel_dq_new", 69},
	}

	t.Log("=== Fingerprint gate (numRegs vs v0.2.0 cert reference) ===")
	t.Logf("%-24s | expected | actual | verdict", "kernel")
	allPass := true
	for _, k := range kernels {
		regs, err := FABwdKernelRegs(k.id)
		if err != nil {
			t.Errorf("%s: cudaFuncGetAttributes failed: %v", k.name, err)
			allPass = false
			continue
		}
		verdict := "OK"
		if regs != k.expected {
			verdict = "MISMATCH"
			allPass = false
		}
		t.Logf("%-24s | %8d | %6d | %s", k.name, k.expected, regs, verdict)
	}
	if !allPass {
		t.Errorf("Fingerprint gate FAILED — nvcc/CUDA/flags mismatch vs cert reference")
	}
}

// TestFABwd_DPrecompute — D-сверка на малой форме.
// D[bh_i, s_i] = sum_d(dO[bh_i, s_i, d] * O[bh_i, s_i, d]) over hd=128.
// Малая форма: bh=1, sl=128, hd=128.
func TestFABwd_DPrecompute(t *testing.T) {
	if err := FABwdLoad(); err != nil {
		t.Skipf("libfa_bwd_sm120.so unavailable: %v", err)
	}
	gomlB, err := backend.Get(backend.CUDA)
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	if s, err := gomlB.Alloc(4); err == nil {
		gomlB.Free(s)
	}

	const (
		bh = 1
		sl = 128
		hd = 128
	)

	// Random F32 values, then encode as FP16 for kernel inputs.
	r := rand.New(rand.NewSource(42))
	nElem := bh * sl * hd
	oF32 := make([]float32, nElem)
	doF32 := make([]float32, nElem)
	for i := range oF32 {
		oF32[i] = float32(r.NormFloat64() * 0.5)
		doF32[i] = float32(r.NormFloat64() * 0.3)
	}

	// Encode F32 → FP16 (little-endian uint16 buffers).
	oFP16 := f32ToFP16Bytes(oF32)
	doFP16 := f32ToFP16Bytes(doF32)

	b, _ := backend.Get(backend.CUDA)
	oGPU, _ := b.ToDevice(backend.CUDADevice(0), &_cpuStorage{data: oFP16})
	defer b.Free(oGPU)
	doGPU, _ := b.ToDevice(backend.CUDADevice(0), &_cpuStorage{data: doFP16})
	defer b.Free(doGPU)
	dGPU, _ := b.Alloc(bh * sl * 4) // F32
	defer b.Free(dGPU)

	if err := FABwdDPrecompute(_ptr(oGPU), _ptr(doGPU), _ptr(dGPU), bh, sl, hd, 0); err != nil {
		t.Fatal(err)
	}
	if s, ok := b.(interface{ Sync() error }); ok {
		s.Sync()
	}

	dGPU_host := _download(b, dGPU, bh*sl)

	// CPU F32 reference (dequantize FP16 to F32 first — same round-trip as kernel).
	oF32Round := fp16BytesToF32(oFP16)
	doF32Round := fp16BytesToF32(doFP16)
	dRef := make([]float32, bh*sl)
	for i := 0; i < bh*sl; i++ {
		var s float32
		for d := 0; d < hd; d++ {
			s += doF32Round[i*hd+d] * oF32Round[i*hd+d]
		}
		dRef[i] = s
	}

	var maxAbs, maxRel float32
	for i := range dGPU_host {
		d := float32(math.Abs(float64(dGPU_host[i] - dRef[i])))
		mag := float32(math.Abs(float64(dRef[i])))
		r := d / (mag + 1e-30)
		if d > maxAbs {
			maxAbs = d
		}
		if r > maxRel {
			maxRel = r
		}
	}
	t.Logf("D-precompute smoke (bh=%d, sl=%d, hd=%d): worst |D_gpu-D_ref|=%.3e (rel=%.3e)",
		bh, sl, hd, maxAbs, maxRel)
	// FP16 round + F32 accum reduction, sqrt(hd)·eps ≈ 6.8e-7 fp32 term dominated by FP16 error.
	// Floor: 5e-3 abs (FP16 accumulation class).
	if maxAbs > 5e-3 {
		t.Errorf("D worst abs %.3e > floor 5e-3", maxAbs)
	}
	for i := 0; i < 3; i++ {
		t.Logf("  D[%d]=%.6f (ref=%.6f)", i, dGPU_host[i], dRef[i])
	}
}

// TestFABwd_CanonicalChain — canonical form full chain smoke.
// bh=128, sl=8192, hd=128 (v0.2.0 cert form).
// L from fa_forward_train — первая живая стыковка G1-выхода с G2-входом!
// Критерий: запуск без ошибок + все выходы (dV, dK, dQ) non-NaN + non-Inf.
func TestFABwd_CanonicalChain(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode — canonical form allocates ~700 MB, takes ~50 ms compute")
	}
	if err := FABwdLoad(); err != nil {
		t.Skipf("libfa_bwd_sm120.so unavailable: %v", err)
	}
	if err := FALoad(); err != nil {
		t.Skipf("libfa_sm120.so unavailable: %v", err)
	}
	gomlB, err := backend.Get(backend.CUDA)
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	if s, err := gomlB.Alloc(4); err == nil {
		gomlB.Free(s)
	}
	faCtx, err := FACreate()
	if err != nil {
		t.Fatalf("FACreate: %v", err)
	}
	defer faCtx.Destroy()

	const (
		bh     = 128
		sl     = 8192
		hd     = 128
		causal = 0
		window = 0
	)
	strideDs := (sl + 15) & ^15
	softmaxScale := float32(1.0 / math.Sqrt(float64(hd)))

	nElem := bh * sl * hd

	// Prepare inputs: Q,K,V constant 1.0 (FP8 0x38).
	q := make([]byte, nElem)
	k := make([]byte, nElem)
	v := make([]byte, nElem)
	fp8One := byte(0x38)
	for i := range q {
		q[i] = fp8One
		k[i] = fp8One
		v[i] = fp8One
	}
	// dO: random FP16.
	r := rand.New(rand.NewSource(123))
	doF32 := make([]float32, nElem)
	for i := range doF32 {
		doF32[i] = float32(r.NormFloat64() * 0.1)
	}
	doFP16 := f32ToFP16Bytes(doF32)

	b, _ := backend.Get(backend.CUDA)
	qGPU, _ := b.ToDevice(backend.CUDADevice(0), &_cpuStorage{data: q})
	defer b.Free(qGPU)
	kGPU, _ := b.ToDevice(backend.CUDADevice(0), &_cpuStorage{data: k})
	defer b.Free(kGPU)
	vGPU, _ := b.ToDevice(backend.CUDADevice(0), &_cpuStorage{data: v})
	defer b.Free(vGPU)
	oGPU, _ := b.Alloc(nElem * 2)
	defer b.Free(oGPU)
	lGPU, _ := b.Alloc(bh * sl * 4)
	defer b.Free(lGPU)
	doGPU, _ := b.ToDevice(backend.CUDADevice(0), &_cpuStorage{data: doFP16})
	defer b.Free(doGPU)
	dGPU, _ := b.Alloc(bh * sl * 4)
	defer b.Free(dGPU)
	dSnatGPU, _ := b.Alloc(bh * sl * strideDs) // FP8 padded
	defer b.Free(dSnatGPU)
	dSTGPU, _ := b.Alloc(bh * sl * strideDs)
	defer b.Free(dSTGPU)
	dVGPU, _ := b.Alloc(nElem * 4) // F32 [bh, sl, hd]
	defer b.Free(dVGPU)
	dKGPU, _ := b.Alloc(nElem * 4)
	defer b.Free(dKGPU)
	dQGPU, _ := b.Alloc(nElem * 4)
	defer b.Free(dQGPU)

	// Zero-init dV, dK, dQ (kernel contract).
	zeroGPU := func(s backend.Storage, bytes int) {
		zeros := make([]byte, bytes)
		zGPU, _ := b.ToDevice(backend.CUDADevice(0), &_cpuStorage{data: zeros})
		// Copy to target buffer? Simpler: overwrite in place via Copy.
		_ = b.Copy(s, zGPU, bytes)
		b.Free(zGPU)
	}
	zeroGPU(dVGPU, nElem*4)
	zeroGPU(dKGPU, nElem*4)
	zeroGPU(dQGPU, nElem*4)

	t.Log("=== A-LLM-1 G2 canonical chain smoke ===")
	t.Logf("Form: bh=%d, sl=%d, hd=%d, causal=%d", bh, sl, hd, causal)

	// Step 1: G1 fa_forward_train → L (первая живая стыковка G1-выхода с G2-входом!)
	if err := faCtx.ForwardTrain(_ptr(qGPU), _ptr(kGPU), _ptr(vGPU),
		_ptr(oGPU), _ptr(lGPU),
		bh, sl, hd, causal, window, softmaxScale, 0); err != nil {
		t.Fatalf("fa_forward_train: %v", err)
	}
	if s, ok := b.(interface{ Sync() error }); ok {
		s.Sync()
	}
	t.Log("  [1/4] fa_forward_train PASS -- L produced (G1→G2 live stitch)")

	// Step 2: D-precompute
	if err := FABwdDPrecompute(_ptr(oGPU), _ptr(doGPU), _ptr(dGPU), bh, sl, hd, 0); err != nil {
		t.Fatalf("d_precompute: %v", err)
	}
	if s, ok := b.(interface{ Sync() error }); ok {
		s.Sync()
	}
	t.Log("  [2/4] gt_fa_bwd_d_precompute PASS")

	// Step 3: merged (ds_gen + dV_p1)
	if err := FABwdMerged(_ptr(qGPU), _ptr(kGPU), _ptr(vGPU),
		_ptr(doGPU), _ptr(lGPU), _ptr(dGPU),
		_ptr(dSnatGPU), _ptr(dSTGPU), _ptr(dVGPU),
		bh, sl, hd, causal, window, softmaxScale, 0); err != nil {
		t.Fatalf("merged: %v", err)
	}
	if s, ok := b.(interface{ Sync() error }); ok {
		s.Sync()
	}
	t.Log("  [3a/4] gt_fa_bwd_merged PASS (dV + dS_nat + dS_T)")

	// Step 4a: dk_new
	if err := FABwdDK(_ptr(qGPU), _ptr(dSTGPU), _ptr(dKGPU),
		bh, sl, hd, causal, window, softmaxScale, 0); err != nil {
		t.Fatalf("dk_new: %v", err)
	}
	if s, ok := b.(interface{ Sync() error }); ok {
		s.Sync()
	}
	t.Log("  [3b/4] gt_fa_bwd_dk PASS")

	// Step 4b: dq_new
	if err := FABwdDQ(_ptr(kGPU), _ptr(dSnatGPU), _ptr(dQGPU),
		bh, sl, hd, causal, window, softmaxScale, 0); err != nil {
		t.Fatalf("dq_new: %v", err)
	}
	if s, ok := b.(interface{ Sync() error }); ok {
		s.Sync()
	}
	t.Log("  [4/4] gt_fa_bwd_dq PASS")

	// Non-NaN / non-Inf check on ends of each output (sample ~1000 elements).
	checkClean := func(name string, s backend.Storage, nSample int) {
		host := _download(b, s, nSample)
		nanCount := 0
		infCount := 0
		var minV, maxV float32
		minV, maxV = math.MaxFloat32, -math.MaxFloat32
		for _, v := range host {
			if math.IsNaN(float64(v)) {
				nanCount++
			}
			if math.IsInf(float64(v), 0) {
				infCount++
			}
			if v < minV {
				minV = v
			}
			if v > maxV {
				maxV = v
			}
		}
		t.Logf("  %s [sample %d]: min=%.3e max=%.3e NaN=%d Inf=%d", name, nSample, minV, maxV, nanCount, infCount)
		if nanCount > 0 {
			t.Errorf("%s has %d NaN values in first %d", name, nanCount, nSample)
		}
		if infCount > 0 {
			t.Errorf("%s has %d Inf values in first %d", name, infCount, nSample)
		}
	}
	checkClean("dV", dVGPU, 1000)
	checkClean("dK", dKGPU, 1000)
	checkClean("dQ", dQGPU, 1000)
	checkClean("D",  dGPU,  bh)
	checkClean("L",  lGPU,  bh)

	t.Log("=== canonical chain smoke PASS ===")
}

// -----------------------------------------------------------------------------
// FP16 encoding helpers (test-only, IEEE 754 half-precision little-endian).
// -----------------------------------------------------------------------------

func f32ToFP16Bytes(a []float32) []byte {
	out := make([]byte, len(a)*2)
	for i, v := range a {
		h := f32ToFP16(v)
		out[i*2] = byte(h)
		out[i*2+1] = byte(h >> 8)
	}
	return out
}

func fp16BytesToF32(b []byte) []float32 {
	out := make([]float32, len(b)/2)
	for i := range out {
		h := uint16(b[i*2]) | uint16(b[i*2+1])<<8
		out[i] = fp16ToF32(h)
	}
	return out
}

// f32ToFP16: IEEE 754 binary16, round-to-nearest-even.
func f32ToFP16(v float32) uint16 {
	bits := math.Float32bits(v)
	sign := uint16((bits >> 16) & 0x8000)
	expF := int32((bits >> 23) & 0xFF)
	mantF := bits & 0x7FFFFF

	if expF == 0xFF {
		// Inf / NaN
		if mantF != 0 {
			return sign | 0x7E00 // quiet NaN
		}
		return sign | 0x7C00 // Inf
	}
	if expF == 0 {
		return sign // signed zero (subnormal round-to-zero)
	}
	newExp := expF - 127 + 15
	if newExp >= 0x1F {
		return sign | 0x7C00 // overflow → Inf
	}
	if newExp <= 0 {
		// subnormal
		if newExp < -10 {
			return sign
		}
		mantF |= 0x800000 // implicit leading 1
		shift := uint32(14 - newExp)
		mant16 := uint16(mantF >> shift)
		// round-to-nearest-even
		roundBit := (mantF >> (shift - 1)) & 1
		mant16 += uint16(roundBit)
		return sign | mant16
	}
	mant16 := uint16(mantF >> 13)
	// round-to-nearest-even
	roundBit := (mantF >> 12) & 1
	if roundBit != 0 {
		lowBits := mantF & 0x1FFF
		if lowBits != 0x1000 || (mant16&1) != 0 {
			mant16++
			if mant16 == 0x400 {
				mant16 = 0
				newExp++
				if newExp >= 0x1F {
					return sign | 0x7C00
				}
			}
		}
	}
	return sign | uint16(newExp<<10) | mant16
}

func fp16ToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h & 0x3FF)
	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// subnormal
		e := int32(-14)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		mant &= 0x3FF
		return math.Float32frombits(sign | uint32((e+127)<<23) | (mant << 13))
	}
	if exp == 0x1F {
		if mant == 0 {
			return math.Float32frombits(sign | 0x7F800000)
		}
		return math.Float32frombits(sign | 0x7FC00000)
	}
	return math.Float32frombits(sign | uint32((int32(exp)-15+127)<<23) | (mant << 13))
}

// unused guard
var _ = unsafe.Sizeof(0)
