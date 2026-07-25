package cuda

// A-LLM-1 G1 (2026-07-25): L (LSE) correctness test для fa_forward_train.
//
// Тесты:
//   TestFAForwardTrain_Version — probe: library loads, version string OK.
//   TestFAForwardTrain_L_Uniform — bh=3, sl=280, hd=128 (asymmetric).
//                                  Q/K constant → L uniform predictable.
//                                  Проверяет: fa_forward_train launch success,
//                                  no CUDA errors, L values in valid range.
//   TestFAForwardTrain_L_Layout — bh=3, sl=280, hd=128 (asymmetric).
//                                 Q varies per row (2^bh_i) → L varies per row.
//                                 Проверяет: L раскладка [bh, sl] row-major,
//                                 off-by-one в страйде убил бы этот тест.
//
// FP8 e4m3 encoding (для конструирования Q,K вручную):
//   sign(1) + exp(4, bias=7) + mantissa(3). No inf. NaN = 0x7F/0xFF.
//   Value = 1.0  →  exp=7, m=0  →  0x38
//   Value = 2.0  →  exp=8, m=0  →  0x40
//   Value = 4.0  →  exp=9, m=0  →  0x48

import (
	"math"
	"os"
	"runtime"
	"testing"
	"unsafe"

	"github.com/djeday123/goml/backend"
)

func fp8E4M3FromFloat(v float32) byte {
	// Simple encoding for exact powers of 2 (used in test).
	// Not a full e4m3 encoder — covers only our test values (1, 2, 4, 8, 16).
	switch v {
	case 0.0:
		return 0x00
	case 1.0:
		return 0x38
	case 2.0:
		return 0x40
	case 4.0:
		return 0x48
	case 8.0:
		return 0x50
	case 16.0:
		return 0x58
	}
	panic("fp8E4M3FromFloat: unsupported test value")
}

func fp8E4M3ToFloat(b byte) float32 {
	// Decode E4M3 (bias=7). Handles subnormals & normals except NaN/max.
	if b == 0x00 || b == 0x80 {
		return 0.0
	}
	sign := float32(1.0)
	if b&0x80 != 0 {
		sign = -1.0
		b &= 0x7F
	}
	exp := int(b>>3) & 0x0F
	m := int(b & 0x07)
	if exp == 0 {
		// subnormal
		return sign * float32(m) / 8.0 * float32(math.Pow(2, -6))
	}
	return sign * (1.0 + float32(m)/8.0) * float32(math.Pow(2, float64(exp-7)))
}

// setupFAContext — helper: skip if libfa_sm120.so unavailable, else create context.
func setupFAContext(t *testing.T) *FAContext {
	t.Helper()
	if err := FALoad(); err != nil {
		t.Skipf("libfa_sm120.so unavailable: %v", err)
	}

	// Ensure a CUDA primary context is initialized (needed for FA cudaGetDevice/cudaGetDeviceProperties).
	// Use goml.cuda backend's ensureInit path.
	gomlB, err := backend.Get(backend.CUDA)
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	// Trigger init via harmless alloc.
	if s, err := gomlB.Alloc(4); err == nil {
		gomlB.Free(s)
	}

	ctx, err := FACreate()
	if err != nil {
		t.Fatalf("FACreate: %v", err)
	}
	t.Cleanup(func() { ctx.Destroy() })
	return ctx
}

func TestFAForwardTrain_Version(t *testing.T) {
	if err := FALoad(); err != nil {
		t.Skipf("libfa_sm120.so unavailable: %v", err)
	}
	v, err := FAVersion()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("libfa_sm120 version: %s", v)
	if v == "" {
		t.Errorf("empty version string")
	}
}

func TestFAForwardTrain_L_Uniform(t *testing.T) {
	ctx := setupFAContext(t)
	const (
		bh     = 3   // asymmetric batch·heads (не степень 2)
		sl     = 280 // asymmetric sequence length (не степень 2)
		hd     = 128
		causal = 0
		window = 0
	)
	softmaxScale := float32(1.0 / math.Sqrt(float64(hd)))

	// Q,K,V = 1.0 uniform → decoded fp8 = 1.0, dot = hd, uniform scores.
	// scale_Q = scale_K = 1.0 (no per-tensor rescaling needed, values already fp8-native 1.0).
	// user_scale = softmax_scale * 1 * 1 = softmax_scale = 1/sqrt(hd).
	nElem := bh * sl * hd
	q := make([]byte, nElem)
	k := make([]byte, nElem)
	v := make([]byte, nElem)
	fp8One := fp8E4M3FromFloat(1.0)
	for i := range q {
		q[i] = fp8One
		k[i] = fp8One
		v[i] = fp8One
	}

	b, _ := backend.Get(backend.CUDA)
	qGPU, _ := b.ToDevice(backend.CUDADevice(0), &_cpuStorage{data: q})
	defer b.Free(qGPU)
	kGPU, _ := b.ToDevice(backend.CUDADevice(0), &_cpuStorage{data: k})
	defer b.Free(kGPU)
	vGPU, _ := b.ToDevice(backend.CUDADevice(0), &_cpuStorage{data: v})
	defer b.Free(vGPU)
	oGPU, _ := b.Alloc(bh * sl * hd * 2) // FP16 output
	defer b.Free(oGPU)
	lGPU, _ := b.Alloc(bh * sl * 4) // F32 LSE [bh, sl]
	defer b.Free(lGPU)

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := ctx.ForwardTrain(
		_ptr(qGPU), _ptr(kGPU), _ptr(vGPU), _ptr(oGPU), _ptr(lGPU),
		bh, sl, hd, causal, window,
		softmaxScale, 0); err != nil {
		t.Fatal(err)
	}
	if s, ok := b.(interface{ Sync() error }); ok {
		s.Sync()
	}

	// Download L.
	lHost := _download(b, lGPU, bh*sl)

	// Reference: uniform scores = hd·1·1·softmax_scale = sqrt(hd) = 11.3137
	// m_i = sqrt(hd), sum_exp = sl (all exp(0)=1), L = sqrt(hd) + log(sl)
	lRef := float32(math.Sqrt(float64(hd)) + math.Log(float64(sl)))

	var maxDiff, maxRel float32
	for i, lv := range lHost {
		d := float32(math.Abs(float64(lv - lRef)))
		r := d / float32(math.Abs(float64(lRef)))
		if d > maxDiff {
			maxDiff = d
		}
		if r > maxRel {
			maxRel = r
		}
		if i < 3 {
			t.Logf("  L[%d]=%.6f (ref=%.6f, diff=%.3e)", i, lv, lRef, d)
		}
	}
	t.Logf("uniform L test: shape=[bh=%d, sl=%d], ref=%.6f, worst |L_gpu-L_ref|=%.3e (rel=%.3e)",
		bh, sl, lRef, maxDiff, maxRel)
	// FP8 e4m3 quantization noise + F32 accum + log/exp approx: floor 5e-3 rel.
	// Pre-registered: sqrt(hd)·eps · hd ~ 11.3·6e-8·128 = 8.7e-5 abs, ·f8 factor ~ 1e-3 abs.
	if maxRel > 5e-3 {
		t.Errorf("L worst rel %.3e > floor 5e-3", maxRel)
	}
}

func TestFAForwardTrain_L_Layout(t *testing.T) {
	ctx := setupFAContext(t)
	const (
		bh     = 3
		sl     = 280
		hd     = 128
		causal = 0
		window = 0
	)
	softmaxScale := float32(1.0 / math.Sqrt(float64(hd)))

	// Q[bh_i, :, :] = 2^bh_i (per-row variation) → L per row varies:
	//   bh_i=0: Q=1, dot_ij = hd, scores = sqrt(hd), L = sqrt(hd) + log(sl)
	//   bh_i=1: Q=2, dot_ij = 2*hd, scores = 2*sqrt(hd), L = 2*sqrt(hd) + log(sl)
	//   bh_i=2: Q=4, dot_ij = 4*hd, scores = 4*sqrt(hd), L = 4*sqrt(hd) + log(sl)
	// Detects [bh, sl] layout: off-by-one в страйде даст неверный L[bh_i·sl + s].
	// K,V=1.
	nElem := bh * sl * hd
	q := make([]byte, nElem)
	k := make([]byte, nElem)
	v := make([]byte, nElem)
	for i := 0; i < bh; i++ {
		qVal := fp8E4M3FromFloat(float32(int(1) << i)) // 1, 2, 4
		for s := 0; s < sl; s++ {
			for d := 0; d < hd; d++ {
				q[i*sl*hd+s*hd+d] = qVal
				k[i*sl*hd+s*hd+d] = fp8E4M3FromFloat(1.0)
				v[i*sl*hd+s*hd+d] = fp8E4M3FromFloat(1.0)
			}
		}
	}

	b, _ := backend.Get(backend.CUDA)
	qGPU, _ := b.ToDevice(backend.CUDADevice(0), &_cpuStorage{data: q})
	defer b.Free(qGPU)
	kGPU, _ := b.ToDevice(backend.CUDADevice(0), &_cpuStorage{data: k})
	defer b.Free(kGPU)
	vGPU, _ := b.ToDevice(backend.CUDADevice(0), &_cpuStorage{data: v})
	defer b.Free(vGPU)
	oGPU, _ := b.Alloc(bh * sl * hd * 2)
	defer b.Free(oGPU)
	lGPU, _ := b.Alloc(bh * sl * 4)
	defer b.Free(lGPU)

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := ctx.ForwardTrain(
		_ptr(qGPU), _ptr(kGPU), _ptr(vGPU), _ptr(oGPU), _ptr(lGPU),
		bh, sl, hd, causal, window,
		softmaxScale, 0); err != nil {
		t.Fatal(err)
	}
	if s, ok := b.(interface{ Sync() error }); ok {
		s.Sync()
	}
	lHost := _download(b, lGPU, bh*sl)

	// Reference per row.
	sqrtHd := float32(math.Sqrt(float64(hd)))
	logSl := float32(math.Log(float64(sl)))
	refPerRow := []float32{
		1*sqrtHd + logSl, // bh_i=0
		2*sqrtHd + logSl, // bh_i=1
		4*sqrtHd + logSl, // bh_i=2
	}

	t.Logf("layout L test: shape=[bh=%d, sl=%d], per-row refs = %.4f, %.4f, %.4f",
		bh, sl, refPerRow[0], refPerRow[1], refPerRow[2])
	for row := 0; row < bh; row++ {
		var maxDiff, maxRel float32
		for s := 0; s < sl; s++ {
			lv := lHost[row*sl+s]
			d := float32(math.Abs(float64(lv - refPerRow[row])))
			r := d / float32(math.Abs(float64(refPerRow[row])))
			if d > maxDiff {
				maxDiff = d
			}
			if r > maxRel {
				maxRel = r
			}
		}
		t.Logf("  row %d: L[0]=%.6f L[end]=%.6f ref=%.6f, worst |diff|=%.3e (rel=%.3e)",
			row, lHost[row*sl], lHost[row*sl+sl-1], refPerRow[row], maxDiff, maxRel)
		if maxRel > 5e-3 {
			t.Errorf("row %d L rel %.3e > 5e-3 (layout mismatch? off-by-one в [bh, sl] страйде?)",
				row, maxRel)
		}
	}

	// Cross-check: если layout был [sl, bh] transposed, то L[1] (index в flat)
	// был бы refPerRow[1] (следующий bh) вместо refPerRow[0] (следующий s).
	// В корректной [bh, sl] layout: L[0..sl-1]=refRow0, L[sl..2sl-1]=refRow1, ...
	// Проверка cross: L[sl-1] (последний s row 0) должен ≈ refRow0, не refRow1.
	if math.Abs(float64(lHost[sl-1]-refPerRow[0])) > 5e-3*math.Abs(float64(refPerRow[0])) {
		t.Errorf("layout cross-check FAIL: L[sl-1]=%.4f ≠ refRow0=%.4f (потенциально [sl, bh] layout вместо [bh, sl])",
			lHost[sl-1], refPerRow[0])
	}
}

// -----------------------------------------------------------------------------
// Helpers: small wrappers to reuse goml.cuda machinery without importing test
// helpers from abjexam.
// -----------------------------------------------------------------------------

type _cpuStorage struct {
	data []byte
}

func (c *_cpuStorage) Bytes() []byte           { return c.data }
func (c *_cpuStorage) ByteLen() int            { return len(c.data) }
func (c *_cpuStorage) Device() backend.Device  { return backend.CPU0 }
func (c *_cpuStorage) Free()                   {}
func (c *_cpuStorage) Ptr() unsafe.Pointer {
	if len(c.data) == 0 {
		return nil
	}
	return unsafe.Pointer(&c.data[0])
}

// _ptr — extract device pointer via type-assertion.
func _ptr(s backend.Storage) uintptr {
	type devPtrer interface{ DevicePtr() uintptr }
	if dp, ok := s.(devPtrer); ok {
		return dp.DevicePtr()
	}
	return uintptr(s.Ptr())
}

var _ unsafe.Pointer

// _download — copy device F32 buffer to host slice.
func _download(b backend.Backend, s backend.Storage, n int) []float32 {
	cpuS, err := b.ToDevice(backend.CPU0, s)
	if err != nil {
		panic(err)
	}
	buf := cpuS.Bytes()[:n*4]
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := uint32(buf[i*4]) | uint32(buf[i*4+1])<<8 |
			uint32(buf[i*4+2])<<16 | uint32(buf[i*4+3])<<24
		out[i] = math.Float32frombits(bits)
	}
	return out
}

// unused import guards.
var _ = os.Getenv
