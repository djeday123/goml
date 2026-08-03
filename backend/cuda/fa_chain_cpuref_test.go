package cuda

// A-LLM-5 П.6г (2026-08-03): изолированная цепочка ЧИСЛОМ — расширение живого
// CPU-F64 D-ref из G2-smoke (TestFABwd_DPrecompute) на dV/dQ/dK.
// Референс НЕ плодится второй: та же форма/паттерн (bh=1, sl=128, hd=128),
// та же дисциплина; добавлены полные формулы цепочки.
//
// Floor 5e-3 abs (записан ДО прогона, боевой класс A/B из ТЗ A-LLM-4).
// Оговорка (до прогона): GPU-путь квантизует dS в e4m3 direct-cast (3-бит
// мантисса, порог 2^-9) — ref dS НЕ квантуется; расхождение на dQ/dK несёт
// квант-шум dS. ПРОГНОЗ: суммирование 128 членов усредняет rel~6% кванта
// до ~0.5% net -> abs класса 1e-3 < 5e-3. При промахе — два числа.

import (
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/djeday123/goml/backend"
)

func TestFABwd_ChainVsCPURef(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	if err := FALoad(); err != nil {
		t.Skipf("libfa_sm120.so unavailable: %v", err)
	}
	if err := FABwdLoad(); err != nil {
		t.Skipf("libfa_bwd_sm120.so unavailable: %v", err)
	}
	b, err := backend.Get(backend.CUDA)
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	if s, err := b.Alloc(4); err == nil {
		b.Free(s)
	}
	faCtx, err := FACreate()
	if err != nil {
		t.Fatalf("FACreate: %v", err)
	}
	defer faCtx.Destroy()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	const (
		bh = 1
		sl = 128
		hd = 128
	)
	n := bh * sl * hd
	softmaxScale := 1.0 / math.Sqrt(float64(hd))

	// Входы: O(1)-коды e4m3 (контракт A-LLM-5), decoded <= 1.
	r := rand.New(rand.NewSource(777))
	mk := func() []byte {
		out := make([]byte, n)
		for i := range out {
			var code byte
			for {
				code = byte(r.Intn(0x78))
				if math.Abs(e4m3Decode(code)) <= 1.0 {
					break
				}
			}
			if r.Intn(2) == 1 {
				code |= 0x80
			}
			out[i] = code
		}
		return out
	}
	q, k, v := mk(), mk(), mk()
	doF32 := make([]float32, n)
	for i := range doF32 {
		doF32[i] = float32(r.NormFloat64() * 0.005)
	}

	// GPU chain (через общий harness контракт-теста; dO seed внутри = 321,
	// поэтому для сверки пересчитаем ref с тем же dO).
	rDo := rand.New(rand.NewSource(321))
	for i := range doF32 {
		doF32[i] = float32(rDo.NormFloat64() * 0.005)
	}
	oH, lH, _ := faContractChainRun(t, b, faCtx, q, k, v, bh, sl, hd, float32(softmaxScale))
	// Повторный прогон для скачивания dV/dK/dQ: harness их не возвращает —
	// делаем прямой прогон здесь с сохранением градиентов.
	up := func(data []byte) backend.Storage {
		s, err := b.ToDevice(backend.CUDADevice(0), &_cpuStorage{data: data})
		if err != nil {
			t.Fatalf("upload: %v", err)
		}
		return s
	}
	qG, kG, vG := up(q), up(k), up(v)
	defer b.Free(qG)
	defer b.Free(kG)
	defer b.Free(vG)
	oG, _ := b.Alloc(n * 2)
	defer b.Free(oG)
	lG, _ := b.Alloc(bh * sl * 4)
	defer b.Free(lG)
	zero := func(s backend.Storage, bytes int) {
		z, _ := b.ToDevice(backend.CUDADevice(0), &_cpuStorage{data: make([]byte, bytes)})
		b.Copy(s, z, bytes)
		b.Free(z)
	}
	zero(oG, n*2)
	zero(lG, bh*sl*4)
	if err := faCtx.ForwardTrain(_ptr(qG), _ptr(kG), _ptr(vG), _ptr(oG), _ptr(lG),
		bh, sl, hd, 0, 0, float32(softmaxScale), 0); err != nil {
		t.Fatalf("fwd: %v", err)
	}
	doG := up(f32ToFP16Bytes(doF32))
	defer b.Free(doG)
	dG, _ := b.Alloc(bh * sl * 4)
	defer b.Free(dG)
	if err := FABwdDPrecompute(_ptr(oG), _ptr(doG), _ptr(dG), bh, sl, hd, 0); err != nil {
		t.Fatalf("d: %v", err)
	}
	sPad := (sl + 15) & ^15
	dSnat, _ := b.Alloc(bh * sl * sPad)
	defer b.Free(dSnat)
	dST, _ := b.Alloc(bh * sl * sPad)
	defer b.Free(dST)
	dV, _ := b.Alloc(n * 4)
	defer b.Free(dV)
	dK, _ := b.Alloc(n * 4)
	defer b.Free(dK)
	dQ, _ := b.Alloc(n * 4)
	defer b.Free(dQ)
	zero(dV, n*4)
	zero(dK, n*4)
	zero(dQ, n*4)
	if err := FABwdMerged(_ptr(qG), _ptr(kG), _ptr(vG), _ptr(doG), _ptr(lG), _ptr(dG),
		_ptr(dSnat), _ptr(dST), _ptr(dV), bh, sl, hd, 0, 0, float32(softmaxScale), 0); err != nil {
		t.Fatalf("merged: %v", err)
	}
	if err := FABwdDK(_ptr(qG), _ptr(dST), _ptr(dK), bh, sl, hd, 0, 0, float32(softmaxScale), 0); err != nil {
		t.Fatalf("dk: %v", err)
	}
	if err := FABwdDQ(_ptr(kG), _ptr(dSnat), _ptr(dQ), bh, sl, hd, 0, 0, float32(softmaxScale), 0); err != nil {
		t.Fatalf("dq: %v", err)
	}
	dVh := _download(b, dV, n)
	dKh := _download(b, dK, n)
	dQh := _download(b, dQ, n)
	dH := _download(b, dG, bh*sl)

	// CPU-F64 ref: полная цепочка формулами из decoded значений.
	qd := make([]float64, n)
	kd := make([]float64, n)
	vd := make([]float64, n)
	for i := 0; i < n; i++ {
		qd[i] = e4m3Decode(q[i])
		kd[i] = e4m3Decode(k[i])
		vd[i] = e4m3Decode(v[i])
	}
	do64 := make([]float64, n)
	for i := range doF32 {
		do64[i] = float64(doF32[i]) // ref на F32-значениях (до F16-кванта; шум F16 ~6e-8 << floor)
	}
	S := make([]float64, sl*sl)
	for i := 0; i < sl; i++ {
		for j := 0; j < sl; j++ {
			var acc float64
			for d := 0; d < hd; d++ {
				acc += qd[i*hd+d] * kd[j*hd+d]
			}
			S[i*sl+j] = acc * softmaxScale
		}
	}
	P := make([]float64, sl*sl)
	Lref := make([]float64, sl)
	for i := 0; i < sl; i++ {
		mx := S[i*sl]
		for j := 1; j < sl; j++ {
			if S[i*sl+j] > mx {
				mx = S[i*sl+j]
			}
		}
		var sum float64
		for j := 0; j < sl; j++ {
			e := math.Exp(S[i*sl+j] - mx)
			P[i*sl+j] = e
			sum += e
		}
		for j := 0; j < sl; j++ {
			P[i*sl+j] /= sum
		}
		Lref[i] = mx + math.Log(sum)
	}
	Oref := make([]float64, n)
	for i := 0; i < sl; i++ {
		for d := 0; d < hd; d++ {
			var acc float64
			for j := 0; j < sl; j++ {
				acc += P[i*sl+j] * vd[j*hd+d]
			}
			Oref[i*hd+d] = acc
		}
	}
	Dref := make([]float64, sl)
	dPm := make([]float64, sl*sl)
	for i := 0; i < sl; i++ {
		var acc float64
		for d := 0; d < hd; d++ {
			acc += Oref[i*hd+d] * do64[i*hd+d]
		}
		Dref[i] = acc
		for j := 0; j < sl; j++ {
			var a2 float64
			for d := 0; d < hd; d++ {
				a2 += do64[i*hd+d] * vd[j*hd+d]
			}
			dPm[i*sl+j] = a2
		}
	}
	dS := make([]float64, sl*sl)
	for i := 0; i < sl; i++ {
		for j := 0; j < sl; j++ {
			dS[i*sl+j] = P[i*sl+j] * (dPm[i*sl+j] - Dref[i])
		}
	}
	dVref := make([]float64, n)
	dQref := make([]float64, n)
	dKref := make([]float64, n)
	for j := 0; j < sl; j++ {
		for d := 0; d < hd; d++ {
			var acc float64
			for i := 0; i < sl; i++ {
				acc += P[i*sl+j] * do64[i*hd+d]
			}
			dVref[j*hd+d] = acc
		}
	}
	for i := 0; i < sl; i++ {
		for d := 0; d < hd; d++ {
			var accQ float64
			for j := 0; j < sl; j++ {
				accQ += dS[i*sl+j] * kd[j*hd+d]
			}
			dQref[i*hd+d] = accQ * softmaxScale
		}
	}
	for j := 0; j < sl; j++ {
		for d := 0; d < hd; d++ {
			var accK float64
			for i := 0; i < sl; i++ {
				accK += dS[i*sl+j] * qd[i*hd+d]
			}
			dKref[j*hd+d] = accK * softmaxScale
		}
	}

	worst := func(name string, gpu []float32, ref []float64) float64 {
		var mx float64
		for i := range ref {
			d := math.Abs(float64(gpu[i]) - ref[i])
			if d > mx {
				mx = d
			}
		}
		return mx
	}
	const floor = 5e-3
	checks := []struct {
		name string
		gpu  []float32
		ref  []float64
	}{
		{"O", oH, Oref}, {"L", lH, Lref}, {"D", dH, Dref},
		{"dV", dVh, dVref}, {"dK", dKh, dKref}, {"dQ", dQh, dQref},
	}
	fails := 0
	for _, c := range checks {
		w := worst(c.name, c.gpu, c.ref)
		verdict := "PASS"
		if w > floor {
			verdict = "FAIL"
			fails++
		}
		t.Logf("CHAIN-REF %-3s worst|GPU-F64ref|=%.4e (floor %.0e) -> %s", c.name, w, floor, verdict)
	}
	if fails > 0 {
		t.Errorf("П.6г: %d/6 выше floor 5e-3", fails)
	} else {
		t.Logf("П.6г PASS: вся цепочка (O/L/D/dV/dK/dQ) в floor 5e-3 vs CPU-F64 ref")
	}
	_ = lH
}
