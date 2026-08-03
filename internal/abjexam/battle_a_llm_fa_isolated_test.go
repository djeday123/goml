package abjexam

// A-LLM-4: изолирующий минимальный тест FA-цепочки ВНЕ трансформера.
// Цель: локализовать звено context-зависимого отказа (O=0 при живом L,
// amax=0 между прогонами) на форме A/B (BH=4, S=2048, HD=128).
// НЕ дебаг FA-lib (вне скоупа) — локализация для отчёта.

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"testing"

	"github.com/djeday123/goml/backend"
	_ "github.com/djeday123/goml/backend/cpu"
	gomlcuda "github.com/djeday123/goml/backend/cuda"
	adapter "github.com/djeday123/goml/backend/gotorch"
)

func TestALLM_FAChain_Isolated(t *testing.T) {
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
	if err := gomlcuda.FALoad(); err != nil {
		t.Skipf("libfa_sm120.so unavailable: %v", err)
	}
	if err := gomlcuda.FABwdLoad(); err != nil {
		t.Skipf("libfa_bwd_sm120.so unavailable: %v", err)
	}
	faCtx, err := gomlcuda.FACreate()
	if err != nil {
		t.Fatalf("FACreate: %v", err)
	}
	defer faCtx.Destroy()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Форма параметризуется: FA_ISO_SL (default 2048) — проверка гипотезы
	// "v121r-train валиден только на больших sl" третьей точкой.
	BH := 4
	S := 2048
	if v := os.Getenv("FA_ISO_SL"); v != "" {
		fmt.Sscanf(v, "%d", &S)
	}
	if v := os.Getenv("FA_ISO_BH"); v != "" {
		fmt.Sscanf(v, "%d", &BH)
	}
	// FA_ISO_CONST=1 -> входы константа FP8 1.0 и scale=softmaxScale (зеркало
	// канонического G2-теста) вместо квантованных N(0,0.02).
	constInputs := os.Getenv("FA_ISO_CONST") == "1" || os.Getenv("FA_ISO_CONST") == "2"
	// FA_ISO_NATIVE=1 -> все FA-буферы аллоцируются НАТИВНЫМ goml.cuda backend
	// (gomlB, полученным ДО adapter.Enable) — контекст канонического G2-теста.
	// Данные ходят через host-staging (cert-only паттерн).
	nativeCtx := os.Getenv("FA_ISO_NATIVE") == "1"
	const HD = 128
	n := BH * S * HD
	softmaxScale := float32(1.0 / math.Sqrt(float64(HD)))
	sync := func() {
		if s, ok := adB.(interface{ Sync() error }); ok {
			s.Sync()
		}
	}
	stat := func(bk backend.Backend, name string, s backend.Storage, cnt int) (nan, zero int, mx float64) {
		h := gpuToHost(bk, s, cnt)
		for _, v := range h {
			if math.IsNaN(float64(v)) {
				nan++
			} else if v == 0 {
				zero++
			} else if a := math.Abs(float64(v)); a > mx {
				mx = a
			}
		}
		t.Logf("  %-12s nan=%d zero=%d/%d max|.|=%.4e", name, nan, zero, cnt, mx)
		return
	}

	// H1: b.Copy канал (использован в attnFABwdBlock (g)).
	{
		a1, _ := adB.Alloc(1024 * 4)
		a2, _ := adB.Alloc(1024 * 4)
		pat := make([]float32, 1024)
		for i := range pat {
			pat[i] = float32(i) * 0.5
		}
		uploadInto(adB, a1, f32ToBytes(pat))
		zeros := make([]float32, 1024)
		uploadInto(adB, a2, f32ToBytes(zeros))
		if err := adB.Copy(a2, a1, 1024*4); err != nil {
			t.Fatalf("H1 Copy: %v", err)
		}
		sync()
		back := gpuToHost(adB, a2, 1024)
		var worst float64
		for i := range back {
			if d := math.Abs(float64(back[i] - pat[i])); d > worst {
				worst = d
			}
		}
		t.Logf("H1 b.Copy roundtrip: max|Δ|=%.3e (0 = канал чист)", worst)
		adB.Free(a1)
		adB.Free(a2)
	}

	// Синтетические входы ~N(0, 0.02) — класс боевых значений Q/K/V.
	r := rand.New(rand.NewSource(77))
	mk := func() []float32 {
		h := make([]float32, n)
		for i := range h {
			h[i] = float32(r.NormFloat64()) * 0.02
		}
		return h
	}
	qH, kH, vH := mk(), mk(), mk()
	hostAmax := func(h []float32) float32 {
		var mx float32
		for _, v := range h {
			if v < 0 {
				v = -v
			}
			if v > mx {
				mx = v
			}
		}
		return mx
	}
	up := func(h []float32) backend.Storage {
		s, err := adB.ToDevice(backend.CUDADevice(0), &cpuStorage{data: f32ToBytes(h)})
		if err != nil {
			t.Fatalf("upload: %v", err)
		}
		return s
	}
	qF32, kF32, vF32 := up(qH), up(kH), up(vH)
	defer adB.Free(qF32)
	defer adB.Free(kF32)
	defer adB.Free(vF32)

	// Бэкенд для FA-буферов: адаптер (боевой стык) или нативный goml.cuda.
	faB := adB
	if nativeCtx {
		faB = gomlB
		t.Logf("NATIVE-режим: FA-буферы в контексте goml.cuda (до adapter.Enable)")
	}
	// Локальные буферы (тест-проба; общий faBlockBufs двухконтекстный, здесь
	// проще одноконтекстные аллокации на faB).
	type isoBufs struct {
		QFP8, KFP8, VFP8    backend.Storage
		OFP16, LGPU         backend.Storage
		AmaxQ, AmaxK, AmaxV backend.Storage
		ScaleQ, ScaleK, ScaleV backend.Storage
	}
	fb := isoBufs{}
	mustAl := func(bytes int) backend.Storage {
		s, e := faB.Alloc(bytes)
		if e != nil {
			t.Fatalf("iso alloc: %v", e)
		}
		return s
	}
	fb.QFP8 = mustAl(n)
	fb.KFP8 = mustAl(n)
	fb.VFP8 = mustAl(n)
	fb.OFP16 = mustAl(n * 2)
	fb.LGPU = mustAl(BH * S * 4)
	fb.AmaxQ = mustAl(4)
	fb.AmaxK = mustAl(4)
	fb.AmaxV = mustAl(4)
	fb.ScaleQ = mustAl(4)
	fb.ScaleK = mustAl(4)
	fb.ScaleV = mustAl(4)
	defer func() {
		for _, s := range []backend.Storage{fb.QFP8, fb.KFP8, fb.VFP8, fb.OFP16,
			fb.LGPU, fb.AmaxQ, fb.AmaxK, fb.AmaxV, fb.ScaleQ, fb.ScaleK, fb.ScaleV} {
			faB.Free(s)
		}
	}()

	// Звено 1: quantize x3 (с zero-amax + Sync, зеркало блока), 2 прогона подряд —
	// проверка повторяемости amax (в Stage1 run1 дал 4.7e-3, run2 дал 0).
	// В CONST-режиме пропускается (входы грузятся напрямую; квантизатор на
	// n>=134M имеет свою границу LAUNCH_FAILED — вне скоупа).
	quantRounds := 2
	if constInputs {
		quantRounds = 0
	}
	for round := 1; round <= quantRounds; round++ {
		zeroF := []byte{0, 0, 0, 0}
		uploadInto(adB, fb.AmaxQ, zeroF)
		uploadInto(adB, fb.AmaxK, zeroF)
		uploadInto(adB, fb.AmaxV, zeroF)
		if err := gtB.QuantizeF32ToF8E4M3(qF32, fb.QFP8, fb.ScaleQ, fb.AmaxQ, n); err != nil {
			t.Fatalf("quantize Q: %v", err)
		}
		sync()
		if err := gtB.QuantizeF32ToF8E4M3(kF32, fb.KFP8, fb.ScaleK, fb.AmaxK, n); err != nil {
			t.Fatalf("quantize K: %v", err)
		}
		sync()
		if err := gtB.QuantizeF32ToF8E4M3(vF32, fb.VFP8, fb.ScaleV, fb.AmaxV, n); err != nil {
			t.Fatalf("quantize V: %v", err)
		}
		sync()
		aq := gpuToHost(adB, fb.AmaxQ, 1)[0]
		ak := gpuToHost(adB, fb.AmaxK, 1)[0]
		av := gpuToHost(adB, fb.AmaxV, 1)[0]
		t.Logf("Звено-1 round %d: amaxQ=%.4e (host %.4e) amaxK=%.4e (host %.4e) amaxV=%.4e (host %.4e)",
			round, aq, hostAmax(qH), ak, hostAmax(kH), av, hostAmax(vH))
	}
	// FP8-нули: доля нулевых байт в QFP8.
	{
		qb := make([]byte, 256)
		// readback первых 256 байт QFP8 через хост-путь невозможен gpuToHost (f32);
		// оценим через деквант: quantize сам не проверяем повторно — достаточно amax выше.
		_ = qb
	}

	amaxQ := gpuToHost(adB, fb.AmaxQ, 1)[0]
	amaxK := gpuToHost(adB, fb.AmaxK, 1)[0]
	amaxV := gpuToHost(adB, fb.AmaxV, 1)[0]
	scaleQ := amaxQ / 448.0
	scaleK := amaxK / 448.0
	scaleV := amaxV / 448.0
	faScale := softmaxScale * scaleQ * scaleK
	if nativeCtx && !constInputs {
		t.Fatalf("FA_ISO_NATIVE=1 требует FA_ISO_CONST=1 (host-staging без адаптерных ядер)")
	}
	if constInputs {
		if os.Getenv("FA_ISO_CONST") == "2" {
			// REALISH-режим: легальные случайные e4m3-байты (без NaN-кода 0xFF),
			// масштаб decoded ~+-2..448, faScale малый — зеркало боевого стыка
			// (кванты реальных N(0,0.02) с amax~2 дают decoded до +-448).
			rb := rand.New(rand.NewSource(99))
			smallMag := os.Getenv("FA_ISO_CONST_SMALL") == "1"
			mkFP8 := func() []byte {
				out := make([]byte, n)
				for i := range out {
					var v byte
					if smallMag {
						// decoded ~0.15..1.75 (проверка гипотезы FP16-S-accum overflow)
						v = byte(0x28 + rb.Intn(0x3d-0x28))
					} else {
						v = byte(rb.Intn(0x7e)) // весь диапазон, NaN-код исключён
					}
					if rb.Intn(2) == 1 {
						v |= 0x80
					}
					out[i] = v
				}
				return out
			}
			uploadInto(faB, fb.QFP8, mkFP8())
			uploadInto(faB, fb.KFP8, mkFP8())
			uploadInto(faB, fb.VFP8, mkFP8())
			scaleQ, scaleK, scaleV = 0.00471, 0.00466, 0.00481
			faScale = softmaxScale * scaleQ * scaleK
			t.Logf("REALISH-режим: random e4m3, faScale=%.4e (боевой класс)", faScale)
		} else {
			// Зеркало G2-теста: Q=K=V const FP8 1.0 (0x38), scale=softmaxScale.
			ones := make([]byte, n)
			for i := range ones {
				ones[i] = 0x38
			}
			uploadInto(faB, fb.QFP8, ones)
			uploadInto(faB, fb.KFP8, ones)
			uploadInto(faB, fb.VFP8, ones)
			scaleQ, scaleK, scaleV = 1.0, 1.0, 1.0
			faScale = softmaxScale
			t.Logf("CONST-режим: входы FP8=1.0, faScale=softmaxScale")
		}
	}
	t.Logf("scales: Q=%.4e K=%.4e V=%.4e faScale=%.4e", scaleQ, scaleK, scaleV, faScale)

	// Хостовая распаковка F16 (native-режим не может звать адаптерный Cast).
	f16ToF32Host := func(raw []byte) []float32 {
		out := make([]float32, len(raw)/2)
		for i := range out {
			h := uint16(raw[i*2]) | uint16(raw[i*2+1])<<8
			sign := uint32(h>>15) << 31
			exp := uint32(h>>10) & 0x1f
			man := uint32(h) & 0x3ff
			var bits uint32
			switch {
			case exp == 0 && man == 0:
				bits = sign
			case exp == 0x1f:
				bits = sign | 0x7f800000 | man<<13
			case exp == 0:
				e := uint32(127 - 15 + 1)
				for man&0x400 == 0 {
					man <<= 1
					e--
				}
				man &= 0x3ff
				bits = sign | e<<23 | man<<13
			default:
				bits = sign | (exp+112)<<23 | man<<13
			}
			out[i] = math.Float32frombits(bits)
		}
		return out
	}
	statHost := func(name string, h []float32) {
		nan, zero := 0, 0
		var mx float64
		for _, v := range h {
			if math.IsNaN(float64(v)) {
				nan++
			} else if v == 0 {
				zero++
			} else if a := math.Abs(float64(v)); a > mx {
				mx = a
			}
		}
		t.Logf("  %-12s nan=%d zero=%d/%d max|.|=%.4e", name, nan, zero, len(h), mx)
	}
	rawDownload := func(s backend.Storage, bytes int) []byte {
		// gpuToHost работает по 4-байтовым словам; для F16 читаем как f32-слова
		// и разбираем побайтово.
		words := gpuToHost(faB, s, bytes/4)
		out := make([]byte, bytes)
		for i, w := range words {
			b4 := math.Float32bits(w)
			out[i*4+0] = byte(b4)
			out[i*4+1] = byte(b4 >> 8)
			out[i*4+2] = byte(b4 >> 16)
			out[i*4+3] = byte(b4 >> 24)
		}
		return out
	}

	// Звено 2: fa_forward_train, 3 прогона подряд — контекст-повторяемость O.
	for round := 1; round <= 3; round++ {
		zo := make([]byte, n*2)
		uploadInto(faB, fb.OFP16, zo)
		zl := make([]byte, BH*S*4)
		uploadInto(faB, fb.LGPU, zl)
		if err := faCtx.ForwardTrain(devPtr(fb.QFP8), devPtr(fb.KFP8), devPtr(fb.VFP8),
			devPtr(fb.OFP16), devPtr(fb.LGPU), int32(BH), int32(S), int32(HD), 0, 0, faScale, 0); err != nil {
			t.Fatalf("fa fwd round %d: %v", round, err)
		}
		sync()
		t.Logf("Звено-2 round %d (fa_forward_train):", round)
		statHost("O(F16->32)", f16ToF32Host(rawDownload(fb.OFP16, n*2)))
		lHost := gpuToHost(faB, fb.LGPU, BH*S)
		statHost("L", lHost)
	}

	// Звено 3: bwd-четвёрка на этих O/L. dO F16 готовится на ХОСТЕ (native-safe).
	doH := make([]float32, n)
	for i := range doH {
		doH[i] = float32(r.NormFloat64()) * 0.005
	}
	f32ToF16Host := func(h []float32) []byte {
		out := make([]byte, len(h)*2)
		for i, v := range h {
			b32 := math.Float32bits(v)
			sign := uint16(b32>>16) & 0x8000
			exp := int32(b32>>23)&0xff - 127 + 15
			man := b32 & 0x7fffff
			var h16 uint16
			switch {
			case exp <= 0:
				h16 = sign // flush-to-zero субнормалей (класс 5e-3 сюда не попадает)
			case exp >= 0x1f:
				h16 = sign | 0x7c00
			default:
				h16 = sign | uint16(exp)<<10 | uint16(man>>13)
			}
			out[i*2] = byte(h16)
			out[i*2+1] = byte(h16 >> 8)
		}
		return out
	}
	doF16, _ := faB.Alloc(n * 2)
	defer faB.Free(doF16)
	uploadInto(faB, doF16, f32ToF16Host(doH))
	dBuf, _ := faB.Alloc(BH * S * 4)
	defer faB.Free(dBuf)
	if err := gomlcuda.FABwdDPrecompute(devPtr(fb.OFP16), devPtr(doF16), devPtr(dBuf), BH, S, HD, 0); err != nil {
		t.Fatalf("d_precompute: %v", err)
	}
	sync()
	sPad := (S + 15) & ^15
	dSnat, _ := faB.Alloc(BH * S * sPad)
	defer faB.Free(dSnat)
	dST, _ := faB.Alloc(BH * S * sPad)
	defer faB.Free(dST)
	dV, _ := faB.Alloc(n * 4)
	defer faB.Free(dV)
	dK, _ := faB.Alloc(n * 4)
	defer faB.Free(dK)
	dQ, _ := faB.Alloc(n * 4)
	defer faB.Free(dQ)
	zeroN := make([]byte, n*4)
	uploadInto(faB, dV, zeroN)
	uploadInto(faB, dK, zeroN)
	uploadInto(faB, dQ, zeroN)
	if err := gomlcuda.FABwdMerged(devPtr(fb.QFP8), devPtr(fb.KFP8), devPtr(fb.VFP8),
		devPtr(doF16), devPtr(fb.LGPU), devPtr(dBuf),
		devPtr(dSnat), devPtr(dST), devPtr(dV), BH, S, HD, 0, 0, faScale, 0); err != nil {
		t.Fatalf("merged: %v", err)
	}
	sync()
	if err := gomlcuda.FABwdDK(devPtr(fb.QFP8), devPtr(dST), devPtr(dK),
		BH, S, HD, 0, 0, softmaxScale*scaleV*scaleQ, 0); err != nil {
		t.Fatalf("dk: %v", err)
	}
	if err := gomlcuda.FABwdDQ(devPtr(fb.KFP8), devPtr(dSnat), devPtr(dQ),
		BH, S, HD, 0, 0, softmaxScale*scaleV*scaleK, 0); err != nil {
		t.Fatalf("dq: %v", err)
	}
	sync()
	t.Logf("Звено-3 (bwd-четвёрка):")
	stat(faB, "D", dBuf, BH*S)
	stat(faB, "dV", dV, n)
	stat(faB, "dK", dK, n)
	stat(faB, "dQ", dQ, n)
}
