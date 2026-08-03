package cuda

// A-LLM-5 П.2б/2в (2026-08-03): КОНТРАКТ-ТЕСТ магнитудного контракта FA-ядер —
// отсутствовавший класс теста, из-за которого контракт Н4 жил неписаным.
//
// Позитив: входы полного e4m3-разнообразия, нормированные к decoded O(1)
// (новая конвенция scale=amax) -> вся цепочка (fwd-train + D + merged + dk + dq)
// живая, L в прогнозной границе |L| <= faScale*hd*max^2 + ln(sl).
// Негатив (log-only, НЕ FAIL): decoded полного диапазона +-448 (старая
// конвенция amax/448) -> документированный NaN.
//   2026-08-03 Н4: документированная ловушка — FP16-акки QK/O MMA
//   (f16.e4m3.e4m3.f16) переполняются, см. A_LLM4_fa_integration.md.
// Чувствительностная проба ([[feedback-lnv-smoke-weak-gate]]): возмущение одной
// строки V -> O обязан измениться. ln(V)-smoke больше не гейт живости.

import (
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/djeday123/goml/backend"
)

// e4m3 decode (host, для контроля магнитуд синтезированных кодов).
func e4m3Decode(b byte) float64 {
	sign := 1.0
	if b&0x80 != 0 {
		sign = -1.0
	}
	exp := int((b >> 3) & 0xf)
	man := int(b & 0x7)
	if exp == 0xf && man == 0x7 {
		return math.NaN()
	}
	if exp == 0 {
		return sign * float64(man) * math.Pow(2, -9)
	}
	return sign * (1 + float64(man)/8) * math.Pow(2, float64(exp-7))
}

func faContractChainRun(t *testing.T, b backend.Backend, faCtx *FAContext,
	q, k, v []byte, bh, sl, hd int, scale float32) (oHost []float32, lHost []float32,
	bwdStats map[string][2]int) {
	n := bh * sl * hd
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
		int32(bh), int32(sl), int32(hd), 0, 0, scale, 0); err != nil {
		t.Fatalf("fa_forward_train: %v", err)
	}
	// O F16 -> host F32.
	oRawS, err := b.ToDevice(backend.CPU0, oG)
	if err != nil {
		t.Fatalf("download O: %v", err)
	}
	oRaw := oRawS.Bytes()[:n*2]
	oHost = make([]float32, n)
	for i := 0; i < n; i++ {
		h := uint16(oRaw[i*2]) | uint16(oRaw[i*2+1])<<8
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
		oHost[i] = math.Float32frombits(bits)
	}
	lHost = _download(b, lG, bh*sl)

	// Bwd-четвёрка: dO random F16 малой магнитуды.
	r := rand.New(rand.NewSource(321))
	doF32 := make([]float32, n)
	for i := range doF32 {
		doF32[i] = float32(r.NormFloat64() * 0.005)
	}
	doG := up(f32ToFP16Bytes(doF32))
	defer b.Free(doG)
	dG, _ := b.Alloc(bh * sl * 4)
	defer b.Free(dG)
	if err := FABwdDPrecompute(_ptr(oG), _ptr(doG), _ptr(dG), bh, sl, hd, 0); err != nil {
		t.Fatalf("d_precompute: %v", err)
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
		_ptr(dSnat), _ptr(dST), _ptr(dV), bh, sl, hd, 0, 0, scale, 0); err != nil {
		t.Fatalf("merged: %v", err)
	}
	if err := FABwdDK(_ptr(qG), _ptr(dST), _ptr(dK), bh, sl, hd, 0, 0, scale, 0); err != nil {
		t.Fatalf("dk: %v", err)
	}
	if err := FABwdDQ(_ptr(kG), _ptr(dSnat), _ptr(dQ), bh, sl, hd, 0, 0, scale, 0); err != nil {
		t.Fatalf("dq: %v", err)
	}
	countBad := func(s backend.Storage, cnt int) [2]int {
		h := _download(b, s, cnt)
		nan, inf := 0, 0
		for _, x := range h {
			if math.IsNaN(float64(x)) {
				nan++
			} else if math.IsInf(float64(x), 0) {
				inf++
			}
		}
		return [2]int{nan, inf}
	}
	bwdStats = map[string][2]int{
		"D":  countBad(dG, bh*sl),
		"dV": countBad(dV, n),
		"dK": countBad(dK, n),
		"dQ": countBad(dQ, n),
	}
	return oHost, lHost, bwdStats
}

func TestFAContract_FullRangeInputs(t *testing.T) {
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

	forms := []struct{ bh, sl int }{{4, 2048}, {128, 8192}}
	const hd = 128
	softmaxScale := float32(1.0 / math.Sqrt(float64(hd)))

	for _, f := range forms {
		n := f.bh * f.sl * hd
		r := rand.New(rand.NewSource(555))
		// ПОЗИТИВ: полное e4m3-разнообразие мантисс/экспонент, нормированное
		// к decoded O(1): коды с decoded <= 1.0 (exp-поле <= 7 => |v| <= 1.875?
		// нет: exp=7 -> 2^0*(1+m/8) <= 1.875; берем exp <= 6 и 7 c m=0 => <= 1.0).
		mkUnit := func() []byte {
			out := make([]byte, n)
			for i := range out {
				var code byte
				for {
					code = byte(r.Intn(0x78)) // exp<=14? нет: 0x77 max => exp poле
					d := math.Abs(e4m3Decode(code))
					if d <= 1.0 {
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
		q, k, v := mkUnit(), mkUnit(), mkUnit()
		// Контроль синтеза: max |decoded|.
		var maxDec float64
		for _, c := range q {
			if d := math.Abs(e4m3Decode(c)); d > maxDec {
				maxDec = d
			}
		}
		oH, lH, bwd := faContractChainRun(t, b, faCtx, q, k, v, f.bh, f.sl, hd, softmaxScale)
		statBad := func(name string, h []float32) (nan int, mx float64) {
			for _, x := range h {
				if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
					nan++
				} else if a := math.Abs(float64(x)); a > mx {
					mx = a
				}
			}
			t.Logf("  [bh=%d sl=%d] %s: bad=%d max|.|=%.4e", f.bh, f.sl, name, nan, mx)
			return
		}
		nanO, _ := statBad("O", oH)
		nanL, maxL := statBad("L", lH)
		// Прогноз границы L: faScale*hd*max^2 + ln(sl).
		lBound := float64(softmaxScale)*float64(hd)*maxDec*maxDec + math.Log(float64(f.sl))
		t.Logf("  [bh=%d sl=%d] maxDecoded=%.4f, L-граница прогноза=%.3f, факт max L=%.3f", f.bh, f.sl, maxDec, lBound, maxL)
		fails := 0
		if nanO > 0 || nanL > 0 {
			t.Errorf("КОНТРАКТ FAIL [bh=%d sl=%d]: NaN/Inf в O(%d)/L(%d) при O(1)-входах", f.bh, f.sl, nanO, nanL)
			fails++
		}
		if maxL > lBound {
			t.Errorf("КОНТРАКТ FAIL [bh=%d sl=%d]: L=%.3f выше прогнозной границы %.3f", f.bh, f.sl, maxL, lBound)
			fails++
		}
		for name, s := range bwd {
			t.Logf("  [bh=%d sl=%d] bwd %s: nan=%d inf=%d", f.bh, f.sl, name, s[0], s[1])
			if s[0]+s[1] > 0 {
				t.Errorf("КОНТРАКТ FAIL [bh=%d sl=%d]: NaN/Inf в %s", f.bh, f.sl, name)
				fails++
			}
		}

		// ЧУВСТВИТЕЛЬНОСТНАЯ ПРОБА (2в): возмущаем строку V (bh=0, s=0) -> O меняется.
		v2 := append([]byte(nil), v...)
		for d := 0; d < hd; d++ {
			v2[d] ^= 0x08 // сдвиг мантиссы младшего бита каждой компоненты строки
		}
		oH2, _, _ := faContractChainRun(t, b, faCtx, q, k, v2, f.bh, f.sl, hd, softmaxScale)
		var maxDO float64
		for i := range oH {
			if d := math.Abs(float64(oH[i] - oH2[i])); d > maxDO {
				maxDO = d
			}
		}
		t.Logf("  [bh=%d sl=%d] чувствительность: max|dO| после возмущения строки V = %.4e", f.bh, f.sl, maxDO)
		if maxDO == 0 {
			t.Errorf("ЧУВСТВИТЕЛЬНОСТЬ FAIL [bh=%d sl=%d]: O не изменился — attention мертв (класс Н4/FA-out-zero)", f.bh, f.sl)
		}

		if f.bh == 4 {
			// НЕГАТИВ (log-only; 2026-08-03 Н4: документированная ловушка старой
			// конвенции amax/448): decoded полного диапазона до +-448 -> NaN.
			mkFull := func() []byte {
				out := make([]byte, n)
				for i := range out {
					code := byte(r.Intn(0x7e))
					if r.Intn(2) == 1 {
						code |= 0x80
					}
					out[i] = code
				}
				return out
			}
			oN, lN, _ := faContractChainRun(t, b, faCtx, mkFull(), mkFull(), mkFull(),
				f.bh, f.sl, hd, softmaxScale*1.9e-5) // масштаб класса старой конвенции
			nanON, _ := 0, 0.0
			for _, x := range oN {
				if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
					nanON++
				}
			}
			nanLN := 0
			for _, x := range lN {
				if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
					nanLN++
				}
			}
			t.Logf("  НЕГАТИВ (документированная ловушка, log-only): full-range decoded -> O bad=%d/%d, L bad=%d/%d",
				nanON, len(oN), nanLN, len(lN))
		}
		if fails == 0 {
			t.Logf("КОНТРАКТ [bh=%d sl=%d] PASS: цепочка живая на O(1)-входах, L в границе", f.bh, f.sl)
		}
	}
}
