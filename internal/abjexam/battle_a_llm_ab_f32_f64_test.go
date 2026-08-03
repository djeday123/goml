package abjexam

// A-LLM-3 Шаг 4 (+ Шаг 5): переклассификация GPU-F32-recon из "эталона"
// в "первый измеряемый" — A/B vs CPU-F64 арбитр, N=5 прогонов (П-4).
//
// ПРОГНОЗЫ (записаны ДО прогона):
//   Formula-floor (5-е применение sqrt-правила, та же параметризация что mixed-floor
//   F32-cert): floor_abs = C*sqrt(N_stages)*eps_F32*scale*amplif,
//   C=50, amplif=20, eps_F32=1.19e-7, scale=max(|ana_F64|,|ana_GPU|), minFloor=1e-5
//   при nStages>=8.
//   Нон-детерминизм GPU (full-tensor, из 9e1e3f0): dWq ~3.4e-3, dW1 ~6.6e-2,
//   dEmbed ~9.4e-2. ПРОГНОЗ: при таком спреде formula-floor на длинных цепях
//   провалится на части прогонов — документируемый floor пути будет определяться
//   нон-детерминизмом, не sqrt-накоплением. Оба числа фиксируются в raw.
//
//   Sign-of-life (Шаг 5): initial ~3.49, после 10 шагов lr=1e-2 ~2.6-2.7 (по efd833c/9e1e3f0).

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

func TestALLM_ABF32vsF64_N5(t *testing.T) {
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

	cfg := f64CertCfg()
	inp, tgt := f64CertTokens(cfg)

	// CPU-F64 арбитр (bit-det, Шаг 2 PASS).
	w64 := newBattleAF64Weights(cfg, 31)
	c64 := newBattleAF64Cache(cfg)
	loss64 := fwdBattleAF64(w64, c64, inp, tgt)
	g64 := bwdBattleAF64(w64, c64, inp, tgt)
	t.Logf("F64-арбитр: loss=%.10f", loss64)

	// GPU state — тот же seed 31 => веса bit-идентичны (F32).
	rInit := rand.New(rand.NewSource(31))
	st, err := NewBattleAState(cfg, rInit, adB)
	if err != nil {
		t.Fatalf("NewBattleAState: %v", err)
	}
	defer st.FreeAll(adB)
	sc, err := NewBattleAScratchF32(cfg, adB)
	if err != nil {
		t.Fatalf("NewBattleAScratchF32: %v", err)
	}
	defer sc.FreeAll(adB)
	bs, err := NewBattleABwdScratch(cfg, adB)
	if err != nil {
		t.Fatalf("NewBattleABwdScratch: %v", err)
	}
	defer bs.FreeAll(adB)
	grads, err := NewBattleAGrads(cfg, adB)
	if err != nil {
		t.Fatalf("NewBattleAGrads: %v", err)
	}
	defer grads.FreeAll(adB)

	// N=5 прогонов fwd+bwd, снапшоты 10 тензоров каждого прогона.
	const N = 5
	type runSnap struct {
		loss   float64
		DWout  []float32
		DNorm  []float32 // DNormOut
		DWq    []float32
		DWv    []float32
		DWo    []float32
		DN1    []float32 // DNorm1[0]
		DN2    []float32 // DNorm2[0]
		DW1    []float32
		DW2    []float32
		DEmbed []float32
	}
	snaps := make([]runSnap, N)
	for r := 0; r < N; r++ {
		loss, err := fwdBattleAF32(adB, st, sc, inp, tgt)
		if err != nil {
			t.Fatalf("run %d fwd: %v", r, err)
		}
		if err := zeroGrads(adB, grads); err != nil {
			t.Fatalf("run %d zeroGrads: %v", r, err)
		}
		if err := bwdBattleAF32(adB, st, sc, bs, grads); err != nil {
			t.Fatalf("run %d bwd: %v", r, err)
		}
		if s, ok := adB.(interface{ Sync() error }); ok {
			s.Sync()
		}
		snaps[r] = runSnap{
			loss:   loss,
			DWout:  gpuToHost(adB, grads.DWout, cfg.D*cfg.V),
			DNorm:  gpuToHost(adB, grads.DNormOut, cfg.D),
			DWq:    gpuToHost(adB, grads.Layers[0].DWq, cfg.D*cfg.D),
			DWv:    gpuToHost(adB, grads.Layers[0].DWv, cfg.D*cfg.D),
			DWo:    gpuToHost(adB, grads.Layers[0].DWo, cfg.D*cfg.D),
			DN1:    gpuToHost(adB, grads.Layers[0].DNorm1, cfg.D),
			DN2:    gpuToHost(adB, grads.Layers[0].DNorm2, cfg.D),
			DW1:    gpuToHost(adB, grads.Layers[0].DW1, cfg.D*cfg.FFN),
			DW2:    gpuToHost(adB, grads.Layers[0].DW2, cfg.FFN*cfg.D),
			DEmbed: gpuToHost(adB, grads.DEmbed, cfg.V*cfg.D),
		}
		t.Logf("GPU run %d: loss=%.10f", r+1, loss)
	}

	// Нон-детерминизм: max попарного |Δ| по каждому классу (full-tensor, П-4).
	pairwiseMax := func(get func(*runSnap) []float32) float32 {
		var mx float32
		for a := 0; a < N; a++ {
			for b := a + 1; b < N; b++ {
				xa, xb := get(&snaps[a]), get(&snaps[b])
				for i := range xa {
					d := xa[i] - xb[i]
					if d < 0 {
						d = -d
					}
					if d > mx {
						mx = d
					}
				}
			}
		}
		return mx
	}
	t.Logf("=== Шаг 4 нон-детерминизм GPU-F32 (N=%d, max попарный |Δ|, full-tensor) ===", N)
	ndClasses := []struct {
		name string
		get  func(*runSnap) []float32
	}{
		{"DWout", func(s *runSnap) []float32 { return s.DWout }},
		{"DNormOut", func(s *runSnap) []float32 { return s.DNorm }},
		{"DWq[0]", func(s *runSnap) []float32 { return s.DWq }},
		{"DWv[0]", func(s *runSnap) []float32 { return s.DWv }},
		{"DWo[0]", func(s *runSnap) []float32 { return s.DWo }},
		{"DNorm1[0]", func(s *runSnap) []float32 { return s.DN1 }},
		{"DNorm2[0]", func(s *runSnap) []float32 { return s.DN2 }},
		{"DW1[0]", func(s *runSnap) []float32 { return s.DW1 }},
		{"DW2[0]", func(s *runSnap) []float32 { return s.DW2 }},
		{"DEmbed", func(s *runSnap) []float32 { return s.DEmbed }},
	}
	for _, c := range ndClasses {
		t.Logf("  ND %-10s max|Δ|=%.3e (свойство пути, не дефект — вердикт ревьюера 8b3cf23)", c.name, pairwiseMax(c.get))
	}

	// A/B на 10 cert-точках: formula-floor ДО прогона, факт по всем 5 прогонам.
	wqIdx := (cfg.D/2)*cfg.D + (cfg.D / 3)
	w1Idx := (cfg.D/2)*cfg.FFN + (cfg.FFN / 3)
	w2Idx := (cfg.FFN/2)*cfg.D + (cfg.D / 3)
	wvIdx := (cfg.D/2)*cfg.D + (cfg.D / 3)
	woIdx := (cfg.D/2)*cfg.D + (cfg.D / 3)
	woutIdx := (cfg.D/2)*cfg.V + (cfg.V / 3)
	embIdx := int(inp[0])*cfg.D + (cfg.D / 3)
	argMaxAbs := func(xs []float64) int {
		best, bestV := 0, math.Abs(xs[0])
		for i := 1; i < len(xs); i++ {
			if a := math.Abs(xs[i]); a > bestV {
				best, bestV = i, a
			}
		}
		return best
	}
	gNOIdx := argMaxAbs(g64.DNormOut)
	gN1Idx := argMaxAbs(g64.Layers[0].DNorm1)
	gN2Idx := argMaxAbs(g64.Layers[0].DNorm2)

	type abPoint struct {
		name   string
		idx    int
		nStg   int
		f64    float64
		gpu    func(*runSnap) []float32
	}
	abPoints := []abPoint{
		{"Wout(top)", woutIdx, 1, g64.DWout[woutIdx], func(s *runSnap) []float32 { return s.DWout }},
		{"NormOut(g)", gNOIdx, 2, g64.DNormOut[gNOIdx], func(s *runSnap) []float32 { return s.DNorm }},
		{"Wo[L=0]", woIdx, 4, g64.Layers[0].DWo[woIdx], func(s *runSnap) []float32 { return s.DWo }},
		{"W2[L=0]", w2Idx, 4, g64.Layers[0].DW2[w2Idx], func(s *runSnap) []float32 { return s.DW2 }},
		{"Norm2[0](g)", gN2Idx, 5, g64.Layers[0].DNorm2[gN2Idx], func(s *runSnap) []float32 { return s.DN2 }},
		{"W1[L=0]", w1Idx, 6, g64.Layers[0].DW1[w1Idx], func(s *runSnap) []float32 { return s.DW1 }},
		{"Wv[L=0]", wvIdx, 8, g64.Layers[0].DWv[wvIdx], func(s *runSnap) []float32 { return s.DWv }},
		{"Norm1[0](g)", gN1Idx, 9, g64.Layers[0].DNorm1[gN1Idx], func(s *runSnap) []float32 { return s.DN1 }},
		{"Wq[L=0]", wqIdx, 10, g64.Layers[0].DWq[wqIdx], func(s *runSnap) []float32 { return s.DWq }},
		{"Embed[i0,d]", embIdx, 12, g64.DEmbed[embIdx], func(s *runSnap) []float32 { return s.DEmbed }},
	}
	const epsF32 = 1.19e-7
	const cSqrt = 50.0
	const scaleAmpl = 20.0
	t.Logf("=== Шаг 4 A/B GPU-F32 vs CPU-F64 (10 точек, все %d прогонов) ===", N)
	formulaFails := 0
	for _, p := range abPoints {
		var worst float64
		var worstRun int
		for r := 0; r < N; r++ {
			gv := float64(p.gpu(&snaps[r])[p.idx])
			d := math.Abs(gv - p.f64)
			if d > worst {
				worst = d
				worstRun = r + 1
			}
		}
		// Formula-floor (записан до прогона): scale = max(|f64|, |gpu run-1|).
		scale := math.Abs(p.f64)
		if g := math.Abs(float64(p.gpu(&snaps[0])[p.idx])); g > scale {
			scale = g
		}
		if scale < 1e-6 {
			scale = 1e-6
		}
		fl := cSqrt * math.Sqrt(float64(p.nStg)) * epsF32 * scale * scaleAmpl
		if p.nStg >= 8 && fl < 1e-5 {
			fl = 1e-5
		}
		verdict := "FORMULA-PASS"
		if worst > fl {
			verdict = "FORMULA-FAIL (документированный floor = факт. worst)"
			formulaFails++
		}
		relWorst := worst / math.Max(math.Abs(p.f64), 1e-15)
		t.Logf("  A/B %-12s f64=%+.6e worst|Δ|=%.3e (run %d, rel=%.2e) formulaFloor=%.3e → %s",
			p.name, p.f64, worst, worstRun, relWorst, fl, verdict)
	}
	t.Logf("Шаг 4 итог: formula-floor провален на %d/10 точках; документированный floor пути = worst|Δ| из raw выше", formulaFails)

	// === Шаг 5: sign-of-life 10 SGD-шагов lr=1e-2 (тот же state; веса не менялись A/B-прогонами) ===
	t.Logf("=== Шаг 5 sign-of-life: 10 SGD steps lr=1e-2 ===")
	losses := make([]float64, 0, 11)
	if l0, err := fwdBattleAF32(adB, st, sc, inp, tgt); err == nil {
		losses = append(losses, l0)
	}
	for step := 0; step < 10; step++ {
		loss, err := trainStepBattleAF32(adB, st, sc, bs, grads, inp, tgt, 1e-2)
		if err != nil {
			t.Fatalf("sign-of-life step %d: %v", step, err)
		}
		losses = append(losses, loss)
	}
	for i, l := range losses {
		t.Logf("  step %d: loss=%.4f", i, l)
	}
	if len(losses) >= 11 {
		t.Logf("Sign-of-life: initial=%.4f after10=%.4f Δ=%+.4f", losses[0], losses[10], losses[0]-losses[10])
	}
}
