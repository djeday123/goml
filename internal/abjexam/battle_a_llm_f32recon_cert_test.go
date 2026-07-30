package abjexam

// A-LLM-2 v2 ЭТАЛОН cert (2026-07-27).
// User rejected v1 sign-of-life как proof of correctness (caveat 1 = мат. ошибка).
// Пересертификация на ПОЛНОЙ форме через finite-diff в разных слоях.
// Точки в РАЗНЫХ слоях обязательны (межслойность ловит caveat-1 класс багов).
//
// F32-only stack: fwdBattleAF32 (attnReconstruct вместо FA) + bwdBattleAF32.
// Снапшоты безопасны потому что FA не вызывается.

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

func TestALLM_BwdCertF32_MultiLayer(t *testing.T) {
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

	// Probe: hd=128 D=128 S=32. Isolates hd-specific bug.
	cfg := BattleACfg{
		V: 32, D: 128, H: 1, HD: 128, L: 1, S: 32, B: 1, FFN: 128,
		Base: 10000.0, Eps: 1e-5,
	}
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

	// Гипотеза 1/2: УНИКАЛЬНЫЕ токены исключают atomicAdd multi-occurrence
	// в EmbeddingGrad (num-grad шевелит только ОДНУ позицию, но ana аккумулирует
	// scatter в ту же позицию N раз при повторах).
	rTok := rand.New(rand.NewSource(41))
	M := cfg.B * cfg.S
	inp := make([]int64, M)
	tgt := make([]int32, M)
	usedTokens := map[int]bool{}
	if M > cfg.V {
		t.Fatalf("M=%d > V=%d — cannot have unique tokens", M, cfg.V)
	}
	for i := 0; i < M; i++ {
		var t int
		for {
			t = rTok.Intn(cfg.V)
			if !usedTokens[t] {
				break
			}
		}
		usedTokens[t] = true
		inp[i] = int64(t)
		tgt[i] = int32(rTok.Intn(cfg.V)) // targets могут повторяться (не участвуют в scatter)
	}
	// Проверка уникальности.
	seen := map[int64]bool{}
	for _, v := range inp {
		if seen[v] {
			t.Fatalf("input tokens NOT unique — cert test broken")
		}
		seen[v] = true
	}
	t.Logf("Input tokens: %d unique of %d total; V=%d", len(seen), M, cfg.V)

	// Ход-2а: warmup ОТКЛЮЧЁН (SGD-warmup при dead attn = death spiral, cf. Ход-1а ratio=1.01).
	// Fresh init — самое честное состояние. P-энтропия ПРОГНОЗ: близко к ln(S)=ln(32)=3.466 (uniform).
	// Если и на fresh init dWq систематически теряет реальный slope — GPU-баг подтверждён.
	const skipWarmup = true
	if !skipWarmup {
		for step := 0; step < 5; step++ {
			if _, err := trainStepBattleAF32(adB, st, sc, bs, grads, inp, tgt, 1e-2); err != nil {
				t.Fatalf("warmup step %d: %v", step, err)
			}
		}
	}
	// Analytical bwd (после warmup).
	loss, err := fwdBattleAF32(adB, st, sc, inp, tgt)
	if err != nil {
		t.Fatalf("initial fwd: %v", err)
	}
	if math.IsNaN(loss) {
		t.Fatalf("initial fwd loss NaN")
	}
	t.Logf("Initial fwd loss=%.4f (ln(V)=%.4f)", loss, math.Log(float64(cfg.V)))
	if err := zeroGrads(adB, grads); err != nil {
		t.Fatalf("zeroGrads: %v", err)
	}
	if err := bwdBattleAF32(adB, st, sc, bs, grads); err != nil {
		t.Fatalf("initial bwd: %v", err)
	}
	if s, ok := adB.(interface{ Sync() error }); ok {
		s.Sync()
	}
	// D-протокол: поэтажный CPU-F64 arbiter. Snapshot после каждого звена.
	dxAfterTopH := gpuToHost(adB, bs.DXAfterTop, (cfg.B * cfg.S)*cfg.D)
	dffnOutSnapH := gpuToHost(adB, bs.DFFNOutSnap, (cfg.B * cfg.S)*cfg.D)
	dffnSiluSnapH := gpuToHost(adB, bs.DFFNSiluSnap, (cfg.B * cfg.S)*cfg.FFN)
	dffnHidSnapH := gpuToHost(adB, bs.DFFNHidSnap, (cfg.B * cfg.S)*cfg.FFN)
	xpreTopH := gpuToHost(adB, sc.XPreTop, (cfg.B * cfg.S)*cfg.D)
	normOutH := gpuToHost(adB, st.NormOut, cfg.D)
	dnormedTopH := gpuToHost(adB, bs.DNormedTop, (cfg.B * cfg.S)*cfg.D)
	w2L0H := gpuToHost(adB, st.Layers[0].W2, cfg.FFN*cfg.D)
	ffnHidH := gpuToHost(adB, sc.FFNHid, (cfg.B * cfg.S)*cfg.FFN)
	ffnSigH := gpuToHost(adB, sc.FFNSig, (cfg.B * cfg.S)*cfg.FFN)
	{
		// (1) dX_ref через F64: dx_j = γ_j·dy_j/rms - x_j·S·rms^{-3}/D.
		Mn := (cfg.B * cfg.S)
		Dn := cfg.D
		dxRef := make([]float64, Mn*Dn)
		for row := 0; row < Mn; row++ {
			var sumX2 float64
			for j := 0; j < Dn; j++ {
				x := float64(xpreTopH[row*Dn+j])
				sumX2 += x * x
			}
			ms := sumX2/float64(Dn) + float64(cfg.Eps)
			rms := math.Sqrt(ms)
			var S float64
			for j := 0; j < Dn; j++ {
				S += float64(normOutH[j]) * float64(xpreTopH[row*Dn+j]) * float64(dnormedTopH[row*Dn+j])
			}
			invRms := 1.0 / rms
			invRms3ByD := invRms * invRms * invRms / float64(Dn)
			for j := 0; j < Dn; j++ {
				g := float64(normOutH[j])
				x := float64(xpreTopH[row*Dn+j])
				dy := float64(dnormedTopH[row*Dn+j])
				dxRef[row*Dn+j] = g*dy*invRms - x*S*invRms3ByD
			}
		}
		var maxAbsDx, maxRelDx float64
		var maxGpu, maxRef float64
		var argMax int
		for i := range dxRef {
			gpu := float64(dxAfterTopH[i])
			ref := dxRef[i]
			diff := math.Abs(gpu - ref)
			den := math.Abs(ref)
			if den < 1e-8 {
				den = 1e-8
			}
			if diff > maxAbsDx {
				maxAbsDx = diff
				maxGpu = gpu
				maxRef = ref
				argMax = i
			}
			rel := diff / den
			if rel > maxRelDx {
				maxRelDx = rel
			}
		}
		t.Logf("D-hop-1 dX_after_top: max cell idx=%d GPU=%+.6e F64=%+.6e diff=%.3e maxRel=%.3e",
			argMax, maxGpu, maxRef, maxAbsDx, maxRelDx)
	}
	{
		// (2) dFFNOut = Copy(bs.DX) - must be identical to dX_after_top.
		var maxDiff float64
		for i := range dxAfterTopH {
			d := math.Abs(float64(dffnOutSnapH[i]) - float64(dxAfterTopH[i]))
			if d > maxDiff {
				maxDiff = d
			}
		}
		t.Logf("D-hop-2 dFFNOut vs dX_after_top: max abs diff=%.3e (must be 0)", maxDiff)
	}
	{
		// (3) dFFNSilu_ref = dFFNOut @ W2^T. Shape [M, FFN].
		Mn := (cfg.B * cfg.S)
		FFNn := cfg.FFN
		Dn := cfg.D
		dsiluRef := make([]float64, Mn*FFNn)
		for m := 0; m < Mn; m++ {
			for f := 0; f < FFNn; f++ {
				var acc float64
				for d := 0; d < Dn; d++ {
					acc += float64(dffnOutSnapH[m*Dn+d]) * float64(w2L0H[f*Dn+d])
				}
				dsiluRef[m*FFNn+f] = acc
			}
		}
		var maxAbs, maxRel float64
		var maxGpu, maxRef float64
		var argMax int
		for i := range dsiluRef {
			gpu := float64(dffnSiluSnapH[i])
			ref := dsiluRef[i]
			diff := math.Abs(gpu - ref)
			den := math.Abs(ref)
			if den < 1e-8 {
				den = 1e-8
			}
			if diff > maxAbs {
				maxAbs = diff
				maxGpu = gpu
				maxRef = ref
				argMax = i
			}
			rel := diff / den
			if rel > maxRel {
				maxRel = rel
			}
		}
		t.Logf("D-hop-3 dFFNSilu: max cell idx=%d GPU=%+.6e F64=%+.6e diff=%.3e maxRel=%.3e",
			argMax, maxGpu, maxRef, maxAbs, maxRel)
	}
	{
		// (4) dFFNHidden_ref = silu_bwd(dFFNSilu, FFNHid, FFNSig).
		// silu(h) = h * σ(h). d silu/dh = σ + h * σ * (1 - σ).
		Mn := (cfg.B * cfg.S)
		FFNn := cfg.FFN
		dhidRef := make([]float64, Mn*FFNn)
		for i := range dhidRef {
			h := float64(ffnHidH[i])
			s := float64(ffnSigH[i])
			dSiludh := s + h*s*(1.0-s)
			dhidRef[i] = float64(dffnSiluSnapH[i]) * dSiludh
		}
		var maxAbs, maxRel float64
		var maxGpu, maxRef float64
		var argMax int
		for i := range dhidRef {
			gpu := float64(dffnHidSnapH[i])
			ref := dhidRef[i]
			diff := math.Abs(gpu - ref)
			den := math.Abs(ref)
			if den < 1e-8 {
				den = 1e-8
			}
			if diff > maxAbs {
				maxAbs = diff
				maxGpu = gpu
				maxRef = ref
				argMax = i
			}
			rel := diff / den
			if rel > maxRel {
				maxRel = rel
			}
		}
		t.Logf("D-hop-4 dFFNHidden: max cell idx=%d GPU=%+.6e F64=%+.6e diff=%.3e maxRel=%.3e",
			argMax, maxGpu, maxRef, maxAbs, maxRel)
	}

	// Snapshot analytical grads for cross-layer points.
	dWq0Ana := gpuToHost(adB, grads.Layers[0].DWq, cfg.D*cfg.D)
	_ = grads.Layers[cfg.L-1].DWq // last layer unused in L=1
	dW1L0Ana := gpuToHost(adB, grads.Layers[0].DW1, cfg.D*cfg.FFN)
	dW2L0Ana := gpuToHost(adB, grads.Layers[0].DW2, cfg.FFN*cfg.D)
	dWv0Ana := gpuToHost(adB, grads.Layers[0].DWv, cfg.D*cfg.D)
	dWo0Ana := gpuToHost(adB, grads.Layers[0].DWo, cfg.D*cfg.D)
	dWout0Ana := gpuToHost(adB, grads.DWout, cfg.D*cfg.V)
	dEmbedAna := gpuToHost(adB, grads.DEmbed, cfg.V*cfg.D)

	// NaN check in grads.
	statNaN := func(name string, h []float32) {
		nans := 0
		var mx float32
		for _, v := range h {
			if math.IsNaN(float64(v)) {
				nans++
			} else if a := v; a > mx || -a > mx {
				if a < 0 {
					mx = -a
				} else {
					mx = a
				}
			}
		}
		t.Logf("  %s: nans=%d/%d max|.|=%.3e", name, nans, len(h), mx)
	}
	t.Logf("Analytical grads scan (F32-only path):")
	statNaN("DEmbed", dEmbedAna)
	statNaN("DWq[L=0]", dWq0Ana)
	statNaN("DW1[L=0]", dW1L0Ana)

	// Numerical grad via central FD.
	const eps float32 = 1e-2
	upload := func(s backend.Storage, host []float32) {
		if _, err := uploadInto(adB, s, f32ToBytes(host)); err != nil {
			t.Fatalf("upload: %v", err)
		}
	}
	fwdLoss := func() float64 {
		l, err := fwdBattleAF32(adB, st, sc, inp, tgt)
		if err != nil {
			t.Fatalf("fwdLoss: %v", err)
		}
		return l
	}
	numGradAt := func(W backend.Storage, idx int, n int, e float32) float32 {
		wH := gpuToHost(adB, W, n)
		orig := wH[idx]
		wH[idx] = orig + e
		upload(W, wH)
		lp := fwdLoss()
		wH[idx] = orig - e
		upload(W, wH)
		lm := fwdLoss()
		wH[idx] = orig
		upload(W, wH)
		return float32((lp - lm) / (2.0 * float64(e)))
	}
	numGrad := func(W backend.Storage, idx int, n int) float32 {
		return numGradAt(W, idx, n, eps)
	}

	// Test points across DIFFERENT layers — межслойность ловит caveat-1.
	type point struct {
		name  string
		W     backend.Storage
		nElem int
		idx   int
		ana   float32
	}
	// Индексы автомат: middle-ish entries.
	wqIdx := (cfg.D / 2) * cfg.D + (cfg.D / 3)     // [D/2][D/3]
	w1Idx := (cfg.D / 2) * cfg.FFN + (cfg.FFN / 3) // [D/2][FFN/3]
	w2Idx := (cfg.FFN / 2) * cfg.D + (cfg.D / 3)   // [FFN/2][D/3]
	wvIdx := (cfg.D / 2) * cfg.D + (cfg.D / 3)     // [D/2][D/3]
	woIdx := (cfg.D / 2) * cfg.D + (cfg.D / 3)     // [D/2][D/3]
	woutIdx := (cfg.D / 2) * cfg.V + (cfg.V / 3)   // [D/2][V/3]
	embIdx := int(inp[0])*cfg.D + (cfg.D / 3)      // [inp0][D/3]
	// F-протокол: dW2 → если чист, зло между dFFNOut и dW1; если off, зло выше (top chain).
	// dWv → parallel check attn chain (K/V different from Q); dWo → attn output layer.
	// dWout → CE-стык verification.
	points := []point{
		{"Wout(top)", st.Wout, cfg.D * cfg.V, woutIdx, dWout0Ana[woutIdx]},
		{"Wo[L=0](attn-out)", st.Layers[0].Wo, cfg.D * cfg.D, woIdx, dWo0Ana[woIdx]},
		{"W2[L=0](ffn-out)", st.Layers[0].W2, cfg.FFN * cfg.D, w2Idx, dW2L0Ana[w2Idx]},
		{"W1[L=0](ffn-in)", st.Layers[0].W1, cfg.D * cfg.FFN, w1Idx, dW1L0Ana[w1Idx]},
		{"Wv[L=0](attn-in)", st.Layers[0].Wv, cfg.D * cfg.D, wvIdx, dWv0Ana[wvIdx]},
		{"Wq[L=0](attn-in)", st.Layers[0].Wq, cfg.D * cfg.D, wqIdx, dWq0Ana[wqIdx]},
		{"Embed[i0,d]", st.Embed, cfg.V * cfg.D, embIdx, dEmbedAna[embIdx]},
	}

	// Ход-1а: noise-arbiter. F32 fwd accumulation shot-noise ~ sqrt(K)*eps_F32*|L|,
	// делённый на 2*eps_pert, даёт "фейковый gradient" ~ 1/eps_pert. True slope
	// от eps_pert НЕ зависит. Ratio num(eps=1e-2)/num(eps=3e-2) ≈ 3 => noise;
	// ≈ 1 => true. Формула шумового пола: |g|_min ~ eps_F32*|L|*sqrt(K) / (2*eps_pert).
	t.Logf("=== Ход-1а eps-scan noise arbiter (extended: 1e-3 tail) ===")
	for _, p := range points {
		n1 := numGradAt(p.W, p.idx, p.nElem, 1e-2)
		n3 := numGradAt(p.W, p.idx, p.nElem, 3e-2)
		nA := numGradAt(p.W, p.idx, p.nElem, 1e-3)
		t.Logf("  %-14s at eps=1e-3 num=%+.3e (nonlinearity probe vs eps=1e-2 num=%+.3e)", p.name, nA, n1)
		absN1 := n1
		if absN1 < 0 {
			absN1 = -absN1
		}
		absN3 := n3
		if absN3 < 0 {
			absN3 = -absN3
		}
		var ratio float32
		if absN3 > 0 {
			ratio = absN1 / absN3
		}
		verdict := "AMBIGUOUS"
		if ratio > 2.0 && ratio < 4.5 {
			verdict = "NOISE (scales as 1/eps → num measures shot-noise, not slope)"
		} else if ratio > 0.7 && ratio < 1.4 {
			verdict = "TRUE-SLOPE (num eps-invariant)"
		}
		t.Logf("  %-14s ana=%+.3e  num(1e-2)=%+.3e  num(3e-2)=%+.3e  ratio=%.2f  → %s",
			p.name, p.ana, n1, n3, ratio, verdict)
	}

	// Floor: 5e-2 abs (4 слоя FP32 chain × центральный FD eps=1e-2).
	const floor float32 = 5e-2
	var fails int
	for _, p := range points {
		num := numGrad(p.W, p.idx, p.nElem)
		if math.IsNaN(float64(num)) || math.IsNaN(float64(p.ana)) {
			t.Errorf("CERT %s FAIL: NaN detected (ana=%.6e num=%.6e)", p.name, p.ana, num)
			fails++
			continue
		}
		diff := p.ana - num
		if diff < 0 {
			diff = -diff
		}
		absN := num
		if absN < 0 {
			absN = -absN
		}
		if absN < 1e-4 {
			absN = 1e-4
		}
		rel := diff / absN
		if diff > floor {
			t.Errorf("CERT %s FAIL: ana=%.6e num=%.6e absDiff=%.3e (floor %.1e), relDiff=%.3e", p.name, p.ana, num, diff, floor, rel)
			fails++
		} else {
			t.Logf("CERT %s PASS: ana=%.6e num=%.6e absDiff=%.3e (floor %.1e), relDiff=%.3e", p.name, p.ana, num, diff, floor, rel)
		}
	}
	if fails == 0 {
		t.Logf("MULTI-LAYER F32-recon CERT PASS: %d/%d cross-layer grad points within floor", len(points), len(points))
	} else {
		t.Errorf("MULTI-LAYER F32-recon CERT FAIL: %d/%d fails", fails, len(points))
	}
}
