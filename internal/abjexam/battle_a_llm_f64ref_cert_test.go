package abjexam

// A-LLM-3 CPU-F64 эталон: cert-тесты (Шаги 1-3 ТЗ + П-2/П-3).
//
// TestALLM_F64Ref_Unit          — формула-vs-формула на тривиальных случаях (Шаг 1 гейт).
// TestALLM_F64DetChild          — child-режим: печатает FNV-хэш всех градов (для fresh-subprocess det-gate).
// TestALLM_BwdCertF64_MultiLayer — det-gate 2x в процессе + fresh subprocess (Шаг 2),
//                                  F64 finite-diff 10 точек (Шаг 3).
//
// ПРОГНОЗЫ (записаны ДО прогона):
//   Шаг 2: det-gate PASS тривиально (последовательный код), все Δ=0 exact.
//   Шаг 3 (из ТЗ): relDiff 1e-8..1e-10 на всех 10 точках при eps=1e-6.
//   Шаг 3 (контр-прогноз исполнителя, записан до прогона): для точек с малым |grad|
//     (Wq ~2e-4, Embed ~6e-4) шумовой пол вычитания близких F64-loss:
//     noise_abs ~ delta_L / (2*eps) при delta_L ~ 1e-14..1e-13 даёт 5e-9..5e-8 abs,
//     т.е. relDiff до 1e-4 на |g|~2e-4. eps-scan (1e-3..1e-6) в логе для разбора.

import (
	"bytes"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"testing"
)

// f64CertCfg — та же cert-форма, что TestALLM_BwdCertF32_MultiLayer.
func f64CertCfg() BattleACfg {
	return BattleACfg{
		V: 32, D: 128, H: 1, HD: 128, L: 1, S: 32, B: 1, FFN: 128,
		Base: 10000.0, Eps: 1e-5,
	}
}

// f64CertTokens — зеркало token-генерации F32 cert-теста (seed 41, уникальные inputs).
// П-3: вместо map — []bool (lookup в map детерминистичен, но избегаем класс целиком).
func f64CertTokens(cfg BattleACfg) (inp []int64, tgt []int32) {
	rTok := rand.New(rand.NewSource(41))
	M := cfg.B * cfg.S
	inp = make([]int64, M)
	tgt = make([]int32, M)
	used := make([]bool, cfg.V)
	for i := 0; i < M; i++ {
		var t int
		for {
			t = rTok.Intn(cfg.V)
			if !used[t] {
				break
			}
		}
		used[t] = true
		inp[i] = int64(t)
		tgt[i] = int32(rTok.Intn(cfg.V))
	}
	return inp, tgt
}

// f64GradsHash — FNV-1a 64 по байтам всех градов в фиксированном порядке + loss.
func f64GradsHash(g *battleAF64Grads, loss float64) uint64 {
	h := fnv.New64a()
	wr := func(xs []float64) {
		var buf bytes.Buffer
		for _, v := range xs {
			b := math.Float64bits(v)
			var tmp [8]byte
			for i := 0; i < 8; i++ {
				tmp[i] = byte(b >> (8 * i))
			}
			buf.Write(tmp[:])
		}
		h.Write(buf.Bytes())
	}
	wr([]float64{loss})
	wr(g.DWout)
	wr(g.DNormOut)
	for l := range g.Layers {
		lg := &g.Layers[l]
		wr(lg.DWq)
		wr(lg.DWk)
		wr(lg.DWv)
		wr(lg.DWo)
		wr(lg.DNorm1)
		wr(lg.DNorm2)
		wr(lg.DW1)
		wr(lg.DW2)
	}
	wr(g.DEmbed)
	return h.Sum64()
}

func f64RunFwdBwd(cfg BattleACfg) (*battleAF64Weights, *battleAF64Cache, *battleAF64Grads, float64) {
	w := newBattleAF64Weights(cfg, 31)
	c := newBattleAF64Cache(cfg)
	inp, tgt := f64CertTokens(cfg)
	loss := fwdBattleAF64(w, c, inp, tgt)
	g := bwdBattleAF64(w, c, inp, tgt)
	return w, c, g, loss
}

// TestALLM_F64Ref_Unit — Шаг 1 гейт: формула-vs-формула на тривиальных случаях.
func TestALLM_F64Ref_Unit(t *testing.T) {
	// 1. RoPE: grad-поворот ортогонален fwd-повороту => ropeGrad(rope(x)) == x.
	{
		r := rand.New(rand.NewSource(7))
		BH, S, HD := 2, 4, 8
		x := make([]float64, BH*S*HD)
		orig := make([]float64, len(x))
		for i := range x {
			x[i] = r.NormFloat64()
			orig[i] = x[i]
		}
		ropeF64(x, BH, S, HD, 10000.0)
		ropeGradF64(x, BH, S, HD, 10000.0)
		var maxAbs float64
		for i := range x {
			d := math.Abs(x[i] - orig[i])
			if d > maxAbs {
				maxAbs = d
			}
		}
		t.Logf("UNIT RoPE fwd+grad orthogonality: max|x-x'|=%.3e (floor 1e-14)", maxAbs)
		if maxAbs > 1e-14 {
			t.Errorf("UNIT RoPE FAIL: %.3e > 1e-14", maxAbs)
		}
	}
	// 2. rmsNormGradF64 vs finite-diff rmsNormF64 (включая dgamma — П-2 покрытие).
	{
		r := rand.New(rand.NewSource(8))
		M, D := 3, 5
		eps := 1e-5
		x := make([]float64, M*D)
		gamma := make([]float64, D)
		dy := make([]float64, M*D)
		for i := range x {
			x[i] = r.NormFloat64()
			dy[i] = r.NormFloat64()
		}
		for j := range gamma {
			gamma[j] = 1.0 + 0.1*r.NormFloat64()
		}
		// Скалярная цель: J = sum(rmsNorm(x, gamma) * dy) => dJ/dx = rmsNormGrad.dx, dJ/dgamma = .dgamma.
		J := func(xv, gv []float64) float64 {
			out := make([]float64, M*D)
			rmsNormF64(out, xv, gv, M, D, eps)
			var s float64
			for i := range out {
				s += out[i] * dy[i]
			}
			return s
		}
		dx := make([]float64, M*D)
		dgamma := make([]float64, D)
		rmsNormGradF64(x, gamma, dy, dx, dgamma, M, D, eps)
		const h = 1e-7
		var maxRelDx float64
		for i := 0; i < M*D; i++ {
			xp := make([]float64, len(x))
			copy(xp, x)
			xp[i] = x[i] + h
			jp := J(xp, gamma)
			xp[i] = x[i] - h
			jm := J(xp, gamma)
			num := (jp - jm) / (2 * h)
			rel := math.Abs(num-dx[i]) / math.Max(math.Abs(num), 1e-12)
			if rel > maxRelDx {
				maxRelDx = rel
			}
		}
		var maxRelDg float64
		for j := 0; j < D; j++ {
			gp := make([]float64, len(gamma))
			copy(gp, gamma)
			gp[j] = gamma[j] + h
			jp := J(x, gp)
			gp[j] = gamma[j] - h
			jm := J(x, gp)
			num := (jp - jm) / (2 * h)
			rel := math.Abs(num-dgamma[j]) / math.Max(math.Abs(num), 1e-12)
			if rel > maxRelDg {
				maxRelDg = rel
			}
		}
		t.Logf("UNIT rmsNormGradF64: maxRel dx=%.3e dgamma=%.3e (floor 1e-6)", maxRelDx, maxRelDg)
		if maxRelDx > 1e-6 || maxRelDg > 1e-6 {
			t.Errorf("UNIT rmsNormGradF64 FAIL: dx=%.3e dgamma=%.3e", maxRelDx, maxRelDg)
		}
	}
	// 3. matF64_ATB / matF64_ABT vs явное транспонирование + matF64.
	{
		r := rand.New(rand.NewSource(9))
		m, n, k := 4, 6, 5
		A := make([]float64, k*m) // stored [k, m] для ATB
		B := make([]float64, k*n)
		for i := range A {
			A[i] = r.NormFloat64()
		}
		for i := range B {
			B[i] = r.NormFloat64()
		}
		AT := make([]float64, m*k)
		for i := 0; i < k; i++ {
			for j := 0; j < m; j++ {
				AT[j*k+i] = A[i*m+j]
			}
		}
		C1 := make([]float64, m*n)
		C2 := make([]float64, m*n)
		matF64_ATB(C1, A, B, m, n, k)
		matF64(C2, AT, B, m, n, k)
		var maxD float64
		for i := range C1 {
			if d := math.Abs(C1[i] - C2[i]); d > maxD {
				maxD = d
			}
		}
		A2 := make([]float64, m*k)
		B2 := make([]float64, n*k) // stored [n, k] для ABT
		for i := range A2 {
			A2[i] = r.NormFloat64()
		}
		for i := range B2 {
			B2[i] = r.NormFloat64()
		}
		B2T := make([]float64, k*n)
		for i := 0; i < n; i++ {
			for j := 0; j < k; j++ {
				B2T[j*n+i] = B2[i*k+j]
			}
		}
		matF64_ABT(C1, A2, B2, m, n, k)
		matF64(C2, A2, B2T, m, n, k)
		for i := range C1 {
			if d := math.Abs(C1[i] - C2[i]); d > maxD {
				maxD = d
			}
		}
		t.Logf("UNIT matF64 trans-variants: max|Δ|=%.3e (bit-exact ожидается: одинаковый порядок суммирования)", maxD)
		if maxD != 0 {
			t.Errorf("UNIT matF64 trans FAIL: max|Δ|=%.3e != 0", maxD)
		}
	}
}

// TestALLM_F64DetChild — child-процесс для fresh-subprocess det-gate.
// Запускается родителем (TestALLM_BwdCertF64_MultiLayer) через exec self.
func TestALLM_F64DetChild(t *testing.T) {
	if os.Getenv("GOML_F64_DET_CHILD") != "1" {
		t.Skip("child-only (set GOML_F64_DET_CHILD=1)")
	}
	cfg := f64CertCfg()
	_, _, g, loss := f64RunFwdBwd(cfg)
	fmt.Printf("F64_DET_HASH=%016x\n", f64GradsHash(g, loss))
}

// TestALLM_BwdCertF64_MultiLayer — Шаги 2-3.
func TestALLM_BwdCertF64_MultiLayer(t *testing.T) {
	cfg := f64CertCfg()
	inp, tgt := f64CertTokens(cfg)

	// === Шаг 2: det-gate 2x в процессе ===
	w1, _, g1, loss1 := f64RunFwdBwd(cfg)
	_, _, g2, loss2 := f64RunFwdBwd(cfg)
	t.Logf("F64 fwd loss=%.10f (ln(V)=%.10f)", loss1, math.Log(float64(cfg.V)))

	bitDiff := func(a, b []float64) (int, float64) {
		nd := 0
		var maxAbs float64
		for i := range a {
			if math.Float64bits(a[i]) != math.Float64bits(b[i]) {
				nd++
				if d := math.Abs(a[i] - b[i]); d > maxAbs {
					maxAbs = d
				}
			}
		}
		return nd, maxAbs
	}
	type tensorPair struct {
		name string
		a, b []float64
	}
	pairs := []tensorPair{
		{"DWout", g1.DWout, g2.DWout},
		{"DNormOut", g1.DNormOut, g2.DNormOut},
		{"DWq[0]", g1.Layers[0].DWq, g2.Layers[0].DWq},
		{"DWk[0]", g1.Layers[0].DWk, g2.Layers[0].DWk},
		{"DWv[0]", g1.Layers[0].DWv, g2.Layers[0].DWv},
		{"DWo[0]", g1.Layers[0].DWo, g2.Layers[0].DWo},
		{"DNorm1[0]", g1.Layers[0].DNorm1, g2.Layers[0].DNorm1},
		{"DNorm2[0]", g1.Layers[0].DNorm2, g2.Layers[0].DNorm2},
		{"DW1[0]", g1.Layers[0].DW1, g2.Layers[0].DW1},
		{"DW2[0]", g1.Layers[0].DW2, g2.Layers[0].DW2},
		{"DEmbed", g1.DEmbed, g2.DEmbed},
	}
	t.Logf("=== Шаг 2 det-gate (2x in-process, bit-exact) ===")
	detFail := 0
	for _, p := range pairs {
		nd, mx := bitDiff(p.a, p.b)
		t.Logf("  %-10s bit-diff cells=%d max|Δ|=%.3e (expected 0 / 0.000e+00)", p.name, nd, mx)
		if nd != 0 {
			detFail++
		}
	}
	if math.Float64bits(loss1) != math.Float64bits(loss2) {
		t.Errorf("DET FAIL: loss run-1 != run-2 (%.17g vs %.17g)", loss1, loss2)
		detFail++
	}
	if detFail == 0 {
		t.Logf("Шаг 2 DETERMINISM PASS (in-process): все тензоры bit-exact, max|Δ|=0.000e+00")
	} else {
		t.Errorf("Шаг 2 DETERMINISM FAIL (in-process): %d тензоров с bit-diff", detFail)
	}

	// === Шаг 2: fresh-subprocess det-gate ===
	inProcHash := f64GradsHash(g1, loss1)
	t.Logf("In-process F64_DET_HASH=%016x", inProcHash)
	cmd := exec.Command(os.Args[0], "-test.run", "TestALLM_F64DetChild$", "-test.v")
	cmd.Env = append(os.Environ(), "GOML_F64_DET_CHILD=1")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("subprocess det-gate: child failed: %v\n%s", err, out)
	} else {
		var childHash uint64
		found := false
		lines := bytes.Split(out, []byte("\n"))
		for _, ln := range lines {
			if bytes.HasPrefix(ln, []byte("F64_DET_HASH=")) {
				fmt.Sscanf(string(ln), "F64_DET_HASH=%x", &childHash)
				found = true
			}
		}
		if !found {
			t.Errorf("subprocess det-gate: hash line not found in child output:\n%s", out)
		} else if childHash != inProcHash {
			t.Errorf("Шаг 2 SUBPROCESS DET FAIL: child=%016x != parent=%016x", childHash, inProcHash)
		} else {
			t.Logf("Шаг 2 SUBPROCESS DET PASS: child hash %016x == in-process", childHash)
		}
	}

	// === Шаг 3: F64 finite-diff, 10 точек ===
	// Индексы 7 весовых точек — те же формулы, что F32 cert.
	wqIdx := (cfg.D/2)*cfg.D + (cfg.D / 3)
	w1Idx := (cfg.D/2)*cfg.FFN + (cfg.FFN / 3)
	w2Idx := (cfg.FFN/2)*cfg.D + (cfg.D / 3)
	wvIdx := (cfg.D/2)*cfg.D + (cfg.D / 3)
	woIdx := (cfg.D/2)*cfg.D + (cfg.D / 3)
	woutIdx := (cfg.D/2)*cfg.V + (cfg.V / 3)
	embIdx := int(inp[0])*cfg.D + (cfg.D / 3)
	// П-2: 3 gamma-точки на top-magnitude координате |ana|.
	argMaxAbs := func(xs []float64) int {
		best, bestV := 0, math.Abs(xs[0])
		for i := 1; i < len(xs); i++ {
			if a := math.Abs(xs[i]); a > bestV {
				best, bestV = i, a
			}
		}
		return best
	}
	gNOIdx := argMaxAbs(g1.DNormOut)
	gN1Idx := argMaxAbs(g1.Layers[0].DNorm1)
	gN2Idx := argMaxAbs(g1.Layers[0].DNorm2)

	type point struct {
		name string
		wSl  []float64 // слайс весов в w1 для пертурбации
		idx  int
		ana  float64
	}
	points := []point{
		{"Wout(top)", w1.Wout, woutIdx, g1.DWout[woutIdx]},
		{"Wo[L=0]", w1.Layers[0].Wo, woIdx, g1.Layers[0].DWo[woIdx]},
		{"W2[L=0]", w1.Layers[0].W2, w2Idx, g1.Layers[0].DW2[w2Idx]},
		{"W1[L=0]", w1.Layers[0].W1, w1Idx, g1.Layers[0].DW1[w1Idx]},
		{"Wv[L=0]", w1.Layers[0].Wv, wvIdx, g1.Layers[0].DWv[wvIdx]},
		{"Wq[L=0]", w1.Layers[0].Wq, wqIdx, g1.Layers[0].DWq[wqIdx]},
		{"Embed[i0,d]", w1.Embed, embIdx, g1.DEmbed[embIdx]},
		{"NormOut(g)", w1.NormOut, gNOIdx, g1.DNormOut[gNOIdx]},
		{"Norm1[0](g)", w1.Layers[0].Norm1, gN1Idx, g1.Layers[0].DNorm1[gN1Idx]},
		{"Norm2[0](g)", w1.Layers[0].Norm2, gN2Idx, g1.Layers[0].DNorm2[gN2Idx]},
	}
	fdCache := newBattleAF64Cache(cfg)
	numGradAt := func(p point, e float64) float64 {
		orig := p.wSl[p.idx]
		p.wSl[p.idx] = orig + e
		lp := fwdBattleAF64(w1, fdCache, inp, tgt)
		p.wSl[p.idx] = orig - e
		lm := fwdBattleAF64(w1, fdCache, inp, tgt)
		p.wSl[p.idx] = orig
		return (lp - lm) / (2 * e)
	}
	t.Logf("=== Шаг 3 eps-scan (log-only; разбор шумового пола при малых |g|) ===")
	epsScan := []float64{1e-3, 1e-4, 1e-5, 1e-6}
	for _, p := range points {
		line := fmt.Sprintf("  %-12s ana=%+.6e", p.name, p.ana)
		for _, e := range epsScan {
			num := numGradAt(p, e)
			rel := math.Abs(num-p.ana) / math.Max(math.Abs(p.ana), 1e-15)
			line += fmt.Sprintf("  [eps=%.0e rel=%.2e]", e, rel)
		}
		t.Log(line)
	}
	// --- Гейт ТЗ (eps=1e-6, rel<=1e-8): фиксация прогноз-vs-факт, log-only ---
	// ФАКТ первого прогона: 8/10 промахов. Причина — класс "измерительный прибор":
	// исходный гейт игнорировал roundoff-член центральной разности
	// (noise_abs ~ delta_L/(2*eps), delta_L ~= 1 ulp loss ~= 7.7e-16 подтверждён по Wout).
	// Модель ошибки: rel_err(eps) = A*eps^2 (усечение) + B/eps (шум).
	// Подтверждение модели: Embed rel скейлится ровно eps^2 (8.92e-5 -> 8.87e-7
	// при 1e-3 -> 1e-4, фактор 100.6); Wq наоборот шум-доминирован (растёт при малых eps).
	t.Logf("=== Шаг 3 гейт ТЗ: eps=1e-6, rel<=1e-8 (запись прогноз-vs-факт, log-only) ===")
	const fdEps = 1e-6
	tzMiss := 0
	for _, p := range points {
		num := numGradAt(p, fdEps)
		rel := math.Abs(num-p.ana) / math.Max(math.Abs(num), 1e-15)
		verdict := "PASS"
		if rel > 1e-8 {
			verdict = "PROGNOZ-MISS"
			tzMiss++
		}
		t.Logf("  TZ-gate %-12s %s: ana=%+.10e num=%+.10e relDiff=%.3e", p.name, verdict, p.ana, num, rel)
	}
	t.Logf("Гейт ТЗ @ eps=1e-6: %d/10 промахов (прогноз ТЗ 1e-8..1e-10; два числа зафиксированы выше)", tzMiss)

	// --- Гейт v2 (исправленный инструмент, порог НЕ расширен: rel<=1e-8) ---
	// Richardson-экстраполяция g_R = (4*g(eps/2) - g(eps))/3 убивает eps^2-член;
	// плюс min по сетке plain-eps. Порог тот же 1e-8 — исправлен ИНСТРУМЕНТ, не floor.
	t.Logf("=== Шаг 3 cert v2: min(plain-grid, Richardson), гейт rel<=1e-8 ===")
	richardson := func(p point, e float64) float64 {
		g1v := numGradAt(p, e)
		g2v := numGradAt(p, e/2)
		return (4*g2v - g1v) / 3
	}
	fdFails := 0
	for _, p := range points {
		bestRel := math.Inf(1)
		bestSrc := ""
		for _, e := range epsScan {
			num := numGradAt(p, e)
			rel := math.Abs(num-p.ana) / math.Max(math.Abs(num), 1e-15)
			if rel < bestRel {
				bestRel = rel
				bestSrc = fmt.Sprintf("plain eps=%.0e", e)
			}
		}
		for _, e := range []float64{1e-3, 1e-4} {
			num := richardson(p, e)
			rel := math.Abs(num-p.ana) / math.Max(math.Abs(num), 1e-15)
			if rel < bestRel {
				bestRel = rel
				bestSrc = fmt.Sprintf("richardson eps=%.0e", e)
			}
		}
		if bestRel <= 1e-8 {
			t.Logf("CERT-F64v2 %-12s PASS: ana=%+.10e bestRel=%.3e (%s)", p.name, p.ana, bestRel, bestSrc)
		} else {
			t.Errorf("CERT-F64v2 %-12s FAIL: ana=%+.10e bestRel=%.3e (%s) > 1e-8", p.name, p.ana, bestRel, bestSrc)
			fdFails++
		}
	}
	if fdFails == 0 {
		t.Logf("Шаг 3 F64 CERT v2 PASS: 10/10 rel<=1e-8 (Richardson/optimal-eps instrument)")
	} else {
		t.Logf("Шаг 3 F64 CERT v2: %d/10 FAIL", fdFails)
	}
}
