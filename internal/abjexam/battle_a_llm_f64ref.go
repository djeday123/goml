package abjexam

// A-LLM-3 CPU-F64 ЭТАЛОН (2026-08-03, Шаг 1 ТЗ HANDOFF_ref_dewrapper.md + П-1..П-3).
//
// bwdBattleAF64/fwdBattleAF64 — полная цепочка cert-формы на CPU в float64.
// НОЛЬ GPU-вызовов. Детерминизм by construction: последовательный код,
// НИКАКИХ итераций по map (П-3), никаких горутин.
//
// П-1 контракт зеркальности: путь зеркалит GPU-F32-recon КАК ЕСТЬ:
//   - attention БЕЗ causal mask (в attention_recon.go маски нет);
//   - RoPE half-split пары (i, i+HD/2), freq = base^(-2i/HD), pos = row % S
//     (зеркало PTX rope_f32, gotorch/v6/cuda/ptx_kernels.go:2429);
//   - RMSNorm: inv_rms = 1/sqrt(mean(x^2)+eps) (зеркало rmsnorm_f32:1646);
//   - CE: per-row NLL c max-subtract logsumexp, dLogits=(softmax-onehot)/M
//     (зеркало cross_entropy_f32, goml/backend/cuda/kernels_b.go:736);
//   - silu-bwd: dsilu/dh = sig + h*sig*(1-sig) (зеркало silu_bwd);
//   - softmax-bwd: dS = P*(dP - sum_k(P*dP)) (зеркало softmax_bwd_f32).
//
// П-2: dgamma зеркалирован с PTX rmsnorm_grad_f32 (ptx_kernels.go:1856):
//   Phase 1: S = sum_j(gamma_j*x_j*dy_j), sum(x^2) — F64-аккумуляторы;
//   Phase 3: dx_j = gamma_j*dy_j*inv_rms - x_j*S*inv_rms^3/D;
//            dgamma_j += dy_j*x_j*inv_rms (у нас: последовательная сумма по строкам,
//            вместо atomicAdd — детерминизм by construction).
//   ВЕРДИКТ ЗЕРКАЛИРОВАНИЯ: PTX-формула совпадает с учебной производной RMSNorm,
//   расхождений НЕ обнаружено (отдельного пункта отчёта не требуется, фиксируется фактом).
//
// dEmbed: последовательный scatter-цикл (зеркало embedding_grad_f32 atomicAdd,
// но порядок фиксирован → bit-детерминизм).

import (
	"math"
	"math/rand"
)

// battleAF64LayerW — host-веса слоя в F64. Значения bit-идентичны F32-инициализации
// NewBattleAState (та же rand-последовательность, конверсия F32->F64 точна).
type battleAF64LayerW struct {
	Norm1 []float64 // [D]
	Wq    []float64 // [D, D]
	Wk    []float64 // [D, D]
	Wv    []float64 // [D, D]
	Wo    []float64 // [D, D]
	Norm2 []float64 // [D]
	W1    []float64 // [D, FFN]
	W2    []float64 // [FFN, D]
}

type battleAF64Weights struct {
	Cfg     BattleACfg
	Embed   []float64 // [V, D]
	Layers  []battleAF64LayerW
	NormOut []float64 // [D]
	Wout    []float64 // [D, V]
}

// newBattleAF64Weights — зеркало NewBattleAState: тот же порядок потребления rand
// (Embed, затем per-layer Wq,Wk,Wv,Wo,W1,W2, затем Wout; гаммы = 1.0 без rand).
func newBattleAF64Weights(cfg BattleACfg, seed int64) *battleAF64Weights {
	r := rand.New(rand.NewSource(seed))
	const scaleW = float32(0.02)
	toF64 := func(h []float32) []float64 {
		out := make([]float64, len(h))
		for i, v := range h {
			out[i] = float64(v)
		}
		return out
	}
	ones := func(n int) []float64 {
		out := make([]float64, n)
		for i := range out {
			out[i] = 1.0
		}
		return out
	}
	w := &battleAF64Weights{Cfg: cfg}
	w.Embed = toF64(initF32Slice(cfg.V*cfg.D, r, scaleW))
	w.Layers = make([]battleAF64LayerW, cfg.L)
	for l := 0; l < cfg.L; l++ {
		lw := &w.Layers[l]
		lw.Norm1 = ones(cfg.D)
		lw.Norm2 = ones(cfg.D)
		lw.Wq = toF64(initF32Slice(cfg.D*cfg.D, r, scaleW))
		lw.Wk = toF64(initF32Slice(cfg.D*cfg.D, r, scaleW))
		lw.Wv = toF64(initF32Slice(cfg.D*cfg.D, r, scaleW))
		lw.Wo = toF64(initF32Slice(cfg.D*cfg.D, r, scaleW))
		lw.W1 = toF64(initF32Slice(cfg.D*cfg.FFN, r, scaleW))
		lw.W2 = toF64(initF32Slice(cfg.FFN*cfg.D, r, scaleW))
	}
	w.NormOut = ones(cfg.D)
	w.Wout = toF64(initF32Slice(cfg.D*cfg.V, r, scaleW))
	return w
}

// battleAF64Cache — все fwd-промежуточные для bwd (аналог snapshots F32-пути).
type battleAF64Cache struct {
	XPreAttn  [][]float64 // [L][M*D]
	Normed1   [][]float64 // [L][M*D]
	QPerm     [][]float64 // [L][BH*S*HD] post-RoPE
	KPerm     [][]float64 // [L][BH*S*HD] post-RoPE
	VPerm     [][]float64 // [L][BH*S*HD]
	P         [][]float64 // [L][BH*S*S] softmax probs
	OAttn     [][]float64 // [L][BH*S*HD]
	QBuf      [][]float64 // [L][M*D] inv-permuted OAttn
	XPreFFN   [][]float64 // [L][M*D]
	Normed2   [][]float64 // [L][M*D]
	FFNHid    [][]float64 // [L][M*FFN]
	FFNSig    [][]float64 // [L][M*FFN]
	FFNSilu   [][]float64 // [L][M*FFN]
	XPreTop   []float64   // [M*D]
	NormedTop []float64   // [M*D]
	Logits    []float64   // [M*V]
	Probs     []float64   // [M*V] softmax(logits) per row
	Loss      float64
}

type battleAF64LayerG struct {
	DNorm1 []float64 // [D]
	DWq    []float64 // [D, D]
	DWk    []float64 // [D, D]
	DWv    []float64 // [D, D]
	DWo    []float64 // [D, D]
	DNorm2 []float64 // [D]
	DW1    []float64 // [D, FFN]
	DW2    []float64 // [FFN, D]
}

type battleAF64Grads struct {
	DEmbed   []float64 // [V, D]
	Layers   []battleAF64LayerG
	DNormOut []float64 // [D]
	DWout    []float64 // [D, V]
}

// --- Примитивы (последовательные, без map) ---

// matF64: C[m,n] = A[m,k] @ B[k,n], row-major.
func matF64(C, A, B []float64, m, n, k int) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var acc float64
			for p := 0; p < k; p++ {
				acc += A[i*k+p] * B[p*n+j]
			}
			C[i*n+j] = acc
		}
	}
}

// matF64_ATB: C[m,n] = A^T @ B, A stored [k,m], B stored [k,n].
func matF64_ATB(C, A, B []float64, m, n, k int) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var acc float64
			for p := 0; p < k; p++ {
				acc += A[p*m+i] * B[p*n+j]
			}
			C[i*n+j] = acc
		}
	}
}

// matF64_ABT: C[m,n] = A @ B^T, A stored [m,k], B stored [n,k].
func matF64_ABT(C, A, B []float64, m, n, k int) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var acc float64
			for p := 0; p < k; p++ {
				acc += A[i*k+p] * B[j*k+p]
			}
			C[i*n+j] = acc
		}
	}
}

// rmsNormF64: y_j = gamma_j * x_j / sqrt(mean(x^2)+eps), по строкам [M, D].
func rmsNormF64(dst, x, gamma []float64, M, D int, eps float64) {
	for row := 0; row < M; row++ {
		var sumX2 float64
		for j := 0; j < D; j++ {
			v := x[row*D+j]
			sumX2 += v * v
		}
		invRms := 1.0 / math.Sqrt(sumX2/float64(D)+eps)
		for j := 0; j < D; j++ {
			dst[row*D+j] = gamma[j] * x[row*D+j] * invRms
		}
	}
}

// rmsNormGradF64 — зеркало PTX rmsnorm_grad_f32 (П-2):
//   S = sum_j(gamma_j*x_j*dy_j); dx_j = gamma_j*dy_j*invRms - x_j*S*invRms^3/D;
//   dgamma_j += dy_j*x_j*invRms (последовательно по строкам).
// dgamma НЕ обнуляется здесь (аккумулирует, как контракт PTX "pre-zeroed").
func rmsNormGradF64(x, gamma, dy, dxOut, dgamma []float64, M, D int, eps float64) {
	for row := 0; row < M; row++ {
		var sumX2, S float64
		for j := 0; j < D; j++ {
			xv := x[row*D+j]
			sumX2 += xv * xv
			S += gamma[j] * xv * dy[row*D+j]
		}
		invRms := 1.0 / math.Sqrt(sumX2/float64(D)+eps)
		invRms3ByD := invRms * invRms * invRms / float64(D)
		for j := 0; j < D; j++ {
			xv := x[row*D+j]
			dyv := dy[row*D+j]
			dxOut[row*D+j] = gamma[j]*dyv*invRms - xv*S*invRms3ByD
			dgamma[j] += dyv * xv * invRms
		}
	}
}

// ropeF64 — зеркало PTX rope_f32: rows = BH*S, pos = row % S, half-split пары.
// r0 = x0*cos - x1*sin; r1 = x0*sin + x1*cos; angle = pos * base^(-2i/HD).
func ropeF64(x []float64, BH, S, HD int, base float64) {
	half := HD / 2
	for row := 0; row < BH*S; row++ {
		pos := row % S
		for i := 0; i < half; i++ {
			freq := math.Pow(base, -2.0*float64(i)/float64(HD))
			angle := float64(pos) * freq
			c, s := math.Cos(angle), math.Sin(angle)
			x0 := x[row*HD+i]
			x1 := x[row*HD+i+half]
			x[row*HD+i] = x0*c - x1*s
			x[row*HD+i+half] = x0*s + x1*c
		}
	}
}

// ropeGradF64 — зеркало rope_grad_f32: поворот на минус-угол.
// dx0 = dy0*cos + dy1*sin; dx1 = -dy0*sin + dy1*cos.
func ropeGradF64(dy []float64, BH, S, HD int, base float64) {
	half := HD / 2
	for row := 0; row < BH*S; row++ {
		pos := row % S
		for i := 0; i < half; i++ {
			freq := math.Pow(base, -2.0*float64(i)/float64(HD))
			angle := float64(pos) * freq
			c, s := math.Cos(angle), math.Sin(angle)
			d0 := dy[row*HD+i]
			d1 := dy[row*HD+i+half]
			dy[row*HD+i] = d0*c + d1*s
			dy[row*HD+i+half] = -d0*s + d1*c
		}
	}
}

// permuteSHD_HSD: src [B][S,H,HD] -> dst [B][H,S,HD] (зеркало transpose_shd_hsd_f32 per batch).
func permuteSHD_HSD(dst, src []float64, B, S, H, HD int) {
	for b := 0; b < B; b++ {
		base := b * S * H * HD
		for s := 0; s < S; s++ {
			for h := 0; h < H; h++ {
				copy(dst[base+(h*S+s)*HD:base+(h*S+s)*HD+HD],
					src[base+(s*H+h)*HD:base+(s*H+h)*HD+HD])
			}
		}
	}
}

// permuteHSD_SHD: src [B][H,S,HD] -> dst [B][S,H,HD] (обратная).
func permuteHSD_SHD(dst, src []float64, B, S, H, HD int) {
	for b := 0; b < B; b++ {
		base := b * S * H * HD
		for h := 0; h < H; h++ {
			for s := 0; s < S; s++ {
				copy(dst[base+(s*H+h)*HD:base+(s*H+h)*HD+HD],
					src[base+(h*S+s)*HD:base+(h*S+s)*HD+HD])
			}
		}
	}
}

// softmaxRowsF64: row-wise softmax c max-subtract, [rows, cols].
func softmaxRowsF64(dst, src []float64, rows, cols int) {
	for r := 0; r < rows; r++ {
		mx := src[r*cols]
		for j := 1; j < cols; j++ {
			if src[r*cols+j] > mx {
				mx = src[r*cols+j]
			}
		}
		var sum float64
		for j := 0; j < cols; j++ {
			e := math.Exp(src[r*cols+j] - mx)
			dst[r*cols+j] = e
			sum += e
		}
		inv := 1.0 / sum
		for j := 0; j < cols; j++ {
			dst[r*cols+j] *= inv
		}
	}
}

// newBattleAF64Cache — аллокация всех промежуточных.
func newBattleAF64Cache(cfg BattleACfg) *battleAF64Cache {
	M := cfg.B * cfg.S
	BH := cfg.B * cfg.H
	c := &battleAF64Cache{}
	allocL := func(n int) [][]float64 {
		out := make([][]float64, cfg.L)
		for l := 0; l < cfg.L; l++ {
			out[l] = make([]float64, n)
		}
		return out
	}
	c.XPreAttn = allocL(M * cfg.D)
	c.Normed1 = allocL(M * cfg.D)
	c.QPerm = allocL(BH * cfg.S * cfg.HD)
	c.KPerm = allocL(BH * cfg.S * cfg.HD)
	c.VPerm = allocL(BH * cfg.S * cfg.HD)
	c.P = allocL(BH * cfg.S * cfg.S)
	c.OAttn = allocL(BH * cfg.S * cfg.HD)
	c.QBuf = allocL(M * cfg.D)
	c.XPreFFN = allocL(M * cfg.D)
	c.Normed2 = allocL(M * cfg.D)
	c.FFNHid = allocL(M * cfg.FFN)
	c.FFNSig = allocL(M * cfg.FFN)
	c.FFNSilu = allocL(M * cfg.FFN)
	c.XPreTop = make([]float64, M*cfg.D)
	c.NormedTop = make([]float64, M*cfg.D)
	c.Logits = make([]float64, M*cfg.V)
	c.Probs = make([]float64, M*cfg.V)
	return c
}

// fwdBattleAF64 — зеркало fwdBattleAF32. Возвращает mean-NLL loss, заполняет cache.
func fwdBattleAF64(w *battleAF64Weights, c *battleAF64Cache,
	inp []int64, tgt []int32) float64 {
	cfg := w.Cfg
	M := cfg.B * cfg.S
	BH := cfg.B * cfg.H
	D, H, HD, S, V, FFN := cfg.D, cfg.H, cfg.HD, cfg.S, cfg.V, cfg.FFN
	scale := 1.0 / math.Sqrt(float64(HD))
	eps := float64(cfg.Eps)
	base := float64(cfg.Base)

	// Embedding gather.
	X := make([]float64, M*D)
	for m := 0; m < M; m++ {
		copy(X[m*D:m*D+D], w.Embed[int(inp[m])*D:int(inp[m])*D+D])
	}

	q := make([]float64, M*D)
	k := make([]float64, M*D)
	v := make([]float64, M*D)
	attnOut := make([]float64, M*D)
	ffnOut := make([]float64, M*D)
	sTemp := make([]float64, BH*S*S)

	for l := 0; l < cfg.L; l++ {
		lw := &w.Layers[l]
		copy(c.XPreAttn[l], X)
		// 1. RMSNorm1.
		rmsNormF64(c.Normed1[l], X, lw.Norm1, M, D, eps)
		// 2. Q/K/V.
		matF64(q, c.Normed1[l], lw.Wq, M, D, D)
		matF64(k, c.Normed1[l], lw.Wk, M, D, D)
		matF64(v, c.Normed1[l], lw.Wv, M, D, D)
		// 3. Permute [B][S,H,HD] -> [B][H,S,HD].
		permuteSHD_HSD(c.QPerm[l], q, cfg.B, S, H, HD)
		permuteSHD_HSD(c.KPerm[l], k, cfg.B, S, H, HD)
		permuteSHD_HSD(c.VPerm[l], v, cfg.B, S, H, HD)
		// 4. RoPE in-place на Q, K.
		ropeF64(c.QPerm[l], BH, S, HD, base)
		ropeF64(c.KPerm[l], BH, S, HD, base)
		// 5. Attention БЕЗ causal mask (П-1): S=Q@K^T*scale; P=softmax; O=P@V, per bh.
		for bh := 0; bh < BH; bh++ {
			qb := c.QPerm[l][bh*S*HD:]
			kb := c.KPerm[l][bh*S*HD:]
			vb := c.VPerm[l][bh*S*HD:]
			sb := sTemp[bh*S*S:]
			for i := 0; i < S; i++ {
				for j := 0; j < S; j++ {
					var acc float64
					for d := 0; d < HD; d++ {
						acc += qb[i*HD+d] * kb[j*HD+d]
					}
					sb[i*S+j] = acc * scale
				}
			}
			softmaxRowsF64(c.P[l][bh*S*S:bh*S*S+S*S], sTemp[bh*S*S:bh*S*S+S*S], S, S)
			matF64(c.OAttn[l][bh*S*HD:bh*S*HD+S*HD], c.P[l][bh*S*S:bh*S*S+S*S], vb[:S*HD], S, HD, S)
		}
		// 6. Inverse permute O -> QBuf [M, D].
		permuteHSD_SHD(c.QBuf[l], c.OAttn[l], cfg.B, S, H, HD)
		// 7. Wo.
		matF64(attnOut, c.QBuf[l], lw.Wo, M, D, D)
		// 8. Residual.
		for i := 0; i < M*D; i++ {
			X[i] += attnOut[i]
		}
		copy(c.XPreFFN[l], X)
		// 9. RMSNorm2.
		rmsNormF64(c.Normed2[l], X, lw.Norm2, M, D, eps)
		// 10. FFN: W1 -> sigmoid -> silu -> W2.
		matF64(c.FFNHid[l], c.Normed2[l], lw.W1, M, FFN, D)
		for i := 0; i < M*FFN; i++ {
			c.FFNSig[l][i] = 1.0 / (1.0 + math.Exp(-c.FFNHid[l][i]))
			c.FFNSilu[l][i] = c.FFNHid[l][i] * c.FFNSig[l][i]
		}
		matF64(ffnOut, c.FFNSilu[l], lw.W2, M, D, FFN)
		// 11. Residual.
		for i := 0; i < M*D; i++ {
			X[i] += ffnOut[i]
		}
	}
	copy(c.XPreTop, X)
	// Top RMSNorm.
	rmsNormF64(c.NormedTop, X, w.NormOut, M, D, eps)
	// Logits.
	matF64(c.Logits, c.NormedTop, w.Wout, M, V, D)
	// CE: per-row NLL (зеркало cross_entropy_f32).
	var lossSum float64
	for m := 0; m < M; m++ {
		row := c.Logits[m*V : m*V+V]
		mx := row[0]
		for j := 1; j < V; j++ {
			if row[j] > mx {
				mx = row[j]
			}
		}
		var sum float64
		for j := 0; j < V; j++ {
			e := math.Exp(row[j] - mx)
			c.Probs[m*V+j] = e
			sum += e
		}
		inv := 1.0 / sum
		for j := 0; j < V; j++ {
			c.Probs[m*V+j] *= inv
		}
		logZ := mx + math.Log(sum)
		lossSum += logZ - row[tgt[m]]
	}
	c.Loss = lossSum / float64(M)
	return c.Loss
}

// bwdBattleAF64 — зеркало bwdBattleAF32 (тот же порядок звеньев). Возвращает свежие grads.
func bwdBattleAF64(w *battleAF64Weights, c *battleAF64Cache,
	inp []int64, tgt []int32) *battleAF64Grads {
	cfg := w.Cfg
	M := cfg.B * cfg.S
	BH := cfg.B * cfg.H
	D, H, HD, S, V, FFN := cfg.D, cfg.H, cfg.HD, cfg.S, cfg.V, cfg.FFN
	scale := 1.0 / math.Sqrt(float64(HD))
	eps := float64(cfg.Eps)
	base := float64(cfg.Base)

	g := &battleAF64Grads{
		DEmbed:   make([]float64, cfg.V*D),
		Layers:   make([]battleAF64LayerG, cfg.L),
		DNormOut: make([]float64, D),
		DWout:    make([]float64, D*V),
	}
	for l := 0; l < cfg.L; l++ {
		lg := &g.Layers[l]
		lg.DNorm1 = make([]float64, D)
		lg.DWq = make([]float64, D*D)
		lg.DWk = make([]float64, D*D)
		lg.DWv = make([]float64, D*D)
		lg.DWo = make([]float64, D*D)
		lg.DNorm2 = make([]float64, D)
		lg.DW1 = make([]float64, D*FFN)
		lg.DW2 = make([]float64, FFN*D)
	}

	// dLogits = (softmax - onehot) / M.
	dLogits := make([]float64, M*V)
	invM := 1.0 / float64(M)
	for m := 0; m < M; m++ {
		for j := 0; j < V; j++ {
			dLogits[m*V+j] = c.Probs[m*V+j] * invM
		}
		dLogits[m*V+int(tgt[m])] -= invM
	}
	// dWout = NormedTop^T @ dLogits.
	matF64_ATB(g.DWout, c.NormedTop, dLogits, D, V, M)
	// dNormedTop = dLogits @ Wout^T.
	dNormedTop := make([]float64, M*D)
	matF64_ABT(dNormedTop, dLogits, w.Wout, M, D, V)
	// Top RMSNorm bwd (dgamma -> DNormOut).
	dX := make([]float64, M*D)
	rmsNormGradF64(c.XPreTop, w.NormOut, dNormedTop, dX, g.DNormOut, M, D, eps)

	dFFNOut := make([]float64, M*D)
	dFFNSilu := make([]float64, M*FFN)
	dFFNHidden := make([]float64, M*FFN)
	dNormed := make([]float64, M*D)
	dTmp := make([]float64, M*D)
	dQBuf := make([]float64, M*D)
	dO := make([]float64, BH*S*HD)
	dQPerm := make([]float64, BH*S*HD)
	dKPerm := make([]float64, BH*S*HD)
	dVPerm := make([]float64, BH*S*HD)
	dP := make([]float64, BH*S*S)
	dS := make([]float64, BH*S*S)
	dQ := make([]float64, M*D)
	dK := make([]float64, M*D)
	dV := make([]float64, M*D)

	for l := cfg.L - 1; l >= 0; l-- {
		lw := &w.Layers[l]
		lg := &g.Layers[l]

		// FFN bwd.
		copy(dFFNOut, dX)
		// dFFNSilu = dFFNOut @ W2^T. W2 stored [FFN, D].
		matF64_ABT(dFFNSilu, dFFNOut, lw.W2, M, FFN, D)
		// dW2 = FFNSilu^T @ dFFNOut.
		matF64_ATB(lg.DW2, c.FFNSilu[l], dFFNOut, FFN, D, M)
		// silu_bwd: dh = dsilu * (sig + h*sig*(1-sig)).
		for i := 0; i < M*FFN; i++ {
			h := c.FFNHid[l][i]
			sg := c.FFNSig[l][i]
			dFFNHidden[i] = dFFNSilu[i] * (sg + h*sg*(1.0-sg))
		}
		// dNormed(FFN) = dFFNHidden @ W1^T. W1 stored [D, FFN].
		matF64_ABT(dNormed, dFFNHidden, lw.W1, M, D, FFN)
		// dW1 = Normed2^T @ dFFNHidden.
		matF64_ATB(lg.DW1, c.Normed2[l], dFFNHidden, D, FFN, M)
		// dRMSNorm2 -> dTmp; dgamma2; dX += dTmp.
		rmsNormGradF64(c.XPreFFN[l], lw.Norm2, dNormed, dTmp, lg.DNorm2, M, D, eps)
		for i := 0; i < M*D; i++ {
			dX[i] += dTmp[i]
		}
		// Attention bwd. dAttnOut = dX.
		dAttnOut := dX // читается только; копия не нужна (зеркало b.Copy сохраняет значение)
		// dQBuf = dAttnOut @ Wo^T.
		matF64_ABT(dQBuf, dAttnOut, lw.Wo, M, D, D)
		// dWo = QBuf^T @ dAttnOut.
		matF64_ATB(lg.DWo, c.QBuf[l], dAttnOut, D, D, M)
		// dQBuf [B][S,H,HD] -> dO [B][H,S,HD].
		permuteSHD_HSD(dO, dQBuf, cfg.B, S, H, HD)
		// attnReconstructBwd per bh: dV=P^T@dO; dP=dO@V^T; dS=P*(dP-rowsum(P*dP));
		// dQ=dS@K*scale; dK=dS^T@Q*scale.
		for bh := 0; bh < BH; bh++ {
			pb := c.P[l][bh*S*S : bh*S*S+S*S]
			vb := c.VPerm[l][bh*S*HD : bh*S*HD+S*HD]
			qb := c.QPerm[l][bh*S*HD : bh*S*HD+S*HD]
			kb := c.KPerm[l][bh*S*HD : bh*S*HD+S*HD]
			dob := dO[bh*S*HD : bh*S*HD+S*HD]
			matF64_ATB(dVPerm[bh*S*HD:bh*S*HD+S*HD], pb, dob, S, HD, S)
			matF64_ABT(dP[bh*S*S:bh*S*S+S*S], dob, vb, S, S, HD)
			// softmax bwd по строкам.
			dpb := dP[bh*S*S : bh*S*S+S*S]
			dsb := dS[bh*S*S : bh*S*S+S*S]
			for i := 0; i < S; i++ {
				var dot float64
				for j := 0; j < S; j++ {
					dot += pb[i*S+j] * dpb[i*S+j]
				}
				for j := 0; j < S; j++ {
					dsb[i*S+j] = pb[i*S+j] * (dpb[i*S+j] - dot)
				}
			}
			// dQ = dS @ K * scale (K stored [S, HD] — без транспонирования).
			matF64(dQPerm[bh*S*HD:bh*S*HD+S*HD], dsb, kb, S, HD, S)
			// dK = dS^T @ Q * scale.
			matF64_ATB(dKPerm[bh*S*HD:bh*S*HD+S*HD], dsb, qb, S, HD, S)
			for i := 0; i < S*HD; i++ {
				dQPerm[bh*S*HD+i] *= scale
				dKPerm[bh*S*HD+i] *= scale
			}
		}
		// RoPE bwd на dQPerm, dKPerm.
		ropeGradF64(dQPerm, BH, S, HD, base)
		ropeGradF64(dKPerm, BH, S, HD, base)
		// Inverse permute [B][H,S,HD] -> [B][S,H,HD].
		permuteHSD_SHD(dQ, dQPerm, cfg.B, S, H, HD)
		permuteHSD_SHD(dK, dKPerm, cfg.B, S, H, HD)
		permuteHSD_SHD(dV, dVPerm, cfg.B, S, H, HD)
		// dNormed = dQ@Wq^T + dK@Wk^T + dV@Wv^T.
		matF64_ABT(dNormed, dQ, lw.Wq, M, D, D)
		matF64_ABT(dTmp, dK, lw.Wk, M, D, D)
		for i := 0; i < M*D; i++ {
			dNormed[i] += dTmp[i]
		}
		matF64_ABT(dTmp, dV, lw.Wv, M, D, D)
		for i := 0; i < M*D; i++ {
			dNormed[i] += dTmp[i]
		}
		// Weight grads.
		matF64_ATB(lg.DWq, c.Normed1[l], dQ, D, D, M)
		matF64_ATB(lg.DWk, c.Normed1[l], dK, D, D, M)
		matF64_ATB(lg.DWv, c.Normed1[l], dV, D, D, M)
		// dRMSNorm1 -> dTmp; dgamma1; dX += dTmp.
		rmsNormGradF64(c.XPreAttn[l], lw.Norm1, dNormed, dTmp, lg.DNorm1, M, D, eps)
		for i := 0; i < M*D; i++ {
			dX[i] += dTmp[i]
		}
	}
	// dEmbed: последовательный scatter (детерминизм by construction).
	for m := 0; m < M; m++ {
		row := int(inp[m]) * D
		for j := 0; j < D; j++ {
			g.DEmbed[row+j] += dX[m*D+j]
		}
	}
	return g
}
