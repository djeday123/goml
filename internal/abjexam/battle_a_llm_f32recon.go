package abjexam

// A-LLM-2 v2 F32-only ЭТАЛОН (2026-07-27).
//
// User rejected v1 sign-of-life as correctness proof: caveat 1 (per-layer Normed
// подменён финальным) — известная мат. ошибка. v2 попытка снапшотов в общем
// BattleAScratch триггерила FA-instability (feedback-fa-fwd-scratch-alloc-instability).
//
// Новый подход: **отдельная F32-only stack, FA не вызывается**. Attention =
// attnReconstructFwd (Stage 4 cert PASS). Снапшоты безопасны потому что нет FA.
// Это ЭТАЛОН навсегда — цель которого производить корректные grads.
//
// FA-path bwdBattleA (в battle_a_llm_bwd.go) — production speed path,
// A/B verified vs этот эталон при встройке FA-bwd (следующий шаг после cert).

import (
	"fmt"
	"math"
	"runtime"
	"unsafe"

	"github.com/djeday123/goml/backend"
	gotorchAdapter "github.com/djeday123/goml/backend/gotorch"
	"github.com/djeday123/goml/core"
)

// BattleAScratchF32 -- отдельный scratch для F32-only fwd/bwd.
// Не пересекается с BattleAScratch (FA-path). Все snapshots per-layer.
type BattleAScratchF32 struct {
	Cfg BattleACfg
	M   int
	BH  int
	// Input
	InputGPU backend.Storage // int64 [M]
	// Per-step intermediates
	X       backend.Storage // F32 [M, D] — current residual stream
	Normed  backend.Storage // F32 [M, D]
	Q       backend.Storage // F32 [M, D]
	K       backend.Storage // F32 [M, D]
	V       backend.Storage // F32 [M, D]
	QPerm   backend.Storage // F32 [BH, S, HD] post-RoPE
	KPerm   backend.Storage // F32 [BH, S, HD] post-RoPE
	VPerm   backend.Storage // F32 [BH, S, HD]
	OAttn   backend.Storage // F32 [BH, S, HD] attnRecon output
	SScr    backend.Storage // F32 [BH, S, S] attn recompute S
	PScr    backend.Storage // F32 [BH, S, S] softmax P
	QScaled backend.Storage // F32 [BH, S, HD] Q*scale for recon
	AttnOut backend.Storage // F32 [M, D] after inverse permute + Wo
	FFNHid  backend.Storage // F32 [M, FFN]
	FFNSig  backend.Storage // F32 [M, FFN]
	FFNSilu backend.Storage // F32 [M, FFN]
	FFNOut  backend.Storage // F32 [M, D]
	// Output
	Logits backend.Storage // F32 [M, V]
	Loss   backend.Storage // F32 [M]
	GradL  backend.Storage // F32 [M, V]
	// Per-layer snapshots for correct bwd (caveat-1 fix).
	// Memory для BattleA (B=1, D=512, S=2048, L=4):
	//   2 M*D*4 * L snapshots (XPreAttn+XPreFFN) = 32 MB
	//   3 BH*S*HD*4 * L (QPerm/KPerm/VPerm) = 48 MB
	//   1 BH*S*HD*4 * L (OAttn) = 16 MB
	//   + XPreTop M*D*4 = 4 MB
	//   Total ~100 MB (влезает свободно).
	XPreAttn  []backend.Storage // F32 [M, D] × L
	XPreFFN   []backend.Storage // F32 [M, D] × L
	QPermSnap []backend.Storage // F32 [BH, S, HD] × L
	KPermSnap []backend.Storage
	VPermSnap []backend.Storage
	OAttnSnap []backend.Storage // F32 [BH, S, HD] × L (для recompute P в bwd)
	XPreTop   backend.Storage   // F32 [M, D]
}

func NewBattleAScratchF32(cfg BattleACfg, b backend.Backend) (*BattleAScratchF32, error) {
	sc := &BattleAScratchF32{Cfg: cfg, M: cfg.B * cfg.S, BH: cfg.B * cfg.H}
	M, BH, D, FFN, V, S, HD := sc.M, sc.BH, cfg.D, cfg.FFN, cfg.V, cfg.S, cfg.HD
	al := func(bytes int) backend.Storage {
		s, err := b.Alloc(bytes)
		if err != nil {
			panic(err)
		}
		return s
	}
	sc.InputGPU = al(M * 8)
	sc.X = al(M * D * 4)
	sc.Normed = al(M * D * 4)
	sc.Q = al(M * D * 4)
	sc.K = al(M * D * 4)
	sc.V = al(M * D * 4)
	sc.QPerm = al(BH * S * HD * 4)
	sc.KPerm = al(BH * S * HD * 4)
	sc.VPerm = al(BH * S * HD * 4)
	sc.OAttn = al(BH * S * HD * 4)
	sc.SScr = al(BH * S * S * 4)
	sc.PScr = al(BH * S * S * 4)
	sc.QScaled = al(BH * S * HD * 4)
	sc.AttnOut = al(M * D * 4)
	sc.FFNHid = al(M * FFN * 4)
	sc.FFNSig = al(M * FFN * 4)
	sc.FFNSilu = al(M * FFN * 4)
	sc.FFNOut = al(M * D * 4)
	sc.Logits = al(M * V * 4)
	sc.Loss = al(M * 4)
	sc.GradL = al(M * V * 4)
	sc.XPreAttn = make([]backend.Storage, cfg.L)
	sc.XPreFFN = make([]backend.Storage, cfg.L)
	sc.QPermSnap = make([]backend.Storage, cfg.L)
	sc.KPermSnap = make([]backend.Storage, cfg.L)
	sc.VPermSnap = make([]backend.Storage, cfg.L)
	sc.OAttnSnap = make([]backend.Storage, cfg.L)
	for l := 0; l < cfg.L; l++ {
		sc.XPreAttn[l] = al(M * D * 4)
		sc.XPreFFN[l] = al(M * D * 4)
		sc.QPermSnap[l] = al(BH * S * HD * 4)
		sc.KPermSnap[l] = al(BH * S * HD * 4)
		sc.VPermSnap[l] = al(BH * S * HD * 4)
		sc.OAttnSnap[l] = al(BH * S * HD * 4)
	}
	sc.XPreTop = al(M * D * 4)
	return sc, nil
}

func (sc *BattleAScratchF32) FreeAll(b backend.Backend) {
	if sc == nil {
		return
	}
	free := func(s backend.Storage) {
		if s != nil {
			b.Free(s)
		}
	}
	free(sc.InputGPU)
	free(sc.X)
	free(sc.Normed)
	free(sc.Q)
	free(sc.K)
	free(sc.V)
	free(sc.QPerm)
	free(sc.KPerm)
	free(sc.VPerm)
	free(sc.OAttn)
	free(sc.SScr)
	free(sc.PScr)
	free(sc.QScaled)
	free(sc.AttnOut)
	free(sc.FFNHid)
	free(sc.FFNSig)
	free(sc.FFNSilu)
	free(sc.FFNOut)
	free(sc.Logits)
	free(sc.Loss)
	free(sc.GradL)
	for i := range sc.XPreAttn {
		free(sc.XPreAttn[i])
		free(sc.XPreFFN[i])
		free(sc.QPermSnap[i])
		free(sc.KPermSnap[i])
		free(sc.VPermSnap[i])
		free(sc.OAttnSnap[i])
	}
	free(sc.XPreTop)
}

// fwdBattleAF32 -- ЧИСТЫЙ F32 forward. FA не вызывается. attnReconstructFwd в attention block.
// Все per-layer snapshots сохраняются для точного bwd.
func fwdBattleAF32(b backend.Backend, st *BattleAState, sc *BattleAScratchF32,
	inputTokens []int64, targetTokens []int32) (loss float64, err error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	cfg := st.Cfg
	M := sc.M
	BH := sc.BH
	D := cfg.D
	H := cfg.H
	HD := cfg.HD
	S := cfg.S
	B := cfg.B
	V := cfg.V

	gtB, ok := b.(*gotorchAdapter.Backend)
	if !ok {
		return 0, fmt.Errorf("fwdBattleAF32 requires gotorch adapter, got %T", b)
	}
	softmaxScale := float32(1.0 / math.Sqrt(float64(HD)))

	// Upload input tokens.
	if _, err := uploadInto(b, sc.InputGPU, int64ToBytes(inputTokens)); err != nil {
		return 0, fmt.Errorf("upload tokens: %w", err)
	}
	// Embedding.
	if err := b.Embedding(sc.X, st.Embed, sc.InputGPU, V, D, M, core.Float32); err != nil {
		return 0, fmt.Errorf("embedding: %w", err)
	}

	batchStride := uintptr(S * D * 4)
	for l := 0; l < cfg.L; l++ {
		lw := &st.Layers[l]

		// SNAPSHOT X BEFORE RMSNorm1.
		if err := b.Copy(sc.XPreAttn[l], sc.X, M*D*4); err != nil {
			return 0, fmt.Errorf("layer %d snapshot XPreAttn: %w", l, err)
		}
		// 1. RMSNorm1.
		if err := gtB.RMSNormF32(sc.X, lw.Norm1, sc.Normed, M, D, cfg.Eps); err != nil {
			return 0, fmt.Errorf("layer %d RMSNorm1: %w", l, err)
		}
		// 2. Q/K/V matmuls.
		if err := b.MatMul(sc.Q, sc.Normed, lw.Wq, core.Shape{M, D}, core.Shape{D, D}, core.Float32); err != nil {
			return 0, fmt.Errorf("layer %d Wq: %w", l, err)
		}
		if err := b.MatMul(sc.K, sc.Normed, lw.Wk, core.Shape{M, D}, core.Shape{D, D}, core.Float32); err != nil {
			return 0, fmt.Errorf("layer %d Wk: %w", l, err)
		}
		if err := b.MatMul(sc.V, sc.Normed, lw.Wv, core.Shape{M, D}, core.Shape{D, D}, core.Float32); err != nil {
			return 0, fmt.Errorf("layer %d Wv: %w", l, err)
		}
		// 3. Permute [S, H, hd] -> [H, S, hd] per batch.
		qBase := devPtr(sc.Q)
		kBase := devPtr(sc.K)
		vBase := devPtr(sc.V)
		qpBase := devPtr(sc.QPerm)
		kpBase := devPtr(sc.KPerm)
		vpBase := devPtr(sc.VPerm)
		for bi := 0; bi < B; bi++ {
			off := uintptr(bi) * batchStride
			if err := launchTransposeSHD_HSDPtr(b, qpBase+off, qBase+off, S, H, HD); err != nil {
				return 0, fmt.Errorf("layer %d Q permute batch %d: %w", l, bi, err)
			}
			if err := launchTransposeSHD_HSDPtr(b, kpBase+off, kBase+off, S, H, HD); err != nil {
				return 0, fmt.Errorf("layer %d K permute batch %d: %w", l, bi, err)
			}
			if err := launchTransposeSHD_HSDPtr(b, vpBase+off, vBase+off, S, H, HD); err != nil {
				return 0, fmt.Errorf("layer %d V permute batch %d: %w", l, bi, err)
			}
		}
		// 4. RoPE in-place.
		if err := gtB.RoPEF32(sc.QPerm, sc.QPerm, BH, 1, S, HD, cfg.Base); err != nil {
			return 0, fmt.Errorf("layer %d RoPE Q: %w", l, err)
		}
		if err := gtB.RoPEF32(sc.KPerm, sc.KPerm, BH, 1, S, HD, cfg.Base); err != nil {
			return 0, fmt.Errorf("layer %d RoPE K: %w", l, err)
		}
		// SNAPSHOT post-RoPE Q/K/V.
		if err := b.Copy(sc.QPermSnap[l], sc.QPerm, BH*S*HD*4); err != nil {
			return 0, fmt.Errorf("layer %d snapshot QPerm: %w", l, err)
		}
		if err := b.Copy(sc.KPermSnap[l], sc.KPerm, BH*S*HD*4); err != nil {
			return 0, fmt.Errorf("layer %d snapshot KPerm: %w", l, err)
		}
		if err := b.Copy(sc.VPermSnap[l], sc.VPerm, BH*S*HD*4); err != nil {
			return 0, fmt.Errorf("layer %d snapshot VPerm: %w", l, err)
		}
		// 5. F32 attention via attnReconstructFwd (NO FA, NO FP8).
		if err := attnReconstructFwd(b, gtB, sc.QPerm, sc.KPerm, sc.VPerm,
			sc.OAttn, sc.SScr, sc.PScr, sc.QScaled,
			BH, S, HD, softmaxScale); err != nil {
			return 0, fmt.Errorf("layer %d attn recon: %w", l, err)
		}
		// SNAPSHOT OAttn.
		if err := b.Copy(sc.OAttnSnap[l], sc.OAttn, BH*S*HD*4); err != nil {
			return 0, fmt.Errorf("layer %d snapshot OAttn: %w", l, err)
		}
		// 6. Inverse permute [H, S, hd] -> [S, H, hd] into sc.Q (reused как buf).
		oBase := devPtr(sc.OAttn)
		qBufBase := devPtr(sc.Q)
		for bi := 0; bi < B; bi++ {
			off := uintptr(bi) * batchStride
			if err := launchTransposeSHD_HSDPtr(b, qBufBase+off, oBase+off, H, S, HD); err != nil {
				return 0, fmt.Errorf("layer %d O inv-permute batch %d: %w", l, bi, err)
			}
		}
		// 7. Wo matmul.
		if err := b.MatMul(sc.AttnOut, sc.Q, lw.Wo, core.Shape{M, D}, core.Shape{D, D}, core.Float32); err != nil {
			return 0, fmt.Errorf("layer %d Wo: %w", l, err)
		}
		// 8. Residual X += AttnOut.
		if err := b.Add(sc.X, sc.X, sc.AttnOut, core.Shape{M, D}, core.Shape{M, D}, core.Shape{M, D}, core.Float32); err != nil {
			return 0, fmt.Errorf("layer %d residual attn: %w", l, err)
		}
		// SNAPSHOT X BEFORE RMSNorm2.
		if err := b.Copy(sc.XPreFFN[l], sc.X, M*D*4); err != nil {
			return 0, fmt.Errorf("layer %d snapshot XPreFFN: %w", l, err)
		}
		// 9. RMSNorm2.
		if err := gtB.RMSNormF32(sc.X, lw.Norm2, sc.Normed, M, D, cfg.Eps); err != nil {
			return 0, fmt.Errorf("layer %d RMSNorm2: %w", l, err)
		}
		// 10. FFN: W1 -> Sigmoid -> Silu(x * sigmoid(x)) -> W2.
		if err := b.MatMul(sc.FFNHid, sc.Normed, lw.W1, core.Shape{M, D}, core.Shape{D, cfg.FFN}, core.Float32); err != nil {
			return 0, fmt.Errorf("layer %d FFN W1: %w", l, err)
		}
		if err := b.Sigmoid(sc.FFNSig, sc.FFNHid, core.Shape{M, cfg.FFN}, core.Float32); err != nil {
			return 0, fmt.Errorf("layer %d Sigmoid: %w", l, err)
		}
		if err := b.Mul(sc.FFNSilu, sc.FFNHid, sc.FFNSig, core.Shape{M, cfg.FFN}, core.Shape{M, cfg.FFN}, core.Shape{M, cfg.FFN}, core.Float32); err != nil {
			return 0, fmt.Errorf("layer %d FFN Silu Mul: %w", l, err)
		}
		if err := b.MatMul(sc.FFNOut, sc.FFNSilu, lw.W2, core.Shape{M, cfg.FFN}, core.Shape{cfg.FFN, D}, core.Float32); err != nil {
			return 0, fmt.Errorf("layer %d FFN W2: %w", l, err)
		}
		// 11. Residual X += FFNOut.
		if err := b.Add(sc.X, sc.X, sc.FFNOut, core.Shape{M, D}, core.Shape{M, D}, core.Shape{M, D}, core.Float32); err != nil {
			return 0, fmt.Errorf("layer %d residual FFN: %w", l, err)
		}
	}
	// SNAPSHOT X BEFORE top RMSNorm.
	if err := b.Copy(sc.XPreTop, sc.X, M*D*4); err != nil {
		return 0, fmt.Errorf("snapshot XPreTop: %w", err)
	}
	// Final RMSNorm.
	if err := gtB.RMSNormF32(sc.X, st.NormOut, sc.Normed, M, D, cfg.Eps); err != nil {
		return 0, fmt.Errorf("final RMSNorm: %w", err)
	}
	// Logits = Normed @ Wout.
	if err := b.MatMul(sc.Logits, sc.Normed, st.Wout, core.Shape{M, D}, core.Shape{D, V}, core.Float32); err != nil {
		return 0, fmt.Errorf("Wout: %w", err)
	}
	// CE-fused kernel.
	targetsBytes := make([]byte, M*4)
	for i, v := range targetTokens {
		u := uint32(v)
		targetsBytes[i*4+0] = byte(u)
		targetsBytes[i*4+1] = byte(u >> 8)
		targetsBytes[i*4+2] = byte(u >> 16)
		targetsBytes[i*4+3] = byte(u >> 24)
	}
	targetsGPU, err := b.ToDevice(backend.CUDADevice(0), &cpuStorage{data: targetsBytes})
	if err != nil {
		return 0, fmt.Errorf("upload targets: %w", err)
	}
	defer b.Free(targetsGPU)

	logitsPtr := devPtr(sc.Logits)
	targetsPtr := devPtr(targetsGPU)
	lossPtr := devPtr(sc.Loss)
	gradLogitsPtr := devPtr(sc.GradL)
	nRows := uint32(M)
	vocab := uint32(V)
	invBs := float32(1.0 / float32(M))
	ceParams := []unsafe.Pointer{
		unsafe.Pointer(&logitsPtr), unsafe.Pointer(&targetsPtr),
		unsafe.Pointer(&lossPtr), unsafe.Pointer(&gradLogitsPtr),
		unsafe.Pointer(&nRows), unsafe.Pointer(&vocab), unsafe.Pointer(&invBs),
	}
	if lc, ok := b.(interface {
		Launch(name string, gx, gy, gz, bx, by, bz uint32, params []unsafe.Pointer) error
	}); ok {
		if err := lc.Launch("cross_entropy_f32", uint32(M), 1, 1, 256, 1, 1, ceParams); err != nil {
			return 0, fmt.Errorf("CE kernel: %w", err)
		}
	}
	if s, ok := b.(interface{ Sync() error }); ok {
		s.Sync()
	}
	lossHost := gpuToHost(b, sc.Loss, M)
	var lossSum float64
	for _, v := range lossHost {
		lossSum += float64(v)
	}
	return lossSum / float64(M), nil
}

// bwdBattleAF32 -- бackward для F32-only stack. Использует snapshots для корректных grads.
// grads.DEmbed/DWout/DNormOut/L[*].D* обнуляются через zeroGrads (RMSNormGrad атомно аккумулирует).
func bwdBattleAF32(b backend.Backend, st *BattleAState, sc *BattleAScratchF32,
	bs *BattleABwdScratch, grads *BattleAGrads) error {
	cfg := st.Cfg
	M := sc.M
	BH := sc.BH
	D := cfg.D
	H := cfg.H
	HD := cfg.HD
	S := cfg.S
	B := cfg.B
	V := cfg.V
	FFN := cfg.FFN
	softmaxScale := float32(1.0 / math.Sqrt(float64(HD)))

	gtB, ok := b.(*gotorchAdapter.Backend)
	if !ok {
		return fmt.Errorf("bwdBattleAF32 requires gotorch adapter, got %T", b)
	}
	// D-протокол helper: plain b.MatMul с host-transpose для transB=true (M,n,k,F,T pattern).
	// Bug class isolated: MatMulF32Ex дает EXACT ZERO partial для (M=32,X,X,F,T).
	matmulPlainTB := func(dst, aStor, bStor backend.Storage, m, n, k int) error {
		bH := gpuToHost(b, bStor, n*k)
		bT := make([]float32, k*n)
		for i := 0; i < n; i++ {
			for j := 0; j < k; j++ {
				bT[j*n+i] = bH[i*k+j]
			}
		}
		tmp, err := b.Alloc(k * n * 4)
		if err != nil {
			return err
		}
		defer b.Free(tmp)
		if _, err := uploadInto(b, tmp, f32ToBytes(bT)); err != nil {
			return err
		}
		return b.MatMul(dst, aStor, tmp, core.Shape{m, k}, core.Shape{k, n}, core.Float32)
	}
	_ = matmulPlainTB

	// PRE-WARMUP отключён — вызывал регрессию (все iters dead).

	// Б-1 DIAG helper (top-level).
	diagTop := func(name string, s backend.Storage, n int) {
		h := gpuToHost(b, s, n)
		var mx float32
		for _, v := range h {
			a := v
			if a < 0 { a = -a }
			if a > mx { mx = a }
		}
		fmt.Printf("Б-1 TOP  %-16s |max|=%.3e\n", name, mx)
	}
	diagTop("GradL(from-CE)", sc.GradL, M*V)
	diagTop("sc.Normed(fwd)", sc.Normed, M*D)
	// Output layer: dLogits already in sc.GradL from CE kernel.
	// dWout = Normed^T @ dLogits (Normed = post-top RMSNorm output, from sc.Normed at fwd end).
	if err := gtB.MatMulF32Ex(sc.Normed, sc.GradL, grads.DWout, D, V, M, true, false); err != nil {
		return fmt.Errorf("dWout: %w", err)
	}
	diagTop("dWout(1st)", grads.DWout, D*V)
	// F-протокол probe: dNormedTop = dLogits @ Wout^T через plain b.MatMul (host-transpose Wout).
	// Если downstream грады улучшаются - MatMulF32Ex wrapper дает systematic ошибку не только
	// на exact-zero (attnReconstructBwd) но и на 30-70% недо/переоценку.
	{
		woutH := gpuToHost(b, st.Wout, D*V)
		woutT := make([]float32, V*D)
		for i := 0; i < D; i++ {
			for j := 0; j < V; j++ {
				woutT[j*D+i] = woutH[i*V+j]
			}
		}
		tmp, err := b.Alloc(V * D * 4)
		if err != nil {
			return fmt.Errorf("dNormedTop transpose alloc: %w", err)
		}
		defer b.Free(tmp)
		if _, err := uploadInto(b, tmp, f32ToBytes(woutT)); err != nil {
			return fmt.Errorf("dNormedTop transpose upload: %w", err)
		}
		if err := b.MatMul(bs.DNormedTop, sc.GradL, tmp, core.Shape{M, V}, core.Shape{V, D}, core.Float32); err != nil {
			return fmt.Errorf("dNormedTop (plain probe): %w", err)
		}
		fmt.Printf("F-PROBE: dNormedTop computed via plain b.MatMul (not MatMulF32Ex)\n")
	}
	diagTop("dNormedTop(1st)", bs.DNormedTop, M*D)
	// Top RMSNorm bwd — использует XPreTop.
	if err := gtB.RMSNormGradF32(sc.XPreTop, st.NormOut, bs.DNormedTop, bs.DX, grads.DNormOut, M, D, cfg.Eps); err != nil {
		return fmt.Errorf("top RMSNormGrad: %w", err)
	}
	// D-протокол snapshot: bs.DX сразу после RMSNormGradTop (до residual-adds).
	if err := b.Copy(bs.DXAfterTop, bs.DX, M*D*4); err != nil {
		return fmt.Errorf("D-snap DXAfterTop: %w", err)
	}

	batchStride := uintptr(S * D * 4)
	// Per-layer bwd (REVERSE).
	for l := cfg.L - 1; l >= 0; l-- {
		lw := &st.Layers[l]
		lg := &grads.Layers[l]

		// FFN bwd.
		if err := b.Copy(bs.DFFNOut, bs.DX, M*D*4); err != nil {
			return fmt.Errorf("layer %d dFFNOut copy: %w", l, err)
		}
		// D-протокол snapshot DFFNOut сразу после copy (только L=0 для cert).
		if l == 0 {
			if err := b.Copy(bs.DFFNOutSnap, bs.DFFNOut, M*D*4); err != nil {
				return fmt.Errorf("D-snap DFFNOutSnap: %w", err)
			}
		}
		// D-протокол localized: MatMulF32Ex(dFFNOut, W2, dFFNSilu, F, T) даёт
		// EXACT ZERO на некоторых cells (context-dep zero bug). Fix: plain b.MatMul.
		// dFFNSilu[M, FFN] = dFFNOut[M, D] @ W2^T[D, FFN]. W2 stored [FFN, D].
		{
			w2H := gpuToHost(b, lw.W2, FFN*D)
			w2T := make([]float32, D*FFN)
			for i := 0; i < FFN; i++ {
				for j := 0; j < D; j++ {
					w2T[j*FFN+i] = w2H[i*D+j]
				}
			}
			tmp, err := b.Alloc(D * FFN * 4)
			if err != nil {
				return fmt.Errorf("layer %d dFFNSilu W2T alloc: %w", l, err)
			}
			if _, err := uploadInto(b, tmp, f32ToBytes(w2T)); err != nil {
				b.Free(tmp)
				return fmt.Errorf("layer %d dFFNSilu W2T upload: %w", l, err)
			}
			if err := b.MatMul(bs.DFFNSilu, bs.DFFNOut, tmp, core.Shape{M, D}, core.Shape{D, FFN}, core.Float32); err != nil {
				b.Free(tmp)
				return fmt.Errorf("layer %d dFFNSilu plain: %w", l, err)
			}
			b.Free(tmp)
		}
		if l == 0 {
			if err := b.Copy(bs.DFFNSiluSnap, bs.DFFNSilu, M*FFN*4); err != nil {
				return fmt.Errorf("D-snap DFFNSiluSnap: %w", err)
			}
		}
		// dW2 = FFNSilu^T @ dFFNOut. FFNSilu пересчитываем: RMSNorm2(XPreFFN[l]) -> Normed2, W1@Normed2 -> hidden, sigmoid, silu.
		// Для простоты: recompute FFNSilu.
		if err := gtB.RMSNormF32(sc.XPreFFN[l], lw.Norm2, bs.NormedRecomp, M, D, cfg.Eps); err != nil {
			return fmt.Errorf("layer %d recompute Normed2: %w", l, err)
		}
		if err := b.MatMul(sc.FFNHid, bs.NormedRecomp, lw.W1, core.Shape{M, D}, core.Shape{D, FFN}, core.Float32); err != nil {
			return fmt.Errorf("layer %d recompute FFN W1: %w", l, err)
		}
		if err := b.Sigmoid(sc.FFNSig, sc.FFNHid, core.Shape{M, FFN}, core.Float32); err != nil {
			return fmt.Errorf("layer %d recompute Sigmoid: %w", l, err)
		}
		if err := b.Mul(sc.FFNSilu, sc.FFNHid, sc.FFNSig, core.Shape{M, FFN}, core.Shape{M, FFN}, core.Shape{M, FFN}, core.Float32); err != nil {
			return fmt.Errorf("layer %d recompute FFN Silu Mul: %w", l, err)
		}
		if err := gtB.MatMulF32Ex(sc.FFNSilu, bs.DFFNOut, lg.DW2, FFN, D, M, true, false); err != nil {
			return fmt.Errorf("layer %d dW2: %w", l, err)
		}
		// silu_bwd -> dFFNHidden.
		if err := launchSiluBwd(b, devPtr(bs.DFFNSilu), devPtr(sc.FFNHid), devPtr(sc.FFNSig), devPtr(bs.DFFNHidden), M*FFN); err != nil {
			return fmt.Errorf("layer %d silu_bwd: %w", l, err)
		}
		if l == 0 {
			if err := b.Copy(bs.DFFNHidSnap, bs.DFFNHidden, M*FFN*4); err != nil {
				return fmt.Errorf("D-snap DFFNHidSnap: %w", err)
			}
		}
		// dNormed(FFN) = dFFNHidden @ W1^T.
		if err := matmulPlainTB(bs.DNormed, bs.DFFNHidden, lw.W1, M, D, FFN); err != nil {
			return fmt.Errorf("layer %d dNormed(FFN) plain: %w", l, err)
		}
		// dW1 = Normed2^T @ dFFNHidden.
		if err := gtB.MatMulF32Ex(bs.NormedRecomp, bs.DFFNHidden, lg.DW1, D, FFN, M, true, false); err != nil {
			return fmt.Errorf("layer %d dW1: %w", l, err)
		}
		// dRMSNorm2 -> add to dX.
		if err := gtB.RMSNormGradF32(sc.XPreFFN[l], lw.Norm2, bs.DNormed, bs.DAttnOut, lg.DNorm2, M, D, cfg.Eps); err != nil {
			return fmt.Errorf("layer %d RMSNormGrad2: %w", l, err)
		}
		if err := b.Add(bs.DX, bs.DX, bs.DAttnOut, core.Shape{M, D}, core.Shape{M, D}, core.Shape{M, D}, core.Float32); err != nil {
			return fmt.Errorf("layer %d dX += dRMSNorm2: %w", l, err)
		}
		// Attention bwd. dAttnOut = dX (residual).
		if err := b.Copy(bs.DAttnOut, bs.DX, M*D*4); err != nil {
			return fmt.Errorf("layer %d dAttnOut copy: %w", l, err)
		}
		// dQ_buf = dAttnOut @ Wo^T.
		if err := matmulPlainTB(bs.DQ, bs.DAttnOut, lw.Wo, M, D, D); err != nil {
			return fmt.Errorf("layer %d dQ_buf(Wo) plain: %w", l, err)
		}
		// dWo = Q_buf^T @ dAttnOut. Q_buf = inv-permuted OAttnSnap[l].
		// Recompute Q_buf via inverse permute OAttnSnap[l] into sc.Q.
		oBase := devPtr(sc.OAttnSnap[l])
		qBufBase := devPtr(sc.Q)
		for bi := 0; bi < B; bi++ {
			off := uintptr(bi) * batchStride
			if err := launchTransposeSHD_HSDPtr(b, qBufBase+off, oBase+off, H, S, HD); err != nil {
				return fmt.Errorf("layer %d Q_buf recompute inv-permute batch %d: %w", l, bi, err)
			}
		}
		if err := gtB.MatMulF32Ex(sc.Q, bs.DAttnOut, lg.DWo, D, D, M, true, false); err != nil {
			return fmt.Errorf("layer %d dWo: %w", l, err)
		}
		// dQ_buf [M, D] -> inverse permute to dOAttn [BH, S, HD].
		dqBase := devPtr(bs.DQ)
		doBase := devPtr(bs.DOF32)
		for bi := 0; bi < B; bi++ {
			off := uintptr(bi) * batchStride
			if err := launchTransposeSHD_HSDPtr(b, doBase+off, dqBase+off, S, H, HD); err != nil {
				return fmt.Errorf("layer %d dOAttn permute batch %d: %w", l, bi, err)
			}
		}
		// Б-1 DIAG: chain magnitudes for layer 0.
		diagAbs := func(name string, s backend.Storage, n int) {
			h := gpuToHost(b, s, n)
			var mx float32
			for _, v := range h {
				a := v
				if a < 0 { a = -a }
				if a > mx { mx = a }
			}
			fmt.Printf("Б-1 L=%d %-16s |max|=%.3e\n", l, name, mx)
		}
		if l == 0 {
			diagAbs("dAttnOut(post-copy)", bs.DAttnOut, M*D)
			diagAbs("dQ_buf(dA@Wo^T)", bs.DQ, M*D)
			diagAbs("dOF32(inv-perm)", bs.DOF32, BH*S*HD)
		}
		// attnReconstructBwd на per-layer snapshots. Нужен P (пересчитаем).
		if err := attnReconstructFwd(b, gtB, sc.QPermSnap[l], sc.KPermSnap[l], sc.VPermSnap[l],
			bs.OReconDbg, bs.SReconTemp, bs.PRecon, bs.QScaledTmp,
			BH, S, HD, softmaxScale); err != nil {
			return fmt.Errorf("layer %d recon fwd for P: %w", l, err)
		}
		if err := attnReconstructBwd(b, gtB, sc.QPermSnap[l], sc.KPermSnap[l], sc.VPermSnap[l], bs.PRecon,
			bs.DOF32, bs.DQPerm, bs.DKPerm, bs.DVPerm, bs.DPTemp, bs.DSTemp,
			BH, S, HD, softmaxScale); err != nil {
			return fmt.Errorf("layer %d recon bwd: %w", l, err)
		}
		// Ход-1а/2а: активирую для локализации звена смерти в attnReconstructBwd.
		if l == 0 {
			doH := gpuToHost(b, bs.DOF32, BH*S*HD)
			dqpH := gpuToHost(b, bs.DQPerm, BH*S*HD)
			dpH := gpuToHost(b, bs.DPTemp, BH*S*S)
			dsH := gpuToHost(b, bs.DSTemp, BH*S*S)
			var maxDO, maxDQP, maxDP, maxDS float32
			for _, v := range doH {
				if a := v; a < 0 {
					if -a > maxDO {
						maxDO = -a
					}
				} else if a > maxDO {
					maxDO = a
				}
			}
			for _, v := range dqpH {
				if a := v; a < 0 {
					if -a > maxDQP {
						maxDQP = -a
					}
				} else if a > maxDQP {
					maxDQP = a
				}
			}
			for _, v := range dpH {
				if a := v; a < 0 {
					if -a > maxDP {
						maxDP = -a
					}
				} else if a > maxDP {
					maxDP = a
				}
			}
			for _, v := range dsH {
				if a := v; a < 0 {
					if -a > maxDS {
						maxDS = -a
					}
				} else if a > maxDS {
					maxDS = a
				}
			}
			fmt.Printf("DEBUG bwd L=0: |dOF32|=%.3e, |dP|=%.3e, |dS|=%.3e, |dQPerm|=%.3e\n", maxDO, maxDP, maxDS, maxDQP)
		}
		// RoPE bwd Q, K.
		if err := gtB.RoPEGradF32(bs.DQPerm, bs.DQPerm, BH, 1, S, HD, cfg.Base); err != nil {
			return fmt.Errorf("layer %d RoPE bwd Q: %w", l, err)
		}
		if err := gtB.RoPEGradF32(bs.DKPerm, bs.DKPerm, BH, 1, S, HD, cfg.Base); err != nil {
			return fmt.Errorf("layer %d RoPE bwd K: %w", l, err)
		}
		if l == 0 {
			diagAbs("dQPerm(post-RoPE-bwd)", bs.DQPerm, BH*S*HD)
			diagAbs("dKPerm(post-RoPE-bwd)", bs.DKPerm, BH*S*HD)
		}
		// Inverse permute [BH, S, HD] -> [B, S, H, HD].
		dqpBase := devPtr(bs.DQPerm)
		dkpBase := devPtr(bs.DKPerm)
		dvpBase := devPtr(bs.DVPerm)
		dQfBase := devPtr(bs.DQ)
		dKfBase := devPtr(bs.DK)
		dVfBase := devPtr(bs.DV)
		for bi := 0; bi < B; bi++ {
			off := uintptr(bi) * batchStride
			if err := launchTransposeSHD_HSDPtr(b, dQfBase+off, dqpBase+off, H, S, HD); err != nil {
				return fmt.Errorf("layer %d dQ inv-permute batch %d: %w", l, bi, err)
			}
			if err := launchTransposeSHD_HSDPtr(b, dKfBase+off, dkpBase+off, H, S, HD); err != nil {
				return fmt.Errorf("layer %d dK inv-permute batch %d: %w", l, bi, err)
			}
			if err := launchTransposeSHD_HSDPtr(b, dVfBase+off, dvpBase+off, H, S, HD); err != nil {
				return fmt.Errorf("layer %d dV inv-permute batch %d: %w", l, bi, err)
			}
		}
		if l == 0 {
			diagAbs("dQ(post-inv-perm)", bs.DQ, M*D)
			diagAbs("dK(post-inv-perm)", bs.DK, M*D)
		}
		// Recompute Normed1 = RMSNorm(XPreAttn[l], Norm1) для dWq/dWk/dWv.
		if err := gtB.RMSNormF32(sc.XPreAttn[l], lw.Norm1, bs.NormedRecomp, M, D, cfg.Eps); err != nil {
			return fmt.Errorf("layer %d recompute Normed1: %w", l, err)
		}
		if l == 0 {
			diagAbs("NormedRecomp1", bs.NormedRecomp, M*D)
		}
		// dNormed = dQ @ Wq^T + dK @ Wk^T + dV @ Wv^T.
		if err := matmulPlainTB(bs.DNormed, bs.DQ, lw.Wq, M, D, D); err != nil {
			return fmt.Errorf("layer %d dNormed(Q) plain: %w", l, err)
		}
		if err := matmulPlainTB(bs.DAttnOut, bs.DK, lw.Wk, M, D, D); err != nil {
			return fmt.Errorf("layer %d dNormed(K) plain: %w", l, err)
		}
		if err := b.Add(bs.DNormed, bs.DNormed, bs.DAttnOut, core.Shape{M, D}, core.Shape{M, D}, core.Shape{M, D}, core.Float32); err != nil {
			return fmt.Errorf("layer %d dNormed sum K: %w", l, err)
		}
		if err := matmulPlainTB(bs.DAttnOut, bs.DV, lw.Wv, M, D, D); err != nil {
			return fmt.Errorf("layer %d dNormed(V) plain: %w", l, err)
		}
		if err := b.Add(bs.DNormed, bs.DNormed, bs.DAttnOut, core.Shape{M, D}, core.Shape{M, D}, core.Shape{M, D}, core.Float32); err != nil {
			return fmt.Errorf("layer %d dNormed sum V: %w", l, err)
		}
		// Weight grads.
		if err := gtB.MatMulF32Ex(bs.NormedRecomp, bs.DQ, lg.DWq, D, D, M, true, false); err != nil {
			return fmt.Errorf("layer %d dWq: %w", l, err)
		}
		if l == 0 {
			diagAbs("dWq(final)", lg.DWq, D*D)
		}
		if err := gtB.MatMulF32Ex(bs.NormedRecomp, bs.DK, lg.DWk, D, D, M, true, false); err != nil {
			return fmt.Errorf("layer %d dWk: %w", l, err)
		}
		if err := gtB.MatMulF32Ex(bs.NormedRecomp, bs.DV, lg.DWv, D, D, M, true, false); err != nil {
			return fmt.Errorf("layer %d dWv: %w", l, err)
		}
		// dRMSNorm1 -> add to dX.
		if err := gtB.RMSNormGradF32(sc.XPreAttn[l], lw.Norm1, bs.DNormed, bs.DAttnOut, lg.DNorm1, M, D, cfg.Eps); err != nil {
			return fmt.Errorf("layer %d RMSNormGrad1: %w", l, err)
		}
		if err := b.Add(bs.DX, bs.DX, bs.DAttnOut, core.Shape{M, D}, core.Shape{M, D}, core.Shape{M, D}, core.Float32); err != nil {
			return fmt.Errorf("layer %d dX += dRMSNorm1: %w", l, err)
		}
	}
	// Embedding bwd.
	if err := gtB.EmbeddingGradF32(sc.InputGPU, bs.DX, grads.DEmbed, V, D, M); err != nil {
		return fmt.Errorf("dEmbed: %w", err)
	}
	if s, ok := b.(interface{ Sync() error }); ok {
		s.Sync()
	}
	return nil
}

// trainStepBattleAF32 -- fwd + bwd + SGD (host update).
func trainStepBattleAF32(b backend.Backend, st *BattleAState, sc *BattleAScratchF32,
	bs *BattleABwdScratch, grads *BattleAGrads,
	inputTokens []int64, targetTokens []int32, lr float32) (float64, error) {
	loss, err := fwdBattleAF32(b, st, sc, inputTokens, targetTokens)
	if err != nil {
		return 0, fmt.Errorf("fwd: %w", err)
	}
	if err := zeroGrads(b, grads); err != nil {
		return 0, fmt.Errorf("zeroGrads: %w", err)
	}
	if err := bwdBattleAF32(b, st, sc, bs, grads); err != nil {
		return 0, fmt.Errorf("bwd: %w", err)
	}
	if err := applySGD(b, st, grads, lr); err != nil {
		return 0, fmt.Errorf("sgd: %w", err)
	}
	return loss, nil
}
