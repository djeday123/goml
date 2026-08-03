package abjexam

// A-LLM-1 Этап 3 (2026-07-25): forward трансформера BattleA с FA-fwd-with-L.
//
// НОВЫЙ Step (параллельно nn.LLM, не через него) — потому что nn.LLM.ForwardWithCache
// = чистый CPU host loops (см. A_LLM1.md §1). Использует наш battle-инструментарий +
// gotorch RMSNorm/RoPE + свежие FA-fwd-with-L (G1) и bwd .so (G2, не встроен).
//
// BattleA config (fixed hd=128 FA-требование):
//   V=32000, D=512, H=4, hd=128, L=4, S=2048, B ∈ {1, 4}.
//   FFN=4*D=2048 (standard 2-layer, Silu активация).
//   Peak memory ~3.8 GB (см. A_LLM1.md §1.4).
//
// B=1 smoke first (layout [1,S,H,hd] → [H,S,hd] через один permute — тривиально).
// B=4 после — repack + bit-exact сверка logits[0] против B=1.
//
// FP8 amax: per-tensor (user decision #1), сверка первого батча против F32-reconstruct.
// qk_descale: свёртка scale = softmax_scale * scale_Q * scale_K в один параметр (user #2).
// Embedding bwd: kernel P3 с честным floor 1e-4 (user #3, ядро корректно, тест P3
//   имел ошибочный floor — задача #79 отдельно).

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"unsafe"

	"github.com/djeday123/goml/backend"
	gotorchAdapter "github.com/djeday123/goml/backend/gotorch"
	gomlcuda "github.com/djeday123/goml/backend/cuda"
	"github.com/djeday123/goml/core"
)

// BattleACfg — фиксированный transformer config.
type BattleACfg struct {
	V     int // vocab
	D     int // embed dim (must be H*hd)
	H     int // num heads
	HD    int // head dim (must be 128 for FA)
	L     int // num layers
	S     int // seq len
	B     int // batch
	FFN   int // FFN hidden (typically 4*D)
	Base  float32 // RoPE base
	Eps   float32 // RMSNorm eps
}

func DefaultBattleACfg(batch int) BattleACfg {
	return BattleACfg{
		V:    32000,
		D:    512,
		H:    4,
		HD:   128,
		L:    4,
		S:    2048,
		B:    batch,
		FFN:  2048, // standard 4*D, no SwiGLU для упрощения
		Base: 10000.0,
		Eps:  1e-5,
	}
}

// BattleAWeights — веса одного transformer layer.
type BattleAWeights struct {
	Norm1  backend.Storage // RMSNorm gamma [D] F32
	Wq     backend.Storage // [D, D] F32 (D = H*hd)
	Wk     backend.Storage // [D, D] F32
	Wv     backend.Storage // [D, D] F32
	Wo     backend.Storage // [D, D] F32
	Norm2  backend.Storage // RMSNorm gamma [D] F32
	W1     backend.Storage // [D, FFN] F32
	W2     backend.Storage // [FFN, D] F32
}

// BattleAState — все веса + top norm + output projection. (AdamW state в первой
// сессии Этапа 3 не нужен — bwd не запускаем.)
type BattleAState struct {
	Cfg     BattleACfg
	Embed   backend.Storage      // [V, D] F32
	Layers  []BattleAWeights     // L штук
	NormOut backend.Storage      // [D] F32
	Wout    backend.Storage      // [D, V] F32
}

// initF32 — random NormFloat32*scale (детерминистично).
func initF32Slice(n int, r *rand.Rand, scale float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(r.NormFloat64()) * scale
	}
	return out
}

// NewBattleAState — деtermиничная случайная инициализация + upload на GPU.
func NewBattleAState(cfg BattleACfg, r *rand.Rand, b backend.Backend) (*BattleAState, error) {
	if cfg.D != cfg.H*cfg.HD {
		return nil, fmt.Errorf("BattleACfg: D=%d must equal H*hd=%d*%d=%d", cfg.D, cfg.H, cfg.HD, cfg.H*cfg.HD)
	}
	// hd=128 требуется только для FA path. F32-only bypasses this check.
	upload := func(data []float32) (backend.Storage, error) {
		return b.ToDevice(backend.CUDADevice(0), &cpuStorage{data: f32ToBytes(data)})
	}
	st := &BattleAState{Cfg: cfg}
	var err error

	// Init scale ~ 0.02 для weights (стандарт для LLM init).
	scaleW := float32(0.02)
	// Embedding
	if st.Embed, err = upload(initF32Slice(cfg.V*cfg.D, r, scaleW)); err != nil {
		return nil, fmt.Errorf("Embed upload: %w", err)
	}
	// Layers
	st.Layers = make([]BattleAWeights, cfg.L)
	for l := 0; l < cfg.L; l++ {
		lw := &st.Layers[l]
		// Norms — gamma init = 1.0
		gamma1 := make([]float32, cfg.D)
		gamma2 := make([]float32, cfg.D)
		for i := range gamma1 {
			gamma1[i] = 1.0
			gamma2[i] = 1.0
		}
		if lw.Norm1, err = upload(gamma1); err != nil {
			return nil, err
		}
		if lw.Norm2, err = upload(gamma2); err != nil {
			return nil, err
		}
		if lw.Wq, err = upload(initF32Slice(cfg.D*cfg.D, r, scaleW)); err != nil {
			return nil, err
		}
		if lw.Wk, err = upload(initF32Slice(cfg.D*cfg.D, r, scaleW)); err != nil {
			return nil, err
		}
		if lw.Wv, err = upload(initF32Slice(cfg.D*cfg.D, r, scaleW)); err != nil {
			return nil, err
		}
		if lw.Wo, err = upload(initF32Slice(cfg.D*cfg.D, r, scaleW)); err != nil {
			return nil, err
		}
		if lw.W1, err = upload(initF32Slice(cfg.D*cfg.FFN, r, scaleW)); err != nil {
			return nil, err
		}
		if lw.W2, err = upload(initF32Slice(cfg.FFN*cfg.D, r, scaleW)); err != nil {
			return nil, err
		}
	}
	// Final norm + output
	gammaOut := make([]float32, cfg.D)
	for i := range gammaOut {
		gammaOut[i] = 1.0
	}
	if st.NormOut, err = upload(gammaOut); err != nil {
		return nil, err
	}
	if st.Wout, err = upload(initF32Slice(cfg.D*cfg.V, r, scaleW)); err != nil {
		return nil, err
	}
	return st, nil
}

func (st *BattleAState) FreeAll(b backend.Backend) {
	if st == nil {
		return
	}
	if st.Embed != nil {
		b.Free(st.Embed)
	}
	for _, l := range st.Layers {
		b.Free(l.Norm1)
		b.Free(l.Norm2)
		b.Free(l.Wq)
		b.Free(l.Wk)
		b.Free(l.Wv)
		b.Free(l.Wo)
		b.Free(l.W1)
		b.Free(l.W2)
	}
	if st.NormOut != nil {
		b.Free(st.NormOut)
	}
	if st.Wout != nil {
		b.Free(st.Wout)
	}
}

// BattleAScratch — все буферы hot-loop, pre-allocated (паттерн A-2 BattleScratch).
type BattleAScratch struct {
	Cfg BattleACfg
	M   int // batch*seq
	BH  int // batch*heads

	// Per-step: input + intermediates
	InputGPU backend.Storage // int64 tokens [M]

	// Layer scratches (могут переиспользоваться между layers)
	X          backend.Storage // F32 [M, D] — current residual stream
	Normed     backend.Storage // F32 [M, D]
	Q          backend.Storage // F32 [M, D] (виден как [B, S, H, hd])
	K          backend.Storage // F32 [M, D]
	V          backend.Storage // F32 [M, D]
	QPerm      backend.Storage // F32 [BH, S, hd] — after permute [B,S,H,hd]->[BH,S,hd]
	KPerm      backend.Storage // F32 [BH, S, hd]
	VPerm      backend.Storage // F32 [BH, S, hd]
	QFP8       backend.Storage // uint8 [BH, S, hd]
	KFP8       backend.Storage // uint8 [BH, S, hd]
	VFP8       backend.Storage // uint8 [BH, S, hd]
	ScaleQ     backend.Storage // F32 [1] per-tensor
	ScaleK     backend.Storage // F32 [1]
	ScaleV     backend.Storage // F32 [1]
	AmaxQ      backend.Storage // F32 [1]
	AmaxK      backend.Storage // F32 [1]
	AmaxV      backend.Storage // F32 [1]
	OFP16      backend.Storage // uint16 [BH, S, hd]
	LGPU       backend.Storage // F32 [BH, S] — LSE от FA
	OF32       backend.Storage // F32 [BH, S, hd] — after FP16->F32 cast
	AttnOut    backend.Storage // F32 [M, D] — after inverse permute + Wo
	FFNHidden  backend.Storage // F32 [M, FFN]
	FFNSigmoid backend.Storage // F32 [M, FFN] — sigmoid(hidden) для Silu
	FFNSilu    backend.Storage // F32 [M, FFN] — hidden * sigmoid = Silu
	FFNOut     backend.Storage // F32 [M, D]

	// Output stage
	Logits backend.Storage // F32 [M, V]
	Loss   backend.Storage // F32 [M]
	GradL  backend.Storage // F32 [M, V] — CE gradient (не используется в fwd-only, для bwd)
}

func NewBattleAScratch(cfg BattleACfg, b backend.Backend) (*BattleAScratch, error) {
	sc := &BattleAScratch{Cfg: cfg, M: cfg.B * cfg.S, BH: cfg.B * cfg.H}
	M, BH, D, FFN, V, S, HD := sc.M, sc.BH, cfg.D, cfg.FFN, cfg.V, cfg.S, cfg.HD
	al := func(bytes int) backend.Storage {
		s, err := b.Alloc(bytes)
		if err != nil {
			panic(err)
		}
		return s
	}
	sc.InputGPU = al(M * 8) // int64
	sc.X = al(M * D * 4)
	sc.Normed = al(M * D * 4)
	sc.Q = al(M * D * 4)
	sc.K = al(M * D * 4)
	sc.V = al(M * D * 4)
	sc.QPerm = al(BH * S * HD * 4)
	sc.KPerm = al(BH * S * HD * 4)
	sc.VPerm = al(BH * S * HD * 4)
	sc.QFP8 = al(BH * S * HD)
	sc.KFP8 = al(BH * S * HD)
	sc.VFP8 = al(BH * S * HD)
	sc.ScaleQ = al(4)
	sc.ScaleK = al(4)
	sc.ScaleV = al(4)
	sc.AmaxQ = al(4)
	sc.AmaxK = al(4)
	sc.AmaxV = al(4)
	sc.OFP16 = al(BH * S * HD * 2)
	sc.LGPU = al(BH * S * 4)
	sc.OF32 = al(BH * S * HD * 4)
	sc.AttnOut = al(M * D * 4)
	sc.FFNHidden = al(M * FFN * 4)
	sc.FFNSigmoid = al(M * FFN * 4)
	sc.FFNSilu = al(M * FFN * 4)
	sc.FFNOut = al(M * D * 4)
	sc.Logits = al(M * V * 4)
	sc.Loss = al(M * 4)
	sc.GradL = al(M * V * 4)
	return sc, nil
}

func (sc *BattleAScratch) FreeAll(b backend.Backend) {
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
	free(sc.QFP8)
	free(sc.KFP8)
	free(sc.VFP8)
	free(sc.ScaleQ)
	free(sc.ScaleK)
	free(sc.ScaleV)
	free(sc.AmaxQ)
	free(sc.AmaxK)
	free(sc.AmaxV)
	free(sc.OFP16)
	free(sc.LGPU)
	free(sc.OF32)
	free(sc.AttnOut)
	free(sc.FFNHidden)
	free(sc.FFNSigmoid)
	free(sc.FFNSilu)
	free(sc.FFNOut)
	free(sc.Logits)
	free(sc.Loss)
	free(sc.GradL)
}

// launchTransposeSHD_HSDPtr -- вызов transpose_shd_hsd_f32 PTX kernel через
// device pointers (для batch loop с pointer-arithmetic offsets).
// src [S, H, hd] -> dst [H, S, hd]. Grid (H, S, 1), Block (hd, 1, 1).
func launchTransposeSHD_HSDPtr(b backend.Backend, dstPtr, srcPtr uintptr, S, H, HD int) error {
	l, ok := b.(interface {
		Launch(name string, gx, gy, gz, bx, by, bz uint32, params []unsafe.Pointer) error
	})
	if !ok {
		return fmt.Errorf("backend has no Launch")
	}
	Su, Hu, HDu := uint32(S), uint32(H), uint32(HD)
	params := []unsafe.Pointer{
		unsafe.Pointer(&dstPtr), unsafe.Pointer(&srcPtr),
		unsafe.Pointer(&Su), unsafe.Pointer(&Hu), unsafe.Pointer(&HDu),
	}
	return l.Launch("transpose_shd_hsd_f32", uint32(H), uint32(S), 1, uint32(HD), 1, 1, params)
}

// fwdBattleA -- forward pass BattleA transformer с FA-fwd-with-L.
// Возвращает scalar loss (per-step-averaged NLL).
//
// LockOSThread: caller (test) должен обёртывать. Здесь используется FA-контекст
// который требует thread pinning.
func fwdBattleA(b backend.Backend, st *BattleAState, sc *BattleAScratch,
	faCtx *gomlcuda.FAContext, inputTokens []int64, targetTokens []int32) (loss float64, err error) {
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
		return 0, fmt.Errorf("fwdBattleA requires gotorch adapter, got %T", b)
	}

	// Upload input tokens (int64 [M]).
	if _, err := uploadInto(b, sc.InputGPU, int64ToBytes(inputTokens)); err != nil {
		return 0, fmt.Errorf("upload tokens: %w", err)
	}

	// -- Embedding: X = Embed[input] -- [M, D]
	if err := b.Embedding(sc.X, st.Embed, sc.InputGPU, V, D, M, core.Float32); err != nil {
		return 0, fmt.Errorf("embedding: %w", err)
	}

	// FA scale composition base: softmax_scale = 1/sqrt(hd).
	softmaxScale := float32(1.0 / math.Sqrt(float64(HD)))

	// -- Layer loop --
	for l := 0; l < cfg.L; l++ {
		lw := &st.Layers[l]

		// 1. RMSNorm(X) -> Normed [M, D]
		if err := gtB.RMSNormF32(sc.X, lw.Norm1, sc.Normed, M, D, cfg.Eps); err != nil {
			return 0, fmt.Errorf("layer %d RMSNorm1: %w", l, err)
		}

		// 2. Q/K/V projections. Wq/Wk/Wv are [D, D], input [M, D], out [M, D].
		if err := b.MatMul(sc.Q, sc.Normed, lw.Wq,
			core.Shape{M, D}, core.Shape{D, D}, core.Float32); err != nil {
			return 0, fmt.Errorf("layer %d Wq: %w", l, err)
		}
		if err := b.MatMul(sc.K, sc.Normed, lw.Wk,
			core.Shape{M, D}, core.Shape{D, D}, core.Float32); err != nil {
			return 0, fmt.Errorf("layer %d Wk: %w", l, err)
		}
		if err := b.MatMul(sc.V, sc.Normed, lw.Wv,
			core.Shape{M, D}, core.Shape{D, D}, core.Float32); err != nil {
			return 0, fmt.Errorf("layer %d Wv: %w", l, err)
		}

		// 3. RoPE применяется в-place на Q, K (не V).
		// RoPEF32 signature: (x, out, batch, heads, seqLen, headDim, base)
		// Input layout: [batch, heads, seqLen, headDim].
		// Наш Q/K сейчас [M, D] = [B*S, H*hd]. View as [B, S, H, hd]. RoPE ожидает
		// [B, H, S, hd] — это другой layout! Нужно permute первым, потом RoPE.
		//
		// УПРОЩЕНИЕ: RoPE'll be applied AFTER permute к [BH, S, hd]. RoPEF32 batch=BH,
		// heads=1 работает на [BH, 1, S, hd] что эквивалентно [BH, S, hd].

		// 4. Permute Q,K,V from [B, S, H, hd] -> [B*H, S, hd] (per-batch loop).
		// Per batch b: src slice offset b*S*D, dst offset b*H*S*hd.
		// Since S*D = S*H*hd = H*S*hd, offsets identical (per-batch stride same).
		//
		// SIMPLIFY B=1: no batch loop needed, treat whole buffer as one transpose call.
		// For B>1: loop over batches, each block transpose.
		Bel := M * D * 4 / B // bytes per batch
		_ = Bel
		qBase := devPtr(sc.Q)
		kBase := devPtr(sc.K)
		vBase := devPtr(sc.V)
		qpBase := devPtr(sc.QPerm)
		kpBase := devPtr(sc.KPerm)
		vpBase := devPtr(sc.VPerm)
		batchStride := uintptr(S * D * 4)
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

		// 5. RoPE в-place на QPerm, KPerm (layout [BH, S, hd]).
		// RoPEF32(x, out, batch, heads, seqLen, headDim, base):
		//   ожидает [batch, heads, seqLen, headDim] row-major.
		//   BH×1×S×HD ≡ BH×S×HD.
		if err := gtB.RoPEF32(sc.QPerm, sc.QPerm, BH, 1, S, HD, cfg.Base); err != nil {
			return 0, fmt.Errorf("layer %d RoPE Q: %w", l, err)
		}
		if err := gtB.RoPEF32(sc.KPerm, sc.KPerm, BH, 1, S, HD, cfg.Base); err != nil {
			return 0, fmt.Errorf("layer %d RoPE K: %w", l, err)
		}

		// 6. Quantize F32 -> FP8 (per-tensor amax).
		// Zero amax buffers before Quantize (kernel does atomicMax; stale value from previous
		// call/layer/step could poison scale). Also Sync between quantize calls (async race).
		nElem := BH * S * HD
		zeroF := []byte{0, 0, 0, 0}
		uploadInto(b, sc.AmaxQ, zeroF)
		uploadInto(b, sc.AmaxK, zeroF)
		uploadInto(b, sc.AmaxV, zeroF)
		// Н3-фикс (A-LLM-5): zero-upload амаксов ОБЯЗАН завершиться до quantize —
		// без Sync копия под нагрузкой исполнялась после ядра и затирала amax
		// (наблюдалось: amax=0 -> FP8-нули; A_LLM4 Н3).
		if s, ok := b.(interface{ Sync() error }); ok {
			s.Sync()
		}
		// A-LLM-5 квант-контракт O(1): Unit-вариант (scale = amax, decoded <= 1) —
		// FP16-акки v121r (QK/O MMA f16.e4m3.e4m3.f16) переполняются на decoded
		// +-448 старой конвенции amax/448 (Н4).
		if err := gtB.QuantizeF32ToF8E4M3Unit(sc.QPerm, sc.QFP8, sc.ScaleQ, sc.AmaxQ, nElem); err != nil {
			return 0, fmt.Errorf("layer %d Q quantize: %w", l, err)
		}
		if s, ok := b.(interface{ Sync() error }); ok {
			s.Sync()
		}
		if err := gtB.QuantizeF32ToF8E4M3Unit(sc.KPerm, sc.KFP8, sc.ScaleK, sc.AmaxK, nElem); err != nil {
			return 0, fmt.Errorf("layer %d K quantize: %w", l, err)
		}
		if s, ok := b.(interface{ Sync() error }); ok {
			s.Sync()
		}
		if err := gtB.QuantizeF32ToF8E4M3Unit(sc.VPerm, sc.VFP8, sc.ScaleV, sc.AmaxV, nElem); err != nil {
			return 0, fmt.Errorf("layer %d V quantize: %w", l, err)
		}
		if s, ok := b.(interface{ Sync() error }); ok {
			s.Sync()
		}

		// 7. Read scales (host) to compose FA scale.
		// Новая конвенция (A-LLM-5): scale_X = amax_X (decoded O(1)).
		// Kernel scale param = softmax_scale * scale_Q * scale_K.
		// V-descale absorb: launcher hardcodes v_descale=1.0; O = P @ V где V raw fp8_decode.
		// O_kernel = P @ V_decoded; O_f32 = O_kernel * scale_V post-hoc (Mul scalar).
		amaxHost := gpuToHost(b, sc.AmaxQ, 1)
		amaxK := gpuToHost(b, sc.AmaxK, 1)
		amaxV := gpuToHost(b, sc.AmaxV, 1)
		scaleQ := amaxHost[0]
		scaleK := amaxK[0]
		scaleV := amaxV[0]
		if scaleQ <= 0 {
			scaleQ = 1.0
		}
		if scaleK <= 0 {
			scaleK = 1.0
		}
		if scaleV <= 0 {
			scaleV = 1.0
		}
		faScale := softmaxScale * scaleQ * scaleK

		// Zero OFP16 and LGPU before FA to ensure fresh state per call.
		{
			ofp16Zero := make([]byte, BH*S*HD*2)
			uploadInto(b, sc.OFP16, ofp16Zero)
			lZero := make([]byte, BH*S*4)
			uploadInto(b, sc.LGPU, lZero)
		}
		// 8. FA-fwd-with-L. Output O -> FP16 [BH, S, HD], L -> F32 [BH, S].
		if err := faCtx.ForwardTrain(
			devPtr(sc.QFP8), devPtr(sc.KFP8), devPtr(sc.VFP8),
			devPtr(sc.OFP16), devPtr(sc.LGPU),
			int32(BH), int32(S), int32(HD),
			0, 0, faScale, 0); err != nil {
			return 0, fmt.Errorf("layer %d fa_forward_train: %w", l, err)
		}

		// 9. Cast O FP16 -> F32.
		if err := gtB.CastF16ToF32(sc.OFP16, sc.OF32, BH*S*HD); err != nil {
			return 0, fmt.Errorf("layer %d O F16->F32: %w", l, err)
		}

		// 10. Scale by scale_V (absorb V descale post-hoc).
		if scaleV != 1.0 {
			if err := scaleInPlaceHost(b, sc.OF32, BH*S*HD, scaleV); err != nil {
				return 0, fmt.Errorf("layer %d O scale_V: %w", l, err)
			}
		}

		// 11. Inverse permute OF32 [BH, S, hd] -> [B, S, H, hd] -> [M, D].
		// [H, S, HD] -> [S, H, HD]: transpose с H<->S параметрами.
		oBase := devPtr(sc.OF32)
		qBufBase := devPtr(sc.Q) // Q reused as inverse-permute buf
		for bi := 0; bi < B; bi++ {
			off := uintptr(bi) * batchStride
			if err := launchTransposeSHD_HSDPtr(b, qBufBase+off, oBase+off, H, S, HD); err != nil {
				return 0, fmt.Errorf("layer %d O inv-permute batch %d: %w", l, bi, err)
			}
		}

		// 12. AttnOut = Q_buf @ Wo -> [M, D]
		if err := b.MatMul(sc.AttnOut, sc.Q, lw.Wo,
			core.Shape{M, D}, core.Shape{D, D}, core.Float32); err != nil {
			return 0, fmt.Errorf("layer %d Wo: %w", l, err)
		}

		// 13. Residual: X = X + AttnOut
		if err := b.Add(sc.X, sc.X, sc.AttnOut,
			core.Shape{M, D}, core.Shape{M, D}, core.Shape{M, D}, core.Float32); err != nil {
			return 0, fmt.Errorf("layer %d residual attn: %w", l, err)
		}

		// 14. RMSNorm2(X) -> Normed
		if err := gtB.RMSNormF32(sc.X, lw.Norm2, sc.Normed, M, D, cfg.Eps); err != nil {
			return 0, fmt.Errorf("layer %d RMSNorm2: %w", l, err)
		}

		// 15. FFN: Silu(Normed @ W1) @ W2.
		if err := b.MatMul(sc.FFNHidden, sc.Normed, lw.W1,
			core.Shape{M, D}, core.Shape{D, cfg.FFN}, core.Float32); err != nil {
			return 0, fmt.Errorf("layer %d FFN W1: %w", l, err)
		}
		// Silu = x * sigmoid(x). Compute sigmoid then mul.
		if err := b.Sigmoid(sc.FFNSigmoid, sc.FFNHidden,
			core.Shape{M, cfg.FFN}, core.Float32); err != nil {
			return 0, fmt.Errorf("layer %d Sigmoid: %w", l, err)
		}
		if err := b.Mul(sc.FFNSilu, sc.FFNHidden, sc.FFNSigmoid,
			core.Shape{M, cfg.FFN}, core.Shape{M, cfg.FFN}, core.Shape{M, cfg.FFN}, core.Float32); err != nil {
			return 0, fmt.Errorf("layer %d FFN Silu Mul: %w", l, err)
		}
		if err := b.MatMul(sc.FFNOut, sc.FFNSilu, lw.W2,
			core.Shape{M, cfg.FFN}, core.Shape{cfg.FFN, D}, core.Float32); err != nil {
			return 0, fmt.Errorf("layer %d FFN W2: %w", l, err)
		}

		// 16. Residual: X = X + FFNOut
		if err := b.Add(sc.X, sc.X, sc.FFNOut,
			core.Shape{M, D}, core.Shape{M, D}, core.Shape{M, D}, core.Float32); err != nil {
			return 0, fmt.Errorf("layer %d residual FFN: %w", l, err)
		}
	}

	// -- Final RMSNorm --
	if err := gtB.RMSNormF32(sc.X, st.NormOut, sc.Normed, M, D, cfg.Eps); err != nil {
		return 0, fmt.Errorf("final RMSNorm: %w", err)
	}

	// -- Logits = Normed @ Wout -> [M, V]
	if err := b.MatMul(sc.Logits, sc.Normed, st.Wout,
		core.Shape{M, D}, core.Shape{D, V}, core.Float32); err != nil {
		return 0, fmt.Errorf("Wout: %w", err)
	}

	// -- CE-fused kernel (loss per row + gradL for future bwd).
	// Convert targetTokens int32 -> bytes, upload.
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

	// -- Sum loss on host --
	lossHost := gpuToHost(b, sc.Loss, M)
	var lossSum float64
	for _, v := range lossHost {
		lossSum += float64(v)
	}
	loss = lossSum / float64(M)
	return loss, nil
}

// -----------------------------------------------------------------------------
// Helpers.
// -----------------------------------------------------------------------------

// sliceStorage — returns a Storage that points to a sub-region of parent.
// Uses type assertion to goml.cuda.Storage.SliceOffsetBytes if available.
// Fallback: fake storage-like wrapper via DevicePtr arithmetic.
type sliceStore struct {
	ptr     uintptr
	byteLen int
}

func (s *sliceStore) Device() backend.Device      { return backend.CUDADevice(0) }
func (s *sliceStore) Ptr() unsafe.Pointer         { return unsafe.Pointer(s.ptr) }
func (s *sliceStore) Bytes() []byte               { return nil }
func (s *sliceStore) ByteLen() int                { return s.byteLen }
func (s *sliceStore) Free()                       {}
func (s *sliceStore) DevicePtr() uintptr          { return s.ptr }

// sliceStorage — DEPRECATED (не используется, MatMul adapter не принимает custom Storage).
// Оставлен для будущих ptr-based утилит.
func sliceStorage(b backend.Backend, parent backend.Storage, elemOff, elemCount int) backend.Storage {
	base := devPtr(parent)
	return &sliceStore{ptr: base + uintptr(elemOff*4), byteLen: elemCount * 4}
}

// uploadInto -- in-place upload from bytes into pre-allocated storage.
// Path 1: goml.cuda.Storage -> CopyHtoD.
// Path 2: any -> ToDevice + Copy fallback.
func uploadInto(b backend.Backend, dst backend.Storage, src []byte) (int, error) {
	if cdst, ok := dst.(*gomlcuda.Storage); ok {
		if err := gomlcuda.CopyHtoD(cdst, src); err != nil {
			return 0, err
		}
		return len(src), nil
	}
	tmp, err := b.ToDevice(backend.CUDADevice(0), &cpuStorage{data: src})
	if err != nil {
		return 0, err
	}
	defer b.Free(tmp)
	if err := b.Copy(dst, tmp, len(src)); err != nil {
		return 0, err
	}
	return len(src), nil
}

// scaleInPlaceHost — умножает F32 tensor на scalar. Простой D2H->host mul->H2D
// (для post-hoc scale_V absorb; O size = BH*S*HD*4 = ~1MB на BattleA B=1, тривиально).
func scaleInPlaceHost(b backend.Backend, s backend.Storage, n int, scale float32) error {
	host := gpuToHost(b, s, n)
	for i := range host {
		host[i] *= scale
	}
	// Upload back.
	if _, err := uploadInto(b, s, f32ToBytes(host)); err != nil {
		return err
	}
	return nil
}
