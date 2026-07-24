package abjexam

// B2-BATTLE: боевой mixed-precision Step на SmallConfig-shape (без attention/FFN).
//
// Форма:  Vocab=32000, Dim=256, Seq=128, batch=8.
// MatMul: [batch*seq=1024, embed=256] × [embed=256, vocab=32000]
//         = 1024×256×32000 = 8.4 GFMA -- 32000× больше чем Tiny.
//
// Скоуп (по ТЗ B2 Этап 2): attention/FFN НЕ в scope (глава А). Один Linear
// projection Embed→Vocab. Данные -- синтетические random.Intn(Vocab) (цель --
// измерение режимов, не тренировка).

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"unsafe"

	"github.com/djeday123/goml/backend"
	gotorchAdapter "github.com/djeday123/goml/backend/gotorch"
	"github.com/djeday123/goml/core"
)

// BattleCfg -- боевая конфигурация.
type BattleCfg struct {
	Vocab int
	Embed int
	Seq   int
	Batch int
}

func DefaultBattleCfg() BattleCfg {
	return BattleCfg{Vocab: 32000, Embed: 256, Seq: 128, Batch: 8}
}

// BattleState -- weights on GPU + AdamW state (только OutW обновляется).
type BattleState struct {
	Cfg                        BattleCfg
	EmbW, LNG, LNB, OutW       backend.Storage
	OutMomM, OutMomV           backend.Storage
}

// initBattleWeightsCPU -- детерминистические CPU F32 веса (scale 0.02).
func initBattleWeightsCPU(cfg BattleCfg, r *rand.Rand) (embW, lng, lnb, outW []float32) {
	embW = make([]float32, cfg.Vocab*cfg.Embed)
	for i := range embW {
		embW[i] = float32(r.NormFloat64() * 0.02)
	}
	lng = make([]float32, cfg.Embed)
	for i := range lng {
		lng[i] = 1.0
	}
	lnb = make([]float32, cfg.Embed)
	outW = make([]float32, cfg.Embed*cfg.Vocab)
	for i := range outW {
		outW[i] = float32(r.NormFloat64() * 0.02)
	}
	return
}

func NewBattleState(cfg BattleCfg, r *rand.Rand, b backend.Backend) (*BattleState, error) {
	embW, lng, lnb, outW := initBattleWeightsCPU(cfg, r)
	upload := func(data []float32) (backend.Storage, error) {
		return b.ToDevice(backend.CUDADevice(0), &cpuStorage{data: f32ToBytes(data)})
	}
	st := &BattleState{Cfg: cfg}
	var err error
	if st.EmbW, err = upload(embW); err != nil {
		return nil, fmt.Errorf("upload EmbW: %w", err)
	}
	if st.LNG, err = upload(lng); err != nil {
		return nil, err
	}
	if st.LNB, err = upload(lnb); err != nil {
		return nil, err
	}
	if st.OutW, err = upload(outW); err != nil {
		return nil, err
	}
	// AdamW state (zero-init).
	zeros := make([]float32, cfg.Embed*cfg.Vocab)
	if st.OutMomM, err = upload(zeros); err != nil {
		return nil, err
	}
	zeros2 := make([]float32, cfg.Embed*cfg.Vocab)
	if st.OutMomV, err = upload(zeros2); err != nil {
		return nil, err
	}
	return st, nil
}

// battleBatch -- синтетический батч random.Intn(Vocab).
func battleBatch(r *rand.Rand, cfg BattleCfg) (inp, tgt []int64) {
	n := cfg.Batch * cfg.Seq
	inp = make([]int64, n)
	tgt = make([]int64, n)
	for i := 0; i < n; i++ {
		inp[i] = int64(r.Intn(cfg.Vocab))
		tgt[i] = (inp[i] + 1) % int64(cfg.Vocab)
	}
	return
}

// trainStepBattle -- боевой Step с precision-dispatch на matmul-линии.
// Возвращает loss + gradOW (F32).
//
// Note: LockOSThread обязателен на боевой форме. goml.cuda primary-ctx
// привязывается к OS-thread через SetCurrent в ensureInit; между
// операциями (Sync/D2H) Go scheduler может мигрировать goroutine
// на другой thread без ctx → INVALID_CONTEXT (R03a Test 5 диагноз).
// На tiny-форме окно миграции слишком узкое, на боевой (softmax 32000×
// вокабуляр ~ desятки мс) миграция стабильно ловится.
func trainStepBattle(b backend.Backend, st *BattleState, inputTokens, targetTokens []int64, step int, prec Precision) (loss float64, gradOW []float32, err error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	cfg := st.Cfg
	m := cfg.Batch * cfg.Seq // rows in matmul input
	inputGPU, err := b.ToDevice(backend.CUDADevice(0), &cpuStorage{data: int64ToBytes(inputTokens)})
	if err != nil {
		return 0, nil, fmt.Errorf("upload tokens: %w", err)
	}
	defer b.Free(inputGPU)

	embedded, _ := b.Alloc(m * cfg.Embed * 4)
	defer b.Free(embedded)
	if err := b.Embedding(embedded, st.EmbW, inputGPU, cfg.Vocab, cfg.Embed, m, core.Float32); err != nil {
		return 0, nil, fmt.Errorf("embedding: %w", err)
	}
	normed, _ := b.Alloc(m * cfg.Embed * 4)
	defer b.Free(normed)
	if err := b.LayerNorm(normed, embedded, st.LNG, st.LNB,
		core.Shape{m, cfg.Embed}, 1, 1e-5, core.Float32); err != nil {
		return 0, nil, fmt.Errorf("layernorm: %w", err)
	}
	logits, _ := b.Alloc(m * cfg.Vocab * 4)
	defer b.Free(logits)

	// ── MatMul линия -- dispatch по precision ─
	switch prec {
	case PrecF32:
		if err := b.MatMul(logits, normed, st.OutW,
			core.Shape{m, cfg.Embed}, core.Shape{cfg.Embed, cfg.Vocab}, core.Float32); err != nil {
			return 0, nil, fmt.Errorf("matmul f32: %w", err)
		}
	case PrecF16:
		gtB, ok := b.(*gotorchAdapter.Backend)
		if !ok {
			return 0, nil, fmt.Errorf("F16 requires gotorch adapter, got %T", b)
		}
		nA := m * cfg.Embed
		nB := cfg.Embed * cfg.Vocab
		normedF16, _ := b.Alloc(nA * 2)
		defer b.Free(normedF16)
		outWF16, _ := b.Alloc(nB * 2)
		defer b.Free(outWF16)
		if err := gtB.CastF32ToF16(normed, normedF16, nA); err != nil {
			return 0, nil, err
		}
		if err := gtB.CastF32ToF16(st.OutW, outWF16, nB); err != nil {
			return 0, nil, err
		}
		if err := gtB.MatMulF16(normedF16, outWF16, logits, m, cfg.Vocab, cfg.Embed); err != nil {
			return 0, nil, fmt.Errorf("matmul f16: %w", err)
		}
	case PrecF8E4M3:
		gtB, ok := b.(*gotorchAdapter.Backend)
		if !ok {
			return 0, nil, fmt.Errorf("F8 requires gotorch adapter, got %T", b)
		}
		nA := m * cfg.Embed
		nB := cfg.Embed * cfg.Vocab
		nC := m * cfg.Vocab
		normedF8, _ := b.Alloc(nA)
		defer b.Free(normedF8)
		outWF8, _ := b.Alloc(nB)
		defer b.Free(outWF8)
		scaleA, _ := b.Alloc(4)
		defer b.Free(scaleA)
		scaleB, _ := b.Alloc(4)
		defer b.Free(scaleB)
		amaxA, _ := b.Alloc(4)
		defer b.Free(amaxA)
		amaxB, _ := b.Alloc(4)
		defer b.Free(amaxB)
		logitsF16, _ := b.Alloc(nC * 2)
		defer b.Free(logitsF16)
		if err := gtB.QuantizeF32ToF8E4M3(normed, normedF8, scaleA, amaxA, nA); err != nil {
			return 0, nil, fmt.Errorf("quant A: %w", err)
		}
		if err := gtB.QuantizeF32ToF8E4M3(st.OutW, outWF8, scaleB, amaxB, nB); err != nil {
			return 0, nil, fmt.Errorf("quant B: %w", err)
		}
		oneBuf := []byte{0, 0, 0x80, 0x3F}
		scaleC, err := b.ToDevice(backend.CUDADevice(0), &cpuStorage{data: oneBuf})
		if err != nil {
			return 0, nil, err
		}
		defer b.Free(scaleC)
		if err := gtB.MatMulF8E4M3(normedF8, outWF8, logitsF16, scaleA, scaleB, scaleC, nil, m, cfg.Vocab, cfg.Embed); err != nil {
			return 0, nil, fmt.Errorf("matmul f8: %w", err)
		}
		if err := gtB.CastF16ToF32(logitsF16, logits, nC); err != nil {
			return 0, nil, err
		}
	}

	probs, _ := b.Alloc(m * cfg.Vocab * 4)
	defer b.Free(probs)
	if err := b.Softmax(probs, logits, core.Shape{m, cfg.Vocab}, 1, core.Float32); err != nil {
		return 0, nil, fmt.Errorf("softmax: %w", err)
	}
	if s, ok := b.(interface{ Sync() error }); ok {
		s.Sync()
	}

	// CE + gradLogits (host).
	probsHost := gpuToHost(b, probs, m*cfg.Vocab)
	gradLogits := make([]float32, m*cfg.Vocab)
	copy(gradLogits, probsHost)
	var lossSum float64
	for i := 0; i < m; i++ {
		tgt := int(targetTokens[i])
		p := probsHost[i*cfg.Vocab+tgt]
		if p < 1e-10 {
			p = 1e-10
		}
		lossSum -= math.Log(float64(p))
		gradLogits[i*cfg.Vocab+tgt] -= 1.0
	}
	loss = lossSum / float64(m)
	invM := float32(1.0 / float32(m))
	for i := range gradLogits {
		gradLogits[i] *= invM
	}

	// Backward outW = normed^T @ gradLogits (CPU F32 -- как gputrain).
	normedHost := gpuToHost(b, normed, m*cfg.Embed)
	gradOW = make([]float32, cfg.Embed*cfg.Vocab)
	for i := 0; i < m; i++ {
		for e := 0; e < cfg.Embed; e++ {
			nv := normedHost[i*cfg.Embed+e]
			for v := 0; v < cfg.Vocab; v++ {
				gradOW[e*cfg.Vocab+v] += nv * gradLogits[i*cfg.Vocab+v]
			}
		}
	}

	// AdamW (GPU через adamw_f32 PTX).
	gradOWGPU, err := b.ToDevice(backend.CUDADevice(0), &cpuStorage{data: f32ToBytes(gradOW)})
	if err != nil {
		return 0, nil, err
	}
	defer b.Free(gradOWGPU)

	b1corr := float32(1.0 - math.Pow(float64(Beta1), float64(step)))
	b2corr := float32(1.0 - math.Pow(float64(Beta2), float64(step)))
	nOW := uint32(cfg.Embed * cfg.Vocab)
	owPtr := devPtr(st.OutW)
	gowPtr := devPtr(gradOWGPU)
	momPtr := devPtr(st.OutMomM)
	vomPtr := devPtr(st.OutMomV)
	lrLoc := LR
	b1Loc := Beta1
	b2Loc := Beta2
	epsLoc := EPS
	wdLoc := WD
	adamParams := []unsafe.Pointer{
		unsafe.Pointer(&owPtr),
		unsafe.Pointer(&gowPtr),
		unsafe.Pointer(&momPtr),
		unsafe.Pointer(&vomPtr),
		unsafe.Pointer(&nOW),
		unsafe.Pointer(&lrLoc),
		unsafe.Pointer(&b1Loc),
		unsafe.Pointer(&b2Loc),
		unsafe.Pointer(&epsLoc),
		unsafe.Pointer(&wdLoc),
		unsafe.Pointer(&b1corr),
		unsafe.Pointer(&b2corr),
	}
	if l, ok := b.(interface {
		Launch(name string, gx, gy, gz, bx, by, bz uint32, params []unsafe.Pointer) error
	}); ok {
		if err := l.Launch("adamw_f32",
			gridSize(int(nOW), 256), 1, 1, 256, 1, 1, adamParams); err != nil {
			return 0, nil, err
		}
	}
	if s, ok := b.(interface{ Sync() error }); ok {
		s.Sync()
	}
	return loss, gradOW, nil
}
