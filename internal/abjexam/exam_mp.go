package abjexam

// B-impl-4: mixed-precision дispatch на matmul-линии Step.
//
// Три precision: F32 (baseline == trainStepGPU), F16 (FP16 IO, F32 out),
// F8E4M3 (FP8 IO, FP16 out → F32).
//
// Только matmul меняется; embedding, layernorm, softmax, CE, backward, adamw --
// как есть (те же F32 paths). Это соответствует ТЗ B-impl-4: matmul-линия
// через новые gotorch пути, остальное как есть.

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/djeday123/goml/backend"
	gotorchAdapter "github.com/djeday123/goml/backend/gotorch"
	"github.com/djeday123/goml/core"
)

// Precision -- параметр Step.
type Precision int

const (
	PrecF32     Precision = 0
	PrecF16     Precision = 1
	PrecF8E4M3  Precision = 2
)

func (p Precision) String() string {
	switch p {
	case PrecF32:
		return "F32"
	case PrecF16:
		return "F16"
	case PrecF8E4M3:
		return "F8E4M3"
	}
	return "?"
}

// trainStepGPUPrecision -- аналог trainStepGPU но с precision-параметром.
// Для F32 -- вызывает b.MatMul напрямую. Для F16/F8 -- через adapter
// extension type-assertion (только gotorch-adapter поддерживает).
func trainStepGPUPrecision(b backend.Backend, st *GPUState, inputTokens, targetTokens []int64, step int, prec Precision) (loss float64, gradOW []float32, err error) {
	cfg := st.Cfg
	inputGPU, err := b.ToDevice(backend.CUDADevice(0), &cpuStorage{data: int64ToBytes(inputTokens)})
	if err != nil {
		return 0, nil, fmt.Errorf("upload tokens: %w", err)
	}
	defer b.Free(inputGPU)

	// ── Forward ─
	embedded, _ := b.Alloc(cfg.Seq * cfg.Embed * 4)
	defer b.Free(embedded)
	if err := b.Embedding(embedded, st.EmbW, inputGPU, cfg.Vocab, cfg.Embed, cfg.Seq, core.Float32); err != nil {
		return 0, nil, fmt.Errorf("embedding: %w", err)
	}
	normed, _ := b.Alloc(cfg.Seq * cfg.Embed * 4)
	defer b.Free(normed)
	if err := b.LayerNorm(normed, embedded, st.LNG, st.LNB,
		core.Shape{cfg.Seq, cfg.Embed}, 1, 1e-5, core.Float32); err != nil {
		return 0, nil, fmt.Errorf("layernorm: %w", err)
	}
	logits, _ := b.Alloc(cfg.Seq * cfg.Vocab * 4)
	defer b.Free(logits)

	// ── Matmul линия -- dispatch по precision ─
	switch prec {
	case PrecF32:
		if err := b.MatMul(logits, normed, st.OutW,
			core.Shape{cfg.Seq, cfg.Embed}, core.Shape{cfg.Embed, cfg.Vocab}, core.Float32); err != nil {
			return 0, nil, fmt.Errorf("matmul f32: %w", err)
		}
	case PrecF16:
		gtB, ok := b.(*gotorchAdapter.Backend)
		if !ok {
			return 0, nil, fmt.Errorf("F16 precision requires gotorch adapter backend, got %T", b)
		}
		nA := cfg.Seq * cfg.Embed
		nB := cfg.Embed * cfg.Vocab
		normedF16, _ := b.Alloc(nA * 2)
		defer b.Free(normedF16)
		outWF16, _ := b.Alloc(nB * 2)
		defer b.Free(outWF16)
		if err := gtB.CastF32ToF16(normed, normedF16, nA); err != nil {
			return 0, nil, fmt.Errorf("cast normed->f16: %w", err)
		}
		if err := gtB.CastF32ToF16(st.OutW, outWF16, nB); err != nil {
			return 0, nil, fmt.Errorf("cast outW->f16: %w", err)
		}
		if err := gtB.MatMulF16(normedF16, outWF16, logits, cfg.Seq, cfg.Vocab, cfg.Embed); err != nil {
			return 0, nil, fmt.Errorf("matmul f16: %w", err)
		}
	case PrecF8E4M3:
		gtB, ok := b.(*gotorchAdapter.Backend)
		if !ok {
			return 0, nil, fmt.Errorf("F8 precision requires gotorch adapter backend, got %T", b)
		}
		nA := cfg.Seq * cfg.Embed
		nB := cfg.Embed * cfg.Vocab
		nC := cfg.Seq * cfg.Vocab
		normedF8, _ := b.Alloc(nA)
		defer b.Free(normedF8)
		outWF8, _ := b.Alloc(nB)
		defer b.Free(outWF8)
		scaleA, _ := b.Alloc(4)
		defer b.Free(scaleA)
		scaleB, _ := b.Alloc(4)
		defer b.Free(scaleB)
		scaleC, _ := b.Alloc(4)
		defer b.Free(scaleC)
		amaxA, _ := b.Alloc(4)
		defer b.Free(amaxA)
		amaxB, _ := b.Alloc(4)
		defer b.Free(amaxB)
		logitsF16, _ := b.Alloc(nC * 2)
		defer b.Free(logitsF16)
		// scaleC = 1.0 (raw output before pixel-level dequantization).
		if err := gtB.QuantizeF32ToF8E4M3(normed, normedF8, scaleA, amaxA, nA); err != nil {
			return 0, nil, fmt.Errorf("quantize normed: %w", err)
		}
		if err := gtB.QuantizeF32ToF8E4M3(st.OutW, outWF8, scaleB, amaxB, nB); err != nil {
			return 0, nil, fmt.Errorf("quantize outW: %w", err)
		}
		// scaleC init to 1.0.
		oneBuf := []byte{0, 0, 0x80, 0x3F}
		scaleCHost, err := b.ToDevice(backend.CUDADevice(0), &cpuStorage{data: oneBuf})
		if err != nil {
			return 0, nil, fmt.Errorf("upload scaleC: %w", err)
		}
		// Copy raw bytes from cpuBridge into pre-allocated scaleC.
		// Simpler: free scaleC alloc, replace pointer.
		b.Free(scaleC)
		scaleC = scaleCHost
		defer b.Free(scaleC)
		if err := gtB.MatMulF8E4M3(normedF8, outWF8, logitsF16, scaleA, scaleB, scaleC, nil, cfg.Seq, cfg.Vocab, cfg.Embed); err != nil {
			return 0, nil, fmt.Errorf("matmul f8: %w", err)
		}
		if err := gtB.CastF16ToF32(logitsF16, logits, nC); err != nil {
			return 0, nil, fmt.Errorf("cast logits: %w", err)
		}
	}

	probs, _ := b.Alloc(cfg.Seq * cfg.Vocab * 4)
	defer b.Free(probs)
	if err := b.Softmax(probs, logits, core.Shape{cfg.Seq, cfg.Vocab}, 1, core.Float32); err != nil {
		return 0, nil, fmt.Errorf("softmax: %w", err)
	}
	if s, ok := b.(interface{ Sync() error }); ok {
		if err := s.Sync(); err != nil {
			return 0, nil, err
		}
	}

	// CE (CPU): read probs, compute loss + gradLogits.
	probsHost := gpuToHost(b, probs, cfg.Seq*cfg.Vocab)
	gradLogits := make([]float32, cfg.Seq*cfg.Vocab)
	copy(gradLogits, probsHost)
	var lossSum float64
	for i := 0; i < cfg.Seq; i++ {
		tgt := int(targetTokens[i])
		p := probsHost[i*cfg.Vocab+tgt]
		if p < 1e-10 {
			p = 1e-10
		}
		lossSum -= math.Log(float64(p))
		gradLogits[i*cfg.Vocab+tgt] -= 1.0
	}
	loss = lossSum / float64(cfg.Seq)
	scale := float32(1.0 / float32(cfg.Seq))
	for i := range gradLogits {
		gradLogits[i] *= scale
	}

	// ── Backward outW (CPU, как в gputrain — оставляем F32 backward) ─
	normedHost := gpuToHost(b, normed, cfg.Seq*cfg.Embed)
	gradOW = make([]float32, cfg.Embed*cfg.Vocab)
	for s := 0; s < cfg.Seq; s++ {
		for e := 0; e < cfg.Embed; e++ {
			nv := normedHost[s*cfg.Embed+e]
			for v := 0; v < cfg.Vocab; v++ {
				gradOW[e*cfg.Vocab+v] += nv * gradLogits[s*cfg.Vocab+v]
			}
		}
	}
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
			return 0, nil, fmt.Errorf("adamw launch: %w", err)
		}
	} else {
		return 0, nil, fmt.Errorf("backend does not support Launch")
	}
	if s, ok := b.(interface{ Sync() error }); ok {
		if err := s.Sync(); err != nil {
			return 0, nil, err
		}
	}
	return loss, gradOW, nil
}
