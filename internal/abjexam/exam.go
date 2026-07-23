// Package abjexam — P1-ABJ экзамен: три параллельные траектории тренировки через
// gputrain-подобную модель (Embed + LN + Linear + Softmax + CE + AdamW-outWeight).
//
// Роль: закрытие долга impl-5c через боевой GPU-путь gputrain (обходит nn.*).
// НЕ trainer, НЕ библиотека — исполняемый экзамен. Не для production import.
//
// cmd/gputrain/main.go остаётся read-only (правило ARCHITECTURE.md переходного
// периода). Логика этого файла — КОПИЯ gputrain trainStep, сжатая в
// вызываемую функцию с параметром Path.
package abjexam

import (
	"fmt"
	"math"
	"math/rand"
	"unsafe"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
	"github.com/djeday123/goml/internal/f64ref"
)

// ── Model config ─ (та же что gputrain/main.go:45-49) ─
type ModelCfg struct {
	Vocab int
	Embed int
	Seq   int
}

var DefaultCfg = ModelCfg{Vocab: 256, Embed: 64, Seq: 16}

// ── ГИПЕР-параметры AdamW из gputrain ─
const (
	LR    float32 = 3e-3
	Beta1 float32 = 0.9
	Beta2 float32 = 0.999
	EPS   float32 = 1e-8
	WD    float32 = 0.01
)

// GPUState — веса на GPU + AdamW state. Аналог `params` из gputrain.
type GPUState struct {
	Cfg          ModelCfg
	EmbW, LNG, LNB, OutW, OutB backend.Storage
	// AdamW state — только outWeight обновляется в gputrain (см. main.go:181-202)
	OutMomM, OutMomV backend.Storage
}

// JState — F64 судья через f64ref (host-side).
type JState struct {
	Cfg   ModelCfg
	EmbW  *f64ref.F64Tensor // [V, E]
	LNG   *f64ref.F64Tensor // [E]
	LNB   *f64ref.F64Tensor // [E]
	OutW  *f64ref.F64Tensor // [E, V]
	// AdamW state:
	Opt *f64ref.AdamWF64
}

// initWeightsCPU — детерминистические веса как в gputrain (rand seed извне),
// scale=0.02 для Embed и OutW, gamma=1, beta=0. Возвращает CPU float32 набор.
func initWeightsCPU(cfg ModelCfg, r *rand.Rand) (embW, lng, lnb, outW, outB []float32) {
	embW = make([]float32, cfg.Vocab*cfg.Embed)
	for i := range embW {
		embW[i] = float32(r.NormFloat64() * 0.02)
	}
	lng = make([]float32, cfg.Embed)
	for i := range lng {
		lng[i] = 1.0
	}
	lnb = make([]float32, cfg.Embed) // zeros
	outW = make([]float32, cfg.Embed*cfg.Vocab)
	for i := range outW {
		outW[i] = float32(r.NormFloat64() * 0.02)
	}
	outB = make([]float32, cfg.Vocab) // zeros
	return
}

// NewGPUState создаёт GPUState с identical numerical weights (тот же rand seed
// снаружи → те же значения). Загружает на GPU через backend.
func NewGPUState(cfg ModelCfg, r *rand.Rand, b backend.Backend) (*GPUState, error) {
	embW, lng, lnb, outW, outB := initWeightsCPU(cfg, r)
	zeros := func(n int) (backend.Storage, error) {
		s, err := b.Alloc(n * 4)
		if err != nil {
			return nil, err
		}
		if err := b.Fill(s, core.Shape{n}, 0, core.Float32); err != nil {
			return nil, err
		}
		return s, nil
	}
	// AdamW state:
	outMomM, err := zeros(cfg.Embed * cfg.Vocab)
	if err != nil {
		return nil, err
	}
	outMomV, err := zeros(cfg.Embed * cfg.Vocab)
	if err != nil {
		return nil, err
	}
	// Weight uploads (Copy с CPU storage backend не всегда работает
	// напрямую — используем ToDevice pattern из gputrain).
	embWS, err := b.ToDevice(backend.CUDADevice(0), &cpuStorage{data: f32ToBytes(embW)})
	if err != nil {
		return nil, err
	}
	lngS, err := b.ToDevice(backend.CUDADevice(0), &cpuStorage{data: f32ToBytes(lng)})
	if err != nil {
		return nil, err
	}
	lnbS, err := b.ToDevice(backend.CUDADevice(0), &cpuStorage{data: f32ToBytes(lnb)})
	if err != nil {
		return nil, err
	}
	outWS, err := b.ToDevice(backend.CUDADevice(0), &cpuStorage{data: f32ToBytes(outW)})
	if err != nil {
		return nil, err
	}
	outBS, err := b.ToDevice(backend.CUDADevice(0), &cpuStorage{data: f32ToBytes(outB)})
	if err != nil {
		return nil, err
	}
	return &GPUState{
		Cfg: cfg, EmbW: embWS, LNG: lngS, LNB: lnbS, OutW: outWS, OutB: outBS,
		OutMomM: outMomM, OutMomV: outMomV,
	}, nil
}

// NewJState создаёт F64-судью с identical initial weights (по тому же массиву
// float32 что был сгенерирован rand'ом).
func NewJState(cfg ModelCfg, r *rand.Rand) *JState {
	embW, lng, lnb, outW, _ := initWeightsCPU(cfg, r)
	embW64 := f32To64Slice(embW)
	lng64 := f32To64Slice(lng)
	lnb64 := f32To64Slice(lnb)
	outW64 := f32To64Slice(outW)
	// Layout: EmbW [V,E], OutW gputrain хранит [E,V] (line 60 randGPU shape).
	emb := f64ref.NewF64Tensor(embW64, []int{cfg.Vocab, cfg.Embed})
	lngT := f64ref.NewF64Tensor(lng64, []int{cfg.Embed})
	lnbT := f64ref.NewF64Tensor(lnb64, []int{cfg.Embed})
	// outW в gputrain: shape [embedDim, vocabSize] = [E, V] (line 60,60,74).
	outWT := f64ref.NewF64Tensor(outW64, []int{cfg.Embed, cfg.Vocab})
	// Только outW обновляется (как в gputrain). Judge Opt только на нём:
	opt := f64ref.NewAdamWF64([]*f64ref.F64Tensor{outWT}, float64(LR))
	opt.Beta1 = float64(Beta1)
	opt.Beta2 = float64(Beta2)
	opt.Eps = float64(EPS)
	opt.WeightDecay = float64(WD)
	return &JState{
		Cfg: cfg, EmbW: emb, LNG: lngT, LNB: lnbT, OutW: outWT, Opt: opt,
	}
}

// ── trainStepGPU — реплика gputrain forward+backward+AdamW в вызываемой форме.
// Возвращает loss + gradOW для аудита (г).
// Backend передаётся снаружи: A использует goml.cuda (backend.Get(CUDA) до
// adapter.Enable), B использует adapter (backend.Get(CUDA) после adapter.Enable).
func trainStepGPU(b backend.Backend, st *GPUState, inputTokens, targetTokens []int64, step int) (loss float64, gradOW []float32, err error) {
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
	if err := b.MatMul(logits, normed, st.OutW,
		core.Shape{cfg.Seq, cfg.Embed}, core.Shape{cfg.Embed, cfg.Vocab}, core.Float32); err != nil {
		return 0, nil, fmt.Errorf("matmul: %w", err)
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

	// CE (CPU): читаем probs, считаем loss + gradLogits.
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

	// ── Backward outW (CPU, как в gputrain) ─
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
	// ── AdamW update outW на GPU через adamw_f32 PTX ─
	gradOWGPU, err := b.ToDevice(backend.CUDADevice(0), &cpuStorage{data: f32ToBytes(gradOW)})
	if err != nil {
		return 0, nil, err
	}
	defer b.Free(gradOWGPU)

	b1corr := float32(1.0 - math.Pow(float64(Beta1), float64(step)))
	b2corr := float32(1.0 - math.Pow(float64(Beta2), float64(step)))
	nOW := uint32(cfg.Embed * cfg.Vocab)

	// Local vars нужны т.к. unsafe.Pointer требует address'able values.
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

// trainStepJudge — F64 forward+CE+backward+AdamW через f64ref.
func trainStepJudge(j *JState, inputTokens, targetTokens []int64) (loss float64, gradOW *f64ref.F64Tensor) {
	cfg := j.Cfg
	// Embedding F64.
	emb := f64ref.NewEmbeddingF64(cfg.Vocab, cfg.Embed)
	// Копируем текущие веса judge'а в этот temporary embedding.
	copy(emb.Weight.Data, j.EmbW.Data)
	embedded := emb.Forward(inputTokens, 1, cfg.Seq) // [1, seq, E]
	// reshape к [seq, E] для LayerNorm (LayerNorm нормализует последнюю ось).
	embedFlat := &f64ref.F64Tensor{Data: embedded.Data, Shape: []int{cfg.Seq, cfg.Embed}}

	ln := &f64ref.LayerNormF64{Gamma: j.LNG, Beta: j.LNB, Eps: 1e-5}
	normed, _ := ln.Forward(embedFlat)

	// MatMul F64: normed [seq, E] @ outW [E, V] → logits [seq, V].
	logits := f64ref.MatMulF64GPU(normed, j.OutW, cfg.Seq, cfg.Embed, cfg.Vocab)

	// CE: превращаем logits в [1, seq, V] для CrossEntropyLossF64.
	logits3D := &f64ref.F64Tensor{Data: logits.Data, Shape: []int{1, cfg.Seq, cfg.Vocab}}
	loss, dLogits := f64ref.CrossEntropyLossF64(logits3D, targetTokens)
	// dLogits [1, seq, V] — reshape в [seq, V] для backward.
	dLogits2D := &f64ref.F64Tensor{Data: dLogits.Data, Shape: []int{cfg.Seq, cfg.Vocab}}

	// gradOW = normed^T @ dLogits: [E, seq] × [seq, V] = [E, V].
	normedT := transposeF64(normed, cfg.Seq, cfg.Embed)
	gradOW = f64ref.MatMulF64GPU(normedT, dLogits2D, cfg.Embed, cfg.Seq, cfg.Vocab)

	// AdamW step: только для OutW.
	for i := range j.Opt.Grads[0].Data {
		j.Opt.Grads[0].Data[i] = gradOW.Data[i]
	}
	j.Opt.Step()

	return loss, gradOW
}

// ── Утилиты ─

func transposeF64(t *f64ref.F64Tensor, rows, cols int) *f64ref.F64Tensor {
	out := f64ref.Zeros(cols, rows)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out.Data[c*rows+r] = t.Data[r*cols+c]
		}
	}
	return out
}

func f32To64Slice(a []float32) []float64 {
	out := make([]float64, len(a))
	for i, v := range a {
		out[i] = float64(v)
	}
	return out
}

func gpuToHost(b backend.Backend, s backend.Storage, n int) []float32 {
	cpuS, err := b.ToDevice(backend.CPU0, s)
	if err != nil {
		panic(err)
	}
	buf := cpuS.Bytes()[:n*4]
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := uint32(buf[i*4]) | uint32(buf[i*4+1])<<8 | uint32(buf[i*4+2])<<16 | uint32(buf[i*4+3])<<24
		out[i] = math.Float32frombits(bits)
	}
	return out
}

func devPtr(s backend.Storage) uintptr {
	type devPtrer interface{ DevicePtr() uintptr }
	if dp, ok := s.(devPtrer); ok {
		return dp.DevicePtr()
	}
	return uintptr(s.Ptr())
}

func gridSize(n, blockSize int) uint32 {
	return uint32((n + blockSize - 1) / blockSize)
}

func f32ToBytes(data []float32) []byte {
	b := make([]byte, len(data)*4)
	for i, v := range data {
		bits := math.Float32bits(v)
		b[i*4] = byte(bits)
		b[i*4+1] = byte(bits >> 8)
		b[i*4+2] = byte(bits >> 16)
		b[i*4+3] = byte(bits >> 24)
	}
	return b
}

func int64ToBytes(data []int64) []byte {
	b := make([]byte, len(data)*8)
	for i, v := range data {
		u := uint64(v)
		for k := 0; k < 8; k++ {
			b[i*8+k] = byte(u >> (k * 8))
		}
	}
	return b
}

type cpuStorage struct{ data []byte }

func (s *cpuStorage) Device() backend.Device { return backend.CPU0 }
func (s *cpuStorage) Ptr() unsafe.Pointer {
	if len(s.data) == 0 {
		return nil
	}
	return unsafe.Pointer(&s.data[0])
}
func (s *cpuStorage) Bytes() []byte { return s.data }
func (s *cpuStorage) ByteLen() int  { return len(s.data) }
func (s *cpuStorage) Free()         { s.data = nil }
