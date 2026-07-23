package f64ref

// Ворота 5c — LLMF64 sanity + numerical grad на sub-модели (по правилу серии,
// каждый компонент отдельно; полная LLM sanity через loss убывает).
//
// TinyConfig-подобный тест: 2 Blocks, Dim=8, Heads=2, Hid=12, Vocab=16.

import (
	"math"
	"math/rand"
	"testing"
)

func buildTinyLLMF64(r *rand.Rand, cfg LLMConfigF64) *LLMF64 {
	emb := NewEmbeddingF64(cfg.VocabSize, cfg.Dim)
	for i := range emb.Weight.Data {
		emb.Weight.Data[i] = r.NormFloat64() * 0.02
	}
	blocks := make([]*TransformerBlockF64, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		attnGamma := Zeros(cfg.Dim)
		for j := range attnGamma.Data {
			attnGamma.Data[j] = 1.0
		}
		attnBeta := Zeros(cfg.Dim)
		ffnGamma := Zeros(cfg.Dim)
		for j := range ffnGamma.Data {
			ffnGamma.Data[j] = 1.0
		}
		ffnBeta := Zeros(cfg.Dim)
		Wq := randTensor(r, cfg.Dim, cfg.Dim)
		Wk := randTensor(r, cfg.Dim, cfg.Dim)
		Wv := randTensor(r, cfg.Dim, cfg.Dim)
		Wo := randTensor(r, cfg.Dim, cfg.Dim)
		W1 := randTensor(r, cfg.FFNHiddenDim, cfg.Dim)
		W2 := randTensor(r, cfg.Dim, cfg.FFNHiddenDim)
		W3 := randTensor(r, cfg.FFNHiddenDim, cfg.Dim)
		blocks[i] = NewTransformerBlockF64(cfg.Dim, cfg.NumHeads, cfg.FFNHiddenDim,
			attnGamma, attnBeta, Wq, Wk, Wv, Wo, ffnGamma, ffnBeta, W1, W2, W3, true)
	}
	normGamma := Zeros(cfg.Dim)
	for j := range normGamma.Data {
		normGamma.Data[j] = 1.0
	}
	normBeta := Zeros(cfg.Dim)
	norm := &LayerNormF64{Gamma: normGamma, Beta: normBeta, Eps: cfg.Eps}
	output := NewLinearF64(randTensor(r, cfg.VocabSize, cfg.Dim), nil)
	return &LLMF64{
		Config: cfg,
		Embed:  emb,
		Blocks: blocks,
		Norm:   norm,
		Output: output,
	}
}

func TestLLMF64_SanityLossDecreasing(t *testing.T) {
	r := rand.New(rand.NewSource(30))
	cfg := LLMConfigF64{
		VocabSize:    16,
		Dim:          8,
		NumLayers:    2,
		NumHeads:     2,
		FFNHiddenDim: 12,
		MaxSeqLen:    16,
		Eps:          1e-5,
	}
	m := buildTinyLLMF64(r, cfg)

	const batch, seq = 2, 4
	tokens := make([]int64, batch*seq)
	targets := make([]int64, batch*seq)
	for i := range tokens {
		tokens[i] = int64(r.Intn(cfg.VocabSize))
		targets[i] = int64(r.Intn(cfg.VocabSize))
	}
	opt := NewAdamWF64(m.AllParams(), 5e-3)

	losses := make([]float64, 10)
	for step := 0; step < 10; step++ {
		logits, cache := m.Forward(tokens, batch, seq)
		loss, dLogits := CrossEntropyLossF64(logits, targets)
		losses[step] = loss
		grads := m.Backward(cache, dLogits)
		if len(grads) != len(opt.Grads) {
			t.Fatalf("step %d: grads len %d != params len %d", step, len(grads), len(opt.Grads))
		}
		for k, g := range grads {
			// Copy grads into opt.Grads slots.
			for i := range g.Data {
				opt.Grads[k].Data[i] = g.Data[i]
			}
		}
		opt.Step()
	}
	t.Logf("LLMF64 10-step sanity losses:")
	for i, l := range losses {
		t.Logf("  step %2d: loss=%.6f", i+1, l)
	}
	if losses[9] >= losses[0] {
		t.Errorf("Loss did NOT decrease over 10 steps: loss[0]=%.4f loss[9]=%.4f", losses[0], losses[9])
	}
	// Sanity: loss не NaN, не Inf.
	for i, l := range losses {
		if math.IsNaN(l) || math.IsInf(l, 0) {
			t.Fatalf("step %d: loss is NaN/Inf: %v", i, l)
		}
	}
}
