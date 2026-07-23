package f64ref

// LLMF64 — эталонная LLM (соответствует goml.nn.LLM pre-norm LLaMA style):
//   tokens → Embedding → N × TransformerBlock → LayerNormFinal → Linear(out)
//
// Cross-entropy loss + backward + Adam step — всё F64.

import (
	"fmt"
	"math"
)

type LLMF64 struct {
	Config LLMConfigF64
	Embed  *EmbeddingF64
	Blocks []*TransformerBlockF64
	Norm   *LayerNormF64
	Output *LinearF64
}

type LLMConfigF64 struct {
	VocabSize    int
	Dim          int
	NumLayers    int
	NumHeads     int
	FFNHiddenDim int
	MaxSeqLen    int
	Eps          float64
}

// AllParams возвращает плоский список параметров (для AdamW).
// Порядок фиксирован: Embed.W, [Block.attnGamma, Block.attnBeta, Wq, Wk, Wv, Wo,
// Block.ffnGamma, Block.ffnBeta, W1, W2, W3] × N, NormGamma, NormBeta, Output.W.
func (m *LLMF64) AllParams() []*F64Tensor {
	params := []*F64Tensor{m.Embed.Weight}
	for _, b := range m.Blocks {
		params = append(params,
			b.AttnNorm.Gamma, b.AttnNorm.Beta,
			b.Attn.Wq.Weight, b.Attn.Wk.Weight, b.Attn.Wv.Weight, b.Attn.Wo.Weight,
			b.FFNNorm.Gamma, b.FFNNorm.Beta,
			b.FFN.W1, b.FFN.W2, b.FFN.W3,
		)
	}
	params = append(params, m.Norm.Gamma, m.Norm.Beta, m.Output.Weight)
	return params
}

type LLMForwardCache struct {
	tokens     []int64
	batch, seq int
	embedOut   *F64Tensor
	blockOuts  []*F64Tensor
	blockCaches []*BlockCache
	normOut    *F64Tensor
	normCache  *LayerNormCache
	logits     *F64Tensor
}

func (m *LLMF64) Forward(tokens []int64, batch, seqLen int) (*F64Tensor, *LLMForwardCache) {
	embOut := m.Embed.Forward(tokens, batch, seqLen)
	blockOuts := make([]*F64Tensor, len(m.Blocks))
	blockCaches := make([]*BlockCache, len(m.Blocks))
	prev := embOut
	for i, b := range m.Blocks {
		out, cache := b.Forward(prev)
		blockOuts[i] = out
		blockCaches[i] = cache
		prev = out
	}
	normOut, normCache := m.Norm.Forward(prev)
	logits := m.Output.Forward(normOut) // [B, S, VocabSize]
	return logits, &LLMForwardCache{
		tokens: tokens, batch: batch, seq: seqLen,
		embedOut: embOut, blockOuts: blockOuts, blockCaches: blockCaches,
		normOut: normOut, normCache: normCache, logits: logits,
	}
}

// CrossEntropyLossF64: mean over batch*seq of -log(softmax(logits)[target]).
// logits: [B, S, V]; targets: []int64 len B*S.
// Возвращает scalar loss + dLogits для backward (уже в форме [B, S, V]).
func CrossEntropyLossF64(logits *F64Tensor, targets []int64) (float64, *F64Tensor) {
	B, S, V := logits.Shape[0], logits.Shape[1], logits.Shape[2]
	if len(targets) != B*S {
		panic("CE: targets len mismatch")
	}
	dLogits := Zeros(B, S, V)
	var totalLoss float64
	for b := 0; b < B; b++ {
		for s := 0; s < S; s++ {
			off := (b*S + s) * V
			// softmax numerically stable.
			mx := math.Inf(-1)
			for i := 0; i < V; i++ {
				if logits.Data[off+i] > mx {
					mx = logits.Data[off+i]
				}
			}
			var sum float64
			exps := make([]float64, V)
			for i := 0; i < V; i++ {
				exps[i] = math.Exp(logits.Data[off+i] - mx)
				sum += exps[i]
			}
			inv := 1.0 / sum
			tgt := int(targets[b*S+s])
			if tgt < 0 || tgt >= V {
				panic(fmt.Sprintf("CE: target %d out of vocab %d", tgt, V))
			}
			pTgt := exps[tgt] * inv
			if pTgt < 1e-30 {
				pTgt = 1e-30
			}
			totalLoss += -math.Log(pTgt)
			// dL/dlogits = softmax - onehot(target), scaled by 1/(B*S) для mean.
			for i := 0; i < V; i++ {
				p := exps[i] * inv
				g := p
				if i == tgt {
					g -= 1.0
				}
				dLogits.Data[off+i] = g / float64(B*S)
			}
		}
	}
	return totalLoss / float64(B*S), dLogits
}

// Backward всей модели: возвращает грады параметров в том же порядке что AllParams.
func (m *LLMF64) Backward(cache *LLMForwardCache, dLogits *F64Tensor) []*F64Tensor {
	// output Linear backward.
	dNormOut, dW_out, _ := m.Output.Backward(cache.normOut, dLogits)
	// LayerNorm final backward.
	dPrev, dNormGamma, dNormBeta := m.Norm.Backward(dNormOut, cache.normCache)
	// blocks backward reverse.
	dxCur := dPrev
	blockDNormGamma := make([]*F64Tensor, len(m.Blocks))
	blockDNormBeta := make([]*F64Tensor, len(m.Blocks))
	blockDWq := make([]*F64Tensor, len(m.Blocks))
	blockDWk := make([]*F64Tensor, len(m.Blocks))
	blockDWv := make([]*F64Tensor, len(m.Blocks))
	blockDWo := make([]*F64Tensor, len(m.Blocks))
	blockDAttnGamma := make([]*F64Tensor, len(m.Blocks))
	blockDAttnBeta := make([]*F64Tensor, len(m.Blocks))
	blockDW1 := make([]*F64Tensor, len(m.Blocks))
	blockDW2 := make([]*F64Tensor, len(m.Blocks))
	blockDW3 := make([]*F64Tensor, len(m.Blocks))
	for i := len(m.Blocks) - 1; i >= 0; i-- {
		bCache := cache.blockCaches[i]
		blk := m.Blocks[i]
		// TransformerBlockF64.Backward return только dX — нужно раскрыть чтобы
		// собрать все веса. Вручную здесь:
		// Реплицируем логику из TransformerBlockF64.Backward.
		dOut := dxCur
		// residual 2: dxAfter1_from_res = dOut; dffnOut = dOut
		dffnOut := dOut
		dNormed2, dW1, dW2, dW3 := blk.FFN.Backward(dffnOut, bCache.ffnCache)
		blockDW1[i] = dW1
		blockDW2[i] = dW2
		blockDW3[i] = dW3
		dxAfter1FromNorm, dGamma2, dBeta2 := blk.FFNNorm.Backward(dNormed2, bCache.lnCache2)
		blockDNormGamma[i] = dGamma2
		blockDNormBeta[i] = dBeta2
		dxAfter1 := Zeros(bCache.x.Shape...)
		for j := range dxAfter1.Data {
			dxAfter1.Data[j] = dOut.Data[j] + dxAfter1FromNorm.Data[j]
		}
		// residual 1.
		dxFromRes := dxAfter1
		dattnOut := dxAfter1
		dNormed1, dWq, dWk, dWv, dWo := blk.Attn.Backward(dattnOut, bCache.mhaCache)
		blockDWq[i] = dWq
		blockDWk[i] = dWk
		blockDWv[i] = dWv
		blockDWo[i] = dWo
		dxFromNorm, dGamma1, dBeta1 := blk.AttnNorm.Backward(dNormed1, bCache.lnCache1)
		blockDAttnGamma[i] = dGamma1
		blockDAttnBeta[i] = dBeta1
		dxNew := Zeros(bCache.x.Shape...)
		for j := range dxNew.Data {
			dxNew.Data[j] = dxFromRes.Data[j] + dxFromNorm.Data[j]
		}
		dxCur = dxNew
	}
	// dxCur — это grad относительно embedOut (входа в первый блок).
	dEmbedW := m.Embed.Backward(cache.tokens, dxCur, cache.batch, cache.seq)

	// Собираем в том же порядке что AllParams.
	grads := []*F64Tensor{dEmbedW}
	for i := 0; i < len(m.Blocks); i++ {
		grads = append(grads,
			blockDAttnGamma[i], blockDAttnBeta[i],
			blockDWq[i], blockDWk[i], blockDWv[i], blockDWo[i],
			blockDNormGamma[i], blockDNormBeta[i],
			blockDW1[i], blockDW2[i], blockDW3[i],
		)
	}
	grads = append(grads, dNormGamma, dNormBeta, dW_out)
	return grads
}
