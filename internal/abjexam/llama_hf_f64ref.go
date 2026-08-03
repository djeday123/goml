// llama_hf_f64ref.go — CPU-F64 forward Llama-класса на ВНЕШНИХ весах
// (HF safetensors), карточка HF-INFERENCE, Фаза I, пункт I.5.
//
// Расширение fwdBattleAF64-пути (battle_a_llm_f64ref.go): переиспользует его
// примитивы matF64_ABT / rmsNormF64 / ropeF64 / permuteSHD_HSD как есть.
// Отличия от BattleA, потребовавшие отдельного forward:
//   - GQA: H_kv < H, kv-голова = h / (H/H_kv) — индексация вместо repeat-kv
//     (память не дублируется; на CPU это бесплатно);
//   - SwiGLU: gate/up/down вместо одноматричного SiLU-MLP BattleA;
//   - causal-маска: скоры считаются только для j<=i, softmax по префиксу
//     (ядра НЕ трогаются — это чисто CPU-линейка);
//   - веса в HF-layout [out,in] — matF64_ABT кушает их без транспонирования;
//   - RoPE: ropeF64 half-split == HF rotate_half бит-в-бит по конвенции
//     (пары (i, i+HD/2), freq = theta^(-2i/HD), см. P4-канон).
// Forward-only: bwd для инференса не нужен.
package abjexam

import (
	"fmt"
	"math"

	"github.com/djeday123/goml/hf"
)

// llamaHFF64Weights — веса модели в F64, HF-layout [out,in] построчно.
type llamaHFF64LayerW struct {
	Norm1, Norm2           []float64 // [D]
	Wq, Wo                 []float64 // [D,D]
	Wk, Wv                 []float64 // [KVD,D]
	Gate, Up               []float64 // [F,D]
	Down                   []float64 // [D,F]
}

type llamaHFF64Weights struct {
	Cfg     *hf.LlamaConfig
	Embed   []float64 // [V,D]
	Layers  []llamaHFF64LayerW
	NormOut []float64 // [D]
	LMHead  []float64 // [V,D]
}

// loadLlamaHFF64 грузит config.json + model.safetensors каталога dir.
// BF16 -> F32 -> F64: оба расширения точны, веса бит-эквивалентны
// тому, что видит HF при torch_dtype=float32.
func loadLlamaHFF64(dir string) (*llamaHFF64Weights, error) {
	cfg, err := hf.LoadLlamaConfig(dir + "/config.json")
	if err != nil {
		return nil, err
	}
	if cfg.TieWordEmbeddings {
		return nil, fmt.Errorf("tie_word_embeddings=true не поддержан в I.5")
	}
	st, err := hf.OpenSafeTensors(dir + "/model.safetensors")
	if err != nil {
		return nil, err
	}
	defer st.Close()

	w := &llamaHFF64Weights{Cfg: cfg, Layers: make([]llamaHFF64LayerW, cfg.NumHiddenLayers)}
	read := func(name string) []float64 {
		if err != nil {
			return nil
		}
		var v []float64
		v, err = st.ReadF64(name)
		return v
	}
	w.Embed = read("model.embed_tokens.weight")
	w.NormOut = read("model.norm.weight")
	w.LMHead = read("lm_head.weight")
	for l := range w.Layers {
		p := fmt.Sprintf("model.layers.%d.", l)
		w.Layers[l] = llamaHFF64LayerW{
			Norm1: read(p + "input_layernorm.weight"),
			Norm2: read(p + "post_attention_layernorm.weight"),
			Wq:    read(p + "self_attn.q_proj.weight"),
			Wk:    read(p + "self_attn.k_proj.weight"),
			Wv:    read(p + "self_attn.v_proj.weight"),
			Wo:    read(p + "self_attn.o_proj.weight"),
			Gate:  read(p + "mlp.gate_proj.weight"),
			Up:    read(p + "mlp.up_proj.weight"),
			Down:  read(p + "mlp.down_proj.weight"),
		}
	}
	if err != nil {
		return nil, err
	}
	return w, nil
}

// fwdLlamaHFF64 — forward промпта toks, возвращает логиты ПОСЛЕДНЕЙ позиции [V].
// Последовательный F64, без map-итераций и горутин — бит-детерминизм
// по построению (религия линеек, как в fwdBattleAF64).
func fwdLlamaHFF64(w *llamaHFF64Weights, toks []int) []float64 {
	c := w.Cfg
	M := len(toks)
	D, V, F := c.HiddenSize, c.VocabSize, c.IntermediateSize
	H, HKV, HD := c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim()
	G := c.KVGroups()
	KVD := HKV * HD
	eps := c.RMSNormEps
	scale := 1.0 / math.Sqrt(float64(HD))

	x := make([]float64, M*D)
	for i, t := range toks {
		copy(x[i*D:(i+1)*D], w.Embed[t*D:(t+1)*D])
	}
	xn := make([]float64, M*D)
	q := make([]float64, M*D)
	k := make([]float64, M*KVD)
	v := make([]float64, M*KVD)
	qh := make([]float64, M*D)
	kh := make([]float64, M*KVD)
	vh := make([]float64, M*KVD)
	oh := make([]float64, M*D)
	attn := make([]float64, M*D)
	ao := make([]float64, M*D)
	g := make([]float64, M*F)
	u := make([]float64, M*F)
	ffn := make([]float64, M*D)
	prow := make([]float64, M)

	for l := range w.Layers {
		lw := &w.Layers[l]
		// 1. pre-attention RMSNorm
		rmsNormF64(xn, x, lw.Norm1, M, D, eps)
		// 2. проекции: y = x @ W^T, W в HF-layout [out,in]
		matF64_ABT(q, xn, lw.Wq, M, D, D)
		matF64_ABT(k, xn, lw.Wk, M, KVD, D)
		matF64_ABT(v, xn, lw.Wv, M, KVD, D)
		// 3. [S,H,HD] -> [H,S,HD]
		permuteSHD_HSD(qh, q, 1, M, H, HD)
		permuteSHD_HSD(kh, k, 1, M, HKV, HD)
		permuteSHD_HSD(vh, v, 1, M, HKV, HD)
		// 4. RoPE (half-split == HF rotate_half), pos = row % M
		ropeF64(qh, H, M, HD, c.RopeTheta)
		ropeF64(kh, HKV, M, HD, c.RopeTheta)
		// 5. causal attention, GQA: голова h читает kv-голову h/G
		for h := 0; h < H; h++ {
			kvh := h / G
			qBase, kvBase := h*M*HD, kvh*M*HD
			for i := 0; i < M; i++ {
				// скоры j <= i c max-subtract softmax по префиксу
				mx := math.Inf(-1)
				for j := 0; j <= i; j++ {
					var acc float64
					for p := 0; p < HD; p++ {
						acc += qh[qBase+i*HD+p] * kh[kvBase+j*HD+p]
					}
					acc *= scale
					prow[j] = acc
					if acc > mx {
						mx = acc
					}
				}
				var sum float64
				for j := 0; j <= i; j++ {
					prow[j] = math.Exp(prow[j] - mx)
					sum += prow[j]
				}
				inv := 1.0 / sum
				dst := oh[qBase+i*HD : qBase+(i+1)*HD]
				for p := 0; p < HD; p++ {
					dst[p] = 0
				}
				for j := 0; j <= i; j++ {
					pj := prow[j] * inv
					src := vh[kvBase+j*HD : kvBase+(j+1)*HD]
					for p := 0; p < HD; p++ {
						dst[p] += pj * src[p]
					}
				}
			}
		}
		// 6. [H,S,HD] -> [S,H,HD] == [M,D], o-proj, residual
		permuteHSD_SHD(attn, oh, 1, M, H, HD)
		matF64_ABT(ao, attn, lw.Wo, M, D, D)
		for i := range x {
			x[i] += ao[i]
		}
		// 7. pre-FFN RMSNorm + SwiGLU: down( silu(gate(x)) * up(x) )
		rmsNormF64(xn, x, lw.Norm2, M, D, eps)
		matF64_ABT(g, xn, lw.Gate, M, F, D)
		matF64_ABT(u, xn, lw.Up, M, F, D)
		for i := range g {
			g[i] = g[i] / (1.0 + math.Exp(-g[i])) * u[i] // silu(g)*u
		}
		matF64_ABT(ffn, g, lw.Down, M, D, F)
		for i := range x {
			x[i] += ffn[i]
		}
	}
	// финальная норма + логиты последней позиции
	rmsNormF64(xn, x, w.NormOut, M, D, eps)
	logits := make([]float64, V)
	matF64_ABT(logits, xn[(M-1)*D:M*D], w.LMHead, 1, V, D)
	return logits
}
