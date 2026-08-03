// Package hf — чтение артефактов Hugging Face (config.json, safetensors,
// tokenizer.json) для инференса Llama-класса на нашем стеке.
// Карточка HF-INFERENCE, Фаза I (CPU-only, ядра не трогает).
package hf

import (
	"encoding/json"
	"fmt"
	"os"
)

// LlamaConfig — поля config.json архитектуры LlamaForCausalLM.
type LlamaConfig struct {
	Architectures         []string `json:"architectures"`
	AttentionBias         bool     `json:"attention_bias"`
	BosTokenID            int      `json:"bos_token_id"`
	EosTokenID            int      `json:"eos_token_id"`
	HiddenAct             string   `json:"hidden_act"`
	HiddenSize            int      `json:"hidden_size"`
	IntermediateSize      int      `json:"intermediate_size"`
	MaxPositionEmbeddings int      `json:"max_position_embeddings"`
	ModelType             string   `json:"model_type"`
	NumAttentionHeads     int      `json:"num_attention_heads"`
	NumHiddenLayers       int      `json:"num_hidden_layers"`
	NumKeyValueHeads      int      `json:"num_key_value_heads"`
	RMSNormEps            float64  `json:"rms_norm_eps"`
	RopeScaling           any      `json:"rope_scaling"`
	RopeTheta             float64  `json:"rope_theta"`
	TieWordEmbeddings     bool     `json:"tie_word_embeddings"`
	TorchDtype            string   `json:"torch_dtype"`
	VocabSize             int      `json:"vocab_size"`
}

// HeadDim = hidden_size / num_attention_heads.
func (c *LlamaConfig) HeadDim() int { return c.HiddenSize / c.NumAttentionHeads }

// KVGroups = num_attention_heads / num_key_value_heads (G для GQA; 1 = MHA).
func (c *LlamaConfig) KVGroups() int { return c.NumAttentionHeads / c.NumKeyValueHeads }

// LoadLlamaConfig читает и валидирует config.json против границ карточки:
// llama-класс, hd in {64,128}, без rope_scaling, без attention_bias.
func LoadLlamaConfig(path string) (*LlamaConfig, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var c LlamaConfig
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("config.json: %w", err)
	}
	if c.ModelType != "llama" {
		return nil, fmt.Errorf("model_type %q: поддерживается только llama", c.ModelType)
	}
	if c.NumAttentionHeads <= 0 || c.HiddenSize%c.NumAttentionHeads != 0 {
		return nil, fmt.Errorf("hidden_size %d не делится на num_attention_heads %d", c.HiddenSize, c.NumAttentionHeads)
	}
	if hd := c.HeadDim(); hd != 64 && hd != 128 {
		return nil, fmt.Errorf("head_dim %d вне {64,128} (граница карточки)", hd)
	}
	if c.NumKeyValueHeads <= 0 || c.NumAttentionHeads%c.NumKeyValueHeads != 0 {
		return nil, fmt.Errorf("num_attention_heads %d не делится на num_key_value_heads %d", c.NumAttentionHeads, c.NumKeyValueHeads)
	}
	if c.RopeScaling != nil {
		return nil, fmt.Errorf("rope_scaling не поддержан в Фазе I: %v", c.RopeScaling)
	}
	if c.AttentionBias {
		return nil, fmt.Errorf("attention_bias=true не поддержан (llama-канон — без bias)")
	}
	return &c, nil
}
