package hf

import (
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// hfDataDir — каталог с артефактами модели (config.json, tokenizer.json,
// model.safetensors). Веса в git не входят — тест скипается без них.
func hfDataDir(t *testing.T) string {
	t.Helper()
	dir := os.Getenv("GOML_HF_DATA")
	if dir == "" {
		dir = filepath.Join("..", "runs", "hf_inference", "tinyllama")
	}
	return dir
}

// expectedLlamaTensors — полный список тензоров LlamaForCausalLM
// (untied) по конфигу: имена и формы в HF-layout [out,in].
func expectedLlamaTensors(c *LlamaConfig) map[string][]int64 {
	D := int64(c.HiddenSize)
	V := int64(c.VocabSize)
	F := int64(c.IntermediateSize)
	KVD := int64(c.NumKeyValueHeads * c.HeadDim())
	exp := map[string][]int64{
		"model.embed_tokens.weight": {V, D},
		"model.norm.weight":         {D},
		"lm_head.weight":            {V, D},
	}
	for l := 0; l < c.NumHiddenLayers; l++ {
		p := fmt.Sprintf("model.layers.%d.", l)
		exp[p+"input_layernorm.weight"] = []int64{D}
		exp[p+"post_attention_layernorm.weight"] = []int64{D}
		exp[p+"self_attn.q_proj.weight"] = []int64{D, D}
		exp[p+"self_attn.k_proj.weight"] = []int64{KVD, D}
		exp[p+"self_attn.v_proj.weight"] = []int64{KVD, D}
		exp[p+"self_attn.o_proj.weight"] = []int64{D, D}
		exp[p+"mlp.gate_proj.weight"] = []int64{F, D}
		exp[p+"mlp.up_proj.weight"] = []int64{F, D}
		exp[p+"mlp.down_proj.weight"] = []int64{D, F}
	}
	return exp
}

// TestSafeTensorsTinyLlama — гейт I.2: заголовок против ожиданий из config.json
// (count/shapes/dtype), загрузка ВСЕХ тензоров с конверсией BF16->F32,
// проверка конечности значений, md5 файла в лог.
func TestSafeTensorsTinyLlama(t *testing.T) {
	dir := hfDataDir(t)
	stPath := filepath.Join(dir, "model.safetensors")
	if _, err := os.Stat(stPath); err != nil {
		t.Skipf("нет весов: %v (скачать model.safetensors в %s)", err, dir)
	}
	cfg, err := LoadLlamaConfig(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatalf("config: %v", err)
	}
	st, err := OpenSafeTensors(stPath)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer st.Close()

	exp := expectedLlamaTensors(cfg)
	if cfg.TieWordEmbeddings {
		delete(exp, "lm_head.weight")
	}
	if got, want := len(st.Tensors), len(exp); got != want {
		t.Errorf("count: %d тензоров, ожидалось %d", got, want)
	}
	for name, shape := range exp {
		ti, ok := st.Tensors[name]
		if !ok {
			t.Errorf("отсутствует %s", name)
			continue
		}
		if len(ti.Shape) != len(shape) {
			t.Errorf("%s: shape %v, ожидалось %v", name, ti.Shape, shape)
			continue
		}
		for i := range shape {
			if ti.Shape[i] != shape[i] {
				t.Errorf("%s: shape %v, ожидалось %v", name, ti.Shape, shape)
				break
			}
		}
		if ti.Dtype != cfgDtype(cfg) {
			t.Errorf("%s: dtype %s, ожидалось %s", name, ti.Dtype, cfgDtype(cfg))
		}
	}
	for name := range st.Tensors {
		if _, ok := exp[name]; !ok {
			t.Errorf("неожиданный тензор %s", name)
		}
	}

	// Загрузка всех тензоров + конечность (NaN/Inf в весах = порча файла).
	var totalElems int64
	for _, name := range st.Names {
		v, err := st.ReadF32(name)
		if err != nil {
			t.Fatalf("ReadF32(%s): %v", name, err)
		}
		for i, x := range v {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				t.Fatalf("%s[%d] = %v (не конечно)", name, i, x)
			}
		}
		totalElems += int64(len(v))
	}
	t.Logf("загружено тензоров: %d, элементов: %d (%.3fB параметров)",
		len(st.Names), totalElems, float64(totalElems)/1e9)

	f, err := os.Open(stPath)
	if err != nil {
		t.Fatalf("md5: %v", err)
	}
	defer f.Close()
	h := md5.New()
	if _, err := io.Copy(h, f); err != nil {
		t.Fatalf("md5: %v", err)
	}
	t.Logf("md5(model.safetensors) = %s", hex.EncodeToString(h.Sum(nil)))
}

func cfgDtype(c *LlamaConfig) string {
	switch c.TorchDtype {
	case "bfloat16":
		return "BF16"
	case "float16":
		return "F16"
	case "float32":
		return "F32"
	}
	return "?"
}

// TestF16ToF32 — точечная проверка half->single (в TinyLlama F16 нет,
// но конвертер обязан быть корректным до появления F16-моделей).
func TestF16ToF32(t *testing.T) {
	cases := []struct {
		h    uint16
		want float32
	}{
		{0x3C00, 1.0},
		{0x3800, 0.5},
		{0xC000, -2.0},
		{0x7BFF, 65504},
		{0x0001, 5.9604645e-08}, // минимальный денормал
		{0x03FF, 6.0975552e-05}, // максимальный денормал
		{0x0000, 0.0},
	}
	for _, c := range cases {
		if got := F16ToF32(c.h); got != c.want {
			t.Errorf("F16ToF32(%#04x) = %v, ожидалось %v", c.h, got, c.want)
		}
	}
	if !math.IsInf(float64(F16ToF32(0x7C00)), 1) {
		t.Errorf("0x7C00 должен быть +Inf")
	}
	if !math.IsNaN(float64(F16ToF32(0x7E00))) {
		t.Errorf("0x7E00 должен быть NaN")
	}
	if math.Float32bits(F16ToF32(0x8000)) != 0x80000000 {
		t.Errorf("0x8000 должен быть -0")
	}
}

// TestBF16ToF32 — битовая точность расширения.
func TestBF16ToF32(t *testing.T) {
	cases := []struct {
		h    uint16
		want float32
	}{
		{0x3F80, 1.0},
		{0x3F00, 0.5},
		{0xC000, -2.0},
		{0x0000, 0.0},
	}
	for _, c := range cases {
		if got := BF16ToF32(c.h); got != c.want {
			t.Errorf("BF16ToF32(%#04x) = %v, ожидалось %v", c.h, got, c.want)
		}
	}
}
