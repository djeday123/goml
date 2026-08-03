package abjexam

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"github.com/djeday123/goml/hf"
)

// Гейт I.5 карточки HF-INFERENCE: CPU-F64 forward TinyLlama на HF-весах,
// top-10 логитов последней позиции против HF-референса (transformers,
// torch_dtype=float32, CPU; снят один раз snap_logits_ref.py).
//
// FLOOR (записан ДО прогона, см. runs/hf_inference/F1_predictions.md):
//   - множество и ПОРЯДОК top-10 id совпадают;
//   - rel = |ours-ref| / max(1,|ref|) <= 1e-3 для каждого из top-10;
//     ожидание ~1e-4: веса бит-эквивалентны (BF16->F32/F64 точны),
//     расхождение — только F32-компьют HF против нашего F64.

type logitsRef struct {
	Model        string  `json:"model"`
	Prompt       string  `json:"prompt"`
	InputIDs     []int   `json:"input_ids"`
	TorchDtype   string  `json:"torch_dtype"`
	Top10IDs     []int   `json:"top10_ids"`
	Top10Logits  []float64 `json:"top10_logits"`
	Top10Tokens  []string  `json:"top10_tokens"`
	ArgmaxID     int     `json:"argmax_id"`
	LogitsFirst8 []float64 `json:"logits_first8"`
}

func TestLlamaHFF64SmokeVsHFRef(t *testing.T) {
	base := os.Getenv("GOML_HF_DATA")
	if base == "" {
		base = filepath.Join("..", "..", "runs", "hf_inference")
	}
	dir := filepath.Join(base, "tinyllama")
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("нет весов: %v", err)
	}
	rawRef, err := os.ReadFile(filepath.Join(base, "logits_ref.json"))
	if err != nil {
		t.Skipf("нет logits_ref.json: %v", err)
	}
	var ref logitsRef
	if err := json.Unmarshal(rawRef, &ref); err != nil {
		t.Fatalf("logits_ref.json: %v", err)
	}

	tok, err := hf.LoadLlamaTokenizer(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	ids := tok.Encode(ref.Prompt, true)
	if len(ids) != len(ref.InputIDs) {
		t.Fatalf("токенизация промпта разошлась с референсом: %v vs %v", ids, ref.InputIDs)
	}
	for i := range ids {
		if ids[i] != ref.InputIDs[i] {
			t.Fatalf("токенизация промпта разошлась с референсом: %v vs %v", ids, ref.InputIDs)
		}
	}

	t0 := time.Now()
	w, err := loadLlamaHFF64(dir)
	if err != nil {
		t.Fatalf("load weights: %v", err)
	}
	tLoad := time.Since(t0)

	t0 = time.Now()
	logits := fwdLlamaHFF64(w, ids)
	tFwd := time.Since(t0)
	t.Logf("веса загружены за %v, forward %d токенов за %v", tLoad, len(ids), tFwd)

	// наш top-10
	idx := make([]int, len(logits))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return logits[idx[a]] > logits[idx[b]] })
	ourTop := idx[:10]

	t.Logf("промпт: %q -> ids %v", ref.Prompt, ids)
	// ТЗ-02 П.2.1: печатаются ВСЕ сравниваемые пары обоих множеств
	// (top10 и first8) — логика гейта не изменена, только полнота печати.
	t.Logf("%-6s %-4s %-8s %-16s %-14s %-14s %s", "set", "rank", "id", "token", "ours(F64)", "HF ref(F32)", "rel")
	worstRel := 0.0
	for r := 0; r < 10; r++ {
		refID := ref.Top10IDs[r]
		rel := math.Abs(logits[refID]-ref.Top10Logits[r]) / math.Max(1.0, math.Abs(ref.Top10Logits[r]))
		if rel > worstRel {
			worstRel = rel
		}
		t.Logf("top10  %-4d %-8d %-16q %-14.8f %-14.8f %.3e", r, refID, ref.Top10Tokens[r], logits[refID], ref.Top10Logits[r], rel)
		if ourTop[r] != refID {
			t.Errorf("rank %d: наш id %d (%q), референс %d (%q)", r, ourTop[r], tok.Decode([]int{ourTop[r]}), refID, ref.Top10Tokens[r])
		}
	}
	if ourTop[0] != ref.ArgmaxID {
		t.Errorf("argmax: наш %d, референс %d", ourTop[0], ref.ArgmaxID)
	}
	t.Logf("greedy next token: %q", tok.Decode([]int{ourTop[0]}))
	for i := 0; i < 8; i++ {
		rel := math.Abs(logits[i]-ref.LogitsFirst8[i]) / math.Max(1.0, math.Abs(ref.LogitsFirst8[i]))
		if rel > worstRel {
			worstRel = rel
		}
		t.Logf("first8 %-4d %-8d %-16s %-14.8f %-14.8f %.3e", i, i, "-", logits[i], ref.LogitsFirst8[i], rel)
	}
	const floor = 1e-3 // класс из карточки, записан до прогона
	t.Logf("worst rel (top-10 + first8) = %.3e, floor = %.0e", worstRel, floor)
	if worstRel > floor {
		t.Fatalf("ГЕЙТ I.5 ПРОВАЛЕН: worst rel %.3e > floor %.0e", worstRel, floor)
	}
	t.Logf("ГЕЙТ I.5 PASS")
}

// TestLlamaHFF64NegativeControl — ТЗ-02 П.3: проба инъективности гейта I.5.
// Доказывает, что тест реально считает forward, а не сверяет референс сам
// с собой: ОДИН элемент весов портится, worst rel ОБЯЗАН превысить floor 1e-3
// (assert на превышение, не на прохождение).
//
// Возмущение (попытка Б): LMHead[ArgmaxID*D+0] += 100.0 — одно значение,
// строка lm_head argmax-токена. Путь порча->логит АНАЛИТИЧЕСКИ прямой:
// Δlogit(argmax) = 100·xn_final[0], без softmax/нормализаций/голов между.
// Попытка А (Layers[0].Wq[0] += 1.0) — задокументированный ПРОГНОЗ-ПРОМАХ:
// worst rel 5.082e-07, одноэлементная порча глубинного веса гасится трактом
// (raw_f1hf/i5_negative_control_attemptA.txt; разбор в F1_predictions.md) —
// негодный инструмент контроля, оставлена в истории как урок.
//
// Копия весов СВОЯ: loadLlamaHFF64 читает файл заново, основной тест не
// затронут (П.3.4). Floor и прогноз величины записаны ДО прогона:
// runs/hf_inference/F1_predictions.md, разделы "ТЗ-02 отрицательный контроль".
func TestLlamaHFF64NegativeControl(t *testing.T) {
	base := os.Getenv("GOML_HF_DATA")
	if base == "" {
		base = filepath.Join("..", "..", "runs", "hf_inference")
	}
	dir := filepath.Join(base, "tinyllama")
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("нет весов: %v", err)
	}
	rawRef, err := os.ReadFile(filepath.Join(base, "logits_ref.json"))
	if err != nil {
		t.Skipf("нет logits_ref.json: %v", err)
	}
	var ref logitsRef
	if err := json.Unmarshal(rawRef, &ref); err != nil {
		t.Fatalf("logits_ref.json: %v", err)
	}

	w, err := loadLlamaHFF64(dir) // своя копия весов, с диска
	if err != nil {
		t.Fatalf("load weights: %v", err)
	}
	pidx := ref.ArgmaxID*w.Cfg.HiddenSize + 0
	orig := w.LMHead[pidx]
	w.LMHead[pidx] += 100.0
	t.Logf("порча: LMHead[%d*D+0] %.8e -> %.8e (+100.0)", ref.ArgmaxID, orig, w.LMHead[pidx])

	logits := fwdLlamaHFF64(w, ref.InputIDs)

	// То же множество сравнений и та же формула rel, что в основном гейте.
	t.Logf("%-6s %-4s %-8s %-14s %-14s %s", "set", "rank", "id", "ours(порчен.)", "HF ref(F32)", "rel")
	worstRel := 0.0
	for r := 0; r < 10; r++ {
		refID := ref.Top10IDs[r]
		rel := math.Abs(logits[refID]-ref.Top10Logits[r]) / math.Max(1.0, math.Abs(ref.Top10Logits[r]))
		if rel > worstRel {
			worstRel = rel
		}
		t.Logf("top10  %-4d %-8d %-14.8f %-14.8f %.3e", r, refID, logits[refID], ref.Top10Logits[r], rel)
	}
	for i := 0; i < 8; i++ {
		rel := math.Abs(logits[i]-ref.LogitsFirst8[i]) / math.Max(1.0, math.Abs(ref.LogitsFirst8[i]))
		if rel > worstRel {
			worstRel = rel
		}
		t.Logf("first8 %-4d %-8d %-14.8f %-14.8f %.3e", i, i, logits[i], ref.LogitsFirst8[i], rel)
	}

	const floor = 1e-3
	t.Logf("worst rel (top-10 + first8) = %.3e, floor = %.0e", worstRel, floor)
	if worstRel <= floor {
		t.Fatalf("ОТРИЦАТЕЛЬНЫЙ КОНТРОЛЬ ПРОВАЛЕН: порча веса невидима гейту, worst rel %.3e <= %.0e", worstRel, floor)
	}
	t.Logf("ОТРИЦАТЕЛЬНЫЙ КОНТРОЛЬ PASS: порча одного веса видна, worst rel %.3e > %.0e", worstRel, floor)
}
