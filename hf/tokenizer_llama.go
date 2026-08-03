package hf

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// LlamaTokenizer — загрузчик HF tokenizer.json для sentencepiece-BPE
// llama-класса (TinyLlama/Llama-2): normalizer = Prepend "▁" + Replace " "->"▁",
// pre_tokenizer = null, model.type = BPE c byte_fallback, decoder =
// Replace "▁"->" " + ByteFallback + Fuse + Strip(1 ведущий пробел).
// Наш собственный BPE (tokenizer/bpe.go) с этим форматом несовместим
// (другая ID-раскладка, нет метапробела) — вердикт I.3: новый загрузчик.
type LlamaTokenizer struct {
	tokens    []string       // id -> токен
	vocab     map[string]int // токен -> id
	ranks     map[string]int // "A B" -> ранг мержа
	isByte    []bool         // id — байтовый токен <0xNN>
	byteVal   []byte         // значение байта для байтовых токенов
	byteID    [256]int       // байт -> id токена <0xNN> (-1 если нет)
	isSpecial []bool         // added_tokens special
	BosID     int
	EosID     int
	UnkID     int
}

type tokenizerJSON struct {
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
	PreTokenizer json.RawMessage `json:"pre_tokenizer"`
	Model        struct {
		Type         string            `json:"type"`
		UnkToken     string            `json:"unk_token"`
		ByteFallback bool              `json:"byte_fallback"`
		Vocab        map[string]int    `json:"vocab"`
		Merges       []json.RawMessage `json:"merges"` // строки "A B" или пары [A,B]
	} `json:"model"`
}

// LoadLlamaTokenizer читает tokenizer.json.
func LoadLlamaTokenizer(path string) (*LlamaTokenizer, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var tj tokenizerJSON
	if err := json.Unmarshal(raw, &tj); err != nil {
		return nil, fmt.Errorf("tokenizer.json: %w", err)
	}
	if tj.Model.Type != "BPE" {
		return nil, fmt.Errorf("model.type %q: поддерживается только BPE", tj.Model.Type)
	}
	if !tj.Model.ByteFallback {
		return nil, fmt.Errorf("byte_fallback=false: ожидается sentencepiece-стиль llama")
	}
	if string(tj.PreTokenizer) != "null" && len(tj.PreTokenizer) != 0 {
		return nil, fmt.Errorf("pre_tokenizer != null не поддержан: %s", tj.PreTokenizer)
	}

	maxID := 0
	for _, id := range tj.Model.Vocab {
		if id > maxID {
			maxID = id
		}
	}
	for _, at := range tj.AddedTokens {
		if at.ID > maxID {
			maxID = at.ID
		}
	}
	t := &LlamaTokenizer{
		tokens:    make([]string, maxID+1),
		vocab:     make(map[string]int, len(tj.Model.Vocab)),
		ranks:     make(map[string]int, len(tj.Model.Merges)),
		isByte:    make([]bool, maxID+1),
		byteVal:   make([]byte, maxID+1),
		isSpecial: make([]bool, maxID+1),
		BosID:     -1, EosID: -1, UnkID: -1,
	}
	for i := range t.byteID {
		t.byteID[i] = -1
	}
	for tok, id := range tj.Model.Vocab {
		t.tokens[id] = tok
		t.vocab[tok] = id
		var b byte
		if n, _ := fmt.Sscanf(tok, "<0x%02X>", &b); n == 1 && len(tok) == 6 {
			t.isByte[id] = true
			t.byteVal[id] = b
			t.byteID[b] = id
		}
	}
	for _, at := range tj.AddedTokens {
		t.tokens[at.ID] = at.Content
		if _, exists := t.vocab[at.Content]; !exists {
			t.vocab[at.Content] = at.ID
		}
		t.isSpecial[at.ID] = at.Special
		switch at.Content {
		case "<s>":
			t.BosID = at.ID
		case "</s>":
			t.EosID = at.ID
		case tj.Model.UnkToken:
			t.UnkID = at.ID
		}
	}
	for rank, rawMerge := range tj.Model.Merges {
		var s string
		if err := json.Unmarshal(rawMerge, &s); err == nil {
			t.ranks[s] = rank
			continue
		}
		var pair [2]string
		if err := json.Unmarshal(rawMerge, &pair); err != nil {
			return nil, fmt.Errorf("merge #%d: ни строка, ни пара: %s", rank, rawMerge)
		}
		t.ranks[pair[0]+" "+pair[1]] = rank
	}
	return t, nil
}

// VocabSize — размер словаря (включая added_tokens).
func (t *LlamaTokenizer) VocabSize() int { return len(t.tokens) }

// Encode: normalizer (Prepend "▁" для непустой строки + " "->"▁"), затем
// ранговый BPE по всей строке (pre_tokenizer отсутствует), затем byte_fallback
// для символов вне словаря. addBOS воспроизводит post_processor (<s> в начале).
func (t *LlamaTokenizer) Encode(text string, addBOS bool) []int {
	ids := []int{}
	if addBOS && t.BosID >= 0 {
		ids = append(ids, t.BosID)
	}
	if text == "" {
		return ids
	}
	norm := "▁" + strings.ReplaceAll(text, " ", "▁")
	syms := make([]string, 0, len(norm))
	for _, r := range norm {
		syms = append(syms, string(r))
	}
	// Классический BPE-цикл: пара минимального ранга, слияние всех её
	// вхождений слева направо. Эквивалентен heap-алгоритму HF: ранги пар,
	// содержащих свежесозданный токен, строго больше ранга его создания.
	for len(syms) > 1 {
		bestRank, bestIdx := int(^uint(0)>>1), -1
		for i := 0; i+1 < len(syms); i++ {
			if r, ok := t.ranks[syms[i]+" "+syms[i+1]]; ok && r < bestRank {
				bestRank, bestIdx = r, i
			}
		}
		if bestIdx < 0 {
			break
		}
		a, b := syms[bestIdx], syms[bestIdx+1]
		merged := make([]string, 0, len(syms))
		for i := 0; i < len(syms); {
			if i+1 < len(syms) && syms[i] == a && syms[i+1] == b {
				merged = append(merged, a+b)
				i += 2
			} else {
				merged = append(merged, syms[i])
				i++
			}
		}
		syms = merged
	}
	for _, s := range syms {
		if id, ok := t.vocab[s]; ok {
			ids = append(ids, id)
			continue
		}
		for _, b := range []byte(s) { // byte_fallback
			if t.byteID[b] >= 0 {
				ids = append(ids, t.byteID[b])
			} else {
				ids = append(ids, t.UnkID)
			}
		}
	}
	return ids
}

// Decode: байтовые токены -> сырые байты, остальные -> "▁"->" ",
// спецтокены — как есть; в конце Strip одного ведущего пробела.
func (t *LlamaTokenizer) Decode(ids []int) string {
	var buf []byte
	for _, id := range ids {
		if id < 0 || id >= len(t.tokens) {
			continue
		}
		switch {
		case t.isByte[id]:
			buf = append(buf, t.byteVal[id])
		case t.isSpecial[id]:
			buf = append(buf, t.tokens[id]...)
		default:
			buf = append(buf, strings.ReplaceAll(t.tokens[id], "▁", " ")...)
		}
	}
	return strings.TrimPrefix(string(buf), " ")
}
