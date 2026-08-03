package hf

import (
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// tokenizerRef — снятый ОДИН РАЗ HF-референс (runs/hf_inference/tokenizer_ref.json,
// снят snap_tokenizer_ref.py, tokenizers==0.21.4). Гейт I.3.
type tokenizerRef struct {
	Model            string `json:"model"`
	TokenizerJSONMD5 string `json:"tokenizer_json_md5"`
	BosProbe         struct {
		Text string `json:"text"`
		IDs  []int  `json:"ids"`
	} `json:"bos_probe"`
	NCases int `json:"n_cases"`
	Cases  []struct {
		Text    string `json:"text"`
		IDs     []int  `json:"ids"`
		Decoded string `json:"decoded"`
	} `json:"cases"`
}

// TestLlamaTokenizerVsHFRef — гейт I.3: encode 100 строк == HF-референсу,
// decode(ids) == HF-декоду, BOS-проба, md5 tokenizer.json == зафиксированному.
func TestLlamaTokenizerVsHFRef(t *testing.T) {
	dir := hfDataDir(t)
	tokPath := filepath.Join(dir, "tokenizer.json")
	refPath := filepath.Join(dir, "..", "tokenizer_ref.json")
	if _, err := os.Stat(tokPath); err != nil {
		t.Skipf("нет tokenizer.json: %v", err)
	}
	rawRef, err := os.ReadFile(refPath)
	if err != nil {
		t.Skipf("нет tokenizer_ref.json: %v", err)
	}
	var ref tokenizerRef
	if err := json.Unmarshal(rawRef, &ref); err != nil {
		t.Fatalf("tokenizer_ref.json: %v", err)
	}
	if len(ref.Cases) != ref.NCases {
		t.Fatalf("референс повреждён: %d кейсов, заявлено %d", len(ref.Cases), ref.NCases)
	}

	rawTok, err := os.ReadFile(tokPath)
	if err != nil {
		t.Fatalf("tokenizer.json: %v", err)
	}
	sum := md5.Sum(rawTok)
	if got := hex.EncodeToString(sum[:]); got != ref.TokenizerJSONMD5 {
		t.Fatalf("md5(tokenizer.json) = %s, референс снят с %s — файлы разошлись", got, ref.TokenizerJSONMD5)
	}

	tok, err := LoadLlamaTokenizer(tokPath)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if tok.BosID != 1 || tok.EosID != 2 || tok.UnkID != 0 {
		t.Fatalf("спецтокены: bos=%d eos=%d unk=%d, ожидалось 1/2/0", tok.BosID, tok.EosID, tok.UnkID)
	}
	if tok.VocabSize() != 32000 {
		t.Fatalf("vocab %d, ожидалось 32000", tok.VocabSize())
	}

	encodeFail, decodeFail := 0, 0
	for i, c := range ref.Cases {
		got := tok.Encode(c.Text, false)
		if !eqInts(got, c.IDs) {
			encodeFail++
			if encodeFail <= 5 {
				t.Errorf("case %d encode %q:\n  got  %v\n  want %v", i, trunc(c.Text), got, c.IDs)
			}
		}
		if dec := tok.Decode(c.IDs); dec != c.Decoded {
			decodeFail++
			if decodeFail <= 5 {
				t.Errorf("case %d decode %q:\n  got  %q\n  want %q", i, trunc(c.Text), dec, c.Decoded)
			}
		}
	}
	if encodeFail+decodeFail > 0 {
		t.Fatalf("ГЕЙТ ПРОВАЛЕН: encode fail %d/%d, decode fail %d/%d",
			encodeFail, len(ref.Cases), decodeFail, len(ref.Cases))
	}
	if got := tok.Encode(ref.BosProbe.Text, true); !eqInts(got, ref.BosProbe.IDs) {
		t.Fatalf("BOS-проба: got %v, want %v", got, ref.BosProbe.IDs)
	}
	t.Logf("ГЕЙТ I.3 PASS: %d/%d encode==HF, %d/%d decode==HF, BOS-проба OK",
		len(ref.Cases)-encodeFail, len(ref.Cases), len(ref.Cases)-decodeFail, len(ref.Cases))
}

func eqInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func trunc(s string) string {
	if len(s) > 40 {
		return s[:40] + "…"
	}
	return s
}
