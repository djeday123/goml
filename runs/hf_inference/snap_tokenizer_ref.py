# Снятие HF-референса токенизатора TinyLlama (I.3, карточка HF-INFERENCE).
# Референс снимается ОДИН РАЗ и фиксируется в tokenizer_ref.json (тест-данные).
# Запуск: venv/bin/python snap_tokenizer_ref.py
import json, hashlib, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.join(HERE, "tinyllama", "tokenizer.json")

from tokenizers import Tokenizer
tok = Tokenizer.from_file(TOK)

# 100 строк: ASCII, юникод, эмодзи, код, пробельные края, цифры, CJK, RU.
strings = []
strings += [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "hello world",
    " hello world",
    "hello  world",  # двойной пробел
    "\thello\n\tworld\n",
    "",
    " ",
    "\n",
    "\t",
]
strings += [
    "Привет, мир!",
    "Съешь же ещё этих мягких французских булок, да выпей чаю.",
    "Тензор внимания причинной маски",
    "ёжик Ёж ЁЖИК",
    "Быстрая коричневая лиса",
    "Многоуровневая архитектура трансформера",
    "два  пробела и\tтаб",
    "русский и english вперемешку",
    "КАПС И строчные",
    "цифры 1234567890 и слова",
]
strings += [
    "你好，世界！",
    "机器学习模型推理",
    "日本語のテキストです。",
    "한국어 텍스트 예시입니다.",
    "中文 English 混合 text",
    "العربية نص تجريبي",
    "עברית טקסט לדוגמה",
    "Ελληνικά κείμενο",
    "ภาษาไทยทดสอบ",
    "हिन्दी परीक्षण पाठ",
]
strings += [
    "😀",
    "🚀🔥💯",
    "emoji 😀 in text",
    "family: 👨‍👩‍👧‍👦 zwj",
    "flags 🇺🇸🇯🇵",
    "skin tone 👍🏽",
    "café naïve résumé",
    "Zürich München Köln",
    "ñandú piñata",
    "øre Å å æ",
]
strings += [
    "func main() {\n\tfmt.Println(\"hello\")\n}",
    "def f(x):\n    return x * 2\n",
    "SELECT * FROM users WHERE id = 42;",
    "x = np.zeros((32, 64), dtype=np.float32)",
    "if (a && b) || !c { return nil }",
    "#include <stdio.h>\nint main(void){return 0;}",
    "curl -sL https://example.com/api?q=1&r=2",
    "{\"key\": \"value\", \"n\": [1, 2, 3]}",
    "<div class=\"row\"><span>hi</span></div>",
    "regex: ^[a-zA-Z0-9_]+$ and \\d{3}-\\d{4}",
]
strings += [
    "path/to/file.txt",
    "C:\\Windows\\System32\\drivers",
    "user@example.com",
    "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "0x1p-52 and 1e-8 and 3.14159",
    "CUDA_VISIBLE_DEVICES=0 ./run.sh --flag=true",
    "a" * 100,
    "ab " * 50,
    "word",
    "  leading and trailing  ",
]
strings += [
    "The transformer architecture relies on self-attention mechanisms.",
    "RMSNorm normalizes activations by their root mean square.",
    "Rotary position embeddings encode relative positions.",
    "Grouped-query attention reduces KV cache memory.",
    "Byte-pair encoding merges frequent symbol pairs iteratively.",
    "In 2026, GPUs compute attention in FP8 precision.",
    "Los modelos de lenguaje generan texto coherente.",
    "Les modèles de langage génèrent du texte.",
    "Sprachmodelle erzeugen kohärenten Text.",
    "I modelli linguistici generano testo.",
]
strings += [
    "1", "12", "123", "1234", "12345",
    "3.14", "-42", "1,000,000", "2048 tokens", "0.999e+10",
]
strings += [
    "don't can't won't it's",
    "well-known state-of-the-art",
    "a.b.c.d.e",
    "one;two;three",
    "quote \"inside\" quotes",
    "'single' and \"double\"",
    "ellipsis... and dash — em",
    "«ёлки» и „кавычки“",
    "50% of $100 is €50",
    "§1.2 ¶3 †note ‡ref",
]
strings += [
    " nbsp here",
    "zero​width​space",
    "combining á é",
    "ﬁ ﬂ ligatures",
    "ROMAN Ⅳ Ⅸ numerals",
    "math ∑∫√ symbols ≤ ≥ ≠",
    "arrows → ← ↑ ↓ ⇒",
    "box ┌─┐│└┘ drawing",
    "ⓤⓝⓘⓒⓞⓓⓔ circled",
    "𝕬𝖓𝖈𝖎𝖊𝖓𝖙 𝔊𝔬𝔱𝔥𝔦𝔠",
]

assert len(strings) == 100, len(strings)

cases = []
for s in strings:
    # add_special_tokens=False: чистый BPE-путь, BOS проверяется отдельным полем ниже
    enc = tok.encode(s, add_special_tokens=False)
    dec = tok.decode(enc.ids, skip_special_tokens=False)
    cases.append({"text": s, "ids": enc.ids, "decoded": dec})

# BOS-поведение (как full pipeline с спецтокенами)
bos_probe = tok.encode("Hello, world!", add_special_tokens=True)

with open(TOK, "rb") as f:
    tok_md5 = hashlib.md5(f.read()).hexdigest()

out = {
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "tokenizer_json_md5": tok_md5,
    "tokenizers_version": __import__("tokenizers").__version__,
    "add_special_tokens": False,
    "bos_probe": {"text": "Hello, world!", "ids": bos_probe.ids},
    "n_cases": len(cases),
    "cases": cases,
}
ref_path = os.path.join(HERE, "tokenizer_ref.json")
with open(ref_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("written", ref_path)
print("tokenizer.json md5:", tok_md5)
print("sample 'Hello, world!' ids (no special):", cases[0]["ids"])
print("sample with BOS:", bos_probe.ids)
roundtrip_fail = sum(1 for c in cases if c["decoded"] != c["text"])
print("HF self-roundtrip mismatches (decode(encode(x)) != x):", roundtrip_fail)
for c in cases:
    if c["decoded"] != c["text"]:
        print("  MISMATCH:", repr(c["text"])[:60], "->", repr(c["decoded"])[:60])
