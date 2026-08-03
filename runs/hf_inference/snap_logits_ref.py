# Снятие HF-референса логитов TinyLlama (I.5, карточка HF-INFERENCE).
# CPU, torch_dtype=float32 (BF16-веса точно расширяются до F32 — та же
# конверсия, что в нашем загрузчике). Референс снимается ОДИН РАЗ.
# Запуск: venv/bin/python snap_logits_ref.py
import json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "tinyllama")
PROMPT = "The capital of France is"

tok = AutoTokenizer.from_pretrained(DIR)
model = AutoModelForCausalLM.from_pretrained(DIR, torch_dtype=torch.float32, device_map="cpu")
model.eval()

enc = tok(PROMPT, return_tensors="pt")
with torch.no_grad():
    out = model(**enc)
logits = out.logits[0, -1]  # [V], float32

top = torch.topk(logits, 10)
ref = {
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "prompt": PROMPT,
    "input_ids": enc["input_ids"][0].tolist(),
    "torch_dtype": "float32",
    "transformers_version": __import__("transformers").__version__,
    "top10_ids": top.indices.tolist(),
    "top10_logits": [float(v) for v in top.values],
    "top10_tokens": [tok.decode([i]) for i in top.indices.tolist()],
    "argmax_id": int(torch.argmax(logits)),
    "logits_first8": [float(v) for v in logits[:8]],
}
path = os.path.join(HERE, "logits_ref.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(ref, f, ensure_ascii=False, indent=1)
print("written", path)
print("input_ids:", ref["input_ids"])
print("top10:", list(zip(ref["top10_ids"], ref["top10_tokens"], ref["top10_logits"])))
