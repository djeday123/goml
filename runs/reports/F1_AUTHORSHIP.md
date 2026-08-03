# F1_AUTHORSHIP — маркер авторства артефактов Фазы I карточки HF-INFERENCE

Дата: 2026-08-03. Сессия: F1-HF (ТЗ-01 П.2.2).

## Инцидент (одной фразой)

Session-close коммит `fffe6ea` параллельной дуги A-LLM-5 широким `git add`
в общем working tree захватил застейдженные файлы сессии F1-HF под своим
сообщением — содержимое цело, авторство фиксируется этим файлом
(правило [[feedback-one-clone-one-session]] введено ТЗ-01 как следствие).

## Чужой коммит-носитель

`fffe6ead7ec0290190ab899225efc2ef41305e75` (A-LLM-5 session close,
2026-08-03 11:25:15 UTC). Параграф сессии F1-HF в хронике: `3d4cb94`.

## Пути сессии F1-HF внутри fffe6ea (21 файл)

Код:
- hf/config.go
- hf/safetensors.go
- hf/safetensors_test.go
- hf/tokenizer_llama.go
- hf/tokenizer_llama_test.go
- internal/abjexam/llama_hf_f64ref.go
- internal/abjexam/llama_hf_f64_smoke_test.go

Данные и референсы:
- runs/hf_inference/.gitignore
- runs/hf_inference/F1_predictions.md
- runs/hf_inference/logits_ref.json
- runs/hf_inference/snap_logits_ref.py
- runs/hf_inference/snap_tokenizer_ref.py
- runs/hf_inference/tokenizer_ref.json
- runs/hf_inference/tinyllama/config.json
- runs/hf_inference/tinyllama/generation_config.json
- runs/hf_inference/tinyllama/tokenizer.json
- runs/hf_inference/tinyllama/tokenizer_config.json

Отчётность:
- runs/reports/F1_HF_INFERENCE_recon.md
- runs/reports/raw_f1hf/i2_i3_gates.txt
- runs/reports/raw_f1hf/i5_smoke.txt
- runs/reports/raw_f1hf/ref_env.txt

Все остальные пути коммита fffe6ea (backend/cuda/fa_*_test.go,
runs/reports/A_LLM5_quant_contract.md, HANDOFF_ref_dewrapper.md,
PROJECT_CHRONICLE.md, runs/reports/0xx-071b_* замер-логи) — работа
дуги A-LLM-5 и её широкого add, к сессии F1-HF отношения не имеют.

## Подтверждение целостности

`git diff origin/main -- hf/ runs/hf_inference/ runs/reports/` в ветке
hf-inference (до добавления этого файла) — вывод пуст, exit 0:
содержимое всех путей Фазы I побайтово равно origin/main.
