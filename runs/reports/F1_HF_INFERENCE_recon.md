# F1 HF-INFERENCE — Фаза I: разведка + инфраструктура без ядер

Дата: 2026-08-03. Карточка: HF-INFERENCE (paper-first, двухфазная).
Сессия: F1-HF. GPU в этой фазе НЕ трогался ни разу (ни одного nvidia-smi
не потребовалось — вся работа CPU/диск, включая гейты).
Прогнозы и floor'ы записаны до прогонов: `runs/hf_inference/F1_predictions.md`.

## 0. Резюме

Все пять пунктов Фазы I закрыты, все гейты зелёные с первого прогона:

| Пункт | Деливерабл | Гейт | Статус |
|---|---|---|---|
| I.1 | таблица config → стек (ниже) | — | закрыт |
| I.2 | `hf/safetensors.go` + `hf/config.go` | 201 тензор, shapes/dtype/конечность/md5 | **PASS** |
| I.3 | `hf/tokenizer_llama.go` | 100/100 encode==HF, 100/100 decode==HF, BOS | **PASS** |
| I.4 | дизайн KV-cache (ниже, бумага) | — | закрыт |
| I.5 | `internal/abjexam/llama_hf_f64ref.go` | top-10 логитов == HF, worst rel 4.3e-7 ≤ 1e-3 | **PASS** |

Greedy-продолжение промпта "The capital of France is" → **"Paris"**
(прогноз подтверждён). Модель: TinyLlama-1.1B-Chat-v1.0.

## 1. Выбор модели

Критерии карточки: hd ∈ {64,128}, вес < 5GB, открытая лицензия.

**TinyLlama/TinyLlama-1.1B-Chat-v1.0**: hd = 2048/32 = **64** ✓ (проверено —
в карточке стоял «hd=?»); BF16-веса 2.2GB ✓; Apache-2.0 ✓; чистый
LlamaForCausalLM без единой экзотики (rope_scaling=null, bias=false).
Бонус: GQA 32/4 — ловушка (а) отрабатывается сразу, а не при первой Llama-2.

Артефакты: `runs/hf_inference/tinyllama/` (config.json, tokenizer.json,
tokenizer_config.json, generation_config.json — в git; model.safetensors
2200119864 байт, **md5 59e1605b3af5f1673eb8396251d6bc46** — не в git,
фиксация целостности здесь и в гейте I.2).

## 2. I.1 — таблица: config.json против нашего стека

| Поле | Значение | Наш аналог | Статус |
|---|---|---|---|
| architectures | LlamaForCausalLM | BattleA-стек (RMSNorm/RoPE/attention) | есть, кроме FFN (см. (е)) |
| attention_bias | false | у нас bias нет вообще | совпадает; true — guard-отказ загрузчика |
| bos/eos_token_id | 1 / 2 | `LlamaTokenizer.BosID/EosID` из added_tokens | есть (новый загрузчик, см. §4) |
| hidden_act | silu | SiLU есть (BattleA FFN) | совпадает |
| hidden_size | 2048 | `BattleACfg.D` | есть |
| initializer_range | 0.02 | не нужен (инференс) | n/a |
| intermediate_size | 5632 | `BattleACfg.FFN` | есть; но FFN-форма другая, см. (е) |
| max_position_embeddings | 2048 | S_max KV-cache (§5) | учтён |
| num_attention_heads | 32 | `BattleACfg.H` | есть |
| num_hidden_layers | 22 | `BattleACfg.L` | есть |
| num_key_value_heads | 4 | **нет — у нас все головы одинаковые** | ловушка (а), вердикт ниже |
| pretraining_tp | 1 | не влияет на инференс | n/a |
| rms_norm_eps | 1e-5 | `BattleACfg.Eps` = 1e-5 | **совпадает битово** |
| rope_scaling | null | у нас scaling нет | совпадает; != null — guard-отказ |
| rope_theta | 10000.0 | `BattleACfg.Base` = 10000.0 | совпадает; параметр проброшен (Llama-3 500000 пройдёт) |
| tie_word_embeddings | false | Embed и Wout раздельные | **совпадает**; true — guard-отказ (ловушка (в)) |
| torch_dtype | bfloat16 | BF16-компьюта нет | конверсия на загрузке, вердикт ниже |
| vocab_size | 32000 | `BattleACfg.V` = 32000 | совпадает |

### Ловушки — вердикты

**(а) GQA (32 Q-голов / 4 KV-голов, G=8).**
- CPU-F64 линейка: **индексация kv-головы `kvh = h/G`** — нулевая лишняя
  память, реализовано в `fwdLlamaHFF64`, подтверждено гейтом I.5.
- GPU Фаза II, prefill через FA (ядро ждёт равные BH у Q и KV): **repeat-kv
  транзиентно на слой**. Числа: развёрнутый KV одного слоя при S=2048 F32 =
  2 (K и V) × 32 (H полных голов после разворота) × 2048 (S) × 64 (hd) ×
  4 (байт F32) = 33 554 432 байт = **33.55 MB (32.0 MiB)** (буфер
  переиспользуется между слоями, ×22 НЕ умножается; исправлено ТЗ-02 —
  ранее стояло ошибочное 67.1 MB, завышение ×2 против собственной формулы).
  Хранение кэша всегда в H_kv=4: 88.0 MiB против 704.0 MiB
  развёрнутого при S=2048 — **экономия ×8**. Decode-шаг (S=1, без FA) GQA
  обслуживается адресацией, разворот не нужен вовсе.
- Нативная GQA в FA-ядрах — НЕ в этой карточке (осознанно).

**(б) RoPE.** theta=10000 == наш Base. Главный риск снят кодом и гейтом:
наша половинная раскладка пар P4-канона (`ropeF64`: пары (i, i+HD/2),
freq = theta^(-2i/HD)) — это **бит-в-бит конвенция HF rotate_half**
(q·cos + rotate_half(q)·sin, x1=первая половина, x2=вторая). Гейт I.5 —
эмпирическое подтверждение на 22 слоях (rel 1e-7 недостижим при неверной
раскладке). rope_scaling (linear/dynamic/yarn, Llama-3) — guard-отказ
в `LoadLlamaConfig`, отдельная будущая работа. Долг Фазы II: GPU-RoPE
gotorch не имеет posOffset (v6/cuda/api.go:252) — для декода с растущим
кэшем обязателен.

**(в) tie_word_embeddings=false** — совпадает с нашей структурой
(Embed [V,D] и lm_head [V,D] раздельные). true (Qwen-мелочь, Llama-3.2-1B) —
guard-отказ загрузчика, поддержка = одна строка (переиспользовать Embed),
отложена до модели, которая её потребует.

**(г) rms_norm_eps=1e-5** — битово равен нашему. Формула идентична:
inv_rms = 1/sqrt(mean(x²)+eps), gamma без beta. Примечание: HF считает
RMSNorm в F32 даже на half-весах — наша F64-линейка строже, GPU-F32 путь
Фазы II идентичен HF по точности класса.

**(д) vocab 32000 и спецтокены.** unk=0, `<s>`=1, `</s>`=2 (added_tokens),
байтовые токены `<0xNN>` = id 3+NN (проверено: эмодзи-кейс референса).
Раскладка нашего собственного BPE (байты 0-255, спец 256-259) несовместима —
но и не нужна: новый загрузчик читает ID прямо из tokenizer.json (§4).

**(е) SwiGLU.** gate[5632,2048], up[5632,2048], down[2048,5632];
формула down(silu(gate(x)) ⊙ up(x)). BattleA FFN — одноматричный SiLU-MLP
(«no SwiGLU для упрощения», battle_a_llm.go:58) — **не совпадает**.
Вердикт: CPU-F64 расширение написано (llama_hf_f64ref.go); GPU-путь
Фазы II = 3 обычных matmul + поэлементное silu·mul gotorch-путями,
FA не затрагивается. Проверка наличия elementwise mul в адаптере — долг
Фазы II (реестр §7).

**Дополнительно (вне списка ловушек).** HF-layout весов Linear —
[out,in] (y = x@Wᵀ): ложится на существующий примитив `matF64_ABT`
**без транспонирования при загрузке**; для GPU Фазы II gotorch имеет
`MatMulF32Ex` с trans-флагами (api.go:284) — тоже без физического
транспонирования. Layout BattleA [in,out] остаётся только в тренировочной
дуге, конфликтов нет.

## 3. I.2 — safetensors-парсер (`hf/safetensors.go`)

Формат: 8 байт LE uint64 (длина заголовка) + JSON + данные. Парсер:
pread по требованию (файл в память целиком не грузится), валидация на
открытии: известный dtype, prod(shape)·sizeof == data_offsets-диапазон,
границы, **непрерывное покрытие секции данных без дыр и перекрытий**.

**Вердикт BF16→F32:** конверсия `uint32(bf16)<<16` — **битово точная**
(BF16 = верхние 16 бит F32; расширение мантиссы нулями без потерь).
Точность: нулевая потеря против того, что видит HF при
torch_dtype=float32 (та же операция). Цена: ×2 память относительно
BF16-хранения (веса TinyLlama: 2.2GB BF16 → 4.4GB F32 на GPU Фазы II —
против 96GB карты несущественно). BF16-компьют в наших ядрах отсутствует
(cuBLAS-обёртка знает CUDA_R_16BF — опция будущих карточек, не этой).
F16-путь written+unit-tested впрок (денормалы/Inf/NaN, TestF16ToF32).

**Гейт I.2 PASS** (`raw_f1hf/i2_i3_gates.txt`): против ожиданий,
выведенных из config.json (у одно-файловой модели index.json нет —
ожидания строже: полный список имён+форм LlamaForCausalLM):
201/201 тензоров, все shapes/dtype совпали, лишних нет, все 1 100 048 384
элементов загружены и конечны, md5 файла совпал с зафиксированным при
скачивании: `59e1605b3af5f1673eb8396251d6bc46`.

## 4. I.3 — токенизатор: вердикт «нужен новый загрузчик»

Наш BPE (`tokenizer/bpe.go`) — byte-level, своя ID-раскладка (байты 0-255,
спец 256-259), без pre-tokenizer/normalizer, свой формат .merges.
TinyLlama — sentencepiece-BPE в HF tokenizer.json: normalizer
Prepend "▁" + Replace " "→"▁", pre_tokenizer=null, byte_fallback=true,
merges ранговые, ID-раскладка unk=0/bos=1/eos=2/байты 3-258.
**Совместимости нет ни по формату, ни по раскладке, ни по алгоритму —
маппер невозможен, написан новый загрузчик** `hf/tokenizer_llama.go`
(читает tokenizer.json напрямую; старый BPE не тронут — он живёт
в тренировочной дуге).

Алгоритм: нормализация (Prepend только для непустой строки — проверено
референсом), ранговый BPE по всей строке (слияние всех вхождений пары
минимального ранга слева направо — эквивалент heap-алгоритма HF: ранг пары
со свежесозданным токеном строго больше ранга его создания), byte_fallback
для символов вне словаря; декод: байтовые токены → сырые байты, "▁"→" ",
Strip одного ведущего пробела.

**Гейт I.3 PASS** (`raw_f1hf/i2_i3_gates.txt`): 100/100 encode ==
HF-референсу, 100/100 decode == HF-референсу, BOS-проба OK. Референс снят
один раз (`runs/hf_inference/snap_tokenizer_ref.py`, tokenizers 0.21.4)
и зафиксирован в `runs/hf_inference/tokenizer_ref.json` (100 строк:
юникод/CJK/RTL, эмодзи+ZWJ, код Go/Python/SQL/regex, пробельные края,
типографика, комбинирующие знаки; md5 tokenizer.json внутри референса,
гейт сверяет — рассинхрон файлов ловится).

## 5. I.4 — дизайн KV-cache (бумага, код Фазы II)

**Раскладка:** per-layer contiguous `[L][2][H_kv][S_max][hd]` F32,
K и V — половины одного буфера слоя. Хранение в H_kv-головах (§2а).

**Память (TinyLlama, F32, один поток генерации):**

| S_max | элементов | байт | итого |
|---|---|---|---|
| 2048 | 2·4·64·2048·22 | 92 274 688 | **88.0 MiB** |
| 8192 | 2·4·64·8192·22 | 369 098 752 | **352.0 MiB** |

(F16/BF16-кэш — вдвое меньше; но кэш Фазы II — F32, как весь боевой
путь: requireF32 в адаптере.) Оговорка: S_max=8192 > max_position_embeddings
= 2048 требует rope-scaling — вне карточки; число дано по ТЗ для масштаба.

**Политика роста: аллокация S_max сразу, без роста.** Обоснование:
0.09-0.35 GB на фоне 4.4GB весов — не ресурс; чанковый рост (256-позиций)
дал бы реаллокации+копии и фрагментацию пула ради экономии, которой нет.
OOM-класс проблем при таких объёмах исключён.

**Где живёт:** gotorch-буферы через `backend.Backend.Alloc`
(адаптер → gotorch.Alloc → cuMemAlloc); владелец — структура `KVCache`
в goml (per-model, Free при закрытии); образец аренды — пул
`backend/cuda/pool.go`. Заполнение prefill: D2D-копия K/V слоя после
проекций; декод: дозапись позиции t (CopyD2D 2·H_kv·hd·4B = 2KB на слой).

**Интерфейс шага генерации (эскиз):**
```
type KVCache struct { layers [][]DeviceBuf; sMax, tCur int }
Append(l int, k, v DeviceView)      // позиция tCur, [H_kv,hd]
K(l int) DeviceView                  // [H_kv, tCur+1, hd]
V(l int) DeviceView
```
Декод-шаг (S=1, БЕЗ FA — подтверждение вердикта карточки): q [H,1,hd];
scores = q·Kᵀ — `MatMulStridedBatchedF32` по H с KV-stride, повторяющим
головы группами G (stride=0 внутри группы — проверить поддержку, иначе
цикл по H одиночными GEMV — при S≤2048 дёшево); softmax
`SoftmaxF32` (rows=H, cols=t+1); out = P·V тем же батч-путём; FA в декоде
не нужен — подтверждено: FA-ядро заточено под train-формы (LSE-выход,
fa_forward.go:161), выгоды при S_q=1 нет.

**Prefill:** FA-fwd causal по слоям с транзиентным repeat-kv (§2а) —
после сертификации II.1.

## 6. I.5 — CPU-F64 смоук против HF-референса

Арбитр прежде скорости: `fwdLlamaHFF64` (internal/abjexam/llama_hf_f64ref.go) —
расширение fwdBattleAF64-пути: те же примитивы `matF64_ABT`/`rmsNormF64`/
`ropeF64`/`permuteSHD_HSD`, поверх них GQA-индексация, SwiGLU и causal-маска
(префиксный softmax; **ядра не тронуты** — маска чисто CPU-шная).
Последовательный F64 без map/горутин — бит-детерминизм по построению.

Референс снят один раз (`snap_logits_ref.py`: transformers 5.14.1,
torch 2.8.0 CPU, dtype=float32) → `runs/hf_inference/logits_ref.json`.
Веса у обоих путей бит-эквивалентны (BF16→F32/F64 точны) — расхождение
только компьют: F32 (HF) против F64 (наш).

**Гейт I.5 PASS** (floor'ы записаны до прогона, raw: `raw_f1hf/i5_smoke.txt`):

```
промпт: "The capital of France is" -> ids [1 450 7483 310 3444 338]  (== референсу)
rank id       token            ours(F64)      HF ref(F32)    rel
0    3681     "Paris"          13.38848375    13.38848686    2.322e-07
1    5982     "located"        12.26868441    12.26868439    1.753e-09
...
9    263      "a"              9.62549408     9.62549591     1.906e-07
greedy next token: "Paris"
worst rel (top-10 + first8) = 4.279e-07, floor = 1e-03
```

Top-10: множество и порядок совпали 10/10, argmax — "Paris".

**Разбор прогнозов (калибровка):** floor 1e-3 — запас 3.4 декады;
прогноз класса ~1e-4 промахнулся на ~2.4 декады в консервативную сторону.
Причина: оценка sqrt(D)·2⁻²⁴ на элемент честна для наивной F32-суммы, но
torch GEMM аккумулирует блочно (FMA + иерархические частичные суммы), что
режет эффективную ошибку на 1-2 декады; амплификация по 22 слоям тоже
переоценена (RMSNorm на каждом слое ренормализует тракт). Скорость: веса
3.5s, forward 6 токенов 3.7s — против прогноза 20-120s (Go-компилятор на
плотных F64-циклах быстрее заложенного консерватива). Оба промаха
безопасной стороны; флор-класс 1e-3 для будущих BF16-моделей оставляем —
он покрывает и модели с менее аккуратным референсом (GPU/F16).

## 7. Реестр долгов и рисков → Фаза II

| # | Долг | Класс |
|---|---|---|
| F2-1 | causal=1 НЕ сертифицирован в FA-ядрах (HIGH из реестра хроники) — II.1, только ПОВЕРХ квант-конвенции decoded O(1) (триггер: зелёный П.6 A-LLM-5) | HIGH |
| F2-2 | GPU-RoPE без posOffset (gotorch api.go:252) — декод с растущим кэшем невозможен без правки | HIGH |
| F2-3 | elementwise silu·mul для SwiGLU в адаптере — проверить наличие/добавить | MED |
| F2-4 | MatMulStridedBatchedF32 со stride=0 (GQA-декод без разворота) — проверить поддержку cuBLAS-обёрткой | MED |
| F2-5 | tie_word_embeddings=true / rope_scaling / attention_bias — guard-отказы, поддержка по мере надобности | LOW |
| F2-6 | chat-template (tokenizer_config) — вне границ карточки (single-stream генерация без чат-обвязки) | LOW |
| F2-7 | tokenizer_ref снят на tokenizers 0.21.4, venv с тех пор обновлён до 0.22.2 (для logits-референса) — при пересъёмке токен-референса зафиксировать версию заново | NOTE |

Фаза II НЕ стартует до зелёного П.6 сессии A-LLM-5 тренировочной дуги
(пересертификация FA на decoded O(1)); детальное ТЗ II — после ревью
этой бумаги.

## 8. Артефакты (Б-0)

- Код: `hf/config.go`, `hf/safetensors.go`, `hf/tokenizer_llama.go`,
  `hf/safetensors_test.go`, `hf/tokenizer_llama_test.go`,
  `internal/abjexam/llama_hf_f64ref.go`,
  `internal/abjexam/llama_hf_f64_smoke_test.go`.
- Данные/референсы: `runs/hf_inference/` (config/tokenizer TinyLlama,
  tokenizer_ref.json, logits_ref.json, snap-скрипты, F1_predictions.md).
- Raw: `runs/reports/raw_f1hf/` (i2_i3_gates.txt, i5_smoke.txt, ref_env.txt).
- Веса не в git: md5 `59e1605b3af5f1673eb8396251d6bc46`, источник
  huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0 (Apache-2.0).
- commit-hash — в параграфе хроники (вставляется при коммите).

СТОП по I.6: ревью бумаги до старта Фазы II.
