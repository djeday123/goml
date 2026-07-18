# 065-мини — GitHub Release заметка на теге v0.2.0

**Chain**: 064_release_final.md `c099586c01ad07d6c83f3d732a77de3c` → **065 `<self>`**

**Правила ТЗ**: тег v0.2.0 НЕ пересоздаётся, код НЕ трогается. Только оформление GitHub Release поверх существующего тега `506b69f` / v0.2.0.

---

## Артефакт-хедер

```
release_v0.2.0/  (unchanged — тег v0.2.0 указывает сюда)
  Git HEAD:     506b69f v0.2.0: initial public release
  Tag:          v0.2.0 (annotated, локальный, ждёт push)
  Remotes:      none configured — Vugar ставит origin

release notes (draft):
  runs/reports/065_release_body.md  md5 f1ca49c7ed811ce847c2892bbc5ec9a3
```

Гейт-тишина: N/A (замеров в 065 нет; чистая работа с текстом).
Правки production ядер: **0**.

---

## §1 Grep-проверка на запрещённые фразы

**Правило ТЗ**: `x2.8`/`2.8x` — нельзя; голые `H100`/`H200` сравнения — нельзя. Разрешено `H100` с оговоркой конвенций и per-dollar; A100 same-convention ×1.5–1.75; H100 — приглашение в «пустой стул».

### 1.a Опубликованное дерево `release_v0.2.0/`

```
$ grep -rniE "x2\.8|2\.8x|2\.8\s*x" release_v0.2.0/
(нет совпадений)

$ grep -rniE "H200" release_v0.2.0/
(нет совпадений)

$ grep -rniE "H100" release_v0.2.0/
(нет совпадений)
```

**Опубликованное дерево — чистое.** Тег v0.2.0 не содержит запрещённых фраз.

Единственное упоминание сравнения — README.md строка 25:

> "FA2 A100 BF16 backward at similar canonical shapes is reported around ~175 TFLOPS in the 10N²d convention (literature estimate, not from this repo's runs)."

С disclaimer конвенции и «literature estimate» — соответствует исправленной карте дуэлей (A100 same-convention).

### 1.b Черновик заметки `065_release_body.md`

```
$ grep -niE "x2\.8|2\.8x|H200" runs/reports/065_release_body.md
(нет совпадений после очистки disavowal-строки)

$ grep -niE "H100" runs/reports/065_release_body.md
43:| **FA3 on H100 (BF16/FP8 bwd)** | H100 has more raw silicon and lower host-side latency;
    FA3's bwd is faster in **absolute** ms than ours. Where we win is **per-dollar**:
    RTX PRO 6000 Blackwell is roughly ×2 cheaper per unit of this workload at time of writing.
44:| **FA3-style bwd on H100 at this exact shape, bwd, causal** | **empty chair at the table** |
    If you have H100 access, please run `benchmark_flash_attention.py`...
```

**H100 упомянут строго по карте дуэлей**: (a) абсолютная скорость FA3 быстрее, (b) мы выигрываем ×2 per-dollar, (c) прямой матч на нашей форме — пустой стул, приглашение владельцам H100.

**Grep-лог чистый** ✓ (запрещённых `x2.8` / `2.8x` / голых `H200` — 0 совпадений и в дереве, и в черновике).

Первая итерация черновика содержала disavowal-строку «No `x2.8` figures, no bare H200 comparisons» — сами строки-запреты содержали запрещённые токены. Убрано: переформулировано в «Bare cross-silicon TFLOPS comparisons at mismatched conventions don't survive a same-convention same-shape audit, so we won't lead with one.» — тот же смысл, чисто.

---

## §2 Черновик заметки — карта секций

Полный текст: `runs/reports/065_release_body.md` md5 `f1ca49c7ed811ce847c2892bbc5ec9a3`.

Секции:
1. **Заголовок-подпись** — «world-first FP8 backward at this shape+arch we could find; open an issue если знаете иначе».
2. **Certified numbers** — таблица nc / causal обе дорожки с явными подписями конвенций (16N²d proj + 10N²d fused; cert 062/063 + clean-clone 064). Включена оговорка про «495 T eff fused» — заявлено как comparison anchor, не как throughput claim (наивная делёжка полной матрицы на causal-wall).
3. **Reproducibility** — одна команда `git clone <REPO_URL> && cd fa-blackwell-fp8 && git checkout v0.2.0 && ./verify.sh`; описание 5 стадий verify; source md5 4 sealed sources (post-SPDX).
4. **Where this sits vs the public field** — карта дуэлей с оговорками:
   - **A100 FA2 BF16 bwd**: same-convention **×1.5** (10N²d) / **×1.75** (16N²d proj). Разное железо, одна конвенция, одна форма.
   - **H100 FA3 BF16/FP8 bwd**: FA3 быстрее в абсолюте, мы **×2 per-dollar**. Разное железо, разный ценник.
   - **H100 direct match на нашей форме**: **empty chair at the table** — приглашение прогнать `benchmark_flash_attention.py` на `bh=128 sl=8192 hd=128 bwd causal` и положить число рядом.
5. **Known limitations** — hd=128 only; sm_120a only; FP8 e4m3 + FP16 O/dO + FP32 acc; causal-wall честный wall, не «full-matrix/causal»; FP64 floors перечислены.
6. **License** — Apache-2.0 + SPDX headers.
7. **Reports** — 062 / 063 / 063-r / 064 краткий указатель.

Все требуемые пункты ТЗ 065 покрыты:
- Таблица обеих дорожек с подписями конвенций ✓
- Строка воспроизводимости `clone && make && verify` (у нас `verify.sh`) ✓
- Карта дуэлей с оговорками (A100 ×1.5–1.75 same-convention; H100 быстрее абсолютно, мы ×2 per-dollar; пустой стул) ✓
- Known limitations (hd=128, sm_120a) ✓

Заголовок как указано в ТЗ:
> **"FP8 FlashAttention backward, world-first -- 42.35 ms nc (415 proj / 260 fused) | 22.21 ms causal (495 eff fused), 30-run cert, sm_120a"**

---

## §3 URL релиза + команда для Vugar

**Статус**: git remote в текущей среде **НЕ настроен** (см. §4 в 064). Push тега не выполнен, GitHub Release создать невозможно до push. Заготовка команды готова.

**Команды для Vugar** (в порядке выполнения):

```bash
# 1. Настроить remote (URL по выбору Vugar — GitHub/GitLab/etc)
cd /data/lib/podman-data/projects/goml/release_v0.2.0
git remote add origin git@github.com:<USER>/fa-blackwell-fp8.git

# 2. Push ветки и тега
git push -u origin main
git push origin v0.2.0

# 3. Создать GitHub Release на существующем теге (тег НЕ пересоздаётся)
gh release create v0.2.0 \
  --title "FP8 FlashAttention backward, world-first -- 42.35 ms nc (415 proj / 260 fused) | 22.21 ms causal (495 eff fused), 30-run cert, sm_120a" \
  --notes-file /data/lib/podman-data/projects/goml/runs/reports/065_release_body.md

# Альтернатива через UI: скопировать содержимое 065_release_body.md
# в поле "Describe this release" на странице github.com/<USER>/fa-blackwell-fp8/releases/new
# → выбрать тег v0.2.0 → вставить заголовок → Publish
```

**URL релиза после publish**:
```
https://github.com/<USER>/fa-blackwell-fp8/releases/tag/v0.2.0
```

**Пост-publish сверка** (Vugar):
```bash
# 1) тег указывает на 506b69f
gh api repos/<USER>/fa-blackwell-fp8/git/refs/tags/v0.2.0 --jq .object.sha
# ожидается: 506b69f... или совпадающий SHA после push

# 2) заметка релиза не содержит запрещённых фраз
gh release view v0.2.0 --json body -q .body | grep -niE "x2\.8|2\.8x|H200" || echo "clean"
# ожидается: clean

# 3) md5 файлов в опубликованном теге == sealed post-SPDX
gh api repos/<USER>/fa-blackwell-fp8/contents/src/fa_bwd_dk_new.cu?ref=v0.2.0 \
    --jq .content | base64 -d | md5sum
# ожидается: eb492e0729ef643280591b8c8dd8a29d
```

---

## §4 Финальная строка

**URL релиза**: (после push от Vugar) `https://github.com/<USER>/fa-blackwell-fp8/releases/tag/v0.2.0`

**md5 черновика заметки**: `f1ca49c7ed811ce847c2892bbc5ec9a3`  (`runs/reports/065_release_body.md`)

**Grep-лог п.1**: **ЧИСТЫЙ** — 0 совпадений на `x2.8`/`2.8x`/`H200` в опубликованном дереве; H100 упомянут строго по карте дуэлей (per-dollar + empty chair).

**Правки production ядер в 065: 0.**
**Тег v0.2.0 НЕ пересоздан.**

### Chain md5

- 062 `b7044db70019e8fa7dea260f9f235b6c`
- 063 `b7c82475ed49ac4821c7346f99a38fb1`
- 063-r `1dce5e445e5b47152c2894ffc7947b30`
- 064 `c099586c01ad07d6c83f3d732a77de3c`
- **065 `cc5c2a7f96aeed162ddf28609703009a`**  (release_body.md `f1ca49c7ed811ce847c2892bbc5ec9a3`)
