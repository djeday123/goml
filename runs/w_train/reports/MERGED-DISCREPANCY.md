# MERGED-DISCREPANCY — какая версия merged_v1 является sealed (44.206ms / 398T)

**Вердикт:** ✅ **sealed merged = pre-052, md5 `2bf32ab7d4c5ecabb4ee2dbf1b5d4b33`.**
Источник-пруф: ledger-отчёт `runs/reports/041_post040_recon_dq.md` (стр. 14, 20) + sealed-архив
`runs/archive/040_sealed/fa_bwd_merged_v1.cu`. Тег `w0-seal-v1` закоммитил `a0b4b190…3012` — это
**отвергнутый экспериментальный кандидат стадии 053** (`.053_cand`), а НЕ production/sealed merged.
Доказательство многократно перекрыто (4 независимых улики, ниже).

Только чтение + git-археология. Ничего не копировалось/собиралось/коммитилось. Грязная рабочая merged не тронута.

---

## ARTIFACT HEADER

```
# бэкапы merged в libs/
-rw-r--r-- 25638 Jul 9 10:22  libs/fa_bwd_merged_v1.cu           (ca064452  = грязная рабочая, откат-в-процессе)
-rw-r--r-- 25638 Jul 9 07:45  libs/fa_bwd_merged_v1.cu.052_pre   (2bf32ab7  = pre-052 = SEALED)

# архивные merged (md5)
2bf32ab7…4b33  runs/archive/040_sealed/fa_bwd_merged_v1.cu           25638 B   <- SEALED (040)
2bf32ab7…4b33  runs/archive/052_pre/fa_bwd_merged_v1.cu             (главный на стадии 052)
2bf32ab7…4b33  runs/archive/053_pre/fa_bwd_merged_v1.cu             (главный на стадии 053)
2bf32ab7…4b33  runs/archive/054_pre/fa_bwd_merged_v1.cu             (главный на стадии 054)
2bf32ab7…4b33  runs/archive/055_pre/fa_bwd_merged_v1.cu             (главный на стадии 055)
a0b4b190…3012  runs/archive/053_pre/fa_bwd_merged_v1.cu.053_cand    31209 B   <- КАНДИДАТ 053 (в теге!)
ca064452…598b  runs/archive/055_pre/fa_bwd_merged_v1.cu.055_1b_cand           <- КАНДИДАТ 055 (=грязная рабочая)
# прочие исторические (не относятся к спору): 033_pre 35ba3d21, 033_sealed deb3a0e1,
#   040_pre 4283cadb, 052_cand 43952e36, 055_1a_cand ee6657d8
```

**Дата разбора (UTC):** 2026-07-09T10:51:04Z · goml @ `2567c2a` (тег `w0-seal-v1`).

---

## Три спорные версии (напоминание)
| md5 | что это | где живёт |
|---|---|---|
| `2bf32ab7…4b33` | **pre-052** | `libs/…052_pre` (бэкап) + `040_sealed/` + главный merged в 052/053/054/055_pre |
| `a0b4b190…3012` | **post-052** (в теге `w0-seal-v1`) | только как `053_pre/…053_cand` (кандидат) |
| `ca064452…598b` | грязная рабочая (не коммитить) | только как `055_pre/…055_1b_cand` (кандидат) |

---

## ШАГ 1 — АРХЕОЛОГИЯ SEALED-БЕНЧА (решающий)

### 1a. Число 44.206ms — это post-041
`runs/reports/041_post040_recon_dq.md`:
- стр. 288: `Медианы: D=0.342, merged=25.029, dk_new=10.397, dq_new=8.436, total=44.206`
- стр. 296: `Post-041 (merged + dq d7a11a3d): 44.206 ms in-chain`
- стр. 343: `033-c → 040 → 041: 46.82 → 44.483 → 44.206 ms E2E`

Переход 040→041 менял **только dq_new** (→ `d7a11a3d`); merged унаследован из 040. Это ledger-официальное число (cert-протокол); альтернативные 40.153/43.911 в 048 помечены «методика, вне леджера».

### 1b. Ledger ПРЯМО называет md5 sealed-merged
Тот же отчёт, «Sealed archive + prod swap + E2E», стр. 14 и 20 — **дословно**:
```
fa_bwd_merged_v1.cu  (md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33 = 040 sealed)
runs/archive/040_sealed/:
fa_bwd_merged_v1.cu  (md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33)
```
→ merged, участвовавший в бенче 44.206ms, зафиксирован ledger'ом как **`2bf32ab7` = 040 sealed = pre-052**.

### 1c. Sealed-архивы подтверждают на диске
- `runs/archive/040_sealed/fa_bwd_merged_v1.cu` → md5 `2bf32ab7` (merged взят отсюда).
- `runs/archive/041_dq_sealed/fa_bwd_dq_new.cu` → md5 `d7a11a3d` (на 041 менялся только dq; merged в этом архиве отсутствует — не менялся).
- Полный sealed-набор 44.206ms: **merged `2bf32ab7` (040)** + dk_new `a9f0ded8` (033) + dq_new `d7a11a3d` (041).

### 1d. Все пост-052 стадии держат pre-052 как production; a0b4b190 — только кандидат
В `052_pre / 053_pre / 054_pre / 055_pre` главный `fa_bwd_merged_v1.cu` = `2bf32ab7` во ВСЕХ.
`a0b4b190` присутствует ЕДИНСТВЕННЫЙ раз — как `053_pre/…053_cand` (кандидат, не принят).

### 1e. Бенч сам себя документирует — rollback к 040
`libs/bench_r2c_e2e.cu:69`:
```
{"kernel_merged_v1", (const void*)fa_bwd_merged_v1::kernel_merged_v1, 252},
      // 055 rollback: обе микро-пробы КРАСНЫЕ, prod = 040 sealed
```
→ после красных микро-проб 055 production откатан к **040 sealed**.

**Итог Шага 1: sealed merged = pre-052 `2bf32ab7`, пруф — ledger `041_post040_recon_dq.md` (стр.14,20) + `040_sealed/`. `a0b4b190` из тега = отвергнутый кандидат 053.**

---

## ШАГ 2 — DIFF pre-052 (`2bf32ab7`) vs post-052 (`a0b4b190`, тег)
`diff libs/…052_pre  runs/archive/053_pre/…053_cand` (обе версии есть на диске).

**Суть изменения 053 — перестройка SMEM-плана загрузки dO (dO half-prefetch / ping-pong):**
- pre-052: единый буфер `smdO` (16384 B = все 64 строки) в aliased-плане поверх `smQ`; SMEM ~46592 B; присутствует `smdS_T_stage` (5120 B).
- post-052 (053): `smdO` разбит на **три 8-КБ плеча** — `smdO_first_A`/`smdO_first_B` (ping-pong для строк 0..31) + `smdO_second` (фикс, строки 32..63); **первая половина dO префетчится в конце qt** (скрытие латентности cp.async, «обход ям 052.1/052.2»); сплит-ридеры Step D по warp, Step H по kb. SMEM 46592→49664 B; `smdS_T_stage` убран из явного layout. Размер файла 25638→31209 B.
- Occupancy 2 блока/SM в обеих (SMEM-limited).

Фактически: **опт пайплайна dO-префетча** (экспериментальная перестройка стейджинга), не откат и не смена алгоритма ядра. (Оценку «лучше/хуже» не даю — по ТЗ; по факту 053 не был принят, prod откатан к 040.)

---

## ШАГ 3 — ПЕРЕКРЁСТНАЯ СВЕРКА СИГНАТУР
Сигнатура `kernel_merged_v1` **побайтово идентична** в pre-052, post-052(тег) и грязной рабочей:
```
kernel_merged_v1(Q, K, V, dO_g, L, D, dS_nat_out, dS_T_out, dV,
                 int bh, int sl, int hd, int causal, int window, float scale)
```
Форвард-декларация и таблица вызова в `bench_r2c_e2e.cu` (стр. 47, 69) совпадают с обеими версиями.

→ 052/053 **сигнатуру НЕ меняли** — правки чисто внутренние (SMEM/prefetch). ABI-конфликта с бесспорными
dk_new/dq_new нет ни у одной версии. **Сигнатуры версии не различают** (нейтральный сигнал); разбор
решается Шагом 1, а не сигнатурами.

---

## ШАГ 4 — ВЕРДИКТ-МАТЕРИАЛ (данные для Vugar, не решение)

| вопрос | ответ |
|---|---|
| md5 merged в sealed-архиве/леджере | **`2bf32ab7` (pre-052)** — `041_post040_recon_dq.md:14,20` + `040_sealed/` |
| суть diff 052→053 | dO half-prefetch: единый smdO(16КБ) → 3×8КБ ping-pong+второй; SMEM 46592→49664; убран smdS_T_stage; +5571 B кода |
| сигнатуры | идентичны во всех трёх → не различают; ABI-конфликта нет |
| что в теге `w0-seal-v1` | `a0b4b190` = `053_cand` = **отвергнутый кандидат**, не production |
| грязная рабочая `ca064452` | `055_1b_cand` (красная микро-проба 055); размер 25638 B ≈ откат к 040-базе в процессе |

### ИТОГОВАЯ СТРОКА
**sealed merged = pre-052, md5 `2bf32ab7d4c5ecabb4ee2dbf1b5d4b33`.**
Воспроизводимый пруф ЕСТЬ: ledger `runs/reports/041_post040_recon_dq.md` (стр.14,20) с явной привязкой
md5↔«040 sealed», сам архив `runs/archive/040_sealed/fa_bwd_merged_v1.cu` (=`2bf32ab7`), и rollback-метка
в `bench_r2c_e2e.cu:69` («prod = 040 sealed»). Тег `w0-seal-v1` содержит НЕВЕРНЫЙ merged (`a0b4b190`,
кандидат 053) — это ошибка запечатывания.

---

## Рекомендация по разблокировке (для клонирования goml_fp8_train)
1. Каноничный sealed merged для копии = `2bf32ab7`, доступен байт-в-байт из двух мест:
   `libs/fa_bwd_merged_v1.cu.052_pre` и `runs/archive/040_sealed/fa_bwd_merged_v1.cu`.
2. Тег стоит перекатать: пере-запечатать `merged_v1` = `2bf32ab7` и пере-тегнуть (напр. `w0-seal-v1a`),
   иначе любой, кто линкуется на `w0-seal-v1`, возьмёт отвергнутый 053-кандидат.
3. Грязную рабочую `libs/fa_bwd_merged_v1.cu` (`ca064452`) НЕ трогал и не откатывал — это решение Vugar.

## Границы соблюдены
Не копировал, не создавал goml_fp8_train, не собирал, не коммитил, грязную рабочую merged не менял.
Только чтение + `md5sum` + `diff` + `grep`. Отчёт — единственный новый файл (untracked), в `runs/w_train/reports/`.
