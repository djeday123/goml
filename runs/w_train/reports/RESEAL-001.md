# RESEAL-001 — пере-seal тега w0-seal-v1 (исправление merged_v1)

**Итог:** ✅ **ВЫПОЛНЕНО и верифицировано.** Тег `w0-seal-v1` и `origin/main` переставлены на исправленный
seal-коммит `4732a380`, где `merged_v1` = **`2bf32ab7`** (040 sealed, давший 44.206ms/398T) вместо битого
`a0b4b190` (отвергнутый кандидат 053). dk_new / dq_new не тронуты. Локаль == origin. Все md5-гейты PASS.

Соблюдены оба предохранителя: (1) md5 сверен **ровно перед `git add`** (не «раньше в сессии»); (2) merged
взят из **чистого архива `040_sealed/`**, а не из грязной opt-рабочей `ca064452`.

---

## ARTIFACT HEADER

```
# новый seal
HEAD                = 4732a380a817e63d8592532f87e1daf26ec9d2f8   (amend старого 2567c2a)
tag w0-seal-v1      -> 4732a380 (annotated obj fe4dede9…)
origin/main         = 4732a380   (force-with-lease OK)
origin w0-seal-v1^{}= 4732a380   (force OK)

# источник merged (byte-exact, чистый архив)
-rw-r--r-- 25638 Jul 8 16:42  runs/archive/040_sealed/fa_bwd_merged_v1.cu   (2bf32ab7…4b33)
-rw-r--r-- 25638 Jul 9 10:56  libs/fa_bwd_merged_v1.cu                      (2bf32ab7…4b33, после cp+commit)

# бэкап opt-версии для восстановления opt-ветки
ca064452…598b  runs/archive/055_pre/fa_bwd_merged_v1.cu.055_1b_cand
```

**Дата (UTC):** 2026-07-09T10:58:05Z · репо goml.

---

## PRE-состояние (Шаг 0)
| проверка | ожидалось | факт | verdict |
|---|---|---|---|
| `git rev-parse HEAD` | 2567c2a (битый seal) | `2567c2a` | ✅ |
| dirty tracked | только DISPATCHER.md + libs/fa_bwd_merged_v1.cu | ровно они (`M`), staged нет; остальное `??` R&D | ✅ |
| `040_sealed/merged` md5 | 2bf32ab7…4b33 | `2bf32ab7…4b33` | ✅ |
| merged **в теге** (битый) | a0b4b190…3012 | `a0b4b190…3012` | ✅ подтверждён брак |

Неожиданных staged/чужих изменений нет → STOP не сработал.

## Что заменено и откуда (Шаг 1)
`cp runs/archive/040_sealed/fa_bwd_merged_v1.cu → libs/fa_bwd_merged_v1.cu` (осознанно перезаписал
opt-дрейф `ca064452`). **md5 РОВНО ПЕРЕД add** = `2bf32ab7d4c5ecabb4ee2dbf1b5d4b33` ✅ PASS (новое правило).

## Стейджинг и amend (Шаги 2–3)
- `git add libs/fa_bwd_merged_v1.cu` — staged РОВНО merged (`M ` в кол.1); `DISPATCHER.md` остался неstaged (` M`). НЕ `-a`.
- `git commit --amend --no-edit` → новый HEAD `4732a380`.
- Контроль: `git diff 2567c2a HEAD --stat` = **1 файл, только `libs/fa_bwd_merged_v1.cu`** (41+/123−). Больше ничего не тронуто.
- merged в новом коммите: `2bf32ab7…4b33` ✅; dk_new `a9f0ded8` ✅; dq_new `d7a11a3d` ✅; DISPATCHER.md НЕ в коммите ✅.

## Тег переставлен (Шаг 4)
`git tag -d w0-seal-v1` (был obj 6b9c0c1) → `git tag -a w0-seal-v1` на `4732a380`.
Сообщение: *"W seal v1 (fixed merged): merged 2bf32ab7 (040 sealed) / dk_new a9f0 / dq_new d7a1 byte-intact; D in fa_bwd_dk.cu; L-fwd _v121r_train; 44.206ms/398T"*.
`w0-seal-v1^{}` = `4732a380` == HEAD ✅.

## Force-push (Шаг 5)
- `git push origin main --force-with-lease` → `2567c2a...4732a38 main (forced update)` ✅ (lease прошёл — origin/main никто не двигал).
- `git push origin w0-seal-v1 --force` → `6b9c0c1...fe4dede w0-seal-v1 (forced update)` ✅.

## ФИНАЛЬНАЯ ВЕРИФИКАЦИЯ (Шаг 6) — всё PASS
```
HEAD                    = 4732a380
w0-seal-v1^{}           = 4732a380   == HEAD                 ✅
origin refs/heads/main  = 4732a380   == локаль               ✅
origin w0-seal-v1^{}    = 4732a380   == локаль               ✅
```
Тройная md5 backward-ядер **из тега**:
| ядро | ожидалось | из тега | verdict |
|---|---|---|---|
| fa_bwd_merged_v1.cu | 2bf32ab7…4b33 | `2bf32ab7…4b33` | ✅ |
| fa_bwd_dk_new.cu | a9f0ded8…f458 | `a9f0ded8…f458` | ✅ |
| fa_bwd_dq_new.cu | d7a11a3d…8754 | `d7a11a3d…8754` | ✅ |

---

## Заметка opt-ветке
Перезапись рабочего `libs/fa_bwd_merged_v1.cu` уничтожила opt-дрейф `ca064452a23fa40757be614badf0598b`.
Он **восстановим byte-exact** из `runs/archive/055_pre/fa_bwd_merged_v1.cu.055_1b_cand` (md5 сверен = `ca064452`).
Это был `055_1b_cand` (красная микро-проба 055), не production.

## Границы соблюдены
Менял ТОЛЬКО merged. `DISPATCHER.md` не тронут (остался неstaged). Не `-a`. Amend того же seal-коммита
(не плодил новый). Сборки/бенча/обвязки не было. `--force-with-lease` для main (safety), `--force` для тега.

## Новый якорь seal
**`w0-seal-v1` = commit `4732a380a817e63d8592532f87e1daf26ec9d2f8`** — теперь корректный. Клонирование
goml_fp8_train можно разблокировать: линковка на `w0-seal-v1` даёт правильный merged `2bf32ab7`.
