# CLONE-GATE0-BLOCKED — чистый клон goml_fp8_train — СТОП на Gate 0 (обновлено)

**Статус:** ⛔ **СТОП на Gate 0.** Путь Vugar добавил — и он подхватился инструментами **Write/Read/Edit**
(канонический путь теперь файлово-записываемый). НО **Bash-песочница его НЕ видит** (её allowlist
фиксируется на старте сессии и не изменился). Клон принципиально требует Bash (`cp -r`, `mkdir`, `git`,
`go build`, `md5sum`), поэтому из ЭТОЙ сессии выполнить нельзя. **Нужен свежий запуск агента** с
`goml_fp8_train` в рабочих каталогах с самого старта. Ничего не скопировано, дерево не создано.

---

## Gate 0 — результаты (повтор после «добавил»)

### Тег-сверка — PASS ✅
| проверка | ожидалось | факт |
|---|---|---|
| `git rev-parse HEAD` | 4732a380 | `4732a380a817e63d8592532f87e1daf26ec9d2f8` ✅ |
| `git rev-parse w0-seal-v1^{}` | == HEAD | `4732a380…` ✅ |
| merged из тега (md5) | 2bf32ab7…4b33 | `2bf32ab7d4c5ecabb4ee2dbf1b5d4b33` ✅ |

### Write-доступ — РАСХОЖДЕНИЕ между инструментами ⚠️
| канал | операция | результат |
|---|---|---|
| **Write tool** | создать `goml_fp8_train/.wtest` | ✅ **OK** (путь подхвачен) |
| Bash | `mkdir -p goml_fp8_train/kernels/backward` | ⛔ blocked (allowlist без goml_fp8_train) |
| Bash | `cp … goml_fp8_train/.cptest` | ⛔ blocked |
| Bash | `rm goml_fp8_train/.wtest` | ⛔ blocked |
| Bash | `test -d goml_fp8_train` | ✅ читается (стат разрешён) |

**Вывод:** добавление рабочего каталога применилось к файловым инструментам агента (Write/Read/Edit), но
**Bash-sandbox allowlist остался прежним** — он привязан к моменту старта сессии. В блок-сообщениях Bash
по-прежнему ровно 5 каталогов + `/root/repos`, без `goml_fp8_train`.

---

## Что нужно от Vugar
Клон нельзя собрать одними Write/Read (пакет `fa_sm120` содержит бинари `.so/.a`; Go-фреймворк — сотни
`.go`; плюс `git init` и `go build` — это только Bash). Поэтому:

**Перезапустить агента заново** — так, чтобы `/data/lib/podman-data/projects/goml_fp8_train` был в рабочих
каталогах **Bash с самого старта** (напр. `--add-dir /data/lib/podman-data/projects/goml_fp8_train` при
запуске новой сессии). В текущей живой сессии Bash-allowlist не перечитывается.

После перезапуска Фазы 1–6 отработаю за один проход (skeleton → sealed-копия из `040_sealed`+`libs` с
md5-сверкой → forward-пакет+L → Go-фреймворк → MANIFEST/README/BUILD_FLAGS → `go build` → `git init`).

## Мусор от проверки (убрать не смог — Bash rm заблокирован)
- `goml_fp8_train/` — создан Write-инструментом при probe.
- `goml_fp8_train/.wtest` — probe-файл (содержимое `probe`). Свежая сессия удалит в Фазе 1, либо `rm` вручную.

## Границы соблюдены
`goml/` не менял (только чтение + этот отчёт). Ничего не копировал, дерево клона не строил, в `/root/repos`
не сваливал, opt-версию merged в `libs/` не читал.

**Дата (UTC):** 2026-07-10 (Gate 0 повтор).
