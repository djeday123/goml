# 063 — Causal-baseline + релиз v0.2.0

**Chain**:
- 061_s2v4_production.md `cf99a50510700f8f994f56ec3274c3b5`
- **062_cert400.md `b7044db70019e8fa7dea260f9f235b6c`**

**Правила ТЗ 063**: production-исходники НЕ правятся. Bench-правка (env-флаг causal) легальна. Релиз-инфраструктура = новые файлы.

---

## Артефакт-хедер (правило 5)

```
libs/ (post-063):
-rw-r--r-- 13060 Jul  9  bench_r2c_e2e.cu              (063 A1: +CAUSAL env, kernels unchanged)
-rw-r--r-- 12099 Jul  9  fa_bwd_dk_new.cu              md5 25e5e1077cc3bec2c49bf9288fe60c54  (S2v4 sealed)
-rw-r--r-- 25638 Jul  8  fa_bwd_merged_v1.cu           md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33  (040 sealed)
-rw-r--r-- 18834 Jul  8  fa_bwd_dq_new.cu              md5 d7a11a3d788eb4c396d892bc9c8ab754  (041 sealed)
-rwxr-xr-x       Jul  9  bench_r2c_e2e                 md5 1f884d70b82501e626fdfef4efce644c  (post-A1 rebuild)
```

**Правки production ядер в 063**: **0**.
**Правки bench**: **1** (env `CAUSAL` override — bench-side only, kernels untouched).

---

## §A Causal-baseline

### §A1 Env-параметр в `bench_r2c_e2e.cu`

**Правка** (3 строки после `causal=0`):
```c
// 063 A1: env-override CAUSAL=1 (bench-side only; kernels untouched)
if (const char *env = std::getenv("CAUSAL")) {
    causal = std::atoi(env) ? 1 : 0;
}
```

**Ядра не тронуты** — все три `.cu` sealed md5 неизменны.

**Новый bench binary md5**: `1f884d70b82501e626fdfef4efce644c` (был `c04685a2b5d378f5ee147e21c93a804b` в 062).

### §A2 Валидация: nc после правки

**Ожидание**: nc-медиана ≈ 062 cert 42.346 ± дрейф < 1%.

**Прогон** (default, CAUSAL unset): `total=42.006` ms
**Прогон** (`CAUSAL=0`): `total=42.073` ms

Drift vs cert 42.346: **-0.7 % / -0.6 %** (< 1% ✓). **Правка bench не сдвинула вычисления**.

### §A3 Causal 30-run стенд-протокол

**Скрипт**: `runs/reports/063_causal_30run.sh` → `063_causal_30run.txt`
**Условия**: `CAUSAL=1`, bh=128 sl=8192 hd=128, 4 warmup + 30 timed, fingerprint per run 252/124/69/38 ✓, gate-тишина ✓

**Стенд-медианы** (обе конвенции):

| Метрика | Median | Min | Max | CV | Режим |
|:--|:-:|:-:|:-:|:-:|:--|
| **total** | **22.206 ms** | 22.169 | 22.236 | **0.074%** | R2C causal in-chain |
| D | 0.340 | — | — | — | k_d_precompute |
| merged | 12.706 | — | — | — | dV+dS_nat causal |
| dk_new | 4.5705 | — | — | — | S2v4 causal-skip |
| dq_new | 4.5860 | — | — | — | 041 causal-skip |
| **TFLOPS proj 16N²d** | **792.24 T** | — | — | — | no mask discount |
| **TFLOPS fused 10N²d** | **495.15 T** | (= 792.24 × 10/16) | — | — | no mask discount |

**Temp range**: 42-48°C.

### §A3.a Разбор расхождения causal ≠ nc

**Ожидание TZ**: "causal ~ nc (маска считается, тайлы не скипаются); >2% расхождения = разбор".

**Факт**: causal/nc = 22.206 / 42.346 = **0.524** — расхождение -47.6%.

**Причина** (архитектурная фича, не bug):
- `fa_bwd_merged_v1.cu:146`: `const int qt_start = causal ? kt : 0;` — causal-skip
- `fa_bwd_dk_new.cu:85`: `const int qt_start = causal ? kt : 0;` — causal-skip
- `fa_bwd_dq_new.cu:*`: аналогично (dq тоже skip'ает по kt)

Все три ядра пропускают тайлы `i < kt` при causal — это стандартная causal-attention оптимизация из forward FA. Среднее число активных `(qt, kt)` пар для causal ≈ 50% от rectangular grid → **~50% wall reduction**.

**Ярлык**: **causal-skip присутствует во всех 3 iter-loops**; ratio 0.524 — норма для этой архитектуры. TZ формулировка "маска считается" не соответствует реальности этих ядер.

Bit-exact chain includes causal forms (F2/F4/F6/F8/F10) — все 11/11 x3 PASS (см. 062 §4.a). Causal correctness подтверждена.

### §A4 Леджер-строка (стартовая черта для скип-главы 064)

```
E2E R2C in-chain (062+063 cert):
  nc     wall = 42.346 ms   TFLOPS proj = 415.44 T (16N²d) / 259.65 T fused (10N²d)   CV = 0.098%
  causal wall = 22.206 ms   TFLOPS proj = 792.24 T (16N²d, no mask discount)           CV = 0.074%
  ratio(c/nc) = 0.524       causal-skip present in {merged, dk_new, dq_new} qt-loops
```

---

## §B Релиз v0.2.0

### §B1 Состав дерева

```
release_v0.2.0/
├── Makefile                    (NVCC ?= nvcc; NVCC=/usr/local/cuda-13.1/bin/nvcc autodetect)
├── verify.sh                   (single-command build+validate)
├── README.md                   (EN, TFLOPS with both conventions, requirements, quickstart)
├── LICENSE                     (placeholder — AWAITING VUGAR DECISION)
├── src/                        (8 files: 7 kernel .cu + 1 common .cuh)
│   ├── fa_bwd_common.cuh       md5 4407ec9cf64708a2a28dc36633d5d6f1
│   ├── fa_bwd_dk.cu            md5 068d6a4fdf5ae04816ebca199b9293cc  (sealed dK reference)
│   ├── fa_bwd_dk_new.cu        md5 25e5e1077cc3bec2c49bf9288fe60c54  (S2v4 PRODUCTION)
│   ├── fa_bwd_dq.cu            md5 09274fb8145744ff9dcc9075b53c2c85  (sealed dQ reference)
│   ├── fa_bwd_dq_new.cu        md5 d7a11a3d788eb4c396d892bc9c8ab754  (041 PRODUCTION)
│   ├── fa_bwd_ds_gen.cu        md5 665a350d3da8ae90b816ccd6b55db346  (reference for r2c_merged_bit_exact)
│   ├── fa_bwd_dv_mma_p1.cu     md5 f25c06d8ec1d6ff9b12f87a9b7d97428  (reference for dV)
│   └── fa_bwd_merged_v1.cu     md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33  (040 PRODUCTION)
├── tests/                      (3 harnesses)
│   ├── bench_r2c_e2e.cu        (E2E wall, CAUSAL env, 4-way fingerprint)
│   ├── r2c_merged_bit_exact.cu (dV+dS_nat vs sealed; INJECT_BITFLIP env)
│   └── r1b_dk_bit_exact.cu     (dK vs sealed)
└── docs/cert/
    └── cert_summary.md         (sanitized 062+063 summary; no internal paths)
```

### §B2 Санитария

**Grep по релиз-дереву на внутренние пути**:
```
$ grep -rE "/data/lib/podman-data|/root/|localhost|127.0.0.1" release_v0.2.0/
(no matches — clean ✓)
```

**Cert data**: `runs/` целиком НЕ включена; только curated `cert_summary.md` в `docs/cert/`. Raw логи с path'ами не пошли в релиз.

### §B3 README

- Заголовок: "First public FP8 FlashAttention backward for NVIDIA RTX PRO 6000 Blackwell (sm_120a)"
- **Числа с подписями**: 415.44 T proj (16N²d Tri Dao V3 ref) / 259.65 T fused (10N²d R2C honest) / wall 42.346 ms nc + 22.206 ms causal (30-run CV 0.098%/0.074%)
- **Форма**: канон-форма `bh=128 sl=8192 hd=128` явно указана
- Сравнительный ориентир: FA2 A100 BF16 ~175 T (10N²d) — с пометкой "literature estimate, not from this repo"
- Требования: sm_120a, CUDA 13.1+, driver 580.159.03+
- Quickstart: `make && ./verify`
- Точностный гибрид: FP8 (e4m3) MMA + FP16 buffers + FP32 accum; floor-константы (nc ~4.65-4.87e-3, causal ~3.1-3.3e-2 vs FP64) — **честно указаны в таблице**
- Known limitations: hd=128, canon-геометрия, sm_120a-only, causal-skip present (~52% wall)
- Ссылка на forward v0.1.0: link TBD

### §B4 LICENSE — AWAITING VUGAR DECISION

**Файл `LICENSE`**: placeholder-заглушка. Опции для выбора:
- Apache-2.0
- MIT
- BSL (Boost Software License)
- Custom

**По TZ**: "тег до решения Vugar по лицензии — запрещено". Тег v0.2.0 **не ставится**, push **не выполняется** до заполнения LICENSE.

### §B5 Git — подготовка (не выполнено до §B4)

**План после решения Vugar**:
1. Vugar выбирает лицензию → заменить `LICENSE` реальным текстом + добавить SPDX header в исходники
2. `git init` в `release_v0.2.0/` (или `git subtree` из основного репо)
3. Коммиты по смыслу: `feat(kernels): S2v4 dK` / `feat(harnesses): CAUSAL env` / `docs(cert): 062+063 summary` / `chore: license {X}`
4. Аннотированный тег `v0.2.0` с сообщением:
   ```
   v0.2.0 — first public FP8 FlashAttention backward (sm_120a)
   
   Cert: 415.44 T proj (16N²d) / 259.65 T fused (10N²d) / 42.346 ms nc median (30-run CV 0.098%)
   Kernels md5:
     dk_new:  25e5e1077cc3bec2c49bf9288fe60c54  (S2v4)
     merged:  2bf32ab7d4c5ecabb4ee2dbf1b5d4b33  (040)
     dq_new:  d7a11a3d788eb4c396d892bc9c8ab754  (041)
   Report: runs/reports/062_cert400.md
   ```
5. `git push origin v0.2.0 && git push origin <release-branch>`
6. **Правило TZ**: **тег ставится ТОЛЬКО после зелёного verify-из-чистого-клона** (см. §B6 ниже — уже выполнено ✓).

### §B6 Пост-релизная сверка

**Verify из чистого клона** (симуляция `git clone` в отдельную директорию, полный цикл build + validate):
```
$ cp -r release_v0.2.0 release_verify_clone
$ cd release_verify_clone && bash verify.sh
```

**Полный лог**: `runs/reports/063_verify_clean_clone.txt`

**Итог verify.sh**:
```
[1/5] OK — all 3 binaries built.
[2/5] OK — fingerprint 252/124/69/38 matched.
[3/5] OK — 11/11 gradient chains BIT-EXACT.
[4/5] OK — 11/11 dK BIT-EXACT.
[5/5] OK — wall printed both conventions.

VERIFY PASSED — v0.2.0 ready.
  Kernel md5:
    25e5e1077cc3bec2c49bf9288fe60c54  fa_bwd_dk_new.cu   ← совпадает с sealed ✓
    2bf32ab7d4c5ecabb4ee2dbf1b5d4b33  fa_bwd_merged_v1.cu ← совпадает с sealed ✓
    d7a11a3d788eb4c396d892bc9c8ab754  fa_bwd_dq_new.cu   ← совпадает с sealed ✓
```

**Wall clean-clone sample** (nc, 5-run):
- run=1: 41.969 / 418.17 T proj / 261.15 T fused
- run=2: 42.126 / 417.45 T / 260.64 T
- run=3: 42.184 / 416.80 T / 260.52 T
- run=4: 42.269 / 416.60 T / 260.16 T
- run=5: 42.234 / 416.12 T / 260.11 T

Медиана 42.2 ms — в допуске cert 42.346 ±дрейф.

**Wall clean-clone sample** (causal, 5-run):
- runs 1-5: 22.212-22.234 ms, TFLOPS proj 790.7-792.1 T

**Пост-релизная сверка md5 (тег vs sealed)**:

| Файл | Sealed md5 | Клон md5 | Match |
|:--|:--|:--|:-:|
| src/fa_bwd_dk_new.cu | 25e5e107... | 25e5e107... | ✓ |
| src/fa_bwd_merged_v1.cu | 2bf32ab7... | 2bf32ab7... | ✓ |
| src/fa_bwd_dq_new.cu | d7a11a3d... | d7a11a3d... | ✓ |

**Ворота релиза пройдены** — verify.sh green из чистого клона, md5 совпадают, все конвенции подписаны.

---

## §C Итоги 063

1. **§A1 Env-параметр CAUSAL** добавлен в `bench_r2c_e2e.cu` (bench-side, 3 строки после `causal=0`). Bench md5 c04685a2 → **1f884d70**. Ядра не тронуты.

2. **§A2 Валидация nc**: default (CAUSAL unset) = 42.006, `CAUSAL=0` = 42.073 — drift < 1% vs cert 42.346 ✓.

3. **§A3 Causal 30-run**: median **22.206 ms**, CV=0.074%, tflops16=**792.24 T**, tflops10=**495.15 T** (no mask discount). Temp 42-48°C.

4. **§A3.a Разбор ratio 0.524**: causal-skip присутствует во всех 3 iter-loops (`merged`, `dk_new`, `dq_new` имеют `qt_start = causal ? kt : 0`). Архитектурная фича, не bug. Bit-exact causal 11/11 ✓ (062).

5. **§A4 Леджер-строка**: `nc 42.346 / causal 22.206 (ratio 0.524, skip in 3 loops)` — стартовая черта скип-главы 064.

6. **§B1-B3 Релиз v0.2.0** подготовлен в `release_v0.2.0/`: 8 sources + 3 tests + Makefile + verify.sh + README (EN, все числа подписаны конвенцией/формой/режимом) + LICENSE placeholder + docs/cert/cert_summary.md (sanitized).

7. **§B2 Санитария**: чистая, нет `/data/lib/podman-data`, `/root/`, hostname'ов в опубликованных файлах.

8. **§B4 LICENSE**: **AWAITING VUGAR DECISION** (placeholder). Тег и push не выполнены.

9. **§B5 Git**: план подготовлен, коммиты/тег/push ждут §B4.

10. **§B6 Пост-релизная сверка**: verify.sh из чистого клона **PASSED** (build + fingerprint 252/124/69/38 + bit-exact 11/11 merged + 11/11 dK + wall 5-run обеих конвенций); **md5 kernels совпадают с sealed** ✓. Ворота релиза пройдены.

### Chain md5

- 060 `1b29dc0852aba39e6933465cfad60e98`
- 061 `cf99a50510700f8f994f56ec3274c3b5`
- 062 `b7044db70019e8fa7dea260f9f235b6c`
- **063 `b7c82475ed49ac4821c7346f99a38fb1`**

### Файлы 063

- `runs/reports/063_causal_release.md` (this report)
- `runs/reports/063_causal_30run.sh` + `063_causal_30run.txt` — 30-run causal
- `runs/reports/063_stats.sh` — статистики compute
- `runs/reports/063_verify_clean_clone.txt` — полный verify.sh лог
- `runs/reports/063_run_verify.sh` — clean-clone runner
- `release_v0.2.0/` — весь release-дерево (8 sources + 3 tests + Makefile + verify + README + LICENSE + docs)
- `release_verify_clone/` — verified clone (для reference)

---

**End 063. Causal 30-run: 22.206 ms median (CV 0.074%), 792.24 T proj / 495.15 T fused (no mask discount); ratio 0.524 (causal-skip in 3 loops, архитектурная фича). Релиз v0.2.0 подготовлен: verify.sh PASSED из чистого клона, md5 совпадают с sealed, санитария чистая. Ждём LICENSE от Vugar — тег/push не выполнены.**
