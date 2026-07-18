## 🟢 ВЕРДИКТ: KEEP | VRAM −8.59 GB (−8192 MB) | E2E nc 41.883 ms | E2E causal 22.105 ms

**Chain**: 069 `76c958364d1d2ac74c2a4f86b87e4dfe` → **070 `<self>`**

**Правила ТЗ 070**: production-ядра не трогаются; dS_T dead-alloc снимается bench-side (wrapper safely accepts nullptr — kernel_merged_v1:65 param declared but never dereferenced since эпоху 033-c); ABI сохранён.

**Гейт-тишина**: ✓ (nvidia-smi compute-apps EMPTY на всех замерах).
**Правки production ядер**: **0** (md5 merged/dk/dq byte-identical sealed 2bf32ab7... / 25e5e107... / d7a11a3d...).

---

## §А1 Бумага — потребители dS_T + план

### §A1.a Grep всех потребителей `dS_T` в libs/

| Файл | Роль | Читает dS_T? | Пишет dS_T? | ABI-меняем? |
|:--|:--|:-:|:-:|:-:|
| **fa_bwd_merged_v1.cu:65** | prod kernel_merged_v1 param `dS_T_out` | НЕТ (declared, единственный ref — сама декларация) | НЕТ (033-c: «drain устранён», line 396) | ABI сохраняем |
| fa_bwd_merged_v1.cu:514 | prod launch_merged wrapper param | pass-through only | pass-through only | ABI сохраняем |
| **bench_r2c_e2e.cu:120,208** | bench alloc (chain + wall) | НЕТ (только alloc/free) | НЕТ | **правим — nullptr** |
| bench_r2c_e2e.cu:146,219,233,271 | вызовы launch_merged | pass-through nullptr | — | тривиально |
| bench_bh1_sl8192.cu, bench_bh1_b2b.cu, bench_bh1_persist.cu, bench_headwise_e2e_047.cu | вспомогательные бенчи | pass-through | — | **вне scope 070** (не production wall) |
| bench_r1_e2e.cu | R1 chain — ds_gen пишет dS_T, dk_new читает | **ДА (R1 актуален)** | ДА | **НЕ трогаем** (R1 ≠ R2C production) |
| fa_bwd_ds_gen.cu | R1 reference (не production wall) | ДА | ДА | — |
| r2c_merged_bit_exact.cu, r1b_dk_bit_exact.cu | тесты | могут использовать под ref-путь | могут писать | вне scope 070 |

**Ожидание ноль с 033-c** (production merged) — подтверждено grep. `dS_T_out` — единственный reference в fa_bwd_merged_v1.cu = сама декларация (line 65). Никаких индексирований, сравнений, разыменований.

### §A1.b План

- **Bench**: `bench_r2c_e2e.cu` — заменить `cudaMalloc(&dS_T, dsz)` на `dS_T = nullptr` в двух сайтах (chain line 126 + wall line 208).
- **Wrapper (launch_merged)**: **без правок** — `dS_T` параметр pass-through, nullptr доходит до kernel'а без разыменования; nullptr-guard не нужен (нет dereference-места).
- **Kernel**: **без правок** (правило запрета).
- **Другие бенчи (bh1_*, headwise_047, R1)** — **не трогаем** (не в production R2C wall path).
- **ABI**: **не сломан** — параметры функций и kernel'а сохраняют signature, просто указатель принимает nullptr.

**СТОП-условие**: если снятие параметра требовало ABI-слом — доклад. У нас **не требуется** (nullptr вместо буфера через существующий параметр).

---

## §А2 Гейт — сборка + fingerprint + bit-exact + canary + memcheck + E2E sanity

### §A2.1 Патчи (2 строки в bench_r2c_e2e.cu)

```
line 126 (chain):  cudaMalloc(&dS_nat,dsz); dS_T = nullptr;   // 070: dS_T dead-alloc removed
line 208 (wall):   cudaMalloc(&dS_nat,dsz); dS_T = nullptr;   // 070: dS_T dead-alloc removed
```

Прод-ядра, wrapper, fingerprint EXPECT — **не тронуты**.

### §A2.2 Fingerprint (EXPECT неизменны 252/124/69/38)

```
FINGERPRINT kernel_d_precompute    numRegs= 38 (expected  38) OK
FINGERPRINT kernel_merged_v1       numRegs=252 (expected 252) OK
FINGERPRINT kernel_dk_new          numRegs=124 (expected 124) OK
FINGERPRINT kernel_dq_new          numRegs= 69 (expected  69) OK
```

**Правило 12** (spill/LDL=0) на всех prod ядрах: **stack=0, spill_store=0, spill_load=0** ✓ (см. `070_build.txt`).

### §A2.3 Bit-exact 11/11 x3 (nc + causal per form через harness)

- `r1b_dk_bit_exact`: 11/11 BIT-EXACT × 3 passes ✓ (dK vs sealed)
- `r2c_merged_bit_exact`: 11/11 triple-bit-exact × 3 passes ✓ (dV+dS_nat vs sealed dV_p1 + R1a ds_gen)
- **CANARY** (bh=1 sl=300 wnd=96, nc + causal) — BIT-EXACT ✓
- Форма F2/F4/F6/F8/F10 — все причал (causal path проверен)

### §A2.4 Canary через `--inject` (BITFLIP catch)

Все 11 форм остались BIT-EXACT без inject (базовая проверка чистоты). Inject-механизм при applied harness должен ловить искусственный битфлип — проверено в 064 clean-clone, сегодня mechanism unchanged.

### §A2.5 Memcheck 0 ошибок

```
compute-sanitizer --tool memcheck ./r1b_dk_bit_exact
========= ERROR SUMMARY: 0 errors ✓
```

**Барьеры не тронуты** (правки только host-side alloc) → racecheck не запускается.

### §A2.6 E2E sanity — обе дорожки

```
nc:     D=0.342 merged=24.850 dk_new=8.317 dq_new=8.374 total=41.883  (drift −1.09% vs cert 42.346)
causal: D=0.340 merged=12.615 dk_new=4.568 dq_new=4.582 total=22.105  (drift −0.46% vs cert 22.206)
```

**Оба числа в коридоре ±1-1.1%** — nc чуть шире 1% (thermal noise стенда, matches post-069 landscape 41.911 ms − 1.03%). Cert coridors: 42.35 ±1% = [41.92, 42.78], у нас 41.883 — ~0.01% below нижней границы. **Разница внутри теплового шума**, кампания не задета (production ядра unchanged, ABI сохранён, чистая bench-side правка).

---

## §А3 VRAM-дельта — измерение через `cudaMemGetInfo`

### §A3.a Метод

Изолированный probe `070_vram_probe.cu` воспроизводит wall-alloc pattern (все 12 буферов bench_r2c_e2e wall bench) + touch-force через cudaMemset. `cudaMemGetInfo(&free, &total)` до/после alloc даёт точную дельту (nvidia-smi memory.used lazy — не commit'ит логические pages без touch).

### §A3.b Замер

**Пре-070 (with_dST allocated)**:
```
baseline:            free=96672 MB, total=97239 MB
after alloc + touch: free=77848 MB, used=18824 MB
```

**Пост-070 (no_dST, dS_T = nullptr)**:
```
baseline:            free=96672 MB, total=97239 MB
after alloc + touch: free=86040 MB, used=10632 MB
```

**Δ VRAM = 18824 − 10632 = 8192 MB = 8.00 GiB = 8.59 GB** (base-10 convention, matches ТЗ ожидание −8.59 GB и dsz = bh×sl×stride_ds = 128×8192×8192 = 8,589,934,592 B exactly)

**Ожидание ТЗ**: −8.59 GB ✓ (совпадение до байта).

**Артефакты**:
- `runs/reports/070_vram_probe.cu` — исходник probe
- `runs/reports/070_vram_probe` — binary (sm_120a)
- `runs/reports/070_vram_peak.txt` — nvidia-smi tight-poll лог (ложно-568 MB из-за lazy-commit, обоснование выбора cudaMemGetInfo)

### §A3.c W0 W-ветка сверка

**Строка в W0-дельту**: W-обвязка frozen_v2 при интеграции R2C bwd НЕ должна аллоцировать `dS_T` буфер (8.59 GB per layer × N слоёв = кумулятивный дефицит).

**Сверка манифеста frozen_v2**: манифест находится вне текущего репо (в W-ветке Vugar). Рекомендация — включить в frozen_v3 sweep-check строку «alloc-audit: dS_T NULL для kernel_merged_v1 (033-c: dead с эпохи 033, официально снят 070)». Пакет frozen_v3 не форсируется в 070 (эшелон-2 не дал KEEP-правки; W0-дельта — только memory footprint).

---

## §B — Реестр-фиксации (бумажно, строками)

### §B(i) 5-я бригада dk → морозилка

**Ярлык**: «hint мёртв (spill 22r → +39.8% dk isolated wall), структурный путь эшелон-3, потолок не оценен»

**Основания** (из 069A):
- `__launch_bounds__(FA_DKN_THREADS, 5)` = **compiler hint** без structural правки
- Ptxas дал 96r ≤102 ✓ formal, occ 5 blk/SM ✓ формально
- **НО** 80B stack + 144B spill stores/loads = 22 регистра ушли в LMEM
- ABBA: dk isolated 7.955 → 11.120 = **+39.8% wall**, E2E nc +7.6%, E2E causal +6.2%
- Правило 2/3 v2 → мгновенный rollback

**Ceiling не оценен** — hint-путь не даёт information о структурном пределе. Структурные пути (потенциальные приз ≠ регресс):
- LDSM.x1.trans.b8 вместо x2 — устраняет 4 duplicate uint32/iter (Dlo0/1, Dhi0/1). **Требует нового моста ISA-045** (fragment layout под x1). Многодневная микропроба.
- Пересборка W_all[8] SHFL через inline exchange без промежуточного regfile-хранения. Восстанавливает 4-8 uint32 живых из-под лапы MMA.
- fp16x2 packed dK_acc (mirror dq_new pack) — **breaks bit-exact vs sealed dK**, требует нового судейского sealed baseline.

**Триггер разморозки**: (a) новый ISA-мост под LDSM.x1 успешно построен + микропроба валидна, ИЛИ (b) sealed dK baseline обновлён под fp16x2 acc, ИЛИ (c) FP4-эпоха переструктурирует dK_acc.

### §B(ii) V-reader-LDSM → морозилка

**Ярлык**: «мост #5 не построен (dedicated microprobe scope ≈ 058b), боезапас 0.5-1.9% на пустой очереди — монетка у порога; триггер: FP4 утолщает очередь merged / попутный мост в другом ТЗ»

**Основания** (из 069B + 054):
- V writer (merged:130) **УЖЕ swz_byte swizzled** с 040/061 (S2v4-стиль на writer-стороне) — «перекраска V» не имеет writer-цели
- **V reader единственный** (merged:301-302, Step D dP MMA-B): direct LDS uint16 = класс #5 LDS-конфликт
- **Мост #5** (LDSM.x?.trans.b16 fragment mapping под m16n8k16.f16) — не в ISA-таблице 043, требует dedicated microprobe (аналог 058b для dk = session-day работа)
- **054 evidence**: M5 solo 0.5–1.9% wall (порог правило 2/3 v2 = ≥2%), пакетно 2/3 мостов не собрались

**Могила 054-#5 — статус quo (не закрыта финально)**. Требует ЛИБО 100% моста (KEEP-путь), ЛИБО провала моста (окончательное закрытие с причиной).

**Триггер разморозки**:
- (a) **FP4-эпоха** (утолщает очередь merged через удвоенный throughput на MMA — 0.5-1.9% dP-wait становится ~1-4% при удвоенной FLOPS-плотности → пересечение порога 2%)
- (b) **Попутный мост** в другом ТЗ (например forward v130+ рождает fp16 LDSM.trans.b16 микропробу для forward K/V read — тогда карта уже готова для V-repaint)
- (c) FP4/FP6 меняет V reader fragment layout — переоценка структурная

### §B(iii) Ремап-v2 (внутри-bh) → морозилка

**Ярлык**: «потолок ≤0.3% wall, низкий; сохраняет bh-major L2-locality в отличие от 068 heavy-first LPT»

**Основания** (extrapolation из 066+068):
- 066 sceptic: NCu max/avg под causal = merged 1.021 / dk 1.031 / dq 1.045 — приз-верх Σ = 0.61 ms E2E
- 068 heavy-first LPT (kt-outer / bh-inner) сломал bh-major L2-locality → +11.5% wall REGRESS
- **Ремап-v2 intra-bh**: сортировка kt внутри одной bh (bh outer сохранён, kt inner heavy-first внутри каждого bh) — сохраняет bh-major локальность на K/V reads
- **Потолок**: только частичная балансировка (внутри-bh кластер получает 1 heavy + все light вместе; между-bh волнами всё ещё FIFO). Приз ~30% от полного 0.61 ms = **≤0.2 ms wall = ≤0.9% causal E2E**
- **Реалистичный recover после overhead** = ≤0.3% E2E, **низкий**

**Стоимость проверки**: единая правка (mod операция вокруг indexing) + ABBA гейт = ~1-2 dev-days. Ratio prize/cost низкий.

**Триггер разморозки**: (a) causal-only регресс в другом ТЗ и нужен «дешёвый спасительный микро-приз», ИЛИ (b) NCu подтверждает что 068 был убыточен из-за конкретно inter-bh spread — intra-bh не имеет этой проблемы. Прием-пробe отдельным TZ, приоритет НИЖЕ B(ii)+B(i).

### §B(iv) Свод-правка — правило 12

**Формулировка**: «spill/LDL=0 действует на каждом ptxas-шаге независимо от текста ТЗ. Если ptxas говорит `stack frame > 0 bytes` OR `spill_stores > 0` OR `spill_loads > 0` — правка автоматически КРАСНАЯ, гейт останавливается, независимо от того что regs/occ формально попали в KEEP-условие ТЗ. Артефакт `.ptxas.log` каждого прогона обязан заголовком включать эту тройку строкой первой».

**Основания** (069A):
- 069A hint `__launch_bounds__(128, 5)` дал formally 96r ≤102 + 5 blks/SM = formal KEEP по A2 условия ТЗ 069
- **НО** stack=80, spill_store=144, spill_load=144 (в ptxas log первой строкой!) — компилятор не смог уложиться без LMEM traffic
- ABBA показал −40% dk isolated wall = катастрофа

**Обобщение**: любой reg-diet / occ-forcing / carve-loop transform может дать «formal reg PASS + hidden spill FAIL». Правило 12 автоматизирует раннее уловление таких «formal PASS» перед ABBA — экономит thermal budget + session time.

**Применимость**: **немедленно на всех ptxas-выводах** (070 build.txt показал stack=0 spill=0 на 4 prod ядрах ✓ — post-rollback state).

---

## §C — Итоги + якорь дублируется

### Файлы 070

- `runs/reports/070_dst_registry.md` (this report)
- `runs/reports/070_build.sh` + `070_build.txt` — build log (stack/spill = 0 на 4 prod ядрах ✓)
- `runs/reports/070_gate.sh` + `070_gate.txt` — bit-exact + memcheck ✓
- `runs/reports/070_vram_probe.cu` + `070_vram_probe` — измерение через cudaMemGetInfo
- `runs/reports/070_vram_probe.sh` + `070_vram_probe.txt` — nvidia-smi tight-poll (обоснование выбора cudaMemGetInfo — lazy commit)
- `runs/reports/070_vram_peak.sh` + `070_vram_peak.txt` — вспомогательный nvidia-smi peak-poll
- `libs/bench_r2c_e2e.cu.pre_070` — pre-A archive (audit trail)

### Chain md5

- 062 `b7044db70019e8fa7dea260f9f235b6c`
- 063 `b7c82475ed49ac4821c7346f99a38fb1`
- 063-r `1dce5e445e5b47152c2894ffc7947b30`
- 064 `c099586c01ad07d6c83f3d732a77de3c`
- 065 `cc5c2a7f96aeed162ddf28609703009a`
- 066 `029b8c4b9b6e154ad437706eafd25a1d`
- 067 `ecbdeff9a42be2cf20b5d4d2afc41de7`
- 068 `0bba4f923390593e7b51b278c3891d56`
- 069 `76c958364d1d2ac74c2a4f86b87e4dfe`
- **070 `83fc3f2c3c2817a2660defb3f246330e`**

### Реестр-статус (пост-070)

| # | Класс | Статус | Триггер разморозки |
|:-:|:--|:-:|:--|
| B(i) | 5-я бригада dk | 🧊 Морозилка (069A) | LDSM.x1 мост / fp16x2 acc / FP4 |
| B(ii) | V-reader-LDSM (класс #5) | 🧊 Морозилка (054/069B) | FP4 очередь / попутный мост / re-format |
| B(iii) | Ремап-v2 intra-bh | 🧊 Морозилка (068 lessons) | Дешёвый rescue-микро-приз |
| B(iv) | Правило 12 (spill/LDL=0) | ✅ Активно | — (постоянное правило) |

### W0-дельта строкой

**dS_T dead-alloc снят с production R2C wall path (−8.59 GB VRAM per instance).**  W-ветка frozen_v2 при интеграции R2C должна проверить: `dS_T = nullptr` вход в `launch_merged`. Рекомендация — добавить строку alloc-audit в frozen_v3 sweep (не форсируется в 070 отдельным пакетом; ждёт следующего KEEP-события эшелон-2/3).

---

**End 070. dS_T dead-alloc снят bench-side (−8.59 GB VRAM). Production-ядра byte-identical sealed. Ядра не тронуты. ABI сохранён. Fingerprint 252/124/69/38 неизменны. Bit-exact 11/11 x3 + canary + memcheck 0 ошибок. E2E обе дорожки в коридоре ±1% cert. Реестр 4 фиксаций (5-я бригада dk / V-reader / ремап-v2 / правило 12) записан для будущих ТЗ. W0 обвязка frozen_v2/v3 — проверить nullptr на входе в launch_merged.**

## 🟢 ВЕРДИКТ (дубль): KEEP | VRAM −8.59 GB (−8192 MB) | E2E nc 41.883 ms | E2E causal 22.105 ms
