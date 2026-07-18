# 063-r — Дочет по приёмке 063: работа-детектор + chrono cross-check + README-чистка

**Chain**:
- 061_s2v4_production.md `cf99a50510700f8f994f56ec3274c3b5`
- 062_cert400.md `b7044db70019e8fa7dea260f9f235b6c`
- **063_causal_release.md `b7c82475ed49ac4821c7346f99a38fb1`**

**Приёмка 063**: секция B принята целиком; секция A принята УСЛОВНО с дырой "работа-детектор не исполнен". 063-r закрывает дыру.

---

## Артефакт-хедер (правило 5)

```
libs/ (post-063r):
-rw-r--r--  13411 Jul  9  bench_r2c_e2e.cu       (+CHRONO env)
-rwxr-xr-x        Jul  9  bench_r2c_e2e          md5 fd7945b735a8d79900f925cabc77dd8f  (post-063r rebuild)
-rw-r--r--  12099 Jul  9  fa_bwd_dk_new.cu       md5 25e5e1077cc3bec2c49bf9288fe60c54  (S2v4 sealed, unchanged)
-rw-r--r--  25638 Jul  8  fa_bwd_merged_v1.cu    md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33  (040 sealed, unchanged)
-rw-r--r--  18834 Jul  8  fa_bwd_dq_new.cu       md5 d7a11a3d788eb4c396d892bc9c8ab754  (041 sealed, unchanged)
```

**Правки production ядер в 063-r**: **0** (все sealed neizменны).
**Правки bench**: **1** (env `CHRONO` = std::chrono cross-check вокруг полной цепи, мимо cudaEvent).

---

## §1 NCu работа-детектор (1 прогон causal vs 1 прогон nc, per-kernel таблица)

**Скрипт**: `runs/reports/063r_ncu_work.sh` → `063r_ncu_work.txt`

**Метрики**: `smsp__inst_executed.sum` (executed instructions), `dram__bytes.sum`, `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`, `launch__grid_size`.

Примечание: `sm__inst_executed_pipe_tensor_op_hmma.sum` вернул `n/a` — для fp8 MMA нужна `qmma`-переменная (не hmma); заменено на `smsp__inst_executed.sum` как proxy для всей исполненной работы.

**Per-kernel таблица**:

| Kernel | Metric | NC | Causal | Ratio (c/nc) | Вердикт |
|:--|:--|:-:|:-:|:-:|:--|
| **merged_v1** | executed insts | 14,921,498,624 | 7,534,772,224 | **0.505** | ~0.5 ✓ |
| merged_v1 | DRAM bytes | 9.80 GB | 5.51 GB | **0.562** | ~0.56 ✓ |
| merged_v1 | LSU wavefronts | 4,063,373,942 | 2,048,496,752 | **0.504** | ~0.5 ✓ |
| **dk_new (S2v4)** | executed insts | 4,087,283,712 | 2,068,938,752 | **0.506** | ~0.5 ✓ |
| dk_new | DRAM bytes | 9.26 GB | 4.99 GB | **0.539** | ~0.54 ✓ |
| dk_new | LSU wavefronts | 2,435,062,108 | 1,227,958,410 | **0.504** | ~0.5 ✓ |
| **dq_new** | executed insts | 4,836,229,120 | 2,447,507,456 | **0.506** | ~0.5 ✓ |
| dq_new | DRAM bytes | 9.25 GB | 4.99 GB | **0.539** | ~0.54 ✓ |
| dq_new | LSU wavefronts | 1,188,268,213 | 598,140,770 | **0.503** | ~0.5 ✓ |
| **d_precompute** | executed insts | 95,420,416 | 95,420,416 | **1.000** | не iter-dep ✓ |
| d_precompute | DRAM bytes | 545 MB | 545 MB | 1.000 | ✓ |
| d_precompute | LSU wavefronts | 7,352,969 | 7,354,297 | 1.000 | ✓ |

### §1.a Вердикт работы-детектора

**Ожидание TZ (честный скип)**: MMA_causal / MMA_nc ~ 0.52-0.55 × nc; DRAM merged-writes ~вдвое меньше.

**Факт**:
- **Executed instructions ratio: 0.505-0.506** для всех 3 iter-loop ядер (merged/dk/dq) — **в честном скип-диапазоне** (ожидание 0.52-0.55 ± 5%)
- **DRAM ratio: 0.54-0.56** — DRAM writes/reads реально сокращены (fits ожидание "вдвое меньше по dS")
- **Wavefronts ratio: 0.50** — SMEM accesses ровно вдвое меньше
- **D (не iter-dep) ratio: 1.000** — правильно, D независимо от causal (contra-control ✓)

**НЕ фантом**: счётчики **не равны nc** (что было бы если работа исполнена, а таймер врёт). Не полу-iters (замерный цикл гоняет полное количество, работа per iter сокращена).

**ВЕРДИКТ**: **работа реально сокращена в ~2× — 22.206 ms wall честный**.

---

## §2 Кросс-проверка std::chrono вокруг полной цепи

**Правка bench** (только bench-side, kernels не тронуты):
```c
// 063-r §2 CHRONO cross-check: измерение std::chrono вокруг полной цепи,
// МИМО cudaEvent-каркаса. Env CHRONO=1 включает.
if (const char *env = std::getenv("CHRONO")) {
    if (std::atoi(env)) {
        CKR(cudaDeviceSynchronize());
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            fa_bwd_dk::launch_d_precompute(...);
            fa_bwd_merged_v1::launch_merged(...);
            fa_bwd_dk_new::launch_dk_new(...);
            fa_bwd_dq_new::launch_dq_new(...);
        }
        CKR(cudaDeviceSynchronize());
        auto t1 = std::chrono::steady_clock::now();
        double chrono_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        printf("wall_chrono_avg_ms = %.4f (iters=%d, causal=%d)\n", chrono_ms/iters, ...);
    }
}
```

**Новый bench md5**: `1f884d70` → **`fd7945b735a8d79900f925cabc77dd8f`**.

**Прогон** `CAUSAL=1 CHRONO=1`:
```
=== CHRONO cross-check ===
  wall_chrono_avg_ms = 22.2848 (iters=20, causal=1)
  vs cudaEvent total = 22.1672  diff = 0.1175 ms (0.53%)
```

**Вердикт §2**: chrono **22.28 ms** vs cudaEvent **22.17 ms** = **diff +0.53%**. Разница — normal host-side driver overhead (chrono includes `cudaDeviceSynchronize()` cost, event framework не include). **22.2 ms wall подтверждён независимым секундомером** (мимо cudaEvent-каркаса).

---

## §3 Ledger-ретро: когда появился `qt_start = causal ? kt : 0`

**Grep по архивам**:
```
qt_start = causal ? kt : 0
```
Присутствует в **033_post_no_dst / 033_pre_merged / 033_sealed / 040_pre / 040_sealed / 041_dq_sealed / ...** и во **всех** более поздних архивах.

Также присутствует в **018_sealed_pack / 021_sealed_piv / 023_sealed_piv / 024_pre_pack_dq / 025_post_pack_dq / 027_post_pack_pi_dq / 029_d5lite_pack_pi** — то есть **как минимум с эпохи 018 (2026-07-07)**, возможно с V3 285.44 T cert (эпоха 61.6 ms).

**Аннуляция ярлыка** "causal==nc, скип не построен":
- **Дата протухания**: TZ 063 (2026-07-09), когда впервые померили causal 30-run.
- **Как только что-либо утверждалось "causal==nc"** — в бумаге TZ 063 (ожидание "маска считается, тайлы не скипаются"). Ошибочное предположение агента, унаследованное от неполного знания истории.

**Canary-вопрос** ("почему это не всплыло раньше"):
- Причinal-wall **не мерился 30-run** со времён V3 cert (эпоха 285.44 T @ 61.6 ms).
- Все cert-runs после V3 (029, 033, 040, 041, 062) были **nc-only** — фокус на достижение 400 proj TFLOPS в nc-конвенции.
- Bit-exact проверял causal формы (F2/F4/F6/F8/F10 в r2c_merged_bit_exact / r1b_dk_bit_exact), но wall causal timing не измерялся.

**Урок в свод**:
> **При первом cert-паке новых конвенций (nc / causal / любые режимы) обе конвенции меряются независимо, не полагаясь на ожидание "≈". Ожидания "маска считается", "накладные равны" — это гипотезы, не факты, пока NCu-детектор и независимый секундомер не подтвердили.**

---

## §4 README-правка ДО тега (release_v0.2.0/README.md)

**Изменения** (санитария метрической дисциплины):

**Удалено** из таблицы cert-numbers:
```diff
- | **TFLOPS, causal (proj 16N²d)** | 792.24 T | No mask-discount; if applied ~half → ~396 T effective |
```
Причина: 792.24 T proj (16N²d, no mask discount) — **фиктивная конвенция по неисполненной работе** (счёт FLOPs полной матрицы делённый на wall, где работа скипнута). Публичная — вводит в заблуждение.

**Добавлено** к таблице (объяснение causal):
```
**Causal path**: the kernels' qt-loops skip tiles with `i < kt` (standard causal-attention optimisation).
Instruction count, DRAM bytes, and shared-memory wavefronts all drop by ~50 % under `CAUSAL=1` —
measured directly with NCu counters, cross-checked with `std::chrono` bypassing the CUDA event framework.
The reported 22.206 ms is honest wall for the actually-executed work. Effective training-step
throughput therefore stays close to the non-causal number (~260 T fused-honest); a naïve
"TFLOPS = FLOPs_full_matrix / wall_causal" number would double-count the skipped triangle
and is **not** reported here.
```

**Обновлено** в "Known limitations":
```
- Causal skip: The causal path takes ~52 % of non-causal wall because tile-loops skip qt < kt
  (standard causal-attention optimisation, verified with NCu instruction/DRAM/wavefront counters
  and std::chrono cross-check). Effective throughput per active tile is comparable to non-causal.
```

**В cert-доке `docs/cert/cert_summary.md`** (внутренний документ) — **обе конвенции с полными подписями сохранены** (по указанию приёмки TZ 063-r).

---

## §5 Ре-verify релиза с обновлённым README

**verify.sh не затронут** README-правкой (build/fingerprint/bit-exact/wall независимы). Повторный запуск не требуется — README-only changes не влияют на функциональность.

Kernel md5 в `release_v0.2.0/src/` **без изменений**:
- fa_bwd_dk_new.cu: `25e5e1077cc3bec2c49bf9288fe60c54`
- fa_bwd_merged_v1.cu: `2bf32ab7d4c5ecabb4ee2dbf1b5d4b33`
- fa_bwd_dq_new.cu: `d7a11a3d788eb4c396d892bc9c8ab754`

---

## §6 Итоги 063-r

1. **§1 NCu работа-детектор** ✓: **executed instructions ratio 0.505-0.506** для 3 iter-loop ядер (merged/dk/dq), DRAM ratio 0.54-0.56, wavefronts 0.50; D (не iter-dep) ratio 1.000 — **работа честно сокращена в ~2×, не фантом**. Прибор подтвердил.

2. **§2 CHRONO cross-check** ✓: chrono 22.28 vs cudaEvent 22.17 = **diff +0.53%** (нормальный host overhead). 22.2 ms wall **подтверждён независимым секундомером** мимо cudaEvent-каркаса.

3. **§3 Ledger-ретро** ✓: `qt_start = causal ? kt : 0` присутствует минимум с **018_sealed_pack (2026-07-07)**, возможно с V3 285.44 T эпохи. **Ярлык "causal==nc, скип не построен" аннулирован с датой** TZ 063. Причина не всплытия: causal-wall не мерился 30-run со времён V3. **Урок в свод**: обе конвенции меряются независимо, ожидания не заменяют факта.

4. **§4 README-чистка** ✓: убрана строка `792.24 T proj (no mask discount)` из публичного README (фиктивная конвенция). Оставлено `wall 22.206 ms (30-run, CV 0.074%)` как истина + сноска "effective training-step throughput ~ nc" с явным объяснением работы-детектора. **Cert-доки 062/063 (внутренние) сохранили обе конвенции с полными подписями** — по приёмке.

5. **Вердикт-строка**: **работа посчитана прибором, 22.206 ms causal ПОДТВЕРЖДЁН** (NCu + chrono). Причinal-skip - honest architectural feature, не bug и не фантом.

### Chain md5

- 061 `cf99a50510700f8f994f56ec3274c3b5`
- 062 `b7044db70019e8fa7dea260f9f235b6c`
- 063 `b7c82475ed49ac4821c7346f99a38fb1`
- **063-r `1dce5e445e5b47152c2894ffc7947b30`**

### Файлы 063-r

- `runs/reports/063r_causal_release_fix.md` (this report)
- `runs/reports/063r_ncu_work.sh` + `063r_ncu_work.txt` — работа-детектор NCu
- `libs/bench_r2c_e2e.cu` — +CHRONO env (bench-side, kernels не тронуты)
- `libs/bench_r2c_e2e` md5 `fd7945b735a8d79900f925cabc77dd8f` (post-063r rebuild)
- `release_v0.2.0/README.md` — правка causal-строки (убрано 792.24 T proj, добавлено объяснение работы)

---

**End 063-r. Работа посчитана: instructions/DRAM/wavefronts ratio ~0.5 для 3 iter-loop ядер, D=1.0 (контроль). Chrono cross-check 22.28 vs event 22.17 (+0.53%). Ярлык "causal==nc" аннулирован с 2026-07-09; causal-skip присутствует с эпохи 018 (2026-07-07). README-правка: убрана фиктивная 792T proj (no mask discount) как публичное число; wall 22.206 ms остаётся истиной. Тег v0.2.0 всё ещё ждёт лицензию Vugar.**
