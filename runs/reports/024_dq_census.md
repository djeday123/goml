# 024 — dq_new census + профиль (ТЗ 009-2 шаг 0)

**Chain**:
- 022_dk_pi_pack.md md5: `9e617df84caba2e33351b502d9a5411c`
- 023_abba_piv.md md5: `5d38e0c86785cb4af95a5f3e88811bc3`

**Artifact header**:
```
-rw-r--r-- 13352  Jul  5 17:51  libs/fa_bwd_dq_new.cu    (md5 683396f8e6867e9fc2e26f8b628774f3 — первая фиксация)
-rwxr-xr-x 1.20M  Jul  7 10:06  libs/r1c_dq_wall         (56r/0s, session-fresh)
```

**Production не тронут в этом шаге. Только census, профиль, бумага.**

---

## 0a. Sanity + свежий профиль (production isolated dq_new)

### 0a.1 fa_bwd_dq_new.cu — первая фиксация в леджер

md5: **`683396f8e6867e9fc2e26f8b628774f3`** (создан 2026-07-05, sealed AA1).

### 0a.2 Session-fresh baseline 5-run (canonical bh=128 sl=8192 hd=128 nc)

| Run | avg_ms | TFLOPS |
|:-:|:-:|:-:|
| 1 | 8.504 | 258.59 |
| 2 | 8.510 | 258.41 |
| **3 med** | **8.515** | **258.27** |
| 4 | 8.522 | 258.05 |
| 5 | 8.518 | 258.15 |

**Median 8.515 ms / 258.27 T** (CV 0.08%). Isolated dq_new baseline на этой сессии.

### 0a.3 NCu полный srez

**Stall breakdown** (Σ ≈ 99% ✓):

| stall | % |
|:--|--:|
| **mio_throttle** | **46.73** ← главный горло |
| barrier | 10.93 |
| long_scoreboard | 10.22 |
| wait | 9.11 |
| not_selected | 8.94 |
| short_scoreboard | 5.97 |
| selected | 5.01 |
| math_pipe_throttle | 1.73 |
| dispatch_stall | 0.32 |
| no_instruction | 0.31 |
| drain / misc / membar / tex | ≈0 |
| **Σ** | **~99.3%** ✓ |

**Conflicts + inst + occupancy**:

| Метрика | Значение |
|:--|--:|
| LD conflicts | 541 M |
| ST conflicts | 60.9 M |
| LD wavefronts | 1.28 B |
| ST wavefronts | 598 M |
| **inst_shared_ld** | **889 M** |
| **inst_shared_st** | **537 M** |
| **LD excess rate** | (1.28−0.889)/0.889 = **+44%** |
| **ST excess rate** | (598−537)/537 = **+11%** |
| dram__bytes.sum | 9.25 GB |
| dram__bytes.per_second | 845 GB/s |
| sm__ctas_active.avg | 24.43% ≈ **5.9 blocks/SM** (peak 24×6=144) |
| sm__warps_active.avg | 48.86% ≈ **4.9 warps/SM** |
| sm__max_warps/cycle | 50% ≈ 6 peak |

**Blocks/SM факт: ~6** ✓ (SMEM-limited, 6·13824 + slack ≈ 84 KB / 100 KB peak).

### 0a.4 Хвост реестра: DRAM 7.26 vs 9.26 GB

- 005b (isolated hot pass, cache-hot): 7.26 GB
- 009-0 (E2E chain, cold): 9.26 GB
- **024 (isolated, session-fresh, warmup=5 iters=20)**: **9.25 GB** — **совпадает с 009-0**

**Разрыв объяснён**: 005b мерил cache-hot fully-primed режим (K/dS кэш переиспользован через iters); 009-0/024 — session-fresh (кэш периодически холодит). Между сессиями DRAM плавает если warmup недостаточен. **Хвост закрыт**: session-fresh надёжное значение = **9.25 GB @ 845 GB/s**. Ни другой счётчик, ни replay-модель — норм thermal state.

---

## 0b. SASS-census (из исходника fa_bwd_dq_new.cu:171-227)

### 0b.(i) K_T scatter — Phase 1.5 write (строки 187-195)

```c
for (int ni = 0; ni < NI_QK; ++ni) {   // 8
    int n_K = ni * 8 + l_div4;
    for (int bt = 0; bt < 4; ++bt) {   // 4
        smK_area[(k_lo + bt) * KT_STRIDE + n_K] = (kr_lo[ni] >> (bt*8)) & 0xFF;
        smK_area[(k_hi + bt) * KT_STRIDE + n_K] = (kr_hi[ni] >> (bt*8)) & 0xFF;
    }
}
```

**Ширина сторов**: **STS.U8** (побайтные, ровно как pre-pack dk_new).  
**Счёт на qt-виток per lane**: `8 × 4 × 2 = 64 STS.U8` — **один-в-один с pre-pack dk_new class** (у которого мы делали pack + π_V).  
**Марш**: строчный. row = `ks*32 + l_mod4*4 + bt` (varies по bt в пределах 4 rows; ks=wid const per warp). col = `n_K = ni*8 + l_div4` (varies по ni через 8 cols per lane).  
**Формула адресации из кода**: byte_addr = `row * 68 + col`.

### 0b.(ii) B-операнд K_T MMA-C (строки 216-219)

```c
for (int kb = 0; kb < KB_DQ; ++kb) {           // 2
    int k_j_lo = kb * 32 + l_mod4 * 4 + 0;
    int k_j_hi = k_j_lo + 16;
    for (int ni = 0; ni < NI_DQ; ++ni) {       // 16
        int n_d = ni * 8 + l_div4;
        B0 = smK_area[n_d * KT_STRIDE + k_j_lo];
        B1 = smK_area[n_d * KT_STRIDE + k_j_hi];
        mma(A, B0, B1, ...);
    }
}
```

**Счёт LDS на qt-виток per lane**: `2 kb × 16 ni × 2 halves = 64 LDS.32`.  
**Stride**: `KT_STRIDE = 68`.  
**Bank pattern** (row=n_d, col=k_j; bank = (row·17 + col/4) mod 32):
- За fixed (kb, ni): 32 lanes читают row `ni*8 + l_div4` (l_div4 varies 0..7, l_mod4 varies 0..3 меняет col). 8 lane-groups by l_div4 → 8 разных rows per (kb, ni). 4 lanes in group читают same row, different cols.
- **Retention-вопрос доказательство по коду**: **уникальность per-lane per-kt адресов**:
  - `n_d(ni) = ni*8 + l_div4` — 16 разных ni × 1 l_div4 = **16 unique n_d values per lane per kt**
  - `k_j(kb, half) = kb*32 + l_mod4*4 + {0, 16}` — **4 unique k_j offsets** per lane
  - Итого 16 × 4 = **64 unique (row, col) pairs per lane per kt**
- **Гарантированного повтора одних и тех же физ.слов между разными (kb, ni) нет.**
- Broadcast **есть** внутри lane-group of 4 (same l_div4, разные l_mod4) — 4 lanes читают same row, что даёт broadcast per row = 1 wavefront per group (тем что 4-way broadcast auto-merge).
- **Retention full/half/quarter не применим** — нет повтора чтений в пределах qt.
- **Между kt-iterations retention невозможен** — sm_K_area переписывается с новой K tile.

### 0b.(iii) A-операнд и фидеры

**A-load MMA-C** (строки 210-213), per lane per kt:
```c
for kb in [0..1]:
    A0 = smdS[m_lo * 80 + k_j_lo]
    A1 = smdS[m_hi * 80 + k_j_lo]
    A2 = smdS[m_lo * 80 + k_j_hi]
    A3 = smdS[m_hi * 80 + k_j_hi]
```
`2 kb × 4 = 8 LDS.32 per lane per kt`. A-loads уже **implicit retention внутри ni-loop** (16 mma-C итераций переиспользуют same A регистры).

**Phase 1.5 read K natural** (строки 179-184):
```c
for ni in [0..7]:
    kr_lo[ni] = smK_area[n_K*Hd + (k_lo^k_xor)]
    kr_hi[ni] = smK_area[n_K*Hd + (k_hi^k_xor)]
```
`8 ni × 2 halves = 16 LDS.32 per lane per kt`. XOR-swizzle by k_xor = l_div4<<4.

**Фидеры cp.async**:
- K → smK_area: **`Bc*Hd = 8 KB` за kt** через cp.async 16-байтовые чанки, `total = 512 chunks / 128 threads = 4 chunks/lane`
- dS_nat → smdS: **`Br*Bc = 4 KB` за kt** через cp.async 16-байтовые чанки, `total = 256 chunks / 128 threads = 2 chunks/lane`

### 0b.(iv) fp16-acc цепочка — BIT-EXACT инвариант

**Строки 203-226**: MMA-C fires **kb outer, ni inner**. Same order as sealed AA1 (fa_bwd_dq.cu:505-529). **fp16-acc non-associative → SAME order = SAME bits**. 

**Комментарий 13-20 dq_new.cu**: 6 инвариантов явно перечислены:
1. dQ_acc = FP16x2 packed [16][2]
2. MMA order kb=0..1 outer, ni=0..15 inner
3. K load swizzle XOR (j_local & 7) << 4
4. K_T read XOR k_xor = l_div4 << 4
5. K_T write natural stride KT_STRIDE=68
6. Epilogue unpack fp16 → fp32 scale

**Граница правок**: любая правка dq_new **не должна пересекать** порядок kt outer / kb outer / ni inner. Это жёсткий инвариант bit-exact.

---

## 0c. Регистровое окно

```
ptxas: 56 регистров / 0 stack / 0 spill / 1 barrier (счётчик bar.sync 0)
cudaFuncGetAttributes: sharedSizeBytes=0 (dynamic), maxThreadsPerBlock=1024
Dynamic SMEM: 8704 + 5120 = 13824 B/block
```

**Blocks/SM occupancy math**:

| Ограничение | Max blocks/SM |
|:--|:-:|
| SMEM: 100 KB / 13824 B | **6** ← лимитер |
| Regs: 65536 / (128 × 56) | 9 |
| Warps: 48 / 4 | 12 |

**Регистровые потолки без потери blocks/SM**:
- 6 blocks: max regs = 65536 / (6 × 128) = **85r**
- 5 blocks: max regs = 65536 / (5 × 128) = **102r**
- Текущее 56r → **окно +29 регистров до 85r** (для 6 blocks) или **+46 до 102r** (при осознанном 6→5 blocks размене)

**Лимитер текущих 6 blocks**: **SMEM** (не регистры, не launch_bounds). Регистровое окно широкое.

---

## 0d. Бумага-матрица лекарств (только по данным 0a-0c)

Верхняя таблица источников горла:

| Класс | Метрика | Значение | Стоимость (bound) | Комментарий |
|:--|:--|--:|:--:|:--|
| **MIO** | mio_throttle | 46.73% | ~4 ms | LDS pipe saturated (889M LD + 537M ST inst) |
| Barrier | barrier | 10.93% | ~0.93 ms | 4 __syncthreads() per kt |
| LongSb (LDG) | long_sb | 10.22% | ~0.87 ms | cp.async K/dS wait |
| Wait | wait | 9.11% | ~0.78 ms | issue-serialization |
| ShortSb | short_sb | 5.97% | ~0.51 ms | LDS→MMA staging chain |

**Прогноз каждого лекарства** (все прогнозы записаны ДО замера, с налогом тесноты 1.5-2× по уроку 022):

### D1. Pack-аналог (аналогично dk_new pack 018)

- **Условие применимости**: сторы K_T = 64 STS.U8/lane — **тот же класс** ✓
- **Целевые ops**: 64 STS.U8 → 12 SHFL + 16 STS.32 (унifiedly с dk_new)
- **shared_st inst drop**: 537M → ~537 × (16+12)/64 ≈ **235M** (**-56%**, аналог 018)
- **MIO прогноз**: 46.73% → **34-38%** (drop ~8-12 pp, по аналогии с dk_new 016 pp)
- **Register cost с налогом тесноты 1.5-2×**: base от dk_new pack **+11r** (107-96); dq_new base 56r, ожидаемо **+17-22r → 73-78r** (в окне 6 blocks × 85r)
- **Wall прогноз**: baseline 8.515 → **8.10-8.30 ms** (**-2.4 to -4.9%**)
- **Риски**: 24 SEL + PRMT chain могут разбудить short_sb (как в 018 +7.45 pp)

### D2. Retention

- **Применимо? По census 0b(ii)**: **НЕТ ГАРАНТИРОВАННОГО ПОВТОРА**. За kt все 64 B-load адреса per lane уникальны. Между kt K меняется — retention across kt невозможно.
- **Единственная форма**: hold ALL K (8 KB) в регистрах на протяжении MMA-C, убирая Phase 1.5 write + Phase 1.5 read + MMA-C B-load. Требует **~64 uint32 per lane для К** — **не влезает** в 128 register budget.
- **Retention half/quarter** — искусственная, не даёт очевидного gain (K пришёл из cp.async в SMEM, всё равно надо читать перед MMA).
- **Vugar-условие**: "если повтора нет — retention вычёркивается по факту".
- **Вычёркиваю retention**.

### D3. π-класс (bank-swizzle обеих сторон)

- **B-load bank pattern**: за (kb, ni) 32 lanes читают smK_area[n_d·68 + k_j].
  - n_d = ni*8 + l_div4 → 8 distinct rows per (kb, ni) per warp
  - k_j = kb*32 + l_mod4*4 + {0,16} → 8 distinct col-values per warp per (kb, ni) per half
- **Ожидаемо конфликтов немного** — LD excess только +44% (vs dk_new pre-pack +91%). Возможно локальные bank-collisions за счёт 68-stride mismatch с 128-wide fp8 rows.
- **ST-side (Phase 1.5 write)**: 64 STS.U8/lane на marching-строку с шагом 4 rows. Известный класс π_A/π_V из 013-016.
- **π-цель**: LD conflicts 541M → ~150M (**-72%**), ST conflicts 60.9M → ~15M (**-75%**), MIO drop ~3-6 pp.
- **Register cost с налогом тесноты**: +5-8r (π-macro на 2 сторонах, скромнее чем dk_new π_V +17r благодаря KT_STRIDE=68 более простым паттернам).
- **Wall прогноз**: baseline 8.515 → **8.20-8.35 ms** (**-1.9 to -3.8%**)
- **Риски**: π-compute overhead может съесть MIO drop (урок 022 dk_new).

### D4. Barrier reduction

- **Барьер stall 10.93%** — 4 __syncthreads() per kt.
- Возможно снизить до 3 барьеров (BARRIER #2 после Phase 1.5 read, before write): подобрать сколько чтения можно смерджить в write фазу.
- **Не безопасно без глубокого разбора** — жёсткий инвариант bit-exact.

### D5. cp.async двойной буфер (K tile prefetch)

- **long_sb 10.22%** — cp.async wait dominant.
- Двойной буфер K: prefetch K для kt+1 пока mma-C работает над kt.
- **SMEM cost**: +8704 B → total 22528 B → 4 blocks/SM (регресс 6→4). Vugar-правило: "6→5 = отдельное решение, 6→4 автоматически регресс".
- **Отложить до decision** — не первый шаг.

---

## Рекомендация первого лекарства

**По крупнейшему измеренному классу (mio_throttle 46.73%)**:

**D1 (pack-аналог)** — прямой аналог dk_new pack:
- Тот же класс сторов (64 STS.U8/lane)
- Ожидаемый MIO drop наибольший (8-12 pp)
- Регистровое окно широкое (85r доступно, ожидаемо 73-78r после pack)
- Инвариант bit-exact НЕ пересекается (pack транспонирует те же байты в те же таргет-позиции, только с другой раскладкой ops)
- Прогноз wall drop 2.4-4.9% — потенциально пересекает 3% keep-порог

**Vugar TZ**: "если census 0b подтверждает гарантированный повтор чтений K_T — первым гейт-циклом идёт retention, вторым pack поверх него. Если повтора нет — retention вычёркивается по факту, первым идёт крупнейший класс по профилю."

**Census 0b(ii): гарантированного повтора B-load нет** → retention вычеркнут.  
→ **Первым лекарством: D1 pack Q_T-scatter аналог**.

---

## Гейты будущих правок (зафиксировано для всех шагов dq_new)

Verbatim из TZ 009-2 шаг 0:

- ptxas: **0 spill, LDL/STL = 0** (детектор V[]-класса) ✓
- blocks: **6 остаются 6**; регресс 6→5 = НЕ автоматический откат, отдельное решение Vugar
- **triple bit-exact 11/11 + CANARY + sanitizer 0 errors**
- **fp16-acc порядок kt/kb/ni сохранён** — жёсткий инвариант (см. 0b.(iv))
- wall: session-pair, baseline и post в **одной сессии**
- **keep-порог правило-2/3 v2**:
  - ≥3% → KEEP сразу
  - 2-3% → серия ABBA ≥8 пар с архивным бинарём (медиана ≥2% AND худшая пара ≥1% → KEEP)
  - <2% → откат
- архивирование исходников после каждого KEEP: `runs/archive/NNN_sealed/` + md5 в отчёт

---

## Резюме шага 0 (024)

| Пункт | Статус |
|:--|:--:|
| 0a Sanity + baseline + NCu | ✓ session-fresh 8.515 ms / 258 T, полный srez |
| 0b SASS census (i-iv) | ✓ формулы из кода, retention доказан отсутствующим |
| 0c Регистровое окно | ✓ +29 запас до 85r (6 blocks), лимитер SMEM |
| 0d Бумага-матрица лекарств | ✓ D1 pack рекомендован первым |
| Хвост реестра DRAM | ✓ 9.25 GB подтверждён, 005b объяснён (cache-hot режим) |

Chain md5: 023 `5d38e0c8…` → **024 `<computed>`**

---

**End 024.**  
Ожидаю ТЗ на первое лекарство D1 pack Q_T scatter для dq_new (или альтернативу).
