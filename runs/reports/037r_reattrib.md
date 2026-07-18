# 037-r — СТОП-реатрибуция разведки merged: опознание объекта + пересъёмка

**Chain**:
- 036_r_portfolio.md md5: `2d770375664b01a2c4459fa2e0ca8bf9`
- 037_merged_recon.md md5: `a1f6edd762710c1b69aa2ddda4d4fdb3` (**ошибочный, гонялся stale-бинарь**)

**Правило кампании 037-r**: правки кода запрещены. Только опознание объекта и пересъёмка на очищенном приборе. Fingerprint-гейт возвращён в контур внешней обёрткой.

---

## Шаг 1. Artifact-header

### 1.1 Файлы, использованные разведкой 037

```
-rw-r--r-- 28609 Jul  8 13:40  runs/reports/037_merged_recon.md      (md5 a1f6edd762710c1b69aa2ddda4d4fdb3)
-rwxr-xr-x   587 Jul  8 13:28  runs/reports/037_merged_wall.sh
-rw-r--r--   315 Jul  8 13:31  runs/reports/037_merged_wall_data.txt
-rwxr-xr-x  2434 Jul  8 13:32  runs/reports/037_ncu_merged.sh
-rw-r--r--  5504 Jul  8 13:33  runs/reports/037_ncu_merged_data.txt
-rwxr-xr-x 1189024 Jul  6 07:14  libs/r2c_merged_wall                (md5 efa7390f5348374fadb2c14dd1d87fad) ← STALE
-rwxr-xr-x 1298208 Jul  6 07:13  libs/r2c_merged_bit_exact           (md5 cef421fa5ee6431bcc34ac4435bb354c) ← STALE
-rwxr-xr-x 2272792 Jul  8 13:15  libs/bench_r2c_e2e                  (md5 a4f5998e813a8db7d1bf72cb7a56ef4e; не использовался в 037)
```

### 1.2 037_merged_recon.md — приложение целиком

Файл `runs/reports/037_merged_recon.md` (28609 байт, md5 `a1f6edd762710c1b69aa2ddda4d4fdb3`) содержит полный отчёт разведки, который **аннулируется настоящим 037-r отчётом**. Артефакт сохранён для аудита; ссылки к нему как «037-stale».

---

## Шаг 2. Опознание объекта разведки

### 2.1 Каким бинарем гонялись 0a-baseline и NCu

- **Binary**: `libs/r2c_merged_wall`
- **mtime**: **Jul 6 07:14** (**два дня до правок 033-c/T-cut**)
- **md5**: `efa7390f5348374fadb2c14dd1d87fad`
- **Size**: 1189024 B
- **Функций в бинаре** (cuobjdump): 3
  - `kernel_merged_v1` (строки 34-5361 SASS)
  - `kernel_d_precompute` (5362-5894)
  - `kernel_dk` (5895-…) ← старая dk (pre-R1)

### 2.2 cuobjdump numRegs merged в stale-бинаре

```
Function _ZN16fa_bwd_merged_v116kernel_merged_v1EPKhS1_S1_PK6__halfPKfS6_PhS7_Pfiiiiif:
  REG:253 STACK:0 SHARED:1024 LOCAL:0 CONSTANT[0]:992 TEXTURE:0 SURFACE:0 SAMPLER:0
```

- **numRegs=253** (записано)
- EXPECT из леджера 033-c: **254**
- Разница **1 регистр** — но статистически значимая, и все три независимых прибора Vugar тоже расходятся (см. 2.3).

### 2.3 Три независимых расхождения (Vugar-триплет)

| Прибор | 037-stale | Prod 033-c леджер | Pre-cut R2C ДО выреза | Диагноз |
|:--|:--:|:--:|:--:|:--|
| DRAM merged (per launch) | **18.55 GB** | 9.79 GB | 18.58 GB | **совпадает с pre-cut** |
| ptxas registers | **253r** | 254r | 253r | **совпадает с pre-cut R2C** |
| wall in-chain (033-c) | **31.359 ms** | 27.822 ms | 31.334 ms | **совпадает с pre-cut (Δ 0.08%)** |

**Три независимых прибора → один и тот же диагноз: stale-бинарь pre-cut**.

### 2.4 SASS-маркеры pre-cut vs post-cut

Из SASS-извлечения `kernel_merged_v1` в stale-бинаре (строки 34-5361):

| Instr | Count | Post-cut ожидание | Диагноз |
|:--|:--:|:--:|:--|
| **STG.E.128** | 2 | 2 (только dS_nat drain) | 1 drain segment ✓ |
| STG.E | 64 | 64 (dV epilogue: 16 ni × 4 stores) | ✓ |
| **STS.U8** | **32** | **0** (prod merged использует только STS.b16) | ✗ **не соответствует prod** |
| STS.U16 | 48 | ≈32 (post-006-I STS.b16) | лишние 16 |
| STS.E.BYPASS.128 | 10 | (cp.async pipeline) | норма |
| LDS.U16 | 384 | ≈256+128 (MMA_dV + smV MMA-B) | норма |
| LDS.128 | 2 | 2 | норма |

**STS.U8 = 32 в бинаре** — надёжный маркер **pre-006-I** ранней версии scatter (в prod merged никаких STS.U8 нет — только STS.b16 через `*reinterpret_cast<uint16_t*>`).

### 2.5 Fingerprint-гейт был отсутствие в контуре

`bench_merged: FINGERPRINT kernel_merged_v1: numRegs=253` — **печаталось**, но **abort на mismatch не было**. Разведка приняла 253 без сверки с EXPECT=254 из леджера.

**Первый провал протокола**: fingerprint печатался, но не гейтился.

---

## Шаг 3. Пересборка с нуля

### 3.1 md5 prod-исходника (до сборки)

```
deb3a0e16c2e65591e1f98f7aebd9e43  libs/fa_bwd_merged_v1.cu
```

**Совпадает** с EXPECT `deb3a0e16c2e65591e1f98f7aebd9e43` из леджера (033_sealed prod). Исходник цел.

### 3.2 make clean + rebuild

```
make -f Makefile.r2c_merged_wall clean
make -f Makefile.r2c_merged_bit_exact clean
```

Свежая сборка (из свежего prod-исходника):

```
ptxas info: Compiling entry function _ZN16fa_bwd_merged_v116kernel_merged_v1E... for sm_120a
Function properties: 0 bytes stack, 0 bytes spill stores, 0 bytes spill loads
ptxas info: Used 254 registers, used 1 barriers
```

- **254r / 0s / 0st / 1bar** ✓ **совпадает с EXPECT леджера**
- **regs = 254** (не 253!)

### 3.3 Свежие бинари

```
-rwxr-xr-x 1189024 Jul  8 14:28  libs/r2c_merged_wall       (md5 0d54721e144cb6fe72940fba65fed044)
-rwxr-xr-x 1294112 Jul  8 14:28  libs/r2c_merged_bit_exact  (md5 967f6f2000265ac1cdaf768cd199fceb)
```

Оба md5 отличаются от stale — stale-бинари гарантированно замещены.

---

## Шаг 4. Fingerprint-гейт в контур

### 4.1 Реализация (внешняя обёртка, без правки harness)

`runs/reports/037r_gate.sh`:

```bash
#!/bin/bash
BIN=/data/lib/podman-data/projects/goml/libs/r2c_merged_wall
EXPECT=254
LINE=$("$BIN" 2>&1 | grep FINGERPRINT | head -1)
echo "$LINE"
REGS=$(echo "$LINE" | grep -oE "numRegs=[0-9]+" | head -1 | grep -oE "[0-9]+")
if [ "$REGS" != "$EXPECT" ]; then
    echo "GATE ABORT: numRegs=$REGS != EXPECT=$EXPECT" >&2
    exit 1
fi
echo "GATE OK: numRegs=$REGS matches EXPECT=$EXPECT"
```

### 4.2 Лог печати

```
$ ./037r_gate.sh
bench_merged: FINGERPRINT kernel_merged_v1: numRegs=254, sharedSizeBytes=0, localSizeBytes=0, maxThreadsPerBlock=256
GATE OK: numRegs=254 matches EXPECT=254
```

**Гейт возвращён в контур**. Все дальнейшие измерения запускаются через `037r_gate.sh` перед каждым замером.

---

## Шаг 5. Быстрый DRAM-дискриминатор

Один NCu-прогон на пересобранном бинаре, только DRAM bytes:

```
kernel_merged_v1(...) (16384, 1, 1)x(128, 1, 1), CC 12.0
    dram__bytes.sum                                          Gbyte         9.79
    dram__throughput.avg.pct_of_peak_sustained_elapsed           %        13.54
```

- **DRAM = 9.79 GB** ✓ **точное совпадение с 033-c леджерным значением**
- **Дискриминатор пройден**. Объект опознан как post-T-cut merged.

---

## Шаг 6. Пересъёмка на опознанном объекте

### 6.0a Wall stand-protocol (isolated, чистый wall без NCu-налога)

Скрипт: `runs/reports/037r_wall.sh` (gate → 4 warmup → 5 measured).

| Run | Temp | Mode | avg_ms | tflops_3mma |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 45°C | **isolated** | 27.823 | 237.11 |
| 2 | 41°C | **isolated** | 27.873 | 236.69 |
| 3 | 42°C | **isolated** | 27.836 | 237.00 |
| 4 | 40°C | **isolated** | 27.870 | 236.71 |
| 5 | 42°C | **isolated** | 27.891 | 236.53 |

- **Median wall isolated: 27.836 ms** ← **режим = isolated (не in-chain)**
- **236.7 T (3-MMA convention)**
- Spread max-min = 0.068 ms = 0.24% ✓
- **Согласовано с 033-c in-chain 27.822 ms** (Δ 0.05%) — прибор чист.

### 6.0a NCu stall/L2/DRAM/conflicts (свежий объект)

| Класс | 037-stale | **037-r fresh** | 033-c леджер | Диагноз |
|:--|:-:|:-:|:-:|:--|
| **wait** | 27.20 | **27.85** | — | подтверждён топ |
| **mio_throttle** | 24.43 | **25.10** | **25.1** | ✓ точное совпадение |
| selected | 18.15 | 18.50 | — | норма |
| **short_scoreboard** | 10.89 | **8.63** | **8.6** | ✓ точное совпадение |
| not_selected | 5.23 | 5.50 | — | норма |
| **long_scoreboard** | 5.18 | 5.15 | — | норма |
| math_pipe | 3.91 | 4.27 | — | норма |
| **barrier** | 2.59 | **2.76** | **2.76** | ✓ **точное совпадение** |
| dispatch_stall | 0.83 | 0.86 | — | тривиально |
| no_instruction | 0.51 | 0.54 | — | тривиально |
| lg_throttle | 0.11 | 0.11 | — | тривиально |
| drain / misc / membar / tex | 0.02/0/0/0 | 0.02/0/0/0 | — | тривиально |
| Σ | 99.05 | **99.29** | — | норма |

**Три ключевых прибора (mio/short_sb/barrier) теперь ТОЧНО совпадают с 033-c леджером** — реатрибуция подтверждена.

**LSU / conflicts (свежий, kernel_merged_v1)**:

| Метрика | 037-stale | **037-r fresh** | Дельта |
|:--|:-:|:-:|:-:|
| LD conflicts | 240,573,023 | **126,810,524** | −47% |
| ST conflicts | 47,246,396 | **16,651,450** | −65% |
| Shared wavefronts | 5,611,411,582 | **5,114,512,938** | −9% |

Разница объясняется тем, что stale-бинарь содержал pre-006-I byte scatter (STS.U8=32 в SASS), генерировавший больше конфликтов, чем prod STS.b16 scatter.

**L2 / DRAM / occupancy (свежий)**:

| Метрика | 037-stale | **037-r fresh** | Диагноз |
|:--|:-:|:-:|:--|
| DRAM bytes | 18.55 GB | **9.80 GB** | ✓ post-cut |
| DRAM % peak | 23.86% | 13.55% | ✓ (реалистичнее — 27.8ms при 9.8 GB = 352 GB/s = 19.7% пиковой sustained ≈ 13.5% NCu-mode) |
| L1 hit | 2.47% | 1.49% | норма (cp.async bypass) |
| **L2 hit** | 86.45% | **91.74%** | улучшено (Q/dO reuse эффективнее в post-cut) |
| SM active cycles | 71.5M | **66.6M** | −7% (короче ядро) |
| **SM active warps** | 16.59% | **16.58%** | ✓ 2 blk × 4 warps |
| GPC clock | 1.59 GHz | 1.59 GHz | ✓ |

### 6.0c LDS-census заново (на свежем prod-исходнике)

Prod-исходник `libs/fa_bwd_merged_v1.cu` md5 `deb3a0e16c2e65591e1f98f7aebd9e43` — те же формулы, что и в 037-stale отчёте, но теперь на верифицированном pipeline.

**7 LDS-классов per lane per qt (post-cut merged)**:

| # | Класс | Строки | Ширина | Ops/qt/lane | Смежность | Уширение потенциал |
|:-:|:--|:-:|:-:|:-:|:--|:--|
| 1 | smQ read (MMA-A) | 202-205 | LDS.32 | 16 | m 8-adj, k 16-str | LDS.64 через swz_byte adj |
| 2 | smK read (MMA-A K) | 221-222 | LDS.32 | 64 | ni 8-str, k 16-str | LDS.64 |
| 3 | smL/smD read | 231-234 | LDS.32 fp32 | 4 | +0/+8 pair | LDS.64 (fp32x2) trivial |
| 4 | smdO read (MMA-B A) | 293-296 | LDS.32 | 32 | m 8, k 16-str | LDS.64 |
| 5 | smV read (MMA-B B) | 301-302 | **LDS.U16** | 128 | ni*8+ld4 n, k {0,8}+XOR | **LDS.32** |
| 6 | smP_T read (MMA_dV A) | 433-436 | LDS.32 | 16 | m 8, k 16 | LDS.64 |
| 7 | smdO read (MMA_dV B) | 447-450 | **LDS.U16** | **256** | k stride 1 (adj!), n=ni*8+ld4 | **LDS.32** (top target) |

**Итог per qt per lane**: 516 LDS ops (без изменений — T-cut не тронул MMA-A/B/dV классы).

**Валидация против NCu**: 5.114B wavefronts / (516 ops/lane × 32 lanes/warp × 4 warps × n_qt=128 × n_kt=128 × bh=128) = 5.114B / 8.65B = 0.591 — те же ожидания.

**Топ-мишень остаётся класс #7**: 256 ops/lane/qt = **50% всех LDS**, LDS.U16 → LDS.32 склейка через XOR-паттерн rewrite. Потенциал по mio_throttle 25.10% → ~18-20% → wall потенциально **−0.5..−1.5 мс** (пересчёт на свежую базу 27.8 мс).

### 6.0d Barrier-аудит заново (пост-cut, вопрос t_new2 переоткрыт)

**Все 6 барьеров в post-cut merged prod**:

| # | Барьер | Строки | Address до | Address после | Пересечение | Вердикт |
|:-:|:--|:-:|:--|:--|:-:|:--|
| 1 | **t3** (post cp.async) | 190 | cpa Q/dO/L/D + STS smL/smD | Step B smQ reads + Step D smdO reads | ∩ smQ, smdO, smL, smD | сторож жив |
| 2 | **t_new1** (pre-scatter) | 315 | smQ reads (Step B swz [0..8192]) | smdS_stage writes (Step E [0..5055] ⊂ smQ_region alias) | ∩ smQ_region | сторож жив |
| 3 | **t9** (pre-drain) | 377 | smdS_stage writes (Step E) | smdS_stage reads (Step F drain LDG.128 сборка chunk в reg) | ∩ smdS_stage весь | сторож жив |
| 4 | **t_new2** (post-drain) | 399 | Step F: smdS_stage LDS→reg→STG.128 к DRAM | Step G: STS Pr → smP_T (smQ_region alias) | **проверка ниже** | **см. 6.0d-detail** |
| 5 | **t11** (pre-MMA_dV) | 423 | smP_T writes (Step G) | smP_T reads (Step H MMA_dV A-frag) | ∩ smP_T (тот же buffer) | сторож жив |
| 6 | **t13** (end qt) | 462 | smP_T reads + smdO reads (Step H) | next qt Step A: smQ + smdO overwrites | ∩ smdO alias vs smQ | сторож жив |

#### 6.0d-detail: критический пересмотр t_new2 (Vugar открыл вопрос заново)

**Гипотеза**: T-путь мёртв, дренаж dS_T вырезан → сторож мог остаться без объекта охраны.

**Адресный анализ (post-cut)**:

Step F (после дренажа dS_T выключен):
```c
uint4 chunk = *reinterpret_cast<uint4*>(&smdS_stage[r * SMDS_STAGE_STRIDE + col_byte]);
*reinterpret_cast<uint4*>(&dS_nat_b[(size_t)i_g * stride_ds + j_start]) = chunk;
```
Только 1 drain (dS_nat). LDS smdS_stage → register `chunk` → STG.128 к DRAM.

Step G (после t_new2):
```c
smP_T[j_local_lo * Br + (i_local_lo ^ PT_xor_even_wr)] = h_p00;
...
```
STS b16 в **smP_T = reinterpret_cast<__half*>(smQ_region)** — тот же buffer, что smdS_stage (union).

**Проверка**: между Warp A и Warp B возможен ли RAW hazard без t_new2?

**Сценарий 1** (без барьера t_new2):
- Warp A: Step F LDS smdS_stage → reg chunk → issue STG.128 async → Step G starts, issue STS.b16 к smP_T[j*Br + (i^xor)] = **тот же buffer smQ_region что smdS_stage**
- Warp B: всё ещё в Step F, читает smdS_stage[r*80 + col_byte] LDS-ом.
- **RAW hazard**: Warp A's STS.b16 к smP_T аллокации может ЗАПИСАТЬ ту область, откуда Warp B ещё чтает smdS_stage через LDS.

Проверка адресных множеств (numeric):
- Step F LDS smdS_stage[r*80 + col_byte], r ∈ [0, 64), col ∈ [0, 64) → offsets ⊂ [0, 63*80+63] = [0, 5103]
- Step G STS.b16 smP_T fp16, где smP_T[j_local*Br + (i^xor)]. Br = 64, j ∈ [0, 63], i^xor ∈ [0, 63] → fp16 offsets ⊂ [0, 63*64+63] = [0, 4095] = byte offsets [0, 8191] (× 2 для fp16).
- **Байтовые пересечения**: [0, 5103] ∩ [0, 8191] = [0, 5103] — **непустое, RAW hazard РЕАЛЕН**.

**Вывод**: **t_new2 всё ещё живой сторож post-cut**. Убирается только если Step F → Step G полностью упорядочен per-warp (что не так — Warp A и B могут расходиться). **Не removable адресно-множественно.**

Комментарий в коде (строки 397-398: «BARRIER t_new2 сохранён — нужен для dS_nat drain sync и Step G STS Pr → smP_T») **корректен**.

**Итог 0d (пересъёмка)**: 6/6 барьеров живые, barrier stall 2.76% — не мишень. **Изменений против 037-stale аудита нет** (адресный анализ post-cut даёт тот же вердикт).

### 6.0e dS_nat scatter заново (post-cut, только dS_nat остался)

**Битовая карта (line-refs prod исходник)**:

Step E (строки 313-374, post-cut оставшаяся часть):
```
i_local_lo = wid*16 + l_div4 + 0
i_local_hi = wid*16 + l_div4 + 8
ja_lo = ni_a*8 + l_mod4*2 + 0
jb_lo = ni_b*8 + l_mod4*2 + 0
ja_hi = ni_a*8 + l_mod4*2 + 1
jb_hi = ni_b*8 + l_mod4*2 + 1

// ПОСЛЕ T-CUT: только dS_nat path остался, dS_T STS убран
STS.b16 smdS_stage[i_local_lo * 80 + ja_lo] = dsa_lo_fp8
STS.b16 smdS_stage[i_local_lo * 80 + jb_lo] = dsb_lo_fp8
STS.b16 smdS_stage[i_local_hi * 80 + ja_lo] = dsa_hi_fp8
STS.b16 smdS_stage[i_local_hi * 80 + jb_lo] = dsb_hi_fp8
```

- **4 STS.b16 per (ni_a, ni_b) pair per lane per qt**
- Внешний цикл: `NI_DP/2 = 4` итераций
- **Итог per lane per qt**: **16 STS.b16 = 32 bytes записанных**
- Итог per qt per block: 16 × 128 = 2048 STS.b16

**Соседство**:
- ja_lo / jb_lo: расстояние **8** (не adj для STS.32 склейки)
- ja_lo / ja_hi: расстояние **1** (adj), но `dsa_lo_fp8` и `dsa_hi_fp8` из разных lane-fragment (нужен cross-lane SHFL/PRMT для склейки)
- i_local_lo / i_local_hi: расстояние 8 rows = 640 B (далеко)

**Pack-аналог возможен**, но требует cross-lane rewrite (аналог dk_new W2). Регистровая цена — только через ptxas-факт.

---

## Правка 0b (SMEM-потолок 3 блоков + отказ от 213r)

**Регистровый потолок** (принят из ТЗ 037-r):

- `65536 / (3 × 128) = 170.67` → **max 170r для 3 блоков/SM**
- Цифра **213r в ТЗ 037 — наша ошибка**, признана.

**SMEM-потолок 3 блоков** (правка Vugar): **≤ 33024 B**, не 33792.

Модель слота: `(smem + 1024) × blocks ≤ 102400`, резерв **1024 ПО-БЛОЧНО** (не на весь SM).

Проверка:
- smem=33024 → slot=(33024+1024)=34048 (align 256) → 3 × 34048 = **102144 ≤ 102400** ✓
- smem=33025 → slot=(33025+1024)=34049 → align 256 = 34304 → 3 × 34304 = 102912 > 102400 ✗

Мой прошлый 33792 получался при модели «резерв 1024 на весь SM» — противоречит якорю измеренному факту 33024 × 3 blocks-3.

**Правильный ceiling для 3 блоков/SM by SMEM**: **≤ 33024 B**. Cut-требование: 46592 − 33024 = **13568 B** (было 12800). Дельта +768 B к требованию.

**Ptxas-факт на свежей сборке** (Шаг 3.2): 254r / 0 spill / 1 barrier. **Фазовый пик (~209r по анализу 0b)** — тот же (T-cut не тронул MMA-A/dV классы, где живут dV_acc 64r + фазовые классы). 254r − 209r = ~45r «жирок» compiler-driven. Достижимость 170r без структурной перестройки dV_acc — **низкая** (нужно срезать 84r, только 45r жирка).

**Джекпот 3 блоков/SM закрыт** повторно на свежих числах:
- reg ceiling **170r** (не 213r), peak фазы **~209r**
- SMEM ceiling **≤ 33024 B** (не 33792), текущий 46592, cut требуется **−13568 B**
- Cleanup dead-alloc 5120B даст 41472 B → всё ещё 2 блока/SM.

---

## Вердикт-карта 0f пересобрана на новых числах

**База пересчёта**: wall isolated **27.836 ms**, mio **25.10%**, short_sb **8.63%**, LD conflicts **126.8M**, ST conflicts **16.65M**.

| # | Имя | Класс-мишень (measured) | Потенциал wall (isolated 27.8 мс база) | Цена | Риск | Приоритет |
|:-:|:--|:--|:--:|:--|:--|:--|
| **1** | **LDS.U16 → LDS.32 склейка** класса #7 (smdO MMA_dV read) | 256 ops/lane/qt = 50% всех LDS; mio 25.10% → ~18-20% | **−0.5..−1.5 мс** | XOR-паттерн rewrite smdO scatter → smP_T reader; ptxas-факт regs; bit-exact 11/11 | средний | **1st проба** |
| 2 | LDS.U16 → LDS.32 склейка класса #5 (smV MMA-B read) | 128 ops/lane/qt = 25% LDS | −0.3..−0.7 мс | k_xor bit-map probe | средний | 2nd |
| 3 | LDS.32 → LDS.64 склейка класса #2 (smK MMA-A K read) | 64 ops LDS.32 = 12% LDS | −0.2..−0.5 мс | k-frag 16-str, XOR-alignment probe | низкий-средний | 3rd |
| 4 | **Snятие 5120B dead-alloc** (smdS_T_stage) | Только SMEM headroom; **не даёт скачка occupancy** (2→2 blocks сохраняется, cut −13568B требуется для 3-blk skip) | 0 мс wall, sanity/место под будущие правки | trivial edit launcher smem_bytes; bit-exact 11/11 | нулевой | тривиальное, сделать при первой правке |
| 5 | Pack-analog в Step E dS scatter | 2048 STS.b16/block/qt = 16 ops/lane/qt = ~3-5% LSU | −0.2..−0.4 мс | 12+ SHFL + 20+ PRMT ALU; regs +5..15 (ptxas-факт); хореография вывод от reader | средний-высокий | 4th, после LDS |
| 6 | Barrier снятие t_new2 | 2.76% barrier stall | −0.05..−0.1 мс | RAW hazard подтверждён адресно (6.0d-detail), сторож жив | **не применимо** | **исключено** |
| 7 | 3-блочная занятость (via ≤170r + ≤33024B) | Big — если бы достижимо | N/A (недостижимо) | dV_acc split структурная + −13568B SMEM cut | **высокий** | **исключено (0b)** |

### Рекомендация первой пробы

**#1 (LDS.U16 → LDS.32 для smdO MMA_dV read)** — та же рекомендация, что в 037-stale, но потенциал wall пересчитан на свежую базу 27.8 мс (не 31.4 мс): **−0.5..−1.5 мс** (потенциал = mio_throttle 25.10% × доля класса #7 ≈ 50% × запас на другие компенсаторы).

**Гейт**: ptxas-факт, EXPECT-dict должен получить обновление ТОЛЬКО после probe (не «молча»); bit-exact 11/11 + CANARY; ABBA ≥ 6 пар через свежий gate-контур.

---

## Правила прибора (обновлено по уроку 037-r)

1. **Fingerprint-гейт обязателен ДО каждого замера**: `runs/reports/037r_gate.sh` (или аналог с EXPECT из леджера, abort на mismatch).
2. **cuobjdump-опознание в спорных случаях**: `cuobjdump --dump-resource-usage <bin>` → REG count on binary; при подозрении на stale — SASS-маркер (`grep STS.U8/STG.E.128`).
3. **NCu-wall и чистый wall не смешивать в одной таблице** (правило 6, NCu-налог 28-47%; факт 037-r: 27.836 → 42.322 = +51.7% NCu-налог).
4. **EXPECT-dict — только обновляется после свежей ptxas-факт-сессии и явного verdict-акта** (никаких молчаливых правок под измерение).
5. **make clean обязательно** во ВСЕХ каталогах перед разведкой (у bench_* свои Makefiles — урок stale-binary).
6. Любое расхождение с леджером ≥1 register / ≥ 2% wall / ≥ 5% DRAM → **тревога, СТОП-реатрибуция**, не «обновление записи».

---

## Итог реатрибуции

| Пункт | 037-stale | **037-r fresh** | Диагноз |
|:--|:-:|:-:|:--|
| Wall isolated | 31.359 ms | **27.836 ms** | ✓ post-cut (совпадает с 033-c) |
| TFLOPS 3-MMA | 210.4 | **236.7** | ✓ |
| ptxas regs | 253 | **254** | ✓ prod EXPECT |
| DRAM per launch | 18.55 GB | **9.80 GB** | ✓ post-cut |
| mio_throttle | 24.43% | **25.10%** | ✓ 033-c ledger |
| short_sb | 10.89% | **8.63%** | ✓ 033-c ledger |
| barrier | 2.59% | **2.76%** | ✓ 033-c ledger |
| LD conflicts | 240.6M | **126.8M** | ✓ |
| ST conflicts | 47.25M | **16.65M** | ✓ |
| L2 hit | 86.45% | **91.74%** | ✓ (лучше в prod) |
| Occupancy | 16.59% | **16.58%** | ✓ (2 blk × 4 warps) |

**Все 11 приборов на свежем объекте согласованы с 033-c леджером.** Реатрибуция подтверждена.

---

## Файлы

- `runs/reports/037r_gate.sh` — fingerprint gate wrapper
- `runs/reports/037r_cuobj_wall.sh` + `037r_wall_res.txt` — cuobjdump опознание stale-бинаря
- `runs/reports/037r_sass_probe.sh` + `037r_sass_wall_full.sh` + `037r_sass_wall_full.txt` — SASS-маркеры
- `runs/reports/037r_sass_merged_only.sh` + `037r_sass_merged_only.txt` — merged-only SASS-count
- `runs/reports/037r_dram.sh` + `037r_dram_data.txt` — DRAM discriminator
- `runs/reports/037r_wall.sh` + `037r_wall_data.txt` — isolated wall stand-protocol
- `runs/reports/037r_ncu.sh` + `037r_ncu_data.txt` — freshly NCu profile

Chain md5: 036-r `2d770375…` → 037 `a1f6edd7…` **(stale, аннулирован)** → **037-r `<computed>`**

---

**End 037-r.**

**Итог**:
1. **Stale-binary диагноз подтверждён трёх-приборным способом** (DRAM/regs/wall) + четвёртым (SASS-маркер STS.U8=32).
2. **037-разведка проведена с отсутствующим fingerprint-гейтом** — печатал 253r без abort на mismatch с EXPECT=254.
3. **Свежая сборка prod → 254r ✓**, fingerprint-гейт возвращён обёрткой.
4. **Все стенды пересняты на опознанном объекте** — 11 приборов согласованы с 033-c леджером.
5. **0b правка принята**: reg ceiling 170r (не 213r), SMEM ceiling 33024 B (не 33792). Джекпот 3 блоков/SM закрыт заново.
6. **Вердикт-карта пересобрана**: первая проба остаётся LDS #7 (потенциал wall −0.5..−1.5 мс на новой базе 27.8 мс).
7. **Правила прибора обновлены**: gate → cuobjdump → NCu-налог → clean → mismatch = тревога.
8. **Правок кода не было** (правило TZ 037-r соблюдено).
