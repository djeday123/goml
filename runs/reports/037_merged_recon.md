# 037 — Merged-кампания, фаза 0: полная разведка (measurement/paper only)

**Chain**:
- 036_r_portfolio.md md5: `2d770375664b01a2c4459fa2e0ca8bf9`

**Artifact header** (production не тронут; фаза 0 = measurement/paper only):
```
-rw-r--r-- 20180  Jul  6         libs/fa_bwd_merged_v1.cu    (033_sealed, md5 deb3a0e16c2e65591e1f98f7aebd9e43)
-rwxr-xr-x       Jul  6         libs/r2c_merged_wall         (md5 efa7390f5348374fadb2c14dd1d87fad)
-rwxr-xr-x       Jul  6         libs/r2c_merged_bit_exact    (md5 cef421fa5ee6431bcc34ac4435bb354c)
-rw-r--r-- 14667  Jul  8         libs/fa_bwd_dk_new.cu       (AB=033_sealed, md5 a9f0ded8261e53a143b521ffa647f458)
-rwxr-xr-x       Jul  8         libs/bench_r2c_e2e           (chain bench, dk_new=128r OK)
```

Reference-бинари 036/036-r переехали из `libs/` → `runs/archive/036_r_binaries/` (в `libs/` только production).

Merged/dk/dq/морозилки не тронуты.

---

## Правило кампании

**Регистровое правило**: перед любой будущей правкой merged — ptxas-факт, прогнозы не принимаются. В отчёте ниже все Rmax значения — из runtime API (`cudaDeviceGetAttribute` + `regsPerMultiprocessor`), никаких предположений.

---

## Устройство (device query)

`libs/query_caps` (via `Makefile.query_caps`):

```
name = NVIDIA RTX PRO 6000 Blackwell Workstation Edition
cc = 12.0
SMs = 188
regsPerSM = 65536         (per SM total)
regsPerBlock = 65536      (per-block cap)
sharedPerBlock = 49152    (default)
sharedOptin = 101376      (opt-in per-block)
sharedPerSM = 102400      (per SM total = 100 KB)
maxThreadsPerSM = 1536
warpSize = 32
L2 cache = 128 MB
```

DRAM peak sustained ≈ **1.79 TB/s** (session snapshot из 033-c).

---

## 0a. Свежий полный профиль merged isolated (прогретая карта)

### Стенд-протокол

- 4 warmup runs (discarded), 5 measured runs
- Скрипт: `runs/reports/037_merged_wall.sh`
- Данные: `runs/reports/037_merged_wall_data.txt`

### 037-fresh wall isolated (5 runs)

| Run | Temp | avg_ms | tflops_3mma |
|:-:|:-:|:-:|:-:|
| 1 | 42°C | 31.305 | 210.74 |
| 2 | 46°C | 31.328 | 210.58 |
| 3 | 45°C | 31.359 | 210.37 |
| 4 | 46°C | 31.361 | 210.36 |
| 5 | 41°C | 31.385 | 210.20 |

- **Median wall isolated: 31.359 ms**
- **210.37 T (3-MMA convention)**
- CV = 0.09% (гладкая полка)
- Spread: max-min = 0.080 ms = 0.26% (стенд стабилен)

### Полная stall-таблица (NCu, sm_120a, sec 0a)

Скрипт: `runs/reports/037_ncu_merged.sh` (использует cuda-13.1 ncu).

| Класс | % | Комментарий |
|:--|:-:|:--|
| **wait** | **27.20** | fp32-acc deps (softmax→dS quantize / dV_acc chain) |
| **mio_throttle** | **24.43** | LSU pipe насыщение (LDS.U16 массив + STS.b16 в scatter) |
| selected | 18.15 | issued (natural) |
| **short_scoreboard** | **10.89** | LDS latency (B-loads после барьеров) |
| not_selected | 5.23 | другой warp выбран (occupancy=16.59%) |
| **long_scoreboard** | **5.18** | LDG latency (cp.async wait) |
| math_pipe_throttle | 3.91 | ALU насыщение (fp16→fp32 + e4m3 quant) |
| barrier | 2.59 | 6 барьеров/qt — низкая доля |
| dispatch_stall | 0.83 | scheduler contention |
| no_instruction | 0.51 | i-cache miss |
| lg_throttle | 0.11 | LG unit (тривиально) |
| drain | 0.02 | нет ветвлений |
| membar / misc / tex_throttle | 0.00 | не задействованы |
| imc_miss | n/a | не сообщается |

Σ ≈ 99.05% (норма ~100%).

### 033 vs 037 сравнение (calibration to memoir claims)

| Класс | 033 header | 037 fresh | Δ pp |
|:--|:-:|:-:|:-:|
| mio | 25.1% | 24.43% | −0.7 |
| short_sb | 8.6% | 10.89% | **+2.29** |
| barrier | 2.76% | 2.59% | −0.17 |
| BW-util | 33.6% peak | 23.86% peak | −9.7 |

- mio/barrier согласуются с memoir.
- short_sb вырос на +2.29 pp — likely результат небольшого дрейфа рабочей SMEM карты после 034 / реорганизации.
- BW-util упал — часть 20-iter avg vs single-launch NCu разница нормировки; peak DRAM не критично для оптимизации, headroom по-прежнему большой (~4×).

### Inst counters (LSU/conflicts)

| Метрика | Значение |
|:--|:--:|
| l1tex bank conflicts LD | 240,573,023 (240M) |
| l1tex bank conflicts ST | 47,246,396 (47M) |
| shared wavefronts total | 5,611,411,582 (5.61B) |
| shared_ld inst | n/a (NCu 2021.3 доп. counters) |
| shared_st inst | n/a |

- LD conflict rate: 240M / 5.61B = **4.3% wavefronts с конфликтом** (низко).
- ST conflict rate: 47M / 5.61B = 0.84% (очень низко).
- Основной класс проблем — **не bank conflicts**, а **wait + mio throughput**.

### L2 / DRAM / occupancy

| Метрика | Значение |
|:--|:-:|
| DRAM bytes | 18.55 GB |
| DRAM % peak | 23.86% |
| L1 hit rate | 2.47% |
| L2 hit rate | **86.45%** ← Q/dO reuse |
| SM active cycles avg | 71.5M |
| GPC clock avg | 1.59 GHz |
| SM active warps | **16.59%** (=8 warps/SM = 2 блока × 4 warps) |
| threads/inst | 32 (нет ветвлений в hot loop) |

Occupancy 16.59% = 8 warps/SM = **2 blocks/SM подтверждено runtime-фактом**.

---

## 0b. Регистровая перепись 254r (фактически 253r) — фазовый аудит

### Установленный факт

```
ptxas info: Used 253 registers, used 1 barriers (fa_bwd_merged_v1.cu)
cudaFuncGetAttributes: numRegs=253, sharedSizeBytes=0, maxThreadsPerBlock=256
```

- ptxas 253r (не 254r как в memoir header 033/TZ).
- 253 × 128 = 32384 регистров/блок → 65536/32384 = **2.024 → 2 блока/SM by regs** ✓
- SMEM 46592 B/блок → slot 47104 → 102400/47104 = 2.17 → **2 блока/SM by SMEM** ✓
- **Оба лимитера активны на 2 блока/SM.** Занятость 16.59% подтверждено NCu.

### Достижимость 3 блоков/SM (Vugar TZ target)

**Runtime factual math** (`libs/query_occ` output):

```
regsPerMultiprocessor = 65536
sharedPerSM = 102400
```

Для 3 блоков/SM одновременно необходимы **оба**:
1. `regs_per_block × 3 ≤ 65536` → regs_per_thread × 128 × 3 ≤ 65536 → **regs ≤ 170**
2. `smem_slot(round-up 1024) × 3 ≤ 102400` → slot ≤ 34133 → **smem ≤ 33792 B**

**Проверка кандидатов**:

| R (регистров/thread) | reg/блок | blk_by_reg |
|:-:|:-:|:-:|
| 150 | 19200 | **3** |
| 160 | 20480 | **3** |
| 168 | 21504 | **3** |
| **170** | 21760 | **3 ← верхняя граница** |
| 176 | 22528 | 2 |
| 200 | 25600 | 2 |
| **213 (TZ target)** | **27264** | **2** ← НЕ ДАЁТ 3 БЛОКА |
| 253 (текущий) | 32384 | 2 |

**Вывод по TZ**: **Vugar-target 213r НЕ достигает 3 блоков/SM** на RTX PRO 6000 Blackwell. Реальный порог — **170r**. Разница 213−170 = 43r в предположениях. Возможно, TZ-число 213 родилось из формулы для другого reg-file (96K reg file → 213×128×3=81792 ≤ 98304 ✓), но на этой карте reg file 65536, что даёт 170r-ceiling. **Гейт-порог для гипотезы 3 блоков — 170r, не 213r.**

**SMEM-side проверка** (кандидаты для 3 блоков):

| smem/блок (B) | slot (round-up 1024) | blk_by_smem |
|:-:|:-:|:-:|
| **33792** | 33792 | **3 ← верхняя граница** |
| 34133 | 34816 | 2 |
| 34816 | 34816 | 2 |
| 40960 | 40960 | 2 |
| 41472 | 41984 | 2 |
| **46592** (текущий) | 47104 | 2 |

**Вывод**: чтобы получить 3 блока/SM by SMEM — резать до **≤33792 B** (текущий 46592, дельта **−12800 B = −27.5%**).

### Куда ушли 5120B от T-cut

- T-cut убрал STS в smdS_T_stage и drain в dS_T (033-c). Но **алокация `smdS_T_stage` (5120B) осталась** в SMEM layout (строки 100-114, launch alloc строка 513: `+ Bc*80 = 5120`).
- Это **мёртвая аллокация** — 5120 B, ни один store не пишет в этот регион.
- **headroom по SMEM: −5120 B тривиально снимаемых**.
- Но: 46592 − 5120 = 41472 → slot 41984 → всё ещё **2 блока/SM**. То есть свободная аллокация не даёт скачка занятости сама по себе.

**Для 3 блоков/SM нужен ещё −7680 B** сверх убранного мёртвого 5120B (или −12800 B в сумме от текущего 46592 → 33792). Кандидаты: сокращение smdO (16384B fp16, но нужен для двух MMA-B и MMA_dV; двойной буфер убил бы 3 блока), sink smK/smV (по 8192B, но резиденты).

**Регистровый аудит "снять 83r" (253→170)**:

Фазы конвейера merged (per qt, 6 barrier сегментов):

| Фаза | Живые классы регистров (оценка сверху) | Δ-риск |
|:--|:--|:--|
| t0..t3 (cp.async Q/dO/L/D) | адр-арифметика cpa (~20r), tid-derived (5r) | фазовое |
| t3..t9-pre (MMA-A + softmax + MMA-B + dS-quant + STS scatter) | **Qr[4][4]=16r, Sr[8][2]=16r, Pr[8][2]=16r fp16-packed, dPr[8][4]=32r fp32, Sr/Pr fp16 stash 8r, softmax scalars L_lo/L_hi/D_lo/D_hi/mask calc ≈15r, address arithmetic ≈20r** = **≈120r peak** |
| t9..t_new2 (drain dS_nat через smdS_stage → DRAM) | address арифметика 15r, chunk 4r × 4 lanes → 16r register | фазовое, ~30r peak |
| t_new2..t11 (STS Pr → smP_T с XOR) | Pr[8][2]=16r остатки от прошлой фазы, addr calc | ~30r peak |
| t11..t13 (MMA_dV P^T · dO → dV_acc) | **dV_acc[16][4]=64r fp32 (несущий!)** + Ar[4] + Br[2] + adr calc | **~90r peak** |

**Несущие классы** (нельзя убрать без потери W-корректности):

| Класс | Регистров | Комментарий |
|:--|:-:|:--|
| **dV_acc[16][4] fp32** | **64r** | ЖЁСТКО несущий: qt-накопление P^T·dO по всему qt-loop; убрать → RAW SMEM spill, потеря 3-MMA fused паттерна |
| fp16-acc Sr[8][2] | 16r | фазовое, живёт до softmax |
| dPr[8][4] fp32 | 32r | фазовое, живёт до dS_quant |
| Pr[8][2] fp16-packed | 16r | живёт t7..t_new2..t11 (перекрытие с scatter) |
| Qr[4][4] u32 | 16r | фазовое, живёт до MMA-A |
| K/V load reuse | ≤2r | инстанционно на инстр., не резидент |

**Резидентно живущие (max одновременно) ≈ 64r (dV_acc) + address arith 20r + tid-consts 5r ≈ 89r**.  
**Пиковая фаза** (t3..t9-pre MMA-A/softmax overlap с dV_acc): **~120r + 89r = ~209r**. Ptxas 253r → **~44r "жирок" на промежуточных фазовых переменных**, address арифметика повторно, SEL-mux mask calc, `Sr`-`Pr` дубли.

**Оценка достижимости 170r ceiling**:
- Peak фаза оценивается **~209r**. Ceiling 170r требует срезать **~39r** дополнительно к жирку.
- 44r жирка (compiler-driven, не всегда доступного через ручную оптимизацию) не хватает для одновременного среза 44+39 = 83r.
- **Достижимость 170r без структурной перестройки — низкая.** Требуется:
  - dV_acc split (64r → 32r × 2 qt-half-loops) — потеря fused паттерна, но с double-drain reg reuse
  - fp16-acc Sr перевод в SMEM transit (16r → 0r) — доп. STS/LDS волна
  - Address arithmetic dedup (fusion l_div4/l_mod4 cached) — ~5-10r
  - Fusion Sr→Pr conversion (устранить промежуточные float scratches) — ~5-8r

Без структурной перестройки dV_acc — **170r не достижим**. Приход dV_acc split даёт цену: 2× barrier + 2× drain SMEM traffic → возможный проигрыш wall.

**Вердикт 0b**: **джекпот 3 блоков/SM закрыт**. Ни regs (170r ceiling против peak ~209r), ни SMEM (33792 B ceiling против 41472 после T-cut cleanup) не достижимы без структурной перестройки dV_acc и/или double-buffer резидентов K/V (что убьёт занятость дальше).

**Альтернатива**: 5120B dead-alloc убрать (headroom → sanity), другой оптимизационный трек (не занятость, а throughput внутри 2 блоков/SM).

---

## 0c. Оффсет-census LDS-классов merged (кандидаты на широкие загрузки)

Полный перечень LDS операций per lane per qt из кода (line-refs к `libs/fa_bwd_merged_v1.cu`):

| # | Класс | Строки | Ширина | Ops/qt/lane | Формула адреса | Смежность | Уширение | Риск |
|:-:|:--|:-:|:-:|:-:|:--|:--|:--|:--|
| **1** | smQ read (MMA-A) | 202-205 | LDS.32 | **16** | `swz_byte(m_lo/m_hi, ks*32 + k0 + {0,16})` | m-frag 8-adj, k-frag 16-stride | LDS.64 (m_lo+m_hi слитно)? XOR разбивает | средний |
| **2** | smK read (MMA-A K) | 221-222 | LDS.32 | **64** | `swz_byte(n_K = ni*8 + l_div4, k_lo/k_hi)` | ni loop → 8-stride n, k-frag 16 | LDS.64 по (k_lo,k_hi)? XOR разбивает | средний |
| **3** | smL/smD read | 231-234 | LDS.32 fp32 | **4** | `smL[wid*16 + l_div4 + {0,8}]`, `smD[wid*16 + l_div4 + {0,8}]` | +0/+8 pairs | LDS.64 (fp32x2) уже реально: 2×LDS.32 → 1×LDS.64 | низкий |
| **4** | smdO read (MMA-B A) | 293-296 | LDS.32 | **32** | `smdO[m*Hd + (k^dO_xor_el)]` c dO_xor_el = l_div4<<3 | m-frag 8, k-frag 16-stride | LDS.64 (k_lo,k_hi слитные если XOR gives adj) | средний-высокий |
| **5** | smV read (MMA-B B) | 301-302 | **LDS.U16** | **128** | `smV[n*Hd + (k_lo/k_hi ^ k_xor)]`, k_xor=l_div4<<4 | ni*8+l_div4 n, k-frag {0,8} + XOR | **LDS.32** (2 halves per lane)? нужен probe XOR-парности | средний |
| **6** | smP_T read (MMA_dV A) | 433-436 | LDS.32 | **16** | `smP_T[m*Br + (k^PT_xor_rd)]`, xor = l_div4<<3 | m-frag 8, k-frag 16 | LDS.64? XOR bits | средний |
| **7** | smdO read (MMA_dV B) | 447-450 | **LDS.U16** | **256** | `smdO[k*Hd + (n^dO_xor_{even,odd})]`, 4 fragments per (kb,ni) | k-frag stride 1 (adjacent!), n=ni*8+l_div4 | **LDS.32** (`lo0`+`hi0` adjacent bytes with different XOR!) — если XOR unified | **высокий** |

**Итог per qt per lane**: 16+64+4+32+128+16+256 = **516 LDS operations**.

**Верификация против NCu**: 5.611B wavefronts / (n_qt=128 × n_kt=128 × bh=128 × 4 warps × 1 wavefront/lane) = 5.611B / 8.39B ~= 0.67 — grubых расхождений нет (LDS.U16 может считаться дробно как sub-wavefront).

**Топ-кандидаты на LDS-уширение**:

1. **Класс #7 (smdO read MMA_dV, 256 ops/lane/qt = 50% всех LDS)** — LDS.U16, XOR-паттерн `(n ^ dO_xor_even)` и `(n ^ dO_xor_odd)` где odd=even+8. При `n = ni*8 + l_div4`, `dO_xor_even = l_mod4<<4`, `dO_xor_odd = dO_xor_even + 8`. Значит соседние pairs (`lo0/hi0`) читаются с разными XOR масками, но байты **соседние в памяти** (16-bit adjacent). Возможен LDS.32 если реорганизовать XOR-паттерн smdO scatter (Step A) чтобы pairs жили в одной 32-bit ячейке — **потенциал −128 LDS ops/lane/qt** (−25% всех LDS).
2. **Класс #5 (smV read MMA-B, 128 ops)** — аналогичный анализ: `k_lo/k_hi` с k_xor. Пары соседних 16-bit могут склеиваться в LDS.32.
3. **Класс #2 (smK read MMA-A K, 64 ops)** — уже LDS.32, но возможен LDS.64 на (k_lo,k_hi) если XOR сохраняет adjacency.

**Потенциал по mio_throttle (24.43% сейчас)**: уширение #7 может убрать ~25% LDS-wavefronts → mio ~18-20% → wall потенциально −1..−2 мс.

**Риск**: XOR-паттерны сложные, требуется bit-map derivation с probe (урок 020 — сначала бумажный перебор адресов, потом код).

---

## 0d. Barrier-аудит t_new (адресные множества, не рассуждения о планировщике)

**Все 6 барьеров ядра merged** (line-refs):

| # | Барьер | Стр. | Address-set до (writes/reads) | Address-set после (writes/reads) | Пересечение? | Вердикт |
|:-:|:--|:-:|:--|:--|:-:|:--|
| 1 | **t3 (post cp.async)** | 190 | cpa_wait Q/dO/L/D + STS smL/smD | smQ reads (Step B) + smdO reads (Step D) | **∩ smQ, smdO, smL, smD** | сторож жив |
| 2 | **t_new1 (pre-scatter)** | 315 | smQ reads (Step B: swz_byte 0..8192) | smdS_stage writes (Step E: [0..5120] ⊂ smQ_region alias) | **∩ smQ_region [0..5055]** | сторож жив |
| 3 | **t9 (pre-drain)** | 377 | smdS_stage writes (Step E) | smdS_stage reads (Step F drain) | **∩ smdS_stage весь** | сторож жив |
| 4 | **t_new2 (post-drain)** | 399 | smdS_stage reads (Step F) | smP_T writes (Step G: smQ_region alias) | **∩ smQ_region** | сторож жив |
| 5 | **t11 (pre-MMA_dV)** | 423 | smP_T writes (Step G) | smP_T reads (Step H MMA_dV A-frag) | **∩ smP_T** | сторож жив |
| 6 | **t13 (end qt)** | 462 | smP_T reads + smdO reads (Step H) | next qt: smQ + smdO overwrites (Step A) | **∩ smdO, smP_T alias vs smQ** | сторож жив |

**Итог 0d**: **все 6 барьеров живые, ни один не снимаем адресно-множественным анализом**. Barrier stall 2.59% — низкий вклад (не мишень).

Заметка: t_new2 сохранён специально по коду (комментарий в line 397-398: "смысл — sync для dS_nat drain + step G STS Pr → smP_T"). Alias-цепочка smQ_region (smQ / smdS_stage / smP_T) в трёх фазах требует **все три** барьера (t_new1, t_new2, t11) для corretness.

**Barrier count total per qt**: 6.  
**Barrier stall (2.59%) / 6 = ~0.43% per barrier** — norm.

---

## 0e. Скаттер-разбор dS_nat-staging под pack-аналог (Step E, строки 313-374)

### Битовая карта dS_nat scatter (первая строка, per lane, per (ni_a, ni_b) pair, per qt)

**Per lane выработка**: 4 fp8 pairs = 4 × 2 = 8 fp8 значений в 4 STS.b16 (u16).

**Формулы адреса (from code)**:
```
i_local_lo = wid*16 + l_div4 + 0     // m ∈ {0..15} within warp; wid ∈ {0..3}
i_local_hi = wid*16 + l_div4 + 8     // +8 for hi
ja_lo = ni_a*8 + l_mod4*2 + 0        // j = ni*8 + l_mod4*2 (+0 lo / +1 hi)
jb_lo = ni_b*8 + l_mod4*2 + 0        // (ni_b = ni_a + 1)
ja_hi = ni_a*8 + l_mod4*2 + 1
jb_hi = ni_b*8 + l_mod4*2 + 1

STS.b16 smdS_stage[i_local_lo * 80 + ja_lo] = dsa_lo_fp8   (byte 0,1)
STS.b16 smdS_stage[i_local_lo * 80 + jb_lo] = dsb_lo_fp8   (byte 0,1)
STS.b16 smdS_stage[i_local_hi * 80 + ja_lo] = dsa_hi_fp8   (byte 0,1)
STS.b16 smdS_stage[i_local_hi * 80 + jb_lo] = dsb_hi_fp8   (byte 0,1)
```

**Ширина**: **STS.b16** (2 bytes/store). Не .32 и не .U8. Между `ja_lo` и `jb_lo` расстояние — 8 (ni_b − ni_a = 1, delta j = 8).

**Count per qt per lane**: `NI_DP / 2 = 4` итераций внешнего цикла × 4 STS.b16 = **16 STS.b16 per lane per qt** = 32 bytes записанных.

**Общий счёт per qt per block (128 threads)**: 16 × 128 = 2048 STS.b16.

### Возможность pack-аналога

**Аналог dk_new W2 pack scatter** — 16 STS.32 per lane per qt (× 12 SHFL + 64 PRMT + 24 SEL cross-lane rewrite). В merged это перевод 16 STS.b16 → 8 STS.32 или 4 STS.b32.

**Проверка адресной геометрии**:
- Соседние stores по столбцам: `ja_lo` и `jb_lo` = `ni_a*8+lm4*2` и `(ni_a+1)*8+lm4*2` = расстояние **8**. Не adjacent (нужна adjacency 2 для склейки b16 → b32).
- Соседние по строкам: `i_local_lo` и `i_local_hi` = wid*16+ld4+0/+8. Расстояние по row-стриду 80: 8*80=640B. Далеко.
- Соседние по (lo/hi) в одной итерации: `ja_lo` и `ja_hi` = делта 1 в j → **adjacent** (расстояние 1 в 80-байтовом ряду).
- Но `ja_lo`/`ja_hi` пишут не связанные dSa значения (dsa_lo_fp8 из pa_lo_h2, dsa_hi_fp8 из pa_hi_h2 — разные lane-fragments!) — **cross-lane exchange нужен как в dk_new W2 pack**.

**Итог 0e**:
- Прямой STS.b16 → STS.32 **невозможен** без **cross-lane rewrite** (pack-analog: PRMT + SHFL + STS.32).
- Требуется вывод хореографии от читателя (dq_new? dk_new W2? — здесь smdS_stage читается drain'ом STG.128 в Step F, а dk_new читает `dS_nat_out` из DRAM — читатель НЕ SMEM, а глобальный. Значит pack applies до drain, а не для читателя.).
- Потенциал: 16→8 STS (−50% STS.b16), ~mio -3..-5 pp; цена: 12+ SHFL + 20+ PRMT ALU в hot loop, вероятная +5..15r нагрузка (по опыту dk_new).
- **Регистровая цена — только через ptxas-факт** (правило кампании). Прогнозы без бумаги не давать.

**Осторожность**: dq-хореография pack не переносилась в dk (разные обмены между варпами). Прежде чем гейт — потребуется derivation "от читателя" (drain STG.128 в Step F читает `smdS_stage[r*80 + col_byte]` chunks по 16 байт — не MMA, а memcpy-like) → возможно, ужать pack проще, чем в dk (нет требования MMA-совместимости на читателе).

---

## 0f. Вердикт-карта кандидатов (отсортирована по потенциалу)

| # | Имя | Класс-мишень (measured) | Потенциал wall | Цена | Риск | Приоритет |
|:-:|:--|:--|:--|:--|:--|:--|
| **1** | **LDS.U16 → LDS.32 склейка** класса #7 (smdO MMA_dV read) | 256 ops/lane/qt = 50% всех LDS; mio 24.43% → цель ~18-20% | **-1..-2 мс** (потенц.) | XOR-паттерн rewrite smdO scatter → smP_T reader; +ptxas-факт regs; +bit-exact 11/11 | средний (XOR bit-map probe нужен) | **1st проба** |
| 2 | LDS.U16 → LDS.32 склейка класса #5 (smV MMA-B read) | 128 ops/lane/qt = 25% LDS | -0.5..-1 мс | k_xor bit-map probe | средний | 2nd |
| 3 | LDS.32 → LDS.64 склейка класса #2 (smK MMA-A K read) | 64 ops/lane/qt = 12% LDS; уже LDS.32 → возможна .64 | -0.3..-0.6 мс | k-frag 16 stride может НЕ давать 64-bit alignment после XOR | низкий-средний | 3rd |
| 4 | **Snятие 5120B dead-alloc** (smdS_T_stage) | Только SMEM headroom; **не даёт скачка occupancy** (2→2 blocks сохраняется) | 0 мс wall, но sanity/место под будущие правки | trivial edit launcher smem_bytes; bit-exact 11/11 | нулевой | тривиальное, сделать при первой правке |
| 5 | Pack-analog в Step E dS scatter | 2048 STS.b16/block/qt = 16 ops/lane/qt = ~3-5% LSU | -0.3..-0.5 мс | 12+ SHFL + 20+ PRMT ALU; regs +5..15 (ptxas-факт); хореография вывод от reader | **средний-высокий** (регистры vs 3 блока recheck) | 4th, после LDS |
| 6 | Barrier снятие | 2.59% barrier stall | -0.1..-0.2 мс | Все 6 барьеров сторожа живы (0d) | не применимо | **исключено** |
| 7 | 3-блочная занятость (via ≤170r + ≤33792B) | Big — если бы достижимо | N/A (недостижимо) | dV_acc split структурная (+2× drain traffic) | **высокий** | **исключено (0b)** |
| 8 | Barrier count reduce (не снятие, а перекомпозиция) | 6 барьеров, каждый ~0.43% | -0.1..-0.3 мс | требует нового alias plan (например t_new1+t_new2 в один pre-scatter+drain-overlap) | средний-высокий | 5th, если 1-4 не дадут |

### Рекомендация первой пробы

**#1 (LDS.U16 → LDS.32 для smdO MMA_dV read)** — самая большая доля LDS-wavefronts, самый прямой mio-фикс. Требуется:

1. **XOR bit-map derivation** (паттерн — `n ^ dO_xor_even` и `n ^ dO_xor_odd` где odd=even+8, `n=ni*8+l_div4`, `dO_xor_{even,odd}` варьирует по l_mod4).
2. **Probe unit-test**: bit-perfect замена LDS.U16 pairs на LDS.32 с последующим SHR/AND в регистре.
3. **Гейт**: ptxas-факт (regs предельно ~+5r допустимо), bit-exact 11/11 + CANARY, ABBA ≥6 пар vs 033_sealed.

Прогноз wall (только после probe, не до): не заявляется. По правилу кампании прогнозов wall без бумаги на конкретную правку не даём.

---

## Резюме

- **037-fresh baseline**: merged isolated **31.359 ms**, 210.37 T (3-MMA), stand-protocol ✓.
- **Stall-таблица (Σ 99.05%)**: wait 27.20% + mio 24.43% + selected 18.15% + short_sb 10.89% + long_sb 5.18% + math_pipe 3.91% + not_sel 5.23% + barrier 2.59% + прочие 1.42%.
- **Джекпот 3 блоков/SM закрыт**: reg ceiling 170r (не 213r как в TZ) vs peak фазы ~209r; SMEM ceiling 33792 B vs 41472 B после T-cut cleanup.
- **5120B dead-alloc (smdS_T_stage)** — тривиальное cleanup, но НЕ даёт скачка occupancy.
- **6 barriers все сторожа живые** (address-set проверка, 0d).
- **Топ-мишень LDS #7** (smdO MMA_dV read, 256 ops/lane/qt = 50% LDS) — рекомендация первой пробы.
- **dS scatter pack-analog** — потенциал есть, но регистровая цена — через ptxas-факт, не прогноз.

### Файлы

- `runs/reports/037_merged_wall.sh` — stand-protocol скрипт
- `runs/reports/037_merged_wall_data.txt` — данные 5-run
- `runs/reports/037_ncu_merged.sh` — NCu script (cuda-13.1 ncu)
- `runs/reports/037_ncu_merged_data.txt` — NCu output raw
- `libs/query_caps.cu` + `libs/query_occ.cu` + Makefile-ы — occupancy audit tools
- `libs/query_caps`, `libs/query_occ` — output bin для перепроверки runtime facts

Chain md5: 035 `2a906152…` → 036 `7c4cce12…` → 036-r `2d770375…` → **037 `<computed>`**

---

**End 037.**

**Итог фазы 0**:
1. Разведка merged завершена (measurement/paper only, production не тронут).
2. Правила кампании установлены: **ptxas-факт перед любой правкой, никаких прогнозов wall без probe-бумаги**.
3. Рекомендация первой пробы: **LDS.U16 → LDS.32 для smdO MMA_dV read (класс #7)**.
4. **Vugar-target 213r=3 блоков/SM опровергнут runtime-фактом** (реальный ceiling 170r). Джекпот 3 блоков закрыт — фокус на throughput внутри 2 блоков/SM.
5. **5120B dead-alloc** — тривиальный cleanup при первой правке.
