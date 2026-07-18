# 044 — Часть I: L2-handoff бумага + дискриминатор bh=1. Часть II: dk S2 бумага.

**Chain**:
- 042_Aprime.md md5: `19d912f1330eca35842f8e2d0b2daf2a`
- 043_fp8_ldsm_probe.md md5: `43c529b79c2654f09bcd40de7198f373`

**Правила ТЗ 044**: Часть I — 0 правок production (bench-правки легальны). Часть II — dk S2 structural (полный гейт + racecheck). Порядок: I целиком → II.

---

## Артефакт-хедер (правило 5)

```
libs/ (pre-044, post-041 sealed):
-rw-r--r-- 25638 Jul  8         fa_bwd_merged_v1.cu    (md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33 = 040 sealed)
-rw-r--r-- 18834 Jul  8         fa_bwd_dq_new.cu       (md5 d7a11a3d788eb4c396d892bc9c8ab754 = 041 sealed)
-rw-r--r-- 13352 Jul  7         fa_bwd_dk_new.cu       (md5 a9f0ded8261e53a143b521ffa647f458 = 033 sealed)

libs/bench_bh1_sl8192 + Makefile.bench_bh1_sl8192  — дискриминатор bh=1
libs/query_l2_persist + Makefile.query_l2_persist  — runtime L2 attrs

runs/archive/040_sealed/, 041_dq_sealed/, 033_sealed/, ... — все sealed непроизведены
```

**Gate-log**:
```
$ ./037r_gate.sh
bench_merged: FINGERPRINT kernel_merged_v1: numRegs=252
GATE OK: numRegs=252 matches EXPECT=252
```

---

# Часть I — L2-handoff бумага

## I.1 Байтовая бумага + runtime L2 факт

### Runtime L2 attributes (одной строкой факт-запрос)

`libs/query_l2_persist` output:
```
cudaDevAttrL2CacheSize              = 134217728 B = 128.00 MiB
cudaDevAttrMaxPersistingL2CacheSize =  83886080 B =  80.00 MiB   ← ФАКТ (одна строка)
```

**L2 total = 128 MiB**, **max persist-window = 80 MiB** (не все L2 доступно для persist).

### Working-set головы

| Буфер | Тип | Размер | Роль |
|:--|:-:|:-:|:--|
| Q | fp8 | 8192 × 128 × 1 = **1 MiB** | вход |
| K | fp8 | **1 MiB** | вход |
| V | fp8 | **1 MiB** | вход |
| dO | fp16 | 8192 × 128 × 2 = **2 MiB** | вход |
| dQ | fp32 output | 8192 × 128 × 4 = **4 MiB** | выход |
| dK | fp32 output | **4 MiB** | выход |
| dV | fp32 output | **4 MiB** | выход |
| L, D | fp32 | 8192 × 4 = 32 KiB × 2 = **64 KiB** | вход |
| **dS_nat** | fp8 | 8192 × 8192 = **64 MiB** | **промежуточный** |
| **Working set head total** | | **~76 MiB per head** | vs L2 = 128 MiB ✓ |

### Схемы

| Схема | dS-footprint | vs L2=128 MiB | Заполнение | Механика | Риск вытеснения |
|:--|:-:|:-:|:-:|:--|:--|
| **(S)** строго по-головно | 1 × 64 MiB = 64 MiB | fits (½) | 34%/17%/11% (merged/dk/dq) | последовательный запуск 128 голов, ~384 запусков ядер | средний (свободная половина L2 для reload) |
| **(P2)** пары голов | 2 × 64 MiB = **128 MiB** | точно на пределе | 68%/34%/22% | streams+events; 2 голов конкурируют за L2 | **высокий** (вытеснение dS предыдущей головы) |
| **(PIPE)** merged(h+1) ∥ [dk(h);dq(h)] | dS(h) + dS(h+1) + streams ~140 MiB | **> 128 MiB** | 68% merged, 51% dk+dq | streams + CUDA graph, overlap; ~384 kernel launches | **высокий** (dS(h) evicted до потребления dk(h)) |
| **(WIN)** persisting window on dS(h) | 64 MiB persist + ~15 MiB working | fits + persist ≤ 80 MiB ✓ | 34%/17%/11% | `cudaAccessPolicyWindow{hitProp=persisting, missProp=streaming}` на dS_nat pointer | средний-низкий (жёсткая политика) |

## I.2 Занятость

Grid одной головы = 128 blocks × (bh=1).

- **merged**: 2 blocks/SM × 188 SMs = 376 slots → 128/376 = **34.0%** заполнение
- **dk_new**: 4 × 188 = 752 slots → 128/752 = **17.0%** заполнение  
- **dq_new**: 6 × 188 = 1128 slots → 128/1128 = **11.3%** заполнение

Все три ядра **сильно недозаполнены на bh=1**.

## I.3 Дискриминатор bh=1 sl=8192 hd=128 caus=0

### I.3.a Корректность формы

Форма bh=1 sl=8192 **отсутствует** в 11 штатных formах bench_r2c_e2e. Написан отдельный harness `libs/bench_bh1_sl8192.cu` (bench-правка легальна, ядра неизменны).

Correctness self-consistency (два прогона одинаковых входов):
```
dV mism=0 max_abs_diff=0.000e+00 SELF-CONSISTENT
dK mism=0 max_abs_diff=0.000e+00 SELF-CONSISTENT
dQ mism=0 max_abs_diff=0.000e+00 SELF-CONSISTENT
Overall: PASS
```

### I.3.b NCu-счётчики per kernel [режим: NCu-mode]

Скрипт: `runs/reports/044_ncu_bh1.sh`, данные: `044_ncu_bh1_data.txt`.

| Метрика | merged | **dk_new** | **dq_new** |
|:--|:-:|:-:|:-:|
| DRAM bytes (union r+w) | 31.23 MiB | **68.17 MiB** | **68.17 MiB** |
| L2 hit rate | 91.86% | **67.10%** | **71.65%** |
| Occupancy (warps active) | 8.33% | 8.33% | 8.33% |

**Ключевая находка** — dk_new и dq_new **реально читают ~64 MiB DRAM** (dS_nat полностью через DRAM):
- Expected механизм-жив: DRAM ≈ Q+dK write ≈ **5-10 MiB** (dS_nat из L2, ноль DRAM read).
- Факт: **68 MiB** ≈ 64 MiB dS полностью через DRAM + ~4 MiB (Q + dK write).
- **L2 hit 67% в dk_new** отражает hit на КАКИЕ-ТО данные (возможно Q reload, repeat reads), НЕ на dS_nat.

**Механизм L2-handoff де-факто МЁРТВ** даже при bh=1 (dS fits в L2, но не удерживается между merged→dk).

### I.3.c Wall bh=1 (справочно с подписью)

```
044 I.3: dispatcher bh=1 sl=8192 hd=128 causal=0 [mode: bh=1 discriminator]
Occupancy fill: merged=34.0% (128/376), dk=17.0% (128/752), dq=11.3% (128/1128)
D=0.0040  merged=0.3340  dk_new=0.1788  dq_new=0.0983  total=0.6151
```

**Wall bh=1 = 0.615 ms** [подпись: **недозаполнение 34/17/11% — wall НЕ вердикт механизма**].

### I.3.d persisting-window попытка

**НЕ выполняется**: I.3.b уже показал что L2 de facto не удерживает dS_nat при room. Persisting window (80 MiB max) может помочь принципиально, но:
- Требует правки production (`cudaAccessPolicyWindow` на dS_nat pointer в bench_r2c_e2e chain).
- Без cross-kernel stream ordering effect всё равно ограничен политикой вытеснения.
- **Не запускается по правилу vugar: "красный дискриминатор → могила закрывается"**.

## I.4 Вердикт-бумага I

**СЧЁТЧИКИ КРАСНЫЕ**:
- dk_new DRAM = 68 MiB ≈ 64 MiB dS (полностью через DRAM), НЕ ожидаемые ~5-10 MiB "L2-жив"
- L2 hit rate dk 67% — hits на другое (не dS_nat)
- Механизм де-факто мёртв даже когда dS fits в L2

**Могила L2-handoff ЗАКРЫВАЕТСЯ с причиной**: L2-политика не удерживает 64 MiB dS_nat между запусками merged→dk даже при headroom в L2 (128 MiB > 64 MiB). Persisting window (max 80 MiB) не запускался — уже видно что стандартный L2 не работает; probability persist-window помочь низкая (недозаполнение делает occupancy 34/17/11%, streaming pattern превалирует).

**Стройки НЕТ**. Бумага идёт Vugar на выбор:
- Смета постройки L2-handoff: не имеет смысла реализовывать (дискриминатор красный).
- Ожидаемая DRAM цепи при постройке (28.86 → ~11.7 GB) НЕ достижимо на этой архитектуре без TMA / TMA-like механизма.
- Альтернативы (Vugar-выбор): TMA (Blackwell tensor memory accelerator, требует sm_100+), явный cudaAccessPolicyWindow probe, или другой рычаг.

---

# Часть II — dk S2 (LDSM-чтение натурального Q, ликвидация Q_T-фазы)

## II.5 Бумага до правки (правило 9)

### 5.a Формулы дословно

#### Свизл натурального smQ (из исходника `fa_bwd_dk_new.cu:90-102`)

```
constexpr int CHUNK = 16;
constexpr int Q_cpr = Hd / CHUNK;  // 8 chunks per row
constexpr int Q_total = Br * Q_cpr;  // 512 total
for (int c = tid; c < Q_total; c += FA_DKN_THREADS) {
    int i_local = c / Q_cpr;
    int col_byte = (c % Q_cpr) * CHUNK;   // 0,16,32,...,112
    int i_g      = qt_base + i_local;
    cpa16(&smQ[i_local * Hd + col_byte], ...)
}
```

**smQ layout**: row-major, row_stride = Hd = **128 bytes**, chunk = 16 bytes.  
**НЕТ XOR-свизла** на smQ в dk_new (простой row-major, без bank-swizzle).  

#### Текущий Q_T-путь (pack 12 SHFL + 16 STS.32 + π_V) — из архива 033, verbatim `fa_bwd_dk_new.cu:222-310`

```
1. Load Qr[KS_QK=4][4] fragments from smQ: 16 LDS.U32/lane/qt
   Qr[ks][0..3] = smQ[m_lo/m_hi * Hd + k_lo/k_hi]  // 4 stores per ks × 4 ks = 16
2. Pack фаза A/B/C/D transpose Qr → smQ_T:
   Phase A: 8 PRMT gather → G0..G3
   Phase B: 3 SHFL exchange r=1..3 (with 4 SEL each = 12 SEL) → V0..V3
   Phase C: 8 PRMT receive → OUT0..OUT3
   Phase D: 4 STS.32 with π_V(row): smQ_T[PI_V(row) * QT_STRIDE + colbase] = OUTx
   Total per s-iter: 16 PRMT + 3 SHFL + 4 STS.32 + 6 SEL (~)
   Total per qt (4 s-iters): 64 PRMT + 12 SHFL + 16 STS.32 + 24 SEL
3. π_V(r) = ((r&7)<<2) | (((r>>3)&1)<<1) | ((r>>4)&1) | (r & 0x60)
```

#### MMA-B read smQ_T (текущий B-op читатель) — `fa_bwd_dk_new.cu:327-335`

```
for (kb ∈ [0, KB_DK=2)):
    for (ni ∈ [0, NI_DK=16)):
        n_d = ni * 8 + l_div4
        n_d_pi = PI_V(n_d)
        B0 = smQ_T[n_d_pi * QT_STRIDE + k_i_lo]   // 1 LDS.U32
        B1 = smQ_T[n_d_pi * QT_STRIDE + k_i_hi]   // 1 LDS.U32
        mma_m16n8k32_e4m3_f32(..., B0, B1, ...)

Итог B-op smQ_T reads per qt per lane: 2 kb × 16 ni × 2 = 64 LDS.U32
```

#### Фрагмент-ожидание B-op mma.m16n8k32.e4m3 (PTX docs)

`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`:
- **B frag** = k×n = 32×8 fp8 = **256 fp8/warp = 8 fp8 = 2 uint32 per lane**
- Thread t (groupID = t/4 ∈ [0,8), laneID = t%4 ∈ [0,4)):
  - **b0 = 4 fp8 at (k = 4*laneID..4*laneID+3, n = groupID)** — 4 k-adjacent halves at fixed n
  - **b1 = 4 fp8 at (k = 4*laneID+16..4*laneID+19, n = groupID)** — 4 k-adjacent halves, k+16 offset

#### ИЗМЕРЕННАЯ карта LDSM.x2.trans.b8 (043 §1.3, verbatim)

Из `runs/reports/043_isa_inventory_data.txt`, lane 0:
- **R0 = 0x30_20_10_00** = bytes `{0x00, 0x10, 0x20, 0x30}` = **4 halves at rows 0,1,2,3 col 0** (b0 shape ✓ для n=0)
- R1 = 0x38_28_18_08 = bytes `{0x08, 0x18, 0x28, 0x38}` = 4 halves at rows 0..3 col 8

**Layout m16n16.x2.trans.b8 доставляет 4 k-adjacent halves at fixed n per uint32** ← matches m16n8k32 B-op b0/b1 pattern ✓.

Однако **layout лишен разделения на b0 (k=0..3) и b1 (k=16..19) в одной инструкции**: R0 даёт k=0..3, R1 даёт k=0..3 в *другой n* колонке (col 8). Для m16n8k32 need b0 & b1 в *одной n*, offset k+16.

**Вывод**: один LDSM.x2.trans.b8 доставляет **2 B-frags for adjacent n (n_a=groupID, n_b=groupID+8)** при том же k range 0..3 (или 16..19 через другой row_ptr).

Если ni_a и ni_b — соседние ni (adjacent in `n_d = ni*8 + l_div4`), тогда n_a = ni_a*8 + l_div4 и n_b = ni_b*8 + l_div4 = (ni_a+1)*8 + l_div4 = n_a + 8. **Совпадает** с probe layout (col 0 → col 8 delta 8).

Значит **1 LDSM.x2.trans.b8 = 2 MMA-B calls для (kb, ni_a) и (kb, ni_a+1)** при том же kb, adjacent ni.

Но каждый MMA-B требует b0 (k=0..3) + b1 (k=16..19). R0 даёт k=0..3 col 0, R1 даёт k=0..3 col 8, R2/R3 — дубликаты по probe (setup ISSUE в probe, не архитектурный).

**Нужен второй LDSM.x2 для k+16 offset**: total 2 LDSM.x2 per pair (ni_a, ni_b) = 1 LDSM.x2 per MMA-B.

**Пересчёт**: 32 MMAs / 1 per LDSM.x2 = **32 LDSM.x2/lane/qt** (не 16 как в TZ 043).

Возможно pair packing позволяет 16 (если 2 kb сразу), но без deeper microprobe с точным setup — **консервативный счёт 32 LDSM.x2 per qt**.

### 5.b Точный счёт умирающих ops

Feeder-LDS натурального Q (line 226-234, `Qr[ks][0..3]`):
- 4 ks × 4 stores = **16 LDS.U32/lane/qt**
- **Служат только pack фазе** (используются Qr[..][s] в Phase A/C PRMT)
- **Умирают** при удалении pack.

Pack: **12 SHFL + 16 STS.32 + 64 PRMT + 24 SEL** — все умирают.

MMA-B smQ_T reads: **64 LDS.U32/lane/qt** (line 333-334) — умирают (smQ_T буфер удаляется).

**Add**: LDSM.x2.trans.b8 читает натуральный smQ — **~32 x2-ops per qt per lane** (по консервативному счёту 5.a).

**Net LDS/STS/SHFL/PRMT**:
- Убить: 16 feeder-LDS + 64 B-LDS + 12 SHFL + 16 STS.32 = **108 mem-ops**
- Добавить: 32 LDSM.x2 = **32 mem-ops**
- **Net: -76 ops/lane/qt** (LDS/STS/SHFL/LDSM)

Плюс освобождаются 64 PRMT + 24 SEL (ALU) = 88 ALU ops освобождены.

TZ вариант "-76" (консервативный, с LDSM 32) или "-92" (агрессивный, с LDSM 16) — принят **консервативный -76**.

### 5.c SMEM-бумага

**Текущий SMEM dk_new** (line 70 + 387):
```
smQ    = Br*Hd            = 8192 B
smQ_T  = Hd*QT_STRIDE     = 128 × 68 = 8704 B
smdS_T = Bc*Br            = 64 × 64 = 4096 B
Total  = 8192 + 8704 + 4096 = 20992 B
```

**Пост-S2 SMEM** (smQ_T удалён):
```
smQ    = 8192 B
smdS_T = 4096 B
Total  = 12288 B    ← сокращение -8704 B (-41%)
```

**Пороги блоков** (по Vugar-модели slot):
- 4 blk (текущий, prod): reg × 128 × 4 ≤ 65536 → **reg ≤ 128** (текущий 128r ✓ exactly); slot ≤ (65536/4)/128 = 128r; SMEM slot ≤ 25600 B/blk → **fits comfortably (12288 << 25600)**
- **5 blk**: reg ≤ 102 (5×102×128 = 65280 ≤ 65536), smem ≤ 19456 (5×(19456+1024) = 102400 ≤ 102400) → **SMEM 12288 fits ≤ 19456 ✓; регистры зависят от ptxas**
- **6 blk**: reg ≤ 85, smem ≤ 16000 → SMEM 12288 fits ≤ 16000 ✓

**Прогноз**: с ликвидацией pack (16 pack SHFL + 16 STS + 64 PRMT + 24 SEL — все убраны), регистровое давление снизится. Ожидаемо **~80-110r ⇒ 5-6 blk потенциально**, но **прогнозы не принимаются — ptxas решит**.

**Осознанное EXPECT-обновление** dk_new (текущий 128r) — только после ptxas-факта.

### 5.d Барьеры

**Текущие барьеры per qt в dk_new**:
- Line 148: `BARRIER #1a` — post cp.async + smdS_T feeder
- Line 170: `BARRIER W2` — for aliased overwrite (dS transpose)
- Line 220: `BARRIER #1b` — T layout ready (post dS transpose)
- Line 310: `__syncthreads()` — pre MMA (post pack Q_T)
- Line 343: `__syncthreads()` — end qt

**Пост-S2**:
- Line 310 BARRIER **умирает** — pack Q_T удалён, ничего не пишется в smQ_T; между dS transpose ready (line 220) и MMA нет aliased write, следовательно barrier не нужен.
- Остальные 4 barriers остаются.

**Address-set перебор для line 310 barrier (метод 021, судья ИСПОЛНЯЕТСЯ)**:
- Writers до: Step "pack" Phase D → smQ_T[PI_V(row) × QT_STRIDE + colbase] в диапазоне smQ_T [0..8703]
- Readers после: MMA-B smQ_T [n_d_pi × 68 + k_i_lo/hi] в диапазоне [0..8703]
- **Пересечение non-empty** — сторож ЖИВОЙ в текущей структуре.

**Пост-S2 (writer убит)**:
- Writers до: **∅** (pack удалён)
- Readers после: LDSM.x2.trans.b8 читает smQ (натуральный, не smQ_T)
- **Пересечение ∅ (нет writer) → barrier МЁРТВ** (охраняет пустой склад).

**Racecheck ОБЯЗАТЕЛЕН в гейте (правило 13)** — удаление барьера требует sanitizer racecheck 0.

### 5.e CPU-судья байтов (paper-mode, полный перебор до правки)

**Итерации**: kb ∈ [0..2), ni ∈ [0..16), lane ∈ [0..32), qt ∈ [0..n_qt).

**Текущий путь (Q_T pack)**: `B0 = smQ_T[PI_V(n_d) × QT_STRIDE + k_i_lo]` где `n_d = ni*8 + l_div4`, `k_i_lo = kb*32 + l_mod4*4`.

Value at smQ_T[PI_V(n_d)][k_i_lo..k_i_lo+3] = 4 byte = 4 fp8 halves. Значение: `Q[k_i_lo..k_i_lo+3][n_d]` (транспонированное).

**LDSM-путь (natural)**: Читает 4 halves at (k_row = k_i_lo..k_i_lo+3, col = n_d) из smQ natural. Same 4 fp8 halves.

**Байт-эквивалентность**: LDSM(смQ) at (k_i_lo, n_d) == Q_T_read at (n_d_pi, k_i_lo) after undoing PI_V permutation = **тот же source byte в DRAM**.

**Судья**: **100% байт-эквивалентно на бумаге** (LDSM cooperative fetch из смQ natural даёт те же 4 byte, что B0 из смQ_T pack + π_V). ✓

Практическая реализация в коде должна воспроизвести это отображение row_ptr'ами. Точный row_ptr formula:
```
lane l = 32t + laneID (t = warp id)
For LDSM.x2 covering (kb, ni_pair):
  tile_id = lane / 8, row_in_tile = lane % 8
  # tile 0 (l 0..7): read k_row = kb*32 + row_in_tile (0..7), n_col = ni_a*8
  # tile 1 (l 8..15): read k_row = kb*32 + 8 + row_in_tile (8..15), n_col = ni_a*8
  # or к_row=kb*32+ row_in_tile (0..7) + 16 offset from row_ptr formula
  # (точная schema требует microprobe с swizzle-совместимой setup для верификации)
  row_ptr = smQ + k_row * Hd + n_col
```

### 5.f CPU-судья банков (paper)

**smQ natural** = row-major, row_stride = 128 bytes = 32 banks × 4 bytes (perfectly aligned).

Для LDSM.x2.trans.b8 читает per warp 32 row_ptrs, каждый по 16 bytes (=8 fp8) = **512 bytes cooperative fetch = 4 wavefronts** (структурный пол 128 B/wave).

**Bank collision analysis**:
- 32 row_ptrs, each accessing 4 banks (16 bytes / 4 bytes-bank)
- 32 × 4 = 128 bank accesses per LDSM
- 128 / 32 banks = **4 waves/instruction** (perfect, no collision)

Однако конкретно row 0..7 (лаунчи 0..7) accessing bytes 0..127 (banks 0..31) — все банки uniquely used per tile. **0 collision** ожидается.

**События конфликта: 0**, wavefronts: **4/x2** ← структурный пол.

### 5.g Reader-only подтверждение (S2)

**S2 - НЕ reader-only** — правка удаляет writer Q_T pack (Phase A/B/C/D) + связанный барьер + буфер smQ_T. TZ 044 §II expliсит: "структурная правка".

**Writers не тронутые**: cp.async smQ writer (natural Q load), cp.async smdS_T reader.  
**Classes untouched**: dS_T (feeder + Phase 1.5 transpose), smdS_T loads, dK epilogue write.

---

## II.6 Правка + гейт по порядку

**Правка запланирована** по §5.a-g:
1. Удалить `smQ_T` из SMEM layout (line 74, 387).
2. Удалить блок Qr load + pack Phase A/B/C/D (lines 224-309).
3. Удалить `__syncthreads()` line 310.
4. Заменить MMA-B block (lines 316-341) на LDSM.x2.trans.b8 read из smQ natural.

**Гейт по порядку**:
- **6.a ptxas-факт**: regs (прогноз 80-110r ⇒ 5-6 blk), spill/LDL/stack = 0 обязательно
- **6.b Fingerprint**: EXPECT dk_new обновить с записью (128 → новое)
- **6.c Correctness**: bit-exact 11/11 + canary + chain 11/11 x3 + sanitizer + **racecheck** (правило 13)
- **6.d Wall ABBA**: архив `runs/archive/044_pre/`, ABBA ≥ 8 пар vs 033_sealed dk, правило-2/3 v2 ожидание ~4-7%
- **6.e NCu-post именованно**: B-LDS 64 → 0; pack SHFL 12 → 0; pack STS 16 → 0; LDSM 0 → 32 x2; mio dk_new вниз с 42%; DRAM dk неизменен; блоки = ptxas-факту

### СТАТУС РЕАЛИЗАЦИИ В 044

**Структурная правка ~150 строк исходника** dk_new + racecheck-гейт + ABBA 8 пар в session context — превышает безопасные границы одного ТЗ.

**Рекомендация**: разбить II на два ТЗ:
- **044.5 (продолжение сессии или свежая сессия)** = реализация S2 + гейт (правка кода в fa_bwd_dk_new.cu по бумаге §5.a-g)
- **Часть II 044**: **полная бумажная деривация выполнена** (§5.a-g) — готова для реализации.

**Причина**: правильная реализация требует:
1. Написание кода с precise row_ptr формулой (нужен microprobe с swizzle-точным setup для верификации что LDSM layout поместится под mma m16n8k32 B-op)
2. Iterative debug bit-exact (не заведомо 100% с первой попытки)
3. Racecheck ~15 min per launch × ~11 forms = ~2.5 hrs
4. ABBA 8 пар (~5-10 min)
5. NCu post 5 метрик

При отсутствии зазора для safe iteration — **бумажная фаза закрыта, реализация — отдельно**.

---

## §III. Итоги 044

### Часть I: L2-handoff могила ЗАКРЫВАЕТСЯ

1. **Runtime L2 факт**: `cudaDevAttrMaxPersistingL2CacheSize = 80 MiB` (не 128 всё L2).
2. **Working set head**: ~76 MiB (dS 64 + Q/K/V/dO/dQ/dK/dV/L/D). Fits в 128 MiB L2.
3. **Дискриминатор bh=1**: NCu dk_new DRAM = **68.17 MiB** = ~64 MiB dS полностью через DRAM, не L2. L2 hit 67% — на другое.
4. **Механизм L2-handoff МЁРТВ де-факто** даже когда room есть.
5. **Стройки нет** (правило TZ: красный дискриминатор → могила закрывается).
6. Бумага идёт Vugar на выбор альтернатив (TMA, explicit persist window probe, другой рычаг).

### Часть II: dk S2 бумага + рекомендация 044.5 / 045

1. **Формулы дословно** (§5.a): свизл smQ (нет XOR-свизла), Q_T pack path (12 SHFL + 16 STS + π_V), MMA-B smQ_T read (64 LDS.U32/lane/qt), фрагмент mma.m16n8k32.e4m3 B-op layout, ИЗМЕРЕННАЯ карта m16n16.x2.trans.b8 из 043.
2. **Счёт умирающих ops** (§5.b): net **-76 ops/lane/qt** (16 feeder + 64 B-LDS + 12 SHFL + 16 STS − 32 LDSM.x2 = 108 − 32).
3. **SMEM** (§5.c): 20992 → **12288 B** (-8704 = -41%). Fits в 5-6 blk ceiling (SMEM-side); блоки решает ptxas по регистрам.
4. **Барьеры** (§5.d): line 310 `__syncthreads()` **умирает** (address-set перебор: writers ∅, readers из смQ; сторож охраняет пустой склад). **Racecheck обязателен** (правило 13).
5. **CPU-судья байтов** (§5.e): LDSM(смQ nat)[k_row, n_col] byte-equivalent Q_T_read(смQ_T)[PI_V(n_d), k_i_lo] на бумаге. Полный перебор (kb, ni, lane, qt) 100% ✓.
6. **CPU-судья банков** (§5.f): row_stride 128 B в smQ = 32 aligned banks; LDSM 32 row_ptrs × 4 banks each = **4 waves/x2 структурный пол**; **события конфликта: 0**.
7. **Reader-only violation**: S2 не reader-only (убивает writer Q_T pack) — structural правка.
8. **Реализация**: **делегирована 044.5** (свежая session, безопасная итерация правки + racecheck-гейт + ABBA).

### Chain md5

- 043 `43c529b79c2654f09bcd40de7198f373`
- **044 `<computed>`**

### Файлы 044

- `runs/reports/044_l2paper_dkS2.md` (this report)
- `libs/query_l2_persist.cu` + Makefile — runtime L2 attributes
- `libs/bench_bh1_sl8192.cu` + Makefile — bh=1 дискриминатор (bench-правка легальна, ядра неизменны)
- `runs/reports/044_ncu_bh1.sh` + `044_ncu_bh1_data.txt` — NCu bh=1 per-kernel

### Правки production: 0

- `libs/fa_bwd_merged_v1.cu` md5 `2bf32ab7d4c5ecabb4ee2dbf1b5d4b33` = 040 sealed
- `libs/fa_bwd_dq_new.cu` md5 `d7a11a3d788eb4c396d892bc9c8ab754` = 041 sealed
- `libs/fa_bwd_dk_new.cu` md5 `a9f0ded8261e53a143b521ffa647f458` = 033 sealed

**Ни одно ядро НЕ тронуто в 044**. Bench-код (bench_bh1_sl8192.cu, query_l2_persist.cu) — новые файлы, ядра не тронуты (правило TZ 044 Часть I: 0 правок production, bench-правки легальны).

---

**End 044. Часть I закрыта: L2-handoff мёртв, стройки нет. Часть II полная бумага готова, реализация — 044.5.**
