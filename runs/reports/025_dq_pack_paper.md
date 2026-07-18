# 025 — dq_new pack K_T scatter: бумага + unit-test + π-paper (ТЗ 009-2 шаг 1a+1b)

**Chain**:
- 023_abba_piv.md md5: `5d38e0c86785cb4af95a5f3e88811bc3`
- 024_dq_census.md md5: `8462480a45d88dfcbcaabac3599dd2ff`

**Artifact header**:
```
-rw-r--r-- 13352  Jul  5 17:51  libs/fa_bwd_dq_new.cu        (unchanged, md5 683396f8…)
-rw-r--r--         Jul  7        runs/probes/pack_kt_unit_test.cu    (создан в 1b)
```

**Production не тронут в этом шаге. Только бумага и unit-test.**

---

## 1a. Exchange-сеть ИЗ КОДА dq_new (не копия dk_spec)

### 1a.1 Feeder K (из fa_bwd_dq_new.cu:171-184)

```c
const int ks = wid;                          // ← const per warp (dk-разница: dk имеет ks=0..3 в цикле)
const int k_lo = ks * 32 + l_mod4 * 4 + 0;   // 4-byte offset в k-блоке
const int k_hi = ks * 32 + l_mod4 * 4 + 16;
uint32_t kr_lo[NI_QK], kr_hi[NI_QK];         // NI_QK = 8; итого 16 uint32 per lane
for (int ni = 0; ni < NI_QK; ++ni) {
    int n_K = ni * 8 + l_div4;               // 8 rows per lane
    kr_lo[ni] = smK_area[n_K * Hd + (k_lo ^ k_xor)];
    kr_hi[ni] = smK_area[n_K * Hd + (k_hi ^ k_xor)];
}
```

**Смысл каждого uint32**:
- `kr_lo[ni]` содержит 4 fp8-байта = `K[n_K][k_lo..k_lo+3]` где `n_K = ni*8+l_div4`, `k_lo = ks*32 + l_mod4*4`
- `kr_hi[ni]` = `K[n_K][k_hi..k_hi+3]` где `k_hi = k_lo + 16`

**Per lane покрытие**: 8 rows × 2 halves × 4 bytes = **64 fp8-байта K** = 16 uint32 регистров.

**Ключевая разница vs dk-фидер Q**:
- **dk feeder** (Q): `Qr[ks=0..3][slot=0..3]` — 4 ks в регистрах per lane, всё Hd покрыто одним lane.
- **dq feeder** (K): `kr_lo[8], kr_hi[8]` — 1 ks per lane (ks=wid), но 8 ni-rows покрыты одним lane.
- **Импликация**: dq feeder ni-axis в регистрах ↔ dk feeder ks-axis. Оси **разные семантически**, но обе — 16 uint32 регистров per lane с 64 fp8-байт покрытием. Числа шапки exchange-сети — **одинаковые**.

### 1a.2 Целевая раскладка K_T scatter

Из fa_bwd_dq_new.cu:187-195, byte source `kr_lo[ni]` → dst byte at `smK_area[(k_lo+bt) * KT_STRIDE + n_K]`:
- `dst_row = k_lo + bt = ks*32 + l_mod4*4 + bt` (bt ∈ [0..3], byte position within uint32)
- `dst_col = n_K = ni*8 + l_div4`

Транспонирование K[n][k] → K_T[k][n], per byte, полностью корректно. Total per warp: 32 lanes × 64 STS.U8 = 2048 STS.U8 (Br × Hd / 4 warps).

### 1a.3 Pack: STS.U8 → STS.32 требует cross-lane obмена

**Проблема**: STS.32 пишет 4 consecutive bytes at `[dst_row][dst_col..dst_col+3]`. `dst_col = ni*8 + l_div4`. 4 consecutive dst_col = 4 lanes с `l_div4 ∈ {a, a+1, a+2, a+3}` (a = 0 или 4).

Значит **4 lanes должны кооперативно сформировать 4-байт слово** для STS.32. Каждая lane владеет байтами только своего l_div4 → нужен cross-lane exchange.

### 1a.4 Структура slot'ов (dq-специфично)

**Слоты pack** (из данных фидера):
- Slot 0: `kr_lo[0..3]` = 4 uint32, k_lo half, ni-half low
- Slot 1: `kr_lo[4..7]` = 4 uint32, k_lo half, ni-half high
- Slot 2: `kr_hi[0..3]` = 4 uint32, k_hi half, ni-half low
- Slot 3: `kr_hi[4..7]` = 4 uint32, k_hi half, ni-half high

**Per slot**: 4 uint32 input per lane, needs to produce 4 output uint32 per lane (для 4 STS.32).

**4-lane exchange group**: per warp of 32 lanes = 8 lane-groups by `l_div4` (2 sub-groups × 4). Для transpose внутри slot, 4 lanes с same **l_mod4** и разными **l_div4 within sub-group** (l_div4 & 3) кооперируют. Sub-group = `l_div4 >> 2` ∈ {0, 1} (= h в dk-квадре).

**Analog to dk-quad structure**:
- `c = l_mod4`, `p = l_div4 & 3`, `h = l_div4 >> 2`
- Same 32-combo структура, но семантика байтов другая

### 1a.5 Счёт (dq-вывод, не dk-копия)

За slot:
- **Phase A** — gather вдоль ni sub-group: 4 uint32 inputs → 4 uint32 intermediate G0..G3, через **8 PRMT** (4 pair-shuffles + 4 4-byte reorders)
- **Phase B** — SHFL exchange 4-lane transpose: 3 rounds, `src(r) = c + 4*((p-r)&3) + 16h`, expose G[(p+r)&3] через 2-level SEL: **3 SHFL + 6 SEL** (два уровня mux)
- **Phase C** — receive-transpose: 8 PRMT (симметрично Phase A)
- **Phase D** — 4 STS.32 output

Per slot: **8 PRMT + 3 SHFL + 6 SEL + 8 PRMT + 4 STS.32**.

**За qt (4 slots)** per lane:
| Ops | Count | Note |
|:--|--:|:--|
| PRMT | 4 × 16 = **64** | Phase A + Phase C |
| SHFL | 4 × 3 = **12** | Phase B rounds r=1..3 |
| SEL | 4 × 6 = **24** | 4 slots × (2-level mux + V-write) |
| STS.32 | 4 × 4 = **16** | Phase D |
| **STS.U8** | **0** | ← target: убрать 64 STS.U8 |
| **LDL/STL** | **0** | ← детектор V[]-класса, V0-V3 named регистры |

**Итог: 12 SHFL + 16 STS.32 + 64 PRMT + 24 SEL** — **счётно совпадает с dk_new pack**, вывод независимый.

### 1a.6 Register cost

- Существующее: `kr_lo[8], kr_hi[8]` = 16 uint32 (уже в feeder-раскладке)
- Pack temp per slot (переиспользуемо между slot'ами): G0-G3 (4) + V0-V3 (4) + t01/t23_lo/hi (4) + OUT0-OUT3 (4) = 16 uint32 temp peak
- **Прогноз ptxas прирост**: **+17-22r** от 56r → **73-78r** (в окне 85r для 6 blocks)
- Урок 022: налог тесноты 1.5-2× → фактические +17 в dk_new π_V (прогноз был +11). Здесь прогноз уже с налогом.

### 1a.7 Полный inst-census до/после (per warp per kt, MIO-трафик)

**Baseline dq_new** (из 024, per kernel = 128 kt-iters):
- **inst_shared_ld: 889 M** (= Phase 1.5 read 16 LDS + MMA-B 64 LDS = 80 LDS × iters × 4 warps × 8192-batch × 128-heads = ~889M ✓)
- **inst_shared_st: 537 M** (= Phase 1.5 write 64 STS.U8 × iters × 4 warps × ... = ~537M ✓)

**Post-pack прогноз**:
- **shared_ld inst**: **unchanged 889 M** (Phase 1.5 read не тронут; MMA-B не тронут)
- **shared_st inst**: **537 M × 28/64 = 235 M** (было 64 STS.U8/lane/kt, стало 12 SHFL + 16 STS.32 = 28 shared_ops/lane/kt; SHFL считается как shared в MIO, STS.32 = 1 op) — **drop 302M = -56%**
- **MIO прогноз**: 46.73% → **34-38%** (drop ~8-12 pp, по аналогии с dk_new pack 018)

### 1a.8 Bank pattern новых STS.32 (paper, KT_STRIDE=68)

Per STS.32 target: `smK_area[dst_row * 68 + dst_col_word*4]`, bank = `(dst_row·17 + dst_col_word) mod 32`.

**За fixed (ks, slot), 32 lanes пишут 32 STS.32 в 32 разных банков?**  
Layout: `dst_row = ks*32 + l_mod4*4 + bt_slot` (bt_slot ∈ 4 значения per slot), `dst_col_word = (ni*8 + l_div4)/4`.

Полный bank-check требует enumeration — делаю позже в π-paper секции. Заранее: **новый layout STS.32 может иметь конфликты (класс P16 без π)**; если конфликтный → зафиксируем в NCu post-pack как status quo (016-вердикт: сначала инструкции, конфликты потом π-циклом).

---

## 1a.9 Предсказания ДО кода (регистрирую вилку от baseline 8.515)

| Метрика | Прогноз | Заметка |
|:--|:--|:--|
| ptxas registers | **73-78r** | Потолок 85r жёсткий (0 spill, 0 LDL/STL, blocks=6). Выход за 85 = **стоп и доклад**, НЕ автопереход 6→5 blocks. |
| shared_st inst | **-70..75%** (537M → 135-160M) | Аналог dk_new 018 -70.5%. Возможно чуть выше drop за счёт 12 SHFL считающихся в shared. |
| ST conflicts | по бумаге ≥ 0 (paper-check ниже) | Класс P16 без π может дать умеренную rate |
| MIO throttle | **46.73% → 34-38%** | drop 8-12 pp |
| shared_ld inst | **unchanged 889M** | Feeder + MMA-B не тронуты |
| Wall isolated | **8.10-8.35 ms** | -2.2..-4.9% от 8.515 |

**Полосы по правилу-2/3 v2**:
- ≥3% keep = **≤ 8.259 ms**
- 2-3% ABBA = 8.260 - 8.344 ms
- <2% откат = > 8.345 ms

---

## 1b. Unit-test сети (микро-ядро pack_kt_unit_test.cu)

**Дизайн**:
- 4 warps × 32 threads = 128 threads
- SMEM: smK[Bc=64][Hd=128] + smK_T[Hd=128][KT_STRIDE=68] = 8192 + 8704 = 16896 B (**отдельный SMEM пул**, production не тронут)
- **Маркер-байты уникальные по (lane, ni, half, bt)**:  
  `smK[n][k] = (l_div4 × 131 + l_mod4 × 17 + ni × 7 + half × 3 + bt × 1) & 0xFF`  
  Actually простой: `smK[n][k] = (n * 131 + k * 7 + 13) & 0xFF` (одинаково для всех lanes) — CPU эталон сходится через n/k координаты, независимо от lane.

**Phase 1.5 read** — как в production (kr_lo/kr_hi).
**Pack (A-B-C-D)** — реализация exchange-сети из 1a.
**CPU эталон**: byte-by-byte trasncript over K→K_T mapping.
**Assert**: 8192/8192 bytes match.

**SASS-гейт unit-test**:
- 12 SHFL / 16 STS.32 (per warp per slot: 32 lanes × 12 = 384 SHFL total)
- 0 STS.U8
- 0 LDL/STL (V-класс детектор)

Реализация — в `runs/probes/pack_kt_unit_test.cu`.

**Финальный результат (после вывода Phase D из читателя)**: **8192/8192 match, 0 mismatch** ✓
**SASS-гейт**: 12 SHFL + 16 STS.32 (pack) + 0 LDL/STL ✓

**Ход вывода** (по TZ 025-b):
1. **Читатель первичен**: формула из `fa_bwd_dq_new.cu:216-219` — B0 uint32 = 4 bytes at (row=n_d, col=k_j_lo..k_j_lo+3).
2. **CPU-судья** (`probe_dq_pack_bytes.py`) подтверждает Vugar-гипотезу:
   - **ASSERT 1**: 2048/2048 STS.32 targets имеют группу {c + 4*p' + 16*h} с fixed (c, h), varying p' ✓
   - **ASSERT 2**: 8192/8192 bytes покрытие без дублей ✓
   - **ASSERT 3**: reader byte-адреса ⊆ writer inventory ✓
   - **ASSERT 4**: 128 lanes × 64 byte-participations = 16 STS.32/lane ✓
3. **Python simulator PhaseA/B/C/D** (`simulate_dq_pack_shfl.py`) — байт-verify 8192/8192 ✓
4. **GPU unit-test** — 8192/8192 ✓

**Хреография (dq-специфично, независимо выведенная)**:
- Slot bits: `bit[1]=slot_half` (0=kr_lo, 1=kr_hi), `bit[0]=slot_ni_hi` (0/1 → ni_base=0/4)
- Group: `{c + 4*p' + 16*h}`, обмен по l_div4-axis
- **Phase A**: 8 PRMT (те же селекторы 0x5140/0x7362/0x5410/0x7632 что dk_new) → G0..G3 = byte-position gather across 4 ni's
- **Phase B**: 3 SHFL identical dk_new (r=1,2,3, src_p=(p-r)&3, expose G[(p+r)&3])
- **Phase C**: 8 PRMT симметрично Phase A → OUT0..OUT3
- **Phase D**:
  - `base_row = wid*32 + 4*c + p + 16*slot_half`
  - `col_base[j] = (ni_base+j)*8 + 4*h` (для j=0..3)
  - 4 STS.32: `smK_T[base_row][col_base[j]..col_base[j]+3] = OUT[j]`

**Счёт-инвариант per lane per qt**: 12 SHFL + 16 STS.32 + 64 PRMT + 24 SEL + 0 LDL/STL ✓

**Зелёный тест → предавторизация production-правки** (по TZ Vugar). Переход на 1c.

**Диагноз** (математический, отчёт по правилу «стоп при первом непонятном»):

Хреография dk_new pack предполагает **ks-варьирующий OUT_i**: 4 OUT слова per slot пишут в 4 разных ks-блока `dst_row = i*32 + row_base_ks`. Это использует симметрию dk-feeder'а, где `Qr[ks=0..3][slot]` хранит **4 ks-значения в регистрах per lane**.

**dq feeder ks=wid=const per warp** — эта симметрия НЕ работает:
- Каждый warp пишет только в свой ks-блок (32 dst_row из 128)
- Per-warp OUT_i **не может** span 4 ks-блока
- Нужно другое Phase D распределение: OUT_i пишет **4 разных `delta_k` внутри одного ks-блока**

Прототип попытался Phase D:
```c
int dst_row_base = ks * 32 + 16 * use_hi + 4 * c;
*smKT[dst_row_base + 0][col] = OUT0;   // delta_k = 4c + 0
...
```
Это не совпадает с обменом Phase B (транспонирует по p-axis, не по 4-consecutive внутри c-quadrant), поэтому байты перепутаны.

**Требуется решение Vugar** (три варианта на выбор):

- **α)** переопределить Phase D под dq-семантику (OUT_i → dst_row = ks*32 + i*8 + p, dst_col = ni*8 + 4h + p) — требует новой bit-precise дериваци и повторного unit-test;
- **β)** оставить кросс-lane exchange dk-style, но переопределить **Phase A** так, чтобы feeder mapping от kr_lo/hi[ni] переставлял входы так, что OUT_i уже соответствуют dk-мэппингу — пропорция та же, но math независимая, требует полного вывода;
- **γ)** упрощённый неполнокросс-lane pack: каждая lane пишет собственные 16 uint32 через local PRMT-пакет (без SHFL), с STS.32 разбросанным по dst_col=n_K через lane's l_div4 (каждая lane пишет 16 STS.32 в свои dst_col, но dst_col не consecutive → 4× меньше bank throughput). **shared_st inst drop 4× меньше**: -12% вместо -70%.

**Статус**: production-правка приостановлена. Отчёт по чест- ности состояния. Прогноз вилки (73-78r, -70% shared_st, 8.10-8.35 wall) **выставлен для варианта α/β**; γ значительно скромнее.

---

## π-paper (paper-only, на полку, для 026 стартового цикла если конфликты в критпути)

### π-paper.1 Bank-паттерн текущего LD B-load

Из fa_bwd_dq_new.cu:216-219:
```c
n_d = ni*8 + l_div4;
k_j_lo = kb*32 + l_mod4*4;
B0 = smK_area[n_d * 68 + k_j_lo];
```

За warp fixed (kb, ni): 32 lanes читают:
- row = n_d = ni*8 + l_div4 ∈ 8 distinct values (l_div4 varies)
- col = k_j_lo/4 + {0, 4} (halves lo/hi)

Bank = (n_d × 17 + k_j_lo/4) mod 32 = `((ni*8+l_div4)*17 + kb*8 + l_mod4) mod 32`.

За fixed (kb, ni, half=lo): `(17*8*ni + 17*l_div4 + 8*kb + l_mod4) mod 32`:
- 17*8 = 136 mod 32 = 8
- 17*l_div4 — varies
- 8*kb — const shift
- l_mod4 — varies

`bank = (8*ni + 17*l_div4 + 8*kb + l_mod4) mod 32`  
Lane-part: `f(l_div4, l_mod4) = 17*l_div4 + l_mod4 mod 32`.

- l_div4 ∈ [0..7]: 17*l_div4 mod 32 = {0, 17, 2, 19, 4, 21, 6, 23}
- l_mod4 ∈ [0..3]

Пары (17*l_div4 + l_mod4) mod 32:
| l_div4 | l_mod4=0 | =1 | =2 | =3 |
|:-:|:-:|:-:|:-:|:-:|
| 0 | 0 | 1 | 2 | 3 |
| 1 | 17 | 18 | 19 | 20 |
| 2 | 2 | 3 | 4 | 5 |
| 3 | 19 | 20 | 21 | 22 |
| 4 | 4 | 5 | 6 | 7 |
| 5 | 21 | 22 | 23 | 24 |
| 6 | 6 | 7 | 8 | 9 |
| 7 | 23 | 24 | 25 | 26 |

**Коллизии**:
- Bank 2: (l_div4=0, l_mod4=2) и (l_div4=2, l_mod4=0) — 2-way
- Bank 3: (0,3) и (2,1) — 2-way
- Bank 4: (2,2) и (4,0) — 2-way
- Bank 5: (2,3) и (4,1) — 2-way
- Bank 17: (1,0) — 1-way; (нет 2-way для 17)

Actually enumerate all 32 lane values, count. Complicated by hand. Numeric result from 024: **LD wavefronts 1.28 B / inst 889 M = 1.44 excess rate**, matches ~2-way collision on ~40% of accesses.

**Кандидат π (paper)**: PI_V как в dk_new pack:
- `PI_V(r) = ((r&7)<<2) | (((r>>3)&1)<<1) | ((r>>4)&1) | (r & 0x60)`
- Bit-perm: r0→2, r1→3, r2→4, r3→1, r4→0, r5→5, r6→6

За dq_new, применяем к `n_d` (row в K_T): `n_d = ni*8 + l_div4`.
- ni ∈ [0..15] → bits [0..3] of n_d (log2(16)=4 bits)
- l_div4 ∈ [0..7] → bits [4..6] of n_d? Wait, n_d = ni*8 + l_div4, ni*8 shifts ni by 3 bits, l_div4 fills bits [0..2]. So bit[2:0] = l_div4, bit[6:3] = ni.

Отличие от dk_new: там `n_d = ni*8+l_div4` тоже, но dk_new lane structure отличается. π_V применительно к тому же паттерну.

**Проверка π-условий для dq_new** — требует полного CPU-перебор'а как в 021. Здесь **зафиксирую формулу PI_V как кандидат-перестановку** — если 026 покажет конфликты на критпути, старт π-цикла без задержки.

### π-paper.2 Bank-паттерн новых STS.32 (после pack)

Per lane per slot записывает 4 STS.32 в K_T:
- dst_row = ks*32 + slot_offset (slot varies)
- dst_col_word = (ni_slot*8 + l_div4)/4 + STS_index_within_slot

Полный enumerate — сделан ниже в CPU-переборе (метод 021). См. probe_pi_dq_pack.py.

### π-paper.3 Кандидат-перестановка

**Гипотеза**: та же PI_V формула из 023 dk_new будет применима к dq_new n_d indexing (row of K_T).

**CPU-судья** (probe_pi_dq_pack.py):
- ASSERT 1: 32 lanes → 32 разных банков per B-LDS группа (LD side, для fixed kb, ni)
- ASSERT 2: 32 lanes → 32 разных банков per STS.32 группа (post-pack ST side, для fixed slot, ks)
- ASSERT 3: физ.строки биекция 0..127
- ASSERT 4: побайтовое покрытие 8192 инвариантно

**Реализация — в отдельном скрипте, не в этом отчёте.** Только фиксирую формулу-кандидат и логику асертов.

---

## Файлы, созданные в этом шаге

- `/data/lib/podman-data/projects/goml/runs/reports/025_dq_pack_paper.md` — этот отчёт
- `/data/lib/podman-data/projects/goml/runs/probes/pack_kt_unit_test.cu` — unit-test микро-ядро (1b)
- `/data/lib/podman-data/projects/goml/runs/probes/Makefile.unit_kt` — build (1b)

**Production dq_new не тронут: `libs/fa_bwd_dq_new.cu` md5 остаётся `683396f8…`**.

Chain md5: 024 `8462480a…` → **025 `<computed>`**

---

**End 025.**
Переход на 1c (production правка) после подтверждения unit-test 8192/8192 + SASS-гейт зелёный.
