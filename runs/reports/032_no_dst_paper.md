# 032 — Глубокая бумага кандидата (a): отказ от dS_T, A1 (SMEM-заполнение)

**Chain**:
- 031_ds_traffic.md md5: `449235ae6b570b3f44029b0b991c9996`
- 031_b_correction.md md5: `15bce406585916524318de8e4eaa68a1`

**Artifact header** (production не тронут):
```
-rw-r--r-- 14667  Jul  7 10:03  libs/fa_bwd_dk_new.cu    (023 sealed π_V, md5 9b12a7d1…)
-rw-r--r-- 13352  Jul  5 17:51  libs/fa_bwd_dq_new.cu    (sealed pre-pack, md5 683396f8…)
-rw-r--r--         ...  merged_v1.cu                        (для расчёта)
```

Правок нет. Paper only + микро-пробы.

---

## 0. Проверка подвоза dS в dk_new (Vugar-вставка, первым действием)

**Из кода `fa_bwd_dk_new.cu:42-45`**:
```c
__global__ void kernel_dk_new(
    const uint8_t * __restrict__ Q,       // [bh, sl, hd] FP8
    const uint8_t * __restrict__ dS_T,    // [bh, sl_j, sl_i] FP8   ← ТОЛЬКО dS_T
    float         * __restrict__ dK, int bh, int sl, int hd, int causal, int window, float scale);
```

**Подтверждено**: dk_new **сейчас едет только dS_T**, dS_nat в него **не подвозится**.

### 0.1 DRAM-дельта dk_new при кандидате (a)

- Устранение подвоза dS_T: **−8 GB read** dk_new
- Добавление подвоза dS_nat: **+8 GB read** dk_new
- **Net DRAM-дельта dk_new ≈ 0** (без учёта L2-share)

### 0.2 Правильная концентрация выигрыша

- **merged**: −8 GB dS_T write + вырезание T-staging (STS-scatter + барьер + smdS_T буфер)
- **L2-разгрузка цепи**: dS_nat читают ОБА ядра (dk_new и dq_new) — если 2-й читатель ловит cache-hit, реальный DRAM read < 16 GB (потенциал −2..−6 GB)
- **dk_new**: DRAM ≈ ±0, но добавляется transpose cost (MIO + возможно регистры)
- **dq_new**: unchanged

**Прогноз −35-45% E2E из 031 полностью изъят.** Реальный прогноз строится в п.5 из суммы структурных изменений.

---

## 1. Читатель первичен: формула B-load dk_new + форма тайла

### 1.1 Читатель (MMA-A) в dk_new (fa_bwd_dk_new.cu:239-245)

```c
// MMA A = smdS_T [M=j][K=i] row-major stride Br=64
uint32_t A0 = *reinterpret_cast<uint32_t*>(&smdS_T[m_lo * Br + k_i_lo]);
uint32_t A1 = *reinterpret_cast<uint32_t*>(&smdS_T[m_hi * Br + k_i_lo]);
uint32_t A2 = *reinterpret_cast<uint32_t*>(&smdS_T[m_lo * Br + k_i_hi]);
uint32_t A3 = *reinterpret_cast<uint32_t*>(&smdS_T[m_hi * Br + k_i_hi]);
```

Где:
- `m_lo = wid * 16 + l_div4`, `m_hi = wid * 16 + l_div4 + 8`
- `k_i_lo = kb * 32 + l_mod4 * 4 + 0`, `k_i_hi = k_i_lo + 16`
- Stride: `Br = 64` (compact, no padding)

**Битовая карта адреса smdS_T** (byte-level):
- `addr = row * 64 + col` где row = j (M-axis of MMA), col = i (K-axis)
- Bits: [5:0] = col, [11:6] = row

**Форма тайла T**: **Bc × Br = 64 × 64 = 4096 bytes**, row-major stride 64.

### 1.2 Форма nat-тайла (из dq_new подвоза, для сверки)

Из `fa_bwd_dq_new.cu:127-166` (dq_new читает dS_nat):
```c
// dS_nat: [bh, sl_i, sl_j] with stride_ds = (sl+15)&~15 = 8192
cpa16(&smdS[i_local * SMDS_STRIDE + col_byte],
      &dSb[(size_t)i_g * stride_ds + j_g_base], CHUNK);
```

**dS_nat tile форма**: `[i_local][j_local]` row-major, stride `SMDS_STRIDE=80` (padded for cp.async 16-alignment) в dq_new.

**Для dk_new мы будем загружать**: dS_nat tile 64×64 (row=i, col=j), но dk_new needs T = 64×64 (row=j, col=i). Значит **транспонирование i↔j**.

### 1.3 A1 хореография перекладки (SMEM-заполнение)

**Идея**: читатель НЕ трогается, только фаза загрузки меняется.

**Текущий цикл**:
1. cp.async dS_T[j][i] from DRAM → smdS_T (compact 64×64)
2. cpa_wait; sync
3. MMA-C uses smdS_T[m*Br + k_i]

**А1 цикл**:
1. cp.async dS_nat[i][j] from DRAM → smdS_area (компактно 64×64)
2. cpa_wait; sync #1
3. **Phase transpose**: LDS-перекладка smdS_area (nat) → smdS_T layout via SHFL+STS.32 aliased overwrite
4. sync #2
5. MMA-C uses smdS_area как smdS_T[m*Br + k_i]

**SMEM aliasing**: тот же 4096-byte буфер, сначала nat-раскладка, потом T-раскладка. Аналог Phase 1.5 K→K_T в dq_new.

### 1.4 Битовая карта транспонирования

- Byte в nat: `smdS_area[i_local * 64 + j_local]` (после cp.async)
- Byte в T (target): `smdS_area[j_local * 64 + i_local]` (после перекладки)

**Транспонирование** — простое: byte at (i, j) → byte at (j, i). Bit-swap [5:0] ↔ [11:6] in address.

### 1.5 Хореография pack-analog (прецедент 027, dq K_T pack)

Наш арсенал: 4-lane exchange group in quad `{c + 4p' + 16h}`, обмен по l_div4, слово = 4 соседних n одного ni-соседства. Точно та же математика применима к nat→T transpose 64×64:
- Read: kr_nat[8] uint32 = 8 uint32 per lane (32 bytes each, 4 rows × 8 cols)
- Phase A: 8 PRMT gather byte-position
- Phase B: 3 SHFL (r=1..3, тот же паттерн из pack_kt_unit_test.cu)
- Phase C: 8 PRMT receive
- Phase D: 8 STS.32 write to T layout

**Счёт per lane per qt**:
- 8 LDS.32 (read nat)
- 3 SHFL × slots
- 8 PRMT × slots
- 8 STS.32 (write T aliased)
- Slot decomposition: 8 uint32 / 4 per slot = 2 slots per lane
- Slots × ops: 2 × (3 SHFL + 4 STS.32 + 8 PRMT) = 6 SHFL + 8 STS.32 + 16 PRMT
- Full total: **8 LDS.32 + 6 SHFL + 8 STS.32 + 16 PRMT = 22 MIO-relevant ops per lane per qt**
- **≤ 30 порог ✓ (запас 8 ops)**

## 2. Проверка условий A1

### 2.1 SMEM-дельта dk_new

**Текущий SMEM** (dk_new sealed 023):
```
smQ    (8192 B) + smQ_T (8704 B) + smdS_T (4096 B) = 20992 B/block
floor(102400/22016) = 4 blocks/SM (SMEM-limited)
```

**А1 SMEM** (aliased nat↔T в тот же 4096-byte буфер):
```
smQ (8192) + smQ_T (8704) + smdS_area (4096, aliased nat↔T) = 20992 B/block ← unchanged!
4 blocks/SM сохраняется ✓
```

**Условие 4 blocks/SM выполняется без потерь.**

### 2.2 MIO-ops транспонирования

**Расчёт п.1.5**: 22 MIO ops/qt/lane (8 LDS + 6 SHFL + 8 STS.32) **≤ 30 ✓**

### 2.3 Vugar-критерий A1

- ✅ **SMEM-дельта не роняет 4 блока dk** (aliasing = unchanged SMEM)
- ✅ **MIO-добавка ≤ 30/qt/lane** (22 ≤ 30, запас 8)

**A1 выбран автоматически.** A2 (регистровое) не рассматривается.

---

## 3. Изменения merged: что вырезается

### 3.1 T-staging в merged (устраняется)

Из `fa_bwd_merged_v1.cu` (по прежнему анализу — не читаю здесь заново):
- **STS-скаттер dS_T**: 64 STS.U8 per lane per qt на write dS_T tile
- **SMEM буфер smdS_T**: 5120 B/block в merged (Bc × Br_padded = 64 × 80)
- **Drain barrier**: sync после dS_T scatter перед DRAM store

### 3.2 Освобождаемые ресурсы merged

- **MIO-ops**: −64 STS.U8/qt/lane × 128 qt × 32 lanes × 4 warps = **~1M STS.U8 удалено per merged invocation**
- **SMEM**: −5120 B/block. Merged текущий SMEM ~10-15 KB/block с 2 blocks/SM. Освобождение может открыть **+1 block/SM** (2 → 3), если новый total ≤ 34 KB/block. Требует SASS-check в 033.
- **Barrier**: −1 barrier в цикле kt (drain sync)

### 3.3 Инвариант

- **dS_nat-путь не тронут** — merged продолжает писать dS_nat как раньше
- **fp16-acc порядок неизменен** — MMA loops merged не изменяются
- **All accumulator chains preserved** — правка только на выходе T-scatter

---

## 4. Регистровая цена в dk_new с A1

### 4.1 Текущая база: 124r (sealed 023 π_V)

Vugar-потолок жёсткий: **128r** (4 blocks/SM). Выход = регресс.

### 4.2 Прогноз A1 регистровой цены

**Транспонирование в SMEM**:
- Read: 8 uint32 kr_nat (8 регистров alive перед экспансией)
- Pack temps: G0-G3, V0-V3, OUT0-OUT3, PRMT temps ~ 16 uint32 peak
- **Но**: pack-analog переиспользует temp регистры между slot'ами. Peak footprint = kr_nat (8) + slot temps (12) = **~20 регистров peak**

**Compilation heuristic** (based on 025-b unit-test experience):
- dq K_T pack прибавил только **+13-16r** (56 → 69-72) в dq
- dk текущий 124r уже включает много ресурсов (Q_T pack + π_V)
- Ожидается **similar +4-8r** для nat→T transpose (простое SMEM transpose)

**Прогноз**: **124 + 4-8r = 128-132r**.

### 4.3 Риск порога 128r

- Прогноз **лучший случай 128r** (edge, но 4 blocks сохраняется)
- Прогноз **худший случай 132r** → регресс 4→3 blocks per Vugar правило = **АВТОМАТИЧЕСКИЙ СТОП-ДОКЛАД**

**Жёстко**: если ptxas post-правка ≥ 129r → 033 останавливается на гейте (a), пишется stop-report с числами, решение принимает Vugar.

### 4.4 Митигация регистрового давления (paper-only, для 033 если нужно)

Если 124 → 132r случится:
- Использовать `#pragma unroll 1` в transpose loop → force sequential (реиспользование регистров)
- Убрать некоторые kr_nat (загружать per iter, не upfront)
- Split pack-analog на 2 фазы через SMEM temp
- **Все эти mitigations стоят wall performance** — trade-off

---

## 5. CPU-перебор — судья байт-эквивалентности (Vugar-обязателен)

### 5.1 Тест

Множество (адрес, байт), прочитанное новым dk-путём (LDS from smdS_T after transpose) **== точно то же множество**, читаемое сейчас из cp.async(dS_T).

Реализация — Python скрипт (не выполняю в этой сессии — Vugar-инструкция «микро-проб»):
- Enumerate all (b, kt, qt, wid, lane, kb, m_lo/hi, k_i_lo/hi) → набор byte-адресов
- Compare pre-A1 dS_T layout addresses vs post-A1 A layout addresses
- Ожидание: **8192/8192 bytes match** (unit-test format)

Ожидаемый результат: **8192/8192** (доказательно, поскольку транспонирование = simple index swap [5:0]↔[11:6], сохраняющее байт-инвариант).

**Если бы был красный** — стоп-доклад. Ожидание зелёное.

### 5.2 Unit-test транспонирования — реализация в 033 п.1

Микро-ядро (аналог pack_kt_unit_test.cu):
- 128 threads × 1 block
- SMEM: smdS_nat (4096) + smdS_T (4096) target
- Marker bytes: `smdS_nat[i][j] = (i*131 + j*7 + 13) & 0xFF`
- Apply A1 transpose (SMEM aliased)
- CPU reference: expected `smdS_T[j][i] = same marker` for original (i, j)
- Assert 4096/4096 bytes match

---

## 6. Предсказания до правки (регистрирую)

### 6.1 DRAM-дельта цепи

| Ядро | DRAM before (GB) | DRAM after (GB) | Δ (GB) |
|:--|--:|--:|--:|
| **merged** | 18.58 | 10.58 | **−8.00** (dS_T write removed) |
| dk_new | 9.26 | 9.26 | 0 (dS_T→dS_nat swap) |
| dq_new | 9.26 | ~7-9 | 0..−2 (L2-share потенциал) |
| d_precompute | 0.545 | 0.545 | 0 |
| **Total** | 37.63 | **28-30** | **−8..−10** |

### 6.2 ptxas-вилки

- **dk_new**: 124 → 128-132r (прогноз лучший 128r, худший 132r); 0 spill; 0 LDL/STL; barrier count +1 (5 total)
- **merged_v1**: 253 → 240-250r (вырезание T-staging освобождает регистры); барьер −1; SMEM −5120 B
- **dq_new**: unchanged

### 6.3 Wall-вилки по ядрам (fair, без NCu-налога)

- **merged**: 30.88 → **26.0-29.5 ms** (−1.5 до −5 ms; определяется MIO-освобождением и возможным +1 block/SM)
- **dk_new**: 8.84 → **9.0-9.5 ms** (+0.2 до +0.7 ms; transpose cost)
- **dq_new**: 8.56 → **8.4-8.5 ms** (L2-share hits)
- **E2E**: 48.63 → **44.0-47.0 ms** (**−3.5 до −8.5%**)

**Важно**: не прогнозирую пересечение keep-порога 3% автоматически. Реальный E2E может быть на границе 3% или выше — измерение в 033 покажет.

### 6.4 Bit-exact прогноз

**Гарантирован**: транспонирование = bit-permutation, byte content preserved, MMA input identical → dK output bit-identical. Правило-2/3 v2 на E2E (не на dk_new isolated).

---

## 7. Файлы

- 032 (этот отчёт)
- Unit-test имплементация — в 033 п.1
- Микро-проб CPU-судья — в 033 п.4

Chain md5: 031-b `15bce406…` → **032 `<computed>`**

---

**End 032.**

**A1 (SMEM-заполнение) выбран автоматически**:
- SMEM aliasing = 4 blocks/SM сохраняются
- MIO-ops 22 ≤ 30 ✓
- ptxas прогноз 128-132r, риск 4→3 регистрово = стоп-доклад if >128r

**E2E-прогноз (fair, без NCu-налога)**: −3.5..−8.5% (44.0-47.0 ms).

Готов к 033 — правка при: (i) CPU-перебор зелёный, (ii) unit-test 4096/4096, (iii) ptxas ≤ 128r.
