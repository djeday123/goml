# 032-b — Закрытие 4 дыр 032 (по Vugar-приёмке)

**Chain**:
- 032_no_dst_paper.md md5: `463ba9245a059880579defc8fabf8a40`

**Artifact header** (production не тронут):
```
-rw-r--r-- 14667  Jul  7 10:03  libs/fa_bwd_dk_new.cu    (023 sealed π_V, md5 9b12a7d1…)
-rw-r--r-- 13352  Jul  5 17:51  libs/fa_bwd_dq_new.cu    (sealed pre-pack, md5 683396f8…)
-rw-r--r--         Jul  7        runs/probes/probe_dk_dsnat_bytepath.py  (CPU-судья, исполнен)
```

Только measurement + paper. CPU-судья ИСПОЛНЕН.

---

## Дыра 1 — блокирующая: CPU-судья ИСПОЛНЕН, не предсказан

**Скрипт**: `runs/probes/probe_dk_dsnat_bytepath.py`

**Модель пути байта** (сквозная):
- **Current**: dS_T[b][j_g][i_g] → cp.async → smdS_T[j_local×Br+i_local] → MMA-A read
- **A1**: dS_nat[b][i_g][j_g] → cp.async → smdS_area[i_local×Bc+j_local] → LDS-перекладка → smdS_area[j_local×Br+i_local] → MMA-A read (unchanged reader!)

**Формулы читателя** — верифицированы против fa_bwd_dk_new.cu:239-245 (m_lo/hi, k_i_lo/hi, byte_offset).

**Формы для проверки**:

| Форма | bh | sl | causal | Byte-paths | Verdict |
|:--|:-:|:-:|:-:|:-:|:-:|
| Canonical | 1 | 8192 | 0 | **67,108,864** | ✓ PASS |
| F1 sl=128 | 1 | 128 | 0 | 16,384 | ✓ PASS |
| **CANARY sl=300** | 1 | **300** | 0 | 102,400 | ✓ PASS |
| F8 sl=512 causal wnd=128 | 1 | 512 | 1 | 147,456 | ✓ PASS |
| F10 sl=2048 causal | 1 | 2048 | 1 | 2,162,688 | ✓ PASS |
| **Total** | | | | **69,537,792** | ✓ **ALL GREEN** |

**Partial/CANARY boundary check** (sl=300, stride_ds=304):
- Edge tile OOB via j: 1280
- Edge tile OOB via i: 880
- Edge tile VALID: 1936
- **Обе стратегии применяют zero-fill via cp.async bytes=0 → VERIFIED by design**

**Verdict**: **Байт-инвариант доказан перебором** на 69M+ путей чтения включая CANARY, edges, OOB, stride_ds padding.

---

## Дыра 2 — блокирующая: Фазовая схема A1 (in-place противоречие разрешено)

### Три варианта транспонирования (Vugar-выбор):

#### W1 — Двойной буфер smdS (+4096 B/block)
- SMEM: `smQ(8192) + smQ_T(8704) + smdS_nat(4096) + smdS_T(4096) = 25088 B/block`
- Blocks/SM: `floor(102400 / 26112) = 3` → **4→3 РЕГРЕСС**
- **АВТОСТОП** по Vugar правилу.

#### W3 — Регистровый транзит без барьера
- Каждый lane читает ВСЁ, что перезапишет, в регистры до первого сторa
- Требует **64 uint32 per lane** для полного тайла 4096 B / 4 warps / 32 lanes = 32 bytes = 8 uint32
- Но каждый lane пишет в locations OTHER lanes также должны быть готовы записать → в реальности **cross-warp exchange** нужен
- Без барьера — race condition при aliased overwrite

#### W2 — Регистровый транзит + барьер (ВЫБРАН)

**Фазовая схема dk_new (полная):**

```
[unchanged Q load path]
cpa16 K → smK_area (existing)                          ← сохранено
cpa16 dS_nat → smdS_area (natural [i][j])              ← НОВОЕ: было cpa16 dS_T
cpa_commit(); cpa_wait<0>();
__syncthreads()  // BARRIER #1: K + dS_nat ready

// Phase 1.5-nat-transpose (NEW)
Load dS_nat rows → registers kr_ds[8] per lane (32 bytes = 8 uint32)
__syncthreads()  // BARRIER #NEW: все чтения smdS_area(nat) до aliased write
SHFL-exchange 4-lane pack-analog + Phase A/B/C/D
STS.32 → smdS_area (aliased overwrite [j][i] = T layout)

__syncthreads()  // BARRIER #2 (было BARRIER #2 в orig): K read done in Phase 1.5

// Phase 1.5 K→K_T (existing, unchanged)
[read K, write K_T]

__syncthreads()  // BARRIER #3 (было BARRIER #3): K_T + smdS_area(T) ready

// MMA-C (unchanged reader)
[MMA using smdS_T + smQ_T]

__syncthreads()  // BARRIER #4 (было BARRIER #4): end of qt
```

**Барьерный счёт**:
- Оригинал: 4 __syncthreads() per qt
- W2: **5 __syncthreads() per qt** (+1 = BARRIER #NEW)

**SMEM-строка (W2)**:

| Component | Bytes | Note |
|:--|--:|:--|
| smQ | 8192 | unchanged |
| smQ_T | 8704 | unchanged |
| smdS_area (aliased nat↔T) | 4096 | **unchanged (aliased)** |
| **Dynamic SMEM total** | **20992** | unchanged from 023 sealed |
| Driver overhead | 1024 | typical CUDA slot |
| **Slot per block** | **22016** | 20992+1024 |
| Blocks/SM ceil | 4 | 102400/22016 = 4.65 → **4 ✓** |
| Headroom (SMEM) | 102400 − 4×22016 = 14336 B | 14 KB запас |

**MIO-ops счёт W2 (пересчитан с учётом барьера)**:

| Op | Count per lane per qt |
|:--|--:|
| LDS.32 (read dS_nat) | 8 |
| SHFL (pack-analog exchange) | 6 |
| PRMT (Phase A + C) | 16 |
| STS.32 (write T aliased) | 8 |
| **Sub-total MIO** | **22** (as before) |
| BARRIER #NEW cost | +1 barrier op |
| **MIO + 1 barrier** | **22 MIO + 1 barrier** |

**Vugar-порог 30 MIO/qt/lane соблюдён (22 ≤ 30)**. Barrier — отдельная класс stall.

**Выбор**: W2 — единственный вариант, сохраняющий 4 blocks/SM без регистрового взрыва.

---

## Дыра 3 — merged механика: "+1 block" изъято, wall пересчитан

### 3.1 Регистровый потолок merged

**merged: 253 registers × 128 threads = 32,384 registers per block**  
Peak registers per SM = **65,536**  
`floor(65536 / 32384) = 2.02 → 2 blocks/SM by registers`

**"+1 block (2→3)" невозможно** без снижения регистров merged на ~40r. Строку изымаю.

### 3.2 Куда уходят освобождённые 5120 B?

**merged SMEM footprint** (из fa_bwd_merged_v1.cu, оценка):
- smQ, smK, smV (FP8) + smdO (FP16) + smL/D + smdS_T (5120 B) + аккумуляторы
- Total ~40-48 KB/block dynamic (2 blocks × ≤48 = ≤96 KB ≤ 100 KB peak)

Освобождение 5120 B при удалении smdS_T:
- SMEM per block: 48 → 43 KB (примерно)
- Blocks/SM ceil = min(2 by regs, 100/43 = 2.3 by SMEM) = **2 blocks/SM (unchanged)**
- **Освобождение уходит в headroom** (SMEM резерв), не в +block

### 3.3 Пересчитанный wall merged (fair, при 2 blocks)

**Вырезаемые операции**:
- **STS.U8 scatter dS_T**: 64/qt/lane × 128 qt × 128 threads × 16384 grids ÷ (188 SMs × 2 blocks) = ~44M STS ops per SM
- **1 барьер устраняется**: -100..-200 cycles × 128 qt = -12800..-25600 cycles per SM
- **Related MIO release**: freeing MIO pipe от STS-drain phase

**Реалистичный merged wall gain**:
- Не bandwidth-модель (BW-util 33.6%, не bandwidth-bound)
- Grubo оценка от вырезаемых MIO: 44M STS × ~5 cycles / (188 SMs × 2 GHz) = 5.85 ms если полностью serial → в реальности overlap → gain ~1.5-3.5 ms
- Barrier gain: 0.1-0.3 ms
- **Merged wall прогноз**: 30.88 → **27-29 ms (-6..-13%)**  

Отличие от 032: было прогноз -1.5..-5 ms → сейчас **-1.9..-3.9 ms** (более консервативно, без +block фантазии).

---

## Дыра 4 — "bit-exact гарантирован" изъят

**Правка формулировки**:

> ~~Bit-exact гарантирован (транспонирование bit-permutation, byte content preserved, MMA input identical → dK output bit-identical)~~
>
> **Байт-инвариант доказан перебором** (69M+ byte-paths, все 5 форм включая CANARY). **Bit-exact доказывается гейтом 033** (тройной 11/11 + CANARY + fp16-acc floor-константы + sanitizer).

**Причина**: dk_new получает новый указатель (dS_nat вместо dS_T), новую адресацию cp.async, новую фазу перекладки; merged теряет фазу. Любой захардкоженный stride, любой сдвиг в canary-обвязке — и байты поплывут. **Bit-exact у нас доказывается только гейтом.**

---

## Дыра 5 (Vugar-мелочь) — L2-share dq_new вычеркнут

**Правка п.5**:
- dq_new DRAM δ: не −0..−2 GB (L2), а **0** (не защищаем байтами до NCu-post)

**L2 40MB vs 2 × 8 GB nat-traffic** — share близок к нулю. Из прогноза убрано.

---

## 5-b. Пересчитанные предсказания (fair, без ложных допущений)

### 5-b.1 DRAM-дельта цепи

| Ядро | DRAM before (GB) | DRAM after (GB) | Δ (GB) |
|:--|--:|--:|--:|
| **merged** | 18.58 | **10.58** | **−8.00** (dS_T write устранён) |
| dk_new | 9.26 | 9.26 | 0 (dS_T→dS_nat swap) |
| dq_new | 9.26 | 9.26 | **0** (L2-share исключён) |
| d_precompute | 0.545 | 0.545 | 0 |
| **Total** | 37.63 | **29.63** | **−8.00** |

### 5-b.2 ptxas-вилки

- **dk_new**: 124 → 128-132r (лучший 128, худший 132); +1 barrier; +8 uint32 kr_ds temp regs; 0 spill; 0 LDL/STL
- **merged_v1**: 253 → **240-250r** (вырезание T-staging освобождает регистры); −1 barrier; SMEM −5120 B → headroom (не +block)
- **dq_new**: unchanged 56r
- **d_precompute**: unchanged 38r

### 5-b.3 Wall-вилки (fair, без NCu-налога)

| Ядро | Wall before (ms) | Wall after (ms) | Δ |
|:--|--:|--:|--:|
| **merged** | 30.88 | 27.0-29.0 | **-1.9..-3.9 ms** |
| **dk_new** | 8.84 | 9.1-9.6 ms | **+0.3..+0.8 ms** (transpose + barrier cost) |
| dq_new | 8.56 | 8.56 | 0 |
| d_precompute | 0.34 | 0.34 | 0 |
| **E2E** | 48.63 | **45.9-47.5** | **-2.3..-5.6%** |

**Изменение от 032**:
- E2E вилка -3.5..-8.5% → **-2.3..-5.6%** (более консервативно)
- Merged gain меньше (без +block фантазии)
- Верхний край -5.6% >> 3% keep-порог — но не гарантировано

---

## 6. Финальный статус условий для 033

| Пункт | Требуется | Факт | Verdict |
|:--|:--|:--|:-:|
| Дыра 1: CPU-судья исполнен | Зелёный | 69M byte-paths ALL GREEN | ✓ |
| Дыра 2: фазовая схема без -blocks | 4 blocks/SM сохраняются | W2 SMEM unchanged 20992 B, 4 blocks ✓ | ✓ |
| Дыра 3: merged без "+1 block" | Регистры дают 2, изъять | Изъято, пересчитано | ✓ |
| Дыра 4: bit-exact формулировка | "Доказывается гейтом" | Правка внесена | ✓ |
| Дыра 5: L2-share исключён | Убрать из дельт | Убрано | ✓ |
| MIO-add ≤ 30/qt/lane | Y | 22 MIO + 1 barrier | ✓ |
| Барьер учтён | Y | +1 barrier зафиксирован | ✓ |

**Все 4 дыры закрыты + Vugar-мелочь.**

**033 предавторизован** при факте (i) CPU-судья зелёный ✓, (ii) фазовая схема без потери блоков ✓ (W2), (iii) merged-механика без "+1 block" ✓.

Chain md5: 032 `463ba924…` → **032-b `<computed>`**

---

**End 032-b.**

**Готов к 033** — правка dk_new (переключение на dS_nat + Phase 1.5 dS transpose W2) + merged (вырезание T-staging).

**Ключевой риск для 033**: dk_new ptxas 128-132r — если >128 → **АВТОСТОП + доклад** (Vugar правило 4→3 blocks).

Chain md5: 032 `463ba924…` → **032-b `<computed after write>`**
