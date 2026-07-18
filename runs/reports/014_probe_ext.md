# 014 — Bank-probe ext: STS-scatter (P6/P7) + A-load (P8) + XOR-swz (P9) (2026-07-06)

## Статус: PATH B. Правку QT_STRIDE НЕ делаю. Production код без изменений.

## ARTIFACT-HEADER

### Кросс-верификация
```
md5 013_bank_probe.md:  90d6e458baebf49442791af8df2f09d1
md5 012_dk_stride80.md: 24087f5925db2a7c8386b78ae04ad0e1
```

### ls -la runs/probes/
```
-rw-r--r--  6086 Jul 6 16:15  fa_probe_bank.cu               (9 template patterns)
-rw-r--r--   499 Jul 6 15:54  Makefile
-rwxr-xr-x       Jul 6 16:15  fa_probe_bank                  (rebuilt with P6-P9)
-rw-r--r--       Jul 6 16:15  probe_build.log
-rwxr-xr-x   704 Jul 6 16:15  014_probe_ext.sh
-rwxr-xr-x   620 Jul 6 15:59  013_probe_ncu.sh
-rwxr-xr-x   405 Jul 6 15:57  n_scan.sh
```

Production файл `libs/fa_bwd_dk_new.cu` — БЕЗ изменений (QT_STRIDE=68 unchanged).

## 014-a — Формулы ИЗ КОДА (не из модели)

### STS Q_T-scatter (fa_bwd_dk_new.cu:154-169, isolated)
```c
for (int ks = 0; ks < KS_QK; ++ks) {
    int k_lo_base = ks * 32 + l_mod4 * 4;     // row varies: l_mod4 → +4 per unit
    int m_lo_q    = wid * 16 + l_div4;        // col varies: l_div4 → +1 per unit
    for (int bt = 0; bt < 4; ++bt) {
        smQ_T[(k_lo_base + bt) * QT_STRIDE + m_lo_q] = byte;   // STS.U8
    }
}
```
- **byte_addr = (l_mod4*4 + bt + ks*32) × QT_STRIDE + (wid*16 + l_div4)**
- **Классификация**: `l_div4` варьирует **column** (по 1 байту) → 4 lanes по l_mod4 → same row + adjacent bytes (same 4-byte word) → coalesce
- Vugar-класс "**колонный скаттер**" подтверждён по коду

### A-load dS_T (fa_bwd_dk_new.cu:182-185, isolated)
```c
uint32_t A0 = *(uint32_t*)&smdS_T[m_lo * Br + k_i_lo];
// m_lo = wid*16 + l_div4; Br = 64 (constexpr FA_DKN_BR); k_i_lo = kb*32 + l_mod4*4
```
- **byte_addr = (wid*16 + l_div4) × 64 + (kb*32 + l_mod4*4)**
- Stride=Br=64 (жёстко зашит)

## 014-b — Предсказания (зарегистрированы ДО прогона)

Paper-analysis:

| Pattern | Формула (per lane, fixed ks/kb/bt/wid=0) | Bank calc | Paper prediction |
|---------|-------------------------------------------|-----------|:----------------:|
| **P6** STS @68 | `l_mod4*4*68 + l_div4` | 4 rows × 2 word-groups = 8 distinct banks {0,1,4,5,8,9,12,13} | **0.00 events/store** |
| **P7** STS @80 | `l_mod4*4*80 + l_div4` | 4 rows collapse to 4 banks {0,1,16,17} × 2 hits each = 2-way | **1.00 events/store** |
| **P8** A-load @Br=64 | `l_div4*64 + l_mod4*4` | 8 banks {0..3, 16..19}, 4 lanes/bank = 4-way | **3.00 events/load** |
| **P9** B-load @68 + XOR ((l_div4>>1)&3)<<4 | `l_div4*68 + (l_mod4*4)^swz` | sorted starts {0,3,6,12,17,18,23,29}, min diff 1 = 2-way | **1.00 events/load** |

## 014-c — Измерения (N=1M iters × 4 warps = 4M base inst)

| Pattern | conflicts LD | conflicts ST | wavefronts LD | wavefronts ST | inst LD | inst ST | events/inst | verdict |
|---------|:------------:|:------------:|:-------------:|:-------------:|:-------:|:-------:|:-----------:|:-------:|
| **P4** B@68 | 4,000,000 | 0 | 8,000,000 | 64 | 4,000,000 | 64 | **1.00 LD (2-way)** | ✓ |
| **P5** B@80 | 0 | 0 | 4,000,000 | 64 | 4,000,000 | 64 | **0.00 LD** | ✓ perfect |
| **P6** STS @68 | 0 | 0 | 4 | 4,000,064 | 4 | 4,000,064 | **0.00 ST** | ✓ paper hit |
| **P7** STS @80 | 0 | **4,000,000** | 4 | **8,000,064** | 4 | 4,000,064 | **1.00 ST (2-way)** | ✓ paper hit |
| **P8** A @Br=64 | 12,000,000 | 0 | 16,000,000 | 64 | 4,000,000 | 64 | **3.00 LD (4-way)** | ✓ paper 011 подтверждён |
| **P9** B@68+swz | 4,000,000 | 0 | 8,000,000 | 64 | 4,000,000 | 64 | **1.00 LD (2-way)** | ✓ swizzle НЕ помогает |

**Все 4 предсказания попали в точку.** Модель 011 полностью подтверждена калиброванными измерениями.

## 014-d — Vugar decision tree (предрешён)

> **Ветка A**: P6@80 ≤ 0.3 events/store → правка QT_STRIDE 68→80
> **Ветка B**: P6@80 ≥ 1 event/store → правку НЕ делать

**Измерено P7 (P6-analog @80) = 1.00 events/store**.

→ **PATH B ACTIVATED.** Правку QT_STRIDE 68→80 **НЕ ДЕЛАЮ**.

### Обоснование net-эффекта в dk_new
- **LD conflicts saved** (fix S=68→80 для B-load): 537 M events (P4 → P5)
- **ST conflicts added** (fix S=68→80 для Q_T-scatter): 
  - Runtime STS.U8 per launch: 64 STS/qt × 128 qt × 65536 warps = 537 M warp-inst
  - × 1.00 events/inst = **+537 M ST conflicts**
- **Net change ≈ 0** (LD savings compensated by ST added) — net-effect правки STRIDE **нулевой**
- **Wall последствие**: не улучшить, риск неопределённого перехода из LD-bottleneck в ST-bottleneck без bank-conflict gain

## 014-e — P8 (A-load) верифицирован

A-load pattern (`byte_addr = l_div4*64 + l_mod4*4`) при stride Br=64 (жёстко зашит):
- **3.00 events/load** measured = 4-way conflict paper 011 ✓
- Runtime dk_new: 8 A-loads/qt × 3.00 × 128 qt × 65536 warps = **201 M A-class events**
- **Br=64 нельзя изменить** без пересмотра MMA layout (m16n8k32 требует m-fragment 16 rows × 4 bytes = 64-byte tile)

## 014-f — P9 (XOR swizzle) провалился

XOR-swz `((l_div4>>1) & 3) << 4` даёт 2-way (1.00 events/load) — не решает 2-way overlap стрида 17.

**Причина** (paper): starts {0,3,6,12,17,18,23,29}, min diff = 1. Групповая структура stride-17 неисправима простым XOR-классом.

Для perfect 1-way нужна нестандартная mapping l_div4 → bank-start:
- Target: {0, 4, 8, 12, 16, 20, 24, 28}
- Delta from current {0, 17, 2, 19, 4, 21, 6, 23}: {0, 19, 6, 25, 12, 31, 18, 5}
- Не простая XOR/AND формула — нужен lookup-table или composite mapping

Простой XOR-класс из семейства `((r>>k)&m)<<n` не находит perfect 1-way для S=17.

## 014-g — Что дальше (options for Vugar)

### O1: **Оставить status quo** (dk_new @68, wall 9.42 ms)
- Нет прогресса, но нет regressa. Основа R2C стоит.

### O2: **Swizzle-ТЗ** (после разбора)
- Целевая formula: joint conflict-free для **B-load** (l_div4*68 + k_i^swz) И **Q_T-scatter** (l_mod4*4*68 + l_div4^swz2)
- Non-trivial: разные bank patterns на LD и ST путях
- Composite XOR (row-dependent per pattern) или non-XOR permutation

### O3: **Смена MMA layout** (радикально)
- Alternative fragment shapes could avoid the stride-17 sawtooth
- Требует переделки MMA loop, dS_T layout, эпилога dK
- Стоимость: full-rewrite dk_new + bit-exact re-cert 11/11

### O4: **Переход к другому кандидату** (не dk_new)
- merged mio 24.56% — второй по величине bottleneck
- Merged double-buffer qt cp.async (если SMEM позволит)
- Полировки dk_new/dq_new "хвостов" в E2E чейне

## 014-h — SMEM закрытие расхождения

`fa_bwd_dk_new.cu:243` (launcher factual):
```c
const int smem_bytes = Br * hd + hd * FA_DKN_QT_STRIDE + Bc * Br
                    =  64*128 + 128*68 + 64*64
                    =  8192  + 8704  + 4096
                    =  20992 B (dynamic)
```
`cudaFuncGetAttributes.sharedSizeBytes` = **0** (all dynamic, since kernel uses `extern __shared__`).

- Dynamic SMEM: **20992 B** (факт, из launcher)
- Driver SMEM: **1024 B** (NVIDIA runtime overhead per block)
- **Total per-block SMEM: 22016 B** (не 22040 из 011 — округление error, теперь закрыто)
- Slot 4 blocks: (dyn + driver) × blocks ≤ 102400 → **dyn ≤ 24576 B**
- Headroom current: 24576 − 20992 = **3584 B**

## Резюме 014

- ✅ **Формулы из кода**: STS-scatter колонный (l_div4 варьирует col), A-load @Br=64 hardcoded
- ✅ **Все 4 предсказания попали в точку** (P6=0, P7=1, P8=3, P9=1)
- ⛔ **P7 (P6@80) = 1.00 events/store** — **PATH B ACTIVATED**
- ⛔ **Правка QT_STRIDE 68→80 НЕ делается** (net-effect ≈ 0, LD savings compensated ST additions)
- ✅ P8 подтверждает A-loading 4-way conflict (paper 011)
- ⛔ P9 XOR-свизла ((r>>1)&3)<<4 не даёт perfect 1-way для S=17
- ✅ SMEM factual: 20992 dyn / 22016 total / 24576 slot / **headroom 3584 B**

**Правку не делаю. dk_new остаётся @ QT_STRIDE=68 with wall 9.42 ms. Жду Vugar-решение по O1-O4.**

### Мой рекомендация: **O4** (переход к merged mio-полировке)

dk_new bank-conflict полировка требует нетривиальной формулы (P9 не работает) или пересмотра MMA layout. **Merged mio_throttle 24.56%** и **short_scoreboard 10.9%** — более доступные цели с меньшим риском.
