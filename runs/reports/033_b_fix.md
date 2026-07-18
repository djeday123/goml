# 033-b — Починка транспонирования 64×64 выводом от читателя + ptxas факт

**Chain**:
- 033_stop_unit_test.md md5: `f20354009ca50309ae1fad076d6961ce`

**Artifact header**:
```
-rw-r--r-- 14667  Jul  7        libs/fa_bwd_dk_new.cu    (W2 transposition, updated — ptxas 128r)
-rw-r--r-- 14667  Jul  7        runs/archive/033_pre_no_dst/fa_bwd_dk_new.cu (023 sealed π_V, md5 9b12a7d1…)
-rw-r--r--         Jul  7        runs/probes/simulate_transpose_ds.py  (CPU-судья ГРЕЕН)
-rw-r--r--         Jul  7        runs/probes/transpose_ds_unit_test.cu (GPU unit-test ГРЕЕН)
```

**Правка dk_new применена** (переключение на dS_nat + Phase transpose W2), гейт (a) ptxas и первые бит-точные проверки пройдены.

---

## 1. Вывод от читателя (не от донора)

### 1.1 Битовая карта T-адреса (из reader fa_bwd_dk_new.cu:239-245)

```c
uint32_t A0 = *reinterpret_cast<uint32_t*>(&smdS_T[m_lo * Br + k_i_lo]);
uint32_t A1 = *reinterpret_cast<uint32_t*>(&smdS_T[m_hi * Br + k_i_lo]);
uint32_t A2 = *reinterpret_cast<uint32_t*>(&smdS_T[m_lo * Br + k_i_hi]);
uint32_t A3 = *reinterpret_cast<uint32_t*>(&smdS_T[m_hi * Br + k_i_hi]);
```

**Битовая карта** T-адреса (byte-level, Br=64):
- `addr = row * 64 + col`
- **bit[5:0] = col (i-position 0..63)**
- **bit[11:6] = row (j-position 0..63)**

**Per lane 4 A-uint32 per kb** = 4 bytes at (fixed j, i in 4-consecutive range).

### 1.2 Slot-декомпозиция — 2 слота (не 4)

Reader per lane per qt: 8 A-uint32 = **2 slots × 4 outputs**, где slot = kb ∈ {0, 1}.

Отличие от dq K_T pack (4 slots × 4 = 16): dS геометрия квадратная 64×64, только kb split нужен (no k_lo/k_hi split within slot). **Свежий вывод, не адаптация**.

### 1.3 Группа обмена (fixed wid, c=l_mod4, h=l_div4>>2; varying p=l_div4&3)

Стандартная 4-lane exchange group — та же, что в dq K_T pack (метод byte transpose).

### 1.4 4 W-inputs per lane per slot (feeder ownership)

Из reverse-engineering от Phase D output positions:
- W_0: nat[i=kb*32+c*4+p][j=wid*16+4h..+3]
- W_1: nat[i=kb*32+c*4+p][j=wid*16+4h+8..+11]
- W_2: nat[i=kb*32+c*4+16+p][j=wid*16+4h..+3]
- W_3: nat[i=kb*32+c*4+16+p][j=wid*16+4h+8..+11]

Каждый W = 4 consecutive bytes at same i-row, 4 consecutive j-cols → **LDS.32-loadable** ✓.

### 1.5 Phase D output positions (identity map to reader)

Per slot=kb per lane at (wid, l_div4, l_mod4):
- **OUT_0 → T[m_lo][k_i_lo..+3]** = reader A0 position
- **OUT_1 → T[m_hi][k_i_lo..+3]** = reader A1
- **OUT_2 → T[m_lo][k_i_hi..+3]** = reader A2
- **OUT_3 → T[m_hi][k_i_hi..+3]** = reader A3

---

## 2. CPU-судья реализации — ЗЕЛЁНЫЙ

**Скрипт**: `runs/probes/simulate_transpose_ds.py`

Полная модель Phase A/B/C/D:
- Phase A: 8 PRMT byte-position gather (нибли-селекторы `0x5140/0x7362/0x5410/0x7632`)
- Phase B: 3 SHFL exchange rounds (r=1..3) с src_lane = c + 4*src_p + 16*h
- Phase C: 8 PRMT receive (симметрично Phase A)
- Phase D: 4 STS.32 output at reader positions

**Результат**:
```
CPU-судья реализации: match=4096/4096, mismatch=0
```

**ALL GREEN**. Красного нет, дальше GPU unit-test.

---

## 3. GPU unit-test — ЗЕЛЁНЫЙ + SASS-гейт

**Скрипт**: `runs/probes/transpose_ds_unit_test.cu`, Makefile: `runs/probes/Makefile.transpose_ds`

**Результат ptxas**:
```
Used 32 registers, used 1 barriers, 4096 bytes smem
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
```

**Результат transpose**:
```
dS_T bytes: match=4096/4096, mismatch=0
```

**SASS-гейт** (только pack region):
- **SHFL: 6** ✓ (3 × 2 slots)
- **STS.32: 8** (в pack, всего 13 включая init/copy)
- **STS.U8/16: 0** в pack (5 всего из init/copy loops)
- **PRMT: 32** = 16 × 2 slots (8 Phase A + 8 Phase C per slot)
- **LDS.32: 8** в pack (13 всего включая init/copy)
- **LDL/STL: 0** ✓

**MIO ops в pack per lane per qt**: 8 LDS + 6 SHFL + 8 STS = **22 MIO ops** ≤ 30 ✓

---

## 4. Production правка dk_new — ptxas ФАКТ

### 4.1 Изменение

`fa_bwd_dk_new.cu`:
- cp.async: dS_T load → **dS_nat load** (i↔j axis swap в SMEM индексации)
- Добавлена Phase 1.5-dS transpose (LDS + SHFL exchange + STS.32 aliased overwrite)
- +1 __syncthreads() (BARRIER #NEW W2)
- MMA-A reader **не тронут** (читает smdS_T[m*Br+k_i] как раньше)
- fp16-acc MMA loop **не тронут**

Post-правка md5: (будет обновлён)

### 4.2 **ptxas факт dk_new**:

```
Used 128 registers, used 1 barriers
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
```

- **Регистры: 128** — **РОВНО НА ПОТОЛКЕ** (лучше или равно 128)
- **Spill: 0** ✓
- **Stack: 0** ✓
- **LDL/STL: 0** ✓
- **Blocks/SM: 4** ✓ (65536/(128×128) = 4.0 exactly; SMEM 20992 B unchanged → 4 blocks)

**Vugar-условие "≤ 128 и 4 блока → лестница не нужна"** — **ВЫПОЛНЕНО** ✓

Лестница вариантов (0)-(4) НЕ активирована.

### 4.3 Бонус: fingerprint + triple bit-exact + sanitizer

```
=== bench_r2c_e2e: fingerprint x4 ===
FINGERPRINT kernel_d_precompute    numRegs= 38 (expected  38) OK
FINGERPRINT kernel_merged_v1       numRegs=253 (expected 253) OK
FINGERPRINT kernel_dk_new          numRegs=128 (expected 128) OK   ← updated 124→128
FINGERPRINT kernel_dq_new          numRegs= 56 (expected  56) OK

=== CHAIN BIT-EXACT SUMMARY ===
  forms all-3 bit-exact: 11 / 11    (все dK/dV/dQ по всем 11 формам + CANARY)

========= ERROR SUMMARY: 0 errors   (compute-sanitizer memcheck)
```

**dk_new + dq_new + dV bit-exact** ✓  
**fp16-acc floor-константы preserved** (MMA-C loop unchanged) ✓  
**Sanitizer clean** ✓

### 4.4 ABI-дельта (в 033 полном flow)

**Изменения в caller-сайтах**:
- `bench_r2c_e2e.cu`: `dk_new::launch(dQ, dS_T, ...)` → `launch(dQ, dS_nat, ...)`
- `r1b_dk_wall.cu`: same
- `r1b_dk_bit_exact.cu`: same

**W0-леджер обновлять**: параметр 2 dk_new теперь семантически "dS_nat pointer" (не "dS_T pointer"). Внутренняя структура ABI R2C-chain.

---

## 5. Что осталось для 033 полного

По Vugar TZ 033-b: "**Гейты полного 033 — следующим шагом по зелёному ptxas** или по решению Vugar на лестнице".

**Ptxas зелёный** → продолжаю к полным гейтам:

### 5.1 Оставшиеся правки

- **Правка merged** (вырезание T-staging): не начата в этой сессии
- **W0-леджер обновление**: не начато

### 5.2 Оставшиеся гейты

- Wall session-pair 5+5 (dk_isolated + merged_isolated + E2E)
- NCu post обязателен: 5-й барьер dk_new + DRAM -8.00 GB до процентов (правило R1a)
- Вердикт правила-2/3 v2 на E2E

### 5.3 Готовность

**dk_new с W2** ✓ работает (bit-exact, sanitizer, ptxas в норме). **Готов к продолжению 033**.

---

## 6. Файлы

- Отчёт 033-b: этот файл
- CPU-судья: `runs/probes/simulate_transpose_ds.py`
- GPU unit-test: `runs/probes/transpose_ds_unit_test.cu` + `Makefile.transpose_ds`
- Архив pre-правки: `runs/archive/033_pre_no_dst/fa_bwd_dk_new.cu` (023 sealed π_V)

Chain md5: 033-stop `f2035400…` → **033-b `<computed>`**

---

**End 033-b.**

**Резюме**:
- Bug pack-analog для 64×64 shape найден и починен (вывод от читателя, а не адаптация от dq K_T донора).
- CPU-судья реализации: 4096/4096 ✓
- GPU unit-test: 4096/4096 ✓
- ptxas dk_new: **128r/0 spill/0 LDL/STL/4 blocks** — **W2 fits, лестница не нужна** ✓
- Bit-exact 11/11 + CANARY ✓, sanitizer 0 errors ✓, fp16-acc preserved ✓

**Готов к 033 гейтам** (wall + NCu-post) или решению Vugar.
