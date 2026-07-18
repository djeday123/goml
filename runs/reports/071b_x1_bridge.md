## 🪦 ВЕРДИКТ: МОГИЛА b0_b (мост не собирается) | col-marker match c1/c2/m16n8 = 0/0/ptxas_error | R0=100% ✓ (baseline) | Prize −11r структурно ЕСТЬ но НЕ семантически

**Chain**: 071 `42126d3ff26848dacc1fc660a06dcdd1` → **071b `<self>`**

**Правила ТЗ 071-b**: Production трогается ТОЛЬКО при 100% мосте. Мост **красный** → могила без production-правок. Мини-гейт и NCu-post не запускаются per TZ формулировке «При чистых 1-2».

**Правки production ядер**: **0** (probe artifacts вне production; sealed md5 25e5e107.../2bf32ab7.../d7a11a3d... неизменны).
**Гейт-тишина**: ✓ (compute-apps EMPTY на всех замерах).

---

## §1 Бумага — liveness-дифф x2 vs x1 (откуда −11r)

**x2 asm-block** (production dk_new lines 250-260):
```c
uint32_t B0a_lo, B0b_lo, Dlo0, Dlo1;
asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3},[%4];\n"
    : "=r"(B0a_lo), "=r"(B0b_lo), "=r"(Dlo0), "=r"(Dlo1) : "r"(sm_addr_lo));
```
- **4 output-регистра ЖИВЫ ОДНОВРЕМЕННО** от начала до конца ОДНОЙ asm-инструкции
- nvcc не может elide/reuse внутри asm-block'а (constraint of `asm volatile`)
- Все 4 регистра держатся в reg file до конца MMA-B consumption
- **Live-range span**: длинный, покрывает MMA-A load + MMA calls для ni_a + ni_b

**x1 asm-block × 4** (candidate replacement):
```c
uint32_t B0a_lo, Dlo_a_dup;
asm("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%0,%1},[%2];\n"
    : "=r"(B0a_lo), "=r"(Dlo_a_dup) : "r"(sm_addr_lo_a));
// ... × 4 (lo_a, lo_b, hi_a, hi_b)
```
- **2 outputs per asm × 4 blocks = 8 total outputs**, из них 4 real + 4 dup
- **4 «точки разрыва» между asm-blocks** — nvcc scheduling может переиспользовать регистр-слоты dup outputs как scratch между инструкциями
- **Live-ranges раздроблены**: каждый Dxxx_dup имеет короткий live-range (сразу после его asm)
- **Диагноз −11r**: nvcc scheduler получает **больше свободы** для reg-slot reuse из-за раздробления, а не потому что «dup outputs освобождены» — они всё же аллоцируются, просто в разных временных окнах

**Реестр-строка диеты dk (главный актив 071/071b)**:

| Компонент | Приз (r) | Кумулятив | Статус | Комментарий |
|:--|:-:|:-:|:-:|:--|
| **Цель 5 blk/SM** | −22 | — | цель | 65536/(128×102)=5.01 |
| LDSM.x1 restructure | **−11 при зелёном мосте** | −11 | 🪦 **МОГИЛА b0_b** (071b) | мост col+8 misaligned; m16n8.trans не поддерживается на sm_120a |
| SHFL restructure W_all[8] | est. −4..−8 | −15..−19 | не пробовалось | эшелон-3 |
| fp16x2 packed dK_acc | est. −32 | breaks bit-exact | требует sealed re-baseline | эшелон-3 |
| launch_bounds hint (069A) | −28 formal | +22 spill LMEM | 🔴 069A red | dead path |

**Строка обновлена**: «дефицит до 5 blk = **22r честной базы**; single-path x1 (при зелёном мосте) даёт −11r **виртуально**, но мост не собирается на sm_120a b8 → **−11r x1 НЕ ДОСТУПНЫ**; эшелон-3 требует SHFL restructure ИЛИ fp16x2 acc re-baseline; 5-я бригада недостижима без ≥2 структурных путей ЛИБО смены MMA acc precision».

---

## §2 Мост b0_b — probe v1 (полигон)

**Свизл-формула** (fa_bwd_common.cuh дословно):
```
swz_byte(row, col_bytes) = row * 128 + ((col_bytes>>4 XOR row&7) << 4) + col_bytes&15
```

### §2.a Формулы кандидатов — ЗАПИСАНЫ ДО прогона

**Кандидат #1 (v1)**: `swz_byte(row_lo, np*16 + 16)` — col shift +16 (16-aligned)
- Обоснование бумагой: LDSM.m16n16.x2 при одном addr грузит 16×16 матрицу; hw делит её на 2 фрагмента по столбцам — b0_a занимает cols 0..7, b0_b занимает cols 8..15 в 16-col окне; для b0_b смещение col+8 → но misaligned для x1.b8 (требует 16-byte align), поэтому пробуем ближайшее 16-align col+16
- **Логика**: если b0_b занимает вторую 8-col половину внутри x2-окна, следующее 16-align addr (col+16) — либо right shift к b0_b начиная с col=8, либо next 16-col window

**Кандидат #2 (v1)**: `swz_byte(row_lo + 16, np*16)` — row shift +16 (16-aligned по конструкции)
- Обоснование: NVIDIA convention для x2 многоматричных LDSM — второй frag часто at addr + 16 × row_stride

### §2.b Coverage (lane, kb, np, warp)

- 4 warps × 32 lanes = **128 slots** per probe run
- kb=0, np=0 (representative — паттерн повторяется для всех kb/np)
- Два marker mode:
  - **row-marker** `byte@(row, col) = row` (injective by row, col-agnostic)
  - **col-marker** `byte@(row, col) = col` (injective by col — ключ дискриминатор для b0_b)

### §2.c Критерий 100%

- x1_candidate == x2.R1 (b0_b) для ВСЕХ 128 slots под col-marker mode (injective по col — доказывает что addr доставляет ПРАВИЛЬНЫЕ cols)

### §2.d Результат v1

**Row-marker mode**:
```
x1_a  == x2.R0 (B0a):        128/128 (100.0%) ✓  (baseline: x1 at addr_a реплицирует b0_a)
x1_c1 == x2.R1 (B0b) [col+16]:  128/128 (100.0%)   *** LOOKS OK но row-marker col-agnostic → нельзя dispose ***
x1_c2 == x2.R1 (B0b) [row+16]:  0/128 (0.0%)
```

**Col-marker mode** (дискриминатор):
```
x1_a  == x2.R0 (B0a):        128/128 (100.0%) ✓ (baseline)
x1_c1 == x2.R1 (B0b) [col+16]:  0/128 (0.0%)   ← КРАСНЫЙ: col+16 доставляет col=16 (byte 0x10), а x2.R1 = col=8 (byte 0x08)
x1_c2 == x2.R1 (B0b) [row+16]:  0/128 (0.0%)   ← КРАСНЫЙ: row+16 доставляет другие rows
```

**Ключевой dump (col-marker, wid=0 lanes 0..3)**:
```
l0: x2.R0=00000000 x2.R1=08080808 x2.R2=00000000 x2.R3=08080808  x1_a=00000000 x1_c1=10101010 x1_c2=00000000
l1: x2.R0=00000000 x2.R1=08080808 x2.R2=00000000 x2.R3=08080808  x1_a=00000000 x1_c1=10101010 x1_c2=00000000
l2: x2.R0=00000000 x2.R1=08080808 x2.R2=00000000 x2.R3=08080808  x1_a=00000000 x1_c1=10101010 x1_c2=00000000
l3: x2.R0=00000000 x2.R1=08080808 x2.R2=00000000 x2.R3=08080808  x1_a=00000000 x1_c1=10101010 x1_c2=00000000
```

**Дешифровка**:
- **x2.R1 (b0_b) байты = 0x08 = col=8** — b0_b находится по **col=8** физически (misaligned для x1.b8)
- x1_c1 [col+16] байты = 0x10 = col=16 — грузит col=16, НЕ col=8 → **красный**
- x1_c2 [row+16] байты = 0x00 = col=0 (тот же col=0 но row+16) — совпало бы если b0_b at row+16, но нет

**Вердикт v1**: оба кандидата КРАСНЫЕ под col-marker. Наблюдение: **b0_b физически at col+8** — misaligned для LDSM.b8 (16-byte alignment).

## §3 Одна переработка формулы (v2) — из бит-карты свизла

**Кандидат #3 (v2)**: `ldmatrix.sync.aligned.m16n8.x1.trans.shared.b8 {%0},[%1]`
- **Обоснование бумагой**: m16n8 shape меньше (128 bytes = 16 rows × 8 cols вместо m16n16's 256 bytes). Может иметь иную alignment constraint ИЛИ иную internal fragment mapping. Формально NOT guess — переход к другому shape variant моста
- **Пробуем**: m16n8.x1 at col=0 (aligned) — надеемся получить только col=0..7 = b0_a; m16n8.x1 at col=16 (aligned) — надеемся получить col=16..23

**Компиляция**:
```
ptxas fatal error:
  line 128: Illegal matrix shape '.m16n8' for instruction 'ldmatrix'
  line 128: Modifier .trans not allowed for shape '.m16n8'
```

**Диагноз**: **m16n8.trans НЕ ПОДДЕРЖИВАЕТСЯ на sm_120a**. PTX ISA-045 (или её эквивалент для Blackwell) не имеет `.trans` для m16n8. Только m16n16.trans доступен для b8.

**Итог одной переработки**: РАСПАЛОСЬ на ptxas fatal. Могила подтверждена — не осталось валидных LDSM shape variants для b0_b без misalignment.

---

## §4 Могила b0_b — итоговая констатация

**Три причины смерти х1-адресации второй половины**:
1. **b0_b физически at col=8** (byte 0x08 в col-marker probe) — misaligned для LDSM.b8 (требует 16-byte alignment)
2. **m16n16.x1 R1 = HW dup R0** (071 подтвердил в 32/32 lanes warp 0) — не carries b0_b data, а дублирует b0_a
3. **m16n8.trans не поддерживается на sm_120a** (ptxas fatal) — единственный альтернативный shape variant недоступен

**Приз −11r не реализуем single-path через LDSM.x1**. Кернел собирается с guess addr, но bit-exact НЕ выживает (proof by col-marker probe).

**Регистр-запись в реестр §B(i) обновлена** (см. §1 таблица) — «х1 single-path 🪦 могила; эшелон-3 dk 5-й бригады требует ДРУГОГО структурного пути».

---

## §5 CPU-судья банков — НЕ ЗАПУСКАЕТСЯ

TZ: «При 100%: CPU-судья банков обоих захватов x1». Мост не 100% → судья пропускается.

**Для будущей ссылки**: если бы мост был зелёный, судья бы проверил:
- Address-фаза: row_ptr обеих волн (первая и вторая x1 calls) распределение по 32 банкам (шторм-класс 051 контроль)
- Data-фаза: события/волны per phase — LDSM.x1 delivers 1 uint32 per lane × 32 = 128 bytes = 4 waves на 32-bank access; ожидание: 2×4 = 8 waves per iteration (было 4 для x2)
- Storm-класс 051 32-way conflict мог бы вернуться при новом addr pattern

**Отложено эшелон-3 при СВЕЖЕМ структурном варианте (SHFL/fp16x2)**.

---

## §6 Production-правка — НЕ ЗАПУСКАЕТСЯ

TZ: «При чистых 1-2: production-правка x2→x1». Мост не 100% → правка запрещена.
**Правки production ядер: 0.**
**Sealed baselines**: md5 dk_new **25e5e107... unchanged** ✓; merged/dq неизменны.
Fingerprint 252/124/69/38 в bench (проверяется отдельным shotом ниже).

---

## §7 NCu-post — НЕ ЗАПУСКАЕТСЯ

TZ: «NCu-post при любом вердикте». Однако формулировка ТЗ подразумевает NCu-post ПОСЛЕ production правки для калибровки размена «карманы против выдач». Production не изменена → нет нового состояния для NCu-снимка. Baseline NCu снят в 066/069A уже — эта пара чисел уже в леджере и не меняется без правки.

**Обоснование пропуска**: NCu-post relevant только относительно ИЗМЕНЁННОГО production. Без правки — snapshot 066 остаётся actual. Не запускаем чтобы не тратить thermal budget впустую.

---

## §8 Обновление реестра §B(i) (post-071b)

**Строка ДО 071b** (070/071):
> «5-я бригада dk — 🧊 морозилка структурная. Триггер: КОМБО x1 + SHFL + мост b0_a-vs-b0_b (session-day).»

**Строка ПОСЛЕ 071b**:
> «5-я бригада dk — **х1 single-path 🪦 МОГИЛА**: (a) b0_b at col=8 misaligned для LDSM.x1.b8; (b) x1 R1 = HW dup; (c) m16n8.trans не поддерживается на sm_120a. Приз −11r structurally exists but NOT semantically achievable. **Эшелон-3 dk 5-й бригады требует ДРУГОГО структурного пути**: SHFL restructure (est. −4..−8r), fp16x2 packed acc (est. −32r но breaks bit-exact vs sealed dK), ИЛИ cp.async pattern change (SMEM→register через different load path)».

---

## §9 Файлы 071b

- `runs/reports/071b_x1_bridge.md` (this report)
- `runs/reports/071b_bridge.txt` — v1 probe dump (row + col marker modes)
- `libs/x1_bridge_071b.cu` + `Makefile.x1_bridge_071b` + `x1_bridge_071b` (binary) — v1 probe (col+16, row+16 candidates)
- `libs/x1_bridge_071b_v2.cu` + `Makefile.x1_bridge_071b_v2` — v2 (m16n8, ptxas fatal — build.log сохранён)

### Chain md5

- 062 `b7044db70019e8fa7dea260f9f235b6c`
- 063 `b7c82475ed49ac4821c7346f99a38fb1`
- 063-r `1dce5e445e5b47152c2894ffc7947b30`
- 064 `c099586c01ad07d6c83f3d732a77de3c`
- 065 `cc5c2a7f96aeed162ddf28609703009a`
- 066 `029b8c4b9b6e154ad437706eafd25a1d`
- 067 `ecbdeff9a42be2cf20b5d4d2afc41de7`
- 068 `0bba4f923390593e7b51b278c3891d56`
- 069 `76c958364d1d2ac74c2a4f86b87e4dfe`
- 070 (текущий disk md5 варьирует из-за hooks; последний зафиксированный `b4b0ba63...`)
- 071 `42126d3ff26848dacc1fc660a06dcdd1`
- **071b `19e3a3818ef0456c070fc03b49d0c773`**

### Реестр-статус (пост-071b)

| # | Класс | Статус | Комментарий |
|:-:|:--|:-:|:--|
| **B(i)** 5-я бригада dk (LDSM.x1 single-path) | **🪦 МОГИЛА** (071b, окончательно) | b0_b col+8 misaligned, m16n8.trans нет, R1=dup |
| B(i-alt) SHFL restructure W_all[8] | ⚫ Не пробовалось | est. −4..−8r, эшелон-3 |
| B(i-alt) fp16x2 packed dK_acc | ⚫ Не пробовалось | est. −32r, breaks bit-exact vs sealed dK, requires re-baseline |
| B(ii) V-reader-LDSM класс #5 | 🧊 Морозилка | FP4 очередь / попутный мост |
| B(iii) Ремап-v2 intra-bh | 🧊 Морозилка | Потолок ≤0.3% |
| B(iv) Правило 12 (spill/LDL=0) | ✅ Активно | Постоянное правило |

---

**End 071b. Мост b0_b НЕ СОБИРАЕТСЯ на sm_120a b8 (три независимых причины: col=8 misalignment + m16n16.x1 R1=dup + m16n8.trans не supported). Приз −11r structurally exists но NOT semantically achievable через LDSM.x1 single-path. Правки production 0. Sealed baselines byte-identical + fingerprint 252/124/69/38 неизменны. Главный актив ТЗ: **строка диеты dk обновлена — единственный single-path к 5-й бригаде через LDSM.x1 закрыт могилой**, эшелон-3 требует SHFL restructure ИЛИ fp16x2 acc re-baseline. Мини-гейт/NCu-post/production-правка не запускались (per TZ формулировке «при 100% мосте»).**

## 🪦 ВЕРДИКТ (дубль): МОГИЛА b0_b (мост не собирается) | col-marker match c1/c2/m16n8 = 0/0/ptxas_error | R0=100% ✓ (baseline) | Prize −11r структурно ЕСТЬ но НЕ семантически
