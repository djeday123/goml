# 033 stop-report — unit-test транспонирования 1024/4096, дальнейшая правка остановлена

**Chain**:
- 032_b_holes.md md5: `8aad4e120306d4a7b2a14f32405f492f`

**Artifact header** (production НЕ тронут):
```
-rw-r--r-- 14667  Jul  7 10:03  libs/fa_bwd_dk_new.cu    (023 sealed π_V, md5 9b12a7d1…)
-rw-r--r-- 13352  Jul  5 17:51  libs/fa_bwd_dq_new.cu    (sealed pre-pack, md5 683396f8…)
-rw-r--r--         Jul  7        runs/probes/transpose_ds_unit_test.cu   (unit-test, RED)
-rw-r--r--         Jul  7        runs/probes/Makefile.transpose_ds
```

**Правки dk_new / merged НЕ начаты.** Стоп на юнит-тесте по правилу 033 (пункт 1: юнит → правка).

---

## 1. Unit-test результат

**Задача**: micro-kernel 128 threads, 4096-byte tile (64×64), pack-analog Phase A/B/C/D в SMEM aliased.

**Реализация**: `runs/probes/transpose_ds_unit_test.cu`, ptxas: **28r / 0 spill / 4096 B smem / 1 barrier**.

**Результат**:
```
MISM j=0 i=16 expect=0x3d got=0x7d
MISM j=0 i=17 expect=0xc0 got=0x00
MISM j=0 i=18 expect=0x43 got=0x83
MISM j=0 i=19 expect=0xc6 got=0x06
MISM j=0 i=20 expect=0x49 got=0x89
dS_T bytes: match=1024/4096, mismatch=3072
```

**1024/4096 = 25% match, 75% mismatch.**

## 2. Диагноз

**Analysis**:
- Expected at (j=0, i=16): source_nat(16, 0) = (16*131 + 0*7 + 13) & 0xFF = 0x3d
- Got at (j=0, i=16): 0x7d = source_nat(0, 16) = (0*131 + 16*7 + 13) & 0xFF

**Байты НЕ транспонируются** — источник копируется в ту же позицию.

**Root cause**: pack-analog из dq K_T pack (128×64 shape, unit-tested 8192/8192 in 025-b) неверно адаптирован под dS transpose (64×64 shape). Ключевые различия:
- dq K_T: `ks = wid` (4 warps × 32 = 128 k-values, spans Hd=128)
- dS transpose: 64 j-values only, ks=wid с ks*32 max = 96 overflows valid range [0..63]
- Slot decomposition (2 slots vs 4), feeder mapping требует переработки под 64×64 квадратную форму
- Phase D output row mapping (`base_row = wid*16 + 4c + p`) не всегда корректно попадает в T-строку

**Match rate 25%** предполагает что 1 из 4 slot-групп работает, остальные пишут на wrong positions.

---

## 3. Стоп-протокол активирован

По TZ 033 (Vugar): "**Порядок: юнит-тест → правка dk_new** ..." — юнит-тест обязателен перед code changes.

**Красный юнит-тест → стоп-доклад, НЕ силовая правка**.

**НЕ выполнены**:
- Правка dk_new (не начата)
- Правка merged (не начата)
- ABI-дельта (не начата)
- Гейты (не начаты)

**Выполнены**:
- Юнит-тест транспонирования: **RED (1024/4096)**
- ptxas юнит-теста: 28r/0spill/1barrier/4096 B (для reference, но правильность важнее)

---

## 4. Лестница вариантов для решения Vugar (по 033-вставке)

По TZ 033: "при ptxas dk_new >128r — автостоп остаётся, но доклад обязан содержать лестницу вариантов". Формально это про ptxas, но здесь применим аналог — красный юнит-тест требует лестницы вариантов дальше:

### (1) Переработать pack-analog под 64×64
- **Проблема**: 64×64 квадратная форма отличается от 128×64 dq K_T (прямоугольная). Feeder ks=wid не работает (ks*32 max = 96 > 63)
- **Требуется**: пересчёт slot decomposition, feeder mapping (возможно split wid → pair-groups), Phase D coordinate math
- **Оценка сложности**: 2-4 часа design + debug + unit-test
- **Регистровая цена**: ~28r в юнит-тесте (без production overhead), в dk_new production **+8-14r над 124r baseline → 132-138r**
- **Риск порога 128r**: **высокий** (аналог 032-b прогноза, но с уже RED-статусом юнит-теста)

### (2) Straightforward LDS.32 → registers → STS.U8 (no SHFL)
- **Схема**: каждый lane читает 8 LDS.32 (32 bytes), пишет 32 STS.U8 в T positions
- **Bit-correctness**: гарантирован (no cross-lane exchange, own bytes только)
- **Bit-invariant**: тавтологически преserved
- **MIO-ops**: 8 LDS.32 + 32 STS.U8 = **40 MIO ops/qt/lane** — **превышает Vugar-порог 30** ✗
- Register cost: минимальный (+2-4r)
- **Не проходит** Vugar auto-authorization critеrию (MIO ≤ 30)

### (3) Полтайла × 2 такта (Vugar-предложение)
- **Схема**: транспонировать 32×64 в 1 такт, затем 32×64 в 2-й такт
- Между тактами: sync + partial STS
- Регистры: половина от полного (~4 uint32 per lane per tact)
- Барьеры: +2 (per tact)
- **Требует**: новый design pack-analog для 32×64 rectangular (аналог dq K_T shape)
- **Оценка**: 3-5 часов
- **Регистровая цена**: +4-8r над baseline (меньше чем полный tile)
- **MIO-ops**: ~15 per tact × 2 = 30 total (граничит с порогом)

### (4) Справка размена π_V в dk_new
- **Текущий 023 sealed dk_new**: 124r с π_V, 4 blocks/SM
- **Возврат π_V (снять 023)**: -17r → **107r baseline**, освобождает окно для transpose staging (~17r headroom)
- **Цена размена**: +2.4% dk_new wall (по 023 ABBA-серии) = **+0.21 ms dk_new wall**
- **Compensating gain**: merged transpose winning ~1.9-3.9 ms (по 032-b)
- **Net E2E**: (-1.9..-3.9) + 0.21 = **-1.7..-3.7 ms → -3.5..-7.6% E2E**
- **Bit-exact**: сохранён (π_V — layout-only, снятие не меняет байты)
- **Complexity**: очень низкая (revert 023 patch first, then re-apply new transpose within larger register window)

**Рекомендация**: **(4) размен π_V** — самый низкорисковый путь.

Обоснование:
- Открывает 17 регистров headroom → transpose легко влезет с любой из вариантов (1)-(3)
- Merged winning остаётся тот же (~1.9-3.9 ms)
- π_V cost ~0.21 ms << merged gain
- Bit-exact сохраняется без изменений
- Юнит-тест проще спроектировать без tight register constraints

---

## 5. Файлы

- Стоп-доклад: этот отчёт
- Unit-test (RED): `runs/probes/transpose_ds_unit_test.cu`
- Makefile: `runs/probes/Makefile.transpose_ds`

Chain md5: 032-b `8aad4e12…` → **033 stop `<computed>`**

---

**End 033 stop-report.**

**Правки production не начаты.** Ожидаю решение Vugar по лестнице:
1. Переработать pack-analog 64×64 (2-4 часа design, риск >128r)
2. LDS→STS.U8 (не проходит MIO ≤ 30)
3. Полтайла × 2 такта (3-5 часов, MIO ≈ 30 граница)
4. **Размен π_V** (низкий риск, -1.7..-3.7 ms E2E) ← **рекомендация**
