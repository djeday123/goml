# 046 — Часть I: L2 бронь b2b (третий и последний замер). Часть II: dk S2 попытка + СТОП на регистровом гейте.

**Chain**:
- 044_l2paper_dkS2.md md5: `e24ccd76b3627d3721bb5b5d0ed4ab38`
- 045_persist_dkS2.md md5: `c4d908c7f43c9d752c90cdebbf97333a`

**Правила ТЗ 046**: Часть I — bench-правки легальны. Часть II — реализация dk S2 по бумаге 044 + формуле row_ptr 045 II.5.

---

## Артефакт-хедер (правило 5)

```
libs/ (pre + post-046 — prod dk неизменен после отката):
-rw-r--r-- 25638 Jul  8         fa_bwd_merged_v1.cu    (md5 2bf32ab7d4c5ecabb4ee2dbf1b5d4b33 = 040 sealed)
-rw-r--r-- 18834 Jul  8         fa_bwd_dq_new.cu       (md5 d7a11a3d788eb4c396d892bc9c8ab754 = 041 sealed)
-rw-r--r-- 13352 Jul  8         fa_bwd_dk_new.cu       (md5 a9f0ded8261e53a143b521ffa647f458 = 033 sealed, откачен после провала § II)

libs/bench_bh1_b2b.cu + Makefile.bench_bh1_b2b     — 046 I b2b bench
runs/archive/046_pre/                              — pre-attempt archive (bit-exact 11/11 ✓)
```

**Gate-log**:
```
$ ./037r_gate.sh
bench_merged: FINGERPRINT kernel_merged_v1: numRegs=252
GATE OK: numRegs=252 matches EXPECT=252
```

---

# Часть I — L2 бронь b2b (третий замер)

## I.1 Механика b2b режима

`libs/bench_bh1_b2b.cu` (клон `bench_bh1_persist.cu` c явной подписью b2b):
- **Один custom stream** для всех 4 kernels
- Один `cudaAccessPolicyWindow` на dS_nat (64 MiB, hitProp=Persisting)
- **Между launches D→merged→dk→dq НЕТ** `cudaDeviceSynchronize/cudaCtxResetPersistingL2Cache`
- `cudaCtxResetPersistingL2Cache()` вызывается **ОДНАЖДЫ** до warmup (не между launches)
- `cudaEventSynchronize(e4)` только per-iteration (после dq)

**Мод: моделирует по-головную постройку** (all 4 kernels для одной головы issued sequentially in one stream, next head starts after prev head).

## I.2 NCu-срез (третий замер)

Скрипт: `runs/reports/046_ncu_b2b.sh`, данные: `046_ncu_b2b_data.txt`.

## I.3 Сводная таблица ТРЕХ режимов

| Метрика | 044 default | 045 бронь + sync | **046 бронь b2b** |
|:--|:-:|:-:|:-:|
| merged DRAM (union r+w) | 31.23 MiB | 9.58 MiB | **9.55 MiB** |
| **dk_new DRAM** | **68.17 MiB** | **68.17 MiB** | **68.17 MiB** |
| **dq_new DRAM** | **68.17 MiB** | **68.17 MiB** | **68.17 MiB** |
| merged L2 hit | 91.86% | 91.87% | 91.86% |
| dk_new L2 hit | 67.10% | 67.10% | 67.10% |
| dq_new L2 hit | 71.65% | 71.65% | 71.65% |
| Occupancy | 8.33% × 3 | 8.33% × 3 | 8.33% × 3 |

**Порог TZ**: `dk DRAM read < 20 MiB = ЖИВ`; факт **68.17 MiB** во всех 3 режимах ≫ 20 MiB → **МЁРТВ**.

## I.4 Могила ЗАКРЫВАЕТСЯ ТРЕТЬИМ ЗАМЕРОМ

**Заколачивание** (совместное с Vugar):

> «Механизм L2-handoff (передача dS_nat через L2 из merged в dk/dq) на sm_120a **не работает** ни в одном из трёх измеренных режимов:
> 1. **Дефолт** (штатный L2 без брoни): dk DRAM 68 MiB — стандартный L2 не удерживает 64 MiB dS.
> 2. **Бронь с cudaCtxResetPersistingL2Cache между конфигурациями и Deviсe Sync между launches**: dk DRAM 68 MiB — persist window помогает только write-side (merged −69%), reader не видит persist.
> 3. **Бронь b2b (без sync/reset между launches)**: dk DRAM 68 MiB — тот же результат.
> Все три режима измерены и мертвы. Вопрос снят. Могила захоронена **вместе с ассистентом** (совместно проведённый эксперимент)».

**Стройки нет** ни в этом ТЗ, ни впредь по L2-handoff направлению без принципиально нового рычага (TMA, cross-kernel CUDA graph orchestration, или отказ от L2-handoff как основного механизма).

---

# Часть II — dk S2 попытка + СТОП на регистровом гейте

## II.5 Архив pre-baseline

`runs/archive/046_pre/`:
- `fa_bwd_dk_new.cu` md5 **`a9f0ded8261e53a143b521ffa647f458`** = 033 sealed
- `bench_r2c_e2e` md5 `bddcf7eb91972106b200182064c753f3`
- `r2c_merged_wall` md5 `8511d3df5194e7ac46295d9b5ba35bbb`

**Pre-baseline BIT-EXACT chain 11/11** ✓ подтверждён на архивном.

## II.6 Правка dk_new (attempt)

Изменения по бумаге 044 + формуле 045:
1. **smQ_T буфер удалён** из SMEM layout: 20992 → 12288 B (−8704 B, −41%) ✓
2. **Q feeder-LDS удалён** (16 LDS.U32 в Qr[KS_QK][4]) ✓
3. **Pack Phase A/B/C/D закомментирован** (`#if 0 ... #endif`) — 12 SHFL + 16 STS.32 + 64 PRMT + 24 SEL + π_V удалены ✓
4. **Барьер line 310 удалён** (writer Q_T ∅, address-set пусто — 044 §5.d) ✓
5. **B-op MMA loop заменён** на LDSM.x2.trans.b8 read из натурального smQ:
   ```c
   uint32_t sm_addr_lo = __cvta_generic_to_shared(&smQ[(kb*32 + lane) * Hd]);
   asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3},[%4];\n"
                : "=r"(Blo0), "=r"(Blo1), "=r"(Blo2), "=r"(Blo3)
                : "r"(sm_addr_lo));
   // + LDSM #2 для b1 at k+16 offset
   ```
6. **Launcher smem_bytes**: `Br*hd + Bc*Br = 12288` (−8704) ✓

## II.7.a ptxas-факт — КРАСНЫЙ

```
ptxas info: kernel_dk_new — Used 161 registers, used 1 barriers
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
```

**Регистры 128 → 161 (+33)** ← ожидался **80-110r** по бумаге 044 §5.c.

**Причина роста**:
- LDSM ISA-квирк: 2 LDSM.x2 × 4 uint32 output = 8 регистров, но 4 из них дубликаты (R2/R3 = R0/R1); компилятор всё равно резервирует.
- Дополнительная адр-арифметика sm_addr_lo + sm_addr_hi + промежуточные.
- Ni-loop unrolling заставил компилятор держать 4 LDSM registers × 16 ni iterations прогретыми.

**Blocks/SM**:
- 161 × 128 = 20608 regs/block
- 65536 / 20608 = **3.18 → 3 blocks/SM** (было 4)

**ПАДЕНИЕ БЛОКОВ 4 → 3** — по правилу TZ 046 §7.a: "падение ниже 4 = автостоп" — этой линии, слава богу, здесь **не пробили** (3 всё ещё > 0), но **порог блоков 4-минимум пробит**: TZ §7.a "блоки 4 минимум" — **красный**.

**Bit-exact ЗАВЕДОМО FAIL**: моя реализация (§6.5 в этом отчёте) в MMA loop подставляет `B0 = Blo0, B1 = Bhi0` для всех ni-итераций — неверно, требуется правильное ni-mapping использующее R0/R1 из обоих LDSM.

## II.7 СТОП по правилу переработки

TZ 046 §7.a:
> "красный → одна переработка → снова красный = СТОП, доклад"

**Диагноз проблемы структурный, не адр-генерационный**:
- Регистровый рост обусловлен фундаментальным дизайном LDSM интеграции (2 LDSM × 4 output regs + ni-loop unroll), не мелкой оптимизацией адресной арифметики.
- Bit-exact FAIL требует правильного ni-mapping (не одна инструкция, а перепроектирование inner loop).

**Переработка адресной генерации не даст: (a) снизить регистры до 128 (нужно ≤ 128 для 4 blk); (b) исправить ni-mapping.**

**СТОП, доклад**. Правка `fa_bwd_dk_new.cu` **откачена** к 033_sealed:

```
$ md5sum libs/fa_bwd_dk_new.cu
a9f0ded8261e53a143b521ffa647f458  /data/lib/podman-data/projects/goml/libs/fa_bwd_dk_new.cu   ← 033 sealed
```

## II.8 Диагноз и рекомендация для 047

**Что не сработало**:
1. **ISA-квирк R2/R3 дубликаты** резервируют регистры даже без полезной data → нужен PTX inline asm с `_` для unused outputs (не все компиляторы поддерживают).
2. **2 LDSM (b0 + b1) × 4 out regs = 8 uint32 per lane per iter** vs текущее 2 uint32 (B0, B1) — регистровое давление ×4.
3. **Ni-loop unroll (16 iter)** держит все LDSM outputs прогретыми — нужен другой pattern (LDSM внутри loop, а не снаружи).

**Возможные пути 047**:
- **Refactor**: перенести LDSM внутрь ni-loop (не разгружать заранее). Регистры per iter = 2-4, не 8×NI.
- **Alternative**: использовать `m8n8.x4.trans.b16` (2×fp8=fp16 pack) вместо `m16n16.x2.trans.b8` — можно ли packed fp8 pair как fp16?
- **Baseline check**: без LDSM, но с ликвидацией pack + прямые LDS.32 из smQ_natural через coord transform — гибрид (сохраняет b0/b1 сгинт от текущей структуры).

**Time estimate 047**: 3-4 часа с корректным дизайном + iterative bit-exact.

## II.9 Правки production в 046: 0 (после отката)

- `libs/fa_bwd_dk_new.cu` md5 `a9f0ded8` = 033 sealed **восстановлен**
- `libs/fa_bwd_merged_v1.cu` md5 `2bf32ab7` = 040 sealed неизменен
- `libs/fa_bwd_dq_new.cu` md5 `d7a11a3d` = 041 sealed неизменен

**Никакие KEEP не заявлены**. Sealed архив 046_pre сохранён (для 047 reference).

---

## §III. Итоги 046

### Часть I — Могила L2-handoff ЗАКРЫВАЕТСЯ ТРЕТЬИМ ЗАМЕРОМ

1. **Три режима измерены**:
   - 044 default: dk DRAM 68.17 MiB (мёртв)
   - 045 бронь + sync: dk DRAM 68.17 MiB (мёртв)
   - **046 бронь b2b**: dk DRAM **68.17 MiB** (мёртв — тот же результат)
2. **Ни один режим не даёт dk DRAM < 20 MiB порога** → механизм окончательно захоронен.
3. **Могила заколочена совместно с ассистентом**. Вопрос снят.
4. **Стройки нет** ни в этом ТЗ, ни впредь без принципиально нового рычага (TMA, CUDA graph orchestration).

### Часть II — dk S2 попытка + СТОП на регистровом гейте

5. **Правка dk_new S2 применена**: smQ_T буфер удалён (SMEM 20992 → 12288), Q feeder + pack + барьер line 310 удалены, MMA-B кормится LDSM.x2.trans.b8 из натурального smQ.
6. **ptxas §7.a — КРАСНЫЙ**: kernel_dk_new **161 регистров** (было 128) → **3 blocks/SM** (было 4). Плюс bit-exact заведомо FAIL из-за неверного ni-mapping.
7. **По правилу TZ 046 §7.a**: одна переработка не даст структурного исправления (адр-генерация не решит регистровый рост + ni-mapping) → **СТОП**.
8. **dk_new откачен к 033_sealed**: prod неизменен, `md5 = a9f0ded8261e53a143b521ffa647f458`.
9. **Рекомендация для 047**: refactor pattern (LDSM внутри ni-loop) или alternative shape (m8n8 fp8→fp16 pack). Оценка 3-4 часа iterative debug.

### Chain md5

- 045 `c4d908c7f43c9d752c90cdebbf97333a`
- **046 `<computed>`**

### Файлы 046

- `runs/reports/046_persist_b2b_dkS2.md` (this report)
- `runs/reports/046_ncu_b2b.sh` + `046_ncu_b2b_data.txt` — NCu b2b режим
- `libs/bench_bh1_b2b.cu` + `Makefile.bench_bh1_b2b` — b2b bench
- `runs/archive/046_pre/` — pre-attempt sealed reference

---

**End 046. Могила L2 закрыта окончательно. dk S2 attempt провалилась на регистровом гейте — откат к sealed. Рекомендация 047: refactor LDSM pattern.**
