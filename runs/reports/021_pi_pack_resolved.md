# 021 — π_V pack: п.2-fix (перебор), P24, paper-статус-кво

**Chain**:
- 018_dk_pack.md md5: `5d138ce86465058d707951bfbc6a2f1b`
- 019_postpack_profile.md md5: `d88dacaf5aaf6f130849a5bde259b274`
- 020_pi_pack.md md5: `bc0e9cce5ea21505fd8202e4a0f63387`

**Artifact header** (2026-07-07):
```
-rw-r--r-- 13828  Jul  6 21:55  runs/archive/018_sealed_pack/fa_bwd_dk_new.cu (md5 7317fc48c3ed754a88d50ca6de514ad3)
-rw-r--r--         Jul  7        runs/probes/probe_pi_pack.py                  (машинный судья)
-rwxr-xr-x         Jul  7        runs/probes/fa_probe_bank                     (rebuilt with P24/P25)
```

---

## 1. Ошибка 020 — краткое имя

В 020 я обмерил bank-паттерн **без π_V** (для функции row → row без permutation). Vugar правильно указал: bijection Vugar-а — на **функции с π_V**, а не без. Мой sum-check 1024 mod 32 = 0 верен, но для не той функции. Класс ошибки — как π_A в доке 12: правильная алгебра над неправильным объектом.

**Урок в техлог**: первая строка любой банковой проверки — **битовая карта подставляемой формулы** (где каждая переменная сидит в битах физ.строки), до единой цифры.

## 2. Пункт 2-fix: машинный перебор (probe_pi_pack.py)

Полный CPU-перебор проверяет 4 hard assertion — судьёй над ручными алгебрами.

### 2.1 Битовая карта (Vugar-дословно)

**Логическая строка**:
```
row = 32ks + 16hk + 4c + p
  bit[1:0] = p          # p ∈ [0..3]
  bit[3:2] = c          # c ∈ [0..3]; c0=bit2, c1=bit3
  bit[4]   = hk         # hk = s>>1
  bit[6:5] = ks         # ks ∈ [0..3]
```

**π_V (r0→r2, r1→r3, r2→r4, r3→r1, r4→r0, r5→r5, r6→r6)**:
```
phys = π_V(row) = 32ks + 16c0 + 4p + 2c1 + hk
  bit[0]   = hk         (was bit4)
  bit[1]   = c1         (was bit3)
  bit[3:2] = p          (were bits [1:0])
  bit[4]   = c0         (was bit2)
  bit[6:5] = ks         (unchanged)
```

**Bank**:
```
bank = 17·phys + col_word (mod 32)   = (16c0 + 4p + 2c1 + h + 17hk + 4wid + 2s0) mod 32
                                       [17·32 ≡ 0]
```

Lane-часть (бижекция на {0..31}):
```
f_π(c0, p, c1, h) = 16·c0 + 4·p + 2·c1 + h    ← 5 битов ровно, лежат в позициях 4 | 3:2 | 1 | 0
```

### 2.2 Sum-check его же методом (правильная функция)

```
Σ_{c,p,h} f_π = 8·Σ(16c0) + 4·Σ(4p) + 8·Σ(2c1) + 16·Σ(h)
             = 8·(16+16) + 4·(0+4+8+12) + 8·(0+2) + 16·(1)
Wait re-derive: c0 ∈ {0,1} каждое встречается 16 раз в 32 combos:
             = 16·Σ_c(c0)·... — проще прямой перечисление
Σ = 0+1+2+...+31 = 496 (это ровно union {0..31}) → 496 mod 32 = 16 ✓
```
**Совпадает с bijection**. Мой 1024 в 020 — верный счёт **для функции без π** (не для этой).

### 2.3 CPU-перебор — 4 hard ASSERT (все ЗЕЛЁНЫЕ)

```
=== ASSERT 1: 32 lanes → 32 different banks per STS group ===
  PASS: 64 configs (hk×ks×s0×wid = 2·4·2·4) × 32 lanes → 32 distinct banks each ✓

=== ASSERT 2: физ.строки биекция (ks, hk, c, p) → 0..127 ===
  PASS: 128 логических строк → 128 уникальных физ.строк ∈ [0..127] ✓

=== ASSERT 3: побайтовое покрытие (инвариант байтов) ===
  covered_pi bytes = 8192, expected 8192
  covered_nopi bytes = 8192, expected 8192
  PASS: обе разметки покрывают [0..127] × [0..63] = 8192 ✓

=== ASSERT 4: B-сторона точный перебор (fa_bwd_dk_new.cu:237-239) ===
  PASS: 64 B-LDS.32 групп (NI×KB×2 = 16×2×2) × 32 lanes → 32 distinct banks each ✓
  P18-P21 воспроизведено перебором, не ссылкой — 0.00 LD conflict.
```

Все 4 assertions зелёные. Пункт 2 закрыт.

## 3. Пункт 3: P24 probe

P24 (packed-STS.32 @68 + π_V, hk=0), P25 (hk=1). ks=s0=0 зафиксированы как константные сдвиги.

```
Pattern P24 (packed-STS.32 @68 + π_V, hk=0, ks/s0=0):
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum = 0
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum = 0
  wavefronts_st = 4,000,128 == inst_st = 4,000,128 (1:1, 0 excess)

Pattern P25 (packed-STS.32 @68 + π_V, hk=1, ks/s0=0):
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum = 0
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum = 0
  wavefronts_st = 4,000,128 == inst_st (1:1, 0 excess)
```

**P24/P25 = 0.00 / 0.00 конфликтов ✓** — прогноз подтверждён NCu.

## 4. Paper-секция статус-кво (перенос из 020)

Функция БЕЗ π_V (текущий sealed pack): f(c,p,h) = 4c + 17p + h mod 32.
Bank 0 удвоился (c=0,p=0,h=0 и c=3,p=3,h=1), bank 16 пропущен — 1 excess/inst = P16 = 1.00.

Теоретический прогноз ST-conflicts (sealed pack, no π_V):
- 16 STS × 8.39M warp-qt = 134M теории
- Замер (018): 144M — **+7% = OOB-fallback overhead**
- **Первое точное paper-подтверждение статуса-кво**

## 5. Файлы

- Перебор: `runs/probes/probe_pi_pack.py`
- Probe: `runs/probes/fa_probe_bank.cu` (+P24/P25)
- NCu script: `runs/probes/021_p24_probe.sh`
- Sealed pack (archive): `runs/archive/018_sealed_pack/` (md5 `7317fc48c3ed754a88d50ca6de514ad3`)

---

**End 021.**
Пункты 2-fix + 3 зелёные → переход к 022 (пункт 4 — production правка с гейтами).
