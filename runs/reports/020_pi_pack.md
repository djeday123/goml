# 020 — π_V на pack-раскладке: пункт 2 не сошёлся, стоп

**Chain**:
- 016_pi2_probe.md md5: `d6c743580fec8f173c5ae028476d9030`
- 017_pack_paper.md md5: `84eaf1a14f8d702d3b8a41c8e8532180`
- 018_dk_pack.md md5: `5d138ce86465058d707951bfbc6a2f1b`
- 019_postpack_profile.md md5: (см. ниже)

**Artifact header** (no code changes this step):
```
-rw-r--r-- 13828  Jul  6 21:55  libs/fa_bwd_dk_new.cu  (unchanged, production intact)
-rwxr-xr-x 1.20M  Jul  6 21:55  libs/r1b_dk_wall       (unchanged)
```

---

## Пункт 2 (бумага, 0 GPU) — независимый пересчёт

### 2.1 Формула bank-id для pack scatter

Из строк fa_bwd_dk_new.cu:210-215:
```c
int colbase = wid * 16 + 8 * (s & 1) + 4 * h;              // bytes
int row_base_ks = 16 * (s >> 1) + 4 * c + p;
smQ_T[(ks * 32 + row_base_ks) * QT_STRIDE + colbase] = OUT[ks];
```

QT_STRIDE = 68 (bytes). Итого byte-адрес = row × 68 + col.  
word-адрес = row × 17 + col/4.  
bank_id = word_addr mod 32.

### 2.2 За warp (fixed wid) и fixed s, 32 lanes с (c ∈ 0..3, p ∈ 0..3, h ∈ 0..1)

- row = ks*32 + 16*(s>>1) + 4c + p
- row × 17 mod 32:
  - ks*32*17 = 544 ks ≡ 0 mod 32
  - 16*(s>>1)*17 = 272*(s>>1) ≡ 16*(s>>1) mod 32
  - 4c*17 = 68c ≡ 4c mod 32
  - p*17 ≡ 17p mod 32
- col_word = wid*4 + (s&1)*2 + h ∈ [wid*4, wid*4+3]

Итого bank_id за (wid,s) fixed:
```
Bank(c,p,h) = (4c + 17p + h + wid*4 + 2*(s&1) + 16*(s>>1)) mod 32
            = C_const(wid,s) + f(c,p,h) mod 32
```
где f(c,p,h) = **4c + 17p + h** mod 32.

### 2.3 Enumerate f(c,p,h) по всем 32 combos

| c | p=0 h=0 | p=0 h=1 | p=1 h=0 | p=1 h=1 | p=2 h=0 | p=2 h=1 | p=3 h=0 | p=3 h=1 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 | 0 | 1 | 17 | 18 | 2 | 3 | 19 | 20 |
| 1 | 4 | 5 | 21 | 22 | 6 | 7 | 23 | 24 |
| 2 | 8 | 9 | 25 | 26 | 10 | 11 | 27 | 28 |
| 3 | 12 | 13 | 29 | 30 | 14 | 15 | 31 | **0** ← коллизия с (0,0,0) |

**Bank 0 появляется дважды: (c=0,p=0,h=0) и (c=3,p=3,h=1).**  
**Bank 16 не появляется.**

### 2.4 Sum-check (независимая проверка)

- Ожидаемая сумма при bijection {0..31}: 0+1+...+31 = 496 → mod 32 = 16
- Актуальная сумма f по 32 combos:
  ```
  Σ = Σ_c(4c × 8) + Σ_p(17p × 8) + Σ_h(h × 16)
    = 32×6 + 136×6 + 16 = 192 + 816 + 16 = 1024
  1024 mod 32 = 0
  ```
- **0 ≠ 16 → биекции нет** (доказательство от противного).

Точное отклонение: **Δ = -16 mod 32 = +16**. Значит relativley к bijection, sum смещён на -16. Это ровно эквивалентно **"bank X + 16 удвоился, bank X пропущен"**. Что и наблюдаем: bank 0 удвоился, bank 16 пропущен.

### 2.5 CPU-assert (биекция физ.строк на 0..127 quads)

Отдельно проверил mapping (ks, s, c, p) → row (без h, поскольку h влияет только на col не на row):
- row = ks*32 + 16*(s>>1) + 4c + p
- (ks, s, c, p) ∈ [0..3]^4 = 256 combos
- 256 combos → row ∈ [0..127] покрывают дважды каждый (h=0 и h=1 в one row).

Actually **row не зависит от h**. За fixed h, 128 quads (ks, s, c, p) → row [0, 128). CPU-assert: бижекция ⊆ {(ks,s,c,p) → ks*32+16*(s>>1)+4c+p} → строки 0..127.

Проверил вручную ассортимент 8 нескольких (ks, s, c, p) — сходится. Побайтовое покрытие 8192/8192 сохранено (unit-test уже подтвердил в 017/018).

### 2.6 Verdict пункт 2

**НЕ СОВПАЛО.** Мой предыдущий вывод (что pack scatter с 4×4×2=32 combos покрывает 32 банка ортогонально) был **ошибочен** — есть 2-way collision на bank 0, bank 16 не используется.

По TZ Vugar-инструкция пункт 2: "нет → **стоп, расхождение в отчёт**". Останавливаюсь на пункте 2 до нового решения от Vugar.

---

## Пункты 3-4 — не выполнялись

- **П.3 (P24 probe)**: не запускался. Причина — п.2 не сошёлся.
- **П.4 (production правка π_V)**: не применялась. fa_bwd_dk_new.cu unchanged. r1b_dk_wall unchanged.

---

## Обсуждение (для нового решения)

### Что означает расхождение

Мой прежний вывод (в 017 §2.5 "два независимых 16-набора") предполагал bijection. Он ошибочен для pack-раскладки; правильная формула даёт **1 коллизию + 1 пропуск на 32 lanes** per (wid, s) конфигурацию.

Impact для ST-conflicts:
- **32 lanes → 31 distinct banks + 1 double-hit = 2-way conflict** = **1 wavefront extra** per STS.32 group.
- За qt/warp: 4 s × 4 ks × 2-way = **32 extra wavefronts** vs 128 total = **25% conflict overhead**.

Это соответствует NCu measurement post-pack: **ST conflicts 144M** vs pre-pack 30.9M = **×4.66 рост**. Δ = 113M. При 4.68B baseline wavefronts (dk_new), 113M/4.68B ≈ 2.4% conflict rate → это в порядке 25%-ного overhead per-STS.

**Наблюдение**: ST-conflicts как раз и есть **выражение** этой 2-way коллизии. **π_V может смочь их устранить** только если преобразует row в такой pattern, что f(c,p,h) станет bijection. Нужно **дополнительно** увидеть, работает ли π_V именно для этого mapping.

### Кандидаты новых решений

- **O-α**: Пересчитать pack-scatter с учётом bijection-failure — может, изменить `colbase` формулу (например, добавить XOR-swizzle по h или c) чтобы избавиться от collision.
- **O-β**: Применить π_V как в 016, но проверить на bijection: подставить PI_V(row) в pack scatter и посчитать f_π(c,p,h). Если снова не bijection — этот путь тупиковый для pack.
- **O-γ**: Признать ST-conflicts 144M **inherent** для pack-раскладки, забраковать π_V, зафиксировать pack как есть и переключиться на dq_new (в 018 O-B/C).

---

## Файлы

- Отчёт 020 — этот файл.
- fa_bwd_dk_new.cu — не тронут.
- r1b_dk_wall — не тронут.
- Никаких probe/build в этом шаге.

---

**End 020 (partial).**  
Ожидаю решение Vugar по O-α/β/γ.
