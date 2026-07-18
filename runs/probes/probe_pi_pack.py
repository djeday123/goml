#!/usr/bin/env python3
"""
021 пункт 2-fix: CPU-перебор bank-разметки для pack Q_T scatter С π_V.

Логическая строка (из fa_bwd_dk_new.cu:210-215):
  row = 32*ks + 16*hk + 4*c + p        где hk = s>>1, s0 = s&1
  colbase(bytes) = 16*wid + 8*s0 + 4*h
  QT_STRIDE = 68

π_V permutation (из 016):
  PI_V(r) = ((r&7)<<2) | (((r>>3)&1)<<1) | ((r>>4)&1) | (r & 0x60)
  Битовая карта: r0→phys2, r1→phys3, r2→phys4, r3→phys1, r4→phys0, r5→phys5, r6→phys6

Layout битов row:
  bit[1:0] = p, bit[3:2] = c (c0=b2, c1=b3), bit[4] = hk, bit[6:5] = ks

Физ. строка (после π_V):
  bit[0] = hk (was bit4), bit[1] = c1, bit[3:2] = p, bit[4] = c0, bit[6:5] = ks
  phys = 32ks + 16c0 + 4p + 2c1 + hk

bank = 17*phys + col_word (mod 32)
     = (16c0 + 4p + 2c1 + 17hk + 4wid + 2s0 + h) mod 32   [17*32ks ≡ 0]

Lane-часть (в биекцию на {0..31}): 16c0 + 4p + 2c1 + h
"""

QT_STRIDE = 68  # bytes
WARP_BANKS = 32

def pi_v(r):
    return ((r & 7) << 2) | (((r >> 3) & 1) << 1) | ((r >> 4) & 1) | (r & 0x60)

def row_logical(ks, hk, c, p):
    return 32*ks + 16*hk + 4*c + p

def colbase_bytes(wid, s0, h):
    return 16*wid + 8*s0 + 4*h

def bank_of(byte_addr):
    return (byte_addr // 4) % 32

def word_addr(row_phys, col_bytes):
    return (row_phys * QT_STRIDE + col_bytes) // 4

# ============================================================
# ASSERT 1: за fixed (ks, s, wid), 32 lanes (c,p,h) кроют 32 разных банка
# ============================================================
def assert1_lanes_bijection():
    print("=== ASSERT 1: 32 lanes → 32 different banks per STS group ===")
    failed = 0
    for hk in [0, 1]:
        for ks in range(4):
            for s0 in [0, 1]:
                for wid in range(4):
                    banks = set()
                    banks_list = []
                    for c in range(4):
                        for p in range(4):
                            for h in [0, 1]:
                                row_log = row_logical(ks, hk, c, p)
                                row_phys = pi_v(row_log)
                                col = colbase_bytes(wid, s0, h)
                                b = bank_of(row_phys*QT_STRIDE + col)
                                banks.add(b)
                                banks_list.append(b)
                    if len(banks) != 32:
                        # find duplicates
                        from collections import Counter
                        cnt = Counter(banks_list)
                        dups = {k:v for k,v in cnt.items() if v>1}
                        missing = set(range(32)) - banks
                        print(f"  FAIL (hk={hk} ks={ks} s0={s0} wid={wid}): {len(banks)}/32 banks; dups={dups}, missing={sorted(missing)}")
                        failed += 1
    if failed == 0:
        print(f"  PASS: 64 configs (hk×ks×s0×wid = 2·4·2·4) × 32 lanes → 32 distinct banks each ✓")
    return failed

# ============================================================
# ASSERT 2: физ.строки — биекция (ks, s, c, p) → 0..127 (за фиксированное h)
# ============================================================
def assert2_rows_bijection():
    print("\n=== ASSERT 2: физ.строки биекция (ks, hk, c, p) → 0..127 ===")
    phys_seen = {}
    for ks in range(4):
        for hk in [0, 1]:
            for c in range(4):
                for p in range(4):
                    row_log = row_logical(ks, hk, c, p)
                    row_phys = pi_v(row_log)
                    if row_phys in phys_seen:
                        print(f"  FAIL: phys {row_phys} from ({ks},{hk},{c},{p}) also from {phys_seen[row_phys]}")
                        return 1
                    phys_seen[row_phys] = (ks, hk, c, p)
    if len(phys_seen) == 128 and min(phys_seen) == 0 and max(phys_seen) == 127:
        print(f"  PASS: 128 логических строк → 128 уникальных физ.строк ∈ [0..127] ✓")
        return 0
    print(f"  FAIL: {len(phys_seen)} phys, min={min(phys_seen)}, max={max(phys_seen)}")
    return 1

# ============================================================
# ASSERT 3: побайтовое покрытие идентично production (инвариант байтов)
# ============================================================
def assert3_byte_invariant():
    """
    Production (no π_V): row = 32ks + 16hk + 4c + p; байт-адрес b = row*68 + col + byte_off.
    With π_V: byte-addr = pi_v(row)*68 + col + byte_off.
    Set of byte-addresses per fixed s-slot (over all lanes × 4 STS.32 output stores):
      - каждое STS.32 записывает 4 байта в один row (4 consecutive bytes at colbase..colbase+3).
      - 32 lanes × 4 s-slots × 4 STS.32-outputs × 4 bytes = 32·4·4·4 = 2048 bytes per warp/qt.
      - 4 warps per block: 2048 · 4 = 8192 bytes.
    Check: множество byte-addr for pi=identity vs pi=PI_V — оба покрывают ТЕ ЖЕ 8192 позиции?
    For invariant to hold, PI_V must be a PERMUTATION of row-space [0..127] which it is (ASSERT 2).
    So the set of covered byte-addresses is identical.
    """
    print("\n=== ASSERT 3: побайтовое покрытие (инвариант байтов) ===")
    # Enumerate for one full qt (4 warps × 4 s × 32 lanes × 4 STS.32 outputs × 4 bytes)
    # Each STS.32 write: row = ks_out*32 + row_base_ks; row_base_ks = 16*hk + 4c + p; col = colbase
    # ks_out ∈ [0..3] is the OUTER ks (write target row-block), NOT the input ks.
    # From code:
    #   for s ∈ [0..3]:      # 4 slots
    #     4 STS.32 writes to rows: (0*32 + row_base_ks), (1*32 + row_base_ks), (2*32 + row_base_ks), (3*32 + row_base_ks)
    #     col = colbase(wid, s0=s&1, h)
    # → across all lanes×s×wid, each of 128 rows is written by exactly SOME (wid, s, c, p, h)
    # For each row × col pair we get 4 bytes. Total 128 rows × 64 col-bytes = 8192 bytes.
    covered_pi = set()
    covered_nopi = set()
    for wid in range(4):
        for lane in range(32):
            c = lane & 3
            p = (lane >> 2) & 3
            h = (lane >> 4) & 1
            for s in range(4):
                hk = (s >> 1) & 1
                s0 = s & 1
                col = colbase_bytes(wid, s0, h)
                row_base_ks = 16*hk + 4*c + p
                for ks_out in range(4):
                    row_log = 32*ks_out + row_base_ks
                    row_phys = pi_v(row_log)
                    for byte_off in range(4):
                        covered_pi.add((row_phys, col + byte_off))
                        covered_nopi.add((row_log, col + byte_off))
    print(f"  covered_pi bytes = {len(covered_pi)}, expected 8192")
    print(f"  covered_nopi bytes = {len(covered_nopi)}, expected 8192")
    # Invariance check: PI is bijection of rows → same {(row, col)} SET (as sets of (row, col)),
    # since PI is a permutation of the row axis.
    rows_pi = set(r for r,c in covered_pi)
    rows_nopi = set(r for r,c in covered_nopi)
    cols_pi = set(c for r,c in covered_pi)
    cols_nopi = set(c for r,c in covered_nopi)
    ok = (len(covered_pi) == 8192 and len(covered_nopi) == 8192 and
          rows_pi == rows_nopi and cols_pi == cols_nopi)
    if ok:
        print(f"  PASS: обе разметки покрывают one and the same set — [0..127] × [0..63] = 8192 ✓")
        return 0
    print(f"  FAIL: rows_pi={sorted(rows_pi)[:5]}... rows_nopi={sorted(rows_nopi)[:5]}...")
    return 1

# ============================================================
# ASSERT 4: B-сторона — PI_NI сравнение с P18-P21 паттернами
# ============================================================
def assert4_bside_pattern():
    """
    B-load (точная формула из fa_bwd_dk_new.cu:237-239):
      n_d    = ni * 8 + l_div4            # l_div4 = l/4 ∈ [0..7]
      k_i_lo = kb * 32 + l_mod4 * 4       # l_mod4 = l%4 ∈ [0..3]
      k_i_hi = kb * 32 + l_mod4 * 4 + 16
      B0     = smQ_T[PI_V(n_d) * QT_STRIDE + k_i_lo]
      B1     = smQ_T[PI_V(n_d) * QT_STRIDE + k_i_hi]

    NI = Hd/8 = 16, KB = Br/32 = 2.
    Per lane per (ni, kb): 2 LDS.32 (lo, hi).
    Bijection required: 32 lanes → 32 distinct banks per LDS group.
    """
    print("\n=== ASSERT 4: B-сторона точный перебор (fa_bwd_dk_new.cu:237-239) ===")
    Hd = 128
    Br = 64
    NI = Hd // 8      # 16 outer n-fragments
    KB = Br // 32     # 2 k-blocks (Br=64 → 2×32)
    failed_groups = 0
    total_groups = 0
    for ni in range(NI):
        for kb in range(KB):
            # 2 LDS groups: k_i_lo and k_i_hi
            for kside in (0, 1):     # 0 = lo, 1 = hi
                banks = []
                for lane in range(32):
                    l_div4 = lane >> 2
                    l_mod4 = lane & 3
                    n_d = ni * 8 + l_div4
                    k_i = kb * 32 + l_mod4 * 4 + (16 if kside else 0)
                    row_phys = pi_v(n_d)
                    b = bank_of(row_phys * QT_STRIDE + k_i)
                    banks.append(b)
                uniq = len(set(banks))
                if uniq != 32:
                    from collections import Counter
                    cnt = Counter(banks)
                    dups = {k:v for k,v in cnt.items() if v>1}
                    print(f"  FAIL (ni={ni}, kb={kb}, {'hi' if kside else 'lo'}): {uniq}/32 banks; dups={dups}")
                    failed_groups += 1
                total_groups += 1
    if failed_groups == 0:
        print(f"  PASS: {total_groups} B-LDS.32 групп (NI×KB×2 = {NI}×{KB}×2) × 32 lanes → 32 distinct banks each ✓")
        print(f"  P18-P21 воспроизведено переебор'ом, не ссылкой — 0.00 LD conflict.")
        return 0
    return failed_groups

# ============================================================
# Predicted metrics per Vugar TZ (register for P24)
# ============================================================
def report_predictions():
    print("\n=== PREDICTIONS (P24) — регистрирую ===")
    print("  P24 hk=0: 0.00 LD conflict / 0.00 ST conflict (packed-STS.32 @68 + π_V)")
    print("  P24 hk=1: 0.00 LD conflict / 0.00 ST conflict")
    print("  ks/s0 зафиксированы = 0 (константные сдвиги, не влияют на bijection lane-set)")

# ============================================================
def main():
    print("021 π_V pack — CPU-перебор (машинный судья над двумя ручными алгебрами)\n")
    fails = 0
    fails += assert1_lanes_bijection()
    fails += assert2_rows_bijection()
    fails += assert3_byte_invariant()
    fails += assert4_bside_pattern()
    report_predictions()
    print(f"\n=== SUMMARY: {'ALL GREEN' if fails==0 else f'{fails} FAILED'} ===")
    return fails

if __name__ == "__main__":
    exit(main())
