#!/usr/bin/env python3
"""
027 пункт 2b: CPU-перебор — судья по PI_V для dq_new pack+π.
Проверяет ОБЕ стороны (ST после pack + LD B-load) на биекцию.
"""

QT_STRIDE = 68
Bc, Br, Hd = 64, 64, 128
NI_DQ = 16
KB_DQ = 2

def pi_v(r):
    return ((r & 7) << 2) | (((r >> 3) & 1) << 1) | ((r >> 4) & 1) | (r & 0x60)

def bank_of(byte_addr):
    return (byte_addr // 4) % 32

# ============================================================
# ASSERT 1: ST-side (post-pack Phase D + PI_V on base_row)
# Per warp per (slot, ni_slot, j): 32 lanes issue 32 STS.32.
# Bank should span 32 unique values (no conflict).
# ============================================================
def assert1_st_bijection():
    print("=== ASSERT 1: ST-side bijection (post-pack + PI_V) ===")
    fail = 0
    for wid in range(4):
        for slot in range(4):
            slot_half = (slot >> 1) & 1
            slot_ni_hi = slot & 1
            ni_base = slot_ni_hi * 4
            for j in range(4):
                banks = set()
                for lane in range(32):
                    l_div4 = lane >> 2
                    l_mod4 = lane & 3
                    c = l_mod4
                    p = l_div4 & 3
                    h = l_div4 >> 2
                    ks = wid
                    base_row = ks*32 + 16*slot_half + 4*c + p
                    row_pi = pi_v(base_row)
                    col_base = (ni_base + j)*8 + 4*h
                    byte_addr = row_pi * QT_STRIDE + col_base
                    b = bank_of(byte_addr)
                    banks.add(b)
                if len(banks) != 32:
                    print(f"  FAIL wid={wid} slot={slot} j={j}: {len(banks)}/32 banks")
                    fail += 1
    if fail == 0:
        print(f"  PASS: 4·4·4 = 64 STS.32 групп × 32 lanes → 32 distinct banks each ✓")
    return fail

# ============================================================
# ASSERT 2: LD-side (B-load with PI_V on n_d)
# Reader: smK_area[PI_V(n_d) * KT_STRIDE + k_j_lo/hi], n_d = ni_R*8+l_div4
# ============================================================
def assert2_ld_bijection():
    print("\n=== ASSERT 2: LD-side bijection (B-load + PI_V) ===")
    fail = 0
    for kb in range(KB_DQ):
        for ni_R in range(NI_DQ):
            for half_R in (0, 1):
                banks = set()
                for lane in range(32):
                    l_div4 = lane >> 2
                    l_mod4 = lane & 3
                    n_d = ni_R * 8 + l_div4
                    k_j = kb * 32 + l_mod4 * 4 + (16 if half_R == 1 else 0)
                    row_pi = pi_v(n_d)
                    byte_addr = row_pi * QT_STRIDE + k_j
                    b = bank_of(byte_addr)
                    banks.add(b)
                if len(banks) != 32:
                    print(f"  FAIL kb={kb} ni={ni_R} half={half_R}: {len(banks)}/32 banks")
                    fail += 1
    if fail == 0:
        print(f"  PASS: KB×NI×2 = 2·16·2 = 64 LDS.32 групп × 32 lanes → 32 distinct banks each ✓")
    return fail

# ============================================================
# ASSERT 3: Row bijection (PI_V permutes rows 0..127 bijectively)
# ============================================================
def assert3_row_bijection():
    print("\n=== ASSERT 3: PI_V row bijection [0..127] ===")
    seen = set()
    for r in range(128):
        seen.add(pi_v(r))
    if len(seen) == 128 and min(seen) == 0 and max(seen) == 127:
        print(f"  PASS: 128 rows → 128 unique phys rows ✓")
        return 0
    print(f"  FAIL: {len(seen)}")
    return 1

# ============================================================
# ASSERT 4: Byte-inventory invariant (pack + PI_V covers same 8192 bytes)
# ============================================================
def assert4_byte_invariant():
    print("\n=== ASSERT 4: побайтовое покрытие 8192 (invariant of PI_V permutation) ===")
    covered = set()
    for wid in range(4):
        for lane in range(32):
            l_div4 = lane >> 2
            l_mod4 = lane & 3
            c = l_mod4
            p = l_div4 & 3
            h = l_div4 >> 2
            ks = wid
            for slot in range(4):
                slot_half = (slot >> 1) & 1
                slot_ni_hi = slot & 1
                ni_base = slot_ni_hi * 4
                base_row = ks*32 + 16*slot_half + 4*c + p
                row_pi = pi_v(base_row)
                for j in range(4):
                    col_base = (ni_base + j)*8 + 4*h
                    for b in range(4):
                        addr = row_pi * QT_STRIDE + col_base + b
                        covered.add(addr)
    expected = set()
    for r in range(Hd):
        for cc in range(Bc):
            expected.add(r * QT_STRIDE + cc)
    if covered == expected:
        print(f"  PASS: обе разметки покрывают one and the same 8192 bytes ✓")
        return 0
    diff = expected - covered
    print(f"  FAIL: missing {len(diff)}, extra {len(covered - expected)}")
    return 1

def report_predictions():
    print("\n=== PREDICTIONS (P24-analog для dq pack+π) ===")
    print("  ST (Phase D packed-STS.32 + PI_V): 0.00 conflict")
    print("  LD (B-load with PI_V on n_d):      0.00 conflict")
    print("  Ожидание post-NCu: ST 150M → <25M, LD 543M → ~150M")

def main():
    print("027 pack+π для dq_new — CPU-судья (метод 021)\n")
    fails = 0
    fails += assert1_st_bijection()
    fails += assert2_ld_bijection()
    fails += assert3_row_bijection()
    fails += assert4_byte_invariant()
    report_predictions()
    print(f"\n=== SUMMARY: {'ALL GREEN' if fails==0 else f'{fails} FAILED'} ===")
    return fails

if __name__ == "__main__":
    exit(main())
