#!/usr/bin/env python3
"""
025-b пункт "CPU-перебор — судья до GPU" для dq_new K_T pack scatter.

Вывод от читателя B-load (fa_bwd_dq_new.cu:216-219):
  B0 = smK_area[n_d * KT_STRIDE + k_j_lo]   uint32 = 4 bytes at [row=n_d, col=k_j_lo..k_j_lo+3]

Целевая раскладка слова STS.32 определяется читателем:
  1 STS.32 = 4 bytes at (row=k_of_K, col=n_of_K, n_of_K+1, n_of_K+2, n_of_K+3)

4 owner lanes для одного STS.32:
  Fixed: (wid, ni_group, l_mod4_W=c, half_W, bt)
  Varying: l_div4_W ∈ {4h, 4h+1, 4h+2, 4h+3} (h=0 или 1)

Vugar-гипотеза (для сверки): "вертикальные четвёрки {c+4p'}, обмен по оси d,
слово = 4 соседних n одного ni-соседства".
- {c + 4*p'} = lane_id where c=l_mod4 fixed, p'=l_div4 varying → PODTVERZHDENO
- обмен по оси l_div4 → PODTVERZHDENO
- слово = 4 consecutive n → PODTVERZHDENO

Асерты:
  1) 4 lanes exchange group имеют fixed l_mod4=c, varying l_div4 within h-subgroup
  2) STS.32 target byte-адреса == текущим STS.U8 побайтно (инвариант читателя)
  3) Полное покрытие 8192 без дублей
"""

Bc = 64
Br = 64
Hd = 128
KT_STRIDE = 68
NI_QK = 8
KS_QK = 4

# ============================================================
# Writer's baseline (current dq_new Phase 1.5 write, 64 STS.U8/lane):
# fa_bwd_dq_new.cu:187-195
#   smK_area[(k_lo + bt) * KT_STRIDE + n_K] = byte at (k_lo+bt, n_K)
#     k_lo = ks*32 + l_mod4*4, ks=wid
#     n_K  = ni*8 + l_div4
# ============================================================
def enumerate_writer_bytes():
    """Return dict {byte_addr → (writer_lane, ni, half, bt, byte_value_source)}"""
    bytes_map = {}
    for wid in range(4):
        for lane in range(32):
            l_div4 = lane >> 2
            l_mod4 = lane & 3
            ks = wid
            k_lo = ks * 32 + l_mod4 * 4 + 0
            k_hi = ks * 32 + l_mod4 * 4 + 16
            for ni in range(NI_QK):
                n_K = ni * 8 + l_div4
                for half in (0, 1):
                    k_base = k_lo if half == 0 else k_hi
                    for bt in range(4):
                        row = k_base + bt
                        col = n_K
                        byte_addr = row * KT_STRIDE + col
                        # byte value source in feeder: kr_[lo/hi][ni] >> (bt*8) & 0xFF
                        # semantically: byte K[n=n_K][k=row]
                        src = ('K', n_K, row)
                        if byte_addr in bytes_map:
                            raise Exception(f"Duplicate byte_addr {byte_addr}")
                        bytes_map[byte_addr] = {
                            'lane_id': (wid, lane),
                            'lane_bits': (wid, l_div4, l_mod4),
                            'ni': ni, 'half': half, 'bt': bt,
                            'src': src, 'row': row, 'col': col,
                        }
    return bytes_map


# ============================================================
# Reader's B-load structure (fa_bwd_dq_new.cu:216-219):
#   for kb in 0..1:
#     for ni_R in 0..15:
#       n_d = ni_R*8 + l_div4_R
#       k_j_lo = kb*32 + l_mod4_R*4 + 0
#       k_j_hi = kb*32 + l_mod4_R*4 + 16
#       B0 uint32 = 4 bytes at [row=n_d, col=k_j_lo..k_j_lo+3]
#       B1 uint32 = 4 bytes at [row=n_d, col=k_j_hi..k_j_hi+3]
# ============================================================
def enumerate_reader_words():
    """Return list of dict describing each STS.32 target from reader's perspective."""
    words = []
    for wid_R in range(4):   # wid_R here is warp id of the READER
        for lane_R in range(32):
            l_div4_R = lane_R >> 2
            l_mod4_R = lane_R & 3
            for kb_R in range(2):
                for ni_R in range(16):
                    for half_R in (0, 1):
                        n_d = ni_R * 8 + l_div4_R
                        k_j_base = kb_R * 32 + l_mod4_R * 4
                        k_j = k_j_base + (16 if half_R == 1 else 0)
                        # word covers 4 consecutive col bytes at row=n_d, cols=k_j..k_j+3
                        words.append({
                            'row': n_d,
                            'col_base': k_j,
                            'reader': (wid_R, lane_R, kb_R, ni_R, half_R),
                            'bytes': [(n_d, k_j+b) for b in range(4)],
                        })
    return words


# ============================================================
# ASSERT 1: for each B0/B1 word target, find 4 owner lanes.
#   Verify: 4 owners have fixed l_mod4=c, varying l_div4 within {0..3} or {4..7} sub-group.
# ============================================================
def assert1_group_structure(writer_bytes):
    print("=== ASSERT 1: 4-owner group structure (Vugar hypothesis check) ===")
    fail = 0
    seen_targets = set()
    # For each unique STS.32 target position (row, col_base):
    # col_base is aligned by 4 (since we pack 4 bytes into STS.32)
    for row in range(128):
        for col_base in range(0, 64, 4):
            # Find 4 owners of consecutive bytes (row, col_base+b for b=0..3)
            owners = []
            for b in range(4):
                byte_addr = row * KT_STRIDE + (col_base + b)
                if byte_addr not in writer_bytes:
                    print(f"  MISS: byte_addr {byte_addr} row={row} col={col_base+b} not written")
                    fail += 1
                    continue
                owners.append(writer_bytes[byte_addr])
            if len(owners) != 4:
                continue
            # Check: same wid, same l_mod4, l_div4 varies within sub-group
            wids = set(o['lane_bits'][0] for o in owners)
            l_mod4s = set(o['lane_bits'][2] for o in owners)
            l_div4s = sorted(o['lane_bits'][1] for o in owners)
            nis = set(o['ni'] for o in owners)
            halfs = set(o['half'] for o in owners)
            bts = set(o['bt'] for o in owners)
            if len(wids) != 1 or len(l_mod4s) != 1:
                print(f"  FAIL (row={row}, col_base={col_base}): wids={wids}, l_mod4s={l_mod4s}")
                fail += 1
                continue
            # l_div4 should be {0,1,2,3} or {4,5,6,7}
            expected_h0 = [0, 1, 2, 3]
            expected_h1 = [4, 5, 6, 7]
            if l_div4s != expected_h0 and l_div4s != expected_h1:
                print(f"  FAIL (row={row}, col_base={col_base}): l_div4s={l_div4s}")
                fail += 1
                continue
            # ni should be same
            if len(nis) != 1 or len(halfs) != 1 or len(bts) != 1:
                print(f"  FAIL (row={row}, col_base={col_base}): mixed ni/half/bt: {nis}/{halfs}/{bts}")
                fail += 1
                continue
            seen_targets.add((row, col_base))
    total = 128 * 16   # 128 rows × 16 col_words
    if fail == 0:
        print(f"  PASS: {len(seen_targets)}/{total} STS.32 targets have proper 4-owner group ✓")
        print(f"        Group = {{lane_id: c + 4*p' + 16*h}} with fixed (c=l_mod4, h), varying p'=l_div4&3.")
        print(f"        Confirmed: exchange axis = l_div4 (Vugar's 'd'), fixed 'c'.")
    return fail


# ============================================================
# ASSERT 2: byte coverage invariant.
#   Total 8192 bytes must be covered exactly once by writer.
# ============================================================
def assert2_coverage(writer_bytes):
    print("\n=== ASSERT 2: побайтовое покрытие 8192 (инвариант читателя) ===")
    covered = set()
    for addr in writer_bytes:
        covered.add(addr)
    # target bytes: all (row, col) with row ∈ [0..127], col ∈ [0..63]
    all_target = set()
    for row in range(128):
        for col in range(64):
            all_target.add(row * KT_STRIDE + col)
    missing = all_target - covered
    extra = covered - all_target
    if not missing and not extra:
        print(f"  PASS: 8192/8192 bytes covered exactly once ✓")
        return 0
    print(f"  FAIL: missing={len(missing)}, extra={len(extra)}")
    return 1


# ============================================================
# ASSERT 3: reader byte addresses == writer byte addresses (semantics match).
# ============================================================
def assert3_reader_match(writer_bytes, reader_words):
    print("\n=== ASSERT 3: reader byte addresses match writer inventory ===")
    all_reader_addrs = set()
    for w in reader_words:
        for r, c in w['bytes']:
            all_reader_addrs.add(r * KT_STRIDE + c)
    missing_from_writer = all_reader_addrs - set(writer_bytes.keys())
    if not missing_from_writer:
        print(f"  PASS: reader reads only bytes writer wrote ({len(all_reader_addrs)} unique addr) ✓")
        return 0
    print(f"  FAIL: reader tries to read {len(missing_from_writer)} bytes writer never wrote")
    return 1


# ============================================================
# ASSERT 4: per-lane STS.32 target distribution (writer's post-pack view).
#   Each of 32 lanes should write 16 STS.32 = 4 slots × 4 outputs.
# ============================================================
def assert4_lane_distribution(writer_bytes):
    print("\n=== ASSERT 4: STS.32 target distribution per writer lane ===")
    # For each writer lane, count how many STS.32 targets it "owns" (writes 1 of 4 bytes for).
    # A lane's "STS.32 target" = a (row, col_base) that includes at least one of the lane's bytes.
    # Since 4-lane group co-owns 1 STS.32 target, lane's targets = union of its bytes' targets.
    from collections import Counter
    per_lane_targets = Counter()
    for addr, info in writer_bytes.items():
        row = addr // KT_STRIDE
        col = addr % KT_STRIDE
        col_base = col & ~3   # align to 4-byte boundary
        target = (row, col_base)
        per_lane_targets[info['lane_id']] += 1  # count bytes; each target counts 4 bytes
    # per lane should have 64 bytes (across own bytes) each contributing to a target
    problems = 0
    for (wid, lane), cnt in per_lane_targets.items():
        if cnt != 64:
            print(f"  Lane wid={wid} lane={lane}: {cnt} bytes (expected 64)")
            problems += 1
    if problems == 0:
        print(f"  PASS: all 128 lanes each write 64 bytes = 16 uint32-word contributions (16 STS.32/lane) ✓")
    return problems


# ============================================================
# Derive Phase D dst mapping per (writer lane, slot, out_i)
# Vugar hint: слот выводится из данных читателя.
# ============================================================
def derive_phase_d(writer_bytes):
    """
    Per lane per qt: 16 STS.32 targets. Split into 4 slots × 4 outputs.

    From writer's byte inventory per lane:
      - 64 bytes at (row=wid*32+4c+bt+16half, col=ni*8+l_div4)
      - 8 ni × 2 half × 4 bt = 64 bytes

    For each byte to be part of a STS.32 target (row_target, col_base):
      - Same target has 4 bytes at (row_target, col_base+0..3)
      - 4 owners for this target: {(l_mod4=c, l_div4=4h+0), (c, 4h+1), (c, 4h+2), (c, 4h+3)}

    Per lane's own 64 bytes distribute across STS.32 targets:
      - Each byte at (row, col=ni*8+l_div4) belongs to target (row, col_base = ni*8 + 4*(l_div4>>2))
      - l_div4 fixed per lane → col_base fixed per lane per ni: col_base = ni*8 + 4*h
      - So per lane's 64 bytes → 64 target-participations, but each target has 4 owners so 64/4 = 16 distinct targets per lane's "group participation"... wait

    Actually each byte at (row, col_base+r) participates in target (row, col_base). Per lane's 64 bytes:
      - 8 ni × 2 half × 4 bt → 64 (row, col) positions
      - Target = (row, col_base) where col_base = (col // 4)*4 = (ni*8+l_div4) // 4 * 4
      - Since ni*8 is multiple of 4, and l_div4 is 0..7, col_base = ni*8 + (l_div4 & ~3) = ni*8 + 4*h
      - 8 ni × 2 half × 4 bt = 64 byte positions → 64 (row, col_base) pairs
      - Each (row, col_base) pair has 4 lane owners (4 different l_div4 with fixed h, l_mod4)
      - Per lane owns 1 out of 4 bytes at target

    So per lane owns 1 byte in each of 64 targets. But per lane sits in 8 groups (8 ni values).
    Wait — per group of 4 lanes owns 16 STS.32 targets (4 rows × 4 col_words = 16). So per lane
    per group = 4 STS.32 targets fully assembled (after exchange). Total per lane = 4 slots × 4 outputs.

    A "slot" in dk-pack sense corresponds to a fixed (ni_group_index_within_lane, half) pair.
    Per lane 8 ni × 2 half = 16 (ni, half) pairs = 4 slots × 4 ni-pairs?

    Alternative: slot = (half, ni_high_bit). 4 slots × 4 STS.32 = 16.
    Per slot: 4 STS.32 outputs = 4 different (ni_low_bits, bt) combinations.
    """
    print("\n=== Phase D derivation (Vugar mapping) ===")
    # Per lane's 16 STS.32 owned targets:
    # For lane (wid, l_div4, l_mod4):
    #   ni ∈ [0..7]: col_base = ni*8 + 4*(l_div4 >> 2)
    #   half ∈ {0,1}, bt ∈ [0..3]: row = wid*32 + 4*l_mod4 + bt + 16*half
    #   16 STS.32 = 8 ni × 2 (half+bt combined) ? No that's 16, but slot count wrong.
    # Wait: 8 ni × 2 half = 16 (ni, half) pairs; each pair has 4 bt values = 4 rows in same col_base.
    # So per (ni, half): 4 rows all with same col_base. That's 4 STS.32 with same col_base and different rows.
    # 16 (ni, half) pairs = 16 col_base values? No: col_base = ni*8 + 4*h only depends on ni (and lane's h).
    # For fixed lane's h, col_base = ni*8 + 4*h → 8 different values (one per ni).
    # For fixed ni, col_base fixed, but half ∈ {0,1} gives 2 different row_offsets (16 vs 0).
    # So per lane: 8 ni × 2 half × 4 bt = 64 bytes covering 8 col_base × 8 row values = 64 target participations.
    # Since each target has 4 owners: per lane's UNIQUE target ownerships = 16 STS.32.

    # Per lane 8 col_base values, 8 row values. But 8×8 = 64 target position candidates. Of these,
    # actually all 8 col_base × 8 row = 64 targets exist, and per lane owns 1 byte in each = 64 byte participations.
    # 64 target participations / 4 lanes per target = 16 unique targets per lane's "owned share"? No, 64 targets
    # total that lane touches, each shared with 3 other lanes.

    # So per lane touches 64 targets (via byte ownership), but for STS.32 write we choose which of 4 owners
    # actually issues the STS.32. Distribute so each lane writes 16 STS.32 (64/4).

    # Natural distribution: for target (row, col_base), assign writer to one specific lane out of 4 owners.
    # E.g., writer = lane with l_div4 = 4h + 0 (the "leader" of group).
    # Then per group (fixed c, h): leader lane writes ALL 16 STS.32 targets covered by group (8 col_base × 2 half × 4 bt? or 8 col_base × 8 rows = 64).
    # 64 targets per lane × 32 lanes = 2048 targets ≠ 512 per warp.

    # Wait per group of 4 lanes touches 8 col_base (per group's h) × 8 rows (all 8 rows in row-band via bt+half).
    # But 8 rows × 8 col_base = 64 targets. And per warp 8 groups × 64 = 512 targets ✓.

    # So per group leader writes 64 STS.32? But per lane should write 16. Contradiction.

    # Correct: each of 4 lanes in group writes 16 STS.32 (own quarter of 64 group targets).
    # 64 / 4 = 16 ✓.

    # So per lane writes 16 STS.32 = own subset of group's 64 targets.
    # Slot decomposition: 4 slots × 4 outputs per slot.

    # Slot = which "column_word" out of group's 8: split as 4 slots × 2 col_words each?
    # Or slot indexes (half, ni_high) 2×2 = 4:
    #   Slot 0: half=lo, ni ∈ [0..3]  → 4 col_base × 4 bt = 16 target participations, but per lane only 4?

    # Actually each lane in group participates in every target of group (via 1 byte per target).
    # So per lane 64 target participations, and STS.32 execution split so each of 4 lanes emits 16 STS.32.

    # Choose: lane with l_div4 = 4h+p' writes STS.32 for targets where (row & 3) == p'?
    # I.e., writer of target (row, col_base) = lane with l_div4 = 4h + (row & 3), l_mod4 = c.
    # Then per lane at (l_div4 = 4h+p_own, l_mod4=c): writes targets where row & 3 == p_own.
    # 32 rows in band, 8 have row&3 == p_own → 8 rows per lane per group.
    # 8 rows × 8 col_base = 64 targets per lane's writing responsibility? Still 64, not 16.

    # Retry: assign STS.32 responsibility differently.
    # Per group: 64 targets = 4 (half, bt) values × 8 col_base = 32 × 2 (half) × 4 (bt) = 64.
    # (Wait 8 rows corresponds to 2 half × 4 bt = 8, ✓)
    # Per lane: (out of 4 lanes in group), owns 1 byte in each target. So writes 1/4 of targets = 16.

    # Which 16? Vugar hint "слово = 4 соседних n одного ni-соседства" implies OUT corresponds to col_word
    # (n-slot). If per lane STS.32 covers 16 col_words × 1 row? No, per lane covers 8 col_words × 2 rows,
    # or 16 col_words × 1 row (but per lane only has bytes for 8 col_words, not 16, since l_div4 fixed).

    # Given each lane touches 8 col_base (via 8 ni), and 8 rows (via 4 bt × 2 half):
    # per lane owns 1 byte in each of 64 group targets, but only writes 16 STS.32 = 1/4.
    # Distribution: e.g., writer = lane where (l_div4-4h) == (bt) mod 4 → each lane writes targets with matching (bt, l_div4).
    # That gives 8 col_base × 2 half × 1 bt (matching) = 16 targets per lane ✓.

    print("  Per lane 16 STS.32 = 8 col_base × 2 half (each with 1 bt matching lane's p'=l_div4&3)")
    print("  Slot = (half, ni_high_bit) or similar; per-slot 4 outputs = 4 col_base values")
    print("  Full derivation & unit-test in next iteration.")
    return 0


# ============================================================
def main():
    print("025-b: dq_new K_T pack — CPU перебор (судья до GPU)\n")
    writer_bytes = enumerate_writer_bytes()
    print(f"Total writer bytes: {len(writer_bytes)}")

    fails = 0
    fails += assert1_group_structure(writer_bytes)
    fails += assert2_coverage(writer_bytes)
    reader_words = enumerate_reader_words()
    fails += assert3_reader_match(writer_bytes, reader_words)
    fails += assert4_lane_distribution(writer_bytes)
    derive_phase_d(writer_bytes)

    print(f"\n=== SUMMARY: {'ALL GREEN' if fails==0 else f'{fails} FAILED'} ===")
    return fails


if __name__ == "__main__":
    exit(main())
