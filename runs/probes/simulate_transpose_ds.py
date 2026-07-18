#!/usr/bin/env python3
"""
033-b CPU-судья реализации транспонирования dS 64×64: Phase A/B/C/D модель.
Вывод от читателя dk_new (fa_bwd_dk_new.cu:239-245), не от донора dq.

Битовая карта T-адреса: bit[5:0]=col_i, bit[11:6]=row_j.
Reader per lane per kb: 4 A-uint32 покрывают 4 сabgroups (m_lo/hi × k_i_lo/hi).

За 4-lane exchange group (fixed wid, c=l_mod4, h=l_div4>>2, varying p=l_div4&3):
  Слот kb ∈ {0, 1}, per slot 4 W-uint32 IN + 4 OUT-uint32 out
  4 sub-blocks в T покрывают: (m_lo, k_i_lo), (m_hi, k_i_lo), (m_lo, k_i_hi), (m_hi, k_i_hi)

Assert: full 4096/4096 byte match после Phase A/B/C/D.
"""

Br = 64
Bc = 64
NAT_STRIDE = Bc   # nat row-major stride = 64
T_STRIDE = Br     # T row-major stride = 64


def marker(i, j):
    return (i * 131 + j * 7 + 13) & 0xFF


def prmt_b32(a, b, sel):
    concat = (b << 32) | a
    out = 0
    for i in range(4):
        pos = (sel >> (i * 4)) & 0xF
        byte = (concat >> (pos * 8)) & 0xFF
        out |= byte << (i * 8)
    return out


def simulate():
    # SMEM 4096-byte area, initially nat layout
    smdS_area = bytearray(Br * Bc)
    for i in range(Bc):
        for j in range(Br):
            smdS_area[i * NAT_STRIDE + j] = marker(i, j)

    # Per lane: read 8 W-uint32 (4 per slot × 2 slots)
    kr_ds_per_lane = {}
    for wid in range(4):
        for lane in range(32):
            l_div4 = lane >> 2
            l_mod4 = lane & 3
            c = l_mod4
            p = l_div4 & 3
            h = l_div4 >> 2
            W_all = [0] * 8
            for slot in range(2):  # kb ∈ {0, 1}
                kb = slot
                W_base = slot * 4
                # W_0: nat[i=kb*32+c*4+p][j=wid*16+4h..+3]
                # W_1: nat[i=kb*32+c*4+p][j=wid*16+4h+8..+11]
                # W_2: nat[i=kb*32+c*4+16+p][j=wid*16+4h..+3]
                # W_3: nat[i=kb*32+c*4+16+p][j=wid*16+4h+8..+11]
                i0 = kb * 32 + c * 4 + p
                i1 = i0 + 16
                j0 = wid * 16 + 4 * h
                j1 = j0 + 8
                W_all[W_base + 0] = int.from_bytes(smdS_area[i0 * NAT_STRIDE + j0 : i0 * NAT_STRIDE + j0 + 4], 'little')
                W_all[W_base + 1] = int.from_bytes(smdS_area[i0 * NAT_STRIDE + j1 : i0 * NAT_STRIDE + j1 + 4], 'little')
                W_all[W_base + 2] = int.from_bytes(smdS_area[i1 * NAT_STRIDE + j0 : i1 * NAT_STRIDE + j0 + 4], 'little')
                W_all[W_base + 3] = int.from_bytes(smdS_area[i1 * NAT_STRIDE + j1 : i1 * NAT_STRIDE + j1 + 4], 'little')
            kr_ds_per_lane[(wid, lane)] = W_all

    # Phase transpose per lane per slot
    # G's per lane per slot
    G_per_lane_slot = {}
    for wid in range(4):
        for lane in range(32):
            for slot in range(2):
                W_all = kr_ds_per_lane[(wid, lane)]
                W0 = W_all[slot * 4 + 0]
                W1 = W_all[slot * 4 + 1]
                W2 = W_all[slot * 4 + 2]
                W3 = W_all[slot * 4 + 3]
                # Phase A: gather byte-position across W0..W3
                t01_lo = prmt_b32(W0, W1, 0x5140)
                t01_hi = prmt_b32(W0, W1, 0x7362)
                t23_lo = prmt_b32(W2, W3, 0x5140)
                t23_hi = prmt_b32(W2, W3, 0x7362)
                G0 = prmt_b32(t01_lo, t23_lo, 0x5410)
                G1 = prmt_b32(t01_lo, t23_lo, 0x7632)
                G2 = prmt_b32(t01_hi, t23_hi, 0x5410)
                G3 = prmt_b32(t01_hi, t23_hi, 0x7632)
                G_per_lane_slot[(wid, lane, slot)] = (G0, G1, G2, G3)

    # Phase B: 3 SHFL exchange within 4-lane group (fixed wid, c, h; varying p)
    V_per_lane_slot = {}
    for wid in range(4):
        for lane in range(32):
            for slot in range(2):
                c = lane & 3
                p_own = (lane >> 2) & 3
                h = (lane >> 2) >> 2
                V = list(G_per_lane_slot[(wid, lane, slot)])
                for r in range(1, 4):
                    src_p = (p_own - r) & 3
                    src_lane = c + 4 * src_p + 16 * h
                    src_idx = (src_p + r) & 3   # from src_lane's own view
                    G_at_src = G_per_lane_slot[(wid, src_lane, slot)]
                    val = G_at_src[src_idx]
                    V[src_p] = val
                V_per_lane_slot[(wid, lane, slot)] = V

    # Phase C + D: reorder V -> OUT + write to smdS_area (aliased T layout)
    smdS_area_out = bytearray(Br * Bc)
    for wid in range(4):
        for lane in range(32):
            c = lane & 3
            p_own = (lane >> 2) & 3
            h = (lane >> 2) >> 2
            for slot in range(2):
                V = V_per_lane_slot[(wid, lane, slot)]
                u01_lo = prmt_b32(V[0], V[1], 0x5140)
                u01_hi = prmt_b32(V[0], V[1], 0x7362)
                u23_lo = prmt_b32(V[2], V[3], 0x5140)
                u23_hi = prmt_b32(V[2], V[3], 0x7362)
                OUT = [
                    prmt_b32(u01_lo, u23_lo, 0x5410),
                    prmt_b32(u01_lo, u23_lo, 0x7632),
                    prmt_b32(u01_hi, u23_hi, 0x5410),
                    prmt_b32(u01_hi, u23_hi, 0x7632),
                ]
                # Phase D: write OUT_j to T positions
                kb = slot
                m_lo = wid * 16 + 4 * h + p_own
                m_hi = m_lo + 8
                k_i_lo = kb * 32 + c * 4
                k_i_hi = k_i_lo + 16
                # OUT_0 -> T[m_lo][k_i_lo..+3]
                # OUT_1 -> T[m_hi][k_i_lo..+3]
                # OUT_2 -> T[m_lo][k_i_hi..+3]
                # OUT_3 -> T[m_hi][k_i_hi..+3]
                positions = [
                    (m_lo, k_i_lo),
                    (m_hi, k_i_lo),
                    (m_lo, k_i_hi),
                    (m_hi, k_i_hi),
                ]
                for j_idx, (row, col_start) in enumerate(positions):
                    for b in range(4):
                        smdS_area_out[row * T_STRIDE + col_start + b] = (OUT[j_idx] >> (b * 8)) & 0xFF

    # Verify: T[j][i] should equal marker(i, j) (byte at nat[i][j])
    match = 0
    mism = 0
    for j in range(Br):
        for i in range(Bc):
            expected = marker(i, j)
            got = smdS_area_out[j * T_STRIDE + i]
            if expected == got:
                match += 1
            else:
                if mism < 5:
                    print(f"  MISM j={j} i={i} expect=0x{expected:02x} got=0x{got:02x}")
                mism += 1

    print(f"CPU-судья реализации: match={match}/{Br * Bc}, mismatch={mism}")
    return mism == 0


if __name__ == '__main__':
    ok = simulate()
    exit(0 if ok else 1)
