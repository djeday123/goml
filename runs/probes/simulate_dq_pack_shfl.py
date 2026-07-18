#!/usr/bin/env python3
"""
Симулятор PACK Phase A/B/C/D для dq_new с явными SHFL операциями.
Идентично dk_new pack (fa_bwd_dk_new.cu:154-217), адаптировано под dq feeder.
"""

Bc, Hd, KT_STRIDE = 64, 128, 68
NI_QK = 8

def make_marker(n, k):
    return (n * 131 + k * 7 + 13) & 0xFF

def prmt_b32(a, b, sel):
    """Emulate CUDA PRMT: sel is 4 hex nibbles, each selects a byte from concat(a,b)."""
    concat = (b << 32) | a
    out = 0
    for i in range(4):
        pos = (sel >> (i*4)) & 0xF
        byte = (concat >> (pos*8)) & 0xFF
        out |= byte << (i*8)
    return out

def simulate():
    smK = [[make_marker(n, k) for k in range(Hd)] for n in range(Bc)]
    smK_T = [[0]*KT_STRIDE for _ in range(Hd)]

    # Warp simulation
    for wid in range(4):
        # Load feeder per each lane
        kr_lo_per_lane = [[] for _ in range(32)]
        kr_hi_per_lane = [[] for _ in range(32)]
        for lane in range(32):
            l_div4 = lane >> 2
            l_mod4 = lane & 3
            ks = wid
            k_lo = ks*32 + l_mod4*4
            k_hi = ks*32 + l_mod4*4 + 16
            for ni in range(NI_QK):
                n_K = ni*8 + l_div4
                bs_lo = sum(smK[n_K][k_lo + b] << (b*8) for b in range(4))
                bs_hi = sum(smK[n_K][k_hi + b] << (b*8) for b in range(4))
                kr_lo_per_lane[lane].append(bs_lo)
                kr_hi_per_lane[lane].append(bs_hi)

        # For each slot, Phase A/B/C/D
        for slot in range(4):
            slot_half = (slot >> 1) & 1
            slot_ni_hi = slot & 1
            ni_base = slot_ni_hi * 4

            # Compute G[0..3] per lane (Phase A: byte-position transpose across 4 ni's)
            G_per_lane = [[0]*4 for _ in range(32)]
            for lane in range(32):
                kr_arr = kr_hi_per_lane[lane] if slot_half == 1 else kr_lo_per_lane[lane]
                # W0..W3 = 4 uint32s at ni_base+0..3
                W = [kr_arr[ni_base + i] for i in range(4)]
                # Phase A dk-style: 8 PRMT
                t01_lo = prmt_b32(W[0], W[1], 0x5140)
                t01_hi = prmt_b32(W[0], W[1], 0x7362)
                t23_lo = prmt_b32(W[2], W[3], 0x5140)
                t23_hi = prmt_b32(W[2], W[3], 0x7362)
                G_per_lane[lane][0] = prmt_b32(t01_lo, t23_lo, 0x5410)
                G_per_lane[lane][1] = prmt_b32(t01_lo, t23_lo, 0x7632)
                G_per_lane[lane][2] = prmt_b32(t01_hi, t23_hi, 0x5410)
                G_per_lane[lane][3] = prmt_b32(t01_hi, t23_hi, 0x7632)

            # Phase B: SHFL exchange within 4-lane group (fixed c, h; varying p ∈ [0..3])
            V_per_lane = [[0]*4 for _ in range(32)]
            for lane in range(32):
                c = lane & 3
                p_own = (lane >> 2) & 3
                h = (lane >> 2) >> 2
                # Init V from own G
                V = list(G_per_lane[lane])
                for r in range(1, 4):
                    src_p = (p_own - r) & 3
                    src_lane = c + 4 * src_p + 16 * h
                    # SHFL: current lane receives value from src_lane.
                    # At src_lane: its own p = src_p; its own idx = (src_p + r) & 3.
                    # So val we receive = G[(src_p + r) & 3] at src_lane.
                    src_own_idx = (src_p + r) & 3
                    val = G_per_lane[src_lane][src_own_idx]
                    # Write into V[src_p]
                    V[src_p] = val
                V_per_lane[lane] = V

            # Phase C: reorder V → OUT (8 PRMT symmetric to Phase A)
            for lane in range(32):
                V = V_per_lane[lane]
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
                # Phase D — write STS.32 to K_T
                c = lane & 3
                p_own = (lane >> 2) & 3
                h = (lane >> 2) >> 2
                ks = wid
                base_row = wid*32 + 4*c + p_own + 16*slot_half
                for j in range(4):
                    col_base = (ni_base + j)*8 + 4*h
                    for b in range(4):
                        smK_T[base_row][col_base + b] = (OUT[j] >> (b*8)) & 0xFF

    # Verify
    match, mism = 0, 0
    for k in range(Hd):
        for n in range(Bc):
            expect = make_marker(n, k)
            got = smK_T[k][n]
            if expect == got:
                match += 1
            else:
                if mism < 5:
                    print(f"  MISM k={k} n={n} expect=0x{expect:02x} got=0x{got:02x}")
                mism += 1
    print(f"Phase A/B/C/D sim: {match}/{Hd*Bc} match, {mism} mismatch")
    return mism == 0


if __name__ == "__main__":
    ok = simulate()
    exit(0 if ok else 1)
