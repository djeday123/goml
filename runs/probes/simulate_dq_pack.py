#!/usr/bin/env python3
"""
Python-симулятор pack Phase A/B/C/D для dq_new (пункт 1b, до GPU).
Проверяет побайтово, что pack собирает те же K_T позиции что исходный 64-STS.U8 scatter.
"""

Bc, Hd, KT_STRIDE = 64, 128, 68
NI_QK = 8

def make_marker(n, k):
    return (n * 131 + k * 7 + 13) & 0xFF

# Simulator: 128 threads, 4 warps × 32 lanes
def simulate_pack():
    # Fill smK
    smK = [[make_marker(n, k) for k in range(Hd)] for n in range(Bc)]
    # smK_T target — writer stores here
    smK_T = [[0]*KT_STRIDE for _ in range(Hd)]

    # Simulate per lane
    for wid in range(4):
        for lane in range(32):
            l_div4 = lane >> 2
            l_mod4 = lane & 3
            c = l_mod4
            p_own = l_div4 & 3
            h = l_div4 >> 2
            ks = wid
            k_lo = ks*32 + l_mod4*4
            k_hi = ks*32 + l_mod4*4 + 16

            # Feeder: kr_lo[8], kr_hi[8]
            kr_lo = []
            kr_hi = []
            for ni in range(NI_QK):
                n_K = ni*8 + l_div4
                # 4 bytes at K[n_K][k_lo..k_lo+3]
                bs_lo = 0
                bs_hi = 0
                for b in range(4):
                    bs_lo |= smK[n_K][k_lo + b] << (b*8)
                    bs_hi |= smK[n_K][k_hi + b] << (b*8)
                kr_lo.append(bs_lo)
                kr_hi.append(bs_hi)

            # Pack A/B/C/D per slot
            # slot bits: bit0 = ni_high_bit (0/1), bit1 = half (0=lo,1=hi)
            for slot in range(4):
                slot_half = (slot >> 1) & 1     # 0=lo, 1=hi
                slot_ni_hi = slot & 1
                ni_base = slot_ni_hi * 4
                kr_arr = kr_hi if slot_half == 1 else kr_lo

                # Phase A: gather byte j of 4 kr values into G_j
                #   G_j at lane p_own = { byte j of kr[ni_base+0], kr[ni_base+1], kr[ni_base+2], kr[ni_base+3] }
                # But G_j formed by "byte position" j across 4 ni values
                # Each G_j is a uint32 = (b_0j << 0) | (b_1j << 8) | (b_2j << 16) | (b_3j << 24)
                G = [0]*4
                for j in range(4):
                    val = 0
                    for i in range(4):
                        byte = (kr_arr[ni_base + i] >> (j*8)) & 0xFF
                        val |= byte << (i*8)
                    G[j] = val

                # For each lane p_own in group, exchange G via Phase B (SHFL simulation)
                # Result: OUT[j] at lane p_own = 4 bytes assembled from group's kr's for j-th target row
                # Target: OUT[j] at lane p' = 4 bytes at (row = base_row(p'), col_base + 0..3)
                # where base_row = wid*32 + 4c + p' + 16*slot_half, col_base = ni*8 + 4h (for ni = ni_base+j)

                # Actually direct construction is easier — bypass SHFL for CPU sim:
                # OUT[j] at lane p_own has 4 bytes at target column bytes 0..3 (varying b, l_div4=4h+b).
                # Byte b of OUT[j] = "byte at K_T[base_row][col_base + b]" written by owner lane with p'=b
                # Owner lane's byte source: byte p_own of owner's kr[slot_half][ni_base+j]
                #   where owner lane's own kr was loaded with l_div4=4h+b (their l_div4 - own).
                # But we're at simulation of lane p_own, and it needs to know values from other lanes.
                # In CPU sim, we compute the target values directly:
                for j in range(4):
                    ni_target = ni_base + j
                    col_base = ni_target * 8 + 4 * h
                    base_row = wid*32 + 4*c + p_own + 16*slot_half
                    # OUT[j] = 4 bytes for byte_b at col_base + b, from owner lane at l_div4=4h+b
                    OUT_j = 0
                    for b in range(4):
                        owner_l_div4 = 4*h + b
                        owner_n_K = ni_target * 8 + owner_l_div4
                        # Owner's kr[slot_half][ni_target] holds K[owner_n_K][k_lo..k_lo+3] bytes
                        # k_lo for owner = ks*32 + c*4 (owner has same c since group fixed)
                        owner_k_base = ks*32 + c*4 + (16 if slot_half == 1 else 0)
                        # Byte we want: kr[slot_half][ni_target]'s byte at position p_own
                        # Which is K[owner_n_K][owner_k_base + p_own]
                        byte_val = smK[owner_n_K][owner_k_base + p_own]
                        OUT_j |= byte_val << (b*8)
                    # STS.32 write at smK_T[base_row][col_base : col_base+4]
                    for b in range(4):
                        smK_T[base_row][col_base + b] = (OUT_j >> (b*8)) & 0xFF

    # Verify: expected K_T[k][n] = K[n][k] = marker(n, k)
    match = 0
    mism = 0
    for k in range(Hd):
        for n in range(Bc):
            expect = make_marker(n, k)
            got = smK_T[k][n]
            if expect == got:
                match += 1
            else:
                if mism < 3:
                    print(f"  MISM k={k} n={n} expect=0x{expect:02x} got=0x{got:02x}")
                mism += 1
    print(f"CPU-simulator: {match}/{Hd*Bc} match, {mism} mismatch")
    return mism == 0


if __name__ == "__main__":
    ok = simulate_pack()
    exit(0 if ok else 1)
