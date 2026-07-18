// 071b bridge probe: x2 reference vs x1 candidates для b0_b.
//
// Полигон: макет production smQ с свизлом `swz_byte(row, col_bytes)` дословно.
// Marker byte@(row, col) = ((row & 0x3F) << 2) | ((col >> 4) & 0x3) → 8-bit injective
//   row-domain 0..63 (6 bits), col_chunk-domain 0..7 (3 bits, but col..127 col_chunk 0..7)
//   Encoded: [row(6):col_chunk(2 bits of 3)]. col_chunk masked to 2 bits for byte fit.
//   Actually simpler: row * 4 + (col_chunk & 3) — 8-bit unique for (row 0..63, chunk 0..3).
// Wait — need FULL injectivity so we can decode row and col_chunk from delivered byte.
// Marker: (row << 3) | (col >> 4)  needs 6+3 = 9 bits. Не влезет в byte.
// Alternative: 8-bit "hash" — chunk_id = (col >> 4) & 7 → 3 bits. row & 0x1F = 5 bits.
//   marker = (row5 << 3) | chunk3 — 8 bits, injective over (row&31 × col_chunk&7).
// Coverage: rows 0..63, но row5 only distinguishes 0..31. Aliases 0↔32, 1↔33, ...
// Compromise: split into 2 probes — one with row-marker (byte=row&0xff, ambiguous над col),
//   one with col-marker (byte=col&0xff).

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include "fa_bwd_common.cuh"

#define CK(c) do{auto e=(c); if(e!=cudaSuccess){printf("CUDA err %d @%d\n",(int)e,__LINE__); return 1;}}while(0)

// Marker options — probe both to disambiguate row+col ID.
// marker_row: byte@(row,col) = row  (0..63 injective by row, col-agnostic → detects К-shift/K-inject)
// marker_col: byte@(row,col) = col  (0..127 injective by col, row-agnostic → detects N-shift/N-inject)

__global__ void bridge_probe(
    uint32_t *out_x2_lo, uint32_t *out_x2_hi,      // x2 reference: 4 uint32 per lane per (kb, np, wid)
    uint32_t *out_x1_a,  uint32_t *out_x1_b_c1,    // x1 at addr_a and candidate 1 (col +8)
    uint32_t *out_x1_b_c2,                          // x1 candidate 2 (row +16)
    int marker_mode)   // 0 = row-marker, 1 = col-marker
{
    __shared__ uint8_t smQ[8192];  // 64 rows × 128 cols

    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;

    // Writer: свизл + маркер
    constexpr int Br = 64, Hd = 128, CHUNK = 16;
    constexpr int cpr = Hd / CHUNK;
    constexpr int total = Br * cpr;
    for (int c = tid; c < total; c += 128) {
        int row = c / cpr;
        int col_byte = (c % cpr) * CHUNK;
        int dst = swz_byte(row, col_byte);
        for (int b = 0; b < CHUNK; ++b) {
            uint8_t val;
            if (marker_mode == 0) {
                val = (uint8_t)(row & 0xFF);   // row-marker
            } else {
                val = (uint8_t)((col_byte + b) & 0xFF);   // col-marker
            }
            smQ[dst + b] = val;
        }
    }
    __syncthreads();

    // === Reference x2: prod formula ===
    // We probe kb=0, np=0 case only (representative — same shape for all kb/np).
    // For each warp wid = 0..3
    int kb = 0, np = 0;
    int row_lo = kb * 32 + lane;
    int addr_lo_a = swz_byte(row_lo, np * 16);
    uint32_t sm_addr_a = __cvta_generic_to_shared(&smQ[addr_lo_a]);

    uint32_t B0a, B0b, D0, D1;
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3},[%4];\n"
        : "=r"(B0a), "=r"(B0b), "=r"(D0), "=r"(D1) : "r"(sm_addr_a));

    int slot = wid * 32 + lane;
    out_x2_lo[slot * 2 + 0] = B0a;
    out_x2_lo[slot * 2 + 1] = B0b;
    out_x2_hi[slot * 2 + 0] = D0;
    out_x2_hi[slot * 2 + 1] = D1;

    // === x1 at addr_a (should match B0a) ===
    uint32_t R0a, R1a_dup;
    asm volatile("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%0,%1},[%2];\n"
        : "=r"(R0a), "=r"(R1a_dup) : "r"(sm_addr_a));
    out_x1_a[slot] = R0a;

    // === x1 candidate 1: col shift +16 (was col+8 but LDSM.b8 requires 16-byte align → moved to +16) ===
    int addr_c1 = swz_byte(row_lo, np * 16 + 16);
    uint32_t sm_addr_c1 = __cvta_generic_to_shared(&smQ[addr_c1]);
    uint32_t R0c1, R1c1_dup;
    asm volatile("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%0,%1},[%2];\n"
        : "=r"(R0c1), "=r"(R1c1_dup) : "r"(sm_addr_c1));
    out_x1_b_c1[slot] = R0c1;

    // === x1 candidate 2: row shift +16 (NVIDIA convention: LDSM.x2 loads two matrices at [addr] and [addr + 16*row_stride]) ===
    int addr_c2 = swz_byte(row_lo + 16, np * 16);
    uint32_t sm_addr_c2 = __cvta_generic_to_shared(&smQ[addr_c2]);
    uint32_t R0c2, R1c2_dup;
    asm volatile("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%0,%1},[%2];\n"
        : "=r"(R0c2), "=r"(R1c2_dup) : "r"(sm_addr_c2));
    out_x1_b_c2[slot] = R0c2;
}

int main() {
    printf("=== 071b bridge probe: x2 reference vs x1 candidates for b0_b ===\n");
    printf("swizzle: swz_byte(row, col) = row*128 + ((col>>4 XOR row&7)<<4) + col&15\n");
    printf("candidate #1: swz_byte(row_lo, np*16 + 16)  (col shift +16, 16-aligned)\n");
    printf("candidate #2: swz_byte(row_lo + 16, np*16)  (row shift +16, NVIDIA x2 convention)\n\n");

    uint32_t *d_x2_lo, *d_x2_hi, *d_x1_a, *d_x1_b_c1, *d_x1_b_c2;
    size_t N = 128;
    CK(cudaMalloc(&d_x2_lo, N * 2 * 4));
    CK(cudaMalloc(&d_x2_hi, N * 2 * 4));
    CK(cudaMalloc(&d_x1_a, N * 4));
    CK(cudaMalloc(&d_x1_b_c1, N * 4));
    CK(cudaMalloc(&d_x1_b_c2, N * 4));

    uint32_t x2_lo[256], x2_hi[256], x1_a[128], x1_c1[128], x1_c2[128];

    for (int mode = 0; mode < 2; ++mode) {
        printf("=== MARKER MODE %d (%s) ===\n", mode, mode == 0 ? "row-marker" : "col-marker");
        bridge_probe<<<1, 128>>>(d_x2_lo, d_x2_hi, d_x1_a, d_x1_b_c1, d_x1_b_c2, mode);
        auto e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("kernel err: %s\n", cudaGetErrorString(e)); return 1; }

        CK(cudaMemcpy(x2_lo, d_x2_lo, sizeof(x2_lo), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(x2_hi, d_x2_hi, sizeof(x2_hi), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(x1_a, d_x1_a, sizeof(x1_a), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(x1_c1, d_x1_b_c1, sizeof(x1_c1), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(x1_c2, d_x1_b_c2, sizeof(x1_c2), cudaMemcpyDeviceToHost));

        // Coverage: for each of 4 warps × 32 lanes = 128 slots.
        // Check: x1_a == x2's R0(B0a)? x1_c1 == x2's R1(B0b)? x1_c2 == x2's R1(B0b)?
        int match_a = 0, match_c1 = 0, match_c2 = 0;
        int cover = 0;
        for (int slot = 0; slot < 128; ++slot) {
            uint32_t x2_R0 = x2_lo[slot * 2 + 0];   // B0a
            uint32_t x2_R1 = x2_lo[slot * 2 + 1];   // B0b
            cover++;
            if (x1_a[slot] == x2_R0) match_a++;
            if (x1_c1[slot] == x2_R1) match_c1++;
            if (x1_c2[slot] == x2_R1) match_c2++;
        }
        printf("  slots covered: %d/128\n", cover);
        printf("  x1_a  == x2.R0 (B0a): %d/%d  (%.1f%%)\n", match_a, cover, 100.0 * match_a / cover);
        printf("  x1_c1 == x2.R1 (B0b) [col+8]:   %d/%d  (%.1f%%)\n", match_c1, cover, 100.0 * match_c1 / cover);
        printf("  x1_c2 == x2.R1 (B0b) [row+16]:  %d/%d  (%.1f%%)\n", match_c2, cover, 100.0 * match_c2 / cover);

        // Также дампим 4 lanes из каждого warp для sanity
        printf("  wid=0 lanes 0..3 (row-marker mode):\n");
        for (int lane = 0; lane < 4; ++lane) {
            int slot = 0 * 32 + lane;
            printf("    l%d: x2.R0=%08x x2.R1=%08x x2.R2=%08x x2.R3=%08x  x1_a=%08x x1_c1=%08x x1_c2=%08x\n",
                   lane, x2_lo[slot*2+0], x2_lo[slot*2+1], x2_hi[slot*2+0], x2_hi[slot*2+1],
                   x1_a[slot], x1_c1[slot], x1_c2[slot]);
        }
        printf("\n");
    }

    cudaFree(d_x2_lo); cudaFree(d_x2_hi); cudaFree(d_x1_a); cudaFree(d_x1_b_c1); cudaFree(d_x1_b_c2);
    return 0;
}
