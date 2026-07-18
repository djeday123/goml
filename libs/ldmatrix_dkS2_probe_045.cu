// 045 II.5 микропроба: LDSM.x2.trans.b8 на макете smQ row-major (128B stride).
// Проверяет row_ptr формулу для чтения натурального Q под B-op mma.m16n8k32.e4m3.
//   B-op ожидание per lane l (groupID=l/4, laneID=l%4):
//     b0 = 4 fp8 at (k = 4*laneID..4*laneID+3, n = groupID)
//     b1 = 4 fp8 at (k = 4*laneID+16..4*laneID+19, n = groupID)
//   x2 доставляет 4 uint32 = 16 fp8 halves per lane
//   Row-ptr formula (per lane):
//     tile_id = l / 8, row_in_tile = l % 8
//     k_row = kb*32 + row_in_tile + ((tile_id & 1) ? 8 : 0)   // 0..15 в 2 tiles
//     n_col = ni_pair * 8  (fixed n for tile)
//     hmm — уточним по факту доставки

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA err: %s\n", cudaGetErrorString(e)); exit(1); }} while (0)

__global__ void probe_dkS2(uint32_t *out) {
    // smQ mimic: 64 rows × 128 cols fp8 = 8192 bytes
    __shared__ uint8_t smQ[64 * 128];
    int tid = threadIdx.x;
    // Setup marker: smQ[row][col] = (row << 4) | (col & 0xF) — high nibble row, low nibble col-low
    if (tid < 128) {
        for (int j = 0; j < 64; ++j) {
            int idx = tid + j * 128;  // (row=idx/128, col=idx%128)
            int r = idx / 128, c = idx % 128;
            smQ[idx] = (uint8_t)((r << 4) | (c & 0xF));
        }
    }
    __syncthreads();

    int lane = tid & 31;
    // Row-ptr formula (v3): 2 tiles side-by-side in N. All 32 lanes provide rows 0..31 across n=0..15 range.
    // Actually LDSM.x2 tiles могут быть 16x16 halves каждый — total 32x16 rows or 16x32 cols.
    // Try: all 32 lanes at k_row=lane%32 (0..31), n_col=(lane<16)?0:16 (16B aligned per tile)
    int k_row = lane;                               // 0..31 (все 32 rows нужны для 2 tiles по 16 rows)
    int n_col = 0;                                  // все на одной 16B-aligned column base
    uint32_t sm_addr = __cvta_generic_to_shared(&smQ[k_row * 128 + n_col]);
    uint32_t r0, r1, r2, r3;
    asm volatile(
        "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3},[%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(sm_addr)
    );
    if (tid < 32) { out[tid*4+0]=r0; out[tid*4+1]=r1; out[tid*4+2]=r2; out[tid*4+3]=r3; }
}

static void decode_and_check(uint32_t *h) {
    printf("Layout per lane (bytes decoded as row.col):\n");
    int ok_bop = 0;  // count halves matching B-op b0/b1 shape
    for (int l = 0; l < 32; ++l) {
        int laneID = l & 3, groupID = l >> 2;
        printf("l%02d(g=%d,L=%d):", l, groupID, laneID);
        for (int r = 0; r < 4; ++r) {
            uint32_t v = h[l * 4 + r];
            printf(" R%d=", r);
            for (int b = 0; b < 4; ++b) {
                uint8_t byte = (v >> (b * 8)) & 0xFF;
                int row = byte >> 4, col_lo = byte & 0xF;
                printf("(%d.%X)", row, col_lo);
            }
        }
        printf("\n");
    }
    printf("\n(row.col decoded from byte = (row<<4)|col_lo, expected 4 k-adjacent rows at fixed n per uint32)\n");
}

int main() {
    uint32_t *d_out; uint32_t h_out[128];
    CK(cudaMalloc(&d_out, 128 * sizeof(uint32_t)));
    CK(cudaMemset(d_out, 0xFF, 128 * sizeof(uint32_t)));
    probe_dkS2<<<1, 128>>>(d_out);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel fail: %s\n", cudaGetErrorString(err));
        return 1;
    }
    CK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    printf("045 II.5 probe LDSM.x2.trans.b8 on smQ 64×128 row-major:\n\n");
    decode_and_check(h_out);
    cudaFree(d_out);
    return 0;
}
