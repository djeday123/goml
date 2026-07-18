// 039 ISA-микропроба: ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 на sm_120a.
//   Anti-DCE: результаты пишутся в глобаль.
//   §2.a: компилируется? исполняется?
//   §2.b: раскладка == B-op-ожидание mma.m16n8k16 (2 adj k-halves at same n)?
//   §2.c: тот же тест на СВИЗЛОВАННОМ макете (production XOR-паттерн smdO).
//
// Раскладка ldmatrix.x4.trans.b16 (документация Ampere/Blackwell):
//   Input SMEM layout: 4 tiles × 8×8 halves.
//   After trans: каждый tile транспонирован (row/col swap внутри 8×8).
//   Per lane l ∈ [0, 32): 4 uint32 = 8 halves.
//     R0 = halves(row=2*(l%4),   col=l/4)      | halves(row=2*(l%4)+1, col=l/4)     [tile 0 transposed]
//     R1 = halves(row=2*(l%4)+8, col=l/4)      | halves(row=2*(l%4)+9, col=l/4)     [tile 1 transposed]
//     R2 = halves(row=2*(l%4),   col=l/4+8)    | halves(row=2*(l%4)+1, col=l/4+8)   [tile 2 transposed]
//     R3 = halves(row=2*(l%4)+8, col=l/4+8)    | halves(row=2*(l%4)+9, col=l/4+8)   [tile 3 transposed]
//   Т.е. каждый uint32 пакует 2 halves с ADJACENT row-index (k-adjacent) при фиксированном col (n) -
//   ТОЧНО раскладка B-op mma.m16n8k16.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA err: %s\n", cudaGetErrorString(e)); exit(1); }} while (0)

// --- Проба 2.a-b: flat (без свизла) mini-array 16 rows × 16 cols halves ---
__global__ void probe_trans_flat(uint32_t *out /* [128 uint32] */) {
    __shared__ __half smem[16 * 16];  // 16 rows × 16 cols = 256 halves = 512 bytes

    int tid = threadIdx.x;
    // Setup маркер: smem[row][col] = (row << 8) | col
    if (tid < 128) {
        int idx0 = tid * 2 + 0;
        int idx1 = tid * 2 + 1;
        int row0 = idx0 / 16, col0 = idx0 % 16;
        int row1 = idx1 / 16, col1 = idx1 % 16;
        smem[idx0] = __short_as_half((short)((row0 << 8) | col0));
        smem[idx1] = __short_as_half((short)((row1 << 8) | col1));
    }
    __syncthreads();

    int lane = tid & 31;
    // Row-ptrs: 32 lanes each provide 1 row start; 4 tiles × 8 rows = 32 rows.
    // Lane assignment (standard ldmatrix layout):
    //   Tile 0 (rows 0..7,  cols 0..7 ):  lanes  0..7
    //   Tile 1 (rows 8..15, cols 0..7 ):  lanes  8..15
    //   Tile 2 (rows 0..7,  cols 8..15):  lanes 16..23
    //   Tile 3 (rows 8..15, cols 8..15):  lanes 24..31
    int tile = lane / 8;
    int rowid = lane % 8;
    int row = rowid + ((tile & 1) ? 8 : 0);
    int col_start = (tile & 2) ? 8 : 0;

    uint32_t sm_addr = __cvta_generic_to_shared(&smem[row * 16 + col_start]);

    uint32_t r0, r1, r2, r3;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(sm_addr)
    );

    // Anti-DCE
    if (tid < 32) {
        out[tid * 4 + 0] = r0;
        out[tid * 4 + 1] = r1;
        out[tid * 4 + 2] = r2;
        out[tid * 4 + 3] = r3;
    }
}

// --- Проба 2.c: свизлованный макет (production XOR-паттерн smdO) ---
// SMEM layout: 16 rows × 16 halves (не 128 как в production, сжато для теста).
// XOR = (row & 7) << ??  Так как в production XOR = (i_local & 7) << 4 применяется на byte offset,
// а row_stride в production = 256 bytes, здесь row_stride = 32 bytes (16 halves).
// Пропорционально: XOR = (row & 7) << 1 в element-space = (row & 7) << 2 в byte-space (bits {2,3,4}).
// Но для demonstration оставим XOR = (row & 7) << 3 element = (row & 7) << 4 byte, что не влезет
// в 16 cols (max half offset = 15, XOR up to 56 element = out of range).
// Значит для теста используем компактный layout: 16 rows × 16 cols, XOR = 0 (flat).
// Настоящая проверка свизла - через CPU-судью §5 (полный паттерн против production).

__global__ void probe_trans_swizzled(uint32_t *out /* [128 uint32] */, int *sm_used /* smem trace */) {
    __shared__ __half smem[16 * 128];  // 16 rows × 128 halves (mini production row size!)
    // Halves per row = 128 (matching production Hd)
    // Row stride в bytes = 256

    int tid = threadIdx.x;

    // Setup маркер по production XOR:
    //   byte_addr(row, col_byte) = row * 256 + (col_byte ^ ((row & 7) << 4))
    //   halves per row = 128, col_byte ∈ {0, 16, 32, ...} for chunks
    // Каждый thread пишет несколько halves marker.
    if (tid < 128) {
        // 16 rows × 128 halves = 2048 halves total, 128 threads → 16 halves per thread
        for (int k = 0; k < 16; ++k) {
            int idx = tid * 16 + k;
            int row = idx / 128;
            int col = idx % 128;  // logical col (halves index)
            int col_byte_logical = col * 2;  // logical byte offset
            // Physical byte offset with XOR (production writer)
            int xor_byte = (row & 7) << 4;
            int col_byte_physical = col_byte_logical ^ xor_byte;
            int phys_addr_half = row * 128 + col_byte_physical / 2;
            smem[phys_addr_half] = __short_as_half((short)((row << 8) | col));
        }
    }
    __syncthreads();

    int lane = tid & 31;
    // Read via ldmatrix.trans, provide 32 row-ptrs.
    // For B-op MMA emulation, we want to read k=0..15, n=0..7 (128 halves = 1 B-frag),
    // via 2 tiles × 8×8 (or 1 x4 if 4 tiles).
    // For x4.trans probe: rows 0..15, cols 0..15 (2 tiles of k × 2 tiles of n).
    // Row-ptr assignment:
    //   Tile 0 (k=0..7, n=0..7):     lanes 0..7,   row-ptr = smdO[k=lane%8][n=0]
    //   Tile 1 (k=8..15, n=0..7):    lanes 8..15,  row-ptr = smdO[k=lane%8+8][n=0]
    //   Tile 2 (k=0..7, n=8..15):    lanes 16..23, row-ptr = smdO[k=lane%8][n=8]
    //   Tile 3 (k=8..15, n=8..15):   lanes 24..31, row-ptr = smdO[k=lane%8+8][n=8]
    int rowid = lane % 8;
    int tile = lane / 8;
    int k_row = rowid + ((tile & 1) ? 8 : 0);
    int n_start_logical = (tile & 2) ? 8 : 0;
    // Physical address (post-XOR): byte = k_row * 256 + n_start*2 ^ ((k_row & 7) << 4)
    int n_byte_logical = n_start_logical * 2;  // 0 or 16
    int n_byte_physical = n_byte_logical ^ ((k_row & 7) << 4);
    int phys_addr_half = k_row * 128 + n_byte_physical / 2;

    uint32_t sm_addr = __cvta_generic_to_shared(&smem[phys_addr_half]);
    if (tid < 32) sm_used[tid] = phys_addr_half;

    uint32_t r0, r1, r2, r3;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(sm_addr)
    );

    if (tid < 32) {
        out[tid * 4 + 0] = r0;
        out[tid * 4 + 1] = r1;
        out[tid * 4 + 2] = r2;
        out[tid * 4 + 3] = r3;
    }
}

static void check_result(const char *label, uint32_t *h_out, bool expect_bop) {
    int ok = 0;
    printf("--- %s ---\n", label);
    for (int lane = 0; lane < 32; ++lane) {
        printf("lane=%2d:", lane);
        for (int r = 0; r < 4; ++r) {
            uint32_t v = h_out[lane * 4 + r];
            uint16_t lo = v & 0xFFFF;
            uint16_t hi = (v >> 16) & 0xFFFF;
            int row_lo = (lo >> 8) & 0xFF, col_lo = lo & 0xFF;
            int row_hi = (hi >> 8) & 0xFF, col_hi = hi & 0xFF;
            printf(" R%d=[(%d,%d)|(%d,%d)]", r, row_lo, col_lo, row_hi, col_hi);
            // B-op ожидание: (row_hi - row_lo == 1) && (col_lo == col_hi) — 2 adjacent k, same n
            bool bop = (row_hi == row_lo + 1) && (col_lo == col_hi);
            if (expect_bop && bop && row_lo < 16 && col_lo < 16) ok++;
        }
        printf("\n");
    }
    printf("Ok_count (B-op match): %d / 128\n", ok);
    printf("Criterion 100%%: %s\n\n", ok == 128 ? "YES" : "NO");
}

int main() {
    uint32_t *d_out;
    int *d_sm_used;
    uint32_t h_out[128];
    int h_sm_used[32];
    CK(cudaMalloc(&d_out, 128 * sizeof(uint32_t)));
    CK(cudaMalloc(&d_sm_used, 32 * sizeof(int)));

    // Проба 2.a-b: flat mini-array
    printf("===== Проба 2.a-b: flat mini-array (16×16 halves) =====\n");
    CK(cudaMemset(d_out, 0xFF, 128 * sizeof(uint32_t)));
    probe_trans_flat<<<1, 128>>>(d_out);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Flat kernel launch failed: %s\n", cudaGetErrorString(err));
        printf("VERDICT: ldmatrix.x4.trans.b16 on sm_120a — RUNTIME_ERROR (flat)\n");
        return 1;
    }
    printf("VERDICT: ldmatrix.x4.trans.b16 on sm_120a — COMPILED_AND_RAN (flat)\n");
    CK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    check_result("Flat B-op check", h_out, true);

    // Проба 2.c: свизлованный макет
    printf("===== Проба 2.c: свизлованный макет (production XOR-паттерн) =====\n");
    CK(cudaMemset(d_out, 0xFF, 128 * sizeof(uint32_t)));
    probe_trans_swizzled<<<1, 128>>>(d_out, d_sm_used);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Swizzled kernel launch failed: %s\n", cudaGetErrorString(err));
        printf("VERDICT: swizzled probe — RUNTIME_ERROR\n");
        return 2;
    }
    CK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_sm_used, d_sm_used, sizeof(h_sm_used), cudaMemcpyDeviceToHost));
    printf("Row-ptrs per lane (phys half addr):\n");
    for (int l = 0; l < 32; ++l) {
        if (l % 8 == 0) printf("  ");
        printf("[%2d]=%4d ", l, h_sm_used[l]);
        if ((l + 1) % 8 == 0) printf("\n");
    }
    check_result("Swizzled B-op check", h_out, true);

    cudaFree(d_out);
    cudaFree(d_sm_used);
    return 0;
}
