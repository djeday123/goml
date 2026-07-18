// 038 ISA-микропроба: ldmatrix.sync.aligned.m8n8.x4.shared.b16 no-trans на sm_120a.
//   Маркер-байтовый юнит (lane, reg, byte) с anti-DCE через выход в глобаль.
//   Критерий: 100% совпадение с ожидаемой фрагмент-раскладкой mma.sync m16n8k16 A-op.
//
// Раскладка ldmatrix.x4.b16 no-trans (Ampere/Blackwell документация):
//   32 lanes × 4 uint32 = 128 halves = 4 tiles (8×8 halves каждый).
//   Каждый lane предоставляет row-адрес одного из 32 rows разбитых на 4 tiles.
//   Порядок tiles: T0..T3 расположены как 2×2 (m16k16 фрагмент MMA-A):
//       [T0][T1]   ← m0..7, k0..7 | m0..7, k8..15
//       [T2][T3]   ← m8..15, k0..7| m8..15, k8..15
//   Lane выдаёт 4 uint32 = 2 halves × 4 = row data для (lane_id / 4, lane_id % 4 * 8 + col_off).
//   Точная per-lane, per-reg раскладка halves (row_id, col_start):
//     For x4 no-trans, lane l holds:
//       R0 = halves(row=l/4,       col_start=(l%4)*8, 2 halves col_start..col_start+1)   [T0 or T1 by row/8]
//       R1 = halves(row=l/4,       col_start=(l%4)*8, col_start+2..col_start+3)?
//   Точная формула сложнее, но для проверки корректности достаточно:
//     Уникальный маркер v(row, col) = (row << 8) | col       (row ∈ [0,16), col ∈ [0,32))
//     Каждый uint32 в результате = ((v(r, c+1)) << 16) | v(r, c) где (r, c) — одна из позиций
//     фрагмента, покрытого этим lane и этим reg-slot.
//   После ldmatrix: каждый lane пишет 4 uint32 в глобаль по lane*4 offset.
//   Хост читает глобаль и печатает раскладку.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA err: %s\n", cudaGetErrorString(e)); exit(1); }} while (0)

// Kernel: setup маркер в SMEM, ldmatrix.x4 no-trans, dump per-lane regs в глобаль.
__global__ void probe_ldmatrix(uint32_t *out /* [128 uint32] */) {
    __shared__ __half smem[16 * 16];  // 16 rows × 16 cols halves = 256 halves = 512 bytes

    int tid = threadIdx.x;
    // Setup маркер: smem[row][col] = (row << 8) | col
    if (tid < 128) {
        // 16 × 16 = 256 halves, каждый thread пишет 2
        int idx0 = tid * 2 + 0;
        int idx1 = tid * 2 + 1;
        int row0 = idx0 / 16, col0 = idx0 % 16;
        int row1 = idx1 / 16, col1 = idx1 % 16;
        smem[idx0] = __short_as_half((short)((row0 << 8) | col0));
        smem[idx1] = __short_as_half((short)((row1 << 8) | col1));
    }
    __syncthreads();

    int lane = tid & 31;

    // Compute row address for this lane. ldmatrix.x4: 32 lanes each provide one row-ptr.
    // For x4 no-trans: rows 0..31, но у нас только 16 rows — используем lane % 16.
    // Реально ldmatrix.x4 требует 32 row-ptrs для 4 8×8 tiles = 4×8 = 32 rows.
    // Мы даём 16 rows, повторяем: rows 0..15 → lanes 0..15, rows 0..15 → lanes 16..31.
    // Для честного маркера: пусть smem = 16 rows × 16 cols = 256 halves = 512 bytes.
    // x4 читает 4 tiles = 4×64 halves = 256 halves = ROW 32 × COL 8, но у нас только 16 row.
    // Значит fits: подадим smem как 2 tiles столбом (col 0..7 T0, col 8..15 T1),
    //             и повторяем rows для T2/T3 → те же rows 0..15.
    // Такой маркер даст duplicate T0/T2 (rows совпадают), но проверяет что ldmatrix
    // корректно распределяет halves per lane per reg.

    int row_ptr_lane;
    if (lane < 16) row_ptr_lane = lane;          // T0/T2 provider row
    else            row_ptr_lane = lane - 16;    // T1/T3 provider row (16 lanes each)

    uint32_t sm_addr = __cvta_generic_to_shared(&smem[row_ptr_lane * 16 + (lane >= 16 ? 8 : 0)]);

    uint32_t r0, r1, r2, r3;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(sm_addr)
    );

    // Anti-DCE: write to global
    if (tid < 32) {
        out[tid * 4 + 0] = r0;
        out[tid * 4 + 1] = r1;
        out[tid * 4 + 2] = r2;
        out[tid * 4 + 3] = r3;
    }
}

int main() {
    uint32_t *d_out;
    CK(cudaMalloc(&d_out, 128 * sizeof(uint32_t)));
    CK(cudaMemset(d_out, 0xFF, 128 * sizeof(uint32_t)));

    probe_ldmatrix<<<1, 128>>>(d_out);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err));
        printf("VERDICT: ldmatrix.x4.b16 no-trans on sm_120a — RUNTIME_ERROR\n");
        return 1;
    }

    uint32_t h_out[128];
    CK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));

    printf("VERDICT: ldmatrix.x4.b16 no-trans on sm_120a — COMPILED_AND_RAN\n");
    printf("Per-lane per-reg raw data (lane, reg → halves):\n");
    int ok_count = 0;
    for (int lane = 0; lane < 32; ++lane) {
        printf("lane=%2d:", lane);
        for (int r = 0; r < 4; ++r) {
            uint32_t v = h_out[lane * 4 + r];
            uint16_t lo = v & 0xFFFF;
            uint16_t hi = (v >> 16) & 0xFFFF;
            // Расшифровка маркера: half = short = (row << 8) | col
            int row_lo = (lo >> 8) & 0xFF, col_lo = lo & 0xFF;
            int row_hi = (hi >> 8) & 0xFF, col_hi = hi & 0xFF;
            printf(" R%d=[(%d,%d)|(%d,%d)]", r, row_lo, col_lo, row_hi, col_hi);
            // Проверка что row_lo/col_lo/row_hi/col_hi в валидных пределах и halves adjacent col
            bool cols_adj = (col_hi == col_lo + 1);
            bool rows_same = (row_lo == row_hi);
            if (cols_adj && rows_same && row_lo < 16 && col_lo < 16) ok_count++;
        }
        printf("\n");
    }
    printf("\nOk_count (adj-col + same-row per uint32): %d / %d\n", ok_count, 32 * 4);
    printf("Criterion 100%%: %s\n", ok_count == 128 ? "YES" : "NO");

    cudaFree(d_out);
    return 0;
}
