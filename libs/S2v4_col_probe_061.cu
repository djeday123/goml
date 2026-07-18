// 061 C0: col-проход моста. Маркер = col-позиция в ряду (7 бит, 0..127).
//   Injective по col: 128 unique values ≤ 256 byte states, no collisions by construction.
//   Row constant (const_row=0..N) для проверки порядка байт внутри R0/R1 vs MMA-B fragment expectation.
//
// Domain (per LDSM.x2 output): 32 lanes × 4 regs × 4 bytes = 512 bytes = 128 unique col samples * 4 duplicates.
// Full coverage: 2 kb × 8 np × 32 lanes × 2 lo/hi × 4 regs × 4 bytes = 32768 sample-checks.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CHK(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; }} while(0)

__device__ __host__ inline int swz_byte(int row, int col_bytes) {
    int chunk = col_bytes >> 4;
    int within = col_bytes & 15;
    return row * 128 + ((chunk ^ (row & 7)) << 4) + within;
}

__global__ void probe_kernel(const uint8_t *__restrict__ Q_g, uint32_t *__restrict__ dump) {
    __shared__ uint8_t smQ[64 * 128];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;

    constexpr int CHUNK = 16;
    constexpr int total = 8192 / CHUNK;
    #pragma unroll 4
    for (int c = tid; c < total; c += 128) {
        int row = c / 8;
        int col_byte = (c % 8) * CHUNK;
        int dst_off = swz_byte(row, col_byte);
        int src_off = row * 128 + col_byte;
        uint32_t *dst = (uint32_t*)&smQ[dst_off];
        const uint32_t *src = (const uint32_t*)&Q_g[src_off];
        dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
    }
    __syncthreads();

    if (wid != 0) return;

    #pragma unroll
    for (int kb = 0; kb < 2; ++kb) {
        #pragma unroll
        for (int np = 0; np < 8; ++np) {
            int row_lo = kb * 32 + lane;
            int addr_lo = swz_byte(row_lo, np * 16);
            uint32_t sm_addr_lo = __cvta_generic_to_shared(&smQ[addr_lo]);
            uint32_t R0_lo, R1_lo, R2_lo, R3_lo;
            asm volatile(
                "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(R0_lo), "=r"(R1_lo), "=r"(R2_lo), "=r"(R3_lo) : "r"(sm_addr_lo));

            int row_hi = kb * 32 + (lane & 15) + 16;
            int addr_hi = swz_byte(row_hi, np * 16);
            uint32_t sm_addr_hi = __cvta_generic_to_shared(&smQ[addr_hi]);
            uint32_t R0_hi, R1_hi, R2_hi, R3_hi;
            asm volatile(
                "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(R0_hi), "=r"(R1_hi), "=r"(R2_hi), "=r"(R3_hi) : "r"(sm_addr_hi));

            int base = (kb * 8 + np) * 32 * 8 + lane * 8;
            dump[base + 0] = R0_lo;
            dump[base + 1] = R1_lo;
            dump[base + 2] = R2_lo;
            dump[base + 3] = R3_lo;
            dump[base + 4] = R0_hi;
            dump[base + 5] = R1_hi;
            dump[base + 6] = R2_hi;
            dump[base + 7] = R3_hi;
        }
    }
}

int main() {
    // COL-MARKER: byte@(row, col) = uint8_t(col)
    uint8_t h_Q[64 * 128];
    for (int row = 0; row < 64; ++row)
        for (int col = 0; col < 128; ++col)
            h_Q[row * 128 + col] = (uint8_t)col;

    uint8_t *d_Q;
    uint32_t *d_dump;
    CHK(cudaMalloc(&d_Q, 8192));
    CHK(cudaMalloc(&d_dump, 2 * 8 * 32 * 8 * sizeof(uint32_t)));
    CHK(cudaMemcpy(d_Q, h_Q, 8192, cudaMemcpyHostToDevice));

    probe_kernel<<<1, 128>>>(d_Q, d_dump);
    CHK(cudaGetLastError());
    CHK(cudaDeviceSynchronize());

    uint32_t h_dump[2 * 8 * 32 * 8];
    CHK(cudaMemcpy(h_dump, d_dump, sizeof(h_dump), cudaMemcpyDeviceToHost));

    // Domain injectivity: 128 col values in 8-bit byte, coll-free by construction.
    printf("=== 061 C0 col-проход (col-marker) ===\n");
    printf("Injectivity: 128 unique col-values <= 256 byte states, no collisions by construction ✓\n");
    printf("Coverage: 2 kb × 8 np × 32 lanes × 2 lo/hi × 4 regs × 4 bytes = %d samples\n",
           2 * 8 * 32 * 2 * 4 * 4);

    // Dump kb=0 np=0 LO for all 32 lanes to check byte order in R0..R3
    printf("\n--- Sample dump: kb=0, np=0, LDSM lo, all lanes (bytes = col positions) ---\n");
    for (int lane = 0; lane < 32; ++lane) {
        int base = (0 * 8 + 0) * 32 * 8 + lane * 8;
        uint32_t R0 = h_dump[base + 0];
        uint32_t R1 = h_dump[base + 1];
        uint32_t R2 = h_dump[base + 2];
        uint32_t R3 = h_dump[base + 3];
        uint8_t b[16];
        for (int k = 0; k < 4; ++k) {
            b[k] = (R0 >> (k*8)) & 0xFF;
            b[k+4] = (R1 >> (k*8)) & 0xFF;
            b[k+8] = (R2 >> (k*8)) & 0xFF;
            b[k+12] = (R3 >> (k*8)) & 0xFF;
        }
        printf("lane=%2d: cols={", lane);
        for (int k = 0; k < 16; ++k) printf(" %3d", b[k]);
        printf(" }\n");
    }

    // MMA m16n8k32.e4m3 B-op fragment expectation:
    //   B is 8 cols × 32 rows fp8 (n=8, k=32). Per lane: 8 fp8 = 2 uint32.
    //   For LDSM.x2 = 2 matrices (b0-pair + b1-pair), 4 uint32 total: {B0a, B0b, B1a, B1b}.
    //   Per lane 4 uint32 outputs → maps to 2 MMA-B calls with ni_a/ni_b pair.
    //
    // Expected col positions per R0/R1 (b0-pair, ni_a):
    //   For np=0, col_start_in_row = 0 (row_ptr uses np*16 offset).
    //   R0 first 4 bytes: col positions 0..3 (or lane-permuted).
    //   R0/R1 duplicated per ISA quirk 045 (b0 halves match).

    // Check: each output byte is valid col-index 0..127
    int total_samples = 0, valid_samples = 0;
    int seen_cols[128] = {0};
    for (int kb = 0; kb < 2; ++kb) {
        for (int np = 0; np < 8; ++np) {
            for (int lane = 0; lane < 32; ++lane) {
                for (int lohi = 0; lohi < 2; ++lohi) {
                    for (int r = 0; r < 4; ++r) {
                        int base = (kb * 8 + np) * 32 * 8 + lane * 8;
                        uint32_t R = h_dump[base + lohi * 4 + r];
                        for (int bt = 0; bt < 4; ++bt) {
                            uint8_t v = (R >> (bt * 8)) & 0xFF;
                            total_samples++;
                            if (v < 128) { valid_samples++; seen_cols[v]++; }
                        }
                    }
                }
            }
        }
    }
    printf("\n--- Coverage report ---\n");
    printf("Total samples: %d\n", total_samples);
    printf("Valid col-values (0..127): %d\n", valid_samples);
    printf("Percentage valid: %.2f%%\n", 100.0 * valid_samples / total_samples);
    int unique_cols_seen = 0;
    for (int c = 0; c < 128; ++c) if (seen_cols[c] > 0) unique_cols_seen++;
    printf("Unique cols seen: %d / 128\n", unique_cols_seen);

    // Byte ORDER check within R0 (per-lane): для MMA-B фрагмента m16n8k32.e4m3:
    //   B is 32×8 fp8, per-lane 8 fp8 (2 uint32). Per lane l: R0 = 4 fp8 at (k=0..3, n=f(l)).
    //   Все 4 bytes R0 должны иметь ОДНУ col-позицию (same n), затем R1 — другую col-позицию.
    //   ISA-квирк 045: R2 = R0 dup, R3 = R1 dup.
    printf("\n--- Byte ORDER check within R0/R1 (kb=0, np=0, LDSM lo) ---\n");
    printf("MMA-B expectation: R0 [4 bytes] all at SAME col-pos (n-value), R2=R0 dup\n");
    int r0_uniform = 0, r0_total = 0;
    int r1_diff_from_r0 = 0;
    int r2_dup_r0 = 0, r3_dup_r1 = 0;
    for (int lane = 0; lane < 32; ++lane) {
        int base = (0 * 8 + 0) * 32 * 8 + lane * 8;
        uint32_t R0 = h_dump[base + 0];
        uint32_t R1 = h_dump[base + 1];
        uint32_t R2 = h_dump[base + 2];
        uint32_t R3 = h_dump[base + 3];
        uint8_t r0b[4], r1b[4];
        for (int k = 0; k < 4; ++k) {
            r0b[k] = (R0 >> (k*8)) & 0xFF;
            r1b[k] = (R1 >> (k*8)) & 0xFF;
        }
        bool r0_all_same = (r0b[0] == r0b[1] && r0b[1] == r0b[2] && r0b[2] == r0b[3]);
        bool r1_all_same = (r1b[0] == r1b[1] && r1b[1] == r1b[2] && r1b[2] == r1b[3]);
        r0_total++;
        if (r0_all_same) r0_uniform++;
        if (r1_all_same && r0b[0] != r1b[0]) r1_diff_from_r0++;
        if (R2 == R0) r2_dup_r0++;
        if (R3 == R1) r3_dup_r1++;
        if (lane < 8) printf("lane=%d R0={%d,%d,%d,%d} R1={%d,%d,%d,%d} R2==R0:%s R3==R1:%s\n",
                              lane, r0b[0],r0b[1],r0b[2],r0b[3], r1b[0],r1b[1],r1b[2],r1b[3],
                              (R2==R0)?"YES":"NO", (R3==R1)?"YES":"NO");
    }
    printf("\nR0 uniform (all 4 bytes same col) : %d / %d = %.1f%%\n",
           r0_uniform, r0_total, 100.0*r0_uniform/r0_total);
    printf("R1 uniform + different from R0    : %d / %d = %.1f%%\n",
           r1_diff_from_r0, r0_total, 100.0*r1_diff_from_r0/r0_total);
    printf("R2 duplicates R0 (ISA-квирк 045)  : %d / %d = %.1f%%\n",
           r2_dup_r0, r0_total, 100.0*r2_dup_r0/r0_total);
    printf("R3 duplicates R1 (ISA-квирк 045)  : %d / %d = %.1f%%\n",
           r3_dup_r1, r0_total, 100.0*r3_dup_r1/r0_total);

    bool bridge_100 = (valid_samples == total_samples) &&
                     (unique_cols_seen == 128) &&
                     (r0_uniform == r0_total) &&
                     (r1_diff_from_r0 == r0_total) &&
                     (r2_dup_r0 == r0_total) &&
                     (r3_dup_r1 == r0_total);
    printf("\n=== C0 BRIDGE VERDICT ===\n");
    printf("Col-проход %s\n", bridge_100 ? "100% (byte ORDER соответствует MMA-B фрагменту)"
                                        : "< 100% (byte ordering не совпадает с ожиданием)");
    return bridge_100 ? 0 : 1;
}
