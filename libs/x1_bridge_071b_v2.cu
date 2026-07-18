// 071b bridge v2: третий кандидат — m16n8.x1.trans.b8 (меньший shape, вывод из бит-карты свизла).
// Обоснование: наблюдение под col-marker показало b0_b at col=8 (byte value 0x08 в x2.R1),
// что misaligned для m16n16.x1.b8. Пробуем m16n8 shape — 128 bytes = 16 rows × 8 cols,
// может дать иную alignment ограничения ИЛИ иную internal fragment mapping.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include "fa_bwd_common.cuh"

#define CK(c) do{auto e=(c); if(e!=cudaSuccess){printf("CUDA err %d @%d\n",(int)e,__LINE__); return 1;}}while(0)

__global__ void bridge_probe_v2(uint32_t *out_x2_R1, uint32_t *out_x1_m16n8_a, uint32_t *out_x1_m16n8_b, int marker_mode)
{
    __shared__ uint8_t smQ[8192];

    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;

    constexpr int Br = 64, Hd = 128, CHUNK = 16;
    constexpr int cpr = Hd / CHUNK;
    for (int c = tid; c < Br * cpr; c += 128) {
        int row = c / cpr;
        int col_byte = (c % cpr) * CHUNK;
        int dst = swz_byte(row, col_byte);
        for (int b = 0; b < CHUNK; ++b) {
            uint8_t val = (marker_mode == 0) ? (uint8_t)(row & 0xFF) : (uint8_t)((col_byte + b) & 0xFF);
            smQ[dst + b] = val;
        }
    }
    __syncthreads();

    int kb = 0, np = 0;
    int row_lo = kb * 32 + lane;
    int addr_a = swz_byte(row_lo, np * 16);
    uint32_t sm_a = __cvta_generic_to_shared(&smQ[addr_a]);

    // Reference x2 R1 (b0_b)
    uint32_t B0a, B0b, D0, D1;
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3},[%4];\n"
        : "=r"(B0a), "=r"(B0b), "=r"(D0), "=r"(D1) : "r"(sm_a));

    int slot = wid * 32 + lane;
    out_x2_R1[slot] = B0b;

    // Кандидат #3: m16n8.x1.trans.b8 at aligned addr (col=0)
    // m16n8 shape = 16 rows × 8 cols = 128 bytes total = 4 bytes/lane = 1 uint32/lane output
    uint32_t R_m16n8_a;
    asm volatile("ldmatrix.sync.aligned.m16n8.x1.trans.shared.b8 {%0},[%1];\n"
        : "=r"(R_m16n8_a) : "r"(sm_a));
    out_x1_m16n8_a[slot] = R_m16n8_a;

    // Кандидат #3-alt: m16n8.x1.trans.b8 at col=16 (aligned)
    int addr_b = swz_byte(row_lo, np * 16 + 16);
    uint32_t sm_b = __cvta_generic_to_shared(&smQ[addr_b]);
    uint32_t R_m16n8_b;
    asm volatile("ldmatrix.sync.aligned.m16n8.x1.trans.shared.b8 {%0},[%1];\n"
        : "=r"(R_m16n8_b) : "r"(sm_b));
    out_x1_m16n8_b[slot] = R_m16n8_b;
}

int main() {
    printf("=== 071b v2 bridge probe: m16n8.x1.trans.b8 candidates ===\n\n");

    uint32_t *d_x2_R1, *d_a, *d_b;
    CK(cudaMalloc(&d_x2_R1, 128 * 4));
    CK(cudaMalloc(&d_a, 128 * 4));
    CK(cudaMalloc(&d_b, 128 * 4));

    uint32_t x2_R1[128], m16n8_a[128], m16n8_b[128];

    for (int mode = 0; mode < 2; ++mode) {
        printf("=== MARKER MODE %d (%s) ===\n", mode, mode == 0 ? "row-marker" : "col-marker");
        bridge_probe_v2<<<1, 128>>>(d_x2_R1, d_a, d_b, mode);
        auto e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("kernel err: %s\n", cudaGetErrorString(e)); return 1; }

        CK(cudaMemcpy(x2_R1, d_x2_R1, sizeof(x2_R1), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(m16n8_a, d_a, sizeof(m16n8_a), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(m16n8_b, d_b, sizeof(m16n8_b), cudaMemcpyDeviceToHost));

        int match_a = 0, match_b = 0;
        for (int slot = 0; slot < 128; ++slot) {
            if (m16n8_a[slot] == x2_R1[slot]) match_a++;
            if (m16n8_b[slot] == x2_R1[slot]) match_b++;
        }
        printf("  m16n8.x1 @ col=0  vs x2.R1 (b0_b):  %d/128 (%.1f%%)\n", match_a, 100.0 * match_a / 128);
        printf("  m16n8.x1 @ col=16 vs x2.R1 (b0_b):  %d/128 (%.1f%%)\n", match_b, 100.0 * match_b / 128);

        printf("  wid=0 lanes 0..3:\n");
        for (int lane = 0; lane < 4; ++lane) {
            printf("    l%d: x2.R1=%08x  m16n8_a=%08x  m16n8_b=%08x\n",
                   lane, x2_R1[lane], m16n8_a[lane], m16n8_b[lane]);
        }
        printf("\n");
    }

    cudaFree(d_x2_R1); cudaFree(d_a); cudaFree(d_b);
    return 0;
}
