// 048 §3: микропроба-мост — repeat production паттерна в миниатюре с np-loop.
// Проверяет что LDSM внутри loop даёт корректные b0/b1 для каждой (lane, kb, np)
// против фрагмент-ожидания B-op mma.m16n8k32.e4m3.
// Критерий: 100% на всех (lane, kb, np).

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){fprintf(stderr,"CUDA:%s\n",cudaGetErrorString(e));exit(1);}} while(0)

__global__ void probe_dkS2v2(uint32_t *out) {
    // smQ mimic: 64 rows × 128 cols fp8. Marker: byte = (row<<4)|(col&0xF)
    __shared__ uint8_t smQ[64 * 128];
    int tid = threadIdx.x;
    if (tid < 128) {
        for (int j = 0; j < 64; ++j) {
            int idx = tid + j * 128;
            int r = idx / 128, c = idx % 128;
            smQ[idx] = (uint8_t)((r << 4) | (c & 0xF));
        }
    }
    __syncthreads();

    int lane = tid & 31;
    // np=0, kb=0 iteration (b0 low + b1 high)
    // Test np=0..3 to verify np-mapping (each np changes n_col)
    // Test kb=0, 1 (row offset changes)
    // Total: 2 kb × 4 np × 32 lanes × 4 registers = 1024 output positions
    // Layout: out[kb*4*32*4 + np*32*4 + lane*4 + r]

    #pragma unroll
    for (int kb = 0; kb < 2; ++kb) {
        #pragma unroll
        for (int np = 0; np < 4; ++np) {
            uint32_t sm_lo = __cvta_generic_to_shared(&smQ[(kb*32 + lane) * 128 + np*16]);
            uint32_t R0lo, R1lo, R2lo, R3lo;
            asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3},[%4];"
                : "=r"(R0lo), "=r"(R1lo), "=r"(R2lo), "=r"(R3lo)
                : "r"(sm_lo));

            uint32_t sm_hi = __cvta_generic_to_shared(&smQ[(kb*32 + (lane&15) + 16) * 128 + np*16]);
            uint32_t R0hi, R1hi, R2hi, R3hi;
            asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3},[%4];"
                : "=r"(R0hi), "=r"(R1hi), "=r"(R2hi), "=r"(R3hi)
                : "r"(sm_hi));

            int base = kb*(4*32*4) + np*(32*4) + lane*4;
            if (tid < 32) {
                // Store R0lo (b0 ni_a) + R1lo (b0 ni_b) + R0hi (b1 ni_a) + R1hi (b1 ni_b)
                out[base+0] = R0lo;
                out[base+1] = R1lo;
                out[base+2] = R0hi;
                out[base+3] = R1hi;
            }
        }
    }
}

int main() {
    const int TOTAL = 2*4*32*4;  // 1024
    uint32_t *d_out; uint32_t h[TOTAL];
    CK(cudaMalloc(&d_out, TOTAL*sizeof(uint32_t)));
    CK(cudaMemset(d_out, 0xFF, TOTAL*sizeof(uint32_t)));
    probe_dkS2v2<<<1, 128>>>(d_out);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "kernel fail: %s\n", cudaGetErrorString(err)); return 1; }
    CK(cudaMemcpy(h, d_out, TOTAL*sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Проверка per (kb, np, lane): все 4 регистра matching MMA-B expectation
    // Per lane l (groupID = l/4, laneID = l%4):
    //   b0(ni_a) = 4 halves at (k = 4*laneID..+3 + kb*32, n = groupID + np*8)     <-- fix
    //   b0(ni_b) = 4 halves at (k = 4*laneID..+3 + kb*32, n = groupID + np*8 + 8) ← wait, ni_b=2*np+1 → n = ni_b*8 + groupID
    //   Actually ni_a = 2*np, ni_b = 2*np+1: n_a = 2*np*8 + groupID = 16*np + groupID;
    //                                        n_b = (2*np+1)*8 + groupID = 16*np + 8 + groupID
    //   b1(ni_a) = same n_a, k+16 offset
    //   b1(ni_b) = same n_b, k+16 offset

    int ok = 0, total_slots = 0;
    printf("048 §3 микропроба-мост: LDSM во np-loop (2 kb × 4 np × 32 lane × 4 reg = %d slots)\n\n", TOTAL);
    for (int kb = 0; kb < 2; ++kb) {
        for (int np = 0; np < 4; ++np) {
            for (int lane = 0; lane < 32; ++lane) {
                int groupID = lane / 4, laneID = lane % 4;
                int base = kb*(4*32*4) + np*(32*4) + lane*4;
                uint32_t R0lo = h[base+0], R1lo = h[base+1];
                uint32_t R0hi = h[base+2], R1hi = h[base+3];
                // R0lo = b0(ni_a): expect 4 halves at (k=4*laneID+kb*32..+3, n=groupID+16*np)
                // Check byte 0..3 of R0lo
                for (int bt = 0; bt < 4; ++bt) {
                    uint8_t byte = (R0lo >> (bt*8)) & 0xFF;
                    int r = byte >> 4, c = byte & 0xF;
                    int exp_row_l4 = (4*laneID + bt + kb*32) & 15;  // row bits {0..3}
                    int exp_col_l4 = (groupID + 16*np) & 15;
                    if (r == exp_row_l4 && c == exp_col_l4) ok++;
                    total_slots++;
                }
                // R1lo = b0(ni_b): n = groupID + 16*np + 8
                for (int bt = 0; bt < 4; ++bt) {
                    uint8_t byte = (R1lo >> (bt*8)) & 0xFF;
                    int r = byte >> 4, c = byte & 0xF;
                    int exp_row_l4 = (4*laneID + bt + kb*32) & 15;
                    int exp_col_l4 = (groupID + 16*np + 8) & 15;
                    if (r == exp_row_l4 && c == exp_col_l4) ok++;
                    total_slots++;
                }
                // R0hi = b1(ni_a): k+16 → row bits, same n
                for (int bt = 0; bt < 4; ++bt) {
                    uint8_t byte = (R0hi >> (bt*8)) & 0xFF;
                    int r = byte >> 4, c = byte & 0xF;
                    int exp_row_l4 = (4*laneID + bt + kb*32 + 16) & 15;
                    int exp_col_l4 = (groupID + 16*np) & 15;
                    if (r == exp_row_l4 && c == exp_col_l4) ok++;
                    total_slots++;
                }
                // R1hi = b1(ni_b)
                for (int bt = 0; bt < 4; ++bt) {
                    uint8_t byte = (R1hi >> (bt*8)) & 0xFF;
                    int r = byte >> 4, c = byte & 0xF;
                    int exp_row_l4 = (4*laneID + bt + kb*32 + 16) & 15;
                    int exp_col_l4 = (groupID + 16*np + 8) & 15;
                    if (r == exp_row_l4 && c == exp_col_l4) ok++;
                    total_slots++;
                }
            }
        }
    }
    printf("Match: %d / %d (%.2f%%)\n", ok, total_slots, 100.0*ok/total_slots);
    printf("Verdict: %s\n", (ok == total_slots) ? "PASS — 100%% match, ni-mapping верен" : "FAIL — расписание LDSM разошлось с ожиданием");

    // Sample lane 0/1/4 kb=0 np=0 dump
    printf("\nSample kb=0 np=0:\n");
    for (int l : {0, 1, 4, 8, 16}) {
        int base = 0 + 0 + l*4;
        printf("  lane %2d: R0lo=%08x R1lo=%08x R0hi=%08x R1hi=%08x\n",
               l, h[base+0], h[base+1], h[base+2], h[base+3]);
    }

    cudaFree(d_out);
    return (ok == total_slots) ? 0 : 1;
}
