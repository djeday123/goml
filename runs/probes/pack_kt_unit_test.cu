// pack_kt_unit_test.cu — 025-b unit-test для dq_new K_T pack scatter
//   Phase A/B/C/D дословно перенесены из Python simulate_dq_pack_shfl.py (8192/8192 ✓)
//   Slot: bit1=slot_half (0=kr_lo, 1=kr_hi), bit0=slot_ni_hi (0=ni_base=0, 1=ni_base=4)
//   Group: {c + 4*p' + 16*h} с fixed c, h; варьирующий p' ∈ [0..3]
//   Обмен по l_div4-axis (Vugar's 'd').
//   ЗАПРЕТ: локальные массивы V[] с runtime-индексацией → V0-V3 именованные регистры

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

#define Bc 64
#define Br 64
#define Hd 128
#define KT_STRIDE 68
#define NI_QK 8
#define KS_QK 4

__global__ void pack_kt_unit_kernel(uint8_t *dst_smKT_out) {
    __shared__ uint8_t smK[Bc][Hd];
    __shared__ uint8_t smKT[Hd][KT_STRIDE];

    const int tid    = threadIdx.x;
    const int wid    = tid >> 5;
    const int lane   = tid & 31;
    const int l_div4 = lane >> 2;
    const int l_mod4 = lane & 3;

    // Fill smK: K[n][k] = (n*131 + k*7 + 13) & 0xFF
    for (int idx = tid; idx < Bc*Hd; idx += blockDim.x) {
        int n = idx / Hd;
        int k = idx % Hd;
        smK[n][k] = (uint8_t)((n * 131 + k * 7 + 13) & 0xFF);
    }
    for (int idx = tid; idx < Hd*KT_STRIDE; idx += blockDim.x) {
        int row = idx / KT_STRIDE;
        int col = idx % KT_STRIDE;
        smKT[row][col] = 0;
    }
    __syncthreads();

    // Feeder (k_xor=0 for unit-test)
    const int ks = wid;
    const int k_lo = ks * 32 + l_mod4 * 4;
    const int k_hi = k_lo + 16;
    uint32_t kr_lo[NI_QK], kr_hi[NI_QK];
    #pragma unroll
    for (int ni = 0; ni < NI_QK; ++ni) {
        int n_K = ni * 8 + l_div4;
        kr_lo[ni] = *reinterpret_cast<uint32_t*>(&smK[n_K][k_lo]);
        kr_hi[ni] = *reinterpret_cast<uint32_t*>(&smK[n_K][k_hi]);
    }

    // Coords for group structure
    const int c = l_mod4;
    const int p = l_div4 & 3;
    const int h = l_div4 >> 2;

    // Pack A/B/C/D — 4 slots × 4 outputs
    #pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
        const int slot_half   = (slot >> 1) & 1;  // 0=kr_lo, 1=kr_hi
        const int slot_ni_hi  = slot & 1;         // 0=ni_base=0, 1=ni_base=4
        const int ni_base     = slot_ni_hi * 4;

        // Select 4 uint32 inputs from kr_lo/kr_hi[ni_base..ni_base+3]
        uint32_t W0, W1, W2, W3;
        if (slot_half == 0) {
            W0 = kr_lo[ni_base + 0]; W1 = kr_lo[ni_base + 1];
            W2 = kr_lo[ni_base + 2]; W3 = kr_lo[ni_base + 3];
        } else {
            W0 = kr_hi[ni_base + 0]; W1 = kr_hi[ni_base + 1];
            W2 = kr_hi[ni_base + 2]; W3 = kr_hi[ni_base + 3];
        }

        // Phase A — gather byte j across 4 W's (8 PRMT)
        uint32_t t01_lo, t01_hi, t23_lo, t23_hi;
        asm volatile("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(t01_lo) : "r"(W0), "r"(W1));
        asm volatile("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(t01_hi) : "r"(W0), "r"(W1));
        asm volatile("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(t23_lo) : "r"(W2), "r"(W3));
        asm volatile("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(t23_hi) : "r"(W2), "r"(W3));
        uint32_t G0, G1, G2, G3;
        asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(G0) : "r"(t01_lo), "r"(t23_lo));
        asm volatile("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(G1) : "r"(t01_lo), "r"(t23_lo));
        asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(G2) : "r"(t01_hi), "r"(t23_hi));
        asm volatile("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(G3) : "r"(t01_hi), "r"(t23_hi));

        // Phase B — 3 SHFL exchange (identical to dk_new pack Phase B)
        uint32_t V0 = G0, V1 = G1, V2 = G2, V3 = G3;
        #pragma unroll
        for (int r = 1; r <= 3; ++r) {
            int src_p = (p - r) & 3;
            int src_lane = c + 4 * src_p + 16 * h;
            int idx = (p + r) & 3;
            uint32_t lo = (idx & 1) ? G1 : G0;
            uint32_t hi = (idx & 1) ? G3 : G2;
            uint32_t expose_val = (idx & 2) ? hi : lo;
            uint32_t val = __shfl_sync(0xFFFFFFFF, expose_val, src_lane);
            V0 = (src_p == 0) ? val : V0;
            V1 = (src_p == 1) ? val : V1;
            V2 = (src_p == 2) ? val : V2;
            V3 = (src_p == 3) ? val : V3;
        }

        // Phase C — receive PRMT (8)
        uint32_t u01_lo, u01_hi, u23_lo, u23_hi;
        asm volatile("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(u01_lo) : "r"(V0), "r"(V1));
        asm volatile("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(u01_hi) : "r"(V0), "r"(V1));
        asm volatile("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(u23_lo) : "r"(V2), "r"(V3));
        asm volatile("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(u23_hi) : "r"(V2), "r"(V3));
        uint32_t OUT0, OUT1, OUT2, OUT3;
        asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(OUT0) : "r"(u01_lo), "r"(u23_lo));
        asm volatile("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(OUT1) : "r"(u01_lo), "r"(u23_lo));
        asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(OUT2) : "r"(u01_hi), "r"(u23_hi));
        asm volatile("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(OUT3) : "r"(u01_hi), "r"(u23_hi));

        // Phase D — 4 STS.32 (dq mapping)
        const int base_row = wid * 32 + 4 * c + p + 16 * slot_half;
        const int col_base_0 = (ni_base + 0) * 8 + 4 * h;
        const int col_base_1 = (ni_base + 1) * 8 + 4 * h;
        const int col_base_2 = (ni_base + 2) * 8 + 4 * h;
        const int col_base_3 = (ni_base + 3) * 8 + 4 * h;
        *reinterpret_cast<uint32_t*>(&smKT[base_row][col_base_0]) = OUT0;
        *reinterpret_cast<uint32_t*>(&smKT[base_row][col_base_1]) = OUT1;
        *reinterpret_cast<uint32_t*>(&smKT[base_row][col_base_2]) = OUT2;
        *reinterpret_cast<uint32_t*>(&smKT[base_row][col_base_3]) = OUT3;
    }
    __syncthreads();

    for (int idx = tid; idx < Hd*KT_STRIDE; idx += blockDim.x) {
        dst_smKT_out[idx] = ((uint8_t*)smKT)[idx];
    }
}

int main() {
    uint8_t *d_out;
    cudaMalloc(&d_out, Hd*KT_STRIDE);
    pack_kt_unit_kernel<<<1, 128>>>(d_out);
    cudaDeviceSynchronize();

    uint8_t h_out[Hd*KT_STRIDE];
    cudaMemcpy(h_out, d_out, Hd*KT_STRIDE, cudaMemcpyDeviceToHost);

    int match = 0, mism = 0;
    for (int k = 0; k < Hd; ++k) {
        for (int n = 0; n < Bc; ++n) {
            uint8_t expect = (uint8_t)((n * 131 + k * 7 + 13) & 0xFF);
            uint8_t got = h_out[k * KT_STRIDE + n];
            if (expect == got) match++;
            else {
                if (mism < 5) fprintf(stderr, "MISM k=%d n=%d expect=0x%02x got=0x%02x\n", k, n, expect, got);
                mism++;
            }
        }
    }
    printf("K_T bytes: match=%d/%d, mismatch=%d\n", match, Hd*Bc, mism);
    cudaFree(d_out);
    return (mism == 0) ? 0 : 1;
}
