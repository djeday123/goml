// transpose_ds_unit_test.cu — 033-b unit-test для dS_nat → dS_T aliased transpose (W2)
//   Реализация по CPU-судье simulate_transpose_ds.py (4096/4096 verified).
//   Feeder + Phase D выведены ИЗ ЧИТАТЕЛЯ dk_new (fa_bwd_dk_new.cu:239-245).

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

#define Bc 64
#define Br 64
#define NAT_STRIDE Bc
#define T_STRIDE Br

__global__ void transpose_ds_unit_kernel(uint8_t *dst_smT_out) {
    __shared__ uint8_t smdS_area[Bc * Br];  // 4096 B aliased nat ↔ T

    const int tid    = threadIdx.x;
    const int wid    = tid >> 5;
    const int lane   = tid & 31;
    const int l_div4 = lane >> 2;
    const int l_mod4 = lane & 3;

    // Fill smdS_area with marker bytes in natural [i][j] layout
    for (int idx = tid; idx < Bc * Br; idx += blockDim.x) {
        int i = idx / Br;
        int j = idx % Br;
        smdS_area[i * NAT_STRIDE + j] = (uint8_t)((i * 131 + j * 7 + 13) & 0xFF);
    }
    __syncthreads();

    // Read 8 W-uint32 per lane (2 slots × 4 per slot)
    // Feeder derived from reader (Phase D):
    //   Per slot kb: W_r at lane (wid, c=l_mod4, p=l_div4&3, h=l_div4>>2):
    //     W_0: nat[i=kb*32+c*4+p][j=wid*16+4h..+3]
    //     W_1: nat[i=kb*32+c*4+p][j=wid*16+4h+8..+11]
    //     W_2: nat[i=kb*32+c*4+16+p][j=wid*16+4h..+3]
    //     W_3: nat[i=kb*32+c*4+16+p][j=wid*16+4h+8..+11]
    uint32_t W_all[8];
    const int c = l_mod4;
    const int p = l_div4 & 3;
    const int h = l_div4 >> 2;
    #pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
        int kb = slot;
        int i0 = kb * 32 + c * 4 + p;
        int i1 = i0 + 16;
        int j0 = wid * 16 + 4 * h;
        int j1 = j0 + 8;
        W_all[slot * 4 + 0] = *reinterpret_cast<uint32_t*>(&smdS_area[i0 * NAT_STRIDE + j0]);
        W_all[slot * 4 + 1] = *reinterpret_cast<uint32_t*>(&smdS_area[i0 * NAT_STRIDE + j1]);
        W_all[slot * 4 + 2] = *reinterpret_cast<uint32_t*>(&smdS_area[i1 * NAT_STRIDE + j0]);
        W_all[slot * 4 + 3] = *reinterpret_cast<uint32_t*>(&smdS_area[i1 * NAT_STRIDE + j1]);
    }

    // BARRIER #NEW (W2): all reads before aliased overwrite
    __syncthreads();

    // Phase A/B/C/D per slot
    #pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
        uint32_t W0 = W_all[slot * 4 + 0];
        uint32_t W1 = W_all[slot * 4 + 1];
        uint32_t W2 = W_all[slot * 4 + 2];
        uint32_t W3 = W_all[slot * 4 + 3];

        // Phase A: 8 PRMT
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

        // Phase B: 3 SHFL exchange
        uint32_t V0 = G0, V1 = G1, V2 = G2, V3 = G3;
        #pragma unroll
        for (int r = 1; r <= 3; ++r) {
            int src_p = (p - r) & 3;
            int src_lane = c + 4 * src_p + 16 * h;
            int idx = (p + r) & 3;
            uint32_t lo_g = (idx & 1) ? G1 : G0;
            uint32_t hi_g = (idx & 1) ? G3 : G2;
            uint32_t expose_val = (idx & 2) ? hi_g : lo_g;
            uint32_t val = __shfl_sync(0xFFFFFFFF, expose_val, src_lane);
            V0 = (src_p == 0) ? val : V0;
            V1 = (src_p == 1) ? val : V1;
            V2 = (src_p == 2) ? val : V2;
            V3 = (src_p == 3) ? val : V3;
        }

        // Phase C: reorder V → OUT (8 PRMT)
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

        // Phase D: STS.32 to T layout (aliased overwrite)
        // Reader positions:
        //   OUT_0 → T[m_lo][k_i_lo..+3]
        //   OUT_1 → T[m_hi][k_i_lo..+3]
        //   OUT_2 → T[m_lo][k_i_hi..+3]
        //   OUT_3 → T[m_hi][k_i_hi..+3]
        int kb = slot;
        int m_lo = wid * 16 + 4 * h + p;
        int m_hi = m_lo + 8;
        int k_i_lo = kb * 32 + c * 4;
        int k_i_hi = k_i_lo + 16;
        *reinterpret_cast<uint32_t*>(&smdS_area[m_lo * T_STRIDE + k_i_lo]) = OUT0;
        *reinterpret_cast<uint32_t*>(&smdS_area[m_hi * T_STRIDE + k_i_lo]) = OUT1;
        *reinterpret_cast<uint32_t*>(&smdS_area[m_lo * T_STRIDE + k_i_hi]) = OUT2;
        *reinterpret_cast<uint32_t*>(&smdS_area[m_hi * T_STRIDE + k_i_hi]) = OUT3;
    }
    __syncthreads();

    // Copy T layout to global for verify
    for (int idx = tid; idx < Bc * Br; idx += blockDim.x) {
        dst_smT_out[idx] = smdS_area[idx];
    }
}

int main() {
    uint8_t *d_out;
    cudaMalloc(&d_out, Br * Bc);
    transpose_ds_unit_kernel<<<1, 128>>>(d_out);
    cudaDeviceSynchronize();

    uint8_t h_out[Br * Bc];
    cudaMemcpy(h_out, d_out, Br * Bc, cudaMemcpyDeviceToHost);

    int match = 0, mism = 0;
    for (int j = 0; j < Br; ++j) {
        for (int i = 0; i < Bc; ++i) {
            uint8_t expect = (uint8_t)((i * 131 + j * 7 + 13) & 0xFF);
            uint8_t got = h_out[j * T_STRIDE + i];
            if (expect == got) match++;
            else {
                if (mism < 5) fprintf(stderr, "MISM j=%d i=%d expect=0x%02x got=0x%02x\n", j, i, expect, got);
                mism++;
            }
        }
    }
    printf("dS_T bytes: match=%d/%d, mismatch=%d\n", match, Br * Bc, mism);
    cudaFree(d_out);
    return (mism == 0) ? 0 : 1;
}
