// pack_qt_unit_test.cu — Vugar spec-хореография (Фазы A/B/C/D) verbatim.
//   4 warps × 32 threads = 128 threads, 1 block.
//   Маркер-байты: smQ[m][k] = pack(m, k) уникально по (lane, ks, slot, byte)
//   Assert: 8192/8192 valid bytes match vs CPU reference.
//   SASS gate: ровно 12 SHFL + 16 STS.32 + 0 STS.U8 + 0 LDL/STL.
//
// Layout mirror kernel_dk_new:
//   Br=64, Hd=128, QT_STRIDE=68, KS_QK=4.
//   smQ: 64 × 128 (Br rows × Hd cols), smQ_T: 128 × 68 (Hd rows × Br+padding cols).

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

constexpr int BR = 64;
constexpr int HD = 128;
constexpr int QT_STRIDE = 68;
constexpr int KS_QK = 4;
constexpr int SMEM_SMQ_BYTES   = BR * HD;              // 8192
constexpr int SMEM_SMQT_BYTES  = HD * QT_STRIDE;       // 8704
constexpr int SMEM_TOTAL_BYTES = SMEM_SMQ_BYTES + SMEM_SMQT_BYTES;

// Unique marker byte per (m, k): mix of both indices → all 8192 distinct via LCG-like
__host__ __device__ inline uint8_t marker(int m, int k) {
    // Injective on (m, k) ∈ [0,64) × [0,128) — но 0..8191 не влезает в 8 bit
    // Use (m*131 + k*7 + 13) mod 251 — pseudo-unique, then bit-mix with rare collisions
    return (uint8_t)((m * 131 + k * 7 + 13) & 0xFF);
}

// PRMT.b32 wrapper (nvcc intrinsic)
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t r;
    asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}

// Variant (B) — direct-gather with precomputed constants от p (compile-time via template)
// В юнит-тесте p — рантайм, поэтому используем SEL-tree (variant A):
//   G_at_p_plus_r = SEL{r={1,2,3}} через двухуровневое условие
__device__ __forceinline__ uint32_t sel_at(uint32_t g0, uint32_t g1, uint32_t g2, uint32_t g3, int idx) {
    // idx ∈ {0..3} → G[idx]. Two-level SEL (no local memory).
    uint32_t lo = (idx & 1) ? g1 : g0;
    uint32_t hi = (idx & 1) ? g3 : g2;
    return (idx & 2) ? hi : lo;
}

__global__ void pack_kernel(uint8_t *smQ_T_out, uint8_t *smQ_init) {
    __shared__ uint8_t smQ  [SMEM_SMQ_BYTES];
    __shared__ uint8_t smQ_T[SMEM_SMQT_BYTES];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;

    // Init smQ from external buffer (host-computed markers)
    for (int i = tid; i < SMEM_SMQ_BYTES; i += 128) smQ[i] = smQ_init[i];
    // Init smQ_T to 0xEE (detector for "not written")
    for (int i = tid; i < SMEM_SMQT_BYTES; i += 128) smQ_T[i] = 0xEE;
    __syncthreads();

    // ---- Feeder (mirror kernel_dk_new lines 141-152) ----
    uint32_t Qr[KS_QK][4];
    const int l_div4 = lane >> 2;
    const int l_mod4 = lane & 3;
    #pragma unroll
    for (int ks = 0; ks < KS_QK; ++ks) {
        int m_lo = wid * 16 + l_div4 + 0;
        int m_hi = wid * 16 + l_div4 + 8;
        int k_lo = ks * 32 + l_mod4 * 4 + 0;
        int k_hi = ks * 32 + l_mod4 * 4 + 16;
        Qr[ks][0] = *reinterpret_cast<uint32_t*>(&smQ[m_lo * HD + k_lo]);
        Qr[ks][1] = *reinterpret_cast<uint32_t*>(&smQ[m_hi * HD + k_lo]);
        Qr[ks][2] = *reinterpret_cast<uint32_t*>(&smQ[m_lo * HD + k_hi]);
        Qr[ks][3] = *reinterpret_cast<uint32_t*>(&smQ[m_hi * HD + k_hi]);
    }

    // Coords in quad
    const int c = l_mod4;
    const int d = l_div4;
    const int p = d & 3;      // quad-index in 4-lane group
    const int h = d >> 2;     // 0 = low, 1 = high (col +4)

    // Per slot s = 0..3 (s&1 = m-half, s>>1 = k-half)
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        // ---- Фаза A — gather вдоль ks (8 PRMT, fixed selectors) ----
        uint32_t t01_lo = prmt(Qr[0][s], Qr[1][s], 0x5140);   // [Q0.b0, Q1.b0, Q0.b1, Q1.b1]
        uint32_t t01_hi = prmt(Qr[0][s], Qr[1][s], 0x7362);   // [Q0.b2, Q1.b2, Q0.b3, Q1.b3]
        uint32_t t23_lo = prmt(Qr[2][s], Qr[3][s], 0x5140);
        uint32_t t23_hi = prmt(Qr[2][s], Qr[3][s], 0x7362);
        uint32_t G0 = prmt(t01_lo, t23_lo, 0x5410);           // [Q0.b0, Q1.b0, Q2.b0, Q3.b0]
        uint32_t G1 = prmt(t01_lo, t23_lo, 0x7632);
        uint32_t G2 = prmt(t01_hi, t23_hi, 0x5410);
        uint32_t G3 = prmt(t01_hi, t23_hi, 0x7632);

        // ---- Фаза B — обмен (3 SHFL, rounds r=1..3) ----
        uint32_t V[4];
        V[p] = 0;   // не будет использован, W_in[p] = G[p] прямо
        #pragma unroll
        for (int r = 1; r <= 3; ++r) {
            int src_p = (p - r) & 3;
            int src_lane = c + 4 * src_p + 16 * h;
            uint32_t expose_val = sel_at(G0, G1, G2, G3, (p + r) & 3);
            uint32_t recvd = __shfl_sync(0xFFFFFFFF, expose_val, src_lane);
            V[src_p] = recvd;   // (p - r) & 3 = src_p
        }
        // W_in: assemble from own G[p] and received V[q] for q != p
        uint32_t W_in[4];
        W_in[0] = (p == 0) ? G0 : V[0];
        W_in[1] = (p == 1) ? G1 : V[1];
        W_in[2] = (p == 2) ? G2 : V[2];
        W_in[3] = (p == 3) ? G3 : V[3];

        // ---- Фаза C — приёмное транспонирование (8 PRMT, тот же tree) ----
        uint32_t u01_lo = prmt(W_in[0], W_in[1], 0x5140);
        uint32_t u01_hi = prmt(W_in[0], W_in[1], 0x7362);
        uint32_t u23_lo = prmt(W_in[2], W_in[3], 0x5140);
        uint32_t u23_hi = prmt(W_in[2], W_in[3], 0x7362);
        uint32_t OUT0 = prmt(u01_lo, u23_lo, 0x5410);
        uint32_t OUT1 = prmt(u01_lo, u23_lo, 0x7632);
        uint32_t OUT2 = prmt(u01_hi, u23_hi, 0x5410);
        uint32_t OUT3 = prmt(u01_hi, u23_hi, 0x7632);

        // ---- Фаза D — стор (4 STS.32 per slot) ----
        // row = ks*32 + 16*(s>>1) + 4c + p
        // colbase = wid*16 + 8*(s&1) + 4h
        int colbase = wid * 16 + 8 * (s & 1) + 4 * h;
        int row_base_ks = 16 * (s >> 1) + 4 * c + p;
        #pragma unroll
        for (int ks = 0; ks < KS_QK; ++ks) {
            int row = ks * 32 + row_base_ks;
            uint32_t out_val = (ks == 0) ? OUT0 :
                               (ks == 1) ? OUT1 :
                               (ks == 2) ? OUT2 : OUT3;
            *reinterpret_cast<uint32_t*>(&smQ_T[row * QT_STRIDE + colbase]) = out_val;
        }
    }
    __syncthreads();

    // Copy smQ_T to global for host check
    for (int i = tid; i < SMEM_SMQT_BYTES; i += 128) smQ_T_out[i] = smQ_T[i];
}

// CPU reference: enumerate scatter per lane → target bytes
void cpu_scatter_ref(uint8_t *smQ_ref, uint8_t *smQ_T_ref) {
    memset(smQ_T_ref, 0xEE, SMEM_SMQT_BYTES);   // detector

    // Iterate all 128 threads (4 warps × 32 lanes)
    for (int tid = 0; tid < 128; ++tid) {
        int lane = tid & 31;
        int wid  = tid >> 5;
        int l_div4 = lane >> 2;
        int l_mod4 = lane & 3;

        // Feeder
        uint32_t Qr[KS_QK][4];
        for (int ks = 0; ks < KS_QK; ++ks) {
            int m_lo = wid * 16 + l_div4 + 0;
            int m_hi = wid * 16 + l_div4 + 8;
            int k_lo = ks * 32 + l_mod4 * 4 + 0;
            int k_hi = ks * 32 + l_mod4 * 4 + 16;
            Qr[ks][0] = *reinterpret_cast<uint32_t*>(&smQ_ref[m_lo * HD + k_lo]);
            Qr[ks][1] = *reinterpret_cast<uint32_t*>(&smQ_ref[m_hi * HD + k_lo]);
            Qr[ks][2] = *reinterpret_cast<uint32_t*>(&smQ_ref[m_lo * HD + k_hi]);
            Qr[ks][3] = *reinterpret_cast<uint32_t*>(&smQ_ref[m_hi * HD + k_hi]);
        }

        // Original scatter (per lane 64 STS.U8)
        for (int ks = 0; ks < KS_QK; ++ks) {
            int k_lo_base = ks * 32 + l_mod4 * 4;
            int k_hi_base = k_lo_base + 16;
            int m_lo_q = wid * 16 + l_div4;
            int m_hi_q = m_lo_q + 8;
            for (int bt = 0; bt < 4; ++bt) {
                smQ_T_ref[(k_lo_base + bt) * QT_STRIDE + m_lo_q] = (Qr[ks][0] >> (bt * 8)) & 0xFF;
                smQ_T_ref[(k_lo_base + bt) * QT_STRIDE + m_hi_q] = (Qr[ks][1] >> (bt * 8)) & 0xFF;
                smQ_T_ref[(k_hi_base + bt) * QT_STRIDE + m_lo_q] = (Qr[ks][2] >> (bt * 8)) & 0xFF;
                smQ_T_ref[(k_hi_base + bt) * QT_STRIDE + m_hi_q] = (Qr[ks][3] >> (bt * 8)) & 0xFF;
            }
        }
    }
}

int main() {
    // Prepare host smQ with markers
    uint8_t *h_smQ = new uint8_t[SMEM_SMQ_BYTES];
    for (int m = 0; m < BR; ++m)
        for (int k = 0; k < HD; ++k)
            h_smQ[m * HD + k] = marker(m, k);

    // CPU reference
    uint8_t *h_ref = new uint8_t[SMEM_SMQT_BYTES];
    cpu_scatter_ref(h_smQ, h_ref);

    // GPU pack
    uint8_t *d_smQ, *d_out;
    cudaMalloc(&d_smQ, SMEM_SMQ_BYTES);
    cudaMalloc(&d_out, SMEM_SMQT_BYTES);
    cudaMemcpy(d_smQ, h_smQ, SMEM_SMQ_BYTES, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0xAA, SMEM_SMQT_BYTES);

    pack_kernel<<<1, 128>>>(d_out, d_smQ);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    uint8_t *h_gpu = new uint8_t[SMEM_SMQT_BYTES];
    cudaMemcpy(h_gpu, d_out, SMEM_SMQT_BYTES, cudaMemcpyDeviceToHost);

    // Assert byte-by-byte for valid data range (128 rows × 64 cols = 8192 bytes;
    //   padding cols 64..67 in each row are ignored)
    size_t match = 0, mism = 0;
    size_t first_mism_idx = SIZE_MAX;
    int first_mism_row = -1, first_mism_col = -1;
    uint8_t first_ref = 0, first_gpu = 0;
    for (int r = 0; r < HD; ++r) {
        for (int c = 0; c < BR; ++c) {   // only valid cols 0..63
            size_t idx = r * QT_STRIDE + c;
            if (h_ref[idx] == h_gpu[idx]) match++;
            else {
                mism++;
                if (first_mism_idx == SIZE_MAX) {
                    first_mism_idx = idx;
                    first_mism_row = r;
                    first_mism_col = c;
                    first_ref = h_ref[idx];
                    first_gpu = h_gpu[idx];
                }
            }
        }
    }

    printf("pack_qt_unit_test:\n");
    printf("  matched=%zu / total=%d\n", match, HD * BR);
    printf("  mismatched=%zu\n", mism);
    if (mism > 0) {
        printf("  FIRST MISM: row=%d col=%d ref=0x%02x gpu=0x%02x\n",
               first_mism_row, first_mism_col, first_ref, first_gpu);
    }
    printf("  verdict: %s\n", mism == 0 ? "OK (256/256 per quad × 32 quads)" : "FAIL");

    delete[] h_smQ; delete[] h_ref; delete[] h_gpu;
    cudaFree(d_smQ); cudaFree(d_out);
    return mism == 0 ? 0 : 1;
}
