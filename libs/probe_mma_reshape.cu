// =====================================================================
//  probe_mma_reshape.cu — empirical test of MMA #2 reshape claim.
//
//  Paper claim: MMA #2 (dK = dS^T · Q, sum over i = Br) FORCES Q in
//  transposed [d][i] orientation in smem because K-axis of MMA = Br
//  must align with Q's row dimension. No reshape (incl. "dK^T = Q^T · dS")
//  removes Q-transpose requirement — that just shifts which OPERAND holds
//  Q^T (still transposed somewhere).
//
//  Empirical test: small MMA with 3 arrangements, FP64 golden verifier.
//
//  V0: standard A=dS^T from smdST [j][i], B=Q^T from smQ_T [d][i]
//      Both smem operands TRANSPOSED. SANITY — must PASS.
//  V1: alternative dK^T=Q^T·dS, A=Q^T from smQ_T [d][i], B=dS^T from smdST [j][i]
//      Still requires Q in transposed smem [d][i]. SHOULD PASS — proves arrangement
//      is mathematically equivalent (just different output ordering).
//  V2: dK^T attempt with Q in NATURAL [i][d] smem, A reads as if [d][i]
//      Reads wrong bytes (Q[d,i] instead of Q[i,d]). EXPECTED FAIL.
//      Verifies paper: natural Q can't serve as Q^T operand without transpose.
//
//  Binary verdict:
//    V0 + V1 PASS, V2 FAIL → paper confirmed: Q-transpose mathematically required
//    V2 PASS → paper math wrong, LIVE lever
//
//  Geometry: Br=Bc=Hd=32 (each one MMA M-tile=16 × N-tile=8 × K-tile=32 wait...)
//  For m16n8k32: M=16 fragment, N=8 fragment, K=32 fragment.
//  dK = dS^T·Q has M=Bc, N=Hd, K=Br. For single MMA: Bc=16, Hd=8, Br=32. 1 warp.
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

#define BR 32      // K-axis of MMA #2 (= Q's row dim = sum dim)
#define BC 16      // M-axis of MMA #2 (= dK's row dim)
#define HD 8       // N-axis of MMA #2 (= dK's col dim)
#define THREADS 32

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) {                \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                    \
            cudaGetErrorString(e)); std::exit(1); }} while (0)

__device__ __forceinline__ void mma_e4m3_f32(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// FP8 e4m3 conversions (simplified, sufficient for our deterministic test values)
static inline uint8_t f_to_e4m3(float f) {
    if (f == 0.0f) return 0x00;
    bool sign = f < 0;
    f = fabsf(f);
    if (f > 448.0f) f = 448.0f;
    int exp = (int)floorf(log2f(f));
    if (exp < -9) return 0x00;
    if (exp < -6) {
        int m = (int)(f / powf(2.0f, -9) + 0.5f);
        if (m > 7) m = 7;
        return ((sign ? 0x80 : 0) | m);
    }
    int be = exp + 7;
    if (be > 15) be = 15;
    float mf = f / powf(2.0f, exp) - 1.0f;
    int m = (int)(mf * 8 + 0.5f);
    if (m >= 8) { m -= 8; be += 1; if (be > 15) be = 15; }
    return ((sign ? 0x80 : 0) | ((be & 0xF) << 3) | (m & 0x7));
}
static inline float e4m3_to_f(uint8_t u) {
    int sign = (u >> 7) & 1;
    int exp = (u >> 3) & 0xF;
    int m = u & 0x7;
    float v;
    if (exp == 0) v = (m / 8.0f) * powf(2.0f, -6);
    else v = (1.0f + m / 8.0f) * powf(2.0f, exp - 7);
    return sign ? -v : v;
}

// =====================================================================
// V0 STANDARD: A = dS^T from smdST [Bc][Br], B = Q^T from smQ_T [Hd][Br]
// Both operands TRANSPOSED in smem (current dK arrangement).
// Computes dK[j, d] directly.
// =====================================================================
__global__ void mma_v0_standard(
    const uint8_t *smdST_in,   // [Bc=16][Br=32] = 512 bytes (dS^T)
    const uint8_t *smQT_in,    // [Hd=8][Br=32] = 256 bytes (Q^T)
    float *out)                // [Bc=16][Hd=8] = 128 floats (dK)
{
    extern __shared__ uint8_t smem[];
    uint8_t *smA = smem;             // dS^T
    uint8_t *smB = smem + 512;       // Q^T

    int tid = threadIdx.x;
    for (int e = tid; e < 512; e += THREADS) smA[e] = smdST_in[e];
    for (int e = tid; e < 256; e += THREADS) smB[e] = smQT_in[e];
    __syncwarp();

    int g = tid >> 2;     // 0..7
    int p = tid & 3;      // 0..3

    // A: row-major [M=Bc=16][K=Br=32]
    int m_lo = g + 0;      // 0..7
    int m_hi = g + 8;      // 8..15 (within Bc=16)
    int k_lo = p * 4 + 0;  // 0,4,8,12
    int k_hi = p * 4 + 16; // 16,20,24,28
    uint32_t A0 = *(uint32_t*)&smA[m_lo * BR + k_lo];
    uint32_t A1 = *(uint32_t*)&smA[m_hi * BR + k_lo];
    uint32_t A2 = *(uint32_t*)&smA[m_lo * BR + k_hi];
    uint32_t A3 = *(uint32_t*)&smA[m_hi * BR + k_hi];

    // B: col-major K=Br=32, N=Hd=8 stored [N][K]
    int n = g;             // 0..7 covers Hd=8
    uint32_t B0 = *(uint32_t*)&smB[n * BR + k_lo];
    uint32_t B1 = *(uint32_t*)&smB[n * BR + k_hi];

    float c0=0, c1=0, c2=0, c3=0;
    mma_e4m3_f32(c0, c1, c2, c3, A0, A1, A2, A3, B0, B1, 0.f, 0.f, 0.f, 0.f);

    // C layout: per lane stores 4 floats at (g, p*2), (g, p*2+1), (g+8, p*2), (g+8, p*2+1)
    int cm_lo = g + 0;
    int cm_hi = g + 8;
    int cn_lo = p * 2 + 0;
    int cn_hi = p * 2 + 1;
    out[cm_lo * HD + cn_lo] = c0;
    out[cm_lo * HD + cn_hi] = c1;
    out[cm_hi * HD + cn_lo] = c2;
    out[cm_hi * HD + cn_hi] = c3;
}

// =====================================================================
// V1 ALTERNATIVE: A = Q^T from smQ_T [Hd=8][Br=32], B = dS^T from smdST [Bc=16][Br=32]
// Computes dK^T = Q^T · dS — produces transposed result [Hd][Bc].
// Still requires Q TRANSPOSED in smem (smQ_T [d][i]). Just swaps operand roles.
// Output: dK^T[d, j] → need to transpose at write to produce dK[j, d].
//
// Adjustments: For m16n8k32, M=16. Hd=8 < 16. Use Hd=16 in this probe variant
// (separate input) or stick to 8 and use only half M-tile. Choose half-tile.
//
// For Hd=8 fits in half M-tile (g=0..7), the other half (g=8..15) is unused.
// =====================================================================
__global__ void mma_v1_dkT(
    const uint8_t *smdS_in,    // [Br=32][Bc=16] natural dS = 512 bytes (B operand col-major stored = dS^T row-major = need smdST!)
    const uint8_t *smQT_in,    // [Hd=8][Br=32] = 256 bytes (Q^T)
    float *out)                // [Hd=8][Bc=16] = 128 floats (dK^T)
{
    extern __shared__ uint8_t smem[];
    uint8_t *smA = smem;          // Q^T   (A in this arrangement)
    uint8_t *smB = smem + 256;    // dS    (B in this arrangement = col-major K=i, N=j stored [j][i] = dS^T)

    int tid = threadIdx.x;
    for (int e = tid; e < 256; e += THREADS) smA[e] = smQT_in[e];   // Q^T as A
    // Compute dS^T into smB from natural dS at load time (we have natural dS_in)
    // dS^T[j, i] = dS[i, j], shape [Bc][Br]
    for (int e = tid; e < BC * BR; e += THREADS) {
        int j_b = e / BR;
        int i_b = e % BR;
        smB[e] = smdS_in[i_b * BC + j_b];   // dS[i, j] → smB[j, i]
    }
    __syncwarp();

    int g = tid >> 2;
    int p = tid & 3;

    // A: row-major [M=Hd=8][K=Br=32]. Hd=8, so only g=0..7 valid in M-axis.
    // For lane t: A0/A1 at m=g+0, m=g+8. m=g+8 OOB for Hd=8 → zero out.
    int m_lo = g + 0;
    int m_hi = g + 8;
    int k_lo = p * 4 + 0;
    int k_hi = p * 4 + 16;
    uint32_t A0 = (m_lo < HD) ? *(uint32_t*)&smA[m_lo * BR + k_lo] : 0;
    uint32_t A1 = (m_hi < HD) ? *(uint32_t*)&smA[m_hi * BR + k_lo] : 0;
    uint32_t A2 = (m_lo < HD) ? *(uint32_t*)&smA[m_lo * BR + k_hi] : 0;
    uint32_t A3 = (m_hi < HD) ? *(uint32_t*)&smA[m_hi * BR + k_hi] : 0;

    // B: col-major K=Br=32, N=Bc=16 stored [N=j][K=i] = dS^T row-major. Stride BR=32.
    int n = g;     // 0..7. But N=Bc=16, need n up to 15. Per lane covers only n=0..7.
                   // For half-N coverage. The other half (n=8..15) by other MMA issue.
                   // For minimal probe, just compute first half (n=0..7) and verify.
    uint32_t B0 = *(uint32_t*)&smB[n * BR + k_lo];
    uint32_t B1 = *(uint32_t*)&smB[n * BR + k_hi];

    float c0=0, c1=0, c2=0, c3=0;
    mma_e4m3_f32(c0, c1, c2, c3, A0, A1, A2, A3, B0, B1, 0.f, 0.f, 0.f, 0.f);

    // C: [M=Hd=8][N=Bc, half=8] but here only N=0..7 covered. Output dK^T[d, j].
    int cm_lo = g + 0;
    int cm_hi = g + 8;
    int cn_lo = p * 2 + 0;
    int cn_hi = p * 2 + 1;
    if (cm_lo < HD) {
        out[cm_lo * BC + cn_lo] = c0;
        out[cm_lo * BC + cn_hi] = c1;
    }
    // cm_hi = g+8, max 15, all >= 8 OOB for Hd=8 → skip
}

// =====================================================================
// V2: Try V1 with Q in NATURAL [i][d] smem (no transpose).
// Read smA at same addresses as V1 (interpreted as [d][i]).
// Expected: bytes at smA[m * BR + k] in natural layout = Q[i=m, d=k] — but we WANT Q^T[m, k] = Q[k, m].
// → WRONG bytes → garbage dK^T.
// =====================================================================
__global__ void mma_v2_natural_q(
    const uint8_t *smdS_in,
    const uint8_t *smQ_natural,  // [Br=32][Hd=8] = 256 bytes Q natural
    float *out)
{
    extern __shared__ uint8_t smem[];
    uint8_t *smA = smem;
    uint8_t *smB = smem + 256;

    int tid = threadIdx.x;
    // Load Q natural to smA. Natural layout: [i][d] = 32 × 8 = 256 bytes
    // But A address pattern expects stride BR=32 between "rows". Natural stride = HD=8.
    // To make smA[m*BR + k] valid addressing, we need to lay out smA with stride BR=32.
    // Pad natural Q to stride BR: smA_padded[i][BR] where bytes 0..HD-1 hold Q[i, *] and bytes HD..BR-1 = 0.
    // Or just access natural Q at smA[m*HD + k]? That changes stride.
    //
    // CRITICAL DESIGN CHOICE: the test must use SAME addressing pattern as V1 to be
    // honest comparison. V1 reads smA[m*BR + k]. V2 also reads smA[m*BR + k] but smA
    // stores Q natural with stride BR=32. So smA[m*BR + k] = stored byte at offset m*32 + k.
    //
    // We "fake" the natural-Q smem by writing Q[i, d] into smA[i * BR + d] (padded to stride BR).
    // Then lane reads smA[m * BR + k] = byte at offset m*32 + k = Q[i=m, d=k].
    // Compared to V1: V1 has Q^T at smA, so smA[m*BR + k] = Q^T[m, k] = Q[k, m].
    // V2 gives Q[m, k] instead → WRONG for MMA expecting Q^T values.

    for (int e = tid; e < 32 * BR; e += THREADS) smA[e] = 0;     // zero pad to stride BR
    __syncwarp();
    for (int e = tid; e < BR * HD; e += THREADS) {
        int i_b = e / HD;
        int d_b = e % HD;
        smA[i_b * BR + d_b] = smQ_natural[e];   // Q[i, d] → smA[i*BR + d] padded
    }
    for (int e = tid; e < BC * BR; e += THREADS) {
        int j_b = e / BR;
        int i_b = e % BR;
        smB[e] = smdS_in[i_b * BC + j_b];   // dS^T
    }
    __syncwarp();

    int g = tid >> 2;
    int p = tid & 3;

    // Read A at SAME pattern as V1 (interpreting smA as if [d][i])
    int m_lo = g + 0;
    int m_hi = g + 8;
    int k_lo = p * 4 + 0;
    int k_hi = p * 4 + 16;

    // For natural Q stored at smA[i*BR + d] (padded), reading at smA[m*BR + k]:
    //   = Q[i=m, d=k] for m < BR and k < HD; padded zero otherwise.
    // For HD=8: only k < 8 has Q data. k_lo=0..12, k_hi=16..28 — k_hi >= 16 always OOB → zero.
    // k_lo=0,4,8 in valid range; k_lo=12 → reads byte at smA[m*32+12] = padded 0
    //   (because HD=8, d>=8 is padding).

    uint32_t A0 = (m_lo < 32) ? *(uint32_t*)&smA[m_lo * BR + k_lo] : 0;
    uint32_t A1 = (m_hi < 32) ? *(uint32_t*)&smA[m_hi * BR + k_lo] : 0;
    uint32_t A2 = 0;  // k_hi >= 16 always padded zero
    uint32_t A3 = 0;

    int n = g;
    uint32_t B0 = *(uint32_t*)&smB[n * BR + k_lo];
    uint32_t B1 = *(uint32_t*)&smB[n * BR + k_hi];

    float c0=0, c1=0, c2=0, c3=0;
    mma_e4m3_f32(c0, c1, c2, c3, A0, A1, A2, A3, B0, B1, 0.f, 0.f, 0.f, 0.f);

    int cm_lo = g + 0;
    int cm_hi = g + 8;
    int cn_lo = p * 2 + 0;
    int cn_hi = p * 2 + 1;
    if (cm_lo < HD) {
        out[cm_lo * BC + cn_lo] = c0;
        out[cm_lo * BC + cn_hi] = c1;
    }
}

int main()
{
    // Deterministic small inputs (FP8 e4m3)
    uint8_t dS_h[BR * BC];
    for (int i = 0; i < BR; ++i)
        for (int j = 0; j < BC; ++j)
            dS_h[i * BC + j] = f_to_e4m3(0.05f * ((i + 2 * j + 3) % 11) - 0.2f);

    uint8_t Q_h[BR * HD];
    for (int i = 0; i < BR; ++i)
        for (int d = 0; d < HD; ++d)
            Q_h[i * HD + d] = f_to_e4m3(0.1f * ((3 * i + d + 1) % 7) - 0.25f);

    // Transposed versions
    uint8_t dST_h[BC * BR];
    for (int j = 0; j < BC; ++j)
        for (int i = 0; i < BR; ++i)
            dST_h[j * BR + i] = dS_h[i * BC + j];

    uint8_t QT_h[HD * BR];
    for (int d = 0; d < HD; ++d)
        for (int i = 0; i < BR; ++i)
            QT_h[d * BR + i] = Q_h[i * HD + d];

    // FP64 golden dK[j, d] = sum_i dS[i, j] * Q[i, d]
    double dK_gold[BC * HD];
    for (int j = 0; j < BC; ++j)
        for (int d = 0; d < HD; ++d) {
            double acc = 0.0;
            for (int i = 0; i < BR; ++i)
                acc += e4m3_to_f(dS_h[i * BC + j]) * e4m3_to_f(Q_h[i * HD + d]);
            dK_gold[j * HD + d] = acc;
        }

    // dK^T golden for V1/V2
    double dKT_gold[HD * BC];
    for (int d = 0; d < HD; ++d)
        for (int j = 0; j < BC; ++j)
            dKT_gold[d * BC + j] = dK_gold[j * HD + d];

    uint8_t *dST_d, *QT_d, *dS_d, *Q_d;
    float *out_d;
    CK(cudaMalloc(&dST_d, BC * BR));
    CK(cudaMalloc(&QT_d,  HD * BR));
    CK(cudaMalloc(&dS_d,  BR * BC));
    CK(cudaMalloc(&Q_d,   BR * HD));
    CK(cudaMalloc(&out_d, 256 * sizeof(float)));   // generous
    CK(cudaMemcpy(dST_d, dST_h, BC * BR, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(QT_d,  QT_h,  HD * BR, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dS_d,  dS_h,  BR * BC, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Q_d,   Q_h,   BR * HD, cudaMemcpyHostToDevice));

    const int smem_bytes = 2048;
    CK(cudaFuncSetAttribute(mma_v0_standard, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    CK(cudaFuncSetAttribute(mma_v1_dkT,      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    CK(cudaFuncSetAttribute(mma_v2_natural_q,cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    // === V0 ===
    CK(cudaMemset(out_d, 0, 256 * sizeof(float)));
    mma_v0_standard<<<1, THREADS, smem_bytes>>>(dST_d, QT_d, out_d);
    CK(cudaDeviceSynchronize());
    float out0[BC * HD];
    CK(cudaMemcpy(out0, out_d, BC * HD * sizeof(float), cudaMemcpyDeviceToHost));

    {
        double max_err = 0;
        int pass = 0;
        for (int k = 0; k < BC * HD; ++k) {
            double err = fabs(out0[k] - dK_gold[k]);
            if (err < 0.05) pass++;
            if (err > max_err) max_err = err;
        }
        printf("V0 STANDARD (A=dS^T [j][i], B=Q^T [d][i]):\n");
        printf("  pass within FP8 floor (0.05): %d/%d (%.1f%%)  max_err: %.4f\n",
               pass, BC * HD, 100.0 * pass / (BC * HD), max_err);
        printf("  out[0..7]: ");
        for (int k = 0; k < 8; ++k) printf("%+.4f ", out0[k]);
        printf("\n  golden:    ");
        for (int k = 0; k < 8; ++k) printf("%+.4f ", dK_gold[k]);
        printf("\n\n");
    }

    // === V1 ===
    CK(cudaMemset(out_d, 0, 256 * sizeof(float)));
    mma_v1_dkT<<<1, THREADS, smem_bytes>>>(dS_d, QT_d, out_d);
    CK(cudaDeviceSynchronize());
    float out1[HD * BC];
    CK(cudaMemcpy(out1, out_d, HD * BC * sizeof(float), cudaMemcpyDeviceToHost));
    {
        double max_err = 0;
        int pass = 0;
        for (int k = 0; k < HD * BC; ++k) {
            double err = fabs(out1[k] - dKT_gold[k]);
            if (err < 0.05) pass++;
            if (err > max_err) max_err = err;
        }
        printf("V1 ALTERNATIVE dK^T (A=Q^T [d][i] transposed smem, B=dS^T):\n");
        printf("  pass within FP8 floor (0.05): %d/%d (%.1f%%)  max_err: %.4f\n",
               pass, HD * BC, 100.0 * pass / (HD * BC), max_err);
        printf("  out[0..7]: ");
        for (int k = 0; k < 8; ++k) printf("%+.4f ", out1[k]);
        printf("\n  dK^T gold: ");
        for (int k = 0; k < 8; ++k) printf("%+.4f ", dKT_gold[k]);
        printf("\n\n");
    }

    // === V2 ===
    CK(cudaMemset(out_d, 0, 256 * sizeof(float)));
    mma_v2_natural_q<<<1, THREADS, smem_bytes>>>(dS_d, Q_d, out_d);
    CK(cudaDeviceSynchronize());
    float out2[HD * BC];
    CK(cudaMemcpy(out2, out_d, HD * BC * sizeof(float), cudaMemcpyDeviceToHost));
    {
        double max_err = 0;
        int pass = 0;
        for (int k = 0; k < HD * BC; ++k) {
            double err = fabs(out2[k] - dKT_gold[k]);
            if (err < 0.05) pass++;
            if (err > max_err) max_err = err;
        }
        printf("V2 NATURAL-Q ATTEMPT (A from natural Q [i][d], read as if [d][i]):\n");
        printf("  pass within FP8 floor (0.05): %d/%d (%.1f%%)  max_err: %.4f\n",
               pass, HD * BC, 100.0 * pass / (HD * BC), max_err);
        printf("  out[0..7]: ");
        for (int k = 0; k < 8; ++k) printf("%+.4f ", out2[k]);
        printf("\n  dK^T gold: ");
        for (int k = 0; k < 8; ++k) printf("%+.4f ", dKT_gold[k]);
        printf("\n\n");
    }

    CK(cudaFree(dST_d)); CK(cudaFree(QT_d)); CK(cudaFree(dS_d));
    CK(cudaFree(Q_d)); CK(cudaFree(out_d));
    return 0;
}
