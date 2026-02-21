// QMMA.16832 Probe v2: Diagnose A fragment layout on SM89
// Build: nvcc -O3 -arch=sm_89 -std=c++17 qmma_probe2.cu -o qmma_probe2

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

static inline uint8_t f2e4m3(float f)
{
    if (f == 0.0f)
        return 0x00;
    uint8_t sign = (f < 0) ? 0x80 : 0;
    float af = fabsf(f);
    static const uint8_t tab[] = {
        0x00,
        0x38,
        0x40,
        0x44,
        0x48,
        0x4A,
        0x4C,
        0x4E,
        0x50,
        0x51,
        0x52,
        0x53,
        0x54,
        0x55,
        0x56,
        0x57,
    };
    int idx = (int)af;
    if (idx >= 0 && idx <= 15)
        return sign | tab[idx];
    return 0x00;
}

static inline float e4m3f(uint8_t v)
{
    int s = (v >> 7) & 1, e = (v >> 3) & 0xF, m = v & 0x7;
    if (e == 0xF && m == 0x7)
        return 0.0f / 0.0f;
    float r = (e == 0) ? ldexpf((float)m, -9) : ldexpf(1.0f + m / 8.0f, e - 7);
    return s ? -r : r;
}

static inline float fp16f(uint16_t h)
{
    __half hv;
    memcpy(&hv, &h, 2);
    return __half2float(hv);
}

// ==========================================================================
// Kernel 1: Standard PTX layout (group_id, group_id+8)
// Also dumps actual register values
// ==========================================================================
__global__ void probe_standard(
    const uint8_t *__restrict__ A, // 16x32
    const uint8_t *__restrict__ B, // 8x32
    uint32_t *D_out,               // 32*2
    uint32_t *A_out)               // 32*4 (dump actual a_frag values)
{
    int lane = threadIdx.x;
    int g = lane / 4, t = lane % 4;

    // Standard PTX ISA layout
    uint32_t a0 = *(uint32_t *)&A[(g) * 32 + t * 4];
    uint32_t a1 = *(uint32_t *)&A[(g) * 32 + t * 4 + 16];
    uint32_t a2 = *(uint32_t *)&A[(g + 8) * 32 + t * 4];
    uint32_t a3 = *(uint32_t *)&A[(g + 8) * 32 + t * 4 + 16];

    uint32_t b0 = *(uint32_t *)&B[g * 32 + t * 4];
    uint32_t b1 = *(uint32_t *)&B[g * 32 + t * 4 + 16];

    // Dump A regs
    A_out[lane * 4 + 0] = a0;
    A_out[lane * 4 + 1] = a1;
    A_out[lane * 4 + 2] = a2;
    A_out[lane * 4 + 3] = a3;

    uint32_t c0 = 0, c1 = 0, d0, d1;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1));

    D_out[lane * 2 + 0] = d0;
    D_out[lane * 2 + 1] = d1;
}

// ==========================================================================
// Kernel 2: Alternative layout (group_id*2, group_id*2+1)
// ==========================================================================
__global__ void probe_alt(
    const uint8_t *__restrict__ A,
    const uint8_t *__restrict__ B,
    uint32_t *D_out)
{
    int lane = threadIdx.x;
    int g = lane / 4, t = lane % 4;

    // Alternative: consecutive rows
    uint32_t a0 = *(uint32_t *)&A[(g * 2) * 32 + t * 4];
    uint32_t a1 = *(uint32_t *)&A[(g * 2) * 32 + t * 4 + 16];
    uint32_t a2 = *(uint32_t *)&A[(g * 2 + 1) * 32 + t * 4];
    uint32_t a3 = *(uint32_t *)&A[(g * 2 + 1) * 32 + t * 4 + 16];

    uint32_t b0 = *(uint32_t *)&B[g * 32 + t * 4];
    uint32_t b1 = *(uint32_t *)&B[g * 32 + t * 4 + 16];

    uint32_t c0 = 0, c1 = 0, d0, d1;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1));

    D_out[lane * 2 + 0] = d0;
    D_out[lane * 2 + 1] = d1;
}

// ==========================================================================
// Kernel 3: Try swapping a0/a1 with a2/a3 to test k-split hypothesis
// Maybe a0=k[0:3], a1=k[4:7]... all for same row?
// Layout: all 4 regs = 16 bytes = 16 k-positions for ONE row
// With 8 groups × 4 tids = 32 threads, each holding 16 k-elements of one row?
// That's only 8 rows... need d1 for the other 8.
// ==========================================================================
__global__ void probe_ksplit(
    const uint8_t *__restrict__ A,
    const uint8_t *__restrict__ B,
    uint32_t *D_out)
{
    int lane = threadIdx.x;
    int g = lane / 4, t = lane % 4;

    // k-split: each thread holds 16 consecutive k-elements for ONE row
    // a0 = A[g][t*8+0 : t*8+3]
    // a1 = A[g][t*8+4 : t*8+7]
    // a2 = A[g+8][t*8+0 : t*8+3]
    // a3 = A[g+8][t*8+4 : t*8+7]
    uint32_t a0 = *(uint32_t *)&A[g * 32 + t * 8];
    uint32_t a1 = *(uint32_t *)&A[g * 32 + t * 8 + 4];
    uint32_t a2 = *(uint32_t *)&A[(g + 8) * 32 + t * 8];
    uint32_t a3 = *(uint32_t *)&A[(g + 8) * 32 + t * 8 + 4];

    uint32_t b0 = *(uint32_t *)&B[g * 32 + t * 8];
    uint32_t b1 = *(uint32_t *)&B[g * 32 + t * 8 + 4];

    uint32_t c0 = 0, c1 = 0, d0, d1;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1));

    D_out[lane * 2 + 0] = d0;
    D_out[lane * 2 + 1] = d1;
}

void print_D(const char *label, uint32_t *h_D)
{
    printf("\n%s:\n", label);
    printf("lane  g  t    d0_lo   d0_hi   d1_lo   d1_hi\n");
    for (int lane = 0; lane < 32; lane++)
    {
        int g = lane / 4, t = lane % 4;
        uint32_t d0 = h_D[lane * 2], d1 = h_D[lane * 2 + 1];
        printf("%4d %2d %2d  %7.1f %7.1f %7.1f %7.1f\n",
               lane, g, t,
               fp16f(d0 & 0xFFFF), fp16f(d0 >> 16),
               fp16f(d1 & 0xFFFF), fp16f(d1 >> 16));
    }
}

int main()
{
    printf("=== QMMA.16832 Probe v2 ===\n");

    // Setup: A[m][0] = m+1, B[n][0] = 1.0
    // Expected D[m][n] = m+1 for all n
    uint8_t hA[16 * 32] = {}, hB[8 * 32] = {};
    for (int m = 0; m < 16; m++)
        hA[m * 32 + 0] = f2e4m3((float)(m + 1));
    for (int n = 0; n < 8; n++)
        hB[n * 32 + 0] = f2e4m3(1.0f);

    printf("A values at k=0: ");
    for (int m = 0; m < 16; m++)
        printf("A[%d]=0x%02X(%.0f) ", m, hA[m * 32], e4m3f(hA[m * 32]));
    printf("\n");

    void *dA, *dB;
    uint32_t *dD, *dD2, *dD3, *dAr;
    cudaMalloc(&dA, 16 * 32);
    cudaMalloc(&dB, 8 * 32);
    cudaMalloc(&dD, 64 * 4);
    cudaMalloc(&dD2, 64 * 4);
    cudaMalloc(&dD3, 64 * 4);
    cudaMalloc(&dAr, 128 * 4);
    cudaMemcpy(dA, hA, 16 * 32, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 8 * 32, cudaMemcpyHostToDevice);

    uint32_t hD[64], hD2[64], hD3[64], hAr[128];

    // Test 1: Standard layout
    probe_standard<<<1, 32>>>((uint8_t *)dA, (uint8_t *)dB, dD, dAr);
    cudaDeviceSynchronize();
    cudaMemcpy(hD, dD, 64 * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hAr, dAr, 128 * 4, cudaMemcpyDeviceToHost);

    printf("\n--- A register dump (standard layout) ---\n");
    printf("lane  g  t     a0(hex)     a1(hex)     a2(hex)     a3(hex)\n");
    for (int lane = 0; lane < 8; lane++)
    {
        int g = lane / 4, t = lane % 4;
        printf("%4d %2d %2d  %08X    %08X    %08X    %08X\n",
               lane, g, t,
               hAr[lane * 4 + 0], hAr[lane * 4 + 1], hAr[lane * 4 + 2], hAr[lane * 4 + 3]);
    }
    printf("  ...\n");
    for (int lane = 28; lane < 32; lane++)
    {
        int g = lane / 4, t = lane % 4;
        printf("%4d %2d %2d  %08X    %08X    %08X    %08X\n",
               lane, g, t,
               hAr[lane * 4 + 0], hAr[lane * 4 + 1], hAr[lane * 4 + 2], hAr[lane * 4 + 3]);
    }

    // Decode a few A regs
    printf("\nA reg decode (lane 0, group=0, tid=0):\n");
    printf("  a0 = 0x%08X → bytes: ", hAr[0]);
    for (int b = 0; b < 4; b++)
        printf("0x%02X(%.0f) ", (hAr[0] >> (b * 8)) & 0xFF, e4m3f((hAr[0] >> (b * 8)) & 0xFF));
    printf("\n  → Should be A[0][0:3] = {1, 0, 0, 0}\n");
    printf("  a2 = 0x%08X → bytes: ", hAr[2]);
    for (int b = 0; b < 4; b++)
        printf("0x%02X(%.0f) ", (hAr[2] >> (b * 8)) & 0xFF, e4m3f((hAr[2] >> (b * 8)) & 0xFF));
    printf("\n  → Should be A[8][0:3] = {9, 0, 0, 0}\n");

    print_D("Standard (g, g+8)", hD);

    // Test 2: Alternative layout (g*2, g*2+1)
    probe_alt<<<1, 32>>>((uint8_t *)dA, (uint8_t *)dB, dD2);
    cudaDeviceSynchronize();
    cudaMemcpy(hD2, dD2, 64 * 4, cudaMemcpyDeviceToHost);
    print_D("Alt (g*2, g*2+1)", hD2);

    // Test 3: K-split layout
    probe_ksplit<<<1, 32>>>((uint8_t *)dA, (uint8_t *)dB, dD3);
    cudaDeviceSynchronize();
    cudaMemcpy(hD3, dD3, 64 * 4, cudaMemcpyDeviceToHost);
    print_D("K-split (t*8)", hD3);

    // Summary
    printf("\n=== SUMMARY ===\n");
    printf("Expected: d0_lo = d0_hi = row+1, d1_lo = d1_hi = row2+1\n");
    printf("Standard: d0=%.0f d1=%.0f (lane0)\n",
           fp16f(hD[0] & 0xFFFF), fp16f(hD[1] & 0xFFFF));
    printf("Alt:      d0=%.0f d1=%.0f (lane0)\n",
           fp16f(hD2[0] & 0xFFFF), fp16f(hD2[1] & 0xFFFF));
    printf("K-split:  d0=%.0f d1=%.0f (lane0)\n",
           fp16f(hD3[0] & 0xFFFF), fp16f(hD3[1] & 0xFFFF));

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dD);
    cudaFree(dD2);
    cudaFree(dD3);
    cudaFree(dAr);
    return 0;
}
