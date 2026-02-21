// QMMA.16832 Probe v3: Definitive layout discovery
// Build: nvcc -O3 -arch=sm_89 -std=c++17 qmma_probe3.cu -o qmma_probe3

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
    int idx = (int)(af + 0.5f);
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

// ============================================================================
// Kernel: MMA with direct global memory loads
// ============================================================================
__global__ void mma_direct(
    const uint8_t *__restrict__ A,
    const uint8_t *__restrict__ B,
    uint32_t *D_out)
{
    int lane = threadIdx.x;
    int g = lane / 4, t = lane % 4;

    uint32_t a0 = *(uint32_t *)&A[g * 32 + t * 4];
    uint32_t a1 = *(uint32_t *)&A[g * 32 + t * 4 + 16];
    uint32_t a2 = *(uint32_t *)&A[(g + 8) * 32 + t * 4];
    uint32_t a3 = *(uint32_t *)&A[(g + 8) * 32 + t * 4 + 16];
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

// ============================================================================
// Kernel: MMA through shared memory (same as GEMM kernel loads)
// ============================================================================
__global__ void mma_smem(
    const uint8_t *__restrict__ A,
    const uint8_t *__restrict__ B,
    uint32_t *D_out,
    uint32_t *Bfrag_out)
{
    __shared__ uint8_t sA[16 * 80]; // stride 80
    __shared__ uint8_t sB[8 * 80];

    int lane = threadIdx.x;
    int g = lane / 4, t = lane % 4;

    // Load A: 16 rows × 32 cols, stride 80 in smem
    // 32 threads, load 16 bytes each = 512 bytes = 16×32
    {
        int row = lane / 2;        // 0..15
        int col = (lane % 2) * 16; // 0 or 16
        *(uint4 *)&sA[row * 80 + col] = *(uint4 *)&A[row * 32 + col];
    }

    // Load B: 8 rows × 32 cols, stride 80
    if (lane < 16)
    {
        int row = lane / 2;        // 0..7
        int col = (lane % 2) * 16; // 0 or 16
        *(uint4 *)&sB[row * 80 + col] = *(uint4 *)&B[row * 32 + col];
    }

    __syncthreads();

    // Load fragments from smem (EXACT same pattern as GEMM kernel)
    uint32_t a0 = *(uint32_t *)&sA[(g) * 80 + t * 4];
    uint32_t a1 = *(uint32_t *)&sA[(g) * 80 + t * 4 + 16];
    uint32_t a2 = *(uint32_t *)&sA[(g + 8) * 80 + t * 4];
    uint32_t a3 = *(uint32_t *)&sA[(g + 8) * 80 + t * 4 + 16];
    uint32_t b0 = *(uint32_t *)&sB[(g) * 80 + t * 4];
    uint32_t b1 = *(uint32_t *)&sB[(g) * 80 + t * 4 + 16];

    if (Bfrag_out)
    {
        Bfrag_out[lane * 2 + 0] = b0;
        Bfrag_out[lane * 2 + 1] = b1;
    }

    uint32_t c0 = 0, c1 = 0, d0, d1;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1));

    D_out[lane * 2 + 0] = d0;
    D_out[lane * 2 + 1] = d1;
}

void print_D(const char *label, uint32_t *h, int n)
{
    printf("%s:\n", label);
    printf("lane  g  t    d0_lo   d0_hi   d1_lo   d1_hi\n");
    for (int i = 0; i < n; i++)
    {
        printf("%4d %2d %2d  %7.1f %7.1f %7.1f %7.1f\n",
               i, i / 4, i % 4,
               fp16f(h[i * 2] & 0xFFFF), fp16f(h[i * 2] >> 16),
               fp16f(h[i * 2 + 1] & 0xFFFF), fp16f(h[i * 2 + 1] >> 16));
    }
}

int main()
{
    printf("=== QMMA.16832 Probe v3 — Definitive ===\n");

    void *dA, *dB;
    uint32_t *dD, *dBf;
    cudaMalloc(&dA, 16 * 32);
    cudaMalloc(&dB, 8 * 32);
    cudaMalloc(&dD, 64 * 4);
    cudaMalloc(&dBf, 64 * 4);

    uint8_t hA[16 * 32], hB[8 * 32];
    uint32_t hD[64], hBf[64];

    // ========================================================================
    // TEST 1: d1 test — rows 0-7 get value 1, rows 8-15 get value 2
    // ALL k positions filled, B = 1 everywhere
    // Expected: D[0..7][n] = 1*1*32 = 32, D[8..15][n] = 2*1*32 = 64
    // ========================================================================
    printf("\n--- TEST 1: d1 with rows 0-7=1, rows 8-15=2, B=1 ---\n");
    printf("Expected: d0→32 (rows 0-7), d1→64 (rows 8-15)\n\n");

    for (int m = 0; m < 8; m++)
        for (int k = 0; k < 32; k++)
            hA[m * 32 + k] = f2e4m3(1.0f);
    for (int m = 8; m < 16; m++)
        for (int k = 0; k < 32; k++)
            hA[m * 32 + k] = f2e4m3(2.0f);
    for (int n = 0; n < 8; n++)
        for (int k = 0; k < 32; k++)
            hB[n * 32 + k] = f2e4m3(1.0f);

    cudaMemcpy(dA, hA, 16 * 32, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 8 * 32, cudaMemcpyHostToDevice);

    mma_direct<<<1, 32>>>((uint8_t *)dA, (uint8_t *)dB, dD);
    cudaDeviceSynchronize();
    cudaMemcpy(hD, dD, 64 * 4, cudaMemcpyDeviceToHost);
    print_D("Direct", hD, 32);

    // ========================================================================
    // TEST 2: Column detection — direct vs smem
    // A[m][0]=1, B[n][0]=n+1 (only k=0 nonzero)
    // ========================================================================
    printf("\n--- TEST 2: Column detect, direct vs smem ---\n");
    printf("A[m][0]=1; B[n][0]=n+1. Expected: D[m][n]=n+1\n\n");

    memset(hA, 0, sizeof(hA));
    memset(hB, 0, sizeof(hB));
    for (int m = 0; m < 16; m++)
        hA[m * 32 + 0] = f2e4m3(1.0f);
    for (int n = 0; n < 8; n++)
        hB[n * 32 + 0] = f2e4m3((float)(n + 1));

    cudaMemcpy(dA, hA, 16 * 32, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 8 * 32, cudaMemcpyHostToDevice);

    mma_direct<<<1, 32>>>((uint8_t *)dA, (uint8_t *)dB, dD);
    cudaDeviceSynchronize();
    cudaMemcpy(hD, dD, 64 * 4, cudaMemcpyDeviceToHost);
    print_D("Direct col", hD, 8);

    mma_smem<<<1, 32>>>((uint8_t *)dA, (uint8_t *)dB, dD, dBf);
    cudaDeviceSynchronize();
    cudaMemcpy(hD, dD, 64 * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hBf, dBf, 64 * 4, cudaMemcpyDeviceToHost);
    print_D("Smem col", hD, 8);

    // Verify B fragments match between direct and smem
    printf("\nB frag comparison (smem, first 8 lanes):\n");
    for (int lane = 0; lane < 8; lane++)
    {
        uint32_t b0 = hBf[lane * 2];
        printf("  lane %d (g=%d,t=%d): b0=", lane, lane / 4, lane % 4);
        for (int i = 0; i < 4; i++)
            printf("%.0f ", e4m3f((b0 >> (i * 8)) & 0xFF));
        printf("\n");
    }

    // ========================================================================
    // TEST 3: Full 16x8 computation with unique per-element values
    // A[m][k] = m+1 only at k=0, rest 0
    // B[n][k] = n*16 + 1 only at k=0, rest 0 (for unique products)
    // D[m][n] = (m+1) * (n*16+1), all distinct
    // ========================================================================
    printf("\n--- TEST 3: Unique values per element ---\n");

    memset(hA, 0, sizeof(hA));
    memset(hB, 0, sizeof(hB));
    // Use small distinct values
    // A[m][0] = m+1 (1..16), but 16 overflows our table
    // Use A[m][0] = 1 for all m, B[n][0] = distinct powers
    // Better: A[m][0] = (m%4)+1 so rows 0,4,8,12 have same A value
    //         B[n][0] = n+1

    // Actually simpler: use A[m][0] values 1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8
    // and B[n][0] = 1 for all n
    // Then D[m][n] = A[m][0]*1 = A[m][0]
    // d0 for group g: D[g][*] = g+1  (rows 0-7)
    // d1 for group g: D[g+8][*] = (g+8)%8+1 = g+1 (same as d0!)
    // Can't distinguish d0 and d1 this way!

    // Use: A rows 0-7 have value = m+1, rows 8-15 have value = (m-8+1)*10
    // A[0..7][0] = 1,2,3,4,5,6,7,8
    // A[8..15][0] = 10,0,0,0,0,0,0,0 (only 10 fits our table)
    // Actually table goes to 15, so A[8][0]=10

    for (int m = 0; m < 8; m++)
        hA[m * 32 + 0] = f2e4m3((float)(m + 1));
    for (int m = 8; m < 16 && m - 8 + 9 <= 15; m++)
        hA[m * 32 + 0] = f2e4m3((float)(m - 8 + 9));
    // A[8]=9, A[9]=10, A[10]=11, A[11]=12, A[12]=13, A[13]=14, A[14]=15, A[15]=0 (16 overflows)
    for (int n = 0; n < 8; n++)
        hB[n * 32 + 0] = f2e4m3(1.0f);

    cudaMemcpy(dA, hA, 16 * 32, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 8 * 32, cudaMemcpyHostToDevice);

    printf("A values: ");
    for (int m = 0; m < 16; m++)
        printf("A[%d]=%.0f ", m, e4m3f(hA[m * 32]));
    printf("\n");

    mma_direct<<<1, 32>>>((uint8_t *)dA, (uint8_t *)dB, dD);
    cudaDeviceSynchronize();
    cudaMemcpy(hD, dD, 64 * 4, cudaMemcpyDeviceToHost);
    print_D("Unique rows (all 32 lanes)", hD, 32);

    printf("\nExpected: d0_lo = d0_hi = group_id+1 (rows 0-7)\n");
    printf("          d1_lo = d1_hi = group_id+9 (rows 8-15)\n");
    printf("Actual d1 values per group:\n");
    for (int g = 0; g < 8; g++)
    {
        int lane = g * 4; // tid=0
        float d1_lo = fp16f(hD[lane * 2 + 1] & 0xFFFF);
        float d1_hi = fp16f(hD[lane * 2 + 1] >> 16);
        printf("  group %d: d0={%.0f,%.0f} d1={%.0f,%.0f}  (expected d1={%.0f,%.0f})\n",
               g,
               fp16f(hD[lane * 2] & 0xFFFF), fp16f(hD[lane * 2] >> 16),
               d1_lo, d1_hi,
               (g < 7) ? (float)(g + 9) : 0.0f, (g < 7) ? (float)(g + 9) : 0.0f);
    }

    // ========================================================================
    // TEST 4: Check QMMA count in this binary
    // ========================================================================
    printf("\n--- Run this to count QMMAs: ---\n");
    printf("cuobjdump -sass qmma_probe3 | grep -c QMMA\n");
    printf("Expected: if hardware m16n8k32 → ~4 QMMAs (4 kernels)\n");
    printf("          if hardware m8n8k32  → ~8 QMMAs (2 per PTX mma)\n");

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dD);
    cudaFree(dBf);
    return 0;
}
