// =============================================================================
// SM89 MMA Fragment Mapping Probe
// =============================================================================
// Does ONE mma.sync.aligned.m16n8k32 with known inputs and dumps
// the per-thread register values to reveal exact fragment layout.
//
// A = identity-like: A[row][col] = (row == col/2) ? 1.0 : 0.0
//     (simplified to make output interpretable)
// B = sequential: B[n][k] = n (constant across K)
//
// So C[m][n] = sum_k A[m][k] * B[n][k] ≈ B[n][m_related] * count
//
// Actually simpler: set B[n][k] = 1.0 for k=0 only, 0 elsewhere
// Then C[m][n] = A[m][0] * B[n][0] = A[m][0]
//
// Even simpler: just use known patterns and dump everything.
//
// Build: nvcc -O3 -arch=sm_89 -std=c++17 fp8_probe.cu -o fp8_probe
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// FP8 E4M3 helpers
static uint8_t f2e4m3(float f)
{
    if (f == 0.0f)
        return 0;
    int sign = f < 0 ? 1 : 0;
    float af = fabsf(f);
    int exp = (int)floorf(log2f(af));
    float m = af / ldexpf(1.0f, exp) - 1.0f;
    int m3 = (int)(m * 8.0f + 0.5f);
    if (m3 >= 8)
    {
        m3 = 0;
        exp++;
    }
    int eb = exp + 7;
    if (eb < 1)
        return (sign << 7);
    if (eb > 15)
        eb = 15;
    return (sign << 7) | (eb << 3) | (m3 & 7);
}

static float e4m3_to_f(uint8_t v)
{
    int s = (v >> 7) & 1;
    int e = (v >> 3) & 0xF;
    int m = v & 7;
    if (e == 0xF && m == 7)
        return nanf("");
    float r;
    if (e == 0)
        r = ldexpf((float)m, -9);
    else
        r = ldexpf(1.0f + m / 8.0f, e - 7);
    return s ? -r : r;
}

static float fp16_bits_to_f(uint16_t h)
{
    __half hv;
    memcpy(&hv, &h, 2);
    return __half2float(hv);
}

// =============================================================================
// Probe kernel: 1 warp does 1 MMA, stores fragment info
// =============================================================================
struct ProbeResult
{
    // Per-thread (32 threads)
    uint32_t a_frag[32][4]; // A fragments loaded
    uint32_t b_frag[32][2]; // B fragments loaded
    uint32_t d_frag[32][2]; // D output fragments
    int lane_id[32];
    int group_id[32];
    int tid[32];
};

__device__ void mma_probe(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t &c0, uint32_t &c1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0, %1}, "
        "{%2, %3, %4, %5}, "
        "{%6, %7}, "
        "{%0, %1};\n"
        : "+r"(c0), "+r"(c1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1));
}

__global__ void probe_kernel(
    const uint8_t *__restrict__ A_smem_data, // 16 * 32 = 512 bytes
    const uint8_t *__restrict__ B_smem_data, // 8 * 32 = 256 bytes
    ProbeResult *result)
{
    // Only first warp
    if (threadIdx.x >= 32)
        return;

    __shared__ uint8_t sA[16 * 32]; // 16 rows × 32 cols (MMA_K=32)
    __shared__ uint8_t sB[8 * 32];  // 8 rows × 32 cols

    // Collaborative load
    if (threadIdx.x < 16)
    { // 512/32 = 16 loads per thread... just do it simply
        for (int i = threadIdx.x; i < 512; i += 32)
            sA[i] = A_smem_data[i];
    }
    if (threadIdx.x < 8)
    {
        for (int i = threadIdx.x; i < 256; i += 32)
            sB[i] = B_smem_data[i];
    }
    // Actually just load everything with all threads
    for (int i = threadIdx.x; i < 512; i += 32)
        sA[i] = A_smem_data[i];
    for (int i = threadIdx.x; i < 256; i += 32)
        sB[i] = B_smem_data[i];

    __syncthreads();

    int lane = threadIdx.x;
    int grp = lane / 4;
    int t = lane % 4;

    // Standard PTX fragment loading (as documented)
    // A[16,32]: a0 = A[grp][t*4..t*4+3], a1 = A[grp][t*4+16..t*4+19]
    //           a2 = A[grp+8][t*4..t*4+3], a3 = A[grp+8][t*4+16..t*4+19]
    uint32_t a0 = *(uint32_t *)&sA[grp * 32 + t * 4];
    uint32_t a1 = *(uint32_t *)&sA[grp * 32 + t * 4 + 16];
    uint32_t a2 = *(uint32_t *)&sA[(grp + 8) * 32 + t * 4];
    uint32_t a3 = *(uint32_t *)&sA[(grp + 8) * 32 + t * 4 + 16];

    // B[8,32]: b0 = B[grp][t*4..t*4+3], b1 = B[grp][t*4+16..t*4+19]
    uint32_t b0 = *(uint32_t *)&sB[grp * 32 + t * 4];
    uint32_t b1 = *(uint32_t *)&sB[grp * 32 + t * 4 + 16];

    // MMA
    uint32_t d0 = 0, d1 = 0;
    mma_probe(a0, a1, a2, a3, b0, b1, d0, d1);

    // Store results
    result->a_frag[lane][0] = a0;
    result->a_frag[lane][1] = a1;
    result->a_frag[lane][2] = a2;
    result->a_frag[lane][3] = a3;
    result->b_frag[lane][0] = b0;
    result->b_frag[lane][1] = b1;
    result->d_frag[lane][0] = d0;
    result->d_frag[lane][1] = d1;
    result->lane_id[lane] = lane;
    result->group_id[lane] = grp;
    result->tid[lane] = t;
}

int main()
{
    printf("=== SM89 MMA Fragment Mapping Probe ===\n\n");

    // =========================================================================
    // Test 1: B = column indicator (B[n][0..31] = n for all k)
    //         A = row indicator  (A[m][0..31] = m*16 for all k... too big)
    //
    // Simpler: A = identity-ish, B = row number encoded
    //
    // Actually, simplest probe:
    //   A[m][k] = 1.0 for all m,k
    //   B[n][k] = 0 except B[n][0] = (n+1) as FP8
    //   Then C[m][n] = 1.0 * B[n][0] = (n+1)
    //   So output d0,d1 tells us which n each thread/register gets
    // =========================================================================

    printf("--- Test: B[n][0] = n+1, A = all 1.0 ---\n");
    printf("  Expected: C[m][n] = n+1 for all m\n\n");

    uint8_t hA[16 * 32]; // A[16][32]
    uint8_t hB[8 * 32];  // B[8][32]

    uint8_t one = f2e4m3(1.0f);

    // A = all 1.0
    memset(hA, 0, sizeof(hA));
    for (int m = 0; m < 16; m++)
        for (int k = 0; k < 32; k++)
            hA[m * 32 + k] = one;

    // B[n][k] = 0 except B[n][0] = fp8(n+1)
    memset(hB, 0, sizeof(hB));
    for (int n = 0; n < 8; n++)
        hB[n * 32 + 0] = f2e4m3((float)(n + 1));

    printf("  B values: ");
    for (int n = 0; n < 8; n++)
        printf("B[%d]=%.1f ", n, e4m3_to_f(hB[n * 32]));
    printf("\n\n");

    void *dA, *dB;
    ProbeResult *dR;
    cudaMalloc(&dA, sizeof(hA));
    cudaMalloc(&dB, sizeof(hB));
    cudaMalloc(&dR, sizeof(ProbeResult));
    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);
    cudaMemset(dR, 0, sizeof(ProbeResult));

    probe_kernel<<<1, 32>>>((const uint8_t *)dA, (const uint8_t *)dB, dR);
    cudaDeviceSynchronize();

    ProbeResult hR;
    cudaMemcpy(&hR, dR, sizeof(ProbeResult), cudaMemcpyDeviceToHost);

    // Print output mapping
    printf("  lane  grp  tid |  d0 (lo,hi fp16)       |  d1 (lo,hi fp16)       | → cols?\n");
    printf("  ----  ---  --- | ----------------------  | ---------------------- | -------\n");
    for (int i = 0; i < 32; i++)
    {
        uint32_t d0 = hR.d_frag[i][0];
        uint32_t d1 = hR.d_frag[i][1];
        float d0_lo = fp16_bits_to_f(d0 & 0xFFFF);
        float d0_hi = fp16_bits_to_f(d0 >> 16);
        float d1_lo = fp16_bits_to_f(d1 & 0xFFFF);
        float d1_hi = fp16_bits_to_f(d1 >> 16);

        printf("  %4d  %3d  %3d | %6.1f %6.1f (0x%08X) | %6.1f %6.1f (0x%08X)\n",
               hR.lane_id[i], hR.group_id[i], hR.tid[i],
               d0_lo, d0_hi, d0,
               d1_lo, d1_hi, d1);
    }

    // =========================================================================
    // Interpret: d0_lo and d0_hi should be (n+1) values
    // The n value tells us which B column this register maps to
    // d0 → rows 0-7 (groupID), d1 → rows 8-15 (groupID+8)
    // Each contains 2 packed FP16 values = 2 columns
    // =========================================================================
    printf("\n--- Output column mapping ---\n");
    printf("  If d0_lo = k, then this thread's d0 low half → column (k-1)\n");
    printf("  If d0_hi = k, then this thread's d0 high half → column (k-1)\n\n");

    printf("  tid | d0_lo→col | d0_hi→col | d1_lo→col | d1_hi→col\n");
    printf("  --- | --------- | --------- | --------- | ---------\n");
    // Show one group (all groups should be same for cols)
    for (int t = 0; t < 4; t++)
    {
        int lane = t; // group 0
        uint32_t d0 = hR.d_frag[lane][0];
        uint32_t d1 = hR.d_frag[lane][1];
        float d0_lo = fp16_bits_to_f(d0 & 0xFFFF);
        float d0_hi = fp16_bits_to_f(d0 >> 16);
        float d1_lo = fp16_bits_to_f(d1 & 0xFFFF);
        float d1_hi = fp16_bits_to_f(d1 >> 16);

        printf("  %3d |    %5.1f   |    %5.1f   |    %5.1f   |    %5.1f\n",
               t, d0_lo, d0_hi, d1_lo, d1_hi);
    }

    // =========================================================================
    // Test 2: A = row indicator, B = all 1.0
    //   A[m][k] = 0 except A[m][0] = fp8(m+1)
    //   B = all 1.0
    //   C[m][n] = A[m][0] * 1.0 = (m+1)
    //   This reveals the ROW mapping of output registers
    // =========================================================================
    printf("\n\n--- Test 2: A[m][0] = m+1, B = all 1.0 ---\n");
    printf("  Expected: C[m][n] = m+1\n\n");

    memset(hA, 0, sizeof(hA));
    for (int m = 0; m < 16; m++)
        hA[m * 32 + 0] = f2e4m3((float)(m + 1));

    memset(hB, 0, sizeof(hB));
    for (int n = 0; n < 8; n++)
        for (int k = 0; k < 32; k++)
            hB[n * 32 + k] = one;

    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);
    cudaMemset(dR, 0, sizeof(ProbeResult));

    probe_kernel<<<1, 32>>>((const uint8_t *)dA, (const uint8_t *)dB, dR);
    cudaDeviceSynchronize();
    cudaMemcpy(&hR, dR, sizeof(ProbeResult), cudaMemcpyDeviceToHost);

    printf("  grp | d0_lo (row?) | d0_hi (row?) | d1_lo (row?) | d1_hi (row?)\n");
    printf("  --- | ------------ | ------------ | ------------ | ------------\n");
    for (int g = 0; g < 8; g++)
    {
        int lane = g * 4; // tid=0 for each group
        uint32_t d0 = hR.d_frag[lane][0];
        uint32_t d1 = hR.d_frag[lane][1];
        printf("  %3d |    %6.1f    |    %6.1f    |    %6.1f    |    %6.1f\n",
               g,
               fp16_bits_to_f(d0 & 0xFFFF), fp16_bits_to_f(d0 >> 16),
               fp16_bits_to_f(d1 & 0xFFFF), fp16_bits_to_f(d1 >> 16));
    }

    // =========================================================================
    // Test 3: Both indicators — A[m][0] = m+1, B[n][0] = (n+1)*16
    //   C[m][n] = (m+1) * (n+1) * 16
    //   Each output value is unique → can decode both row and col
    // =========================================================================
    printf("\n\n--- Test 3: A[m][0]=m+1, B[n][0]=(n+1)*16 → C[m][n]=(m+1)*(n+1)*16 ---\n\n");

    memset(hA, 0, sizeof(hA));
    for (int m = 0; m < 16; m++)
        hA[m * 32] = f2e4m3((float)(m + 1));

    memset(hB, 0, sizeof(hB));
    for (int n = 0; n < 8; n++)
        hB[n * 32] = f2e4m3((float)((n + 1) * 16));

    printf("  A vals: ");
    for (int m = 0; m < 16; m++)
        printf("%.0f ", e4m3_to_f(hA[m * 32]));
    printf("\n  B vals: ");
    for (int n = 0; n < 8; n++)
        printf("%.0f ", e4m3_to_f(hB[n * 32]));
    printf("\n\n");

    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);
    cudaMemset(dR, 0, sizeof(ProbeResult));

    probe_kernel<<<1, 32>>>((const uint8_t *)dA, (const uint8_t *)dB, dR);
    cudaDeviceSynchronize();
    cudaMemcpy(&hR, dR, sizeof(ProbeResult), cudaMemcpyDeviceToHost);

    printf("  Expected C[m][n] = (m+1)*(n+1)*16:\n  ");
    for (int n = 0; n < 8; n++)
        printf("  n=%d  ", n);
    printf("\n");
    for (int m = 0; m < 16; m++)
    {
        printf("  m=%2d:", m);
        for (int n = 0; n < 8; n++)
            printf(" %5.0f", (m + 1.0f) * (n + 1.0f) * 16.0f);
        printf("\n");
    }

    printf("\n  Per-thread output (decode: value/(16) → (m+1)*(n+1)):\n");
    printf("  lane grp tid | d0_lo          | d0_hi          | d1_lo          | d1_hi\n");
    printf("  ---- --- --- | -------------- | -------------- | -------------- | ------\n");
    for (int i = 0; i < 32; i++)
    {
        uint32_t d0 = hR.d_frag[i][0];
        uint32_t d1 = hR.d_frag[i][1];
        float v[4] = {
            fp16_bits_to_f(d0 & 0xFFFF),
            fp16_bits_to_f(d0 >> 16),
            fp16_bits_to_f(d1 & 0xFFFF),
            fp16_bits_to_f(d1 >> 16)};
        printf("  %4d %3d %3d |", i, hR.group_id[i], hR.tid[i]);
        for (int j = 0; j < 4; j++)
        {
            // Decode: v = (m+1)*(n+1)*16 → find m,n
            float val = v[j];
            int found_m = -1, found_n = -1;
            for (int m = 0; m < 16 && found_m < 0; m++)
                for (int n = 0; n < 8 && found_m < 0; n++)
                    if (fabsf(val - (m + 1.0f) * (n + 1.0f) * 16.0f) < 1.0f)
                    {
                        found_m = m;
                        found_n = n;
                    }
            if (found_m >= 0)
                printf(" %5.0f→[%2d][%d] |", val, found_m, found_n);
            else
                printf(" %5.0f→[??][?] |", val);
        }
        printf("\n");
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dR);
    return 0;
}
