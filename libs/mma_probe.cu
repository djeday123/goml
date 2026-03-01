// MMA m16n8k16 FP16→F32 Definitive Probe — a1↔a2 SWAPPED
// groupID = lane>>2 (0..7), tid = lane&3 (0..3)
//
// SM89 actual layout (a1↔a2 swapped vs PTX doc):
//   a0: row=gid,   k=tid*2      (k_low,  row_lo)
//   a1: row=gid+8, k=tid*2      (k_low,  row_hi)  ← swapped
//   a2: row=gid,   k=tid*2+8    (k_high, row_lo)  ← swapped
//   a3: row=gid+8, k=tid*2+8    (k_high, row_hi)
//
// Build: nvcc -O3 -arch=sm_89 -std=c++17 libs/mma_probe.cu -o runs/mma_probe -lcudart

#include <stdint.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(c)                                                                               \
    do                                                                                      \
    {                                                                                       \
        cudaError_t e = (c);                                                                \
        if (e != cudaSuccess)                                                               \
        {                                                                                   \
            fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(1);                                                                        \
        }                                                                                   \
    } while (0)

__device__ __forceinline__ uint32_t pk2(const __half *p)
{
    uint32_t r;
    asm("mov.b32 %0, {%1, %2};" : "=r"(r)
        : "h"(*(const unsigned short *)&p[0]),
          "h"(*(const unsigned short *)&p[1]));
    return r;
}
__device__ __forceinline__ uint32_t pk2s(const __half *p0, const __half *p1)
{
    uint32_t r;
    asm("mov.b32 %0, {%1, %2};" : "=r"(r)
        : "h"(*(const unsigned short *)p0),
          "h"(*(const unsigned short *)p1));
    return r;
}
__device__ __forceinline__ void do_mma(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
}

// Test 1: A[row][k]=row, B=1 → D[row][*] = 16*row
__global__ void test1(float *D)
{
    int L = threadIdx.x % 32;
    int gid = L >> 2, tid = L & 3;

    __shared__ __half A[16 * 16], B[16 * 8];
    for (int i = L; i < 256; i += 32)
        A[i] = __float2half((float)(i / 16));
    for (int i = L; i < 128; i += 32)
        B[i] = __float2half(1.0f);
    __syncthreads();

    uint32_t a0 = pk2(&A[gid * 16 + tid * 2]);
    uint32_t a1 = pk2(&A[(gid + 8) * 16 + tid * 2]); // SWAPPED: row=gid+8, k_low
    uint32_t a2 = pk2(&A[gid * 16 + tid * 2 + 8]);   // SWAPPED: row=gid, k_high
    uint32_t a3 = pk2(&A[(gid + 8) * 16 + tid * 2 + 8]);

    uint32_t b0 = pk2s(&B[(tid * 2) * 8 + gid], &B[(tid * 2 + 1) * 8 + gid]);
    uint32_t b1 = pk2s(&B[(tid * 2 + 8) * 8 + gid], &B[(tid * 2 + 9) * 8 + gid]);

    float d0, d1, d2, d3;
    do_mma(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);

    D[gid * 8 + tid * 2] = d0;
    D[gid * 8 + tid * 2 + 1] = d1;
    D[(gid + 8) * 8 + tid * 2] = d2;
    D[(gid + 8) * 8 + tid * 2 + 1] = d3;
}

// Test 2: A=1, B[k][n]=k → D = 120 everywhere
__global__ void test2(float *D)
{
    int L = threadIdx.x % 32;
    int gid = L >> 2, tid = L & 3;

    __shared__ __half A[16 * 16], B[16 * 8];
    for (int i = L; i < 256; i += 32)
        A[i] = __float2half(1.0f);
    for (int i = L; i < 128; i += 32)
        B[i] = __float2half((float)(i / 8));
    __syncthreads();

    uint32_t a0 = pk2(&A[gid * 16 + tid * 2]);
    uint32_t a1 = pk2(&A[(gid + 8) * 16 + tid * 2]); // SWAPPED
    uint32_t a2 = pk2(&A[gid * 16 + tid * 2 + 8]);   // SWAPPED
    uint32_t a3 = pk2(&A[(gid + 8) * 16 + tid * 2 + 8]);

    uint32_t b0 = pk2s(&B[(tid * 2) * 8 + gid], &B[(tid * 2 + 1) * 8 + gid]);
    uint32_t b1 = pk2s(&B[(tid * 2 + 8) * 8 + gid], &B[(tid * 2 + 9) * 8 + gid]);

    float d0, d1, d2, d3;
    do_mma(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);

    D[gid * 8 + tid * 2] = d0;
    D[gid * 8 + tid * 2 + 1] = d1;
    D[(gid + 8) * 8 + tid * 2] = d2;
    D[(gid + 8) * 8 + tid * 2 + 1] = d3;
}

// Test 3: A=I, B=I → D=I (top-left 8x8)
__global__ void test3(float *D)
{
    int L = threadIdx.x % 32;
    int gid = L >> 2, tid = L & 3;

    __shared__ __half A[16 * 16], B[16 * 8];
    for (int i = L; i < 256; i += 32)
        A[i] = __float2half(0.0f);
    for (int i = L; i < 128; i += 32)
        B[i] = __float2half(0.0f);
    __syncthreads();
    if (L < 16)
        A[L * 16 + L] = __float2half(1.0f);
    if (L < 8)
        B[L * 8 + L] = __float2half(1.0f);
    __syncthreads();

    uint32_t a0 = pk2(&A[gid * 16 + tid * 2]);
    uint32_t a1 = pk2(&A[(gid + 8) * 16 + tid * 2]); // SWAPPED
    uint32_t a2 = pk2(&A[gid * 16 + tid * 2 + 8]);   // SWAPPED
    uint32_t a3 = pk2(&A[(gid + 8) * 16 + tid * 2 + 8]);

    uint32_t b0 = pk2s(&B[(tid * 2) * 8 + gid], &B[(tid * 2 + 1) * 8 + gid]);
    uint32_t b1 = pk2s(&B[(tid * 2 + 8) * 8 + gid], &B[(tid * 2 + 9) * 8 + gid]);

    float d0, d1, d2, d3;
    do_mma(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);

    D[gid * 8 + tid * 2] = d0;
    D[gid * 8 + tid * 2 + 1] = d1;
    D[(gid + 8) * 8 + tid * 2] = d2;
    D[(gid + 8) * 8 + tid * 2 + 1] = d3;
}

// Test 4: A[r][k]=r*16+k, B=1 → D[r][c] = 256*r + 120
__global__ void test4_dump(float *out)
{
    int L = threadIdx.x % 32;
    int gid = L >> 2, tid = L & 3;

    __shared__ __half A[16 * 16], B[16 * 8];
    for (int i = L; i < 256; i += 32)
        A[i] = __float2half((float)i);
    for (int i = L; i < 128; i += 32)
        B[i] = __float2half(1.0f);
    __syncthreads();

    uint32_t a0 = pk2(&A[gid * 16 + tid * 2]);
    uint32_t a1 = pk2(&A[(gid + 8) * 16 + tid * 2]); // SWAPPED
    uint32_t a2 = pk2(&A[gid * 16 + tid * 2 + 8]);   // SWAPPED
    uint32_t a3 = pk2(&A[(gid + 8) * 16 + tid * 2 + 8]);

    uint32_t b0 = pk2s(&B[(tid * 2) * 8 + gid], &B[(tid * 2 + 1) * 8 + gid]);
    uint32_t b1 = pk2s(&B[(tid * 2 + 8) * 8 + gid], &B[(tid * 2 + 9) * 8 + gid]);

    float d0, d1, d2, d3;
    do_mma(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);

    out[L * 4 + 0] = d0;
    out[L * 4 + 1] = d1;
    out[L * 4 + 2] = d2;
    out[L * 4 + 3] = d3;
}

void pm(const char *n, float *D, int M, int N)
{
    printf("  %s:\n", n);
    for (int i = 0; i < M; i++)
    {
        printf("    row %2d: ", i);
        for (int j = 0; j < N; j++)
            printf("%7.1f ", D[i * N + j]);
        printf("\n");
    }
}

int main()
{
    float *dD, hD[256];
    CK(cudaMalloc(&dD, 256 * 4));
    printf("=== MMA m16n8k16 Probe — a1/a2 SWAPPED (gid=lane>>2, tid=lane&3) ===\n\n");

    printf("Test 1: A[r][k]=r, B=1 → D[r][*]=16*r (0,16,32,...,240)\n");
    CK(cudaMemset(dD, 0, 256 * 4));
    test1<<<1, 32>>>(dD);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hD, dD, 128 * 4, cudaMemcpyDeviceToHost));
    pm("D", hD, 16, 8);

    printf("\nTest 2: A=1, B[k][n]=k → D=120 everywhere\n");
    CK(cudaMemset(dD, 0, 256 * 4));
    test2<<<1, 32>>>(dD);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hD, dD, 128 * 4, cudaMemcpyDeviceToHost));
    pm("D", hD, 16, 8);

    printf("\nTest 3: A=I, B=I → D=I (top-left 8x8)\n");
    CK(cudaMemset(dD, 0, 256 * 4));
    test3<<<1, 32>>>(dD);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hD, dD, 128 * 4, cudaMemcpyDeviceToHost));
    pm("D", hD, 16, 8);

    printf("\nTest 4: Raw per-lane dump. A[r][k]=r*16+k, B=1\n");
    printf("  Expected: d0,d1 = 256*gid+120 (row=gid), d2,d3 = 256*(gid+8)+120 (row=gid+8)\n");
    CK(cudaMemset(dD, 0, 256 * 4));
    test4_dump<<<1, 32>>>(dD);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hD, dD, 128 * 4, cudaMemcpyDeviceToHost));
    for (int lane = 0; lane < 32; lane++)
    {
        int gid = lane >> 2, tid = lane & 3;
        float exp_lo = 256.0f * gid + 120.0f;
        float exp_hi = 256.0f * (gid + 8) + 120.0f;
        printf("  lane %2d (gid=%d,tid=%d): d0=%7.1f d1=%7.1f d2=%7.1f d3=%7.1f  | exp_lo=%7.1f exp_hi=%7.1f %s\n",
               lane, gid, tid, hD[lane * 4], hD[lane * 4 + 1], hD[lane * 4 + 2], hD[lane * 4 + 3],
               exp_lo, exp_hi,
               (hD[lane * 4] == exp_lo && hD[lane * 4 + 1] == exp_lo &&
                hD[lane * 4 + 2] == exp_hi && hD[lane * 4 + 3] == exp_hi)
                   ? "OK"
                   : "FAIL");
    }

    cudaFree(dD);
}
