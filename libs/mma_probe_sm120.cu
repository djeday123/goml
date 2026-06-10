// MMA fragment-layout probe for sm_120a.
//
// Goal: empirically verify that QMMA m16n8k32 e4m3.e4m3 (the instruction used
// in flash_attention_v66) implements PTX-doc fragment layout on Blackwell
// consumer. Also probe HMMA m16n8k16 f16 as control.
//
// Method:
//   A is 16x32 FP8 (or 16x16 FP16 for HMMA). Rows are filled with row index:
//   A[r][k] = r.   (r ranges 0..15)
//   B is 8x32 (or 8x16) all 1.
//   D[m][n] = sum_k A[m][k] * B[n][k] = K * m * 1 = K * m
//     QMMA m16n8k32: D[m][n] = 32 * m
//     HMMA m16n8k16: D[m][n] = 16 * m
//
// Standard m16n8 D output layout (constant across MMA families):
//   groupID = lane >> 2 = (0..7)
//   threadID = lane & 3 = (0..3)
//   d0 → D[gid + 0, 2*tid + 0]
//   d1 → D[gid + 0, 2*tid + 1]
//   d2 → D[gid + 8, 2*tid + 0]
//   d3 → D[gid + 8, 2*tid + 1]
// → For each lane: d0=d1=32*gid, d2=d3=32*(gid+8).
//
// We probe two A-frag layouts per MMA and report 32/32 pass count for each:
//   Hypothesis "ptx":
//     m16n8k16 HMMA:    a0=(gid,klo)  a1=(gid+8,klo)  a2=(gid,khi)  a3=(gid+8,khi)
//     m16n8k32 QMMA:    a0=(gid,klo)  a1=(gid+8,klo)  a2=(gid,khi)  a3=(gid+8,khi)
//       (klo = tid*2 for HMMA, tid*4 for QMMA; khi = klo + 8/16)
//   Hypothesis "swap" (sm_89 a1↔a2):
//     a0=(gid,klo)  a1=(gid,khi)  a2=(gid+8,klo)  a3=(gid+8,khi)
//
// Build:
//   /usr/local/cuda-13.1/bin/nvcc -arch=sm_120a -O3 -std=c++17 \
//     libs/mma_probe_sm120.cu -o runs/mma_probe_sm120 -lcudart

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); } } while(0)

// ============================================================================
// FP16 helpers
// ============================================================================
__device__ __forceinline__ uint32_t pack_fp16x2(__half lo, __half hi) {
    uint32_t r;
    uint16_t a = *(uint16_t*)&lo;
    uint16_t b = *(uint16_t*)&hi;
    asm("mov.b32 %0, {%1, %2};" : "=r"(r) : "h"(a), "h"(b));
    return r;
}

__device__ __forceinline__ uint8_t f32_to_e4m3(float f) {
    // Cvt one FP32 → e4m3 via x2 path; we duplicate the input and take the low byte.
    uint16_t h2;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;"
                 : "=h"(h2) : "f"(f));
    return (uint8_t)(h2 & 0xff);
}

// ============================================================================
// HMMA m16n8k16 .f32 acc (probe — easier to read than .f16 acc)
// ============================================================================
__device__ __forceinline__ void hmma_m16n8k16_f32(
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

// TEST_KIND=0: A[r][k]=r, B=1  → D[r][n] = K*r
// TEST_KIND=1: A=1, B[k][n]=n  → D[r][n] = K*n
template <bool SWAP, int TEST_KIND>
__global__ void probe_hmma_m16n8k16(float *D_out)
{
    int lane = threadIdx.x & 31;
    int gid = lane >> 2, tid = lane & 3;

    __shared__ __half A[16 * 16];
    __shared__ __half B[16 * 8];
    if constexpr (TEST_KIND == 0) {
        for (int i = lane; i < 16*16; i += 32) A[i] = __float2half((float)(i / 16));
        for (int i = lane; i < 16*8;  i += 32) B[i] = __float2half(1.0f);
    } else {
        for (int i = lane; i < 16*16; i += 32) A[i] = __float2half(1.0f);
        for (int i = lane; i < 16*8;  i += 32) B[i] = __float2half((float)(i % 8));
    }
    __syncthreads();

    int klo = tid * 2;
    int khi = klo + 8;
    uint32_t a0, a1, a2, a3;
    if constexpr (SWAP) {
        // sm_89-style: a1↔a2
        a0 = pack_fp16x2(A[(gid + 0) * 16 + klo + 0], A[(gid + 0) * 16 + klo + 1]);
        a1 = pack_fp16x2(A[(gid + 0) * 16 + khi + 0], A[(gid + 0) * 16 + khi + 1]);
        a2 = pack_fp16x2(A[(gid + 8) * 16 + klo + 0], A[(gid + 8) * 16 + klo + 1]);
        a3 = pack_fp16x2(A[(gid + 8) * 16 + khi + 0], A[(gid + 8) * 16 + khi + 1]);
    } else {
        // PTX doc:   a1=(gid+8, klo),   a2=(gid, khi)
        a0 = pack_fp16x2(A[(gid + 0) * 16 + klo + 0], A[(gid + 0) * 16 + klo + 1]);
        a1 = pack_fp16x2(A[(gid + 8) * 16 + klo + 0], A[(gid + 8) * 16 + klo + 1]);
        a2 = pack_fp16x2(A[(gid + 0) * 16 + khi + 0], A[(gid + 0) * 16 + khi + 1]);
        a3 = pack_fp16x2(A[(gid + 8) * 16 + khi + 0], A[(gid + 8) * 16 + khi + 1]);
    }

    // B layout: m16n8 col-major; PTX doc says
    //   b0 = (k=klo, n=gid), b1 = (k=khi, n=gid) — 2 fp16x2 packed across k
    uint32_t b0 = pack_fp16x2(B[(klo + 0) * 8 + gid], B[(klo + 1) * 8 + gid]);
    uint32_t b1 = pack_fp16x2(B[(khi + 0) * 8 + gid], B[(khi + 1) * 8 + gid]);

    float d0, d1, d2, d3;
    hmma_m16n8k16_f32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);

    D_out[lane * 4 + 0] = d0;
    D_out[lane * 4 + 1] = d1;
    D_out[lane * 4 + 2] = d2;
    D_out[lane * 4 + 3] = d3;
}

// ============================================================================
// QMMA m16n8k32 e4m3.e4m3 .f32 acc
// ============================================================================
__device__ __forceinline__ void qmma_m16n8k32_f32(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
}

template <bool SWAP, int TEST_KIND>
__global__ void probe_qmma_m16n8k32(float *D_out)
{
    int lane = threadIdx.x & 31;
    int gid = lane >> 2, tid = lane & 3;

    __shared__ uint8_t A[16 * 32];
    __shared__ uint8_t B[8 * 32];
    if constexpr (TEST_KIND == 0) {
        for (int i = lane; i < 16*32; i += 32) A[i] = f32_to_e4m3((float)(i / 32));
        for (int i = lane; i < 8*32;  i += 32) B[i] = f32_to_e4m3(1.0f);
    } else {
        for (int i = lane; i < 16*32; i += 32) A[i] = f32_to_e4m3(1.0f);
        for (int i = lane; i < 8*32;  i += 32) B[i] = f32_to_e4m3((float)(i / 32));
    }
    __syncthreads();

    int klo = tid * 4;          // 4 FP8 bytes per uint32
    int khi = klo + 16;
    auto load4 = [&](int row, int kcol) -> uint32_t {
        return *reinterpret_cast<uint32_t*>(&A[row * 32 + kcol]);
    };
    uint32_t a0, a1, a2, a3;
    if constexpr (SWAP) {
        a0 = load4(gid + 0, klo);
        a1 = load4(gid + 0, khi);
        a2 = load4(gid + 8, klo);
        a3 = load4(gid + 8, khi);
    } else {
        a0 = load4(gid + 0, klo);
        a1 = load4(gid + 8, klo);
        a2 = load4(gid + 0, khi);
        a3 = load4(gid + 8, khi);
    }

    // B: 8x32 FP8 col-major view; PTX doc says
    //   b0 = (n=gid, k=klo..klo+3)  (uint32 from B[gid*32 + klo])
    //   b1 = (n=gid, k=khi..khi+3)
    uint32_t b0 = *reinterpret_cast<uint32_t*>(&B[gid * 32 + klo]);
    uint32_t b1 = *reinterpret_cast<uint32_t*>(&B[gid * 32 + khi]);

    float d0, d1, d2, d3;
    qmma_m16n8k32_f32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);

    D_out[lane * 4 + 0] = d0;
    D_out[lane * 4 + 1] = d1;
    D_out[lane * 4 + 2] = d2;
    D_out[lane * 4 + 3] = d3;
}

// ============================================================================
// Host: run both hypotheses for both MMAs, count 32/32 passes
// ============================================================================
//
// Output is structured so that EVERY verdict is justified by visible numbers:
//   - For PASS cases, we print 4 sample lanes showing d == exp.
//   - For FAIL cases, we print first 4 mismatched lanes showing d ≠ exp.
//   - Verdict line ALWAYS comes AFTER the numbers, never before.
// For TEST_KIND=0 (A[r][k]=r, B=1):  D[r][n] = K*r
//   d0=K*gid, d1=K*gid, d2=K*(gid+8), d3=K*(gid+8)
// For TEST_KIND=1 (A=1, B[k][n]=n):  D[r][n] = K*n
//   d0=K*(2*tid), d1=K*(2*tid+1), d2=K*(2*tid), d3=K*(2*tid+1)
static void compute_exp(int test_kind, int K, int gid, int tid, float *e0, float *e1, float *e2, float *e3)
{
    if (test_kind == 0) {
        float lo = (float)K * gid, hi = (float)K * (gid + 8);
        *e0 = lo; *e1 = lo; *e2 = hi; *e3 = hi;
    } else {
        float c0 = (float)K * (2*tid), c1 = (float)K * (2*tid + 1);
        *e0 = c0; *e1 = c1; *e2 = c0; *e3 = c1;
    }
}

static int verify(const float *hD, const char *name, int test_kind, int K)
{
    printf("  -- %s --\n", name);
    int ok = 0;
    for (int lane = 0; lane < 32; ++lane) {
        int gid = lane >> 2, tid = lane & 3;
        float e0,e1,e2,e3;
        compute_exp(test_kind, K, gid, tid, &e0,&e1,&e2,&e3);
        bool pass = (hD[lane*4+0] == e0) && (hD[lane*4+1] == e1)
                 && (hD[lane*4+2] == e2) && (hD[lane*4+3] == e3);
        if (pass) ok++;
    }
    int printed = 0, want_pass = (ok == 32);
    for (int lane = 0; lane < 32 && printed < 4; ++lane) {
        int gid = lane >> 2, tid = lane & 3;
        float e0,e1,e2,e3;
        compute_exp(test_kind, K, gid, tid, &e0,&e1,&e2,&e3);
        bool pass = (hD[lane*4+0] == e0) && (hD[lane*4+1] == e1)
                 && (hD[lane*4+2] == e2) && (hD[lane*4+3] == e3);
        if (pass != want_pass) continue;
        const char *tag = pass ? "match" : " MISS";
        printf("     [%s] lane %2d (gid=%d,tid=%d): d=[%.0f,%.0f,%.0f,%.0f]  exp=[%.0f,%.0f,%.0f,%.0f]\n",
            tag, lane, gid, tid,
            hD[lane*4+0], hD[lane*4+1], hD[lane*4+2], hD[lane*4+3],
            e0, e1, e2, e3);
        printed++;
    }
    printf("     verdict: %d/32 lanes OK\n\n", ok);
    return ok;
}

int main()
{
    float *dD;
    float hD[128];
    CK(cudaMalloc(&dD, 128 * sizeof(float)));

    int dev = 0;
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s  compute=%d.%d\n", prop.name, prop.major, prop.minor);
    printf("\n");

    auto run_hmma = [&](bool swap, int test_kind) {
        CK(cudaMemset(dD, 0, 128*sizeof(float)));
        if (swap) {
            if (test_kind == 0) probe_hmma_m16n8k16<true , 0><<<1,32>>>(dD);
            else                probe_hmma_m16n8k16<true , 1><<<1,32>>>(dD);
        } else {
            if (test_kind == 0) probe_hmma_m16n8k16<false, 0><<<1,32>>>(dD);
            else                probe_hmma_m16n8k16<false, 1><<<1,32>>>(dD);
        }
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hD, dD, 128*sizeof(float), cudaMemcpyDeviceToHost));
    };
    auto run_qmma = [&](bool swap, int test_kind) {
        CK(cudaMemset(dD, 0, 128*sizeof(float)));
        if (swap) {
            if (test_kind == 0) probe_qmma_m16n8k32<true , 0><<<1,32>>>(dD);
            else                probe_qmma_m16n8k32<true , 1><<<1,32>>>(dD);
        } else {
            if (test_kind == 0) probe_qmma_m16n8k32<false, 0><<<1,32>>>(dD);
            else                probe_qmma_m16n8k32<false, 1><<<1,32>>>(dD);
        }
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hD, dD, 128*sizeof(float), cudaMemcpyDeviceToHost));
    };

    int hmma_ptx_t0, hmma_ptx_t1, hmma_swap_t0, hmma_swap_t1;
    int qmma_ptx_t0, qmma_ptx_t1, qmma_swap_t0, qmma_swap_t1;

    printf("############################################################\n");
    printf("# HMMA m16n8k16 .f32 acc  (K=16)\n");
    printf("############################################################\n");

    printf("=== Test 1: A[r][k]=r, B=1  →  D[r][n] = 16*r ===\n");
    run_hmma(false, 0); hmma_ptx_t0  = verify(hD, "PTX-doc layout", 0, 16);
    run_hmma(true , 0); hmma_swap_t0 = verify(hD, "sm_89 swap    ", 0, 16);

    printf("=== Test 2: A=1, B[k][n]=n  →  D[r][n] = 16*n ===\n");
    run_hmma(false, 1); hmma_ptx_t1  = verify(hD, "PTX-doc layout", 1, 16);
    run_hmma(true , 1); hmma_swap_t1 = verify(hD, "sm_89 swap    ", 1, 16);

    printf("############################################################\n");
    printf("# QMMA m16n8k32 kind::f8f6f4 e4m3.e4m3 .f32 acc  (K=32)\n");
    printf("############################################################\n");

    printf("=== Test 1: A[r][k]=r, B=1  →  D[r][n] = 32*r ===\n");
    run_qmma(false, 0); qmma_ptx_t0  = verify(hD, "PTX-doc layout", 0, 32);
    run_qmma(true , 0); qmma_swap_t0 = verify(hD, "sm_89 swap    ", 0, 32);

    printf("=== Test 2: A=1, B[k][n]=n  →  D[r][n] = 32*n ===\n");
    run_qmma(false, 1); qmma_ptx_t1  = verify(hD, "PTX-doc layout", 1, 32);
    run_qmma(true , 1); qmma_swap_t1 = verify(hD, "sm_89 swap    ", 1, 32);

    printf("############################################################\n");
    printf("# Final verdict (both A-test AND B-test must agree)\n");
    printf("############################################################\n");
    bool hmma_ptx_ok = (hmma_ptx_t0 == 32) && (hmma_ptx_t1 == 32);
    bool qmma_ptx_ok = (qmma_ptx_t0 == 32) && (qmma_ptx_t1 == 32);
    printf("  HMMA m16n8k16:  PTX-doc t0=%d/32 t1=%d/32  swap t0=%d/32 t1=%d/32  → %s\n",
        hmma_ptx_t0, hmma_ptx_t1, hmma_swap_t0, hmma_swap_t1,
        hmma_ptx_ok ? "USE PTX-doc (both tests confirm)" : "INCONSISTENT");
    printf("  QMMA m16n8k32:  PTX-doc t0=%d/32 t1=%d/32  swap t0=%d/32 t1=%d/32  → %s\n",
        qmma_ptx_t0, qmma_ptx_t1, qmma_swap_t0, qmma_swap_t1,
        qmma_ptx_ok ? "USE PTX-doc (both tests confirm, v66 matches)" : "INCONSISTENT");

    cudaFree(dD);
    return 0;
}
