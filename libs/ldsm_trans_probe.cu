// Step 2c: probe whether ldmatrix.trans (LDSM.MT88) produces a B-fragment
// layout that matches QMMA m16n8k32 e4m3.e4m3 input expectation.
//
// We don't pre-suppose anything about the trans register order. Method:
//   - Stage a V tile in SMEM in (k_row, n_col) FP8 layout
//   - Load B fragment via ldmatrix.trans (which is documented to return
//     a transposed 8x8 b16 tile in registers)
//   - Feed into QMMA, multiply by A=1 (known good A layout from mma_probe)
//   - Compare D output to expected D[r][n] = K*B_avg per n
//
// Reference test (same as mma_probe Test 2):
//   A = 1 everywhere → D[r][n] = sum_k 1 * B[k][n] = sum_k B[k][n]
// Choose B[k][n] = n  (col-index value) →  D[r][n] = K * n = 32 * n
//
// For each lane (gid, tid), QMMA outputs:
//   d0 = D[gid, 2*tid]       = 32 * (2*tid)
//   d1 = D[gid, 2*tid + 1]   = 32 * (2*tid + 1)
//   d2 = D[gid+8, 2*tid]     = 32 * (2*tid)
//   d3 = D[gid+8, 2*tid + 1] = 32 * (2*tid + 1)
//
// If 32/32 lanes match: ldmatrix.trans gives the right B-frag layout for QMMA.
// If not: the register order differs and we need fixup (or this lever is dead).
//
// IMPORTANT: ldmatrix.x4 loads 4 8x8 fp16/bf16 tiles (= 2 8x16 fp8 effective).
// PTX has variants for b16 only. For FP8 we'd need to handle pairs or use
// ldmatrix.x2 / .x4 with reinterpretation. This probe tests the simplest case
// and we report whether the bytes line up correctly.
//
// Build:
//   /usr/local/cuda-13.1/bin/nvcc -O3 -gencode arch=compute_120a,code=sm_120a \
//     -std=c++17 libs/ldsm_trans_probe.cu -o runs/ldsm_trans_probe -lcudart

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); } } while(0)

__device__ __forceinline__ uint8_t f32_to_e4m3(float f) {
    uint16_t h2;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;" : "=h"(h2) : "f"(f));
    return (uint8_t)(h2 & 0xff);
}

__device__ __forceinline__ uint32_t pack_fp16x2(__half lo, __half hi) {
    uint32_t r;
    uint16_t a = *(uint16_t*)&lo;
    uint16_t b = *(uint16_t*)&hi;
    asm("mov.b32 %0, {%1, %2};" : "=r"(r) : "h"(a), "h"(b));
    return r;
}

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

// HYPOTHESIS A: ldmatrix.trans.x2.m8n8.b16 loads in the order QMMA expects.
//   We stage V as B[k][n] FP8 in SMEM laid out as 32K × 8N.
//   ldmatrix.trans returns 2 .b32 per thread representing the trans tile.
//   QMMA's B-frag for m16n8k32: 2 .b32 per thread, packing (n=gid, k=klo..klo+3)
//   and (n=gid, k=khi..khi+3) — 4 FP8 bytes per .b32.
//
// To match: ldmatrix.trans must produce uint32 with bytes corresponding to
// k_low..k_low+3 (b0) and k_high..k_high+3 (b1) for n=gid.
__global__ void probe_ldsm_trans(float *D_out, int hypothesis)
{
    int lane = threadIdx.x & 31;
    int gid = lane >> 2, tid = lane & 3;

    // A = 1 everywhere (known good PTX-doc A layout)
    // B = 8x32 FP8: B[k][n] = n
    __shared__ uint8_t A[16 * 32];
    __shared__ uint8_t B[32 * 8];   // 32 k-rows × 8 n-cols (stored as [k][n])
    for (int i = lane; i < 16*32; i += 32) A[i] = f32_to_e4m3(1.0f);
    for (int i = lane; i < 32*8;  i += 32) B[i] = f32_to_e4m3((float)(i & 7));
    __syncthreads();

    // Build PTX-doc A frag: a0=(gid,klo) a1=(gid+8,klo) a2=(gid,khi) a3=(gid+8,khi)
    int klo = tid * 4, khi = klo + 16;
    auto loadA = [&](int row, int kcol) -> uint32_t {
        return *reinterpret_cast<uint32_t*>(&A[row * 32 + kcol]);
    };
    uint32_t a0 = loadA(gid + 0, klo);
    uint32_t a1 = loadA(gid + 8, klo);
    uint32_t a2 = loadA(gid + 0, khi);
    uint32_t a3 = loadA(gid + 8, khi);

    uint32_t b0 = 0, b1 = 0;

    if (hypothesis == 0) {
        // CONTROL: direct gmem-style B load matching v66's pattern.
        // b0 = B[n=gid, k=klo..klo+3]  — but B is stored as [k][n]
        // So we need to gather byte-wise: for each k in klo..klo+3, read B[k][gid]
        // and pack into uint32.
        uint8_t bb0[4], bb1[4];
        for (int j = 0; j < 4; j++) {
            bb0[j] = B[(klo + j) * 8 + gid];
            bb1[j] = B[(khi + j) * 8 + gid];
        }
        b0 = *(uint32_t*)bb0;
        b1 = *(uint32_t*)bb1;
    } else if (hypothesis == 1) {
        // HYPOTHESIS: ldmatrix.trans.x2.m8n8.b16 directly produces B-frag for QMMA.
        // ldmatrix expects an 8-row source address per lane: addr[L] = &SMEM[row L].
        // For .x2, 2 8x8 b16 tiles are loaded. With .trans, columns become rows.
        //
        // Source address layout for .x2.trans:
        //   First tile (b0): lanes 0..7 supply addresses of rows 0..7 of an 8x8 b16 source.
        //   Second tile (b1): lanes 8..15 supply addresses of rows 0..7 of another 8x8 source.
        //   With .trans, what was COL within the 8x8 tile becomes the lane's register byte.
        //
        // For FP8 (8-bit), each "row" of the 8x8 b16 source covers 16 FP8 bytes.
        // We treat B[k_row][n_col] as: row stride = 8 bytes (8 n-cols × 1 byte),
        // 8 rows make 64 bytes = 32 FP8 elements in 16-element n strides... this is
        // hairy. The cleanest: B layout for trans.x2 is 8 rows × 16 b16 cols. Each b16
        // col packs 2 FP8 bytes. So 8 rows × 32 FP8 = matches our 32×8 with reordering.
        //
        // For each lane in [0..15], supply address &B[(lane % 8) * stride].
        // For lanes [16..31], unused for .x2 (only first 16 supply).
        //
        // After .trans, each lane gets 2 .b32 representing transposed tile data.
        //
        // We construct address: lane L supplies row (L % 8) of one tile.
        //   lanes 0..7  → rows 0..7 of tile 0 (k=0..7)
        //   lanes 8..15 → rows 0..7 of tile 1 (k=8..15)
        //
        // Stride within row: 16 bytes (8 b16 = 16 FP8).
        uint32_t b_smem_base = __cvta_generic_to_shared(B);
        int src_lane = lane & 15;            // first 16 lanes carry addresses
        int tile_id = (lane >> 3) & 1;       // tile 0 or 1
        int row_in_tile = src_lane & 7;
        // Each row covers 16 bytes; first tile starts at k=0, second at k=16
        // (because we have K=32 and want both halves).
        // Hmm — but the source is B[k][n] with row stride = 8 bytes (n_dim = 8).
        // To get 16-byte rows we'd need n_dim = 16, which we don't have.
        //
        // ldmatrix.x2 expects 16 lanes supplying addresses for 2 tiles of 8 rows each.
        // Each row is 16 bytes (8 b16 elements). With n_dim=8 FP8, a row is 8 bytes.
        // → mismatch. ldmatrix.x2 needs a wider source.
        //
        // CONCLUSION (to be verified by running):
        // direct ldmatrix.trans onto an 8-wide-n FP8 V layout doesn't match natively.
        // We'd need to widen the SMEM layout to 16-wide n strides OR use ldmatrix.x4.
        //
        // For this probe we report HYPOTHESIS=1 as a hypothesis that's likely
        // dimensionally-wrong; if it matches against expected values, layout truly
        // happens to align.
        uint32_t addr = b_smem_base + row_in_tile * 8 + tile_id * 128;
        asm volatile(
            "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];"
            : "=r"(b0), "=r"(b1)
            : "r"(addr));
    }

    float d0, d1, d2, d3;
    qmma_m16n8k32_f32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);

    D_out[lane * 4 + 0] = d0;
    D_out[lane * 4 + 1] = d1;
    D_out[lane * 4 + 2] = d2;
    D_out[lane * 4 + 3] = d3;
}

static int verify(const float *hD, const char *name, int K)
{
    printf("  -- %s --\n", name);
    int ok = 0;
    for (int lane = 0; lane < 32; ++lane) {
        int tid = lane & 3;
        float e0 = (float)K * (2*tid);
        float e1 = (float)K * (2*tid + 1);
        float e2 = e0, e3 = e1;  // row-independent since A=1
        bool pass = (hD[lane*4+0] == e0) && (hD[lane*4+1] == e1)
                 && (hD[lane*4+2] == e2) && (hD[lane*4+3] == e3);
        if (pass) ok++;
    }
    int printed = 0, want = (ok == 32) ? 1 : 0;
    for (int lane = 0; lane < 32 && printed < 4; ++lane) {
        int gid = lane >> 2, tid = lane & 3;
        float e0 = (float)K * (2*tid), e1 = (float)K * (2*tid + 1);
        bool pass = (hD[lane*4+0] == e0) && (hD[lane*4+1] == e1)
                 && (hD[lane*4+2] == e0) && (hD[lane*4+3] == e1);
        int pi = pass ? 1 : 0;
        if (pi != want) continue;
        printf("     [%s] lane %2d (gid=%d,tid=%d): d=[%.0f,%.0f,%.0f,%.0f]  exp=[%.0f,%.0f,%.0f,%.0f]\n",
            pass ? "match" : " MISS", lane, gid, tid,
            hD[lane*4+0], hD[lane*4+1], hD[lane*4+2], hD[lane*4+3],
            e0, e1, e0, e1);
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
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("Device: %s\n\n", p.name);

    printf("############################################################\n");
    printf("# ldmatrix.trans → QMMA m16n8k32 B-frag layout probe\n");
    printf("############################################################\n");

    printf("=== Test: A=1, B[k][n]=n  →  D[r][n] = 32*n ===\n");

    printf("\nHypothesis 0 (CONTROL): direct byte-gather load of B[n,k] uint32\n");
    CK(cudaMemset(dD, 0, 128*sizeof(float)));
    probe_ldsm_trans<<<1, 32>>>(dD, 0);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hD, dD, 128*sizeof(float), cudaMemcpyDeviceToHost));
    int ctrl = verify(hD, "control (byte-gather)", 32);

    printf("Hypothesis 1: ldmatrix.x2.trans.m8n8.b16 from B[k][n] FP8 SMEM\n");
    CK(cudaMemset(dD, 0, 128*sizeof(float)));
    probe_ldsm_trans<<<1, 32>>>(dD, 1);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hD, dD, 128*sizeof(float), cudaMemcpyDeviceToHost));
    int hyp1 = verify(hD, "ldmatrix.x2.trans   ", 32);

    printf("=== Summary ===\n");
    printf("  Control byte-gather:        %d/32  (sanity: should be 32/32)\n", ctrl);
    printf("  ldmatrix.x2.trans hypothesis: %d/32\n", hyp1);
    if (hyp1 == 32) printf("  → ldmatrix.trans matches QMMA B-frag. Safe to use in v68.\n");
    else printf("  → ldmatrix.trans does NOT directly match. Need byte rearrangement, or this lever blocked.\n");

    cudaFree(dD);
    return 0;
}
