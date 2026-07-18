// =====================================================================
//  probe_cast_v.cu — isolate cast-V cost (lever (c) re-examination).
//
//  WHY (per Vugar's pushback on probe_shfl_v_cache):
//   Variant C of (b)-probe removed 25% of MMA #1 V LDS and got 0% wall change.
//   → top-mio source (LDS) is NOT the wall driver — mio queue has headroom.
//   → original dK NCu top-stall attribution (mio 38%, short_sb 25%) is
//     "where stalls happen" not "what holds wall". Each suspect must be
//     killed via separating experiment, dV-style (P2/P3/α discriminated).
//
//  D variant: V stored as FP16 in smem (16 KB instead of 8 KB FP8).
//  Cast cvt.rn.f16x2.e4m3x2 disappears from MMA #1 inner loop entirely.
//  Per (ks, ni): 2 LDS.U32 directly → uint32 B0, B1 (no F2FP).
//  Same NUMBER of LDS issues (128/qt/lane), 2× bytes per issue.
//
//  Occupancy confounder: D needs +8 KB smem → may push to 1 block/SM in
//  real dK. Probe controls this via runtime DUMMY_PAD_KB padding — sweep
//  smem until cudaOccupancyMaxActiveBlocksPerMultiprocessor reports 1.
//  Then run A_padded and D_padded at SAME occupancy → pure cast cost.
//
//  Predicted outcomes:
//    short_sb dominated by cast → D << A_padded (cast removed, no equiv)
//    short_sb dominated by MMA F32-acc → D ≈ A_padded (cast removal irrelevant)
//    mio bandwidth-bound → D > A_padded (2× LDS bytes hurts)
//    mio issue-bound only → D ≈ A_padded
//
//  Reports wall + cudaOccupancy + per-block smem. NCu stall breakdown
//  via separate make target (Makefile.probe_cast → ncu_metrics).
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define VARIANT_A 0   // FP8 V in smem + inline cast (current dK behavior)
#define VARIANT_D 3   // FP16 V in smem, no cast

#ifndef PROBE_VARIANT
#define PROBE_VARIANT VARIANT_A
#endif

#define BR        64
#define BC        64
#define HD        128
#define THREADS   128

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) {                \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                    \
            cudaGetErrorString(e)); std::exit(1); }} while (0)

__device__ __forceinline__ void mma_m16n8k16_f32(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint16_t fp8x2) {
    uint32_t r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(fp8x2));
    return r;
}

// =====================================================================
// Probe kernel.
//   Smem layout depends on variant + runtime dummy padding:
//     A:  smV[8KB FP8]  + smdO[16KB FP16] + dummy[N KB]  =  24+N KB
//     D:  smV[16KB FP16] + smdO[16KB FP16] + dummy[N KB] =  32+N KB
//   Inner qt loop identical structure; only V-fetch differs.
// =====================================================================
__global__ void probe_kernel(
    const uint8_t * __restrict__ V_g,
    const __half  * __restrict__ dO_g,
    float         * __restrict__ out,
    int qt_iters)
{
    const int tid    = threadIdx.x;
    const int wid    = tid >> 5;
    const int lane   = tid & 31;
    const int l_div4 = lane >> 2;
    const int l_mod4 = lane & 3;

    extern __shared__ uint8_t smem_raw[];
#if PROBE_VARIANT == VARIANT_D
    __half *smV  = reinterpret_cast<__half*>(smem_raw);                           // 16 KB
    __half *smdO = reinterpret_cast<__half*>(smem_raw + BC * HD * sizeof(__half));// 16 KB
#else
    uint8_t *smV = smem_raw;                                                      // 8 KB
    __half  *smdO = reinterpret_cast<__half*>(smV + BC * HD);                     // 16 KB
#endif
    // Note: dummy pad bytes after smdO are allocated but never touched —
    // sole purpose is to force lower occupancy via smem ceiling.

    // ---- one-shot smem fill ----
#if PROBE_VARIANT == VARIANT_D
    // Cast FP8 → FP16 at load time (one-shot, not in qt loop).
    // Read uint16_t pairs (2 FP8), cvt to fp16x2, store uint32.
    for (int i = tid; i < (BC * HD) / 2; i += THREADS) {
        uint16_t u8x2 = *reinterpret_cast<const uint16_t*>(&V_g[i * 2]);
        uint32_t fp16x2 = e4m3x2_to_f16x2(u8x2);
        *reinterpret_cast<uint32_t*>(&smV[i * 2]) = fp16x2;
    }
#else
    for (int i = tid; i < BC * HD; i += THREADS) smV[i] = V_g[i];
#endif
    for (int i = tid; i < BR * HD; i += THREADS) smdO[i] = dO_g[i];
    __syncthreads();

    // ---- preload dO into per-lane regs (constant across qt) ----
    uint32_t dOr[8][4];
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        int m_lo = wid * 16 + l_div4 + 0;
        int m_hi = wid * 16 + l_div4 + 8;
        int k_lo = ks * 16 + l_mod4 * 2 + 0;
        int k_hi = ks * 16 + l_mod4 * 2 + 8;
        dOr[ks][0] = *reinterpret_cast<uint32_t*>(&smdO[m_lo * HD + k_lo]);
        dOr[ks][1] = *reinterpret_cast<uint32_t*>(&smdO[m_hi * HD + k_lo]);
        dOr[ks][2] = *reinterpret_cast<uint32_t*>(&smdO[m_lo * HD + k_hi]);
        dOr[ks][3] = *reinterpret_cast<uint32_t*>(&smdO[m_hi * HD + k_hi]);
    }

    float acc[8][4];
    #pragma unroll
    for (int ni = 0; ni < 8; ++ni)
        #pragma unroll
        for (int s = 0; s < 4; ++s) acc[ni][s] = 0.0f;

    // ===== stationary qt loop (≥128 iters for steady-state stall regime) =====
    for (int qt = 0; qt < qt_iters; ++qt) {
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            #pragma unroll
            for (int ni = 0; ni < 8; ++ni) {
                int n    = ni * 8 + l_div4;
                int k_lo = ks * 16 + l_mod4 * 2 + 0;
                int k_hi = ks * 16 + l_mod4 * 2 + 8;
                uint32_t B0, B1;
#if PROBE_VARIANT == VARIANT_D
                // === D: FP16 direct, no cast (2 LDS.U32 per fragment) ===
                B0 = *reinterpret_cast<uint32_t*>(&smV[n * HD + k_lo]);
                B1 = *reinterpret_cast<uint32_t*>(&smV[n * HD + k_hi]);
#else
                // === A: FP8 + cvt (2 LDS.U16 + 2 F2FP per fragment) ===
                uint16_t v0 = *reinterpret_cast<uint16_t*>(&smV[n * HD + k_lo]);
                uint16_t v1 = *reinterpret_cast<uint16_t*>(&smV[n * HD + k_hi]);
                B0 = e4m3x2_to_f16x2(v0);
                B1 = e4m3x2_to_f16x2(v1);
#endif
                mma_m16n8k16_f32(
                    acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3],
                    dOr[ks][0], dOr[ks][1], dOr[ks][2], dOr[ks][3],
                    B0, B1,
                    acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3]);
            }
        }
    }

    float sink = 0.0f;
    #pragma unroll
    for (int ni = 0; ni < 8; ++ni)
        #pragma unroll
        for (int s = 0; s < 4; ++s) sink += acc[ni][s];
    out[blockIdx.x * THREADS + tid] = sink;
}

int main(int argc, char **argv)
{
    int qt_iters = (argc >= 2) ? std::atoi(argv[1]) : 128;
    int warmup   = (argc >= 3) ? std::atoi(argv[2]) : 5;
    int iters    = (argc >= 4) ? std::atoi(argv[3]) : 20;
    int blocks   = (argc >= 5) ? std::atoi(argv[4]) : 128 * 128;
    int dummy_kb = (argc >= 6) ? std::atoi(argv[5]) : 0;

#if PROBE_VARIANT == VARIANT_D
    const size_t base_smem = BC * HD * sizeof(__half) + BR * HD * sizeof(__half); // 32 KB
    const char *vname = "D (FP16 V, no cast)";
#else
    const size_t base_smem = BC * HD + BR * HD * sizeof(__half);                  // 24 KB
    const char *vname = "A (FP8 V + cast)";
#endif
    const size_t smem_bytes = base_smem + (size_t)dummy_kb * 1024;

    const size_t V_bytes   = BC * HD;
    const size_t dO_bytes  = BR * HD * sizeof(__half);
    const size_t out_bytes = (size_t)blocks * THREADS * sizeof(float);

    uint8_t *V_h  = (uint8_t*)std::malloc(V_bytes);
    __half  *dO_h = (__half*)std::malloc(dO_bytes);
    for (size_t i = 0; i < V_bytes; ++i)
        V_h[i] = (uint8_t)(i & 0xFF);
    for (size_t i = 0; i < BR * HD; ++i)
        dO_h[i] = __float2half_rn((float)((i & 0xFF) / 128.0f - 1.0f));

    uint8_t *V_d;  __half *dO_d;  float *out_d;
    CK(cudaMalloc(&V_d,   V_bytes));
    CK(cudaMalloc(&dO_d,  dO_bytes));
    CK(cudaMalloc(&out_d, out_bytes));
    CK(cudaMemcpy(V_d,  V_h,  V_bytes,  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dO_d, dO_h, dO_bytes, cudaMemcpyHostToDevice));

    CK(cudaFuncSetAttribute(probe_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            (int)smem_bytes));

    int active_blocks = -1;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, probe_kernel, THREADS, smem_bytes));

    for (int i = 0; i < warmup; ++i) {
        probe_kernel<<<blocks, THREADS, smem_bytes>>>(V_d, dO_d, out_d, qt_iters);
    }
    CK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) {
        probe_kernel<<<blocks, THREADS, smem_bytes>>>(V_d, dO_d, out_d, qt_iters);
    }
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, t0, t1));

    printf("probe_cast_v: variant=%s smem=%zuKB dummy_pad=%dKB occupancy=%d/SM "
           "blocks=%d qt_iters=%d iters=%d avg_ms=%.4f\n",
           vname, smem_bytes / 1024, dummy_kb, active_blocks,
           blocks, qt_iters, iters, ms / iters);

    std::free(V_h); std::free(dO_h);
    CK(cudaFree(V_d)); CK(cudaFree(dO_d)); CK(cudaFree(out_d));
    return 0;
}
