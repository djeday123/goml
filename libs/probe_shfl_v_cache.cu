// =====================================================================
//  probe_shfl_v_cache.cu — isolated shfl-vs-LDS cost probe for dK MMA #1.
//
//  Purpose (B3.3.b lever screening): the dK kernel's MMA #1 (dO·V^T) reads
//  V from smV via LDS.U16 + cvt.rn.f16x2.e4m3x2 per fragment. parse_stalls_top.py
//  attributed top-10 mio to LDS/LDS.U16 (smV byte-range 0x2000–0x4000) and
//  top-10 short_sb to F2FP.F16.E4M3.UNPACK_B casts + MMAs — DISJOINT sets.
//  Question: does a 32-uint32-per-lane V cache + shfl cooperative read
//  pattern reduce mio enough to be worth the reg pressure (lever (b))?
//
//  Probe variants (selected at compile time via -DPROBE_VARIANT=N):
//    A (0) — pure LDS baseline: 128 LDS.U16 + 128 cvt per qt per lane
//    B (1) — own-cache + shfl: 32 own-regs + 96 shfl.sync.idx per qt per lane
//    C (2) — own-cache + LDS hybrid: 32 own-regs + 96 LDS.U16+cvt per qt per lane
//
//  All three run identical MMA grid (8 ks × 8 ni) and identical dO source.
//  Differences live ONLY in the V-fetch path → mio/short_sb delta is the lever.
//
//  Corrections per Vugar's review:
//   (1) Stationary regime: qt loop iterates QT_ITERS ≥ 128 (matches dK real
//       n_qt=128 at sl=8192, ensures cached-data-reuse-stationary stall regime,
//       not single-shot kernel-warmup transient).
//   (2) Realistic shfl pattern: for B, shfl source lane varies per (ni, ks, lane)
//       — owner = ((ni*11 + ks*5) + l_mod4) & 31. Verified distinct sources
//       across ks (12,17,22,27,0,5,10,15 for ni=4 l_mod4=0). NOT a broadcast.
//   (3) NCu counter: use --page source dump (same path as parse_stalls_top.py)
//       — shfl.sync register-dep stalls fall under short_scoreboard on sm_120a;
//       PC-attribution via smsp__pcsamp_warps_issue_stalled_short_scoreboard.sum
//       lets us separate shfl-stall from MMA-F32-acc-stall (both short_sb class
//       but distinct SASS lines: SHFL.IDX vs HMMA.F32 / QMMA).
//
//  Output: per-variant avg_ms wall + ptxas regs/spill. NCu source dump
//  inspected via existing parse_stalls_top.py to break down mio/short_sb per
//  SASS line.
//
//  Decision rule (Vugar's法):
//   B << A on wall AND mio drops ≥ 10% → build full (b) into dK
//   B ≈ A on wall (delta < 3%) → (b) dead, accept dK 175 T, move to dQ
//   C ~3.6% better than A → confirms hit-rate model (25% × 14.4% mio)
//   C ≈ A → mio not the bottleneck even at 50% LDS reduction → kill (b) too
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define VARIANT_A 0   // pure LDS baseline
#define VARIANT_B 1   // 32-reg cache + shfl for misses
#define VARIANT_C 2   // 32-reg cache + LDS for misses (hit-rate sanity)

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
//   Geometry identical to dK MMA #1 step D:
//     Br=64, Bc=64, Hd=128, 128 threads (4 warps), m16n8k16 over (ks × ni).
//   smV NOT swizzled (matches dK smV layout, line 152 fa_bwd_dk.cu).
//   smdO row-major (matches dK smdO layout, line 198).
//   dO loaded ONCE into per-lane regs outside qt loop (no smdO re-LDS noise).
//   V re-fetched per qt according to variant (where the differences live).
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
    uint8_t *smV  = smem_raw;                                            // 8 KB
    __half  *smdO = reinterpret_cast<__half*>(smV + BC * HD);            // 16 KB

    // ---- one-shot smem fill ----
    for (int i = tid; i < BC * HD;          i += THREADS) smV[i]  = V_g[i];
    for (int i = tid; i < BR * HD;          i += THREADS) smdO[i] = dO_g[i];
    __syncthreads();

#if PROBE_VARIANT != VARIANT_A
    // ---- preload V_cache (32 uint32 per lane = 64 fp16 = 25% of MMA #1 needs) ----
    // Cache holds fragments for ni=0..3 × ks=0..3 × {B0,B1} = 32 entries.
    // For cached fragment at (ni, ks, frag), lane stores ITS OWN (n=ni*8+l_div4,
    // k=ks*16+l_mod4*2+(frag*8)) — same access function as real MMA loop below.
    uint32_t Vcache[32];
    #pragma unroll
    for (int ni_c = 0; ni_c < 4; ++ni_c) {
        #pragma unroll
        for (int ks_c = 0; ks_c < 4; ++ks_c) {
            int n     = ni_c * 8 + l_div4;
            int k_lo  = ks_c * 16 + l_mod4 * 2 + 0;
            int k_hi  = ks_c * 16 + l_mod4 * 2 + 8;
            uint16_t v0 = *reinterpret_cast<uint16_t*>(&smV[n * HD + k_lo]);
            uint16_t v1 = *reinterpret_cast<uint16_t*>(&smV[n * HD + k_hi]);
            int idx_b0 = (ni_c << 3) | (ks_c << 1) | 0;
            int idx_b1 = idx_b0 | 1;
            Vcache[idx_b0] = e4m3x2_to_f16x2(v0);
            Vcache[idx_b1] = e4m3x2_to_f16x2(v1);
        }
    }
#endif

    // ---- preload dO into per-lane regs (constant across all qt) ----
    // Mirrors dK step D dO read pattern (line 310-313 fa_bwd_dk.cu).
    uint32_t dOr[8][4];   // [ks][A0..A3]
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

    // ---- accumulator (per ni F32 quad, sunk to gmem at end) ----
    float acc[8][4];
    #pragma unroll
    for (int ni = 0; ni < 8; ++ni)
        #pragma unroll
        for (int s = 0; s < 4; ++s) acc[ni][s] = 0.0f;

    // ===== stationary qt loop =====
    // qt iter count ≥ 128 guarantees cached-data-reuse-stationary stall regime.
    for (int qt = 0; qt < qt_iters; ++qt) {
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            #pragma unroll
            for (int ni = 0; ni < 8; ++ni) {
                int n    = ni * 8 + l_div4;
                int k_lo = ks * 16 + l_mod4 * 2 + 0;
                int k_hi = ks * 16 + l_mod4 * 2 + 8;
                uint32_t B0, B1;

#if PROBE_VARIANT == VARIANT_A
                // === A: pure LDS + cvt (baseline, current dK behavior) ===
                uint16_t v0 = *reinterpret_cast<uint16_t*>(&smV[n * HD + k_lo]);
                uint16_t v1 = *reinterpret_cast<uint16_t*>(&smV[n * HD + k_hi]);
                B0 = e4m3x2_to_f16x2(v0);
                B1 = e4m3x2_to_f16x2(v1);
#else
                bool cached = (ni < 4) && (ks < 4);
                if (cached) {
                    int idx_b0 = (ni << 3) | (ks << 1) | 0;
                    int idx_b1 = idx_b0 | 1;
                    B0 = Vcache[idx_b0];
                    B1 = Vcache[idx_b1];
                } else {
  #if PROBE_VARIANT == VARIANT_B
                    // === B: shfl.sync.idx, DIFFERENT source per (ni, ks, lane) ===
                    // owner derived to produce distinct src lanes within warp
                    // (NOT broadcast — the only meaningful test of shfl-unit cost).
                    int owner  = ((ni * 11 + ks * 5) + l_mod4) & 31;
                    int idx_b0 = ((ni & 3) << 3) | ((ks & 3) << 1) | 0;
                    int idx_b1 = idx_b0 | 1;
                    B0 = __shfl_sync(0xFFFFFFFF, Vcache[idx_b0], owner);
                    B1 = __shfl_sync(0xFFFFFFFF, Vcache[idx_b1], owner);
  #else // VARIANT_C
                    // === C: LDS+cvt for misses (hit-rate sanity baseline) ===
                    // Same cache miss rate as B, but no shfl — pure LDS path.
                    // If C ≈ A → mio drop from 25% LDS cut is not measurable
                    //          → kill (b) lever entirely.
                    // If C is ~3.6% faster than A → hit-rate model validated.
                    uint16_t v0 = *reinterpret_cast<uint16_t*>(&smV[n * HD + k_lo]);
                    uint16_t v1 = *reinterpret_cast<uint16_t*>(&smV[n * HD + k_hi]);
                    B0 = e4m3x2_to_f16x2(v0);
                    B1 = e4m3x2_to_f16x2(v1);
  #endif
                }
#endif
                mma_m16n8k16_f32(
                    acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3],
                    dOr[ks][0], dOr[ks][1], dOr[ks][2], dOr[ks][3],
                    B0, B1,
                    acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3]);
            }
        }
    }

    // ---- sink acc → gmem (prevents dead-code elimination) ----
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
    int blocks   = (argc >= 5) ? std::atoi(argv[4]) : 128 * 128;   // dK grid

    const size_t V_bytes   = BC * HD;                              // 8 KB
    const size_t dO_bytes  = BR * HD * sizeof(__half);             // 16 KB
    const size_t out_bytes = (size_t)blocks * THREADS * sizeof(float);
    const size_t smem_bytes = V_bytes + dO_bytes;                  // 24 KB / block

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

    const char *vname =
        (PROBE_VARIANT == VARIANT_A) ? "A (pure LDS)"             :
        (PROBE_VARIANT == VARIANT_B) ? "B (own-cache + shfl)"     :
                                       "C (own-cache + LDS)";
    printf("probe_shfl_v_cache: variant=%s blocks=%d threads=%d qt_iters=%d "
           "iters=%d avg_ms=%.4f\n",
           vname, blocks, THREADS, qt_iters, iters, ms / iters);

    std::free(V_h); std::free(dO_h);
    CK(cudaFree(V_d)); CK(cudaFree(dO_d)); CK(cudaFree(out_d));
    return 0;
}
