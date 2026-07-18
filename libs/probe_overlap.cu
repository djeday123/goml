// =====================================================================
//  probe_overlap.cu — E3-минимальный: structural overlap decision probe.
//
//  Question (per Vugar's pushback after probe_cast D revealed 99.4% tensor
//  ceiling for isolated MMA #1 vs real dK 21.82%):
//   Can the 77.6pp tensor-idle structural tax (transpose+barriers between
//   MMA chains) be recovered by interleaving transpose work into the MMA
//   issue stream — at FAIR OCCUPANCY (probe forced equal smem)?
//
//  Three variants in one binary (select via -DPROBE_VARIANT):
//   BASE   (0) — pure MMA chain, no barriers, no transpose work
//                Tensor ceiling reference (≈ probe_cast A's 99.44%).
//   SERIAL (1) — MMA chain split into N_CHUNK groups; between each chunk:
//                __syncthreads + bulk LDS+STS transpose work (mimics dK
//                Phase 1.5 between MMA #1 and MMA #2).
//                Tensor% should drop sharply if structural tax is real.
//   OVERLAP(2) — same total MMA + same total transpose bytes as SERIAL,
//                but transpose work issued INSIDE the MMA loop (1 LDS+STS
//                per MMA). Hardware ILP should dual-issue tensor + LDS
//                pipes. Single barrier at end-of-qt.
//                If tensor% recovers + wall drops vs SERIAL → lever lives.
//
//  Fair occupancy: all 3 variants allocate identical smem (40 KB) →
//  identical 2 blocks/SM expected. No occupancy confound.
//
//  Decision rule (Vugar):
//   OVERLAP tensor% > SERIAL AND OVERLAP wall < SERIAL → B3.4 = restructure
//   dK pipeline (then check projection: +8KB in real dK → 1 block/SM cost
//   per P2/P3 history, weigh overlap benefit vs occupancy loss).
//   OVERLAP ≈ SERIAL on either metric → structural ceiling, 175 T floor.
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define VARIANT_BASE    0
#define VARIANT_SERIAL  1
#define VARIANT_OVERLAP 2

#ifndef PROBE_VARIANT
#define PROBE_VARIANT VARIANT_BASE
#endif

#define BR        64
#define BC        64
#define HD        128
#define THREADS   128

// MMA layout: 8 ks × 8 ni = 64 MMAs per qt per warp (matches probe_cast A).
// Split into N_CHUNK=4 chunks of 2 ks each (= 16 MMAs per chunk).
#define N_KS         8
#define N_NI         8
#define KS_PER_CHUNK 2
#define N_CHUNK      (N_KS / KS_PER_CHUNK)   // 4 chunks

// Transpose work matches dK Phase 1.5 (~16 uint32 R+W per lane per qt).
// SERIAL: bulk between chunks = 4 R+W per chunk.
// OVERLAP: 1 R+W per MMA = 64 R+W per lane per qt (4× more — but spread
// across MMA latency so dual-issue can hide). Adjust transpose iters per
// MMA so total bytes match: 16 transpose ops / 64 MMAs = 1 R+W per 4 MMAs.
#define TRANSPOSE_RW_PER_QT       16        // total R+W pairs per lane per qt
#define TRANSPOSE_RW_PER_CHUNK    (TRANSPOSE_RW_PER_QT / N_CHUNK)   // 4
#define TRANSPOSE_RW_PER_MMA_NUM  TRANSPOSE_RW_PER_QT               // numerator
#define MMAS_PER_QT               (N_KS * N_NI)                     // 64
// In OVERLAP, do 1 transpose op every (64/16)=4 MMAs.
#define MMAS_PER_TRANSPOSE_OP     (MMAS_PER_QT / TRANSPOSE_RW_PER_QT)  // 4

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
//   Smem layout (40 KB, identical across variants → 2 blocks/SM):
//     smdO 16 KB + smV 8 KB + smQ_scratch 8 KB + smQ_T_scratch 8 KB
//   smQ/smQ_T scratch present in all variants (BASE ignores them) — sole
//   purpose is identical smem footprint → identical occupancy.
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
    __half  *smdO    = reinterpret_cast<__half*>(smem_raw);                    // 16 KB
    uint8_t *smV     = reinterpret_cast<uint8_t*>(smdO + BR * HD);             // 8 KB
    uint8_t *smQ     = smV + BC * HD;                                          // 8 KB
    uint8_t *smQ_T   = smQ + BC * HD;                                          // 8 KB

    // ---- one-shot smem fill ----
    for (int i = tid; i < BC * HD; i += THREADS) smV[i] = V_g[i];
    for (int i = tid; i < BR * HD; i += THREADS) smdO[i] = dO_g[i];
    for (int i = tid; i < BC * HD; i += THREADS) smQ[i] = V_g[i];   // dummy
    for (int i = tid; i < BC * HD; i += THREADS) smQ_T[i] = 0;
    __syncthreads();

    // ---- preload dO into per-lane regs (constant across qt) ----
    uint32_t dOr[N_KS][4];
    #pragma unroll
    for (int ks = 0; ks < N_KS; ++ks) {
        int m_lo = wid * 16 + l_div4 + 0;
        int m_hi = wid * 16 + l_div4 + 8;
        int k_lo = ks * 16 + l_mod4 * 2 + 0;
        int k_hi = ks * 16 + l_mod4 * 2 + 8;
        dOr[ks][0] = *reinterpret_cast<uint32_t*>(&smdO[m_lo * HD + k_lo]);
        dOr[ks][1] = *reinterpret_cast<uint32_t*>(&smdO[m_hi * HD + k_lo]);
        dOr[ks][2] = *reinterpret_cast<uint32_t*>(&smdO[m_lo * HD + k_hi]);
        dOr[ks][3] = *reinterpret_cast<uint32_t*>(&smdO[m_hi * HD + k_hi]);
    }

    float acc[N_NI][4];
    #pragma unroll
    for (int ni = 0; ni < N_NI; ++ni)
        #pragma unroll
        for (int s = 0; s < 4; ++s) acc[ni][s] = 0.0f;

    // Per-lane scratch indexing for transpose work (16 R+W per qt total).
    // Each lane reads/writes its own slot range → no inter-lane conflict.
    // Lane t covers byte range [t*64 .. t*64+63] in smQ/smQ_T (8 KB / 128 lanes).
    uint32_t mma_counter = 0;
    uint32_t transpose_counter = 0;

    // ===== stationary qt loop =====
    for (int qt = 0; qt < qt_iters; ++qt) {
        mma_counter = 0;
        transpose_counter = 0;

        #pragma unroll
        for (int chunk = 0; chunk < N_CHUNK; ++chunk) {

            // ---- MMA chunk (KS_PER_CHUNK × N_NI MMAs) ----
            #pragma unroll
            for (int ks_in = 0; ks_in < KS_PER_CHUNK; ++ks_in) {
                int ks = chunk * KS_PER_CHUNK + ks_in;
                #pragma unroll
                for (int ni = 0; ni < N_NI; ++ni) {
                    int n    = ni * 8 + l_div4;
                    int k_lo = ks * 16 + l_mod4 * 2 + 0;
                    int k_hi = ks * 16 + l_mod4 * 2 + 8;
                    uint16_t v0 = *reinterpret_cast<uint16_t*>(&smV[n * HD + k_lo]);
                    uint16_t v1 = *reinterpret_cast<uint16_t*>(&smV[n * HD + k_hi]);
                    uint32_t B0 = e4m3x2_to_f16x2(v0);
                    uint32_t B1 = e4m3x2_to_f16x2(v1);

                    mma_m16n8k16_f32(
                        acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3],
                        dOr[ks][0], dOr[ks][1], dOr[ks][2], dOr[ks][3],
                        B0, B1,
                        acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3]);

#if PROBE_VARIANT == VARIANT_OVERLAP
                    // ----- Interleaved transpose: 1 R+W per MMAS_PER_TRANSPOSE_OP -----
                    // Issue alongside MMA so compiler dual-issues LDS/STS on smem unit
                    // while tensor pipe runs MMA. Stride per-lane access to avoid bank
                    // conflict cluster (each lane owns disjoint scratch region).
                    if ((mma_counter % MMAS_PER_TRANSPOSE_OP) == 0
                        && transpose_counter < TRANSPOSE_RW_PER_QT) {
                        int off = (tid * 64 + transpose_counter * 4) & 0x1FFF;  // 8KB mask
                        uint32_t v = *reinterpret_cast<uint32_t*>(&smQ[off]);
                        *reinterpret_cast<uint32_t*>(&smQ_T[off]) = v;
                        ++transpose_counter;
                    }
                    ++mma_counter;
#endif
                }
            }

#if PROBE_VARIANT == VARIANT_SERIAL
            // ---- Bulk transpose phase between chunks ----
            __syncthreads();
            #pragma unroll
            for (int t = 0; t < TRANSPOSE_RW_PER_CHUNK; ++t) {
                int off = (tid * 64 + (transpose_counter + t) * 4) & 0x1FFF;
                uint32_t v = *reinterpret_cast<uint32_t*>(&smQ[off]);
                *reinterpret_cast<uint32_t*>(&smQ_T[off]) = v;
            }
            transpose_counter += TRANSPOSE_RW_PER_CHUNK;
            __syncthreads();
#endif
        }

#if PROBE_VARIANT != VARIANT_BASE
        // End-of-qt barrier (SERIAL has it implicitly from last chunk barrier;
        // OVERLAP needs one to publish smQ_T writes for next-qt's reads in
        // realistic dK; BASE has nothing to publish).
        __syncthreads();
#endif
    }

    float sink = 0.0f;
    #pragma unroll
    for (int ni = 0; ni < N_NI; ++ni)
        #pragma unroll
        for (int s = 0; s < 4; ++s) sink += acc[ni][s];
    // Also sink last transpose write to prevent dead-elim
    sink += (float)smQ_T[(tid * 4) & 0x1FFF];
    out[blockIdx.x * THREADS + tid] = sink;
}

int main(int argc, char **argv)
{
    int qt_iters = (argc >= 2) ? std::atoi(argv[1]) : 128;
    int warmup   = (argc >= 3) ? std::atoi(argv[2]) : 5;
    int iters    = (argc >= 4) ? std::atoi(argv[3]) : 20;
    int blocks   = (argc >= 5) ? std::atoi(argv[4]) : 128 * 128;
    int dummy_kb = (argc >= 6) ? std::atoi(argv[5]) : 0;

    const size_t base_smem = BR * HD * sizeof(__half)   // smdO 16K
                           + BC * HD                     // smV  8K
                           + BC * HD                     // smQ  8K
                           + BC * HD;                    // smQ_T 8K  → 40K total
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

    const char *vname =
        (PROBE_VARIANT == VARIANT_BASE)    ? "BASE (no transpose, no barriers)"  :
        (PROBE_VARIANT == VARIANT_SERIAL)  ? "SERIAL (chunked MMA + barrier+bulk transpose)" :
                                             "OVERLAP (transpose interleaved into MMA)";
    printf("probe_overlap: variant=%s smem=%zuKB dummy_pad=%dKB occupancy=%d/SM "
           "blocks=%d qt_iters=%d iters=%d avg_ms=%.4f\n",
           vname, smem_bytes / 1024, dummy_kb, active_blocks,
           blocks, qt_iters, iters, ms / iters);

    std::free(V_h); std::free(dO_h);
    CK(cudaFree(V_d)); CK(cudaFree(dO_d)); CK(cudaFree(out_d));
    return 0;
}
