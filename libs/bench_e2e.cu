// =====================================================================
//  bench_e2e.cu — B5 step 3: end-to-end backward pipeline wall + breakdown.
//
//  Pipeline: D-precompute → dV → dK → dQ (D first; dV/dK/dQ independent after D).
//  Per-kernel timing via cudaEvent_t (records between kernels in same stream).
//
//  Outputs:
//    - per-kernel avg wall in chain (vs isolated reference)
//    - % wall breakdown
//    - weighted total TFLOPS (FLOPS_dV + FLOPS_dK + FLOPS_dQ over total wall)
//
//  FLOPS arithmetic (Tri Dao Variant 3):
//    dV: 2 MMA chains (Q·K^T recompute + P^T·dO)         = 4·bh·sl²·hd·cf
//    dK: 3 MMA chains (Q·K^T + dO·V^T + dS^T·Q)          = 6·bh·sl²·hd·cf
//    dQ: 3 MMA chains (Q·K^T + dO·V^T + dS·K)            = 6·bh·sl²·hd·cf
//    D-precompute: bh·sl·hd (element-wise, negligible)
//    Total backward = 16·bh·sl²·hd·cf
//
//  CLI: ./bench_e2e <bh> <sl> <causal> <window> <warmup> <iters>
//  Default: bh=128 sl=8192 causal=0 wnd=0 warmup=5 iters=20.
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "fa_bwd_common.cuh"

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) {                \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                    \
            cudaGetErrorString(e)); std::exit(1); }} while (0)

namespace fa_bwd_dk {
void launch_d_precompute(const __half *O, const __half *dO, float *D,
                          int bh, int sl, int hd, cudaStream_t stream);
void launch_dk(const uint8_t *Q, const uint8_t *K, const uint8_t *V,
               const __half *dO_g, const float *L, const float *D,
               float *dK,
               int bh, int sl, int hd, int causal, int window,
               float scale, cudaStream_t stream);
}
namespace fa_bwd_dq {
void launch_dq(const uint8_t *Q, const uint8_t *K, const uint8_t *V,
               const __half *dO_g, const float *L, const float *D,
               float *dQ,
               int bh, int sl, int hd, int causal, int window,
               float scale, cudaStream_t stream);
}
namespace fa_bwd_dv_mma_p1 {
void launch(const uint8_t *Q, const uint8_t *K, const __half *dO_g,
            const float *L, float *dV,
            int bh, int sl, int hd, int causal, int window,
            float scale, cudaStream_t stream);
}

// V2 fingerprint gate: forward-declare each kernel so cudaFuncGetAttributes works.
namespace fa_bwd_dq { __global__ void kernel_dq(
    const uint8_t *, const uint8_t *, const uint8_t *,
    const __half *, const float *, const float *, float *,
    int, int, int, int, int, float); }
namespace fa_bwd_dk { __global__ void kernel_dk(
    const uint8_t *, const uint8_t *, const uint8_t *,
    const __half *, const float *, const float *, float *,
    int, int, int, int, int, float); }
namespace fa_bwd_dv_mma_p1 { __global__ void kernel_dv_mma_p1(
    const uint8_t *, const uint8_t *, const __half *,
    const float *, float *,
    int, int, int, int, int, float); }

static void e2e_fingerprint_one(const char *label, const void *fptr) {
    cudaFuncAttributes fa;
    cudaError_t e = cudaFuncGetAttributes(&fa, fptr);
    if (e != cudaSuccess) {
        fprintf(stderr, "fingerprint %s: %s\n", label, cudaGetErrorString(e));
        return;
    }
    printf("bench_e2e: FINGERPRINT %s: numRegs=%d, sharedSizeBytes=%zu, "
           "localSizeBytes=%zu, maxThreadsPerBlock=%d\n",
           label, fa.numRegs, fa.sharedSizeBytes, fa.localSizeBytes,
           fa.maxThreadsPerBlock);
}
static void bench_e2e_fingerprint() {
    e2e_fingerprint_one("kernel_dq", (const void*)fa_bwd_dq::kernel_dq);
    e2e_fingerprint_one("kernel_dk", (const void*)fa_bwd_dk::kernel_dk);
    e2e_fingerprint_one("kernel_dv_mma_p1", (const void*)fa_bwd_dv_mma_p1::kernel_dv_mma_p1);
}

int main(int argc, char **argv)
{
    int bh      = (argc >= 2) ? std::atoi(argv[1]) : 128;
    int sl      = (argc >= 3) ? std::atoi(argv[2]) : 8192;
    int causal  = (argc >= 4) ? std::atoi(argv[3]) : 0;
    int window  = (argc >= 5) ? std::atoi(argv[4]) : 0;
    int warmup  = (argc >= 6) ? std::atoi(argv[5]) : 5;
    int iters   = (argc >= 7) ? std::atoi(argv[6]) : 20;
    int hd      = 128;

    const size_t sz  = (size_t)bh * sl * hd;
    const size_t lsz = (size_t)bh * sl;

    printf("bench_e2e: bh=%d sl=%d hd=%d causal=%d window=%d warmup=%d iters=%d\n",
           bh, sl, hd, causal, window, warmup, iters);
    bench_e2e_fingerprint();    // V2 fingerprint gate — all three kernels self-report

    // Random inputs (seed=42 for comparability across runs)
    std::vector<uint8_t> Q8(sz), K8(sz), V8(sz);
    std::vector<__half>  O16(sz), dO16(sz);
    std::vector<float>   L32(lsz);
    {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.6f);
        for (size_t i = 0; i < sz; ++i) {
            Q8[i] = float_to_e4m3_host(dist(rng));
            K8[i] = float_to_e4m3_host(dist(rng));
            V8[i] = float_to_e4m3_host(dist(rng));
            O16[i] = __float2half_rn(dist(rng));
            dO16[i] = __float2half_rn(dist(rng));
        }
        for (size_t i = 0; i < lsz; ++i) L32[i] = dist(rng);
    }

    uint8_t *dQ_g, *dK_g, *dV_g;
    __half  *dO_O_g, *dO_dO_g;
    float   *dL, *dD, *ddV, *ddK, *ddQ;
    CK(cudaMalloc(&dQ_g,  sz));
    CK(cudaMalloc(&dK_g,  sz));
    CK(cudaMalloc(&dV_g,  sz));
    CK(cudaMalloc(&dO_O_g,  sz * sizeof(__half)));
    CK(cudaMalloc(&dO_dO_g, sz * sizeof(__half)));
    CK(cudaMalloc(&dL,  lsz * sizeof(float)));
    CK(cudaMalloc(&dD,  lsz * sizeof(float)));
    CK(cudaMalloc(&ddV, sz * sizeof(float)));
    CK(cudaMalloc(&ddK, sz * sizeof(float)));
    CK(cudaMalloc(&ddQ, sz * sizeof(float)));
    CK(cudaMemcpy(dQ_g, Q8.data(),   sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK_g, K8.data(),   sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV_g, V8.data(),   sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dO_O_g,  O16.data(),  sz * sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dO_dO_g, dO16.data(), sz * sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dL, L32.data(), lsz * sizeof(float), cudaMemcpyHostToDevice));

    float scale = 1.0f / std::sqrt((float)hd);

    cudaEvent_t e_start, e_after_d, e_after_dv, e_after_dk, e_after_dq;
    CK(cudaEventCreate(&e_start));
    CK(cudaEventCreate(&e_after_d));
    CK(cudaEventCreate(&e_after_dv));
    CK(cudaEventCreate(&e_after_dk));
    CK(cudaEventCreate(&e_after_dq));

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        CK(cudaMemsetAsync(ddV, 0, sz * sizeof(float)));
        CK(cudaMemsetAsync(ddK, 0, sz * sizeof(float)));
        CK(cudaMemsetAsync(ddQ, 0, sz * sizeof(float)));
        fa_bwd_dk::launch_d_precompute(dO_O_g, dO_dO_g, dD, bh, sl, hd, 0);
        fa_bwd_dv_mma_p1::launch(dQ_g, dK_g, dO_dO_g, dL, ddV,
                                  bh, sl, hd, causal, window, scale, 0);
        fa_bwd_dk::launch_dk(dQ_g, dK_g, dV_g, dO_dO_g, dL, dD, ddK,
                              bh, sl, hd, causal, window, scale, 0);
        fa_bwd_dq::launch_dq(dQ_g, dK_g, dV_g, dO_dO_g, dL, dD, ddQ,
                              bh, sl, hd, causal, window, scale, 0);
    }
    CK(cudaDeviceSynchronize());

    // Measure: accumulate per-kernel times across iters
    double sum_d = 0.0, sum_dv = 0.0, sum_dk = 0.0, sum_dq = 0.0, sum_total = 0.0;
    for (int i = 0; i < iters; ++i) {
        CK(cudaMemsetAsync(ddV, 0, sz * sizeof(float)));
        CK(cudaMemsetAsync(ddK, 0, sz * sizeof(float)));
        CK(cudaMemsetAsync(ddQ, 0, sz * sizeof(float)));

        CK(cudaEventRecord(e_start));
        fa_bwd_dk::launch_d_precompute(dO_O_g, dO_dO_g, dD, bh, sl, hd, 0);
        CK(cudaEventRecord(e_after_d));
        fa_bwd_dv_mma_p1::launch(dQ_g, dK_g, dO_dO_g, dL, ddV,
                                  bh, sl, hd, causal, window, scale, 0);
        CK(cudaEventRecord(e_after_dv));
        fa_bwd_dk::launch_dk(dQ_g, dK_g, dV_g, dO_dO_g, dL, dD, ddK,
                              bh, sl, hd, causal, window, scale, 0);
        CK(cudaEventRecord(e_after_dk));
        fa_bwd_dq::launch_dq(dQ_g, dK_g, dV_g, dO_dO_g, dL, dD, ddQ,
                              bh, sl, hd, causal, window, scale, 0);
        CK(cudaEventRecord(e_after_dq));
        CK(cudaEventSynchronize(e_after_dq));

        float ms_d, ms_dv, ms_dk, ms_dq, ms_total;
        CK(cudaEventElapsedTime(&ms_d,  e_start,    e_after_d));
        CK(cudaEventElapsedTime(&ms_dv, e_after_d,  e_after_dv));
        CK(cudaEventElapsedTime(&ms_dk, e_after_dv, e_after_dk));
        CK(cudaEventElapsedTime(&ms_dq, e_after_dk, e_after_dq));
        CK(cudaEventElapsedTime(&ms_total, e_start, e_after_dq));
        sum_d  += ms_d;
        sum_dv += ms_dv;
        sum_dk += ms_dk;
        sum_dq += ms_dq;
        sum_total += ms_total;
    }

    double avg_d  = sum_d  / iters;
    double avg_dv = sum_dv / iters;
    double avg_dk = sum_dk / iters;
    double avg_dq = sum_dq / iters;
    double avg_total = sum_total / iters;
    double sum_kernels = avg_d + avg_dv + avg_dk + avg_dq;
    double overhead = avg_total - sum_kernels;

    // FLOPS arithmetic (Tri Dao Variant 3)
    double cf = causal ? 0.5 : 1.0;
    double base = (double)bh * sl * sl * hd * cf;
    double flops_dv = 4.0 * base;   // Q·K^T recompute + P^T·dO
    double flops_dk = 6.0 * base;   // Q·K^T + dO·V^T + dS^T·Q
    double flops_dq = 6.0 * base;   // Q·K^T + dO·V^T + dS·K
    double flops_total = flops_dv + flops_dk + flops_dq;   // = 16·base

    double tflops_dv_iso = flops_dv / (avg_dv * 1e-3) / 1e12;
    double tflops_dk_iso = flops_dk / (avg_dk * 1e-3) / 1e12;
    double tflops_dq_iso = flops_dq / (avg_dq * 1e-3) / 1e12;
    double tflops_e2e = flops_total / (avg_total * 1e-3) / 1e12;

    printf("\n=== Per-kernel breakdown (chain) ===\n");
    printf("  D-precompute  avg_ms=%.4f  (%5.2f%% wall, %s overhead)\n",
           avg_d, 100.0 * avg_d / avg_total, avg_d < 0.5 ? "≈" : ">");
    printf("  dV            avg_ms=%.4f  (%5.2f%% wall, %.2f T in-chain)\n",
           avg_dv, 100.0 * avg_dv / avg_total, tflops_dv_iso);
    printf("  dK            avg_ms=%.4f  (%5.2f%% wall, %.2f T in-chain)\n",
           avg_dk, 100.0 * avg_dk / avg_total, tflops_dk_iso);
    printf("  dQ            avg_ms=%.4f  (%5.2f%% wall, %.2f T in-chain)\n",
           avg_dq, 100.0 * avg_dq / avg_total, tflops_dq_iso);
    printf("  sum_kernels   %.4f  overhead=%.4f (%5.2f%%)\n",
           sum_kernels, overhead, 100.0 * overhead / avg_total);
    printf("\n=== END-TO-END ===\n");
    printf("  avg_total=%.4f ms\n", avg_total);
    printf("  weighted TFLOPS = %.2f T  (causal_factor=%.2f)\n", tflops_e2e, cf);

    printf("\n=== Isolated reference (for L2-contention check) ===\n");
    printf("  dV isolated: ~160.8 T (memory)\n");
    printf("  dK isolated: ~196.1 T (B3.2+f+4)\n");
    printf("  dQ isolated: ~171.9 T (B4.2+b)\n");
    printf("  vs in-chain: dV %.2f T, dK %.2f T, dQ %.2f T\n",
           tflops_dv_iso, tflops_dk_iso, tflops_dq_iso);

    CK(cudaFree(dQ_g)); CK(cudaFree(dK_g)); CK(cudaFree(dV_g));
    CK(cudaFree(dO_O_g)); CK(cudaFree(dO_dO_g));
    CK(cudaFree(dL)); CK(cudaFree(dD));
    CK(cudaFree(ddV)); CK(cudaFree(ddK)); CK(cudaFree(ddQ));
    return 0;
}
