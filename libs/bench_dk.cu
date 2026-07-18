// =====================================================================
//  bench_dk.cu — minimal GPU-only wrapper for dK NCu baseline.
//
//  CLI: ./bench_dk <bh> <sl> <causal> <window> <warmup> <iters> [d_only]
//  Default: bh=128 sl=8192 causal=0 wnd=0 warmup=5 iters=20.
//  d_only=1: measure ONLY D-precompute kernel wall (separate ~0.5-1.5 ms est).
//
//  D-precompute runs ONCE before measurement loop (D vector reused).
//  NCu profile target: kernel_dk (skip kernel_d_precompute via launch-skip
//  + kernel-name filter).
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

#define DK_CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) {              \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                     \
            cudaGetErrorString(e)); std::exit(1); }} while (0)

namespace fa_bwd_dk {
void launch_d_precompute(const __half *O, const __half *dO, float *D,
                          int bh, int sl, int hd, cudaStream_t stream);
void launch_dk(const uint8_t *Q, const uint8_t *K, const uint8_t *V,
               const __half *dO_g, const float *L, const float *D,
               float *dK,
               int bh, int sl, int hd, int causal, int window,
               float scale, cudaStream_t stream);
__global__ void kernel_dk(
    const uint8_t *, const uint8_t *, const uint8_t *,
    const __half *, const float *, const float *, float *,
    int, int, int, int, int, float);
}
static void bench_dk_fingerprint() {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, fa_bwd_dk::kernel_dk) != cudaSuccess) return;
    printf("bench_dk: FINGERPRINT kernel_dk: numRegs=%d, sharedSizeBytes=%zu, "
           "localSizeBytes=%zu, maxThreadsPerBlock=%d\n",
           fa.numRegs, fa.sharedSizeBytes, fa.localSizeBytes, fa.maxThreadsPerBlock);
}

int main(int argc, char **argv)
{
    int bh      = (argc >= 2) ? std::atoi(argv[1]) : 128;
    int sl      = (argc >= 3) ? std::atoi(argv[2]) : 8192;
    int causal  = (argc >= 4) ? std::atoi(argv[3]) : 0;
    int window  = (argc >= 5) ? std::atoi(argv[4]) : 0;
    int warmup  = (argc >= 6) ? std::atoi(argv[5]) : 5;
    int iters   = (argc >= 7) ? std::atoi(argv[6]) : 20;
    int d_only  = (argc >= 8) ? std::atoi(argv[7]) : 0;
    int hd      = 128;

    const size_t sz  = (size_t)bh * sl * hd;
    const size_t lsz = (size_t)bh * sl;

    printf("bench_dk: bh=%d sl=%d hd=%d causal=%d window=%d warmup=%d iters=%d d_only=%d\n",
           bh, sl, hd, causal, window, warmup, iters, d_only);
    bench_dk_fingerprint();    // V2 fingerprint gate

    // Random inputs
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

    uint8_t *dQ, *dKk, *dV;
    __half  *dO_d, *dO_g;
    float   *dL, *dD, *ddK;
    DK_CK(cudaMalloc(&dQ,  sz * sizeof(uint8_t)));
    DK_CK(cudaMalloc(&dKk, sz * sizeof(uint8_t)));
    DK_CK(cudaMalloc(&dV,  sz * sizeof(uint8_t)));
    DK_CK(cudaMalloc(&dO_d, sz * sizeof(__half)));
    DK_CK(cudaMalloc(&dO_g, sz * sizeof(__half)));
    DK_CK(cudaMalloc(&dL,  lsz * sizeof(float)));
    DK_CK(cudaMalloc(&dD,  lsz * sizeof(float)));
    DK_CK(cudaMalloc(&ddK, sz * sizeof(float)));
    DK_CK(cudaMemcpy(dQ,  Q8.data(),   sz  * sizeof(uint8_t), cudaMemcpyHostToDevice));
    DK_CK(cudaMemcpy(dKk, K8.data(),   sz  * sizeof(uint8_t), cudaMemcpyHostToDevice));
    DK_CK(cudaMemcpy(dV,  V8.data(),   sz  * sizeof(uint8_t), cudaMemcpyHostToDevice));
    DK_CK(cudaMemcpy(dO_d, O16.data(), sz * sizeof(__half),  cudaMemcpyHostToDevice));
    DK_CK(cudaMemcpy(dO_g, dO16.data(), sz * sizeof(__half), cudaMemcpyHostToDevice));
    DK_CK(cudaMemcpy(dL,  L32.data(),  lsz * sizeof(float),  cudaMemcpyHostToDevice));

    float scale = 1.0f / std::sqrt((float)hd);

    if (d_only) {
        // ============================
        // D-precompute kernel wall ONLY
        // ============================
        for (int i = 0; i < warmup; ++i) {
            fa_bwd_dk::launch_d_precompute(dO_d, dO_g, dD, bh, sl, hd, 0);
        }
        DK_CK(cudaDeviceSynchronize());

        cudaEvent_t t0, t1;
        DK_CK(cudaEventCreate(&t0));
        DK_CK(cudaEventCreate(&t1));
        DK_CK(cudaEventRecord(t0));
        for (int i = 0; i < iters; ++i) {
            fa_bwd_dk::launch_d_precompute(dO_d, dO_g, dD, bh, sl, hd, 0);
        }
        DK_CK(cudaEventRecord(t1));
        DK_CK(cudaEventSynchronize(t1));
        float ms = 0.0f;
        DK_CK(cudaEventElapsedTime(&ms, t0, t1));
        double avg_ms = ms / iters;

        // D-kernel transfer rate
        double total_bytes = 2.0 * sz * sizeof(__half) + lsz * sizeof(float);
        double gb_per_s = (total_bytes / 1e9) / (avg_ms * 1e-3);
        printf("bench_dk D-only: avg_ms=%.4f  transfer=%.2f GB/s  (total=%.1f MB)\n",
               avg_ms, gb_per_s, total_bytes / 1e6);
    } else {
        // ============================
        // dK kernel wall (D precomputed once)
        // ============================
        fa_bwd_dk::launch_d_precompute(dO_d, dO_g, dD, bh, sl, hd, 0);
        DK_CK(cudaDeviceSynchronize());

        // Warmup dK
        for (int i = 0; i < warmup; ++i) {
            DK_CK(cudaMemset(ddK, 0, sz * sizeof(float)));
            fa_bwd_dk::launch_dk(dQ, dKk, dV, dO_g, dL, dD, ddK,
                                 bh, sl, hd, causal, window, scale, 0);
        }
        DK_CK(cudaDeviceSynchronize());

        cudaEvent_t t0, t1;
        DK_CK(cudaEventCreate(&t0));
        DK_CK(cudaEventCreate(&t1));
        DK_CK(cudaEventRecord(t0));
        for (int i = 0; i < iters; ++i) {
            DK_CK(cudaMemset(ddK, 0, sz * sizeof(float)));
            fa_bwd_dk::launch_dk(dQ, dKk, dV, dO_g, dL, dD, ddK,
                                 bh, sl, hd, causal, window, scale, 0);
        }
        DK_CK(cudaEventRecord(t1));
        DK_CK(cudaEventSynchronize(t1));
        float ms = 0.0f;
        DK_CK(cudaEventElapsedTime(&ms, t0, t1));
        double avg_ms = ms / iters;

        // TFLOPS estimate: dK has 3 MMA chains per qt
        //   Q·K^T  : 2 * bh * sl² * hd * causal_factor
        //   dO·V^T : 2 * bh * sl² * hd * causal_factor
        //   dS^T·Q : 2 * bh * sl² * hd * causal_factor
        // Total ≈ 6 * bh * sl² * hd * causal_factor
        double cf = causal ? 0.5 : 1.0;
        double flops = 6.0 * bh * (double)sl * sl * hd * cf;
        double tflops = flops / (avg_ms * 1e-3) / 1e12;
        printf("bench_dk: avg_ms=%.3f  tflops=%.2f  (causal_factor=%.2f)\n",
               avg_ms, tflops, cf);
    }

    DK_CK(cudaFree(dQ));  DK_CK(cudaFree(dKk)); DK_CK(cudaFree(dV));
    DK_CK(cudaFree(dO_d)); DK_CK(cudaFree(dO_g));
    DK_CK(cudaFree(dL));  DK_CK(cudaFree(dD));  DK_CK(cudaFree(ddK));
    return 0;
}
