// =====================================================================
//  bench_dq.cu — minimal GPU-only wrapper for dQ NCu baseline.
//  Mirror of bench_dk.cu, calls launch_dq instead of launch_dk.
//
//  CLI: ./bench_dq <bh> <sl> <causal> <window> <warmup> <iters>
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

#define DQ_CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) {              \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                     \
            cudaGetErrorString(e)); std::exit(1); }} while (0)

namespace fa_bwd_dk {
void launch_d_precompute(const __half *O, const __half *dO, float *D,
                          int bh, int sl, int hd, cudaStream_t stream);
}

namespace fa_bwd_dq {
void launch_dq(const uint8_t *Q, const uint8_t *K, const uint8_t *V,
               const __half *dO_g, const float *L, const float *D,
               float *dQ,
               int bh, int sl, int hd, int causal, int window,
               float scale, cudaStream_t stream);

// Forward-declare the kernel so cudaFuncGetAttributes has a handle.
__global__ void kernel_dq(
    const uint8_t *, const uint8_t *, const uint8_t *,
    const __half *, const float *, const float *, float *,
    int, int, int, int, int, float);
}

// V2 fingerprint gate: print numRegs / sharedSizeBytes / localSizeBytes / maxThreadsPerBlock
// so replays and benchmark logs can prove *which* build produced the numbers.
static void bench_dq_fingerprint() {
    cudaFuncAttributes fa;
    cudaError_t e = cudaFuncGetAttributes(&fa, fa_bwd_dq::kernel_dq);
    if (e != cudaSuccess) {
        fprintf(stderr, "fingerprint: cudaFuncGetAttributes: %s\n",
                cudaGetErrorString(e));
        return;
    }
    printf("bench_dq: FINGERPRINT kernel_dq: numRegs=%d, sharedSizeBytes=%zu, "
           "localSizeBytes=%zu, maxThreadsPerBlock=%d\n",
           fa.numRegs, fa.sharedSizeBytes, fa.localSizeBytes,
           fa.maxThreadsPerBlock);
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

    printf("bench_dq: bh=%d sl=%d hd=%d causal=%d window=%d warmup=%d iters=%d\n",
           bh, sl, hd, causal, window, warmup, iters);
    bench_dq_fingerprint();    // V2 fingerprint gate — self-report binary identity

    // Random inputs (same seed as bench_dk for direct comparability)
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
    float   *dL, *dD, *ddQ;
    DQ_CK(cudaMalloc(&dQ,  sz * sizeof(uint8_t)));
    DQ_CK(cudaMalloc(&dKk, sz * sizeof(uint8_t)));
    DQ_CK(cudaMalloc(&dV,  sz * sizeof(uint8_t)));
    DQ_CK(cudaMalloc(&dO_d, sz * sizeof(__half)));
    DQ_CK(cudaMalloc(&dO_g, sz * sizeof(__half)));
    DQ_CK(cudaMalloc(&dL,  lsz * sizeof(float)));
    DQ_CK(cudaMalloc(&dD,  lsz * sizeof(float)));
    DQ_CK(cudaMalloc(&ddQ, sz * sizeof(float)));
    DQ_CK(cudaMemcpy(dQ,  Q8.data(),   sz  * sizeof(uint8_t), cudaMemcpyHostToDevice));
    DQ_CK(cudaMemcpy(dKk, K8.data(),   sz  * sizeof(uint8_t), cudaMemcpyHostToDevice));
    DQ_CK(cudaMemcpy(dV,  V8.data(),   sz  * sizeof(uint8_t), cudaMemcpyHostToDevice));
    DQ_CK(cudaMemcpy(dO_d, O16.data(), sz * sizeof(__half),  cudaMemcpyHostToDevice));
    DQ_CK(cudaMemcpy(dO_g, dO16.data(), sz * sizeof(__half), cudaMemcpyHostToDevice));
    DQ_CK(cudaMemcpy(dL,  L32.data(),  lsz * sizeof(float),  cudaMemcpyHostToDevice));

    float scale = 1.0f / std::sqrt((float)hd);

    // D-precompute once
    fa_bwd_dk::launch_d_precompute(dO_d, dO_g, dD, bh, sl, hd, 0);
    DQ_CK(cudaDeviceSynchronize());

    // Warmup dQ
    for (int i = 0; i < warmup; ++i) {
        DQ_CK(cudaMemset(ddQ, 0, sz * sizeof(float)));
        fa_bwd_dq::launch_dq(dQ, dKk, dV, dO_g, dL, dD, ddQ,
                             bh, sl, hd, causal, window, scale, 0);
    }
    DQ_CK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    DQ_CK(cudaEventCreate(&t0));
    DQ_CK(cudaEventCreate(&t1));
    DQ_CK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) {
        DQ_CK(cudaMemset(ddQ, 0, sz * sizeof(float)));
        fa_bwd_dq::launch_dq(dQ, dKk, dV, dO_g, dL, dD, ddQ,
                             bh, sl, hd, causal, window, scale, 0);
    }
    DQ_CK(cudaEventRecord(t1));
    DQ_CK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    DQ_CK(cudaEventElapsedTime(&ms, t0, t1));
    double avg_ms = ms / iters;

    // TFLOPS: dQ has 3 MMA chains, same FLOP arithmetic as dK
    //   Q·K^T  : 2 * bh * sl² * hd * causal_factor
    //   dO·V^T : 2 * bh * sl² * hd * causal_factor
    //   dS·K   : 2 * bh * sl² * hd * causal_factor
    // Total ≈ 6 * bh * sl² * hd * causal_factor
    double cf = causal ? 0.5 : 1.0;
    double flops = 6.0 * bh * (double)sl * sl * hd * cf;
    double tflops = flops / (avg_ms * 1e-3) / 1e12;
    printf("bench_dq: avg_ms=%.3f  tflops=%.2f  (causal_factor=%.2f)\n",
           avg_ms, tflops, cf);

    DQ_CK(cudaFree(dQ));  DQ_CK(cudaFree(dKk)); DQ_CK(cudaFree(dV));
    DQ_CK(cudaFree(dO_d)); DQ_CK(cudaFree(dO_g));
    DQ_CK(cudaFree(dL));  DQ_CK(cudaFree(dD));  DQ_CK(cudaFree(ddQ));
    return 0;
}
