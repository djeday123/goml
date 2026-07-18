// =====================================================================
//  bench_b21_dv_mma.cu — minimal GPU-only wrapper for NCu baseline profile.
//
//  CLI: ./bench_b21_dv_mma <bh> <sl> <causal> <window> <warmup> <iters>
//  Default: bh=1, sl=2048, causal=0, window=0, warmup=5, iters=20.
//
//  NO host CPU forward / no FP64 ref / no validation — only GPU kernel
//  launches for clean NCu instrumentation. Inputs filled with deterministic
//  pseudo-random FP32 → quantized to FP8/FP16 (same path as production
//  harness, so memory layout / cache patterns match).
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "fa_bwd_common.cuh"   // float_to_e4m3_host

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) {                \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                    \
            cudaGetErrorString(e)); exit(1); }} while (0)

namespace fa_bwd_dv_mma_p1_alpha {
void launch(const uint8_t *Q, const uint8_t *K, const __half *dO_g,
            const float *L, float *dV,
            int bh, int sl, int hd, int causal, int window,
            float scale, cudaStream_t stream);
}

int main(int argc, char **argv)
{
    int bh      = (argc >= 2) ? std::atoi(argv[1]) : 1;
    int sl      = (argc >= 3) ? std::atoi(argv[2]) : 2048;
    int causal  = (argc >= 4) ? std::atoi(argv[3]) : 0;
    int window  = (argc >= 5) ? std::atoi(argv[4]) : 0;
    int warmup  = (argc >= 6) ? std::atoi(argv[5]) : 5;
    int iters   = (argc >= 7) ? std::atoi(argv[6]) : 20;
    int hd      = 128;

    const size_t sz  = (size_t)bh * sl * hd;
    const size_t lsz = (size_t)bh * sl;

    printf("bench: bh=%d sl=%d hd=%d causal=%d window=%d warmup=%d iters=%d\n",
           bh, sl, hd, causal, window, warmup, iters);

    // Generate FP32, quantize to FP8 (Q,K) and FP16 (dO) on host.
    std::vector<uint8_t> Q8(sz), K8(sz);
    std::vector<__half>  dO16(sz);
    std::vector<float>   L32(lsz);
    {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.6f);
        for (size_t i = 0; i < sz; ++i) {
            Q8 [i] = float_to_e4m3_host(dist(rng));
            K8 [i] = float_to_e4m3_host(dist(rng));
            dO16[i] = __float2half_rn(dist(rng));
        }
        for (size_t i = 0; i < lsz; ++i) L32[i] = dist(rng);
    }

    uint8_t *dQ = nullptr, *dK = nullptr;
    __half  *ddO = nullptr;
    float   *dL = nullptr, *ddV = nullptr;
    CK(cudaMalloc(&dQ,  sz  * sizeof(uint8_t)));
    CK(cudaMalloc(&dK,  sz  * sizeof(uint8_t)));
    CK(cudaMalloc(&ddO, sz  * sizeof(__half)));
    CK(cudaMalloc(&dL,  lsz * sizeof(float)));
    CK(cudaMalloc(&ddV, sz  * sizeof(float)));
    CK(cudaMemcpy(dQ,  Q8.data(),   sz  * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK,  K8.data(),   sz  * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ddO, dO16.data(), sz  * sizeof(__half),  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dL,  L32.data(),  lsz * sizeof(float),   cudaMemcpyHostToDevice));

    float scale = 1.0f / std::sqrt((float)hd);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        CK(cudaMemset(ddV, 0, sz * sizeof(float)));
        fa_bwd_dv_mma_p1_alpha::launch(dQ, dK, ddO, dL, ddV,
                              bh, sl, hd, causal, window, scale, 0);
    }
    CK(cudaDeviceSynchronize());

    // Measurement loop — these are the launches NCu profiles.
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) {
        CK(cudaMemset(ddV, 0, sz * sizeof(float)));
        fa_bwd_dv_mma_p1_alpha::launch(dQ, dK, ddO, dL, ddV,
                              bh, sl, hd, causal, window, scale, 0);
    }
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, t0, t1));

    double avg_ms = ms / iters;
    // Rough TFLOPs estimate for dV-only:
    //   forward Q·K^T = 2*bh*sl*sl*hd  (with causal/window factor)
    //   exp negligible
    //   P^T·dO        = 2*bh*sl*sl*hd
    // Total ≈ 4*bh*sl*sl*hd. Causal halves Q·K^T flops, dV pass full.
    double causal_factor = causal ? 0.5 : 1.0;
    double flops = 2.0 * bh * (double)sl * sl * hd * causal_factor
                 + 2.0 * bh * (double)sl * sl * hd;
    double tflops = flops / (avg_ms * 1e-3) / 1e12;
    printf("bench: avg_ms=%.3f  tflops=%.2f  (causal_factor=%.2f)\n",
           avg_ms, tflops, causal_factor);

    CK(cudaFree(dQ));   CK(cudaFree(dK));
    CK(cudaFree(ddO));  CK(cudaFree(dL));
    CK(cudaFree(ddV));
    return 0;
}
