// bench_dv.cu — minimal GPU-only wrapper for dV NCu baseline / 30-run seal.
// CLI: ./bench_dv <bh> <sl> <causal> <window> <warmup> <iters>
// Default: bh=128 sl=8192 causal=0 wnd=0 warmup=5 iters=20.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "fa_bwd_common.cuh"

#define DV_CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) {                \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                       \
            cudaGetErrorString(e)); std::exit(1); }} while (0)

namespace fa_bwd_dv_mma_p1 {
void launch(const uint8_t *Q, const uint8_t *K, const __half *dO_g,
            const float *L, float *dV,
            int bh, int sl, int hd, int causal, int window,
            float scale, cudaStream_t stream);
__global__ void kernel_dv_mma_p1(
    const uint8_t *, const uint8_t *, const __half *,
    const float *, float *,
    int, int, int, int, int, float);
}
static void bench_dv_fingerprint() {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, fa_bwd_dv_mma_p1::kernel_dv_mma_p1) != cudaSuccess) return;
    printf("bench_dv: FINGERPRINT kernel_dv_mma_p1: numRegs=%d, sharedSizeBytes=%zu, "
           "localSizeBytes=%zu, maxThreadsPerBlock=%d\n",
           fa.numRegs, fa.sharedSizeBytes, fa.localSizeBytes, fa.maxThreadsPerBlock);
}

int main(int argc, char **argv) {
    int bh      = (argc >= 2) ? std::atoi(argv[1]) : 128;
    int sl      = (argc >= 3) ? std::atoi(argv[2]) : 8192;
    int causal  = (argc >= 4) ? std::atoi(argv[3]) : 0;
    int window  = (argc >= 5) ? std::atoi(argv[4]) : 0;
    int warmup  = (argc >= 6) ? std::atoi(argv[5]) : 5;
    int iters   = (argc >= 7) ? std::atoi(argv[6]) : 20;
    int hd      = 128;

    const size_t sz  = (size_t)bh * sl * hd;
    const size_t lsz = (size_t)bh * sl;

    printf("bench_dv: bh=%d sl=%d hd=%d causal=%d window=%d warmup=%d iters=%d\n",
           bh, sl, hd, causal, window, warmup, iters);
    bench_dv_fingerprint();    // V2 fingerprint gate

    std::vector<uint8_t> Q8(sz), K8(sz);
    std::vector<__half>  dO16(sz);
    std::vector<float>   L32(lsz);
    {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.6f);
        for (size_t i = 0; i < sz; ++i) {
            Q8[i] = float_to_e4m3_host(dist(rng));
            K8[i] = float_to_e4m3_host(dist(rng));
            dO16[i] = __float2half_rn(dist(rng));
        }
        for (size_t i = 0; i < lsz; ++i) L32[i] = dist(rng);
    }

    uint8_t *dQ, *dK;
    __half  *ddO;
    float   *dL, *ddV;
    DV_CK(cudaMalloc(&dQ,  sz * sizeof(uint8_t)));
    DV_CK(cudaMalloc(&dK,  sz * sizeof(uint8_t)));
    DV_CK(cudaMalloc(&ddO, sz * sizeof(__half)));
    DV_CK(cudaMalloc(&dL,  lsz * sizeof(float)));
    DV_CK(cudaMalloc(&ddV, sz * sizeof(float)));
    DV_CK(cudaMemcpy(dQ,  Q8.data(),   sz  * sizeof(uint8_t), cudaMemcpyHostToDevice));
    DV_CK(cudaMemcpy(dK,  K8.data(),   sz  * sizeof(uint8_t), cudaMemcpyHostToDevice));
    DV_CK(cudaMemcpy(ddO, dO16.data(), sz  * sizeof(__half),  cudaMemcpyHostToDevice));
    DV_CK(cudaMemcpy(dL,  L32.data(),  lsz * sizeof(float),   cudaMemcpyHostToDevice));

    float scale = 1.0f / std::sqrt((float)hd);

    for (int i = 0; i < warmup; ++i) {
        DV_CK(cudaMemset(ddV, 0, sz * sizeof(float)));
        fa_bwd_dv_mma_p1::launch(dQ, dK, ddO, dL, ddV,
                                 bh, sl, hd, causal, window, scale, 0);
    }
    DV_CK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    DV_CK(cudaEventCreate(&t0));
    DV_CK(cudaEventCreate(&t1));
    DV_CK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) {
        DV_CK(cudaMemset(ddV, 0, sz * sizeof(float)));
        fa_bwd_dv_mma_p1::launch(dQ, dK, ddO, dL, ddV,
                                 bh, sl, hd, causal, window, scale, 0);
    }
    DV_CK(cudaEventRecord(t1));
    DV_CK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    DV_CK(cudaEventElapsedTime(&ms, t0, t1));
    double avg_ms = ms / iters;

    // dV = 2 MMA chains × 2*bh*sl²*hd = 4*bh*sl²*hd
    double cf = causal ? 0.5 : 1.0;
    double flops = 4.0 * bh * (double)sl * sl * hd * cf;
    double tflops = flops / (avg_ms * 1e-3) / 1e12;
    printf("bench_dv: avg_ms=%.3f  tflops=%.2f  (causal_factor=%.2f)\n",
           avg_ms, tflops, cf);

    DV_CK(cudaFree(dQ)); DV_CK(cudaFree(dK)); DV_CK(cudaFree(ddO));
    DV_CK(cudaFree(dL)); DV_CK(cudaFree(ddV));
    return 0;
}
