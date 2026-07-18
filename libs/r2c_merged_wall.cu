// R2C merged wall canonical (bh=128 sl=8192 hd=128 causal=0, warmup=5 iters=20).

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "fa_bwd_common.cuh"

#define CKR(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e)); std::exit(1); }} while (0)

namespace fa_bwd_dk {
void launch_d_precompute(const __half *O, const __half *dO, float *D,
                          int bh, int sl, int hd, cudaStream_t stream);
}
namespace fa_bwd_merged_v1 {
void launch_merged(const uint8_t *Q, const uint8_t *K, const uint8_t *V,
                    const __half *dO_g, const float *L, const float *D,
                    uint8_t *dS_nat, uint8_t *dS_T, float *dV,
                    int bh, int sl, int hd, int causal, int window,
                    float scale, cudaStream_t stream);
__global__ void kernel_merged_v1(const uint8_t *, const uint8_t *, const uint8_t *,
                                 const __half *, const float *, const float *,
                                 uint8_t *, uint8_t *, float *,
                                 int, int, int, int, int, float);
}

int main() {
    const int bh = 128, sl = 8192, hd = 128, causal = 0, window = 0;
    const int warmup = 5, iters = 20;
    int stride_ds = (sl + 15) & ~15;

    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, fa_bwd_merged_v1::kernel_merged_v1) == cudaSuccess) {
        printf("bench_merged: FINGERPRINT kernel_merged_v1: numRegs=%d, "
               "sharedSizeBytes=%zu, localSizeBytes=%zu, maxThreadsPerBlock=%d\n",
               fa.numRegs, fa.sharedSizeBytes, fa.localSizeBytes, fa.maxThreadsPerBlock);
    }
    printf("bench_merged: bh=%d sl=%d hd=%d causal=%d warmup=%d iters=%d\n",
           bh, sl, hd, causal, warmup, iters);

    size_t sz  = (size_t)bh * sl * hd;
    size_t lsz = (size_t)bh * sl;
    size_t dsz = (size_t)bh * sl * stride_ds;

    std::vector<uint8_t> Q8(sz), K8(sz), V8(sz);
    std::vector<__half>  O16(sz), dO16(sz);
    std::vector<float>   L32(lsz);
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

    uint8_t *dQ, *dK, *dV8;
    __half  *dOG, *dOO;
    float   *dL, *dD, *ddV;
    uint8_t *dS_nat, *dS_T;
    CKR(cudaMalloc(&dQ, sz)); CKR(cudaMalloc(&dK, sz)); CKR(cudaMalloc(&dV8, sz));
    CKR(cudaMalloc(&dOO, sz*sizeof(__half))); CKR(cudaMalloc(&dOG, sz*sizeof(__half)));
    CKR(cudaMalloc(&dL, lsz*sizeof(float))); CKR(cudaMalloc(&dD, lsz*sizeof(float)));
    CKR(cudaMalloc(&ddV, sz*sizeof(float)));
    CKR(cudaMalloc(&dS_nat, dsz)); CKR(cudaMalloc(&dS_T, dsz));
    CKR(cudaMemcpy(dQ, Q8.data(), sz, cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(dK, K8.data(), sz, cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(dV8, V8.data(), sz, cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(dOO, O16.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(dOG, dO16.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(dL, L32.data(), lsz*sizeof(float), cudaMemcpyHostToDevice));

    fa_bwd_dk::launch_d_precompute(dOO, dOG, dD, bh, sl, hd, 0);
    CKR(cudaDeviceSynchronize());

    float scale = 1.0f / sqrtf((float)hd);

    for (int i = 0; i < warmup; ++i) {
        fa_bwd_merged_v1::launch_merged(dQ, dK, dV8, dOG, dL, dD,
                                         dS_nat, dS_T, ddV,
                                         bh, sl, hd, causal, window, scale, 0);
    }
    CKR(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        fa_bwd_merged_v1::launch_merged(dQ, dK, dV8, dOG, dL, dD,
                                         dS_nat, dS_T, ddV,
                                         bh, sl, hd, causal, window, scale, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_total = 0;
    cudaEventElapsedTime(&ms_total, start, stop);
    float avg_ms = ms_total / iters;

    // Merged executes 4 MMAs total (Q·K^T + dO·V^T + P^T·dO + [none for dS as it's a compute path])
    // But per Vugar convention: dS_gen (2 MMA) + dV (2 MMA) merged S = shared → 3 MMA in merged.
    // Actually literal MMA count: Q·K^T (once, shared) + dO·V^T (for dP) + P^T·dO (for dV) = 3 MMA.
    // FLOPS = 6 * base (3 MMA * 2 base each).
    double base = (double)bh * sl * sl * hd;
    double fops = 6.0 * base;
    double tflops = fops / (avg_ms * 1e-3) / 1e12;
    printf("bench_merged: avg_ms=%.3f  tflops_3mma=%.2f\n", avg_ms, tflops);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV8);
    cudaFree(dOO); cudaFree(dOG);
    cudaFree(dL); cudaFree(dD); cudaFree(ddV);
    cudaFree(dS_nat); cudaFree(dS_T);
    return 0;
}
