// R1b dk_new wall isolated: ds_gen produces dS_T; kernel_dk_new isolated bench.
//   Assumes single canonical form (bh=128 sl=8192 hd=128 nc). Prefills dS_T via one
//   ds_gen call in warmup; then measures dk_new iteration only.

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "fa_bwd_common.cuh"

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e)); std::exit(1); }} while (0)

namespace fa_bwd_dk {
void launch_d_precompute(const __half *O, const __half *dO, float *D,
                          int bh, int sl, int hd, cudaStream_t stream);
}
namespace fa_bwd_ds_gen {
void launch_ds_gen(const uint8_t *Q, const uint8_t *K, const uint8_t *V,
                    const __half *dO_g, const float *L, const float *D,
                    uint8_t *dS_out, uint8_t *dS_T_out,
                    int bh, int sl, int hd, int causal, int window,
                    float scale, cudaStream_t stream);
}
namespace fa_bwd_dk_new {
void launch_dk_new(const uint8_t *Q, const uint8_t *dS_T,
                   float *dK, int bh, int sl, int hd, int causal, int window,
                   float scale, cudaStream_t stream);
__global__ void kernel_dk_new(const uint8_t *, const uint8_t *, float *,
                              int, int, int, int, int, float);
}

int main(int argc, char **argv) {
    int bh = 128, sl = 8192, hd = 128, causal = 0;
    int warmup = 5, iters = 20;
    printf("bench_dk_new: bh=%d sl=%d hd=%d causal=%d warmup=%d iters=%d\n",
           bh, sl, hd, causal, warmup, iters);

    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, fa_bwd_dk_new::kernel_dk_new) == cudaSuccess) {
        printf("bench_dk_new: FINGERPRINT kernel_dk_new: numRegs=%d, sharedSizeBytes=%zu, "
               "localSizeBytes=%zu, maxThreadsPerBlock=%d\n",
               fa.numRegs, fa.sharedSizeBytes, fa.localSizeBytes, fa.maxThreadsPerBlock);
    }

    size_t sz  = (size_t)bh * sl * hd;
    size_t lsz = (size_t)bh * sl;
    size_t dsz = (size_t)bh * sl * sl;

    std::vector<uint8_t> Q8(sz), K8(sz), V8(sz);
    std::vector<__half>  O16(sz), dO16(sz);
    std::vector<float>   L32(lsz);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.6f);
    for (size_t i = 0; i < sz; ++i) {
        Q8[i] = float_to_e4m3_host(dist(rng));
        K8[i] = float_to_e4m3_host(dist(rng));
        V8[i] = float_to_e4m3_host(dist(rng));
        O16[i]  = __float2half_rn(dist(rng));
        dO16[i] = __float2half_rn(dist(rng));
    }
    for (size_t i = 0; i < lsz; ++i) L32[i] = dist(rng);

    uint8_t *dQ8, *dK8, *dV8;
    __half *dO_g_g, *dO_O_g;
    float *dL, *dD;
    uint8_t *dS_nat, *dS_T;
    float *ddK;
    CK(cudaMalloc(&dQ8, sz)); CK(cudaMalloc(&dK8, sz)); CK(cudaMalloc(&dV8, sz));
    CK(cudaMalloc(&dO_O_g, sz*sizeof(__half))); CK(cudaMalloc(&dO_g_g, sz*sizeof(__half)));
    CK(cudaMalloc(&dL, lsz*sizeof(float))); CK(cudaMalloc(&dD, lsz*sizeof(float)));
    CK(cudaMalloc(&dS_nat, dsz)); CK(cudaMalloc(&dS_T, dsz));
    CK(cudaMalloc(&ddK, sz*sizeof(float)));

    CK(cudaMemcpy(dQ8, Q8.data(), sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK8, K8.data(), sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV8, V8.data(), sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dO_O_g, O16.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dO_g_g, dO16.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dL, L32.data(), lsz*sizeof(float), cudaMemcpyHostToDevice));

    fa_bwd_dk::launch_d_precompute(dO_O_g, dO_g_g, dD, bh, sl, hd, 0);
    CK(cudaDeviceSynchronize());

    float scale = 1.0f / sqrtf((float)hd);

    // Pre-fill dS_T once (not counted in bench)
    fa_bwd_ds_gen::launch_ds_gen(dQ8, dK8, dV8, dO_g_g, dL, dD, dS_nat, dS_T,
                                  bh, sl, hd, causal, 0, scale, 0);
    CK(cudaDeviceSynchronize());

    // Warmup dk_new
    for (int i = 0; i < warmup; ++i) {
        fa_bwd_dk_new::launch_dk_new(dQ8, dS_nat, ddK, bh, sl, hd, causal, 0, scale, 0);
    }
    CK(cudaDeviceSynchronize());

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) {
        fa_bwd_dk_new::launch_dk_new(dQ8, dS_nat, ddK, bh, sl, hd, causal, 0, scale, 0);
    }
    cudaEventRecord(e);
    CK(cudaDeviceSynchronize());
    float ms; cudaEventElapsedTime(&ms, s, e);
    float avg_ms = ms / iters;
    // "TFLOPS" via 1 mma (dS^T·Q): 2 * bh * sl * sl * hd fma
    double fops = 2.0 * (double)bh * sl * sl * hd;
    double t = fops / (avg_ms * 1e-3) / 1e12;
    printf("bench_dk_new: avg_ms=%.3f  tflops_1mma=%.2f\n", avg_ms, t);

    cudaFree(dQ8); cudaFree(dK8); cudaFree(dV8);
    cudaFree(dO_O_g); cudaFree(dO_g_g);
    cudaFree(dL); cudaFree(dD);
    cudaFree(dS_nat); cudaFree(dS_T); cudaFree(ddK);
    return 0;
}
