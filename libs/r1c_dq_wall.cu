// R1c isolated wall bench: kernel_dq_new only, canonical form.
//   Pre-materializes dS_nat once via ds_gen, then benchmarks dq_new alone.

#include <cstdio>
#include <cstdlib>
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
namespace fa_bwd_ds_gen {
void launch_ds_gen(const uint8_t *Q, const uint8_t *K, const uint8_t *V,
                    const __half *dO_g, const float *L, const float *D,
                    uint8_t *dS_out, uint8_t *dS_T_out,
                    int bh, int sl, int hd, int causal, int window,
                    float scale, cudaStream_t stream);
}
namespace fa_bwd_dq_new {
void launch_dq_new(const uint8_t *K, const uint8_t *dS_nat, float *dQ,
                    int bh, int sl, int hd, int causal, int window,
                    float scale, cudaStream_t stream);
__global__ void kernel_dq_new(const uint8_t *, const uint8_t *, float *,
                              int, int, int, int, int, float);
}

int main() {
    const int bh = 128, sl = 8192, hd = 128, causal = 0, window = 0;
    const int warmup = 5, iters = 20;
    int stride_ds = (sl + 15) & ~15;

    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, fa_bwd_dq_new::kernel_dq_new) == cudaSuccess) {
        printf("bench_dq_new: FINGERPRINT kernel_dq_new: numRegs=%d, "
               "sharedSizeBytes=%zu, localSizeBytes=%zu, maxThreadsPerBlock=%d\n",
               fa.numRegs, fa.sharedSizeBytes, fa.localSizeBytes,
               fa.maxThreadsPerBlock);
    }
    printf("bench_dq_new: bh=%d sl=%d hd=%d causal=%d warmup=%d iters=%d\n",
           bh, sl, hd, causal, warmup, iters);

    size_t sz = (size_t)bh * sl * hd;
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

    uint8_t *dQ8, *dK8, *dV8;
    __half  *dO_g_g, *dO_O_g;
    float   *dL, *dD, *ddQ;
    uint8_t *dS_nat, *dS_T;
    CKR(cudaMalloc(&dQ8, sz)); CKR(cudaMalloc(&dK8, sz)); CKR(cudaMalloc(&dV8, sz));
    CKR(cudaMalloc(&dO_O_g, sz*sizeof(__half))); CKR(cudaMalloc(&dO_g_g, sz*sizeof(__half)));
    CKR(cudaMalloc(&dL, lsz*sizeof(float))); CKR(cudaMalloc(&dD, lsz*sizeof(float)));
    CKR(cudaMalloc(&ddQ, sz*sizeof(float)));
    CKR(cudaMalloc(&dS_nat, dsz)); CKR(cudaMalloc(&dS_T, dsz));
    CKR(cudaMemcpy(dQ8, Q8.data(), sz, cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(dK8, K8.data(), sz, cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(dV8, V8.data(), sz, cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(dO_O_g, O16.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(dO_g_g, dO16.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(dL, L32.data(), lsz*sizeof(float), cudaMemcpyHostToDevice));

    fa_bwd_dk::launch_d_precompute(dO_O_g, dO_g_g, dD, bh, sl, hd, 0);
    CKR(cudaDeviceSynchronize());

    float scale = 1.0f / sqrtf((float)hd);

    // Pre-materialize dS_nat once (ds_gen not part of dq_new wall)
    fa_bwd_ds_gen::launch_ds_gen(dQ8, dK8, dV8, dO_g_g, dL, dD, dS_nat, dS_T,
                                  bh, sl, hd, causal, window, scale, 0);
    CKR(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        fa_bwd_dq_new::launch_dq_new(dK8, dS_nat, ddQ, bh, sl, hd, causal, window, scale, 0);
    }
    CKR(cudaDeviceSynchronize());

    // Timed
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        fa_bwd_dq_new::launch_dq_new(dK8, dS_nat, ddQ, bh, sl, hd, causal, window, scale, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_total = 0;
    cudaEventElapsedTime(&ms_total, start, stop);
    float avg_ms = ms_total / iters;

    // "TFLOPS" via 1 mma (dS·K): 2 * bh * sl * sl * hd fma
    double fops = 2.0 * (double)bh * sl * sl * hd;
    double tflops = fops / (avg_ms * 1e-3) / 1e12;
    printf("bench_dq_new: avg_ms=%.3f  tflops_1mma=%.2f\n", avg_ms, tflops);

    cudaFree(dQ8); cudaFree(dK8); cudaFree(dV8);
    cudaFree(dO_O_g); cudaFree(dO_g_g);
    cudaFree(dL); cudaFree(dD); cudaFree(ddQ);
    cudaFree(dS_nat); cudaFree(dS_T);
    return 0;
}
