// R1a wall standalone: kernel_ds_gen alone, bench-form bh=128 sl=8192 hd=128 (matches bench_dq).
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
__global__ void kernel_ds_gen(
    const uint8_t *, const uint8_t *, const uint8_t *,
    const __half *, const float *, const float *,
    uint8_t *, uint8_t *, int, int, int, int, int, float);
}

int main(int argc, char **argv) {
    int bh    = (argc >= 2) ? std::atoi(argv[1]) : 128;
    int sl    = (argc >= 3) ? std::atoi(argv[2]) : 8192;
    int causal = (argc >= 4) ? std::atoi(argv[3]) : 0;
    int warmup = (argc >= 5) ? std::atoi(argv[4]) : 5;
    int iters  = (argc >= 6) ? std::atoi(argv[5]) : 20;
    int hd = 128;
    printf("bench_ds_gen: bh=%d sl=%d hd=%d causal=%d warmup=%d iters=%d\n",
           bh, sl, hd, causal, warmup, iters);

    // Fingerprint gate
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, fa_bwd_ds_gen::kernel_ds_gen) == cudaSuccess) {
        printf("bench_ds_gen: FINGERPRINT kernel_ds_gen: numRegs=%d, "
               "sharedSizeBytes=%zu, localSizeBytes=%zu, maxThreadsPerBlock=%d\n",
               fa.numRegs, fa.sharedSizeBytes, fa.localSizeBytes,
               fa.maxThreadsPerBlock);
    }

    size_t sz  = (size_t)bh * sl * hd;
    size_t lsz = (size_t)bh * sl;
    int stride_ds = (sl + 15) & ~15;
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
        O16[i]  = __float2half_rn(dist(rng));
        dO16[i] = __float2half_rn(dist(rng));
    }
    for (size_t i = 0; i < lsz; ++i) L32[i] = dist(rng);

    uint8_t *dQ8, *dK8, *dV8;
    __half  *dO_g_g, *dO_O_g;
    float   *dL, *dD;
    uint8_t *dS_out, *dS_T_out;
    CK(cudaMalloc(&dQ8, sz)); CK(cudaMalloc(&dK8, sz)); CK(cudaMalloc(&dV8, sz));
    CK(cudaMalloc(&dO_O_g, sz*sizeof(__half))); CK(cudaMalloc(&dO_g_g, sz*sizeof(__half)));
    CK(cudaMalloc(&dL, lsz*sizeof(float))); CK(cudaMalloc(&dD, lsz*sizeof(float)));
    CK(cudaMalloc(&dS_out, dsz));
    CK(cudaMalloc(&dS_T_out, dsz));

    CK(cudaMemcpy(dQ8, Q8.data(), sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK8, K8.data(), sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV8, V8.data(), sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dO_O_g, O16.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dO_g_g, dO16.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dL, L32.data(), lsz*sizeof(float), cudaMemcpyHostToDevice));

    fa_bwd_dk::launch_d_precompute(dO_O_g, dO_g_g, dD, bh, sl, hd, 0);
    CK(cudaDeviceSynchronize());

    float scale = 1.0f / sqrtf((float)hd);

    for (int i = 0; i < warmup; ++i) {
        fa_bwd_ds_gen::launch_ds_gen(dQ8, dK8, dV8, dO_g_g, dL, dD, dS_out, dS_T_out,
                                      bh, sl, hd, causal, 0, scale, 0);
    }
    CK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        fa_bwd_ds_gen::launch_ds_gen(dQ8, dK8, dV8, dO_g_g, dL, dD, dS_out, dS_T_out,
                                      bh, sl, hd, causal, 0, scale, 0);
    }
    cudaEventRecord(stop);
    CK(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iters;
    // Rough executed matmul TFLOPS (Q·K^T + dO·V^T) = 2 * 2 * bh*sl*sl*hd fma
    double fops = 2.0 * 2.0 * (double)bh * sl * sl * hd;
    double t = fops / (avg_ms * 1e-3) / 1e12;
    printf("bench_ds_gen: avg_ms=%.3f  tflops_2mma=%.2f  (causal_factor=%.2f)\n",
           avg_ms, t, causal ? 0.5 : 1.0);

    cudaFree(dQ8); cudaFree(dK8); cudaFree(dV8);
    cudaFree(dO_O_g); cudaFree(dO_g_g);
    cudaFree(dL); cudaFree(dD); cudaFree(dS_out); cudaFree(dS_T_out);
    return 0;
}
