// 044 I.3: дискриминатор bh=1 sl=8192 hd=128 (bench-сторона, ядра не тронуты).
//   Correctness: dV/dS_nat + dQ/dK vs текущий production цепь (self-consistent test).
//   Wall: 20-iter avg per-kernel (справочно с подписью недозаполнения).
//   NCu отдельным скриптом.

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
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); }} while (0)

namespace fa_bwd_dk {
void launch_d_precompute(const __half *, const __half *, float *, int, int, int, cudaStream_t);
__global__ void kernel_d_precompute(const __half *, const __half *, float *, int, int, int);
}
namespace fa_bwd_merged_v1 {
void launch_merged(const uint8_t *, const uint8_t *, const uint8_t *, const __half *, const float *, const float *,
                    uint8_t *, uint8_t *, float *, int, int, int, int, int, float, cudaStream_t);
__global__ void kernel_merged_v1(const uint8_t *, const uint8_t *, const uint8_t *, const __half *, const float *, const float *,
                                 uint8_t *, uint8_t *, float *, int, int, int, int, int, float);
}
namespace fa_bwd_dk_new {
void launch_dk_new(const uint8_t *, const uint8_t *, float *, int, int, int, int, int, float, cudaStream_t);
__global__ void kernel_dk_new(const uint8_t *, const uint8_t *, float *, int, int, int, int, int, float);
}
namespace fa_bwd_dq_new {
void launch_dq_new(const uint8_t *, const uint8_t *, float *, int, int, int, int, int, float, cudaStream_t);
__global__ void kernel_dq_new(const uint8_t *, const uint8_t *, float *, int, int, int, int, int, float);
}

int main() {
    const int bh = 1, sl = 8192, hd = 128, causal = 0, window = 0;
    const int warmup = 5, iters = 20;
    int stride_ds = (sl + 15) & ~15;

    // Fingerprint x4 (gate обязателен)
    cudaFuncAttributes fa;
    struct { const char *name; const void *fp; int exp; } gate[] = {
        {"D",       (const void*)fa_bwd_dk::kernel_d_precompute,      38},
        {"merged",  (const void*)fa_bwd_merged_v1::kernel_merged_v1, 252},
        {"dk_new",  (const void*)fa_bwd_dk_new::kernel_dk_new,       128},
        {"dq_new",  (const void*)fa_bwd_dq_new::kernel_dq_new,        69},
    };
    int fails = 0;
    for (int i = 0; i < 4; ++i) {
        cudaFuncGetAttributes(&fa, gate[i].fp);
        bool ok = (fa.numRegs == gate[i].exp);
        printf("FINGERPRINT %-8s numRegs=%3d (expected %3d) %s\n",
               gate[i].name, fa.numRegs, gate[i].exp, ok ? "OK" : "MISMATCH");
        if (!ok) fails++;
    }
    if (fails) { fprintf(stderr, "gate FAIL\n"); return 1; }

    printf("\n044 I.3: dispatcher bh=%d sl=%d hd=%d causal=%d [mode: bh=1 discriminator]\n",
           bh, sl, hd, causal);
    printf("Occupancy fill: merged=%.1f%% (128/376), dk=%.1f%% (128/752), dq=%.1f%% (128/1128)\n",
           128.0/376*100, 128.0/752*100, 128.0/1128*100);

    size_t sz  = (size_t)bh * sl * hd;
    size_t lsz = (size_t)bh * sl;
    size_t dsz = (size_t)bh * sl * stride_ds;

    std::vector<uint8_t> Q8(sz), K8(sz), V8(sz);
    std::vector<__half>  O16(sz), dO16(sz);
    std::vector<float>   L32(lsz);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.6f);
    for (size_t i = 0; i < sz;  ++i) { Q8[i]=float_to_e4m3_host(dist(rng));K8[i]=float_to_e4m3_host(dist(rng));V8[i]=float_to_e4m3_host(dist(rng));O16[i]=__float2half_rn(dist(rng));dO16[i]=__float2half_rn(dist(rng)); }
    for (size_t i = 0; i < lsz; ++i)   L32[i] = dist(rng);

    uint8_t *dQ, *dK, *dV8; __half *dOG, *dOO; float *dL, *dD, *ddV;
    uint8_t *dS_nat, *dS_T; float *ddK, *ddQ;
    CKR(cudaMalloc(&dQ, sz));CKR(cudaMalloc(&dK, sz));CKR(cudaMalloc(&dV8, sz));
    CKR(cudaMalloc(&dOO, sz*sizeof(__half)));CKR(cudaMalloc(&dOG, sz*sizeof(__half)));
    CKR(cudaMalloc(&dL, lsz*sizeof(float)));CKR(cudaMalloc(&dD, lsz*sizeof(float)));
    CKR(cudaMalloc(&ddV, sz*sizeof(float)));CKR(cudaMalloc(&ddK, sz*sizeof(float)));CKR(cudaMalloc(&ddQ, sz*sizeof(float)));
    CKR(cudaMalloc(&dS_nat, dsz));CKR(cudaMalloc(&dS_T, dsz));
    CKR(cudaMemcpy(dQ, Q8.data(), sz, cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(dK, K8.data(), sz, cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(dV8, V8.data(), sz, cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(dOO, O16.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(dOG, dO16.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(dL, L32.data(), lsz*sizeof(float), cudaMemcpyHostToDevice));

    float scale = 1.0f / sqrtf((float)hd);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        fa_bwd_dk::launch_d_precompute(dOO, dOG, dD, bh, sl, hd, 0);
        fa_bwd_merged_v1::launch_merged(dQ, dK, dV8, dOG, dL, dD, dS_nat, dS_T, ddV,
                                         bh, sl, hd, causal, window, scale, 0);
        fa_bwd_dk_new::launch_dk_new(dQ, dS_nat, ddK, bh, sl, hd, causal, window, scale, 0);
        fa_bwd_dq_new::launch_dq_new(dK, dS_nat, ddQ, bh, sl, hd, causal, window, scale, 0);
    }
    CKR(cudaDeviceSynchronize());

    // Wall per-kernel
    cudaEvent_t e0, e1, e2, e3, e4;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3); cudaEventCreate(&e4);
    float t_D=0, t_M=0, t_K=0, t_Q=0;
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(e0, 0);
        fa_bwd_dk::launch_d_precompute(dOO, dOG, dD, bh, sl, hd, 0);
        cudaEventRecord(e1, 0);
        fa_bwd_merged_v1::launch_merged(dQ, dK, dV8, dOG, dL, dD, dS_nat, dS_T, ddV,
                                         bh, sl, hd, causal, window, scale, 0);
        cudaEventRecord(e2, 0);
        fa_bwd_dk_new::launch_dk_new(dQ, dS_nat, ddK, bh, sl, hd, causal, window, scale, 0);
        cudaEventRecord(e3, 0);
        fa_bwd_dq_new::launch_dq_new(dK, dS_nat, ddQ, bh, sl, hd, causal, window, scale, 0);
        cudaEventRecord(e4, 0);
        cudaEventSynchronize(e4);
        float ms;
        cudaEventElapsedTime(&ms, e0, e1); t_D += ms;
        cudaEventElapsedTime(&ms, e1, e2); t_M += ms;
        cudaEventElapsedTime(&ms, e2, e3); t_K += ms;
        cudaEventElapsedTime(&ms, e3, e4); t_Q += ms;
    }
    t_D /= iters; t_M /= iters; t_K /= iters; t_Q /= iters;

    printf("\n044 I.3.c wall bh=1 sl=8192 (справочно, недозаполнение 34/17/11%%):\n");
    printf("  D=%.4f  merged=%.4f  dk_new=%.4f  dq_new=%.4f  total=%.4f\n",
           t_D, t_M, t_K, t_Q, t_D+t_M+t_K+t_Q);

    // Correctness self-consistency: два прогона одинаковые входы → одинаковые dK/dV/dQ выходы
    float *ddK_ref, *ddV_ref, *ddQ_ref;
    CKR(cudaMalloc(&ddK_ref, sz*sizeof(float)));
    CKR(cudaMalloc(&ddV_ref, sz*sizeof(float)));
    CKR(cudaMalloc(&ddQ_ref, sz*sizeof(float)));
    CKR(cudaMemset(ddV_ref, 0, sz*sizeof(float)));CKR(cudaMemset(ddK_ref, 0, sz*sizeof(float)));CKR(cudaMemset(ddQ_ref, 0, sz*sizeof(float)));
    fa_bwd_merged_v1::launch_merged(dQ, dK, dV8, dOG, dL, dD, dS_nat, dS_T, ddV_ref,
                                     bh, sl, hd, causal, window, scale, 0);
    fa_bwd_dk_new::launch_dk_new(dQ, dS_nat, ddK_ref, bh, sl, hd, causal, window, scale, 0);
    fa_bwd_dq_new::launch_dq_new(dK, dS_nat, ddQ_ref, bh, sl, hd, causal, window, scale, 0);

    CKR(cudaMemset(ddV, 0, sz*sizeof(float)));CKR(cudaMemset(ddK, 0, sz*sizeof(float)));CKR(cudaMemset(ddQ, 0, sz*sizeof(float)));
    fa_bwd_merged_v1::launch_merged(dQ, dK, dV8, dOG, dL, dD, dS_nat, dS_T, ddV,
                                     bh, sl, hd, causal, window, scale, 0);
    fa_bwd_dk_new::launch_dk_new(dQ, dS_nat, ddK, bh, sl, hd, causal, window, scale, 0);
    fa_bwd_dq_new::launch_dq_new(dK, dS_nat, ddQ, bh, sl, hd, causal, window, scale, 0);
    CKR(cudaDeviceSynchronize());

    auto cmp = [&](const char *tag, float *a, float *b) -> size_t {
        std::vector<float> ha(sz), hb(sz);
        cudaMemcpy(ha.data(), a, sz*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hb.data(), b, sz*sizeof(float), cudaMemcpyDeviceToHost);
        size_t mism = 0; float mx = 0;
        for (size_t i = 0; i < sz; ++i) { float d = fabsf(ha[i]-hb[i]); if (d>0) mism++; if (d>mx) mx=d; }
        printf("  %s mism=%zu max_abs_diff=%.3e %s\n", tag, mism, mx, mism == 0 ? "SELF-CONSISTENT" : "DIVERGENT");
        return mism;
    };
    printf("\n044 I.3.a correctness (self-consistency bh=1 sl=8192):\n");
    size_t m_dv = cmp("dV", ddV_ref, ddV);
    size_t m_dk = cmp("dK", ddK_ref, ddK);
    size_t m_dq = cmp("dQ", ddQ_ref, ddQ);
    printf("Overall: %s\n", (m_dv+m_dk+m_dq==0) ? "PASS" : "FAIL");

    cudaEventDestroy(e0);cudaEventDestroy(e1);cudaEventDestroy(e2);cudaEventDestroy(e3);cudaEventDestroy(e4);
    return 0;
}
