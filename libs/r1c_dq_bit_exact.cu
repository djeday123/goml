// R1c bit-exact test: kernel_dq_new(dS_nat, K) vs sealed AA1 dQ baseline.
//   Pipeline: sealed dQ → dQ_ref;  ds_gen → dS_nat;  dq_new(K, dS_nat) → dQ_gen.
//   Expected: dQ_ref == dQ_gen byte-by-byte (fp32 IEEE, same MMA-C fp16-acc path).
//   Critical fp16-acc order: kt outer → kb inner → ni innermost — identical in both.

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
namespace fa_bwd_dq {
void launch_dq(const uint8_t *Q, const uint8_t *K, const uint8_t *V,
               const __half *dO_g, const float *L, const float *D,
               float *dQ,
               int bh, int sl, int hd, int causal, int window,
               float scale, cudaStream_t stream);
}
namespace fa_bwd_ds_gen {
void launch_ds_gen(const uint8_t *Q, const uint8_t *K, const uint8_t *V,
                    const __half *dO_g, const float *L, const float *D,
                    uint8_t *dS_out, uint8_t *dS_T_out,
                    int bh, int sl, int hd, int causal, int window,
                    float scale, cudaStream_t stream);
}
namespace fa_bwd_dq_new {
void launch_dq_new(const uint8_t *K, const uint8_t *dS_nat,
                   float *dQ,
                   int bh, int sl, int hd, int causal, int window,
                   float scale, cudaStream_t stream);
__global__ void kernel_dq_new(const uint8_t *, const uint8_t *, float *,
                              int, int, int, int, int, float);
}

struct Form { const char *name; int bh; int sl; int causal; int window; };

int main() {
    // Fingerprint gate
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, fa_bwd_dq_new::kernel_dq_new) == cudaSuccess) {
        printf("r1c_dq_bit_exact: FINGERPRINT kernel_dq_new: numRegs=%d, "
               "sharedSizeBytes=%zu, localSizeBytes=%zu, maxThreadsPerBlock=%d\n",
               fa.numRegs, fa.sharedSizeBytes, fa.localSizeBytes,
               fa.maxThreadsPerBlock);
    }
    const int hd = 128;
    Form forms[] = {
        {"F1", 1, 128, 0, 0}, {"F2", 1, 128, 1, 0},
        {"F3", 2, 256, 0, 0}, {"F4", 2, 256, 1, 0},
        {"F5", 4, 384, 0, 0}, {"F6", 4, 384, 1, 0},
        {"F7", 1, 512, 0, 128}, {"F8", 1, 512, 1, 128},
        {"F9", 1, 2048, 0, 0}, {"F10", 1, 2048, 1, 0},
        {"CANARY", 1, 300, 0, 96},
    };
    const int N = sizeof(forms) / sizeof(forms[0]);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.6f);

    int total_ok = 0;
    for (int f = 0; f < N; ++f) {
        Form F = forms[f];
        int bh = F.bh, sl = F.sl;
        size_t sz  = (size_t)bh * sl * hd;
        size_t lsz = (size_t)bh * sl;
        int stride_ds = (sl + 15) & ~15;
        size_t dsz = (size_t)bh * sl * stride_ds;

        std::vector<uint8_t> Q8(sz), K8(sz), V8(sz);
        std::vector<__half>  O16(sz), dO16(sz);
        std::vector<float>   L32(lsz);
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
        float   *dL, *dD;
        float   *ddQ_ref, *ddQ_gen;
        uint8_t *dS_nat, *dS_T;

        CKR(cudaMalloc(&dQ8, sz)); CKR(cudaMalloc(&dK8, sz)); CKR(cudaMalloc(&dV8, sz));
        CKR(cudaMalloc(&dO_O_g, sz*sizeof(__half))); CKR(cudaMalloc(&dO_g_g, sz*sizeof(__half)));
        CKR(cudaMalloc(&dL, lsz*sizeof(float))); CKR(cudaMalloc(&dD, lsz*sizeof(float)));
        CKR(cudaMalloc(&ddQ_ref, sz*sizeof(float)));
        CKR(cudaMalloc(&ddQ_gen, sz*sizeof(float)));
        CKR(cudaMalloc(&dS_nat, dsz)); CKR(cudaMalloc(&dS_T, dsz));

        CKR(cudaMemcpy(dQ8, Q8.data(), sz, cudaMemcpyHostToDevice));
        CKR(cudaMemcpy(dK8, K8.data(), sz, cudaMemcpyHostToDevice));
        CKR(cudaMemcpy(dV8, V8.data(), sz, cudaMemcpyHostToDevice));
        CKR(cudaMemcpy(dO_O_g, O16.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));
        CKR(cudaMemcpy(dO_g_g, dO16.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));
        CKR(cudaMemcpy(dL, L32.data(), lsz*sizeof(float), cudaMemcpyHostToDevice));
        CKR(cudaMemset(ddQ_ref, 0, sz*sizeof(float)));
        CKR(cudaMemset(ddQ_gen, 0, sz*sizeof(float)));

        fa_bwd_dk::launch_d_precompute(dO_O_g, dO_g_g, dD, bh, sl, hd, 0);
        CKR(cudaDeviceSynchronize());

        float scale = 1.0f / sqrtf((float)hd);

        // Reference: sealed AA1 dQ
        fa_bwd_dq::launch_dq(dQ8, dK8, dV8, dO_g_g, dL, dD, ddQ_ref,
                             bh, sl, hd, F.causal, F.window, scale, 0);
        cudaDeviceSynchronize();
        cudaError_t e_seal = cudaGetLastError();
        if (e_seal != cudaSuccess) printf("[%-6s] SEALED_DQ: %s\n", F.name, cudaGetErrorString(e_seal));

        // ds_gen materializes dS_nat + dS_T
        fa_bwd_ds_gen::launch_ds_gen(dQ8, dK8, dV8, dO_g_g, dL, dD,
                                      dS_nat, dS_T,
                                      bh, sl, hd, F.causal, F.window, scale, 0);
        cudaDeviceSynchronize();
        cudaError_t e_gen = cudaGetLastError();
        if (e_gen != cudaSuccess) printf("[%-6s] DS_GEN: %s\n", F.name, cudaGetErrorString(e_gen));

        // dq_new consumes dS_nat + K
        fa_bwd_dq_new::launch_dq_new(dK8, dS_nat, ddQ_gen,
                                     bh, sl, hd, F.causal, F.window, scale, 0);
        cudaDeviceSynchronize();
        cudaError_t e_new = cudaGetLastError();
        if (e_new != cudaSuccess) printf("[%-6s] DQ_NEW: %s\n", F.name, cudaGetErrorString(e_new));

        std::vector<float> h_ref(sz), h_gen(sz);
        CKR(cudaMemcpy(h_ref.data(), ddQ_ref, sz*sizeof(float), cudaMemcpyDeviceToHost));
        CKR(cudaMemcpy(h_gen.data(), ddQ_gen, sz*sizeof(float), cudaMemcpyDeviceToHost));

        // Byte-exact fp32 IEEE compare
        size_t mism = 0;
        double max_abs_diff = 0.0;
        for (size_t p = 0; p < sz; ++p) {
            float r = h_ref[p], g = h_gen[p];
            uint32_t ur = *reinterpret_cast<uint32_t*>(&r);
            uint32_t ug = *reinterpret_cast<uint32_t*>(&g);
            if (ur != ug) {
                mism++;
                double d = std::fabs((double)r - (double)g);
                if (d > max_abs_diff) max_abs_diff = d;
            }
        }
        bool ok = (mism == 0);
        printf("[%-6s bh=%d sl=%4d caus=%d wnd=%d] total=%zu mism=%zu max_abs_diff=%.3e  %s\n",
               F.name, bh, sl, F.causal, F.window, sz, mism, max_abs_diff,
               ok ? "BIT-EXACT" : "MISMATCH");
        if (ok) total_ok++;

        cudaFree(dQ8); cudaFree(dK8); cudaFree(dV8);
        cudaFree(dO_O_g); cudaFree(dO_g_g);
        cudaFree(dL); cudaFree(dD);
        cudaFree(ddQ_ref); cudaFree(ddQ_gen);
        cudaFree(dS_nat); cudaFree(dS_T);
    }

    printf("\n=== SUMMARY ===\n    forms bit-exact: %d / %d\n", total_ok, N);
    return (total_ok == N) ? 0 : 1;
}
