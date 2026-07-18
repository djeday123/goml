// R1b bit-exact test: kernel_dk_new(Q, dS_T) vs sealed dK baseline.
//   Pipeline: sealed dK → dK_ref;  ds_gen → dS_T;  dk_new(Q, dS_T) → dK_gen.
//   Expected: dK_ref == dK_gen byte-by-byte (fp32 IEEE, since same MMA-C path).

#include <cstdio>
#include <cstdlib>
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
void launch_dk(const uint8_t *Q, const uint8_t *K, const uint8_t *V,
               const __half *dO_g, const float *L, const float *D,
               float *dK,
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
namespace fa_bwd_dk_new {
void launch_dk_new(const uint8_t *Q, const uint8_t *dS_T,
                   float *dK,
                   int bh, int sl, int hd, int causal, int window,
                   float scale, cudaStream_t stream);
__global__ void kernel_dk_new(const uint8_t *, const uint8_t *, float *,
                              int, int, int, int, int, float);
}

struct Form { const char *name; int bh; int sl; int causal; int window; };

int main() {
    // Fingerprint gate
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, fa_bwd_dk_new::kernel_dk_new) == cudaSuccess) {
        printf("r1b_dk_bit_exact: FINGERPRINT kernel_dk_new: numRegs=%d, "
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
        // ABI-padded dS row stride for 16-byte alignment of each row's start (cp.async).
        // Canonical sl=8192 → stride==sl (zero impact); CANARY sl=300 → 304 (+1.3% mem).
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
        float   *ddK_ref, *ddK_gen;
        uint8_t *dS_nat, *dS_T;

        CK(cudaMalloc(&dQ8, sz)); CK(cudaMalloc(&dK8, sz)); CK(cudaMalloc(&dV8, sz));
        CK(cudaMalloc(&dO_O_g, sz*sizeof(__half))); CK(cudaMalloc(&dO_g_g, sz*sizeof(__half)));
        CK(cudaMalloc(&dL, lsz*sizeof(float))); CK(cudaMalloc(&dD, lsz*sizeof(float)));
        CK(cudaMalloc(&ddK_ref, sz*sizeof(float)));
        CK(cudaMalloc(&ddK_gen, sz*sizeof(float)));
        CK(cudaMalloc(&dS_nat, dsz)); CK(cudaMalloc(&dS_T, dsz));

        CK(cudaMemcpy(dQ8, Q8.data(), sz, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dK8, K8.data(), sz, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dV8, V8.data(), sz, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dO_O_g, O16.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dO_g_g, dO16.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dL, L32.data(), lsz*sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemset(ddK_ref, 0, sz*sizeof(float)));
        CK(cudaMemset(ddK_gen, 0, sz*sizeof(float)));

        fa_bwd_dk::launch_d_precompute(dO_O_g, dO_g_g, dD, bh, sl, hd, 0);
        CK(cudaDeviceSynchronize());

        float scale = 1.0f / sqrtf((float)hd);

        // Reference: sealed dK
        fa_bwd_dk::launch_dk(dQ8, dK8, dV8, dO_g_g, dL, dD, ddK_ref,
                             bh, sl, hd, F.causal, F.window, scale, 0);
        cudaDeviceSynchronize();
        cudaError_t e_seal = cudaGetLastError();
        if (e_seal != cudaSuccess) printf("[%-6s] SEALED_DK: %s\n", F.name, cudaGetErrorString(e_seal));

        fa_bwd_ds_gen::launch_ds_gen(dQ8, dK8, dV8, dO_g_g, dL, dD,
                                      dS_nat, dS_T,
                                      bh, sl, hd, F.causal, F.window, scale, 0);
        cudaDeviceSynchronize();
        cudaError_t e_gen = cudaGetLastError();
        if (e_gen != cudaSuccess) printf("[%-6s] DS_GEN: %s\n", F.name, cudaGetErrorString(e_gen));

        fa_bwd_dk_new::launch_dk_new(dQ8, dS_nat, ddK_gen,
                                      bh, sl, hd, F.causal, F.window, scale, 0);
        cudaDeviceSynchronize();
        cudaError_t e_new = cudaGetLastError();
        if (e_new != cudaSuccess) printf("[%-6s] DK_NEW: %s\n", F.name, cudaGetErrorString(e_new));

        std::vector<float> h_ref(sz), h_gen(sz);
        CK(cudaMemcpy(h_ref.data(), ddK_ref, sz*sizeof(float), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_gen.data(), ddK_gen, sz*sizeof(float), cudaMemcpyDeviceToHost));

        // Byte-exact compare (fp32 IEEE bit representation)
        size_t total = sz, mism = 0;
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
               F.name, bh, sl, F.causal, F.window, total, mism, max_abs_diff,
               ok ? "BIT-EXACT" : "MISMATCH");
        if (ok) total_ok++;

        cudaFree(dQ8); cudaFree(dK8); cudaFree(dV8);
        cudaFree(dO_O_g); cudaFree(dO_g_g);
        cudaFree(dL); cudaFree(dD);
        cudaFree(ddK_ref); cudaFree(ddK_gen);
        cudaFree(dS_nat); cudaFree(dS_T);
    }

    printf("\n=== SUMMARY ===\n    forms bit-exact: %d / %d\n", total_ok, N);
    return (total_ok == N) ? 0 : 1;
}
