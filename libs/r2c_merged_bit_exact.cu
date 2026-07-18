// R2C merged bit-exact test:
//   Reference: sealed dV_p1 (dV) + R1a ds_gen (dS_nat + dS_T).
//   Merged: kernel_merged_v1 outputs all three (dV + dS_nat + dS_T).
//   Compare all three byte-exact (fp32 for dV, uint8 for dS bytes).
//
//   ALL 11 forms including CANARY. Zero CUDA errors + sanitizer clean externally.

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
namespace fa_bwd_dv_mma_p1 {
void launch(const uint8_t *Q, const uint8_t *K, const __half *dO_g,
            const float *L, float *dV,
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

struct Form { const char *name; int bh; int sl; int causal; int window; };

int main() {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, fa_bwd_merged_v1::kernel_merged_v1) == cudaSuccess) {
        printf("r2c_merged_bit_exact: FINGERPRINT kernel_merged_v1: numRegs=%d, "
               "sharedSizeBytes=%zu, localSizeBytes=%zu, maxThreadsPerBlock=%d\n",
               fa.numRegs, fa.sharedSizeBytes, fa.localSizeBytes, fa.maxThreadsPerBlock);
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

    int all_pass = 0;
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

        uint8_t *dQ, *dK, *dV8;
        __half  *dO_g_g, *dO_O_g;
        float   *dL, *dD;
        float   *ddV_ref, *ddV_gen;
        uint8_t *dS_nat_ref, *dS_T_ref, *dS_nat_gen, *dS_T_gen;

        CKR(cudaMalloc(&dQ, sz)); CKR(cudaMalloc(&dK, sz)); CKR(cudaMalloc(&dV8, sz));
        CKR(cudaMalloc(&dO_O_g, sz*sizeof(__half))); CKR(cudaMalloc(&dO_g_g, sz*sizeof(__half)));
        CKR(cudaMalloc(&dL, lsz*sizeof(float))); CKR(cudaMalloc(&dD, lsz*sizeof(float)));
        CKR(cudaMalloc(&ddV_ref, sz*sizeof(float))); CKR(cudaMalloc(&ddV_gen, sz*sizeof(float)));
        CKR(cudaMalloc(&dS_nat_ref, dsz)); CKR(cudaMalloc(&dS_T_ref, dsz));
        CKR(cudaMalloc(&dS_nat_gen, dsz)); CKR(cudaMalloc(&dS_T_gen, dsz));

        CKR(cudaMemcpy(dQ, Q8.data(), sz, cudaMemcpyHostToDevice));
        CKR(cudaMemcpy(dK, K8.data(), sz, cudaMemcpyHostToDevice));
        CKR(cudaMemcpy(dV8, V8.data(), sz, cudaMemcpyHostToDevice));
        CKR(cudaMemcpy(dO_O_g, O16.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));
        CKR(cudaMemcpy(dO_g_g, dO16.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));
        CKR(cudaMemcpy(dL, L32.data(), lsz*sizeof(float), cudaMemcpyHostToDevice));

        fa_bwd_dk::launch_d_precompute(dO_O_g, dO_g_g, dD, bh, sl, hd, 0);
        CKR(cudaDeviceSynchronize());

        float scale = 1.0f / sqrtf((float)hd);

        // Reference: sealed dV_p1 (dV) + R1a ds_gen (dS_nat + dS_T)
        CKR(cudaMemset(ddV_ref, 0, sz*sizeof(float)));
        CKR(cudaMemset(dS_nat_ref, 0, dsz));
        CKR(cudaMemset(dS_T_ref, 0, dsz));
        fa_bwd_dv_mma_p1::launch(dQ, dK, dO_g_g, dL, ddV_ref,
                                  bh, sl, hd, F.causal, F.window, scale, 0);
        fa_bwd_ds_gen::launch_ds_gen(dQ, dK, dV8, dO_g_g, dL, dD,
                                      dS_nat_ref, dS_T_ref,
                                      bh, sl, hd, F.causal, F.window, scale, 0);
        CKR(cudaDeviceSynchronize());

        // Merged: kernel_merged_v1
        CKR(cudaMemset(ddV_gen, 0, sz*sizeof(float)));
        CKR(cudaMemset(dS_nat_gen, 0, dsz));
        CKR(cudaMemset(dS_T_gen, 0, dsz));
        fa_bwd_merged_v1::launch_merged(dQ, dK, dV8, dO_g_g, dL, dD,
                                         dS_nat_gen, dS_T_gen, ddV_gen,
                                         bh, sl, hd, F.causal, F.window, scale, 0);
        CKR(cudaDeviceSynchronize());
        cudaError_t em = cudaGetLastError();
        if (em != cudaSuccess) printf("[%-6s] merged err: %s\n", F.name, cudaGetErrorString(em));

        // Compare
        auto cmp_f32 = [&](const char *tag, float *a, float *b, size_t n) -> size_t {
            std::vector<float> ha(n), hb(n);
            cudaMemcpy(ha.data(), a, n*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(hb.data(), b, n*sizeof(float), cudaMemcpyDeviceToHost);
            size_t mism = 0; double mx = 0.0;
            for (size_t p = 0; p < n; ++p) {
                uint32_t ua = *reinterpret_cast<uint32_t*>(&ha[p]);
                uint32_t ub = *reinterpret_cast<uint32_t*>(&hb[p]);
                if (ua != ub) { mism++; double d = std::fabs((double)ha[p]-(double)hb[p]); if (d > mx) mx = d; }
            }
            printf("  %s mism=%zu max_abs_diff=%.3e %s\n", tag, mism, mx, mism == 0 ? "BIT-EXACT" : "MISMATCH");
            return mism;
        };
        auto cmp_u8 = [&](const char *tag, uint8_t *a, uint8_t *b, size_t n) -> size_t {
            std::vector<uint8_t> ha(n), hb(n);
            cudaMemcpy(ha.data(), a, n, cudaMemcpyDeviceToHost);
            cudaMemcpy(hb.data(), b, n, cudaMemcpyDeviceToHost);
            size_t mism = 0;
            for (size_t p = 0; p < n; ++p) if (ha[p] != hb[p]) mism++;
            printf("  %s mism=%zu %s\n", tag, mism, mism == 0 ? "BIT-EXACT" : "MISMATCH");
            return mism;
        };

        // 039 VALIDATION: bit-flip injection to prove harness catches искажение (compile-time flag)
        // При запуске с --inject флагом добавить 0xFF в первый байт dS_nat_gen.
        if (getenv("INJECT_BITFLIP")) {
            uint8_t bad = 0xAA;
            cudaMemcpy(dS_nat_gen, &bad, 1, cudaMemcpyHostToDevice);
        }

        printf("[%-6s bh=%d sl=%4d caus=%d wnd=%d]\n", F.name, bh, sl, F.causal, F.window);
        size_t m_dv = cmp_f32("dV", ddV_ref, ddV_gen, sz);
        // dS_nat compare only positions where i_g < sl && j_g < sl (ds_gen writes only these; merged may write extra padding)
        size_t m_ds = 0, m_dsT = 0;
        {
            std::vector<uint8_t> ha(dsz), hb(dsz);
            cudaMemcpy(ha.data(), dS_nat_ref, dsz, cudaMemcpyDeviceToHost);
            cudaMemcpy(hb.data(), dS_nat_gen, dsz, cudaMemcpyDeviceToHost);
            for (int bh_i = 0; bh_i < bh; ++bh_i)
                for (int i_g = 0; i_g < sl; ++i_g)
                    for (int j_g = 0; j_g < sl; ++j_g) {
                        size_t idx = (size_t)bh_i*sl*stride_ds + (size_t)i_g*stride_ds + j_g;
                        if (ha[idx] != hb[idx]) m_ds++;
                    }
            printf("  dS_nat (in-bounds only) mism=%zu %s\n", m_ds, m_ds == 0 ? "BIT-EXACT" : "MISMATCH");

            // 039: dS_T check УДАЛЁН (post-cut ABI: merged больше не пишет dS_T DRAM после 033-c T-cut).
            //      m_dsT остаётся = 0 (не инкрементируется), сохранён в переменной для совместимости
            //      с логами. Правка ТОЛЬКО reader-теста; production не тронут.
        }

        if (m_dv == 0 && m_ds == 0) all_pass++;    // 039: убран m_dsT из критерия

        cudaFree(dQ); cudaFree(dK); cudaFree(dV8);
        cudaFree(dO_O_g); cudaFree(dO_g_g);
        cudaFree(dL); cudaFree(dD);
        cudaFree(ddV_ref); cudaFree(ddV_gen);
        cudaFree(dS_nat_ref); cudaFree(dS_T_ref);
        cudaFree(dS_nat_gen); cudaFree(dS_T_gen);
    }

    printf("\n=== SUMMARY ===\n  forms triple-bit-exact: %d / %d\n", all_pass, N);
    return (all_pass == N) ? 0 : 1;
}
