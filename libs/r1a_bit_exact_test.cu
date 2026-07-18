// R1a bit-exact test:
//   Run kernel_dq_dump (sealed AA1 instrumented, dumps smdS bytes to dS_ref)
//   and kernel_ds_gen (R1a probe, writes dS to dS_gen)
//   on identical inputs. Compare dS_ref vs dS_gen byte-by-byte.
//
//   Expectation: BYTE-EXACT match on all valid (i, j) positions (i < sl && j < sl,
//   plus causal/window mask contributions).
//
//   Forms cycled: F1..F10 + CANARY (mirrors fa_bwd_dq_test config).

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
}
namespace fa_bwd_dq_dump {
void launch_dq_dump(const uint8_t *Q, const uint8_t *K, const uint8_t *V,
                     const __half *dO_g, const float *L, const float *D,
                     float *dQ, uint8_t *dS_scratch,
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

struct Form { const char *name; int bh; int sl; int causal; int window; };

int main() {
    const int hd = 128;
    Form forms[] = {
        {"F1",     1,  128, 0, 0},
        {"F2",     1,  128, 1, 0},
        {"F3",     2,  256, 0, 0},
        {"F4",     2,  256, 1, 0},
        {"F5",     4,  384, 0, 0},
        {"F6",     4,  384, 1, 0},
        {"F7",     1,  512, 0, 128},
        {"F8",     1,  512, 1, 128},
        {"F9",     1, 2048, 0, 0},
        {"F10",    1, 2048, 1, 0},
        {"CANARY", 1,  300, 0, 96},
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
        size_t dsz = (size_t)bh * sl * sl;

        std::vector<uint8_t> Q8(sz), K8(sz), V8(sz);
        std::vector<__half>  O16(sz), dO16(sz);
        std::vector<float>   L32(lsz);
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
        float   *ddQ_ref, *ddQ_gen;
        uint8_t *dS_ref, *dS_gen, *dS_T_gen;

        CK(cudaMalloc(&dQ8, sz));
        CK(cudaMalloc(&dK8, sz));
        CK(cudaMalloc(&dV8, sz));
        CK(cudaMalloc(&dO_O_g, sz * sizeof(__half)));
        CK(cudaMalloc(&dO_g_g, sz * sizeof(__half)));
        CK(cudaMalloc(&dL, lsz * sizeof(float)));
        CK(cudaMalloc(&dD, lsz * sizeof(float)));
        CK(cudaMalloc(&ddQ_ref, sz * sizeof(float)));
        CK(cudaMalloc(&ddQ_gen, sz * sizeof(float)));
        CK(cudaMalloc(&dS_ref, dsz));
        CK(cudaMalloc(&dS_gen, dsz));
        CK(cudaMalloc(&dS_T_gen, dsz));

        CK(cudaMemcpy(dQ8, Q8.data(), sz, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dK8, K8.data(), sz, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dV8, V8.data(), sz, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dO_O_g, O16.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dO_g_g, dO16.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dL, L32.data(), lsz*sizeof(float), cudaMemcpyHostToDevice));

        CK(cudaMemset(dS_ref, 0xFF, dsz));   // sentinel: uninitialized bytes stay 0xFF
        CK(cudaMemset(dS_gen, 0xFF, dsz));
        CK(cudaMemset(dS_T_gen, 0xFF, dsz));

        // D-precompute
        fa_bwd_dk::launch_d_precompute(dO_O_g, dO_g_g, dD, bh, sl, hd, 0);
        CK(cudaDeviceSynchronize());

        float scale = 1.0f / std::sqrt((float)hd);

        // Reference: sealed AA1 dQ instrumented, dumps smdS bytes to dS_ref
        fa_bwd_dq_dump::launch_dq_dump(dQ8, dK8, dV8, dO_g_g, dL, dD,
                                        ddQ_ref, dS_ref,
                                        bh, sl, hd, F.causal, F.window, scale, 0);
        // Candidate: R1a/R1b ds_gen (dual-write dS_nat + dS_T)
        fa_bwd_ds_gen::launch_ds_gen(dQ8, dK8, dV8, dO_g_g, dL, dD,
                                      dS_gen, dS_T_gen,
                                      bh, sl, hd, F.causal, F.window, scale, 0);
        CK(cudaDeviceSynchronize());

        std::vector<uint8_t> h_ref(dsz), h_gen(dsz), h_T_gen(dsz);
        CK(cudaMemcpy(h_ref.data(),   dS_ref,   dsz, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_gen.data(),   dS_gen,   dsz, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_T_gen.data(), dS_T_gen, dsz, cudaMemcpyDeviceToHost));

        // Gate 1: dS_nat vs sealed reference (regression on natural layout).
        size_t g1_compared=0, g1_mism=0, g1_sr=0, g1_sg=0;
        for (size_t p=0; p<dsz; ++p) {
            uint8_t r=h_ref[p], g=h_gen[p];
            bool rs=(r==0xFF), gs=(g==0xFF);
            if (rs&&gs) continue;
            if (rs) { g1_sr++; continue; }
            if (gs) { g1_sg++; continue; }
            g1_compared++;
            if (r!=g) g1_mism++;
        }

        // Gate 2: dS_T[j,i] == dS_nat[i,j] transpose consistency (per-batch).
        //   Iterate all (b, i, j); read h_gen[b*sl*sl + i*sl + j] and h_T_gen[b*sl*sl + j*sl + i].
        size_t g2_compared=0, g2_mism=0, g2_sn=0, g2_sT=0;
        for (int bb=0; bb<bh; ++bb) {
            size_t off = (size_t)bb * sl * sl;
            for (int i=0; i<sl; ++i) {
                for (int j=0; j<sl; ++j) {
                    uint8_t n = h_gen  [off + (size_t)i*sl + j];
                    uint8_t t = h_T_gen[off + (size_t)j*sl + i];
                    bool ns=(n==0xFF), ts=(t==0xFF);
                    if (ns&&ts) continue;
                    if (ns) { g2_sn++; continue; }
                    if (ts) { g2_sT++; continue; }
                    g2_compared++;
                    if (n!=t) g2_mism++;
                }
            }
        }

        bool ok1 = (g1_mism==0) && (g1_sr==0) && (g1_sg==0);
        bool ok2 = (g2_mism==0) && (g2_sn==0) && (g2_sT==0);
        bool ok = ok1 && ok2;
        printf("[%-6s bh=%d sl=%4d caus=%d wnd=%d] "
               "G1 nat vs ref: cmp=%zu mism=%zu sr=%zu sg=%zu | "
               "G2 T vs nat: cmp=%zu mism=%zu sn=%zu sT=%zu  %s\n",
               F.name, bh, sl, F.causal, F.window,
               g1_compared, g1_mism, g1_sr, g1_sg,
               g2_compared, g2_mism, g2_sn, g2_sT,
               ok ? "PASS" : "FAIL");
        if (ok) total_ok++;

        cudaFree(dQ8); cudaFree(dK8); cudaFree(dV8);
        cudaFree(dO_O_g); cudaFree(dO_g_g);
        cudaFree(dL); cudaFree(dD);
        cudaFree(ddQ_ref); cudaFree(ddQ_gen);
        cudaFree(dS_ref); cudaFree(dS_gen); cudaFree(dS_T_gen);
    }

    printf("\n=== SUMMARY ===\n    forms bit-exact: %d / %d\n", total_ok, N);
    return (total_ok == N) ? 0 : 1;
}
