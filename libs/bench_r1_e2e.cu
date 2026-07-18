// R1 E2E honest bench: D → ds_gen → dV(sealed) → dk_new → dq_new.
//   Two runs: (A) sequential single-stream, (B) multi-stream with events.
//   Three-gradient BIT-EXACT verification on 11 forms including CANARY,
//   against sealed references (dQ AA1 / dK sealed / dV Yarus-1).
//   Fingerprint EXPECT-dict abort: 5 kernels must match expected ptxas.
//
//   Streams topology:
//     Stream A: D-precompute → ds_gen → dk_new
//     Stream B: [wait D-event] → dV → [wait ds_gen-event] → dq_new
//   Wait event ensures dq_new sees materialized dS_nat.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "fa_bwd_common.cuh"

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) {                \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                    \
            cudaGetErrorString(e)); std::exit(1); }} while (0)

namespace fa_bwd_dk {
void launch_d_precompute(const __half *O, const __half *dO, float *D,
                          int bh, int sl, int hd, cudaStream_t stream);
void launch_dk(const uint8_t *Q, const uint8_t *K, const uint8_t *V,
               const __half *dO_g, const float *L, const float *D,
               float *dK,
               int bh, int sl, int hd, int causal, int window,
               float scale, cudaStream_t stream);
__global__ void kernel_dk(const uint8_t *, const uint8_t *, const uint8_t *,
                          const __half *, const float *, const float *, float *,
                          int, int, int, int, int, float);
__global__ void kernel_d_precompute(const __half *, const __half *, float *,
                                    int, int, int);
}
namespace fa_bwd_dq {
void launch_dq(const uint8_t *Q, const uint8_t *K, const uint8_t *V,
               const __half *dO_g, const float *L, const float *D,
               float *dQ,
               int bh, int sl, int hd, int causal, int window,
               float scale, cudaStream_t stream);
__global__ void kernel_dq(const uint8_t *, const uint8_t *, const uint8_t *,
                          const __half *, const float *, const float *, float *,
                          int, int, int, int, int, float);
}
namespace fa_bwd_dv_mma_p1 {
void launch(const uint8_t *Q, const uint8_t *K, const __half *dO_g,
            const float *L, float *dV,
            int bh, int sl, int hd, int causal, int window,
            float scale, cudaStream_t stream);
__global__ void kernel_dv_mma_p1(const uint8_t *, const uint8_t *, const __half *,
                                 const float *, float *,
                                 int, int, int, int, int, float);
}
namespace fa_bwd_ds_gen {
void launch_ds_gen(const uint8_t *Q, const uint8_t *K, const uint8_t *V,
                    const __half *dO_g, const float *L, const float *D,
                    uint8_t *dS_out, uint8_t *dS_T_out,
                    int bh, int sl, int hd, int causal, int window,
                    float scale, cudaStream_t stream);
__global__ void kernel_ds_gen(const uint8_t *, const uint8_t *, const uint8_t *,
                              const __half *, const float *, const float *,
                              uint8_t *, uint8_t *,
                              int, int, int, int, int, float);
}
namespace fa_bwd_dk_new {
void launch_dk_new(const uint8_t *Q, const uint8_t *dS_T, float *dK,
                    int bh, int sl, int hd, int causal, int window,
                    float scale, cudaStream_t stream);
__global__ void kernel_dk_new(const uint8_t *, const uint8_t *, float *,
                              int, int, int, int, int, float);
}
namespace fa_bwd_dq_new {
void launch_dq_new(const uint8_t *K, const uint8_t *dS_nat, float *dQ,
                    int bh, int sl, int hd, int causal, int window,
                    float scale, cudaStream_t stream);
__global__ void kernel_dq_new(const uint8_t *, const uint8_t *, float *,
                              int, int, int, int, int, float);
}

// ==================== Fingerprint EXPECT-dict ====================
struct FpExpect { const char *name; const void *fptr; int expected_regs; };

static void fingerprint_gate() {
    FpExpect gate[] = {
        {"kernel_d_precompute", (const void*)fa_bwd_dk::kernel_d_precompute,   38},
        {"kernel_ds_gen",       (const void*)fa_bwd_ds_gen::kernel_ds_gen,    130},
        {"kernel_dv_mma_p1",    (const void*)fa_bwd_dv_mma_p1::kernel_dv_mma_p1, 129},
        {"kernel_dk_new",       (const void*)fa_bwd_dk_new::kernel_dk_new,     96},
        {"kernel_dq_new",       (const void*)fa_bwd_dq_new::kernel_dq_new,     56},
    };
    int n = sizeof(gate) / sizeof(gate[0]);
    int fails = 0;
    for (int i = 0; i < n; ++i) {
        cudaFuncAttributes fa;
        cudaError_t e = cudaFuncGetAttributes(&fa, gate[i].fptr);
        if (e != cudaSuccess) {
            printf("FINGERPRINT %-22s ERROR: %s\n", gate[i].name, cudaGetErrorString(e));
            fails++;
            continue;
        }
        bool ok = (fa.numRegs == gate[i].expected_regs);
        printf("FINGERPRINT %-22s numRegs=%3d (expected %3d) %s\n",
               gate[i].name, fa.numRegs, gate[i].expected_regs, ok ? "OK" : "MISMATCH");
        if (!ok) fails++;
    }
    if (fails) {
        fprintf(stderr, "FINGERPRINT gate FAILED (%d mismatch) — aborting\n", fails);
        std::exit(2);
    }
}

// ==================== BIT-EXACT chain 11 forms × 3 gradients ====================
struct Form { const char *name; int bh; int sl; int causal; int window; };

static int bit_exact_chain() {
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

    int ok_total = 0;
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
        float   *ddQ_ref, *ddK_ref, *ddV_ref;
        float   *ddQ_gen, *ddK_gen, *ddV_gen;
        uint8_t *dS_nat, *dS_T;

        CK(cudaMalloc(&dQ8, sz)); CK(cudaMalloc(&dK8, sz)); CK(cudaMalloc(&dV8, sz));
        CK(cudaMalloc(&dO_O_g, sz*sizeof(__half))); CK(cudaMalloc(&dO_g_g, sz*sizeof(__half)));
        CK(cudaMalloc(&dL, lsz*sizeof(float))); CK(cudaMalloc(&dD, lsz*sizeof(float)));
        CK(cudaMalloc(&ddQ_ref, sz*sizeof(float))); CK(cudaMalloc(&ddK_ref, sz*sizeof(float))); CK(cudaMalloc(&ddV_ref, sz*sizeof(float)));
        CK(cudaMalloc(&ddQ_gen, sz*sizeof(float))); CK(cudaMalloc(&ddK_gen, sz*sizeof(float))); CK(cudaMalloc(&ddV_gen, sz*sizeof(float)));
        CK(cudaMalloc(&dS_nat, dsz)); CK(cudaMalloc(&dS_T, dsz));

        CK(cudaMemcpy(dQ8, Q8.data(), sz, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dK8, K8.data(), sz, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dV8, V8.data(), sz, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dO_O_g, O16.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dO_g_g, dO16.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dL, L32.data(), lsz*sizeof(float), cudaMemcpyHostToDevice));

        fa_bwd_dk::launch_d_precompute(dO_O_g, dO_g_g, dD, bh, sl, hd, 0);
        CK(cudaDeviceSynchronize());

        float scale = 1.0f / std::sqrt((float)hd);

        // Reference chain: sealed dQ + sealed dK + sealed dV
        CK(cudaMemset(ddQ_ref, 0, sz*sizeof(float)));
        CK(cudaMemset(ddK_ref, 0, sz*sizeof(float)));
        CK(cudaMemset(ddV_ref, 0, sz*sizeof(float)));
        fa_bwd_dq::launch_dq(dQ8, dK8, dV8, dO_g_g, dL, dD, ddQ_ref, bh, sl, hd, F.causal, F.window, scale, 0);
        fa_bwd_dk::launch_dk(dQ8, dK8, dV8, dO_g_g, dL, dD, ddK_ref, bh, sl, hd, F.causal, F.window, scale, 0);
        fa_bwd_dv_mma_p1::launch(dQ8, dK8, dO_g_g, dL, ddV_ref, bh, sl, hd, F.causal, F.window, scale, 0);
        CK(cudaDeviceSynchronize());

        // R1 chain: ds_gen → dk_new + dq_new + sealed dV (dV unchanged reference reused)
        CK(cudaMemset(ddQ_gen, 0, sz*sizeof(float)));
        CK(cudaMemset(ddK_gen, 0, sz*sizeof(float)));
        CK(cudaMemset(ddV_gen, 0, sz*sizeof(float)));
        fa_bwd_ds_gen::launch_ds_gen(dQ8, dK8, dV8, dO_g_g, dL, dD, dS_nat, dS_T,
                                      bh, sl, hd, F.causal, F.window, scale, 0);
        fa_bwd_dv_mma_p1::launch(dQ8, dK8, dO_g_g, dL, ddV_gen, bh, sl, hd, F.causal, F.window, scale, 0);
        fa_bwd_dk_new::launch_dk_new(dQ8, dS_T, ddK_gen, bh, sl, hd, F.causal, F.window, scale, 0);
        fa_bwd_dq_new::launch_dq_new(dK8, dS_nat, ddQ_gen, bh, sl, hd, F.causal, F.window, scale, 0);
        CK(cudaDeviceSynchronize());

        auto cmp = [&](const char *tag, float *a, float *b) -> size_t {
            std::vector<float> ha(sz), hb(sz);
            cudaMemcpy(ha.data(), a, sz*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(hb.data(), b, sz*sizeof(float), cudaMemcpyDeviceToHost);
            size_t mism = 0; double mx = 0.0;
            for (size_t p = 0; p < sz; ++p) {
                uint32_t ua = *reinterpret_cast<uint32_t*>(&ha[p]);
                uint32_t ub = *reinterpret_cast<uint32_t*>(&hb[p]);
                if (ua != ub) { mism++; double d = std::fabs((double)ha[p]-(double)hb[p]); if (d > mx) mx = d; }
            }
            printf("  %s mism=%zu max_abs_diff=%.3e %s\n", tag, mism, mx, mism == 0 ? "BIT-EXACT" : "MISMATCH");
            return mism;
        };
        printf("[%-6s bh=%d sl=%4d caus=%d wnd=%d]\n", F.name, bh, sl, F.causal, F.window);
        size_t m_dq = cmp("dQ", ddQ_ref, ddQ_gen);
        size_t m_dk = cmp("dK", ddK_ref, ddK_gen);
        size_t m_dv = cmp("dV", ddV_ref, ddV_gen);
        if (m_dq == 0 && m_dk == 0 && m_dv == 0) ok_total++;

        cudaFree(dQ8); cudaFree(dK8); cudaFree(dV8);
        cudaFree(dO_O_g); cudaFree(dO_g_g);
        cudaFree(dL); cudaFree(dD);
        cudaFree(ddQ_ref); cudaFree(ddK_ref); cudaFree(ddV_ref);
        cudaFree(ddQ_gen); cudaFree(ddK_gen); cudaFree(ddV_gen);
        cudaFree(dS_nat); cudaFree(dS_T);
    }
    printf("\n=== CHAIN BIT-EXACT SUMMARY ===\n  forms all-3 bit-exact: %d / %d\n\n", ok_total, N);
    return (ok_total == N) ? 0 : 1;
}

// ==================== Wall runs ====================
struct WallResult {
    double total_ms;
    double d_ms, dsgen_ms, dv_ms, dk_ms, dq_ms;
};

static WallResult run_sequential(
    uint8_t *dQ, uint8_t *dK, uint8_t *dV, __half *dOO, __half *dOG,
    float *dL, float *dD, float *ddV, float *ddK, float *ddQ,
    uint8_t *dS_nat, uint8_t *dS_T,
    int bh, int sl, int hd, int causal, int window, float scale, int iters)
{
    cudaEvent_t e0, e1, e2, e3, e4, e5;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
    cudaEventCreate(&e3); cudaEventCreate(&e4); cudaEventCreate(&e5);
    double sum_d=0, sum_ds=0, sum_dv=0, sum_dk=0, sum_dq=0, sum_tot=0;
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(e0);
        fa_bwd_dk::launch_d_precompute(dOO, dOG, dD, bh, sl, hd, 0);
        cudaEventRecord(e1);
        fa_bwd_ds_gen::launch_ds_gen(dQ, dK, dV, dOG, dL, dD, dS_nat, dS_T,
                                     bh, sl, hd, causal, window, scale, 0);
        cudaEventRecord(e2);
        fa_bwd_dv_mma_p1::launch(dQ, dK, dOG, dL, ddV, bh, sl, hd, causal, window, scale, 0);
        cudaEventRecord(e3);
        fa_bwd_dk_new::launch_dk_new(dQ, dS_T, ddK, bh, sl, hd, causal, window, scale, 0);
        cudaEventRecord(e4);
        fa_bwd_dq_new::launch_dq_new(dK, dS_nat, ddQ, bh, sl, hd, causal, window, scale, 0);
        cudaEventRecord(e5);
        cudaEventSynchronize(e5);
        float d_ms, ds_ms, dv_ms, dk_ms, dq_ms, tot_ms;
        cudaEventElapsedTime(&d_ms,  e0, e1);
        cudaEventElapsedTime(&ds_ms, e1, e2);
        cudaEventElapsedTime(&dv_ms, e2, e3);
        cudaEventElapsedTime(&dk_ms, e3, e4);
        cudaEventElapsedTime(&dq_ms, e4, e5);
        cudaEventElapsedTime(&tot_ms, e0, e5);
        sum_d+=d_ms; sum_ds+=ds_ms; sum_dv+=dv_ms; sum_dk+=dk_ms; sum_dq+=dq_ms; sum_tot+=tot_ms;
    }
    WallResult r = {sum_tot/iters, sum_d/iters, sum_ds/iters, sum_dv/iters, sum_dk/iters, sum_dq/iters};
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
    cudaEventDestroy(e3); cudaEventDestroy(e4); cudaEventDestroy(e5);
    return r;
}

static double run_streams(
    uint8_t *dQ, uint8_t *dK, uint8_t *dV, __half *dOO, __half *dOG,
    float *dL, float *dD, float *ddV, float *ddK, float *ddQ,
    uint8_t *dS_nat, uint8_t *dS_T,
    int bh, int sl, int hd, int causal, int window, float scale, int iters)
{
    cudaStream_t sA, sB;
    cudaStreamCreate(&sA); cudaStreamCreate(&sB);
    cudaEvent_t ev_D, ev_ds, e_start, e_end;
    cudaEventCreate(&ev_D); cudaEventCreate(&ev_ds);
    cudaEventCreate(&e_start); cudaEventCreate(&e_end);
    double sum = 0;
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(e_start, sA);
        // Stream A: D → ds_gen → dk_new
        fa_bwd_dk::launch_d_precompute(dOO, dOG, dD, bh, sl, hd, sA);
        cudaEventRecord(ev_D, sA);
        fa_bwd_ds_gen::launch_ds_gen(dQ, dK, dV, dOG, dL, dD, dS_nat, dS_T,
                                     bh, sl, hd, causal, window, scale, sA);
        cudaEventRecord(ev_ds, sA);
        fa_bwd_dk_new::launch_dk_new(dQ, dS_T, ddK, bh, sl, hd, causal, window, scale, sA);
        // Stream B: [wait D] → dV;  [wait ds_gen] → dq_new
        cudaStreamWaitEvent(sB, ev_D, 0);
        fa_bwd_dv_mma_p1::launch(dQ, dK, dOG, dL, ddV, bh, sl, hd, causal, window, scale, sB);
        cudaStreamWaitEvent(sB, ev_ds, 0);
        fa_bwd_dq_new::launch_dq_new(dK, dS_nat, ddQ, bh, sl, hd, causal, window, scale, sB);
        cudaEventRecord(e_end, sB);
        // Join: also wait for stream A (dk_new)
        cudaStreamWaitEvent(sB, e_end, 0);
        cudaStreamSynchronize(sA);
        cudaStreamSynchronize(sB);
        float ms; cudaEventElapsedTime(&ms, e_start, e_end);
        sum += ms;
    }
    cudaStreamDestroy(sA); cudaStreamDestroy(sB);
    cudaEventDestroy(ev_D); cudaEventDestroy(ev_ds);
    cudaEventDestroy(e_start); cudaEventDestroy(e_end);
    return sum / iters;
}

int main(int argc, char **argv) {
    int mode_bit_exact = (argc >= 2 && std::string(argv[1]) == "bitexact") ? 1 : 0;

    printf("=== bench_r1_e2e: fingerprint gate ===\n");
    fingerprint_gate();

    if (mode_bit_exact) {
        printf("\n=== BIT-EXACT chain 11 forms × 3 gradients ===\n");
        return bit_exact_chain();
    }

    // Wall runs
    int bh = 128, sl = 8192, hd = 128, causal = 0, window = 0;
    int warmup = 5, iters = 20;

    size_t sz  = (size_t)bh * sl * hd;
    size_t lsz = (size_t)bh * sl;
    int stride_ds = (sl + 15) & ~15;
    size_t dsz = (size_t)bh * sl * stride_ds;

    printf("\nbench_r1_e2e: bh=%d sl=%d hd=%d causal=%d window=%d warmup=%d iters=%d\n",
           bh, sl, hd, causal, window, warmup, iters);
    printf("  transient dS_nat + dS_T = %zu MB\n", (2*dsz) >> 20);

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

    uint8_t *dQ, *dK, *dV;
    __half  *dOG, *dOO;
    float   *dL, *dD, *ddV, *ddK, *ddQ;
    uint8_t *dS_nat, *dS_T;
    CK(cudaMalloc(&dQ, sz)); CK(cudaMalloc(&dK, sz)); CK(cudaMalloc(&dV, sz));
    CK(cudaMalloc(&dOO, sz*sizeof(__half))); CK(cudaMalloc(&dOG, sz*sizeof(__half)));
    CK(cudaMalloc(&dL, lsz*sizeof(float))); CK(cudaMalloc(&dD, lsz*sizeof(float)));
    CK(cudaMalloc(&ddV, sz*sizeof(float))); CK(cudaMalloc(&ddK, sz*sizeof(float))); CK(cudaMalloc(&ddQ, sz*sizeof(float)));
    CK(cudaMalloc(&dS_nat, dsz)); CK(cudaMalloc(&dS_T, dsz));
    CK(cudaMemcpy(dQ, Q8.data(), sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, K8.data(), sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, V8.data(), sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dOO, O16.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dOG, dO16.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dL, L32.data(), lsz*sizeof(float), cudaMemcpyHostToDevice));

    float scale = 1.0f / std::sqrt((float)hd);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        fa_bwd_dk::launch_d_precompute(dOO, dOG, dD, bh, sl, hd, 0);
        fa_bwd_ds_gen::launch_ds_gen(dQ, dK, dV, dOG, dL, dD, dS_nat, dS_T,
                                     bh, sl, hd, causal, window, scale, 0);
        fa_bwd_dv_mma_p1::launch(dQ, dK, dOG, dL, ddV, bh, sl, hd, causal, window, scale, 0);
        fa_bwd_dk_new::launch_dk_new(dQ, dS_T, ddK, bh, sl, hd, causal, window, scale, 0);
        fa_bwd_dq_new::launch_dq_new(dK, dS_nat, ddQ, bh, sl, hd, causal, window, scale, 0);
    }
    CK(cudaDeviceSynchronize());

    printf("\n=== 007-A: SEQUENTIAL (single stream) ===\n");
    WallResult r = run_sequential(dQ, dK, dV, dOO, dOG, dL, dD, ddV, ddK, ddQ, dS_nat, dS_T,
                                   bh, sl, hd, causal, window, scale, iters);
    printf("  total_ms=%.3f\n", r.total_ms);
    printf("  D=%.3f  ds_gen=%.3f  dV=%.3f  dk_new=%.3f  dq_new=%.3f\n",
           r.d_ms, r.dsgen_ms, r.dv_ms, r.dk_ms, r.dq_ms);
    double sum_kernels = r.d_ms + r.dsgen_ms + r.dv_ms + r.dk_ms + r.dq_ms;
    double overhead = r.total_ms - sum_kernels;
    printf("  sum_kernels=%.3f  overhead=%.3f\n", sum_kernels, overhead);

    printf("\n=== 007-B: STREAMS (D → sA(ds_gen→dk_new) || sB(dV→dq_new)) ===\n");
    double streams_ms = run_streams(dQ, dK, dV, dOO, dOG, dL, dD, ddV, ddK, ddQ, dS_nat, dS_T,
                                    bh, sl, hd, causal, window, scale, iters);
    printf("  streams_ms=%.3f\n", streams_ms);

    // FLOPS arithmetic — two conventions
    double base = (double)bh * sl * sl * hd * (causal ? 0.5 : 1.0);
    double flops_16 = 16.0 * base;   // Tri Dao Variant 3 (executed): dV=4 + dK=6 + dQ=6
    double flops_10 = 10.0 * base;   // fused-ideal (no recomputation): dV=2 + dK=4 + dQ=4 (theoretical lower bound)
    // R1 actually executes: ds_gen (Q·K^T + dO·V^T) + dV_sealed (2 MMA) + dk_new (1 MMA) + dq_new (1 MMA) = 6 MMA total
    double flops_r1_actual = 2.0 * 6.0 * base;    // 12·N²·d (6 MMA × 2 FMA)

    printf("\n=== FLOPS conventions ===\n");
    printf("  base = 2 * bh * sl^2 * hd * cf = %.4e\n", base);
    printf("  Tri Dao Variant 3 (executed reference 16N²d): %.4e\n", flops_16);
    printf("  R1 actual executed (6 MMA = 12N²d):           %.4e\n", flops_r1_actual);
    printf("  fused-ideal (10N²d):                          %.4e\n", flops_10);

    printf("\n=== Sequential TFLOPS ===\n");
    printf("  vs 16N²d: %.2f T\n", flops_16 / (r.total_ms * 1e-3) / 1e12);
    printf("  vs 12N²d: %.2f T\n", flops_r1_actual / (r.total_ms * 1e-3) / 1e12);
    printf("  vs 10N²d: %.2f T\n", flops_10 / (r.total_ms * 1e-3) / 1e12);
    printf("\n=== Streams TFLOPS ===\n");
    printf("  vs 16N²d: %.2f T\n", flops_16 / (streams_ms * 1e-3) / 1e12);
    printf("  vs 12N²d: %.2f T\n", flops_r1_actual / (streams_ms * 1e-3) / 1e12);
    printf("  vs 10N²d: %.2f T\n", flops_10 / (streams_ms * 1e-3) / 1e12);

    CK(cudaFree(dQ)); CK(cudaFree(dK)); CK(cudaFree(dV));
    CK(cudaFree(dOO)); CK(cudaFree(dOG));
    CK(cudaFree(dL)); CK(cudaFree(dD));
    CK(cudaFree(ddV)); CK(cudaFree(ddK)); CK(cudaFree(ddQ));
    CK(cudaFree(dS_nat)); CK(cudaFree(dS_T));
    return 0;
}
