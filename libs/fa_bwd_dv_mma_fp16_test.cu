// =====================================================================
//  fa_bwd_dv_mma_fp16_test.cu — B2.1 DIAGNOSTIC harness for FP16 recompute.
//
//  Same forms + tol as fa_bwd_dv_mma_test.cu, but launches the FP16
//  recompute kernel and uploads Q,K as FP16 (no e4m3 quantize).
//
//  Adds worst-N (b, i, d) cluster dump per form to confirm whether
//  worst cases stay clustered at small-i (causal N_eff geometry hypothesis)
//  even when FP8 is eliminated.
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define DV_CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) {            \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                   \
            cudaGetErrorString(e)); exit(1); }} while (0)

namespace fa_bwd_dv_mma_fp16 {
void launch(const __half *Q, const __half *K, const __half *dO_g,
            const float *L, float *dV,
            int bh, int sl, int hd, int causal, int window,
            float scale, cudaStream_t stream);
}

// ====== FP64 forward (same as before) ======
static void fa_fwd_fp64(
    const double *Q, const double *K, const double *V,
    double *O, double *L_out,
    int bh, int sl, int hd, int causal, int window)
{
    double scale = 1.0 / std::sqrt((double)hd);
    for (int b = 0; b < bh; ++b) {
        const double *Qb = Q + b * sl * hd;
        const double *Kb = K + b * sl * hd;
        const double *Vb = V + b * sl * hd;
        double *Ob = O + b * sl * hd;
        double *Lb = L_out + b * sl;
        for (int i = 0; i < sl; ++i) {
            std::vector<double> s(sl);
            double m_i = -INFINITY;
            int any_valid = 0;
            for (int j = 0; j < sl; ++j) {
                if (causal && j > i)                  { s[j] = -INFINITY; continue; }
                if (window > 0 && j < i + 1 - window) { s[j] = -INFINITY; continue; }
                double dot = 0.0;
                for (int d = 0; d < hd; ++d) dot += Qb[i*hd+d] * Kb[j*hd+d];
                s[j] = dot * scale;
                if (s[j] > m_i) m_i = s[j];
                any_valid = 1;
            }
            if (!any_valid) m_i = 0.0;
            double l_i = 0.0;
            std::vector<double> p(sl, 0.0);
            for (int j = 0; j < sl; ++j) {
                if (!std::isfinite(s[j])) continue;
                p[j] = std::exp(s[j] - m_i);
                l_i += p[j];
            }
            for (int d = 0; d < hd; ++d) {
                double acc = 0.0;
                for (int j = 0; j < sl; ++j) acc += p[j] * Vb[j*hd+d];
                Ob[i*hd+d] = acc / (l_i > 0 ? l_i : 1.0);
            }
            Lb[i] = m_i + std::log(l_i > 0 ? l_i : 1.0);
        }
    }
}

static void fa_bwd_dv_fp64(
    const double *Q, const double *K,
    const double *L, const double *dO,
    double *dV,
    int bh, int sl, int hd, int causal, int window)
{
    double scale = 1.0 / std::sqrt((double)hd);
    size_t tot = (size_t)bh * sl * hd;
    for (size_t i = 0; i < tot; ++i) dV[i] = 0.0;
    for (int b = 0; b < bh; ++b) {
        const double *Qb  = Q  + b * sl * hd;
        const double *Kb  = K  + b * sl * hd;
        const double *Lb  = L  + b * sl;
        const double *dOb = dO + b * sl * hd;
        double *dVb       = dV + b * sl * hd;
        for (int i = 0; i < sl; ++i) {
            std::vector<double> P(sl, 0.0);
            for (int j = 0; j < sl; ++j) {
                if (causal && j > i)                  continue;
                if (window > 0 && j < i + 1 - window) continue;
                double dot = 0.0;
                for (int d = 0; d < hd; ++d) dot += Qb[i*hd+d] * Kb[j*hd+d];
                double s = dot * scale;
                P[j] = std::exp(s - Lb[i]);
            }
            for (int j = 0; j < sl; ++j) {
                if (causal && j > i)                  continue;
                if (window > 0 && j < i + 1 - window) continue;
                for (int d = 0; d < hd; ++d)
                    dVb[j*hd+d] += P[j] * dOb[i*hd+d];
            }
        }
    }
}

static void fa_bwd_dv_fp32(
    const float *Q, const float *K,
    const float *L, const float *dO,
    float *dV,
    int bh, int sl, int hd, int causal, int window)
{
    float scale = 1.0f / std::sqrt((float)hd);
    size_t tot = (size_t)bh * sl * hd;
    for (size_t i = 0; i < tot; ++i) dV[i] = 0.0f;
    for (int b = 0; b < bh; ++b) {
        const float *Qb  = Q  + b * sl * hd;
        const float *Kb  = K  + b * sl * hd;
        const float *Lb  = L  + b * sl;
        const float *dOb = dO + b * sl * hd;
        float       *dVb = dV + b * sl * hd;
        for (int i = 0; i < sl; ++i) {
            std::vector<float> P(sl, 0.0f);
            for (int j = 0; j < sl; ++j) {
                if (causal && j > i)                  continue;
                if (window > 0 && j < i + 1 - window) continue;
                float dot = 0.0f;
                for (int d = 0; d < hd; ++d) dot += Qb[i*hd+d] * Kb[j*hd+d];
                float s = dot * scale;
                P[j] = std::exp(s - Lb[i]);
            }
            for (int j = 0; j < sl; ++j) {
                if (causal && j > i)                  continue;
                if (window > 0 && j < i + 1 - window) continue;
                for (int d = 0; d < hd; ++d)
                    dVb[j*hd+d] += P[j] * dOb[i*hd+d];
            }
        }
    }
}

// ====== Compare with worst-N cluster dump ======
struct DiscRow { double abs_err; int b, i, d; double ref, got; };

struct CmpStats {
    size_t n_checked, n_passed;
    size_t n_above_floor, n_passed_above_floor;
    double max_abs, max_rel_above_floor;
    int worst_b, worst_i, worst_d;
    double worst_ref, worst_got;
    std::vector<DiscRow> top_disc;  // top-N by abs_err
};

template <typename Tref>
static CmpStats compare_dv(const float *got, const Tref *ref,
                           int bh, int sl, int hd,
                           double abs_tol, double rel_tol, double sig,
                           int top_n)
{
    CmpStats s{};
    s.worst_b = s.worst_i = s.worst_d = -1;
    size_t tot = (size_t)bh * sl * hd;
    std::vector<DiscRow> all;
    all.reserve(tot);
    for (size_t idx = 0; idx < tot; ++idx) {
        double r = (double)ref[idx];
        double g = (double)got[idx];
        double a = std::fabs(g - r);
        double rel = a / (std::fabs(r) + 1e-30);
        bool ok = a < abs_tol + rel_tol * std::fabs(r);
        s.n_checked++;
        if (ok) s.n_passed++;
        if (a > s.max_abs) {
            s.max_abs = a;
            int b_  = (int)(idx / ((size_t)sl * hd));
            int rem = (int)(idx % ((size_t)sl * hd));
            s.worst_b = b_;
            s.worst_i = rem / hd;
            s.worst_d = rem % hd;
            s.worst_ref = r;
            s.worst_got = g;
        }
        if (std::fabs(r) > sig) {
            s.n_above_floor++;
            if (ok) s.n_passed_above_floor++;
            if (rel > s.max_rel_above_floor) s.max_rel_above_floor = rel;
        }
        DiscRow row;
        row.abs_err = a;
        row.b   = (int)(idx / ((size_t)sl * hd));
        row.i   = (int)((idx % ((size_t)sl * hd)) / hd);
        row.d   = (int)(idx % hd);
        row.ref = r;
        row.got = g;
        all.push_back(row);
    }
    std::partial_sort(all.begin(),
                      all.begin() + std::min((size_t)top_n, all.size()),
                      all.end(),
                      [](const DiscRow &a, const DiscRow &b) { return a.abs_err > b.abs_err; });
    s.top_disc.assign(all.begin(),
                      all.begin() + std::min((size_t)top_n, all.size()));
    return s;
}

static void print_cmp(const char *label, const CmpStats &s)
{
    double pct_tot   = 100.0 * (double)s.n_passed / (double)s.n_checked;
    double pct_above = s.n_above_floor
        ? 100.0 * (double)s.n_passed_above_floor / (double)s.n_above_floor
        : 0.0;
    printf("    %-22s  pass %zu/%zu (%.4f%%)  above-floor %zu/%zu (%.4f%%)  "
           "max_abs %.3e  max_rel_af %.3e  worst@(b=%d,i=%d,d=%d) ref=%.4e got=%.4e\n",
           label, s.n_passed, s.n_checked, pct_tot,
           s.n_passed_above_floor, s.n_above_floor, pct_above,
           s.max_abs, s.max_rel_above_floor,
           s.worst_b, s.worst_i, s.worst_d, s.worst_ref, s.worst_got);
}

static void print_cluster(const char *label, const CmpStats &s)
{
    // Bucket top-20 by i-position to test small-i clustering hypothesis.
    int hist[16] = {0};   // i ∈ [0,1,2,3,4,5,6,7,8,15,16-31,32-63,64-127,128-255,256-511,512+]
    auto bucket = [](int i) -> int {
        if (i < 8) return i;
        if (i < 16) return 8;
        if (i < 32) return 9;
        if (i < 64) return 10;
        if (i < 128) return 11;
        if (i < 256) return 12;
        if (i < 512) return 13;
        if (i < 1024) return 14;
        return 15;
    };
    int top = std::min((int)s.top_disc.size(), 20);
    for (int k = 0; k < top; ++k) hist[bucket(s.top_disc[k].i)]++;
    printf("    %s top-%d worst by i-bucket: "
           "[0..7]=%d,%d,%d,%d,%d,%d,%d,%d  [8-15]=%d [16-31]=%d [32-63]=%d "
           "[64-127]=%d [128-255]=%d [256-511]=%d [512-1023]=%d [1024+]=%d\n",
           label, top,
           hist[0], hist[1], hist[2], hist[3], hist[4], hist[5], hist[6], hist[7],
           hist[8], hist[9], hist[10], hist[11], hist[12], hist[13], hist[14], hist[15]);
}

struct Form {
    const char *name;
    int bh, sl, hd;
    int causal, window;
    bool canary;
};

static bool run_form(const Form &F, unsigned seed,
                     double abs_tol, double rel_tol, double sig)
{
    const int bh = F.bh, sl = F.sl, hd = F.hd;
    const size_t sz  = (size_t)bh * sl * hd;
    const size_t lsz = (size_t)bh * sl;

    printf("\n[%s%s]  bh=%d sl=%d hd=%d causal=%d window=%d (seed=%u)\n",
           F.canary ? "CANARY " : "",
           F.name, bh, sl, hd, F.causal, F.window, seed);

    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 0.6);
    std::vector<double> Q64(sz), K64(sz), V64(sz), dO64(sz);
    for (size_t i = 0; i < sz; ++i) {
        Q64 [i] = dist(rng);
        K64 [i] = dist(rng);
        V64 [i] = dist(rng);
        dO64[i] = dist(rng);
    }

    std::vector<double> O64(sz), L64(lsz);
    fa_fwd_fp64(Q64.data(), K64.data(), V64.data(),
                O64.data(), L64.data(),
                bh, sl, hd, F.causal, F.window);

    std::vector<double> dV64(sz, 0.0);
    fa_bwd_dv_fp64(Q64.data(), K64.data(), L64.data(), dO64.data(),
                   dV64.data(), bh, sl, hd, F.causal, F.window);

    std::vector<float> Q32(sz), K32(sz), dO32(sz);
    std::vector<float> L32(lsz);
    for (size_t i = 0; i < sz; ++i) {
        Q32 [i] = (float)Q64 [i];
        K32 [i] = (float)K64 [i];
        dO32[i] = (float)dO64[i];
    }
    for (size_t i = 0; i < lsz; ++i) L32[i] = (float)L64[i];

    std::vector<float> dV_cpu32(sz, 0.0f);
    fa_bwd_dv_fp32(Q32.data(), K32.data(), L32.data(), dO32.data(),
                   dV_cpu32.data(), bh, sl, hd, F.causal, F.window);

    // FP16 upload for diagnostic kernel (no e4m3 quantize).
    std::vector<__half> Q16(sz), K16(sz), dO16(sz);
    for (size_t i = 0; i < sz; ++i) {
        Q16 [i] = __float2half_rn(Q32 [i]);
        K16 [i] = __float2half_rn(K32 [i]);
        dO16[i] = __float2half_rn(dO32[i]);
    }

    __half *dQ_dev = nullptr, *dK_dev = nullptr, *ddO_dev = nullptr;
    float  *dL_dev = nullptr, *ddV_dev = nullptr;
    DV_CK(cudaMalloc(&dQ_dev,  sz  * sizeof(__half)));
    DV_CK(cudaMalloc(&dK_dev,  sz  * sizeof(__half)));
    DV_CK(cudaMalloc(&ddO_dev, sz  * sizeof(__half)));
    DV_CK(cudaMalloc(&dL_dev,  lsz * sizeof(float)));
    DV_CK(cudaMalloc(&ddV_dev, sz  * sizeof(float)));
    DV_CK(cudaMemcpy(dQ_dev,  Q16.data(),  sz  * sizeof(__half), cudaMemcpyHostToDevice));
    DV_CK(cudaMemcpy(dK_dev,  K16.data(),  sz  * sizeof(__half), cudaMemcpyHostToDevice));
    DV_CK(cudaMemcpy(ddO_dev, dO16.data(), sz  * sizeof(__half), cudaMemcpyHostToDevice));
    DV_CK(cudaMemcpy(dL_dev,  L32.data(),  lsz * sizeof(float),  cudaMemcpyHostToDevice));
    DV_CK(cudaMemset(ddV_dev, 0, sz * sizeof(float)));

    float scale = 1.0f / std::sqrt((float)hd);
    fa_bwd_dv_mma_fp16::launch(
        dQ_dev, dK_dev, ddO_dev, dL_dev, ddV_dev,
        bh, sl, hd, F.causal, F.window, scale, 0);
    DV_CK(cudaDeviceSynchronize());

    std::vector<float> dV_gpu(sz);
    DV_CK(cudaMemcpy(dV_gpu.data(), ddV_dev, sz * sizeof(float),
                     cudaMemcpyDeviceToHost));

    DV_CK(cudaFree(dQ_dev));   DV_CK(cudaFree(dK_dev));
    DV_CK(cudaFree(ddO_dev));  DV_CK(cudaFree(dL_dev));
    DV_CK(cudaFree(ddV_dev));

    CmpStats vs_cpu  = compare_dv<float> (dV_gpu.data(), dV_cpu32.data(),
                                          bh, sl, hd, abs_tol, rel_tol, sig, 20);
    CmpStats vs_fp64 = compare_dv<double>(dV_gpu.data(), dV64.data(),
                                          bh, sl, hd, abs_tol, rel_tol, sig, 20);
    print_cmp ("MMA-FP16 vs CPU-FP32 ", vs_cpu);
    print_cluster("MMA-FP16 vs CPU-FP32 ", vs_cpu);
    print_cmp ("MMA-FP16 vs FP64-gold", vs_fp64);

    bool ok_cpu = vs_cpu.n_above_floor == 0 ||
                  100.0 * vs_cpu.n_passed_above_floor / vs_cpu.n_above_floor >= 99.5;
    bool ok_fp64 = vs_fp64.n_above_floor == 0 ||
                   100.0 * vs_fp64.n_passed_above_floor / vs_fp64.n_above_floor >= 99.5;
    printf("    verdict: CPU-cmp %s  FP64-cmp %s\n",
           ok_cpu  ? "PASS" : "FAIL",
           ok_fp64 ? "PASS" : "FAIL");
    return ok_cpu && ok_fp64;
}

int main(int argc, char **argv)
{
    unsigned seed = 42;
    if (argc >= 2) seed = (unsigned)std::atoi(argv[1]);

    double abs_tol = 1e-3;
    double rel_tol = 5e-3;
    double sig     = 1e-2;
    if (argc >= 3) abs_tol = std::atof(argv[2]);
    if (argc >= 4) rel_tol = std::atof(argv[3]);

    printf("=== B2.1 DIAG: FP16-recompute MMA dV validation ===\n");
    printf("    tol: abs %.0e  rel %.0e  sig %.0e  seed=%u\n",
           abs_tol, rel_tol, sig, seed);
    printf("    Q,K uploaded as FP16 (no e4m3 quantize). Q·K^T via m16n8k16 FP16 -> FP32 acc.\n");

    std::vector<Form> forms = {
        {"F1 bh=1 sl=128  caus=0 wnd=0",   1, 128,  128, 0, 0,   false},
        {"F2 bh=1 sl=128  caus=1 wnd=0",   1, 128,  128, 1, 0,   false},
        {"F3 bh=2 sl=256  caus=0 wnd=0",   2, 256,  128, 0, 0,   false},
        {"F4 bh=2 sl=256  caus=1 wnd=0",   2, 256,  128, 1, 0,   false},
        {"F5 bh=4 sl=384  caus=0 wnd=0",   4, 384,  128, 0, 0,   false},
        {"F6 bh=4 sl=384  caus=1 wnd=0",   4, 384,  128, 1, 0,   false},
        {"F7 bh=1 sl=512  caus=0 wnd=128", 1, 512,  128, 0, 128, false},
        {"F8 bh=1 sl=512  caus=1 wnd=128", 1, 512,  128, 1, 128, false},
        {"F9 bh=1 sl=2048 caus=0 wnd=0",   1, 2048, 128, 0, 0,   false},
        {"F10 bh=1 sl=2048 caus=1 wnd=0",  1, 2048, 128, 1, 0,   false},
        {"sl=300 wnd=96 caus=0",           1, 300,  128, 0, 96,  true},
    };

    int n_pass = 0, n_total = (int)forms.size();
    for (auto &F : forms) {
        bool ok = run_form(F, seed, abs_tol, rel_tol, sig);
        if (ok) n_pass++;
    }
    printf("\n=== SUMMARY ===\n");
    printf("    forms passed: %d / %d\n", n_pass, n_total);
    return (n_pass == n_total) ? 0 : 1;
}
