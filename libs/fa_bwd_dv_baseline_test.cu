// =====================================================================
//  fa_bwd_dv_baseline_test.cu — двухуровневая валидация для B2.0.
//
//  Уровень 1 (CI):    GPU dV (FP32) ↔ CPU FP32 dV-only reference.
//                     hybrid tol: abs 1e-4 + rel 1e-3.
//  Уровень 2 (Debug): GPU dV (FP32 upcast → FP64) ↔ FP64 golden dV.
//                     same hybrid tol — для GPU FP32 vs FP64 ref ожидаем
//                     FP32 floor ~1e-4..1e-3 (выраженный rel_above_floor
//                     должен быть ≤ rel_tol).
//
//  Прогон: 8 форм + канарейка sl=300 wnd=96.
//
//  L canonicalization: L_i = m_i + log(l_i), natural log (как в FP64 ref
//  и как пишет _v121r_train_kernel.cu). Здесь L считается FP64-forward'ом
//  и потом cast в FP32 для GPU-baseline.
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

#define DV_CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) {            \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                   \
            cudaGetErrorString(e)); exit(1); }} while (0)

// External launcher from fa_bwd_dv_baseline.cu.
namespace fa_bwd_dv_baseline {
void launch(const float *Q, const float *K, const float *dO_g,
            const float *L, float *dV,
            int bh, int sl, int hd, int causal, int window,
            float scale, cudaStream_t stream);
}

// ---------------------------------------------------------------------
// FP64 forward — computes L only (we don't need O for dV-only test, but
// the standard formula L = m + log(l) requires both). О тоже считаем,
// он будет нужен для polished FP64 golden если переходим на dQ/dK.
// ---------------------------------------------------------------------
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
                if (causal && j > i)                        { s[j] = -INFINITY; continue; }
                if (window > 0 && j < i + 1 - window)       { s[j] = -INFINITY; continue; }
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

// ---------------------------------------------------------------------
// FP64 dV golden — copy from fa_bwd_cpu_reference_fp64_golden.cu, dV-only.
// ---------------------------------------------------------------------
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

// ---------------------------------------------------------------------
// FP32 dV reference — same algorithm, FP32. CI-level check.
// ---------------------------------------------------------------------
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

// ---------------------------------------------------------------------
// Compare: hybrid tol  |a - b| < abs_tol + rel_tol * |ref|
// Returns: max_abs, max_rel above floor (|ref| > sig), pct passed.
// ---------------------------------------------------------------------
struct CmpStats {
    size_t n_checked, n_passed;
    size_t n_above_floor, n_passed_above_floor;
    double max_abs, max_rel_above_floor;
    int worst_b, worst_i, worst_d;
    double worst_ref, worst_got;
};

static CmpStats compare_dv_fp64(const float *got, const double *ref,
                                int bh, int sl, int hd,
                                double abs_tol, double rel_tol, double sig)
{
    CmpStats s{};
    s.worst_b = s.worst_i = s.worst_d = -1;
    size_t tot = (size_t)bh * sl * hd;
    for (size_t idx = 0; idx < tot; ++idx) {
        double r = ref[idx];
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
    }
    return s;
}

static CmpStats compare_dv_fp32(const float *got, const float *ref,
                                int bh, int sl, int hd,
                                double abs_tol, double rel_tol, double sig)
{
    CmpStats s{};
    s.worst_b = s.worst_i = s.worst_d = -1;
    size_t tot = (size_t)bh * sl * hd;
    for (size_t idx = 0; idx < tot; ++idx) {
        double r = ref[idx];
        double g = got[idx];
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
    }
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

// ---------------------------------------------------------------------
// Form runner.
// ---------------------------------------------------------------------
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
    const size_t sz   = (size_t)bh * sl * hd;
    const size_t lsz  = (size_t)bh * sl;

    printf("\n[%s%s]  bh=%d sl=%d hd=%d causal=%d window=%d (seed=%u)\n",
           F.canary ? "CANARY " : "",
           F.name, bh, sl, hd, F.causal, F.window, seed);

    // ---- Generate FP64 inputs ----
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 0.6);
    std::vector<double> Q64(sz), K64(sz), V64(sz), dO64(sz);
    for (size_t i = 0; i < sz; ++i) {
        Q64 [i] = dist(rng);
        K64 [i] = dist(rng);
        V64 [i] = dist(rng);
        dO64[i] = dist(rng);
    }

    // ---- FP64 forward → L (and O, unused here) ----
    std::vector<double> O64(sz), L64(lsz);
    fa_fwd_fp64(Q64.data(), K64.data(), V64.data(),
                O64.data(), L64.data(),
                bh, sl, hd, F.causal, F.window);

    // ---- FP64 dV golden ----
    std::vector<double> dV64(sz, 0.0);
    fa_bwd_dv_fp64(Q64.data(), K64.data(), L64.data(), dO64.data(),
                   dV64.data(), bh, sl, hd, F.causal, F.window);

    // ---- Cast inputs to FP32 ----
    std::vector<float> Q32(sz), K32(sz), dO32(sz);
    std::vector<float> L32(lsz);
    for (size_t i = 0; i < sz; ++i) {
        Q32 [i] = (float)Q64 [i];
        K32 [i] = (float)K64 [i];
        dO32[i] = (float)dO64[i];
    }
    for (size_t i = 0; i < lsz; ++i) L32[i] = (float)L64[i];

    // ---- CPU FP32 dV reference ----
    std::vector<float> dV_cpu32(sz, 0.0f);
    fa_bwd_dv_fp32(Q32.data(), K32.data(), L32.data(), dO32.data(),
                   dV_cpu32.data(), bh, sl, hd, F.causal, F.window);

    // ---- GPU baseline ----
    float *dQ_dev = nullptr, *dK_dev = nullptr, *ddO_dev = nullptr;
    float *dL_dev = nullptr, *ddV_dev = nullptr;
    DV_CK(cudaMalloc(&dQ_dev,  sz  * sizeof(float)));
    DV_CK(cudaMalloc(&dK_dev,  sz  * sizeof(float)));
    DV_CK(cudaMalloc(&ddO_dev, sz  * sizeof(float)));
    DV_CK(cudaMalloc(&dL_dev,  lsz * sizeof(float)));
    DV_CK(cudaMalloc(&ddV_dev, sz  * sizeof(float)));
    DV_CK(cudaMemcpy(dQ_dev,  Q32.data(),  sz  * sizeof(float), cudaMemcpyHostToDevice));
    DV_CK(cudaMemcpy(dK_dev,  K32.data(),  sz  * sizeof(float), cudaMemcpyHostToDevice));
    DV_CK(cudaMemcpy(ddO_dev, dO32.data(), sz  * sizeof(float), cudaMemcpyHostToDevice));
    DV_CK(cudaMemcpy(dL_dev,  L32.data(),  lsz * sizeof(float), cudaMemcpyHostToDevice));
    DV_CK(cudaMemset(ddV_dev, 0, sz * sizeof(float)));

    float scale = 1.0f / std::sqrt((float)hd);
    fa_bwd_dv_baseline::launch(
        dQ_dev, dK_dev, ddO_dev, dL_dev, ddV_dev,
        bh, sl, hd, F.causal, F.window, scale, 0);
    DV_CK(cudaDeviceSynchronize());

    std::vector<float> dV_gpu(sz);
    DV_CK(cudaMemcpy(dV_gpu.data(), ddV_dev, sz * sizeof(float),
                     cudaMemcpyDeviceToHost));

    DV_CK(cudaFree(dQ_dev));  DV_CK(cudaFree(dK_dev));
    DV_CK(cudaFree(ddO_dev)); DV_CK(cudaFree(dL_dev));
    DV_CK(cudaFree(ddV_dev));

    // ---- Compare: GPU vs CPU FP32 (CI) ----
    CmpStats vs_cpu = compare_dv_fp32(dV_gpu.data(), dV_cpu32.data(),
                                      bh, sl, hd, abs_tol, rel_tol, sig);
    print_cmp("GPU vs CPU-FP32", vs_cpu);

    // ---- Compare: GPU vs FP64 golden (Debug) ----
    CmpStats vs_fp64 = compare_dv_fp64(dV_gpu.data(), dV64.data(),
                                       bh, sl, hd, abs_tol, rel_tol, sig);
    print_cmp("GPU vs FP64-golden", vs_fp64);

    // verdict per form: both checks must pass ≥99.5% above-floor
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

    // Hybrid tolerance (FP32 floor honest).
    double abs_tol = 1e-4;
    double rel_tol = 1e-3;
    double sig     = 1e-2;

    printf("=== B2.0 dV-only baseline validation ===\n");
    printf("    hybrid tol: abs %.0e  rel %.0e  sig %.0e  seed=%u\n",
           abs_tol, rel_tol, sig, seed);

    std::vector<Form> forms = {
        // 8 main forms — exercise (sl, causal, window) corners. hd=128 fixed.
        // Keep sl small to bound CPU O(N^2 hd) cost (FP64 forward is the slowest step).
        {"F1 bh=1 sl=128  caus=0 wnd=0",   1, 128, 128, 0, 0,   false},
        {"F2 bh=1 sl=128  caus=1 wnd=0",   1, 128, 128, 1, 0,   false},
        {"F3 bh=2 sl=256  caus=0 wnd=0",   2, 256, 128, 0, 0,   false},
        {"F4 bh=2 sl=256  caus=1 wnd=0",   2, 256, 128, 1, 0,   false},
        {"F5 bh=4 sl=384  caus=0 wnd=0",   4, 384, 128, 0, 0,   false},
        {"F6 bh=4 sl=384  caus=1 wnd=0",   4, 384, 128, 1, 0,   false},
        {"F7 bh=1 sl=512  caus=0 wnd=128", 1, 512, 128, 0, 128, false},
        {"F8 bh=1 sl=512  caus=1 wnd=128", 1, 512, 128, 1, 128, false},

        // Big forms — stress P-recompute on long sl. CPU FP64 forward is slow
        // here (O(sl^2 * hd) per b); these dominate harness wall time.
        {"F9 bh=1 sl=2048 caus=0 wnd=0",   1, 2048, 128, 0, 0,   false},
        {"F10 bh=1 sl=2048 caus=1 wnd=0",  1, 2048, 128, 1, 0,   false},

        // Canary — partial K-tile (sl % Bc != 0) + window.
        {"sl=300 wnd=96 caus=0",           1, 300, 128, 0, 96,  true},
    };

    int n_pass = 0, n_total = (int)forms.size();
    for (auto &F : forms) {
        bool ok = run_form(F, seed, abs_tol, rel_tol, sig);
        if (ok) n_pass++;
    }
    printf("\n=== SUMMARY ===\n");
    printf("    forms passed: %d / %d\n", n_pass, n_total);
    if (n_pass == n_total) {
        printf("    OVERALL: PASS — B2.0 dV-only baseline correct on all forms + canary.\n");
        return 0;
    } else {
        printf("    OVERALL: FAIL — %d form(s) regressed.\n", n_total - n_pass);
        return 1;
    }
}
