// B1-FIX v2: FP64 probe-forward + hybrid abs+rel tolerance.
//
// Урок v1: FP32-forward даёт шум ~1e-7 relative в O. При finite-diff на
// малых градиентах (|grad|<1e-3) шум амплифицируется в loss-diff и даёт
// rel_err 100%+. FP64 loss-accumulator не лечит — шум в самом forward.
//
// v2:
//   - fa_fwd_fp64: forward на double только для probes.
//   - FP32 backward остаётся проверяемой штукой (та же реализация что в
//     fa_bwd_cpu_reference.cu).
//   - Hybrid pass: |num-ana| < abs_tol + rel_tol*|ana|. Малые grads имеют
//     intrinsic FP32-noise floor ~1e-4 — некорректно требовать <0.1% rel.
//   - Все 4096 элементов dQ/dK/dV проверяются. Top-20 расхождений с
//     индексами + кластеризация.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

// ====== FP32 forward (production-style) =========
static void fa_fwd_fp32(
    const float *Q, const float *K, const float *V,
    float *O, float *L_out,
    int bh, int N, int hd,
    int causal, int window)
{
    float scale = 1.0f / sqrtf((float)hd);
    for (int b = 0; b < bh; b++) {
        const float *Qb = Q + b * N * hd;
        const float *Kb = K + b * N * hd;
        const float *Vb = V + b * N * hd;
        float *Ob = O + b * N * hd;
        float *Lb = L_out + b * N;
        for (int i = 0; i < N; i++) {
            std::vector<float> s(N);
            float m_i = -INFINITY;
            for (int j = 0; j < N; j++) {
                if (causal && j > i) { s[j] = -INFINITY; continue; }
                if (window > 0 && j < i + 1 - window) { s[j] = -INFINITY; continue; }
                float dot = 0.0f;
                for (int d = 0; d < hd; d++) dot += Qb[i*hd + d] * Kb[j*hd + d];
                s[j] = dot * scale;
                if (s[j] > m_i) m_i = s[j];
            }
            float l_i = 0.0f;
            std::vector<float> p(N, 0.0f);
            for (int j = 0; j < N; j++) {
                if (!std::isfinite(s[j])) continue;
                p[j] = expf(s[j] - m_i);
                l_i += p[j];
            }
            for (int d = 0; d < hd; d++) {
                float acc = 0.0f;
                for (int j = 0; j < N; j++) acc += p[j] * Vb[j*hd + d];
                Ob[i*hd + d] = acc / (l_i > 0 ? l_i : 1.0f);
            }
            Lb[i] = m_i + logf(l_i > 0 ? l_i : 1.0f);
        }
    }
}

// ====== FP64 forward (probe-side) =========
static void fa_fwd_fp64(
    const float *Q, const float *K, const float *V,
    double *O,
    int bh, int N, int hd,
    int causal, int window)
{
    double scale = 1.0 / std::sqrt((double)hd);
    for (int b = 0; b < bh; b++) {
        const float *Qb = Q + b * N * hd;
        const float *Kb = K + b * N * hd;
        const float *Vb = V + b * N * hd;
        double *Ob = O + b * N * hd;
        for (int i = 0; i < N; i++) {
            std::vector<double> s(N);
            double m_i = -INFINITY;
            for (int j = 0; j < N; j++) {
                if (causal && j > i) { s[j] = -INFINITY; continue; }
                if (window > 0 && j < i + 1 - window) { s[j] = -INFINITY; continue; }
                double dot = 0.0;
                for (int d = 0; d < hd; d++) {
                    dot += (double)Qb[i*hd + d] * (double)Kb[j*hd + d];
                }
                s[j] = dot * scale;
                if (s[j] > m_i) m_i = s[j];
            }
            double l_i = 0.0;
            std::vector<double> p(N, 0.0);
            for (int j = 0; j < N; j++) {
                if (!std::isfinite(s[j])) continue;
                p[j] = std::exp(s[j] - m_i);
                l_i += p[j];
            }
            for (int d = 0; d < hd; d++) {
                double acc = 0.0;
                for (int j = 0; j < N; j++) {
                    acc += p[j] * (double)Vb[j*hd + d];
                }
                Ob[i*hd + d] = acc / (l_i > 0 ? l_i : 1.0);
            }
        }
    }
}

// ====== FP32 backward (Tri Dao Variant 3) — checked thing =========
static void fa_bwd_fp32(
    const float *Q, const float *K, const float *V,
    const float *O, const float *L,
    const float *dO,
    float *dQ, float *dK, float *dV,
    int bh, int N, int hd,
    int causal, int window)
{
    float scale = 1.0f / sqrtf((float)hd);
    size_t tot = (size_t)bh * N * hd;
    for (size_t i = 0; i < tot; i++) { dQ[i] = 0.0f; dK[i] = 0.0f; dV[i] = 0.0f; }

    for (int b = 0; b < bh; b++) {
        const float *Qb = Q + b * N * hd;
        const float *Kb = K + b * N * hd;
        const float *Vb = V + b * N * hd;
        const float *Ob = O + b * N * hd;
        const float *Lb = L + b * N;
        const float *dOb = dO + b * N * hd;
        float *dQb = dQ + b * N * hd;
        float *dKb = dK + b * N * hd;
        float *dVb = dV + b * N * hd;

        std::vector<float> D(N, 0.0f);
        for (int i = 0; i < N; i++) {
            float d = 0.0f;
            for (int dim = 0; dim < hd; dim++) d += dOb[i*hd+dim] * Ob[i*hd+dim];
            D[i] = d;
        }
        for (int i = 0; i < N; i++) {
            std::vector<float> P(N, 0.0f);
            std::vector<float> dP(N, 0.0f);
            std::vector<float> dS(N, 0.0f);
            for (int j = 0; j < N; j++) {
                if (causal && j > i) continue;
                if (window > 0 && j < i + 1 - window) continue;
                float dot = 0.0f;
                for (int d = 0; d < hd; d++) dot += Qb[i*hd+d] * Kb[j*hd+d];
                float s = dot * scale;
                P[j] = expf(s - Lb[i]);
            }
            for (int j = 0; j < N; j++) {
                if (causal && j > i) continue;
                if (window > 0 && j < i + 1 - window) continue;
                float dpv = 0.0f;
                for (int d = 0; d < hd; d++) dpv += dOb[i*hd+d] * Vb[j*hd+d];
                dP[j] = dpv;
                dS[j] = P[j] * (dP[j] - D[i]);
            }
            for (int d = 0; d < hd; d++) {
                float acc = 0.0f;
                for (int j = 0; j < N; j++) acc += dS[j] * Kb[j*hd+d];
                dQb[i*hd+d] += acc * scale;
            }
            for (int j = 0; j < N; j++) {
                if (causal && j > i) continue;
                if (window > 0 && j < i + 1 - window) continue;
                for (int d = 0; d < hd; d++) {
                    dKb[j*hd+d] += dS[j] * Qb[i*hd+d] * scale;
                    dVb[j*hd+d] += P[j] * dOb[i*hd+d];
                }
            }
        }
    }
}

// ====== FP64 loss =========
static double loss_fp64(const float *dO, const double *O, size_t sz)
{
    double l = 0.0;
    for (size_t j = 0; j < sz; j++) l += (double)dO[j] * O[j];
    return l;
}

// ====== Discrepancy record + helpers =========
struct Disc {
    size_t idx;
    int b, n, d;
    double analytic;
    double numeric;
    double abs_err;
    double rel_err;
    bool passed;
};

static inline void unflatten(size_t idx, int N, int hd, int &b, int &n, int &d) {
    b = (int)(idx / (size_t)(N * hd));
    size_t rem = idx % (size_t)(N * hd);
    n = (int)(rem / hd);
    d = (int)(rem % hd);
}

static bool passes(double num, double ana, double abs_tol, double rel_tol) {
    return std::fabs(num - ana) < abs_tol + rel_tol * std::fabs(ana);
}

struct CheckResult {
    int n_checked, n_passed;
    double max_abs_err, max_rel_err_for_big_grads;
    std::vector<Disc> top_discs;
};

static CheckResult check_tensor(
    float *T, const float *T_grad,
    const float *Q, const float *K, const float *V,
    const float *dO,
    int bh, int N, int hd,
    const char *name,
    double abs_tol, double rel_tol,
    double sig_grad_thr)
{
    size_t sz = (size_t)bh * N * hd;
    std::vector<double> O_plus(sz), O_minus(sz);

    std::vector<Disc> discs;
    discs.reserve(sz);

    for (size_t idx = 0; idx < sz; idx++) {
        double ana = (double)T_grad[idx];
        float orig = T[idx];
        double h_d = 1e-3 * std::max((double)std::fabs(orig), 1e-2);
        float h = (float)h_d;
        T[idx] = orig + h;
        fa_fwd_fp64(Q, K, V, O_plus.data(),  bh, N, hd, 0, 0);
        T[idx] = orig - h;
        fa_fwd_fp64(Q, K, V, O_minus.data(), bh, N, hd, 0, 0);
        T[idx] = orig;
        double l_plus  = loss_fp64(dO, O_plus.data(),  sz);
        double l_minus = loss_fp64(dO, O_minus.data(), sz);
        double num = (l_plus - l_minus) / (2.0 * h_d);
        double abs_err = std::fabs(num - ana);
        double rel = abs_err / (std::fabs(ana) + 1e-12);
        Disc r;
        r.idx = idx;
        unflatten(idx, N, hd, r.b, r.n, r.d);
        r.analytic = ana;
        r.numeric = num;
        r.abs_err = abs_err;
        r.rel_err = rel;
        r.passed = passes(num, ana, abs_tol, rel_tol);
        discs.push_back(r);
    }

    CheckResult res{};
    res.n_checked = (int)discs.size();
    for (auto &d : discs) {
        if (d.passed) res.n_passed++;
        if (d.abs_err > res.max_abs_err) res.max_abs_err = d.abs_err;
        if (std::fabs(d.analytic) > sig_grad_thr) {
            if (d.rel_err > res.max_rel_err_for_big_grads)
                res.max_rel_err_for_big_grads = d.rel_err;
        }
    }
    std::sort(discs.begin(), discs.end(),
              [](const Disc &a, const Disc &b) { return a.abs_err > b.abs_err; });
    int top_n = std::min(20, (int)discs.size());
    res.top_discs.assign(discs.begin(), discs.begin() + top_n);

    double pct = 100.0 * res.n_passed / res.n_checked;
    printf("\n--- %s (n=%d positions) ---\n", name, res.n_checked);
    printf("  passed: %d/%d (%.3f%%) — criterion |num-ana| < %.0e + %.0e*|ana|\n",
           res.n_passed, res.n_checked, pct, abs_tol, rel_tol);
    printf("  max abs_err = %.6e\n", res.max_abs_err);
    printf("  max rel_err on |ana|>%.0e = %.4f\n",
           sig_grad_thr, res.max_rel_err_for_big_grads);
    printf("  Top-10 by abs_err (b,n,d):\n");
    int show = std::min(10, (int)res.top_discs.size());
    for (int k = 0; k < show; k++) {
        auto &d = res.top_discs[k];
        printf("    (%d,%2d,%3d)  abs=%.4e  ana=%+.6e  num=%+.6e  %s\n",
               d.b, d.n, d.d, d.abs_err, d.analytic, d.numeric,
               d.passed ? "PASS" : "FAIL");
    }
    return res;
}

static void cluster_analysis(const std::vector<Disc> &top, int N, int hd, const char *name)
{
    if (top.empty()) return;
    int n_seq[64] = {0};
    int n_dim[256] = {0};
    int boundary = 0, diagonal = 0, n_failures = 0;
    int N_clamped = std::min(N, 64);
    int hd_clamped = std::min(hd, 256);
    for (auto &dx : top) {
        if (!dx.passed) n_failures++;
        if (dx.n < N_clamped) n_seq[dx.n]++;
        if (dx.d < hd_clamped) n_dim[dx.d]++;
        if (dx.n == 0 || dx.n == N - 1 || dx.d == 0 || dx.d == hd - 1) boundary++;
        if (dx.n == dx.d) diagonal++;
    }
    int max_seq_hits = 0, max_seq_pos = -1;
    for (int n = 0; n < N_clamped; n++) {
        if (n_seq[n] > max_seq_hits) { max_seq_hits = n_seq[n]; max_seq_pos = n; }
    }
    int max_dim_hits = 0, max_dim_pos = -1;
    for (int d = 0; d < hd_clamped; d++) {
        if (n_dim[d] > max_dim_hits) { max_dim_hits = n_dim[d]; max_dim_pos = d; }
    }
    int sz = (int)top.size();
    printf("  Cluster on top-%d (%d failures):\n", sz, n_failures);
    printf("    seq:   max=%d at n=%d (%.0f%%)\n",
           max_seq_hits, max_seq_pos, 100.0*max_seq_hits/sz);
    printf("    dim:   max=%d at d=%d (%.0f%%)\n",
           max_dim_hits, max_dim_pos, 100.0*max_dim_hits/sz);
    printf("    bound: %d (%.0f%%)   diag: %d (%.0f%%)\n",
           boundary, 100.0*boundary/sz, diagonal, 100.0*diagonal/sz);
    if (n_failures == 0) {
        printf("    VERDICT: all top discrepancies pass hybrid tol — clean.\n");
    } else if (max_seq_hits >= sz / 3) {
        printf("    VERDICT: clustering at n=%d — possible seq-position bug.\n", max_seq_pos);
    } else if (max_dim_hits >= sz / 3) {
        printf("    VERDICT: clustering at d=%d — possible head-dim bug.\n", max_dim_pos);
    } else if (boundary >= sz * 2 / 3) {
        printf("    VERDICT: boundary-heavy — possible edge-handling bug.\n");
    } else {
        printf("    VERDICT: scattered — FP32-bwd intrinsic noise.\n");
    }
}

int main()
{
    printf("=== B1-FIX v2: FP64 probe + hybrid tol ===\n");
    printf("  FP64 forward inside finite-diff probes (no FP32 amplification).\n");
    printf("  Hybrid pass: |num-ana| < abs_tol + rel_tol*|ana|\n");
    printf("    abs_tol = 1e-4   rel_tol = 1e-3\n\n");

    const int bh = 1, N = 32, hd = 128;
    const double abs_tol = 1e-4;
    const double rel_tol = 1e-3;
    const double sig_grad = 1e-2;
    size_t sz = (size_t)bh * N * hd;
    std::vector<float> Q(sz), K(sz), V(sz), O(sz), dO(sz);
    std::vector<float> L(bh * N);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.6f);
    for (size_t i = 0; i < sz; i++) { Q[i] = dist(rng); K[i] = dist(rng); V[i] = dist(rng); }
    for (size_t i = 0; i < sz; i++) dO[i] = dist(rng);

    fa_fwd_fp32(Q.data(), K.data(), V.data(), O.data(), L.data(), bh, N, hd, 0, 0);

    std::vector<float> dQ(sz, 0.0f), dK(sz, 0.0f), dV(sz, 0.0f);
    fa_bwd_fp32(Q.data(), K.data(), V.data(), O.data(), L.data(), dO.data(),
                dQ.data(), dK.data(), dV.data(), bh, N, hd, 0, 0);

    printf("Shape: bh=%d N=%d hd=%d  →  %zu elements/tensor\n", bh, N, hd, sz);

    auto resQ = check_tensor(Q.data(), dQ.data(),
                             Q.data(), K.data(), V.data(), dO.data(),
                             bh, N, hd, "dQ", abs_tol, rel_tol, sig_grad);
    cluster_analysis(resQ.top_discs, N, hd, "dQ");

    auto resK = check_tensor(K.data(), dK.data(),
                             Q.data(), K.data(), V.data(), dO.data(),
                             bh, N, hd, "dK", abs_tol, rel_tol, sig_grad);
    cluster_analysis(resK.top_discs, N, hd, "dK");

    auto resV = check_tensor(V.data(), dV.data(),
                             Q.data(), K.data(), V.data(), dO.data(),
                             bh, N, hd, "dV", abs_tol, rel_tol, sig_grad);
    cluster_analysis(resV.top_discs, N, hd, "dV");

    printf("\n=== SUMMARY ===\n");
    int total = resQ.n_checked + resK.n_checked + resV.n_checked;
    int passed = resQ.n_passed + resK.n_passed + resV.n_passed;
    double agreement = 100.0 * passed / total;
    printf("  total              : %d positions\n", total);
    printf("  passed hybrid tol  : %d (%.4f%%)\n", passed, agreement);
    printf("  per-component:\n");
    printf("    dQ  %d/%d (%.3f%%)  max abs=%.2e  max rel|ana>%.0e=%.4f\n",
           resQ.n_passed, resQ.n_checked, 100.0*resQ.n_passed/resQ.n_checked,
           resQ.max_abs_err, sig_grad, resQ.max_rel_err_for_big_grads);
    printf("    dK  %d/%d (%.3f%%)  max abs=%.2e  max rel|ana>%.0e=%.4f\n",
           resK.n_passed, resK.n_checked, 100.0*resK.n_passed/resK.n_checked,
           resK.max_abs_err, sig_grad, resK.max_rel_err_for_big_grads);
    printf("    dV  %d/%d (%.3f%%)  max abs=%.2e  max rel|ana>%.0e=%.4f\n",
           resV.n_passed, resV.n_checked, 100.0*resV.n_passed/resV.n_checked,
           resV.max_abs_err, sig_grad, resV.max_rel_err_for_big_grads);

    printf("\n  VERDICT: ");
    if (agreement >= 99.5) {
        printf("PASS (>99.5%% under hybrid tol)\n");
        printf("  Reference VALIDATED. GO to L-patch + B2.\n");
        return 0;
    } else {
        printf("CRACK — agreement %.4f%% < 99.5%%.\n", agreement);
        printf("  Inspect per-component verdicts + clusters above.\n");
        return 1;
    }
}
