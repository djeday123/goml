// B1-FIX-EXTRA: FP64 backward + FP64 finite-diff + tight tol.
//
// Цель — золотой эталон без FP32-floor'а:
//   - probe-forward: FP64 (как было)
//   - **checked thing**: FP64 backward (NEW — механический port FP32→double,
//     математика идентична)
//   - tolerance: abs_tol=1e-10, rel_tol=1e-6 (на 6 порядков жёстче FP32)
//   - finite-diff h = 1e-5 ≈ ε^(1/3) для double — оптимум central diff
//
// Сравнивая результат с FP32-checker'ом, видим:
//   - 0.1% мутация, что НЕ ловилась в FP32 (на noise-floor) — теперь ловится
//   - граница чувствительности уезжает к 1e-6 systematic error
//   - если формулы не математически верны (не "в пределах FP32-шума"),
//     ожидание ~1e-10 не наступит — это диагностика
//
// Артефакт: этот FP64 backward = золотой эталон для B4 (GPU dV/dK/dQ ↔ FP64).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

// ====== FP64 forward (probe-side) =========
static void fa_fwd_fp64(
    const double *Q, const double *K, const double *V,
    double *O, double *L_out,
    int bh, int N, int hd,
    int causal, int window)
{
    double scale = 1.0 / std::sqrt((double)hd);
    for (int b = 0; b < bh; b++) {
        const double *Qb = Q + b * N * hd;
        const double *Kb = K + b * N * hd;
        const double *Vb = V + b * N * hd;
        double *Ob = O + b * N * hd;
        double *Lb = L_out + b * N;
        for (int i = 0; i < N; i++) {
            std::vector<double> s(N);
            double m_i = -INFINITY;
            for (int j = 0; j < N; j++) {
                if (causal && j > i) { s[j] = -INFINITY; continue; }
                if (window > 0 && j < i + 1 - window) { s[j] = -INFINITY; continue; }
                double dot = 0.0;
                for (int d = 0; d < hd; d++) dot += Qb[i*hd + d] * Kb[j*hd + d];
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
                for (int j = 0; j < N; j++) acc += p[j] * Vb[j*hd + d];
                Ob[i*hd + d] = acc / (l_i > 0 ? l_i : 1.0);
            }
            Lb[i] = m_i + std::log(l_i > 0 ? l_i : 1.0);
        }
    }
}

// ====== FP64 backward (golden) — mechanical port of Tri Dao Variant 3 =========
// Это и есть permanent артефакт. Математика идентична FP32 версии, только тип.
static void fa_bwd_fp64(
    const double *Q, const double *K, const double *V,
    const double *O, const double *L,
    const double *dO,
    double *dQ, double *dK, double *dV,
    int bh, int N, int hd,
    int causal, int window)
{
    double scale = 1.0 / std::sqrt((double)hd);
    size_t tot = (size_t)bh * N * hd;
    for (size_t i = 0; i < tot; i++) { dQ[i] = 0.0; dK[i] = 0.0; dV[i] = 0.0; }

    for (int b = 0; b < bh; b++) {
        const double *Qb = Q + b * N * hd;
        const double *Kb = K + b * N * hd;
        const double *Vb = V + b * N * hd;
        const double *Ob = O + b * N * hd;
        const double *Lb = L + b * N;
        const double *dOb = dO + b * N * hd;
        double *dQb = dQ + b * N * hd;
        double *dKb = dK + b * N * hd;
        double *dVb = dV + b * N * hd;

        std::vector<double> D(N, 0.0);
        for (int i = 0; i < N; i++) {
            double d = 0.0;
            for (int dim = 0; dim < hd; dim++) d += dOb[i*hd+dim] * Ob[i*hd+dim];
            D[i] = d;
        }
        for (int i = 0; i < N; i++) {
            std::vector<double> P(N, 0.0);
            std::vector<double> dP(N, 0.0);
            std::vector<double> dS(N, 0.0);
            for (int j = 0; j < N; j++) {
                if (causal && j > i) continue;
                if (window > 0 && j < i + 1 - window) continue;
                double dot = 0.0;
                for (int d = 0; d < hd; d++) dot += Qb[i*hd+d] * Kb[j*hd+d];
                double s = dot * scale;
                P[j] = std::exp(s - Lb[i]);
            }
            for (int j = 0; j < N; j++) {
                if (causal && j > i) continue;
                if (window > 0 && j < i + 1 - window) continue;
                double dpv = 0.0;
                for (int d = 0; d < hd; d++) dpv += dOb[i*hd+d] * Vb[j*hd+d];
                dP[j] = dpv;
                dS[j] = P[j] * (dP[j] - D[i]);
            }
            for (int d = 0; d < hd; d++) {
                double acc = 0.0;
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

// ====== Mutations on FP64 backward output =========
static void apply_mutation_fp64(const std::string &m,
                                double *dQ, double *dK, double *dV,
                                int hd, size_t sz)
{
    if (m == "none") {
        printf("  mutation: NONE (clean FP64 backward)\n");
        return;
    }
    double scale = 1.0 / std::sqrt((double)hd);
    if (m == "dK_drop_scale") {
        for (size_t i = 0; i < sz; i++) dK[i] /= scale;
        printf("  mutation: dK /= scale\n");
    } else if (m == "dV_x_1.01") {
        for (size_t i = 0; i < sz; i++) dV[i] *= 1.01;
        printf("  mutation: dV *= 1.01 (1%%)\n");
    } else if (m == "dQ_sign") {
        for (size_t i = 0; i < sz; i++) dQ[i] = -dQ[i];
        printf("  mutation: dQ sign-flipped\n");
    } else if (m == "dQ_x_1.005") {
        for (size_t i = 0; i < sz; i++) dQ[i] *= 1.005;
        printf("  mutation: dQ *= 1.005 (0.5%%)\n");
    } else if (m == "dQ_x_1.002") {
        for (size_t i = 0; i < sz; i++) dQ[i] *= 1.002;
        printf("  mutation: dQ *= 1.002 (0.2%%)\n");
    } else if (m == "dQ_x_1.001") {
        for (size_t i = 0; i < sz; i++) dQ[i] *= 1.001;
        printf("  mutation: dQ *= 1.001 (0.1%%)\n");
    } else if (m == "dQ_x_1.0001") {
        for (size_t i = 0; i < sz; i++) dQ[i] *= 1.0001;
        printf("  mutation: dQ *= 1.0001 (0.01%%)\n");
    } else if (m == "dQ_x_1.00001") {
        for (size_t i = 0; i < sz; i++) dQ[i] *= 1.00001;
        printf("  mutation: dQ *= 1.00001 (0.001%%)\n");
    } else if (m == "dQ_x_1.000001") {
        for (size_t i = 0; i < sz; i++) dQ[i] *= 1.000001;
        printf("  mutation: dQ *= 1.000001 (0.0001%%)\n");
    } else {
        printf("  WARN: unknown mutation '%s' — using NONE\n", m.c_str());
    }
}

// ====== FP64 loss accumulator =========
static double loss_fp64(const double *dO, const double *O, size_t sz)
{
    double l = 0.0;
    for (size_t j = 0; j < sz; j++) l += dO[j] * O[j];
    return l;
}

// ====== Discrepancy struct =========
struct Disc {
    size_t idx;
    int b, n, d;
    double analytic, numeric, abs_err, rel_err;
    bool passed;
};

static inline void unflatten(size_t idx, int N, int hd, int &b, int &n, int &dd) {
    b = (int)(idx / (size_t)(N * hd));
    size_t rem = idx % (size_t)(N * hd);
    n = (int)(rem / hd);
    dd = (int)(rem % hd);
}

static bool passes(double num, double ana, double abs_tol, double rel_tol) {
    return std::fabs(num - ana) < abs_tol + rel_tol * std::fabs(ana);
}

struct CheckResult {
    int n_checked, n_passed;
    int n_above_floor, n_passed_above_floor;
    int n_below_floor, n_passed_below_floor;
    double max_abs_err, max_rel_err_above_floor;
    std::vector<Disc> top_discs;
};

// ====== Tight FP64 checker =========
static CheckResult check_tensor_fp64(
    double *T, const double *T_grad,
    const double *Q, const double *K, const double *V,
    const double *dO,
    int bh, int N, int hd,
    const char *name,
    double abs_tol, double rel_tol,
    double sig_grad_thr)
{
    size_t sz = (size_t)bh * N * hd;
    std::vector<double> O_plus(sz), L_plus(bh * N);
    std::vector<double> O_minus(sz), L_minus(bh * N);
    std::vector<Disc> discs;
    discs.reserve(sz);

    for (size_t idx = 0; idx < sz; idx++) {
        double ana = T_grad[idx];
        double orig = T[idx];
        // h = ε^(1/3) * max(|x|, 1) ≈ 1e-5 для double
        double h = 1e-5 * std::max(std::fabs(orig), 1.0);
        T[idx] = orig + h;
        fa_fwd_fp64(Q, K, V, O_plus.data(),  L_plus.data(),  bh, N, hd, 0, 0);
        T[idx] = orig - h;
        fa_fwd_fp64(Q, K, V, O_minus.data(), L_minus.data(), bh, N, hd, 0, 0);
        T[idx] = orig;
        double l_plus  = loss_fp64(dO, O_plus.data(),  sz);
        double l_minus = loss_fp64(dO, O_minus.data(), sz);
        double num = (l_plus - l_minus) / (2.0 * h);
        double abs_err = std::fabs(num - ana);
        double rel = abs_err / (std::fabs(ana) + 1e-30);
        Disc r{};
        r.idx = idx;
        unflatten(idx, N, hd, r.b, r.n, r.d);
        r.analytic = ana;
        r.numeric  = num;
        r.abs_err  = abs_err;
        r.rel_err  = rel;
        r.passed   = passes(num, ana, abs_tol, rel_tol);
        discs.push_back(r);
    }
    CheckResult res{};
    res.n_checked = (int)discs.size();
    for (auto &d : discs) {
        if (d.passed) res.n_passed++;
        if (d.abs_err > res.max_abs_err) res.max_abs_err = d.abs_err;
        bool above = std::fabs(d.analytic) > sig_grad_thr;
        if (above) {
            res.n_above_floor++;
            if (d.passed) res.n_passed_above_floor++;
            if (d.rel_err > res.max_rel_err_above_floor)
                res.max_rel_err_above_floor = d.rel_err;
        } else {
            res.n_below_floor++;
            if (d.passed) res.n_passed_below_floor++;
        }
    }
    std::sort(discs.begin(), discs.end(),
              [](const Disc &a, const Disc &b) { return a.abs_err > b.abs_err; });
    int top_n = std::min(10, (int)discs.size());
    res.top_discs.assign(discs.begin(), discs.begin() + top_n);

    double pct_total = 100.0 * res.n_passed / res.n_checked;
    double pct_above = res.n_above_floor ? 100.0 * res.n_passed_above_floor / res.n_above_floor : 0.0;
    printf("\n--- %s (n=%d, criterion |num-ana| < %.0e + %.0e*|ana|) ---\n",
           name, res.n_checked, abs_tol, rel_tol);
    printf("  total                : %d/%d (%.4f%%)\n",
           res.n_passed, res.n_checked, pct_total);
    printf("  above-floor (|ana|>%.0e) : %d/%d (%.4f%%)\n",
           sig_grad_thr, res.n_passed_above_floor, res.n_above_floor, pct_above);
    printf("  max abs_err = %.4e   max rel_err above-floor = %.4e\n",
           res.max_abs_err, res.max_rel_err_above_floor);
    return res;
}

// ====== Dump =========
static void dump_binary_fp64(const char *path,
                             int bh, int N, int hd,
                             const double *Q, const double *K, const double *V,
                             const double *dO,
                             const double *dQ, const double *dK, const double *dV,
                             size_t sz)
{
    FILE *f = fopen(path, "wb");
    if (!f) { printf("dump: cannot open %s\n", path); return; }
    int32_t hdr[3] = { bh, N, hd };
    fwrite(hdr, sizeof(int32_t), 3, f);
    fwrite(Q,  sizeof(double), sz, f);
    fwrite(K,  sizeof(double), sz, f);
    fwrite(V,  sizeof(double), sz, f);
    fwrite(dO, sizeof(double), sz, f);
    fwrite(dQ, sizeof(double), sz, f);
    fwrite(dK, sizeof(double), sz, f);
    fwrite(dV, sizeof(double), sz, f);
    fclose(f);
    printf("dump: wrote %s  (FP64, bh=%d N=%d hd=%d, 7 × %zu doubles)\n",
           path, bh, N, hd, sz);
}

int main(int argc, char **argv)
{
    unsigned seed = 42;
    std::string mutation = "none";
    std::string dump_path;
    if (argc >= 2) seed = (unsigned)std::atoi(argv[1]);
    if (argc >= 3) mutation = argv[2];
    if (argc >= 4) dump_path = argv[3];

    printf("=== B1-FIX-EXTRA FP64 strict check ===\n");
    printf("  seed=%u  mutation=%s  dump=%s\n", seed, mutation.c_str(),
           dump_path.empty() ? "no" : dump_path.c_str());
    printf("  Tolerance: abs=1e-10  rel=1e-6  finite-diff h=1e-5 * max(|x|,1)\n\n");

    const int bh = 1, N = 32, hd = 128;
    const double abs_tol = 1e-10;
    const double rel_tol = 1e-6;
    const double sig_grad = 1e-2;
    size_t sz = (size_t)bh * N * hd;

    std::vector<double> Q(sz), K(sz), V(sz), dO(sz), O(sz);
    std::vector<double> L(bh * N);
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 0.6);
    for (size_t i = 0; i < sz; i++) { Q[i] = dist(rng); K[i] = dist(rng); V[i] = dist(rng); }
    for (size_t i = 0; i < sz; i++) dO[i] = dist(rng);

    fa_fwd_fp64(Q.data(), K.data(), V.data(), O.data(), L.data(), bh, N, hd, 0, 0);

    std::vector<double> dQ(sz, 0.0), dK(sz, 0.0), dV(sz, 0.0);
    fa_bwd_fp64(Q.data(), K.data(), V.data(), O.data(), L.data(), dO.data(),
                dQ.data(), dK.data(), dV.data(), bh, N, hd, 0, 0);
    apply_mutation_fp64(mutation, dQ.data(), dK.data(), dV.data(), hd, sz);

    if (!dump_path.empty()) {
        dump_binary_fp64(dump_path.c_str(), bh, N, hd,
                         Q.data(), K.data(), V.data(), dO.data(),
                         dQ.data(), dK.data(), dV.data(), sz);
        return 0;
    }

    auto resQ = check_tensor_fp64(Q.data(), dQ.data(),
                                  Q.data(), K.data(), V.data(), dO.data(),
                                  bh, N, hd, "dQ", abs_tol, rel_tol, sig_grad);
    auto resK = check_tensor_fp64(K.data(), dK.data(),
                                  Q.data(), K.data(), V.data(), dO.data(),
                                  bh, N, hd, "dK", abs_tol, rel_tol, sig_grad);
    auto resV = check_tensor_fp64(V.data(), dV.data(),
                                  Q.data(), K.data(), V.data(), dO.data(),
                                  bh, N, hd, "dV", abs_tol, rel_tol, sig_grad);

    printf("\n=== SUMMARY seed=%u mutation=%s ===\n", seed, mutation.c_str());
    int total_above = resQ.n_above_floor + resK.n_above_floor + resV.n_above_floor;
    int passed_above = resQ.n_passed_above_floor + resK.n_passed_above_floor + resV.n_passed_above_floor;
    double agreement_above = total_above ? 100.0 * passed_above / total_above : 0.0;
    double max_abs = std::max({resQ.max_abs_err, resK.max_abs_err, resV.max_abs_err});
    double max_rel_above = std::max({resQ.max_rel_err_above_floor,
                                     resK.max_rel_err_above_floor,
                                     resV.max_rel_err_above_floor});
    printf("  above-floor: %d/%d (%.4f%%)\n", passed_above, total_above, agreement_above);
    printf("  max abs_err overall   : %.4e\n", max_abs);
    printf("  max rel_err above-floor: %.4e\n", max_rel_above);

    if (mutation == "none") {
        if (agreement_above >= 99.5) {
            printf("  VERDICT: PASS — FP64 backward consistent with FP64 finite-diff.\n");
            return 0;
        } else {
            printf("  VERDICT: FAIL — formula bug exposed by tight FP64 tol.\n");
            return 1;
        }
    } else {
        printf("  VERDICT (mutation '%s'): ", mutation.c_str());
        if (agreement_above < 99.5) {
            printf("DETECTED (above-floor %.4f%% < 99.5%%)\n", agreement_above);
            return 0;
        } else {
            printf("UNDETECTED (still %.4f%%)\n", agreement_above);
            return 2;
        }
    }
}
