// =====================================================================
//  fa_bwd_e2e_test.cu — B5 step 2: END-TO-END correctness of full backward pipeline.
//
//  Pipeline order: D-precompute → dV → dK → dQ (D first; dV/dK/dQ independent after D).
//  Compares ALL THREE gradients (dV, dK, dQ) vs FP64-golden ON SAME INPUTS,
//  catches stitching bugs (shared buffers, scale, layout mismatches) that
//  per-kernel tests miss.
//
//  Tolerance: FP8 floor (abs 5e-2 + rel 5e-1) — same as per-kernel tests.
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "fa_bwd_common.cuh"

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) {                \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                    \
            cudaGetErrorString(e)); std::exit(1); }} while (0)

// =====================================================================
// External launches.
// =====================================================================
namespace fa_bwd_dk {
void launch_d_precompute(const __half *O, const __half *dO, float *D,
                          int bh, int sl, int hd, cudaStream_t stream);
void launch_dk(const uint8_t *Q, const uint8_t *K, const uint8_t *V,
               const __half *dO_g, const float *L, const float *D,
               float *dK,
               int bh, int sl, int hd, int causal, int window,
               float scale, cudaStream_t stream);
}
namespace fa_bwd_dq {
void launch_dq(const uint8_t *Q, const uint8_t *K, const uint8_t *V,
               const __half *dO_g, const float *L, const float *D,
               float *dQ,
               int bh, int sl, int hd, int causal, int window,
               float scale, cudaStream_t stream);
}
namespace fa_bwd_dv_mma_p1 {
void launch(const uint8_t *Q, const uint8_t *K, const __half *dO_g,
            const float *L, float *dV,
            int bh, int sl, int hd, int causal, int window,
            float scale, cudaStream_t stream);
}

// =====================================================================
// FP64 reference forward (computes L and O).
// =====================================================================
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
            std::vector<double> s(sl, -INFINITY);
            double m_i = -INFINITY;
            for (int j = 0; j < sl; ++j) {
                if (causal && j > i) continue;
                if (window > 0 && j < i + 1 - window) continue;
                double dot = 0.0;
                for (int d = 0; d < hd; ++d) dot += Qb[i*hd+d] * Kb[j*hd+d];
                s[j] = dot * scale;
                if (s[j] > m_i) m_i = s[j];
            }
            if (m_i == -INFINITY) m_i = 0.0;
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

// =====================================================================
// FP64 references — compute dV, dK, dQ in ONE pass (same chain).
//   dS = P · (dP - D), P = exp(scale Q·K - L), dP = dO · V^T
//   dV[j, d] = sum_i P[i, j] * dO[i, d]
//   dK[j, d] = scale * sum_i dS[i, j] * Q[i, d]
//   dQ[i, d] = scale * sum_j dS[i, j] * K[j, d]
// =====================================================================
static void fa_bwd_all_fp64(
    const double *Q, const double *K, const double *V,
    const double *O, const double *L, const double *dO,
    double *dV, double *dK, double *dQ,
    int bh, int sl, int hd, int causal, int window)
{
    double scale = 1.0 / std::sqrt((double)hd);
    size_t tot = (size_t)bh * sl * hd;
    for (size_t i = 0; i < tot; ++i) { dV[i] = 0.0; dK[i] = 0.0; dQ[i] = 0.0; }

    for (int b = 0; b < bh; ++b) {
        const double *Qb  = Q  + b * sl * hd;
        const double *Kb  = K  + b * sl * hd;
        const double *Vb  = V  + b * sl * hd;
        const double *Ob  = O  + b * sl * hd;
        const double *Lb  = L  + b * sl;
        const double *dOb = dO + b * sl * hd;
        double *dVb       = dV + b * sl * hd;
        double *dKb       = dK + b * sl * hd;
        double *dQb       = dQ + b * sl * hd;

        // D[i] per row
        std::vector<double> D(sl, 0.0);
        for (int i = 0; i < sl; ++i)
            for (int d = 0; d < hd; ++d) D[i] += Ob[i*hd+d] * dOb[i*hd+d];

        for (int i = 0; i < sl; ++i) {
            for (int j = 0; j < sl; ++j) {
                if (causal && j > i) continue;
                if (window > 0 && j < i + 1 - window) continue;
                double dot_QK = 0.0;
                for (int d = 0; d < hd; ++d) dot_QK += Qb[i*hd+d] * Kb[j*hd+d];
                double s = dot_QK * scale;
                double P = std::exp(s - Lb[i]);

                double dP = 0.0;
                for (int d = 0; d < hd; ++d) dP += dOb[i*hd+d] * Vb[j*hd+d];

                double dS = P * (dP - D[i]);

                // Accumulate all three gradients in one pass
                for (int d = 0; d < hd; ++d) {
                    dVb[j*hd+d] += P  * dOb[i*hd+d];           // dV (no scale)
                    dKb[j*hd+d] += dS * Qb[i*hd+d] * scale;    // dK (with scale)
                    dQb[i*hd+d] += dS * Kb[j*hd+d] * scale;    // dQ (with scale)
                }
            }
        }
    }
}

// =====================================================================
// Comparator
// =====================================================================
struct CmpStats {
    size_t n_checked, n_passed;
    size_t n_above_floor, n_passed_above_floor;
    double max_abs, max_rel_above_floor;
    int worst_b, worst_x, worst_d;
    double worst_ref, worst_got;
};

template <typename Tref>
static CmpStats compare_grad(const float *got, const Tref *ref,
                             int bh, int sl, int hd,
                             double abs_tol, double rel_tol, double sig)
{
    CmpStats s{};
    s.worst_b = s.worst_x = s.worst_d = -1;
    size_t tot = (size_t)bh * sl * hd;
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
            int b_ = (int)(idx / ((size_t)sl * hd));
            int rem = (int)(idx % ((size_t)sl * hd));
            s.worst_b = b_;
            s.worst_x = rem / hd;
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

static void print_cmp(const char *label, const CmpStats &s, const char *xlabel)
{
    double pct_above = s.n_above_floor
        ? 100.0 * (double)s.n_passed_above_floor / (double)s.n_above_floor : 0.0;
    printf("    %-12s  above-floor %zu/%zu (%.3f%%)  max_abs %.3e  max_rel_af %.3e  "
           "worst@(b=%d,%s=%d,d=%d) ref=%.4e got=%.4e\n",
           label, s.n_passed_above_floor, s.n_above_floor, pct_above,
           s.max_abs, s.max_rel_above_floor,
           s.worst_b, xlabel, s.worst_x, s.worst_d, s.worst_ref, s.worst_got);
}

// =====================================================================
// Form definitions (mirror dV/dK/dQ tests).
// =====================================================================
struct Form { int bh, sl, causal, window; const char *name; };

static const Form FORMS[] = {
    {1, 128,  0, 0,    "F1 bh=1 sl=128  caus=0 wnd=0"},
    {1, 128,  1, 0,    "F2 bh=1 sl=128  caus=1 wnd=0"},
    {2, 256,  0, 0,    "F3 bh=2 sl=256  caus=0 wnd=0"},
    {2, 256,  1, 0,    "F4 bh=2 sl=256  caus=1 wnd=0"},
    {4, 384,  0, 0,    "F5 bh=4 sl=384  caus=0 wnd=0"},
    {4, 384,  1, 0,    "F6 bh=4 sl=384  caus=1 wnd=0"},
    {1, 512,  0, 128,  "F7 bh=1 sl=512  caus=0 wnd=128"},
    {1, 512,  1, 128,  "F8 bh=1 sl=512  caus=1 wnd=128"},
    {1, 2048, 0, 0,    "F9 bh=1 sl=2048 caus=0 wnd=0"},
    {1, 2048, 1, 0,    "F10 bh=1 sl=2048 caus=1 wnd=0"},
};
static const Form CANARY = {1, 300, 0, 96, "CANARY sl=300 wnd=96 caus=0"};

// =====================================================================
// E2E form runner: D-precompute → dV → dK → dQ → compare all three.
// =====================================================================
static bool run_form(const Form &F, unsigned seed,
                     double abs_tol, double rel_tol, double sig)
{
    const int bh = F.bh, sl = F.sl, hd = 128;
    const int causal = F.causal, window = F.window;

    printf("[%s] bh=%d sl=%d hd=%d causal=%d wnd=%d (seed=%u)\n",
           F.name, bh, sl, hd, causal, window, seed);

    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 0.6);

    const size_t sz = (size_t)bh * sl * hd;
    const size_t lsz = (size_t)bh * sl;

    // FP64 inputs
    std::vector<double> Q64(sz), K64(sz), V64(sz), dO64(sz);
    std::vector<double> O64(sz), L64(lsz);
    std::vector<double> dV64_gold(sz), dK64_gold(sz), dQ64_gold(sz);
    for (size_t i = 0; i < sz; ++i) {
        Q64[i]  = dist(rng);
        K64[i]  = dist(rng);
        V64[i]  = dist(rng);
        dO64[i] = dist(rng);
    }
    fa_fwd_fp64(Q64.data(), K64.data(), V64.data(),
                O64.data(), L64.data(), bh, sl, hd, causal, window);
    fa_bwd_all_fp64(Q64.data(), K64.data(), V64.data(),
                    O64.data(), L64.data(), dO64.data(),
                    dV64_gold.data(), dK64_gold.data(), dQ64_gold.data(),
                    bh, sl, hd, causal, window);

    // FP8/FP16 inputs for GPU
    std::vector<uint8_t> Q8(sz), K8(sz), V8(sz);
    std::vector<__half>  O16(sz), dO16(sz);
    std::vector<float>   L32(lsz);
    for (size_t i = 0; i < sz; ++i) {
        Q8[i]  = float_to_e4m3_host((float)Q64[i]);
        K8[i]  = float_to_e4m3_host((float)K64[i]);
        V8[i]  = float_to_e4m3_host((float)V64[i]);
        O16[i] = __float2half_rn((float)O64[i]);
        dO16[i]= __float2half_rn((float)dO64[i]);
    }
    for (size_t i = 0; i < lsz; ++i) L32[i] = (float)L64[i];

    uint8_t *dQ_g, *dK_g, *dV_g;
    __half  *dO_O_g, *dO_dO_g;
    float   *dL, *dD, *ddV, *ddK, *ddQ;
    CK(cudaMalloc(&dQ_g, sz));
    CK(cudaMalloc(&dK_g, sz));
    CK(cudaMalloc(&dV_g, sz));
    CK(cudaMalloc(&dO_O_g,  sz * sizeof(__half)));
    CK(cudaMalloc(&dO_dO_g, sz * sizeof(__half)));
    CK(cudaMalloc(&dL,  lsz * sizeof(float)));
    CK(cudaMalloc(&dD,  lsz * sizeof(float)));
    CK(cudaMalloc(&ddV, sz * sizeof(float)));
    CK(cudaMalloc(&ddK, sz * sizeof(float)));
    CK(cudaMalloc(&ddQ, sz * sizeof(float)));
    CK(cudaMemcpy(dQ_g, Q8.data(),    sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK_g, K8.data(),    sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV_g, V8.data(),    sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dO_O_g,  O16.data(),  sz * sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dO_dO_g, dO16.data(), sz * sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dL, L32.data(), lsz * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemset(ddV, 0, sz * sizeof(float)));
    CK(cudaMemset(ddK, 0, sz * sizeof(float)));
    CK(cudaMemset(ddQ, 0, sz * sizeof(float)));

    float scale = 1.0f / std::sqrt((float)hd);

    // ===== PIPELINE: D-precompute → dV → dK → dQ =====
    fa_bwd_dk::launch_d_precompute(dO_O_g, dO_dO_g, dD, bh, sl, hd, 0);
    fa_bwd_dv_mma_p1::launch(dQ_g, dK_g, dO_dO_g, dL, ddV,
                              bh, sl, hd, causal, window, scale, 0);
    fa_bwd_dk::launch_dk(dQ_g, dK_g, dV_g, dO_dO_g, dL, dD, ddK,
                          bh, sl, hd, causal, window, scale, 0);
    fa_bwd_dq::launch_dq(dQ_g, dK_g, dV_g, dO_dO_g, dL, dD, ddQ,
                          bh, sl, hd, causal, window, scale, 0);
    CK(cudaDeviceSynchronize());

    std::vector<float> dV_got(sz), dK_got(sz), dQ_got(sz);
    CK(cudaMemcpy(dV_got.data(), ddV, sz * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dK_got.data(), ddK, sz * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dQ_got.data(), ddQ, sz * sizeof(float), cudaMemcpyDeviceToHost));

    CmpStats sV = compare_grad(dV_got.data(), dV64_gold.data(), bh, sl, hd, abs_tol, rel_tol, sig);
    CmpStats sK = compare_grad(dK_got.data(), dK64_gold.data(), bh, sl, hd, abs_tol, rel_tol, sig);
    CmpStats sQ = compare_grad(dQ_got.data(), dQ64_gold.data(), bh, sl, hd, abs_tol, rel_tol, sig);

    print_cmp("dV", sV, "j");
    print_cmp("dK", sK, "j");
    print_cmp("dQ", sQ, "i");

    auto ok_pct = [](const CmpStats &s) {
        return s.n_above_floor == 0 ||
               100.0 * s.n_passed_above_floor / s.n_above_floor >= 99.5;
    };
    bool ok = ok_pct(sV) && ok_pct(sK) && ok_pct(sQ);
    printf("    verdict: %s\n\n", ok ? "PASS (all three grads)" : "FAIL");

    cudaFree(dQ_g); cudaFree(dK_g); cudaFree(dV_g);
    cudaFree(dO_O_g); cudaFree(dO_dO_g);
    cudaFree(dL); cudaFree(dD);
    cudaFree(ddV); cudaFree(ddK); cudaFree(ddQ);
    return ok;
}

int main(int argc, char **argv)
{
    unsigned seed = 42;
    double abs_tol = 5e-2;   // FP8 floor (lesson: must match floor)
    double rel_tol = 5e-1;
    double sig = 1e-2;
    if (argc >= 2) seed = (unsigned)std::atoi(argv[1]);
    if (argc >= 3) abs_tol = std::atof(argv[2]);
    if (argc >= 4) rel_tol = std::atof(argv[3]);

    printf("=== B5 step 2: end-to-end backward correctness ===\n");
    printf("    pipeline: D-precompute → dV → dK → dQ (D first)\n");
    printf("    tol: abs %.0e + rel %.0e * |ref| (sig %.0e), seed=%u\n\n",
           abs_tol, rel_tol, sig, seed);

    int n_forms = sizeof(FORMS) / sizeof(FORMS[0]);
    int n_pass = 0;
    for (int k = 0; k < n_forms; ++k) {
        if (run_form(FORMS[k], seed, abs_tol, rel_tol, sig)) n_pass++;
    }
    if (run_form(CANARY, seed, abs_tol, rel_tol, sig)) n_pass++;

    printf("=== SUMMARY ===\n");
    printf("    forms passed: %d / %d (all three grads simultaneously)\n",
           n_pass, n_forms + 1);
    return n_pass == n_forms + 1 ? 0 : 1;
}
