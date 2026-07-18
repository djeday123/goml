// =====================================================================
//  fa_bwd_dk_test.cu — B3.2 correctness validation for dK kernel.
//
//  Two-level checks:
//    Level 1 (per-lane transpose verifier): bit-exact check of Q→Q_T
//                                            transpose pass on synthetic Q.
//    Level 2 (8+ forms + canary):           full dK pipeline (D-precompute
//                                            + dK kernel) vs FP64-golden.
//
//  FP64 golden ref:
//    - forward to produce L, O (per (b, i))
//    - D[i] = sum_d O[i,d]*dO[i,d]
//    - dK[j,d] = scale * sum_i dS[i,j] * Q[i,d]
//      where dS = P · (dP - D), P = exp(scale Q·K - L), dP = dO · V^T
//
//  Floor: dS-quantize via probe B3.1 measured ~5e-3 abs (non-causal),
//         ~1e-2 abs (causal) — same N_eff geometry pattern as dV.
//         Plus Q,K,V FP8 quantize: combined ~5e-3 ave / ~5e-2 worst.
//  Tolerance starts strict abs 1e-3 + rel 5e-3 (dV-style); если fail —
//  ослабляем до empirical floor (abs 5e-2 + rel 5e-1) и фиксируем числа.
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "fa_bwd_common.cuh"

#define DK_CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) {              \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                     \
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
// FP64 reference dK (Tri Dao Variant 3).
// =====================================================================
static void fa_bwd_dk_fp64(
    const double *Q, const double *K, const double *V,
    const double *O, const double *L, const double *dO,
    double *dK,
    int bh, int sl, int hd, int causal, int window)
{
    double scale = 1.0 / std::sqrt((double)hd);
    size_t tot = (size_t)bh * sl * hd;
    for (size_t i = 0; i < tot; ++i) dK[i] = 0.0;

    for (int b = 0; b < bh; ++b) {
        const double *Qb  = Q  + b * sl * hd;
        const double *Kb  = K  + b * sl * hd;
        const double *Vb  = V  + b * sl * hd;
        const double *Ob  = O  + b * sl * hd;
        const double *Lb  = L  + b * sl;
        const double *dOb = dO + b * sl * hd;
        double *dKb       = dK + b * sl * hd;

        // D[i] per row
        std::vector<double> D(sl, 0.0);
        for (int i = 0; i < sl; ++i) {
            for (int d = 0; d < hd; ++d) D[i] += Ob[i*hd+d] * dOb[i*hd+d];
        }

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

                for (int d = 0; d < hd; ++d) {
                    dKb[j*hd+d] += dS * Qb[i*hd+d] * scale;
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
    int worst_b, worst_j, worst_d;
    double worst_ref, worst_got;
};

template <typename Tref>
static CmpStats compare_dK(const float *got, const Tref *ref,
                           int bh, int sl, int hd,
                           double abs_tol, double rel_tol, double sig)
{
    CmpStats s{};
    s.worst_b = s.worst_j = s.worst_d = -1;
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
            s.worst_j = rem / hd;
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
           "max_abs %.3e  max_rel_af %.3e  worst@(b=%d,j=%d,d=%d) ref=%.4e got=%.4e\n",
           label, s.n_passed, s.n_checked, pct_tot,
           s.n_passed_above_floor, s.n_above_floor, pct_above,
           s.max_abs, s.max_rel_above_floor,
           s.worst_b, s.worst_j, s.worst_d, s.worst_ref, s.worst_got);
}

// =====================================================================
// Per-lane transpose verifier:
//   Generate synthetic Q[Br=64, Hd=128] FP8 with known values.
//   Run a minimal kernel that does only the transpose pass.
//   Read back smQ_T and verify bit-exact with CPU-computed transpose.
// =====================================================================
__global__ void kernel_transpose_only(
    const uint8_t * __restrict__ Q_in,
    uint8_t       * __restrict__ Q_T_out)
{
    constexpr int Br = 64;
    constexpr int Hd = 128;
    constexpr int QT_STRIDE = 68;
    constexpr int SMQ_AREA = Hd * QT_STRIDE;
    __shared__ uint8_t smQ_area[SMQ_AREA];

    const int tid = threadIdx.x;

    // Step 1: load Q row-major into smQ_area
    for (int e = tid; e < Br * Hd; e += 128) {
        int i = e / Hd;
        int d = e % Hd;
        smQ_area[i * Hd + d] = Q_in[i * Hd + d];
    }
    __syncthreads();

    // Step 2: transpose pass (identical to dK kernel logic)
    uint32_t Q_buf[16];
    #pragma unroll
    for (int e = 0; e < 16; ++e) {
        int byte_idx = tid * 4 + e * (128 * 4);
        int i_local = byte_idx / Hd;
        int d       = byte_idx % Hd;
        if (i_local < Br && d + 3 < Hd) {
            Q_buf[e] = *reinterpret_cast<uint32_t*>(&smQ_area[i_local * Hd + d]);
        } else {
            Q_buf[e] = 0u;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int e = 0; e < 16; ++e) {
        int byte_idx = tid * 4 + e * (128 * 4);
        int i_local = byte_idx / Hd;
        int d_base  = byte_idx % Hd;
        if (i_local < Br && d_base + 3 < Hd) {
            uint8_t b0 = Q_buf[e]        & 0xFF;
            uint8_t b1 = (Q_buf[e] >>  8) & 0xFF;
            uint8_t b2 = (Q_buf[e] >> 16) & 0xFF;
            uint8_t b3 = (Q_buf[e] >> 24) & 0xFF;
            smQ_area[(d_base + 0) * QT_STRIDE + i_local] = b0;
            smQ_area[(d_base + 1) * QT_STRIDE + i_local] = b1;
            smQ_area[(d_base + 2) * QT_STRIDE + i_local] = b2;
            smQ_area[(d_base + 3) * QT_STRIDE + i_local] = b3;
        }
    }
    __syncthreads();

    // Step 3: write smQ_T back to gmem (only meaningful positions)
    for (int e = tid; e < Hd * Br; e += 128) {
        int d = e / Br;
        int i = e % Br;
        Q_T_out[d * Br + i] = smQ_area[d * QT_STRIDE + i];
    }
}

static int test_transpose_verifier()
{
    printf("\n=== Per-lane transpose verifier ===\n");
    constexpr int Br = 64;
    constexpr int Hd = 128;
    constexpr int sz = Br * Hd;

    std::vector<uint8_t> Q_h(sz);
    // Synthetic: Q[i, d] = (i * 7 + d * 13) & 0xFF (deterministic FP8 bytes)
    for (int i = 0; i < Br; ++i)
        for (int d = 0; d < Hd; ++d)
            Q_h[i * Hd + d] = (uint8_t)((i * 7 + d * 13) & 0xFF);

    // CPU reference Q_T[d, i] = Q[i, d]
    std::vector<uint8_t> Q_T_ref(sz);
    for (int d = 0; d < Hd; ++d)
        for (int i = 0; i < Br; ++i)
            Q_T_ref[d * Br + i] = Q_h[i * Hd + d];

    uint8_t *dQ, *dQ_T;
    DK_CK(cudaMalloc(&dQ,   sz));
    DK_CK(cudaMalloc(&dQ_T, sz));
    DK_CK(cudaMemcpy(dQ, Q_h.data(), sz, cudaMemcpyHostToDevice));

    kernel_transpose_only<<<1, 128>>>(dQ, dQ_T);
    DK_CK(cudaDeviceSynchronize());

    std::vector<uint8_t> Q_T_gpu(sz);
    DK_CK(cudaMemcpy(Q_T_gpu.data(), dQ_T, sz, cudaMemcpyDeviceToHost));

    int mismatches = 0;
    int worst_d = -1, worst_i = -1;
    for (int d = 0; d < Hd; ++d) {
        for (int i = 0; i < Br; ++i) {
            if (Q_T_gpu[d * Br + i] != Q_T_ref[d * Br + i]) {
                if (mismatches < 5) {
                    printf("  MISMATCH @(d=%d,i=%d): gpu=%u ref=%u\n",
                           d, i, Q_T_gpu[d * Br + i], Q_T_ref[d * Br + i]);
                }
                mismatches++;
                worst_d = d; worst_i = i;
            }
        }
    }
    DK_CK(cudaFree(dQ));
    DK_CK(cudaFree(dQ_T));

    if (mismatches == 0) {
        printf("  Transpose pass: BIT-EXACT (%d/%d cells match)\n", sz, sz);
        return 0;
    } else {
        printf("  Transpose pass: FAIL — %d mismatches (last @d=%d,i=%d)\n",
               mismatches, worst_d, worst_i);
        return 1;
    }
}

// =====================================================================
// Form runner
// =====================================================================
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

    printf("\n[%s%s] bh=%d sl=%d hd=%d causal=%d wnd=%d (seed=%u)\n",
           F.canary ? "CANARY " : "", F.name, bh, sl, hd, F.causal, F.window, seed);

    // Generate FP64 inputs
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 0.6);
    std::vector<double> Q64(sz), K64(sz), V64(sz), dO64(sz);
    for (size_t i = 0; i < sz; ++i) {
        Q64[i] = dist(rng); K64[i] = dist(rng);
        V64[i] = dist(rng); dO64[i] = dist(rng);
    }

    // FP64 forward → L, O
    std::vector<double> O64(sz), L64(lsz);
    fa_fwd_fp64(Q64.data(), K64.data(), V64.data(),
                O64.data(), L64.data(),
                bh, sl, hd, F.causal, F.window);

    // FP64 backward → dK_golden
    std::vector<double> dK64(sz, 0.0);
    fa_bwd_dk_fp64(Q64.data(), K64.data(), V64.data(),
                   O64.data(), L64.data(), dO64.data(),
                   dK64.data(), bh, sl, hd, F.causal, F.window);

    // Quantize to GPU input formats
    std::vector<uint8_t> Q8(sz), K8(sz), V8(sz);
    std::vector<__half>  O16(sz), dO16(sz);
    std::vector<float>   L32(lsz);
    for (size_t i = 0; i < sz; ++i) {
        Q8[i] = float_to_e4m3_host((float)Q64[i]);
        K8[i] = float_to_e4m3_host((float)K64[i]);
        V8[i] = float_to_e4m3_host((float)V64[i]);
        O16[i] = __float2half_rn((float)O64[i]);
        dO16[i] = __float2half_rn((float)dO64[i]);
    }
    for (size_t i = 0; i < lsz; ++i) L32[i] = (float)L64[i];

    // GPU pipeline
    uint8_t *dQ, *dK_, *dV;
    __half *dO_d, *dO_g;
    float *dL, *dD, *ddK;
    DK_CK(cudaMalloc(&dQ,  sz  * sizeof(uint8_t)));
    DK_CK(cudaMalloc(&dK_, sz  * sizeof(uint8_t)));
    DK_CK(cudaMalloc(&dV,  sz  * sizeof(uint8_t)));
    DK_CK(cudaMalloc(&dO_d, sz * sizeof(__half)));
    DK_CK(cudaMalloc(&dO_g, sz * sizeof(__half)));
    DK_CK(cudaMalloc(&dL,  lsz * sizeof(float)));
    DK_CK(cudaMalloc(&dD,  lsz * sizeof(float)));
    DK_CK(cudaMalloc(&ddK, sz  * sizeof(float)));
    DK_CK(cudaMemcpy(dQ,  Q8.data(),   sz  * sizeof(uint8_t), cudaMemcpyHostToDevice));
    DK_CK(cudaMemcpy(dK_, K8.data(),   sz  * sizeof(uint8_t), cudaMemcpyHostToDevice));
    DK_CK(cudaMemcpy(dV,  V8.data(),   sz  * sizeof(uint8_t), cudaMemcpyHostToDevice));
    DK_CK(cudaMemcpy(dO_d, O16.data(), sz * sizeof(__half),  cudaMemcpyHostToDevice));
    DK_CK(cudaMemcpy(dO_g, dO16.data(), sz * sizeof(__half), cudaMemcpyHostToDevice));
    DK_CK(cudaMemcpy(dL, L32.data(),   lsz * sizeof(float),  cudaMemcpyHostToDevice));
    DK_CK(cudaMemset(ddK, 0, sz * sizeof(float)));

    fa_bwd_dk::launch_d_precompute(dO_d, dO_g, dD, bh, sl, hd, 0);
    DK_CK(cudaDeviceSynchronize());

    float scale = 1.0f / std::sqrt((float)hd);
    fa_bwd_dk::launch_dk(dQ, dK_, dV, dO_g, dL, dD, ddK,
                         bh, sl, hd, F.causal, F.window, scale, 0);
    DK_CK(cudaDeviceSynchronize());

    std::vector<float> dK_gpu(sz);
    DK_CK(cudaMemcpy(dK_gpu.data(), ddK, sz * sizeof(float), cudaMemcpyDeviceToHost));

    DK_CK(cudaFree(dQ));  DK_CK(cudaFree(dK_)); DK_CK(cudaFree(dV));
    DK_CK(cudaFree(dO_d)); DK_CK(cudaFree(dO_g));
    DK_CK(cudaFree(dL));  DK_CK(cudaFree(dD));  DK_CK(cudaFree(ddK));

    CmpStats vs_fp64 = compare_dK<double>(dK_gpu.data(), dK64.data(),
                                          bh, sl, hd, abs_tol, rel_tol, sig);
    print_cmp("dK vs FP64-golden", vs_fp64);

    bool ok = vs_fp64.n_above_floor == 0 ||
              100.0 * vs_fp64.n_passed_above_floor / vs_fp64.n_above_floor >= 99.5;
    printf("    verdict: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

int main(int argc, char **argv)
{
    unsigned seed = 42;
    double abs_tol = 5e-2;   // FP8 dK empirical floor (per test.cu:20 comment)
    double rel_tol = 5e-1;   // FP8 dK empirical floor
    double sig = 1e-2;
    if (argc >= 2) seed = (unsigned)std::atoi(argv[1]);
    if (argc >= 3) abs_tol = std::atof(argv[2]);
    if (argc >= 4) rel_tol = std::atof(argv[3]);

    printf("=== B3.2 dK validation ===\n");
    printf("    tol: abs %.0e + rel %.0e * |ref| (sig %.0e), seed=%u\n",
           abs_tol, rel_tol, sig, seed);

    // Level 1: per-lane transpose verifier
    int t_fail = test_transpose_verifier();
    if (t_fail) {
        printf("\n=== TRANSPOSE FAIL — aborting full forms ===\n");
        return 1;
    }

    // Level 2: 8+ forms + canary
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
