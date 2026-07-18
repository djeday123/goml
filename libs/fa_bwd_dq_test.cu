// =====================================================================
//  fa_bwd_dq_test.cu — B4.1 correctness scaffold for dQ kernel.
//
//  Mirror of fa_bwd_dk_test.cu adapted for dQ:
//    dQ[i, d] = scale * sum_j dS[i, j] * K[j, d]
//  where dS = P * (dP - D), P = exp(scale Q·K - L), dP = dO · V^T.
//
//  Inner chain (s → P → dP → dS) IDENTICAL to dK's CPU reference.
//  Only the final accumulator differs:
//    dK was:  dKb[j*hd+d] += dS * Qb[i*hd+d] * scale  (own K-tile, sum over i)
//    dQ is:   dQb[i*hd+d] += dS * Kb[j*hd+d] * scale  (own Q-tile, sum over j)
//
//  Build modes:
//    SANITY mode: launch_dq stubbed (returns zeros), test prints CPU golden,
//                 verifies CPU pipeline runs sensibly (no NaN, mask correct).
//    KERNEL mode: real launch_dq from fa_bwd_dq.cu (added in B4.2).
//
//  Tolerance defaults: FP8 dK floor (abs 5e-2 + rel 5e-1) — lesson from dK
//                      (default-strict 1e-3 incorrectly failed correct kernel).
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

#define DQ_CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) {              \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                     \
            cudaGetErrorString(e)); std::exit(1); }} while (0)

// =====================================================================
// External declarations.
//   D-precompute reused from fa_bwd_dk.cu namespace.
//   launch_dq from fa_bwd_dq.cu (or stub) — same signature as launch_dk.
// =====================================================================
namespace fa_bwd_dk {
void launch_d_precompute(const __half *O, const __half *dO, float *D,
                          int bh, int sl, int hd, cudaStream_t stream);
}

namespace fa_bwd_dq {
void launch_dq(const uint8_t *Q, const uint8_t *K, const uint8_t *V,
               const __half *dO_g, const float *L, const float *D,
               float *dQ,
               int bh, int sl, int hd, int causal, int window,
               float scale, cudaStream_t stream);
}

// =====================================================================
// FP64 reference forward (same as dK test — computes L and O).
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
// FP64 reference dQ (Tri Dao Variant 3).
//   Mirror of dK reference. Inner chain s → P → dP → dS identical.
//   Difference: final accumulator on Q-axis (own row i), summed over K-axis (j).
// =====================================================================
static void fa_bwd_dq_fp64(
    const double *Q, const double *K, const double *V,
    const double *O, const double *L, const double *dO,
    double *dQ,
    int bh, int sl, int hd, int causal, int window)
{
    double scale = 1.0 / std::sqrt((double)hd);
    size_t tot = (size_t)bh * sl * hd;
    for (size_t i = 0; i < tot; ++i) dQ[i] = 0.0;

    for (int b = 0; b < bh; ++b) {
        const double *Qb  = Q  + b * sl * hd;
        const double *Kb  = K  + b * sl * hd;
        const double *Vb  = V  + b * sl * hd;
        const double *Ob  = O  + b * sl * hd;
        const double *Lb  = L  + b * sl;
        const double *dOb = dO + b * sl * hd;
        double *dQb       = dQ + b * sl * hd;

        // D[i] per row (reused from forward output O and input dO)
        std::vector<double> D(sl, 0.0);
        for (int i = 0; i < sl; ++i) {
            for (int d = 0; d < hd; ++d) D[i] += Ob[i*hd+d] * dOb[i*hd+d];
        }

        for (int i = 0; i < sl; ++i) {
            for (int j = 0; j < sl; ++j) {
                if (causal && j > i) continue;
                if (window > 0 && j < i + 1 - window) continue;

                // Inner chain identical to dK
                double dot_QK = 0.0;
                for (int d = 0; d < hd; ++d) dot_QK += Qb[i*hd+d] * Kb[j*hd+d];
                double s = dot_QK * scale;
                double P = std::exp(s - Lb[i]);

                double dP = 0.0;
                for (int d = 0; d < hd; ++d) dP += dOb[i*hd+d] * Vb[j*hd+d];

                double dS = P * (dP - D[i]);

                // Final accumulator — DIFFERS from dK.
                // dQ owns row i (Q-tile), accumulates contribution from all j (K columns).
                for (int d = 0; d < hd; ++d) {
                    dQb[i*hd+d] += dS * Kb[j*hd+d] * scale;
                }
            }
        }
    }
}

// =====================================================================
// Comparator: dQ has shape [bh][sl=i_axis][hd=d_axis].
//   Index decoding:  b = idx / (sl*hd);  rem = idx % (sl*hd);  i = rem/hd; d = rem%hd.
//   Worst-position reported as (b, i, d) — i is Q-row axis (NOT dK's j!).
// =====================================================================
struct CmpStats {
    size_t n_checked, n_passed;
    size_t n_above_floor, n_passed_above_floor;
    double max_abs, max_rel_above_floor;
    int worst_b, worst_i, worst_d;
    double worst_ref, worst_got;
};

template <typename Tref>
static CmpStats compare_dQ(const float *got, const Tref *ref,
                           int bh, int sl, int hd,
                           double abs_tol, double rel_tol, double sig)
{
    CmpStats s{};
    s.worst_b = s.worst_i = s.worst_d = -1;
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
            s.worst_i = rem / hd;   // Q-row axis (dQ: own row i)
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

// =====================================================================
// Per-lane K_T transpose verifier (mirror dK Q_T verifier).
//   Isolated kernel that only runs Phase 1.5 (smK → smK_T transpose).
//   Compares smK_T bytes against CPU-computed transpose. Bit-exact (no FP8 noise).
//   Catches bank-layout / scatter-write bugs separately from full pipeline.
// =====================================================================
__global__ void kernel_transpose_only_dq(
    const uint8_t * __restrict__ K_in,
    uint8_t       * __restrict__ K_T_out)
{
    constexpr int Bc = 64;
    constexpr int Hd = 128;
    constexpr int KT_STRIDE = 68;
    constexpr int SMK_AREA = (Bc * Hd > Hd * KT_STRIDE) ? Bc * Hd : Hd * KT_STRIDE;  // 8704
    __shared__ uint8_t smK_area[SMK_AREA];

    const int tid = threadIdx.x;

    // Step 1: load K row-major into smK_area
    for (int e = tid; e < Bc * Hd; e += 128) {
        int j = e / Hd;
        int d = e % Hd;
        smK_area[j * Hd + d] = K_in[j * Hd + d];
    }
    __syncthreads();

    // Step 2: transpose smK → smK_T (Phase 1.5 pattern)
    uint32_t K_buf[16];
    #pragma unroll
    for (int e = 0; e < 16; ++e) {
        int elem_idx_base = tid * 4 + e * (128 * 4);
        int byte_idx = elem_idx_base;
        int j_local = byte_idx / Hd;
        int d = byte_idx % Hd;
        if (j_local < Bc && d + 3 < Hd) {
            K_buf[e] = *reinterpret_cast<uint32_t*>(&smK_area[j_local * Hd + d]);
        } else {
            K_buf[e] = 0u;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int e = 0; e < 16; ++e) {
        int elem_idx_base = tid * 4 + e * (128 * 4);
        int byte_idx = elem_idx_base;
        int j_local = byte_idx / Hd;
        int d_base = byte_idx % Hd;
        if (j_local < Bc && d_base + 3 < Hd) {
            uint8_t b0 = K_buf[e]        & 0xFF;
            uint8_t b1 = (K_buf[e] >>  8) & 0xFF;
            uint8_t b2 = (K_buf[e] >> 16) & 0xFF;
            uint8_t b3 = (K_buf[e] >> 24) & 0xFF;
            smK_area[(d_base + 0) * KT_STRIDE + j_local] = b0;
            smK_area[(d_base + 1) * KT_STRIDE + j_local] = b1;
            smK_area[(d_base + 2) * KT_STRIDE + j_local] = b2;
            smK_area[(d_base + 3) * KT_STRIDE + j_local] = b3;
        }
    }
    __syncthreads();

    // Step 3: dump smK_T (only valid d×j region: Hd × Bc bytes interleaved with padding)
    for (int e = tid; e < Hd * Bc; e += 128) {
        int d = e / Bc;
        int j = e % Bc;
        K_T_out[d * Bc + j] = smK_area[d * KT_STRIDE + j];
    }
}

static int test_transpose_verifier_dq()
{
    constexpr int Bc = 64;
    constexpr int Hd = 128;
    std::vector<uint8_t> K_h(Bc * Hd), K_T_exp(Bc * Hd), K_T_got(Bc * Hd);
    // Synthetic deterministic K
    for (int j = 0; j < Bc; ++j)
        for (int d = 0; d < Hd; ++d)
            K_h[j * Hd + d] = (uint8_t)((j * Hd + d * 7 + 11) & 0xFF);
    // CPU transpose: K_T[d, j] = K[j, d]
    for (int d = 0; d < Hd; ++d)
        for (int j = 0; j < Bc; ++j)
            K_T_exp[d * Bc + j] = K_h[j * Hd + d];

    uint8_t *K_d, *K_T_d;
    DQ_CK(cudaMalloc(&K_d, Bc * Hd));
    DQ_CK(cudaMalloc(&K_T_d, Hd * Bc));
    DQ_CK(cudaMemcpy(K_d, K_h.data(), Bc * Hd, cudaMemcpyHostToDevice));
    kernel_transpose_only_dq<<<1, 128>>>(K_d, K_T_d);
    DQ_CK(cudaDeviceSynchronize());
    DQ_CK(cudaMemcpy(K_T_got.data(), K_T_d, Hd * Bc, cudaMemcpyDeviceToHost));

    int n_match = 0, n_diff = 0;
    int first_diff_d = -1, first_diff_j = -1;
    for (int d = 0; d < Hd; ++d) {
        for (int j = 0; j < Bc; ++j) {
            if (K_T_got[d * Bc + j] == K_T_exp[d * Bc + j]) n_match++;
            else {
                n_diff++;
                if (first_diff_d < 0) { first_diff_d = d; first_diff_j = j; }
            }
        }
    }
    if (n_diff == 0) {
        printf("  K_T transpose: BIT-EXACT (%d/%d cells match)\n", n_match, Hd * Bc);
    } else {
        printf("  K_T transpose: FAIL %d diffs, first at (d=%d, j=%d) exp=0x%02x got=0x%02x\n",
               n_diff, first_diff_d, first_diff_j,
               K_T_exp[first_diff_d * Bc + first_diff_j],
               K_T_got[first_diff_d * Bc + first_diff_j]);
    }
    cudaFree(K_d); cudaFree(K_T_d);
    return n_diff;
}

// =====================================================================
// Form definitions (mirror dK test: F1..F10 + canary).
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
// Form runner: generates inputs, runs FP64 golden + GPU dQ, compares.
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
    std::vector<double> O64(sz), L64(lsz), dQ64_gold(sz);
    for (size_t i = 0; i < sz; ++i) {
        Q64[i]  = dist(rng);
        K64[i]  = dist(rng);
        V64[i]  = dist(rng);
        dO64[i] = dist(rng);
    }

    // FP64 forward → O, L
    fa_fwd_fp64(Q64.data(), K64.data(), V64.data(),
                O64.data(), L64.data(),
                bh, sl, hd, causal, window);

    // FP64 backward dQ
    fa_bwd_dq_fp64(Q64.data(), K64.data(), V64.data(),
                   O64.data(), L64.data(), dO64.data(),
                   dQ64_gold.data(),
                   bh, sl, hd, causal, window);

    // Sanity: count finite entries, find max |dQ_gold|
    size_t finite = 0;
    double max_abs_gold = 0.0;
    for (size_t i = 0; i < sz; ++i) {
        if (std::isfinite(dQ64_gold[i])) finite++;
        if (std::fabs(dQ64_gold[i]) > max_abs_gold) max_abs_gold = std::fabs(dQ64_gold[i]);
    }
    printf("    golden sanity: finite %zu/%zu  max|dQ_gold| %.4e\n",
           finite, sz, max_abs_gold);

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

    // GPU buffers
    uint8_t *dQ_gpu, *dK_gpu, *dV_gpu;
    __half  *dO_d_gpu, *dO_g_gpu;
    float   *dL, *dD, *ddQ;
    DQ_CK(cudaMalloc(&dQ_gpu, sz));
    DQ_CK(cudaMalloc(&dK_gpu, sz));
    DQ_CK(cudaMalloc(&dV_gpu, sz));
    DQ_CK(cudaMalloc(&dO_d_gpu, sz * sizeof(__half)));
    DQ_CK(cudaMalloc(&dO_g_gpu, sz * sizeof(__half)));
    DQ_CK(cudaMalloc(&dL, lsz * sizeof(float)));
    DQ_CK(cudaMalloc(&dD, lsz * sizeof(float)));
    DQ_CK(cudaMalloc(&ddQ, sz * sizeof(float)));
    DQ_CK(cudaMemcpy(dQ_gpu, Q8.data(),  sz, cudaMemcpyHostToDevice));
    DQ_CK(cudaMemcpy(dK_gpu, K8.data(),  sz, cudaMemcpyHostToDevice));
    DQ_CK(cudaMemcpy(dV_gpu, V8.data(),  sz, cudaMemcpyHostToDevice));
    DQ_CK(cudaMemcpy(dO_d_gpu, O16.data(),  sz * sizeof(__half), cudaMemcpyHostToDevice));
    DQ_CK(cudaMemcpy(dO_g_gpu, dO16.data(), sz * sizeof(__half), cudaMemcpyHostToDevice));
    DQ_CK(cudaMemcpy(dL, L32.data(), lsz * sizeof(float), cudaMemcpyHostToDevice));

    // D-precompute (reused from dK pipeline)
    fa_bwd_dk::launch_d_precompute(dO_d_gpu, dO_g_gpu, dD, bh, sl, hd, 0);
    DQ_CK(cudaDeviceSynchronize());

    // dQ launch
    DQ_CK(cudaMemset(ddQ, 0, sz * sizeof(float)));
    float scale = 1.0f / std::sqrt((float)hd);
    fa_bwd_dq::launch_dq(dQ_gpu, dK_gpu, dV_gpu, dO_g_gpu, dL, dD, ddQ,
                         bh, sl, hd, causal, window, scale, 0);
    DQ_CK(cudaDeviceSynchronize());

    std::vector<float> dQ_got(sz);
    DQ_CK(cudaMemcpy(dQ_got.data(), ddQ, sz * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare
    CmpStats stats = compare_dQ(dQ_got.data(), dQ64_gold.data(),
                                bh, sl, hd, abs_tol, rel_tol, sig);
    print_cmp("dQ vs FP64-golden", stats);

    bool ok = stats.n_above_floor == 0 ||
              100.0 * stats.n_passed_above_floor / stats.n_above_floor >= 99.5;
    printf("    verdict: %s\n\n", ok ? "PASS" : "FAIL");

    cudaFree(dQ_gpu); cudaFree(dK_gpu); cudaFree(dV_gpu);
    cudaFree(dO_d_gpu); cudaFree(dO_g_gpu);
    cudaFree(dL); cudaFree(dD); cudaFree(ddQ);
    return ok;
}

int main(int argc, char **argv)
{
    unsigned seed = 42;
    double abs_tol = 5e-2;   // FP8 dQ empirical floor (lesson from dK: defaults must match floor)
    double rel_tol = 5e-1;
    double sig = 1e-2;
    if (argc >= 2) seed = (unsigned)std::atoi(argv[1]);
    if (argc >= 3) abs_tol = std::atof(argv[2]);
    if (argc >= 4) rel_tol = std::atof(argv[3]);

    printf("=== B4.2 dQ validation ===\n");
    printf("    tol: abs %.0e + rel %.0e * |ref| (sig %.0e), seed=%u\n", abs_tol, rel_tol, sig, seed);

    printf("\n=== Per-lane K_T transpose verifier ===\n");
    int t_fail = test_transpose_verifier_dq();
    if (t_fail) {
        printf("\n=== TRANSPOSE FAIL — aborting full forms ===\n");
        return 2;
    }
    printf("\n");

    int n_forms = sizeof(FORMS) / sizeof(FORMS[0]);
    int n_pass = 0;
    for (int k = 0; k < n_forms; ++k) {
        if (run_form(FORMS[k], seed, abs_tol, rel_tol, sig)) n_pass++;
    }
    if (run_form(CANARY, seed, abs_tol, rel_tol, sig)) n_pass++;

    printf("=== SUMMARY ===\n");
    printf("    forms passed: %d / %d\n", n_pass, n_forms + 1);
    return n_pass == n_forms + 1 ? 0 : 1;
}
