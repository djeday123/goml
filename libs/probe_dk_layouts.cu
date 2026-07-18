// =====================================================================
//  probe_dk_layouts.cu — B3.1 dK probe (3 sections).
//
//  Section 1: MMA #1 layout — m16n8k16 row.col.f32.f16.f16.f32
//             dP = dO · V^T, A=dO FP16 row-major, B=V col-major,
//             D=dP F32 acc.  Per-lane verifier bit-exact vs CPU.
//
//  Section 2: MMA #2 layout — m16n8k32 row.col.f32.e4m3.e4m3.f32
//             dK = dS^T · Q, A=dS^T FP8 e4m3 row-major,
//             B=Q FP8 e4m3 col-major, D=dK F32 acc.
//             Per-lane verifier bit-exact vs CPU.
//
//  Section 3: floor dS → FP8 quantize vs FP64-golden.
//             Mini-backward (Br=64, Bc=64, Hd=128), one K-tile, all Q-rows.
//             Compute dK three ways:
//               (a) FP64-golden (no quantize anywhere)
//               (b) FP32-baseline (cast to FP32, no quantize)
//               (c) FP8-dS (FP32 compute, but dS quantized e4m3 before accum)
//             Mask patterns: no-mask, causal=1 (i=0..2 worst case).
//             Report per-pattern max_abs / max_rel_above_floor.
//
//  Artefact: stays in libs/, переиспользуется для dQ probe.
// =====================================================================

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "fa_bwd_common.cuh"  // float_to_e4m3_host, e4m3_to_float_host

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) {                \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                    \
            cudaGetErrorString(e)); std::exit(1); }} while (0)

// =====================================================================
// MMA helpers (PTX wrappers).
// =====================================================================
__device__ __forceinline__ void mma_m16n8k16_f32(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__ void mma_m16n8k32_e4m3_f32(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// =====================================================================
// Section 1 — MMA #1 layout probe (FP16 m16n8k16 → F32 acc).
//   Test: D[16,8] = A[16,16] · B^T   where logically B is "the second operand".
//   Per PTX m16n8k16.row.col: A row-major, B col-major.
//
//   For dP = dO · V^T (in backward dK):
//     A = dO [Br=i, hd=k], row-major.
//     B = V  [hd=k, Bc=j], col-major view of V row-major [Bc, hd] storage.
//     D = dP [Br=i, Bc=j], FP32 acc.
//   This is the same row.col semantic, just operand assignment.
//
//   For the probe we pick generic A/B values; verify lane-level fragment
//   layout matches docs (no a1↔a2 surprise).
// =====================================================================
__global__ void probe_mma1_kernel(
    const __half *smA_init,  // 16×16 fp16
    const __half *smB_init,  // 16×8  fp16 (col-major view)
    float *D_out)            // 32 lanes × 4 fp32
{
    __shared__ __half smA[16 * 16];
    __shared__ __half smB[16 *  8];   // stored col-major: addr (n*16 + k)*2
    const int tid = threadIdx.x;

    if (tid < 32) {
        #pragma unroll
        for (int e = 0; e < 8; ++e) smA[tid * 8 + e] = smA_init[tid * 8 + e];
        #pragma unroll
        for (int e = 0; e < 4; ++e) smB[tid * 4 + e] = smB_init[tid * 4 + e];
    }
    __syncthreads();

    const int lane = tid & 31;
    const int l_div4 = lane >> 2;
    const int l_mod4 = lane & 3;

    // A operand: per PTX m16n8k16 row.col fp16 — same layout as probe verified.
    int m_lo = l_div4 + 0, m_hi = l_div4 + 8;
    int k_lo = l_mod4 * 2 + 0, k_hi = l_mod4 * 2 + 8;
    uint32_t a0 = *reinterpret_cast<uint32_t*>(&smA[m_lo * 16 + k_lo]);
    uint32_t a1 = *reinterpret_cast<uint32_t*>(&smA[m_hi * 16 + k_lo]);
    uint32_t a2 = *reinterpret_cast<uint32_t*>(&smA[m_lo * 16 + k_hi]);
    uint32_t a3 = *reinterpret_cast<uint32_t*>(&smA[m_hi * 16 + k_hi]);

    // B operand: col-major layout in smB [n=8, k=16] storage.
    // Lane reads (k_pair, n) at adjacent k → single LDS.U32.
    int n = l_div4;       // 0..7
    int k0 = l_mod4 * 2 + 0;
    int k1 = l_mod4 * 2 + 8;
    uint32_t b0 = *reinterpret_cast<uint32_t*>(&smB[n * 16 + k0]);
    uint32_t b1 = *reinterpret_cast<uint32_t*>(&smB[n * 16 + k1]);

    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    mma_m16n8k16_f32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

    if (tid < 32) {
        D_out[lane * 4 + 0] = d0;
        D_out[lane * 4 + 1] = d1;
        D_out[lane * 4 + 2] = d2;
        D_out[lane * 4 + 3] = d3;
    }
}

// =====================================================================
// Section 2 — MMA #2 layout probe (FP8 e4m3 m16n8k32 → F32 acc).
//   A = dS^T FP8 e4m3 row-major (16×32 fp8 = 16 rows × 32 cols).
//   B = Q    FP8 e4m3 col-major (32×8  fp8 storage [n=8, k=32]).
//   D = dK contrib FP32 acc.
// =====================================================================
__global__ void probe_mma2_kernel(
    const uint8_t *smA_init,  // 16×32 fp8 = 512 bytes
    const uint8_t *smB_init,  // 8×32  fp8 = 256 bytes (col-major storage)
    float *D_out)             // 32 lanes × 4 fp32
{
    __shared__ uint8_t smA[16 * 32];
    __shared__ uint8_t smB[8 * 32];
    const int tid = threadIdx.x;

    if (tid < 32) {
        #pragma unroll
        for (int e = 0; e < 16; ++e) smA[tid * 16 + e] = smA_init[tid * 16 + e];
        #pragma unroll
        for (int e = 0; e <  8; ++e) smB[tid *  8 + e] = smB_init[tid *  8 + e];
    }
    __syncthreads();

    const int lane = tid & 31;
    const int l_div4 = lane >> 2;
    const int l_mod4 = lane & 3;

    // A operand m16n8k32 fp8 row-major. 4 uint32 per lane (16 fp8 elems).
    int m_lo = l_div4 + 0, m_hi = l_div4 + 8;
    int k_lo = l_mod4 * 4 + 0;
    int k_hi = l_mod4 * 4 + 16;
    uint32_t a0 = *reinterpret_cast<uint32_t*>(&smA[m_lo * 32 + k_lo]);
    uint32_t a1 = *reinterpret_cast<uint32_t*>(&smA[m_hi * 32 + k_lo]);
    uint32_t a2 = *reinterpret_cast<uint32_t*>(&smA[m_lo * 32 + k_hi]);
    uint32_t a3 = *reinterpret_cast<uint32_t*>(&smA[m_hi * 32 + k_hi]);

    // B operand col-major view of [n=8, k=32] storage. 2 uint32 per lane.
    int n = l_div4;
    int k0_b = l_mod4 * 4 + 0;
    int k1_b = l_mod4 * 4 + 16;
    uint32_t b0 = *reinterpret_cast<uint32_t*>(&smB[n * 32 + k0_b]);
    uint32_t b1 = *reinterpret_cast<uint32_t*>(&smB[n * 32 + k1_b]);

    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    mma_m16n8k32_e4m3_f32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

    if (tid < 32) {
        D_out[lane * 4 + 0] = d0;
        D_out[lane * 4 + 1] = d1;
        D_out[lane * 4 + 2] = d2;
        D_out[lane * 4 + 3] = d3;
    }
}

// =====================================================================
// CPU reference for MMA #1: D[m, n] = sum_k A[m, k] * B[k, n]
// (A FP16 → FP32, B FP16 → FP32, acc FP32; B addressing col-major).
// =====================================================================
static void cpu_ref_mma1(const __half *A, const __half *B, float *D)
{
    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 8; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < 16; ++k) {
                float a = (float)A[m * 16 + k];          // A row-major
                float b = (float)B[n * 16 + k];          // B col-major: storage[n, k]
                acc += a * b;
            }
            D[m * 8 + n] = acc;
        }
    }
}

// =====================================================================
// CPU reference for MMA #2: D[m, n] = sum_k A[m, k] * B[k, n]
// (A FP8 → FP32, B FP8 → FP32 via e4m3_to_float_host, acc FP32).
// A row-major [m=16, k=32]. B col-major view of [n=8, k=32].
// =====================================================================
static void cpu_ref_mma2(const uint8_t *A, const uint8_t *B, float *D)
{
    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 8; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < 32; ++k) {
                float a = e4m3_to_float_host(A[m * 32 + k]);
                float b = e4m3_to_float_host(B[n * 32 + k]);
                acc += a * b;
            }
            D[m * 8 + n] = acc;
        }
    }
}

// =====================================================================
// Verifier: compare GPU D[32 lanes × 4 fp32] vs CPU D[16, 8].
//   Lane (m, n) ownership per docs:
//     d0: m=(l/4)+0, n=(l%4)*2+0
//     d1: m=(l/4)+0, n=(l%4)*2+1
//     d2: m=(l/4)+8, n=(l%4)*2+0
//     d3: m=(l/4)+8, n=(l%4)*2+1
// =====================================================================
static int verify_layout(const char *name, const float *D_gpu, const float *D_cpu,
                         double abs_tol)
{
    int mismatches = 0;
    double max_abs = 0;
    int worst_lane = -1, worst_slot = -1;
    float worst_gpu = 0, worst_cpu = 0;

    for (int lane = 0; lane < 32; ++lane) {
        int l_div4 = lane >> 2;
        int l_mod4 = lane & 3;
        int m_lo = l_div4 + 0, m_hi = l_div4 + 8;
        int n_lo = l_mod4 * 2 + 0, n_hi = l_mod4 * 2 + 1;
        struct { int m, n, slot; } map[4] = {
            { m_lo, n_lo, 0 }, { m_lo, n_hi, 1 },
            { m_hi, n_lo, 2 }, { m_hi, n_hi, 3 },
        };
        for (int e = 0; e < 4; ++e) {
            float gpu = D_gpu[lane * 4 + map[e].slot];
            float cpu = D_cpu[map[e].m * 8 + map[e].n];
            double a = std::fabs((double)gpu - (double)cpu);
            if (a > abs_tol) mismatches++;
            if (a > max_abs) {
                max_abs = a;
                worst_lane = lane; worst_slot = map[e].slot;
                worst_gpu = gpu; worst_cpu = cpu;
            }
        }
    }
    printf("  %-50s mismatches=%-3d max_abs=%.4e  worst@lane=%d,slot=%d  gpu=%.4f cpu=%.4f\n",
           name, mismatches, max_abs, worst_lane, worst_slot, worst_gpu, worst_cpu);
    return mismatches;
}

// =====================================================================
// Section 3 — dS → FP8 floor measurement vs FP64 golden.
//
//   Mini-backward chain on (Br=64, Bc=64, Hd=128). Compute dK three ways:
//     (a) FP64 throughout — golden.
//     (b) FP32 throughout, no quantize — control (FP32 floor).
//     (c) FP32 compute, but dS quantized to e4m3 before final accum — measured.
//
//   Inputs random: Q, K, V, dO ~ N(0, 0.6). L computed from FP64 forward.
//   D computed from FP64 forward (D_i = sum_d dO[i,d]*O[i,d]).
//   Mask patterns: no-mask, causal=1 (Br=64 lights up i=0..2 N_eff regime).
// =====================================================================
static void compute_dK_fp64(
    const double *Q, const double *K, const double *V, const double *dO,
    const double *L, const double *D_i,
    double *dK_out,
    int Br, int Bc, int Hd, int causal, int sl)
{
    double scale = 1.0 / std::sqrt((double)Hd);
    for (int j = 0; j < Bc; ++j)
        for (int d = 0; d < Hd; ++d)
            dK_out[j*Hd+d] = 0.0;

    for (int i = 0; i < Br; ++i) {
        for (int j = 0; j < Bc; ++j) {
            if (causal && j > i) continue;
            if (i >= sl || j >= sl) continue;

            double s = 0.0;
            for (int d = 0; d < Hd; ++d) s += Q[i*Hd+d] * K[j*Hd+d];
            s *= scale;
            double P = std::exp(s - L[i]);

            double dP = 0.0;
            for (int d = 0; d < Hd; ++d) dP += dO[i*Hd+d] * V[j*Hd+d];

            double dS = P * (dP - D_i[i]);

            for (int d = 0; d < Hd; ++d)
                dK_out[j*Hd+d] += dS * Q[i*Hd+d] * scale;
        }
    }
}

static void compute_dK_fp32(
    const float *Q, const float *K, const float *V, const float *dO,
    const float *L, const float *D_i,
    float *dK_out,
    int Br, int Bc, int Hd, int causal, int sl)
{
    float scale = 1.0f / std::sqrt((float)Hd);
    for (int j = 0; j < Bc; ++j)
        for (int d = 0; d < Hd; ++d)
            dK_out[j*Hd+d] = 0.0f;

    for (int i = 0; i < Br; ++i) {
        for (int j = 0; j < Bc; ++j) {
            if (causal && j > i) continue;
            if (i >= sl || j >= sl) continue;

            float s = 0.0f;
            for (int d = 0; d < Hd; ++d) s += Q[i*Hd+d] * K[j*Hd+d];
            s *= scale;
            float P = std::exp(s - L[i]);

            float dP = 0.0f;
            for (int d = 0; d < Hd; ++d) dP += dO[i*Hd+d] * V[j*Hd+d];

            float dS = P * (dP - D_i[i]);

            for (int d = 0; d < Hd; ++d)
                dK_out[j*Hd+d] += dS * Q[i*Hd+d] * scale;
        }
    }
}

// dS quantize: FP32 → e4m3 → FP32 (round-trip).
static inline float quantize_e4m3(float x) {
    return e4m3_to_float_host(float_to_e4m3_host(x));
}

static void compute_dK_fp32_dS_quantized(
    const float *Q, const float *K, const float *V, const float *dO,
    const float *L, const float *D_i,
    float *dK_out,
    int Br, int Bc, int Hd, int causal, int sl)
{
    float scale = 1.0f / std::sqrt((float)Hd);
    for (int j = 0; j < Bc; ++j)
        for (int d = 0; d < Hd; ++d)
            dK_out[j*Hd+d] = 0.0f;

    for (int i = 0; i < Br; ++i) {
        for (int j = 0; j < Bc; ++j) {
            if (causal && j > i) continue;
            if (i >= sl || j >= sl) continue;

            float s = 0.0f;
            for (int d = 0; d < Hd; ++d) s += Q[i*Hd+d] * K[j*Hd+d];
            s *= scale;
            float P = std::exp(s - L[i]);

            float dP = 0.0f;
            for (int d = 0; d < Hd; ++d) dP += dO[i*Hd+d] * V[j*Hd+d];

            float dS = P * (dP - D_i[i]);
            float dS_q = quantize_e4m3(dS);       // <- key step

            for (int d = 0; d < Hd; ++d)
                dK_out[j*Hd+d] += dS_q * Q[i*Hd+d] * scale;
        }
    }
}

// FP64 forward to produce exact L and D_i for given inputs.
static void fa_fwd_fp64(
    const double *Q, const double *K, const double *V,
    double *O, double *L_out, double *D_out,
    const double *dO,
    int Br, int Hd, int causal)
{
    double scale = 1.0 / std::sqrt((double)Hd);
    for (int i = 0; i < Br; ++i) {
        std::vector<double> s(Br, -INFINITY), p(Br, 0.0);
        double m_i = -INFINITY;
        for (int j = 0; j < Br; ++j) {
            if (causal && j > i) continue;
            double dot = 0;
            for (int d = 0; d < Hd; ++d) dot += Q[i*Hd+d] * K[j*Hd+d];
            s[j] = dot * scale;
            if (s[j] > m_i) m_i = s[j];
        }
        double l_i = 0;
        for (int j = 0; j < Br; ++j) {
            if (!std::isfinite(s[j])) { p[j] = 0; continue; }
            p[j] = std::exp(s[j] - m_i);
            l_i += p[j];
        }
        if (l_i <= 0) l_i = 1.0;
        for (int d = 0; d < Hd; ++d) {
            double acc = 0;
            for (int j = 0; j < Br; ++j) acc += p[j] * V[j*Hd+d];
            O[i*Hd+d] = acc / l_i;
        }
        L_out[i] = m_i + std::log(l_i);
        double D = 0;
        for (int d = 0; d < Hd; ++d) D += dO[i*Hd+d] * O[i*Hd+d];
        D_out[i] = D;
    }
}

struct FloorStat {
    double max_abs;
    double max_rel_above_floor;
    size_t n_above_floor;
    int worst_j, worst_d;
    double worst_ref, worst_got;
};

template <typename Tref>
static FloorStat compare_dK(const float *got, const Tref *ref,
                            int Bc, int Hd, double sig_floor)
{
    FloorStat s{};
    s.worst_j = s.worst_d = -1;
    for (int j = 0; j < Bc; ++j) {
        for (int d = 0; d < Hd; ++d) {
            double r = (double)ref[j*Hd+d];
            double g = (double)got[j*Hd+d];
            double a = std::fabs(g - r);
            if (a > s.max_abs) {
                s.max_abs = a;
                s.worst_j = j; s.worst_d = d;
                s.worst_ref = r; s.worst_got = g;
            }
            if (std::fabs(r) > sig_floor) {
                s.n_above_floor++;
                double rel = a / std::fabs(r);
                if (rel > s.max_rel_above_floor) s.max_rel_above_floor = rel;
            }
        }
    }
    return s;
}

static void run_section3_case(const char *name, int causal, int sl, unsigned seed)
{
    constexpr int Br = 64, Bc = 64, Hd = 128;
    const size_t sz = (size_t)Br * Hd;

    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 0.6);
    std::vector<double> Q64(sz), K64(sz), V64(sz), dO64(sz);
    for (size_t i = 0; i < sz; ++i) {
        Q64[i] = dist(rng);
        K64[i] = dist(rng);
        V64[i] = dist(rng);
        dO64[i] = dist(rng);
    }

    // FP64 forward → L, O, D
    std::vector<double> O64(sz), L64(Br), D64(Br);
    fa_fwd_fp64(Q64.data(), K64.data(), V64.data(),
                O64.data(), L64.data(), D64.data(),
                dO64.data(), Br, Hd, causal);

    // FP64 dK golden
    std::vector<double> dK64(sz, 0);
    compute_dK_fp64(Q64.data(), K64.data(), V64.data(), dO64.data(),
                    L64.data(), D64.data(),
                    dK64.data(), Br, Bc, Hd, causal, sl);

    // Cast to FP32 (Q,K,V,dO ARE NOT FP8-quantized in this probe — we isolate
    // dS-quantize floor only; Q/K/V FP8 floor will be measured separately in
    // dK baseline kernel).
    std::vector<float> Q32(sz), K32(sz), V32(sz), dO32(sz);
    std::vector<float> L32(Br), D32(Br);
    for (size_t i = 0; i < sz; ++i) {
        Q32[i] = (float)Q64[i];
        K32[i] = (float)K64[i];
        V32[i] = (float)V64[i];
        dO32[i] = (float)dO64[i];
    }
    for (int i = 0; i < Br; ++i) { L32[i] = (float)L64[i]; D32[i] = (float)D64[i]; }

    // FP32 baseline (no quantize)
    std::vector<float> dK_fp32(sz, 0);
    compute_dK_fp32(Q32.data(), K32.data(), V32.data(), dO32.data(),
                    L32.data(), D32.data(),
                    dK_fp32.data(), Br, Bc, Hd, causal, sl);

    // FP32 compute with dS quantized
    std::vector<float> dK_dSq(sz, 0);
    compute_dK_fp32_dS_quantized(
        Q32.data(), K32.data(), V32.data(), dO32.data(),
        L32.data(), D32.data(),
        dK_dSq.data(), Br, Bc, Hd, causal, sl);

    auto stat_fp32 = compare_dK<double>(dK_fp32.data(), dK64.data(), Bc, Hd, 1e-2);
    auto stat_dSq  = compare_dK<double>(dK_dSq.data(),  dK64.data(), Bc, Hd, 1e-2);

    printf("  [%s] FP32 baseline vs FP64: max_abs=%.3e  max_rel_af=%.3e (n_af=%zu)\n",
           name, stat_fp32.max_abs, stat_fp32.max_rel_above_floor, stat_fp32.n_above_floor);
    printf("       dS→FP8 vs FP64:      max_abs=%.3e  max_rel_af=%.3e  worst@(j=%d,d=%d) ref=%.4e got=%.4e\n",
           stat_dSq.max_abs, stat_dSq.max_rel_above_floor,
           stat_dSq.worst_j, stat_dSq.worst_d, stat_dSq.worst_ref, stat_dSq.worst_got);
}

// =====================================================================
// Main: run all 3 sections.
// =====================================================================
int main()
{
    printf("=== Probe dK MMA layouts + dS-quantize floor ===\n\n");

    // -------- Section 1 — MMA #1 layout (FP16×FP16 → F32) --------
    printf("Section 1: MMA #1 m16n8k16.row.col.f32.f16.f16.f32 (dO·V^T)\n");
    {
        std::vector<float> A_f(16*16), B_f(16*8);
        for (int i = 0; i < 16; ++i)
            for (int k = 0; k < 16; ++k)
                A_f[i*16+k] = ((float)i - 7.5f) * 0.125f + (float)k * 0.0625f;
        // B col-major storage [n=8, k=16]: B_f[n*16 + k]
        for (int n = 0; n < 8; ++n)
            for (int k = 0; k < 16; ++k)
                B_f[n*16+k] = ((float)k - 7.5f) * 0.0625f + (float)n * 0.125f;
        std::vector<__half> A_h(16*16), B_h(16*8);
        for (int i = 0; i < 16*16; ++i) A_h[i] = __float2half_rn(A_f[i]);
        for (int i = 0; i < 16*8;  ++i) B_h[i] = __float2half_rn(B_f[i]);

        std::vector<float> D_cpu(16*8);
        cpu_ref_mma1(A_h.data(), B_h.data(), D_cpu.data());

        __half *dA, *dB; float *dD;
        CK(cudaMalloc(&dA, 256*sizeof(__half)));
        CK(cudaMalloc(&dB, 128*sizeof(__half)));
        CK(cudaMalloc(&dD, 32*4*sizeof(float)));
        CK(cudaMemcpy(dA, A_h.data(), 256*sizeof(__half), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, B_h.data(), 128*sizeof(__half), cudaMemcpyHostToDevice));
        probe_mma1_kernel<<<1, 32>>>(dA, dB, dD);
        CK(cudaDeviceSynchronize());
        std::vector<float> D_gpu(32*4);
        CK(cudaMemcpy(D_gpu.data(), dD, 32*4*sizeof(float), cudaMemcpyDeviceToHost));

        verify_layout("MMA #1 (FP16×FP16→F32)", D_gpu.data(), D_cpu.data(), 1e-2);
        CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dD));
    }
    printf("\n");

    // -------- Section 2 — MMA #2 layout (FP8×FP8 → F32) --------
    printf("Section 2: MMA #2 m16n8k32.row.col.f32.e4m3.e4m3.f32 (dS^T·Q)\n");
    {
        std::vector<float> A_f(16*32), B_f(8*32);
        for (int i = 0; i < 16; ++i)
            for (int k = 0; k < 32; ++k)
                A_f[i*32+k] = ((float)i - 7.5f) * 0.0625f + (float)k * 0.03125f;
        for (int n = 0; n < 8; ++n)
            for (int k = 0; k < 32; ++k)
                B_f[n*32+k] = ((float)k - 15.5f) * 0.03125f + (float)n * 0.0625f;
        std::vector<uint8_t> A_8(16*32), B_8(8*32);
        for (int i = 0; i < 16*32; ++i) A_8[i] = float_to_e4m3_host(A_f[i]);
        for (int i = 0; i <  8*32; ++i) B_8[i] = float_to_e4m3_host(B_f[i]);

        std::vector<float> D_cpu(16*8);
        cpu_ref_mma2(A_8.data(), B_8.data(), D_cpu.data());

        uint8_t *dA, *dB; float *dD;
        CK(cudaMalloc(&dA, 16*32));
        CK(cudaMalloc(&dB,  8*32));
        CK(cudaMalloc(&dD, 32*4*sizeof(float)));
        CK(cudaMemcpy(dA, A_8.data(), 16*32, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, B_8.data(),  8*32, cudaMemcpyHostToDevice));
        probe_mma2_kernel<<<1, 32>>>(dA, dB, dD);
        CK(cudaDeviceSynchronize());
        std::vector<float> D_gpu(32*4);
        CK(cudaMemcpy(D_gpu.data(), dD, 32*4*sizeof(float), cudaMemcpyDeviceToHost));

        verify_layout("MMA #2 (FP8×FP8→F32)", D_gpu.data(), D_cpu.data(), 1e-4);
        CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dD));
    }
    printf("\n");

    // -------- Section 3 — dS quantize floor vs FP64-golden --------
    printf("Section 3: dS→FP8 floor measurement (Br=64 Bc=64 Hd=128)\n");
    printf("  Tolerance signal: sig_floor=1e-2 for above-floor (matches dV tol).\n");
    run_section3_case("non-causal sl=64",       0, 64, 42);
    run_section3_case("causal sl=64",           1, 64, 42);
    run_section3_case("non-causal sl=64 seed=7", 0, 64, 7);
    run_section3_case("causal sl=64 seed=7",    1, 64, 7);

    printf("\n=== Probe complete ===\n");
    return 0;
}
