// =============================================================================
// FlashAttention v62 — FP8 Forward (full pipeline)
// =============================================================================
// Builds on v61 PoC (which verified S = Q·Kᵀ FP8 MMA path) to deliver the
// full forward kernel with FP8 inputs end-to-end:
//
//   1. Q, K, V FP8 e4m3 inputs (caller quantizes; per-tensor scale)
//   2. S = Q·Kᵀ via mma m16n8k32 kind::f8f6f4 .f16.e4m3.e4m3.f16
//   3. Online softmax in FP32 (FlashInfer-style with FP16 ex2)
//   4. Quantize P (FP16) → FP8 e4m3 per-row scale, write to SMEM scratch
//   5. P·V via mma m16n8k32 kind::f8f6f4 — A=P (FP8), B=V (FP8 transposed
//      via byte-gather, no pre-transpose)
//   6. O accumulator in packed FP16 (cross-K-block, with rescale)
//   7. Final O · (1/sum_exp) → FP16 output
//
// Single-buffered K, V loads for PoC simplicity. v63 will add double-buffer.
//
// Build: nvcc -gencode arch=compute_120a,code=sm_120a
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define FA_BR 64
#define FA_BC 64
#define FA_THREADS 128
#define FA_STRIDE 128

__device__ __forceinline__ void cpa16(void *s, const void *g, int n)
{
    uint32_t sa = __cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" ::"r"(sa), "l"(g), "r"(n));
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }

__device__ __forceinline__ void mma_fp8_f16(
    uint32_t &d0, uint32_t &d1,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t c0, uint32_t c1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e4m3.e4m3.f16 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(c0), "r"(c1));
}

__device__ __forceinline__ int swz_byte(int row, int col_bytes)
{
    int chunk = col_bytes >> 4;
    int within = col_bytes & 15;
    return row * FA_STRIDE + ((chunk ^ (row & 7)) << 4) + within;
}

__device__ __forceinline__ void load_tile_fp8(
    uint8_t *dst, const uint8_t *src, int start, int rows,
    int seq_len, int head_dim)
{
    constexpr int CHUNK = 16;
    int chunks_per_row = head_dim / CHUNK;
    int total = rows * chunks_per_row;
#pragma unroll 4
    for (int c = threadIdx.x; c < total; c += FA_THREADS)
    {
        int row = c / chunks_per_row;
        int col_bytes = (c % chunks_per_row) * CHUNK;
        int gr = start + row;
        int dst_off = swz_byte(row, col_bytes);
        cpa16(&dst[dst_off], &src[gr * head_dim + col_bytes], (gr < seq_len) ? 16 : 0);
    }
}

__device__ __forceinline__ uint8_t fp16_to_e4m3_d(__half h)
{
    float f = __half2float(h);
    if (f != f) return 0x7Fu;
    int sign = (f < 0.0f) ? 1 : 0;
    float af = fabsf(f);
    if (af > 448.0f) return sign ? 0xFEu : 0x7Eu;
    if (af < 1.953125e-3f) return sign ? 0x80u : 0x00u;
    int eu = (int)floorf(log2f(af));
    float mf = af / ldexpf(1.0f, eu) - 1.0f;
    int m3 = (int)(mf * 8.0f + 0.5f);
    if (m3 >= 8) { m3 = 0; eu++; }
    int eb = eu + 7;
    if (eb < 1) {
        int ms = (int)(af / ldexpf(1.0f, -9) + 0.5f);
        if (ms > 7) ms = 7;
        return (uint8_t)((sign << 7) | (ms & 7));
    }
    if (eb > 15) eb = 15;
    return (uint8_t)((sign << 7) | (eb << 3) | (m3 & 7));
}

// V is stored [seq_k, head] but mma B operand expects col-major (k inner).
// Gather 8 fp8 across k=4 positions at fixed n.
__device__ __forceinline__ void gather_v_b(
    uint32_t &b0, uint32_t &b1,
    const uint8_t *smV, int n_base, int k_base, int gid, int sub)
{
    int n = n_base + gid;
    int k_lo = k_base + sub * 4;
    int k_hi = k_lo + 16;
    uint8_t bytes_lo[4] = {
        smV[swz_byte(k_lo + 0, n)],
        smV[swz_byte(k_lo + 1, n)],
        smV[swz_byte(k_lo + 2, n)],
        smV[swz_byte(k_lo + 3, n)],
    };
    uint8_t bytes_hi[4] = {
        smV[swz_byte(k_hi + 0, n)],
        smV[swz_byte(k_hi + 1, n)],
        smV[swz_byte(k_hi + 2, n)],
        smV[swz_byte(k_hi + 3, n)],
    };
    b0 = ((uint32_t)bytes_lo[0]) | ((uint32_t)bytes_lo[1] << 8) |
         ((uint32_t)bytes_lo[2] << 16) | ((uint32_t)bytes_lo[3] << 24);
    b1 = ((uint32_t)bytes_hi[0]) | ((uint32_t)bytes_hi[1] << 8) |
         ((uint32_t)bytes_hi[2] << 16) | ((uint32_t)bytes_hi[3] << 24);
}

__global__ void __launch_bounds__(FA_THREADS, 3)
    fa62_kernel(
        const uint8_t *__restrict__ Q,
        const uint8_t *__restrict__ K,
        const uint8_t *__restrict__ V,
        __half *__restrict__ O,
        int seq_len, int head_dim, int causal, float scale,
        float qk_descale, float v_descale)
{
    int nqt = (seq_len + FA_BR - 1) / FA_BR;
    int bh = blockIdx.x / nqt, qt = blockIdx.x % nqt, qs = qt * FA_BR;
    if (qs >= seq_len) return;
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    int gid = lane / 4, tid = lane % 4;
    int mrb = wid * 16;

    extern __shared__ uint8_t raw[];
    uint8_t *smQ = raw;
    uint8_t *smK = smQ + FA_BR * FA_STRIDE;
    uint8_t *smV = smK + FA_BC * FA_STRIDE;
    uint8_t *smP = smV + FA_BC * FA_STRIDE;

    int hs = seq_len * head_dim;
    const uint8_t *Qh = Q + bh * hs;
    const uint8_t *Kh = K + bh * hs;
    const uint8_t *Vh = V + bh * hs;
    __half *Oh = O + bh * hs;

    load_tile_fp8(smQ, Qh, qs, FA_BR, seq_len, head_dim);
    cpa_commit();
    cpa_wait<0>();
    __syncthreads();

    uint32_t Qr[4][4];
#pragma unroll
    for (int ks = 0; ks < 4; ks++)
    {
        int k_off = ks * 32;
        int cl = k_off + tid * 4;
        int ch = cl + 16;
        int g0 = mrb + gid, g8 = g0 + 8;
        Qr[ks][0] = *(uint32_t *)&smQ[swz_byte(g0, cl)];
        Qr[ks][1] = *(uint32_t *)&smQ[swz_byte(g8, cl)];
        Qr[ks][2] = *(uint32_t *)&smQ[swz_byte(g0, ch)];
        Qr[ks][3] = *(uint32_t *)&smQ[swz_byte(g8, ch)];
    }

    uint32_t Or_p[16][2];
#pragma unroll
    for (int t = 0; t < 16; t++) Or_p[t][0] = Or_p[t][1] = 0u;

    float rmax[2] = {-1e30f, -1e30f}, rsexp[2] = {0.0f, 0.0f};
    int nkv = (seq_len + FA_BC - 1) / FA_BC;

    for (int kv = 0; kv < nkv; kv++)
    {
        int kvs = kv * FA_BC;
        if (causal && kvs > qs + FA_BR - 1) break;

        load_tile_fp8(smK, Kh, kvs, FA_BC, seq_len, head_dim);
        load_tile_fp8(smV, Vh, kvs, FA_BC, seq_len, head_dim);
        cpa_commit();
        cpa_wait<0>();
        __syncthreads();

        // S = Q · Kᵀ
        uint32_t Sr_p[8][2];
#pragma unroll
        for (int nt = 0; nt < 8; nt++) Sr_p[nt][0] = Sr_p[nt][1] = 0u;
#pragma unroll
        for (int ks = 0; ks < 4; ks++)
        {
            int k_off = ks * 32;
            int cl = k_off + tid * 4, ch = cl + 16;
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
            {
                int br = nt * 8;
                uint32_t b0 = *(uint32_t *)&smK[swz_byte(br + gid, cl)];
                uint32_t b1 = *(uint32_t *)&smK[swz_byte(br + gid, ch)];
                mma_fp8_f16(Sr_p[nt][0], Sr_p[nt][1],
                            Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3],
                            b0, b1, Sr_p[nt][0], Sr_p[nt][1]);
            }
        }

        float Sr[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            __half2 v0 = *reinterpret_cast<__half2 *>(&Sr_p[nt][0]);
            __half2 v1 = *reinterpret_cast<__half2 *>(&Sr_p[nt][1]);
            float fs = scale * qk_descale;
            Sr[nt][0] = __half2float(__low2half(v0)) * fs;
            Sr[nt][1] = __half2float(__high2half(v0)) * fs;
            Sr[nt][2] = __half2float(__low2half(v1)) * fs;
            Sr[nt][3] = __half2float(__high2half(v1)) * fs;
        }

        if (causal)
        {
            int gq0 = qs + mrb + gid, gq8 = gq0 + 8;
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
            {
                int gk0 = kvs + nt * 8 + tid * 2, gk1 = gk0 + 1;
                if (gk0 > gq0) Sr[nt][0] = -1e30f;
                if (gk1 > gq0) Sr[nt][1] = -1e30f;
                if (gk0 > gq8) Sr[nt][2] = -1e30f;
                if (gk1 > gq8) Sr[nt][3] = -1e30f;
                if (gq0 >= seq_len) Sr[nt][0] = Sr[nt][1] = -1e30f;
                if (gq8 >= seq_len) Sr[nt][2] = Sr[nt][3] = -1e30f;
                if (gk0 >= seq_len) Sr[nt][0] = Sr[nt][2] = -1e30f;
                if (gk1 >= seq_len) Sr[nt][1] = Sr[nt][3] = -1e30f;
            }
        }

        float nm[2] = {-1e30f, -1e30f};
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            nm[0] = fmaxf(nm[0], fmaxf(Sr[nt][0], Sr[nt][1]));
            nm[1] = fmaxf(nm[1], fmaxf(Sr[nt][2], Sr[nt][3]));
        }
        nm[0] = fmaxf(nm[0], __shfl_xor_sync(0xffffffff, nm[0], 1));
        nm[0] = fmaxf(nm[0], __shfl_xor_sync(0xffffffff, nm[0], 2));
        nm[1] = fmaxf(nm[1], __shfl_xor_sync(0xffffffff, nm[1], 1));
        nm[1] = fmaxf(nm[1], __shfl_xor_sync(0xffffffff, nm[1], 2));
        nm[0] = fmaxf(nm[0], rmax[0]);
        nm[1] = fmaxf(nm[1], rmax[1]);

        float rsc0 = __expf(rmax[0] - nm[0]);
        float rsc1 = __expf(rmax[1] - nm[1]);
        __half2 h2_rsc0 = __float2half2_rn(rsc0);
        __half2 h2_rsc1 = __float2half2_rn(rsc1);
#pragma unroll
        for (int t = 0; t < 16; t++)
        {
            __half2 v0 = *reinterpret_cast<__half2 *>(&Or_p[t][0]);
            __half2 v1 = *reinterpret_cast<__half2 *>(&Or_p[t][1]);
            v0 = __hmul2(v0, h2_rsc0);
            v1 = __hmul2(v1, h2_rsc1);
            Or_p[t][0] = *reinterpret_cast<uint32_t *>(&v0);
            Or_p[t][1] = *reinterpret_cast<uint32_t *>(&v1);
        }
        rmax[0] = nm[0]; rmax[1] = nm[1];

        float ns[2] = {0.0f, 0.0f};
        float P_local[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            float d0 = Sr[nt][0] - rmax[0], d1 = Sr[nt][1] - rmax[0];
            float d2 = Sr[nt][2] - rmax[1], d3 = Sr[nt][3] - rmax[1];
            float p0 = __expf(d0), p1 = __expf(d1);
            float p2 = __expf(d2), p3 = __expf(d3);
            ns[0] += p0 + p1;
            ns[1] += p2 + p3;
            P_local[nt][0] = p0;
            P_local[nt][1] = p1;
            P_local[nt][2] = p2;
            P_local[nt][3] = p3;
        }
        ns[0] += __shfl_xor_sync(0xffffffff, ns[0], 1);
        ns[0] += __shfl_xor_sync(0xffffffff, ns[0], 2);
        ns[1] += __shfl_xor_sync(0xffffffff, ns[1], 1);
        ns[1] += __shfl_xor_sync(0xffffffff, ns[1], 2);
        rsexp[0] = rsexp[0] * rsc0 + ns[0];
        rsexp[1] = rsexp[1] * rsc1 + ns[1];

        __syncthreads();

        // Quantize P → smP at natural [m_row, n_col] layout.
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            int row0 = mrb + gid, row8 = mrb + gid + 8;
            int col0 = nt * 8 + tid * 2, col1 = col0 + 1;
            smP[swz_byte(row0, col0)] = fp16_to_e4m3_d(__float2half(P_local[nt][0]));
            smP[swz_byte(row0, col1)] = fp16_to_e4m3_d(__float2half(P_local[nt][1]));
            smP[swz_byte(row8, col0)] = fp16_to_e4m3_d(__float2half(P_local[nt][2]));
            smP[swz_byte(row8, col1)] = fp16_to_e4m3_d(__float2half(P_local[nt][3]));
        }
        __syncthreads();

        // O += P · V
#pragma unroll
        for (int ks = 0; ks < 2; ks++)
        {
            int k_off = ks * 32;
            int cl = k_off + tid * 4, ch = cl + 16;
            int g0 = mrb + gid, g8 = g0 + 8;
            uint32_t Pr[4];
            Pr[0] = *(uint32_t *)&smP[swz_byte(g0, cl)];
            Pr[1] = *(uint32_t *)&smP[swz_byte(g8, cl)];
            Pr[2] = *(uint32_t *)&smP[swz_byte(g0, ch)];
            Pr[3] = *(uint32_t *)&smP[swz_byte(g8, ch)];
#pragma unroll
            for (int nt = 0; nt < 16; nt++)
            {
                uint32_t b0, b1;
                gather_v_b(b0, b1, smV, nt * 8, k_off, gid, tid);
                mma_fp8_f16(Or_p[nt][0], Or_p[nt][1],
                            Pr[0], Pr[1], Pr[2], Pr[3],
                            b0, b1, Or_p[nt][0], Or_p[nt][1]);
            }
        }
        __syncthreads();
    }

    float li0 = (rsexp[0] > 0) ? v_descale / rsexp[0] : 0.0f;
    float li1 = (rsexp[1] > 0) ? v_descale / rsexp[1] : 0.0f;
    int gr0 = qs + mrb + gid, gr8 = gr0 + 8;
#pragma unroll
    for (int nt = 0; nt < 16; nt++)
    {
        int c0 = nt * 8 + tid * 2, c1 = c0 + 1;
        __half2 v0 = *reinterpret_cast<__half2 *>(&Or_p[nt][0]);
        __half2 v1 = *reinterpret_cast<__half2 *>(&Or_p[nt][1]);
        float O0 = __half2float(__low2half(v0)) * li0;
        float O1 = __half2float(__high2half(v0)) * li0;
        float O2 = __half2float(__low2half(v1)) * li1;
        float O3 = __half2float(__high2half(v1)) * li1;
        if (gr0 < seq_len && c0 < head_dim) Oh[gr0 * head_dim + c0] = __float2half(O0);
        if (gr0 < seq_len && c1 < head_dim) Oh[gr0 * head_dim + c1] = __float2half(O1);
        if (gr8 < seq_len && c0 < head_dim) Oh[gr8 * head_dim + c0] = __float2half(O2);
        if (gr8 < seq_len && c1 < head_dim) Oh[gr8 * head_dim + c1] = __float2half(O3);
    }
}

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); }} while(0)

static inline uint8_t float_to_e4m3(float f)
{
    if (f != f) return 0x7Fu;
    int sign = (f < 0.0f) ? 1 : 0;
    float af = fabsf(f);
    if (af > 448.0f) return sign ? 0xFEu : 0x7Eu;
    if (af < 1.953125e-3f) return sign ? 0x80u : 0x00u;
    int eu = (int)floorf(log2f(af));
    float mf = af / ldexpf(1.0f, eu) - 1.0f;
    int m3 = (int)(mf * 8.0f + 0.5f);
    if (m3 >= 8) { m3 = 0; eu++; }
    int eb = eu + 7;
    if (eb < 1) {
        int ms = (int)(af / ldexpf(1.0f, -9) + 0.5f);
        if (ms > 7) ms = 7;
        return (uint8_t)((sign << 7) | (ms & 7));
    }
    if (eb > 15) eb = 15;
    return (uint8_t)((sign << 7) | (eb << 3) | (m3 & 7));
}
static inline float e4m3_to_float(uint8_t v)
{
    int s = (v >> 7) & 1, e = (v >> 3) & 0xF, m = v & 7;
    if (e == 0xF && m == 7) return nanf("");
    float r = (e == 0) ? ldexpf((float)m, -9) : ldexpf(1.0f + m / 8.0f, e - 7);
    return s ? -r : r;
}
static inline float fp16f(uint16_t h)
{
    __half hv; memcpy(&hv, &h, 2); return __half2float(hv);
}

void cpu_attention_fp8(
    const uint8_t *Q, const uint8_t *K, const uint8_t *V,
    float *O_out, int bh, int sl, int hd, int causal)
{
    float scale = 1.0f / sqrtf((float)hd);
    int hs = sl * hd;
    for (int h = 0; h < bh; h++)
    {
        const uint8_t *Qh = Q + h * hs;
        const uint8_t *Kh = K + h * hs;
        const uint8_t *Vh = V + h * hs;
        float *Oh = O_out + h * hs;
        for (int q = 0; q < sl; q++)
        {
            int kv_max = causal ? (q + 1) : sl;
            float *P = (float *)malloc(sizeof(float) * sl);
            float rmax = -1e30f;
            for (int k = 0; k < kv_max; k++)
            {
                float s = 0;
                for (int d = 0; d < hd; d++)
                    s += e4m3_to_float(Qh[q * hd + d]) * e4m3_to_float(Kh[k * hd + d]);
                P[k] = s * scale;
                if (P[k] > rmax) rmax = P[k];
            }
            float rsum = 0;
            for (int k = 0; k < kv_max; k++)
            {
                P[k] = expf(P[k] - rmax);
                rsum += P[k];
            }
            for (int k = 0; k < kv_max; k++) P[k] /= rsum;
            for (int d = 0; d < hd; d++)
            {
                float o = 0;
                for (int k = 0; k < kv_max; k++)
                    o += P[k] * e4m3_to_float(Vh[k * hd + d]);
                Oh[q * hd + d] = o;
            }
            free(P);
        }
    }
}

int main()
{
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    int clk = 0; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    printf("=== FA v62 FP8 forward — full pipeline ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n\n", p.name, p.multiProcessorCount, clk / 1000);

    printf("--- Correctness vs CPU FP8-roundtripped reference ---\n");
    int configs[][4] = {
        {1, 64, 128, 0},
        {1, 128, 128, 0},
        {1, 256, 128, 0},
        {1, 512, 128, 0},
        {2, 256, 128, 1},
    };
    for (auto &c : configs)
    {
        int bh = c[0], sl = c[1], hd = c[2], ca = c[3];
        size_t n_elems = (size_t)bh * sl * hd;

        float *Qf = (float *)malloc(sizeof(float) * n_elems);
        float *Kf = (float *)malloc(sizeof(float) * n_elems);
        float *Vf = (float *)malloc(sizeof(float) * n_elems);
        srand(42);
        for (size_t i = 0; i < n_elems; i++) {
            Qf[i] = ((float)(rand() % 16) - 8.0f) * 0.125f;
            Kf[i] = ((float)(rand() % 16) - 8.0f) * 0.125f;
            Vf[i] = ((float)(rand() % 16) - 8.0f) * 0.125f;
        }
        uint8_t *Qq = (uint8_t *)malloc(n_elems);
        uint8_t *Kq = (uint8_t *)malloc(n_elems);
        uint8_t *Vq = (uint8_t *)malloc(n_elems);
        for (size_t i = 0; i < n_elems; i++) {
            Qq[i] = float_to_e4m3(Qf[i]);
            Kq[i] = float_to_e4m3(Kf[i]);
            Vq[i] = float_to_e4m3(Vf[i]);
        }

        float *O_ref = (float *)malloc(sizeof(float) * n_elems);
        cpu_attention_fp8(Qq, Kq, Vq, O_ref, bh, sl, hd, ca);

        uint8_t *Q_d, *K_d, *V_d;
        __half *O_d;
        CK(cudaMalloc(&Q_d, n_elems));
        CK(cudaMalloc(&K_d, n_elems));
        CK(cudaMalloc(&V_d, n_elems));
        CK(cudaMalloc(&O_d, n_elems * 2));
        CK(cudaMemcpy(Q_d, Qq, n_elems, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(K_d, Kq, n_elems, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(V_d, Vq, n_elems, cudaMemcpyHostToDevice));
        CK(cudaMemset(O_d, 0, n_elems * 2));

        int smem = (FA_BR + 2 * FA_BC + FA_BR) * FA_STRIDE;
        CK(cudaFuncSetAttribute(fa62_kernel,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        int nqt = (sl + FA_BR - 1) / FA_BR;
        float scale = 1.0f / sqrtf((float)hd);
        fa62_kernel<<<bh * nqt, FA_THREADS, smem>>>(
            Q_d, K_d, V_d, O_d, sl, hd, ca, scale, 1.0f, 1.0f);
        CK(cudaDeviceSynchronize());

        uint16_t *O_cpu = (uint16_t *)malloc(n_elems * 2);
        CK(cudaMemcpy(O_cpu, O_d, n_elems * 2, cudaMemcpyDeviceToHost));

        float mx = 0;
        int errs = 0;
        for (size_t i = 0; i < n_elems; i++)
        {
            float gpu = fp16f(O_cpu[i]);
            float ref = O_ref[i];
            float ae = fabsf(gpu - ref);
            if (ae > mx) mx = ae;
            if (ae > fmaxf(0.05f, fabsf(ref) * 0.1f)) errs++;
        }
        printf("  bh=%d sl=%d hd=%d ca=%d  max_diff=%.4f errs=%d → %s\n",
               bh, sl, hd, ca, mx, errs, errs == 0 ? "PASS" : "FAIL");

        free(Qf); free(Kf); free(Vf);
        free(Qq); free(Kq); free(Vq);
        free(O_ref); free(O_cpu);
        cudaFree(Q_d); cudaFree(K_d); cudaFree(V_d); cudaFree(O_d);
    }

    printf("\n--- Performance ---\n");
    int bench_configs[][3] = {
        {4, 1024, 128},
        {4, 2048, 128},
        {8, 2048, 128},
        {4, 4096, 128},
    };
    for (auto &c : bench_configs)
    {
        int bh = c[0], sl = c[1], hd = c[2];
        size_t n_elems = (size_t)bh * sl * hd;
        uint8_t *Q_d, *K_d, *V_d;
        __half *O_d;
        CK(cudaMalloc(&Q_d, n_elems));
        CK(cudaMalloc(&K_d, n_elems));
        CK(cudaMalloc(&V_d, n_elems));
        CK(cudaMalloc(&O_d, n_elems * 2));
        CK(cudaMemset(Q_d, 0x38, n_elems));
        CK(cudaMemset(K_d, 0x38, n_elems));
        CK(cudaMemset(V_d, 0x38, n_elems));

        int smem = (FA_BR + 2 * FA_BC + FA_BR) * FA_STRIDE;
        int nqt = (sl + FA_BR - 1) / FA_BR;
        float scale = 1.0f / sqrtf((float)hd);

        for (int i = 0; i < 5; i++)
            fa62_kernel<<<bh * nqt, FA_THREADS, smem>>>(
                Q_d, K_d, V_d, O_d, sl, hd, 0, scale, 1.0f, 1.0f);
        CK(cudaDeviceSynchronize());

        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        int it = 50;
        cudaEventRecord(t0);
        for (int i = 0; i < it; i++)
            fa62_kernel<<<bh * nqt, FA_THREADS, smem>>>(
                Q_d, K_d, V_d, O_d, sl, hd, 0, scale, 1.0f, 1.0f);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        ms /= it;
        double flops = 4.0 * (double)bh * (double)sl * (double)sl * (double)hd;
        double tf = flops / (ms / 1000.0) / 1e12;
        printf("  bh=%d sl=%d hd=%d  time=%.3f ms  perf=%.1f TFLOPS\n",
               bh, sl, hd, ms, tf);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaFree(Q_d); cudaFree(K_d); cudaFree(V_d); cudaFree(O_d);
    }

    return 0;
}
