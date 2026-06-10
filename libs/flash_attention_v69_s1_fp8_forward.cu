// =============================================================================
// FlashAttention v69_s1 — Stage 1: drop smV_T + byte-gather V from smV
// =============================================================================
// ISOLATED CORRECTNESS+CONFLICT test for A-aggressive Stage 1:
//   - Drop smV_T allocation
//   - Drop transpose_v call
//   - In P·V MMA loop, build (b0, b1) via byte-gather from cur_V instead of
//     reading uint32 from smV_T.
//   - smP stays (no longer overlapping cur_V since cur_V is now read in P·V
//     loop). smP becomes a separate 8KB slot.
//   - Net SMEM ~unchanged (drop smV_T -8KB, gain separate smP +8KB). Still
//     1 block/SM. Occupancy NOT improved yet — that comes in Stage 3.
//
// Diagnostic targets:
//   (a) Correctness: max_diff must match v68 (no off-by-one in byte order).
//   (b) NCu: V byte-gather LDS pattern must NOT introduce new 32-way load
//       conflict. The risk: smV stride 128 B may collide for 32 lanes.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define FA_BR 128
#define FA_BC 64
#define FA_THREADS 128   // 4 warps × M_TILES=2 × 16 rows = Br=128
#define FA_STRIDE 128
#define M_TILES 2        // each warp owns 2 m16 sub-tiles in M direction
// v69_s1: smV_T removed; SMV_T_STRIDE no longer used.

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

// Swizzle for smP (stride = FA_BC = 64 bytes per row).
// Still used by P quantize writes and P read in P·V loop.
__device__ __forceinline__ int swz_byte_bc(int row, int col_bytes)
{
    int chunk = col_bytes >> 4;
    int within = col_bytes & 15;
    return row * FA_BC + ((chunk ^ (row & 3)) << 4) + within;
}

// v69_s1: swz_byte_smvt removed (smV_T gone).

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

// Hardware FP16x2 → FP8x2 conversion. cvt.rn.satfinite.e4m3x2.f16x2 is
// available on sm_89+ as a single PTX instruction.
__device__ __forceinline__ uint16_t fp16x2_to_e4m3x2(uint32_t h2)
{
    uint16_t out;
    asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
                 : "=h"(out) : "r"(h2));
    return out;
}

// v69_s1: transpose_v removed (no smV_T).

__global__ void __launch_bounds__(FA_THREADS, 2)
    fa69_s1_kernel(
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
    int mrb = wid * 32;  // M_TILES=2 → each warp owns 32 rows

    extern __shared__ uint8_t raw[];
    uint8_t *smQ = raw;
    uint8_t *smK[2] = {
        smQ + FA_BR * FA_STRIDE,
        smQ + FA_BR * FA_STRIDE + FA_BC * FA_STRIDE,
    };
    uint8_t *smV[2] = {
        smK[1] + FA_BC * FA_STRIDE,
        smK[1] + 2 * FA_BC * FA_STRIDE,
    };
    // v69_s1: smV_T removed. smP is its own slot now (no longer overlaps cur_V,
    // because cur_V is read in P·V loop via byte-gather).
    uint8_t *smP = smV[1] + FA_BC * FA_STRIDE;

    int hs = seq_len * head_dim;
    const uint8_t *Qh = Q + bh * hs;
    const uint8_t *Kh = K + bh * hs;
    const uint8_t *Vh = V + bh * hs;
    __half *Oh = O + bh * hs;

    load_tile_fp8(smQ, Qh, qs, FA_BR, seq_len, head_dim);
    cpa_commit();
    cpa_wait<0>();
    __syncthreads();

    // Qr[ks][mi][r] — 4 k-steps × 2 M-tiles × 4 uint32 (m16k32 A operand)
    uint32_t Qr[4][M_TILES][4];
#pragma unroll
    for (int ks = 0; ks < 4; ks++)
    {
        int k_off = ks * 32;
        int cl = k_off + tid * 4;
        int ch = cl + 16;
#pragma unroll
        for (int mi = 0; mi < M_TILES; mi++)
        {
            int mr = mrb + mi * 16;
            int g0 = mr + gid, g8 = g0 + 8;
            Qr[ks][mi][0] = *(uint32_t *)&smQ[swz_byte(g0, cl)];
            Qr[ks][mi][1] = *(uint32_t *)&smQ[swz_byte(g8, cl)];
            Qr[ks][mi][2] = *(uint32_t *)&smQ[swz_byte(g0, ch)];
            Qr[ks][mi][3] = *(uint32_t *)&smQ[swz_byte(g8, ch)];
        }
    }

    // Or_p[nt][mi][r] — 16 N-tiles × 2 M-tiles × 2 packed uint32 (m16n8 D)
    uint32_t Or_p[16][M_TILES][2];
#pragma unroll
    for (int t = 0; t < 16; t++)
#pragma unroll
        for (int mi = 0; mi < M_TILES; mi++)
            Or_p[t][mi][0] = Or_p[t][mi][1] = 0u;

    // Per-row state: [mi][side] where side=0 is gid row, side=1 is gid+8 row
    float rmax[M_TILES][2] = {{-1e30f, -1e30f}, {-1e30f, -1e30f}};
    float rsexp[M_TILES][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    int nkv = (seq_len + FA_BC - 1) / FA_BC;
    int kv_max_blocks = causal ? ((qs + FA_BR - 1) / FA_BC + 1) : nkv;
    if (kv_max_blocks > nkv) kv_max_blocks = nkv;

    // Pre-load iter 0 only (commit g0). Iter i prefetches iter i+1 → other bank.
    load_tile_fp8(smK[0], Kh, 0, FA_BC, seq_len, head_dim);
    load_tile_fp8(smV[0], Vh, 0, FA_BC, seq_len, head_dim);
    cpa_commit();

    for (int kv = 0; kv < kv_max_blocks; kv++)
    {
        int kvs = kv * FA_BC;
        int buf = kv & 1;
        uint8_t *cur_K = smK[buf];
        uint8_t *cur_V = smV[buf];
        uint8_t *nxt_K = smK[buf ^ 1];
        uint8_t *nxt_V = smV[buf ^ 1];

        // Wait until current iter's K, V land.
        cpa_wait<0>();
        __syncthreads();

        // v69_s1: no transpose_v. cur_V holds V[k][n] (untransposed); P·V loop
        // reads V via byte-gather from cur_V. smP has its own slot (declared above).

        // Prefetch iter kv+1's data into the OTHER bank.
        if (kv + 1 < kv_max_blocks)
        {
            load_tile_fp8(nxt_K, Kh, (kv + 1) * FA_BC, FA_BC, seq_len, head_dim);
            load_tile_fp8(nxt_V, Vh, (kv + 1) * FA_BC, FA_BC, seq_len, head_dim);
        }
        cpa_commit();

        // S = Q · Kᵀ — K B-operand loaded once per (nt, ks), reused across mi.
        uint32_t Sr_p[8][M_TILES][2];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
                Sr_p[nt][mi][0] = Sr_p[nt][mi][1] = 0u;
#pragma unroll
        for (int ks = 0; ks < 4; ks++)
        {
            int k_off = ks * 32;
            int cl = k_off + tid * 4, ch = cl + 16;
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
            {
                int br = nt * 8;
                uint32_t b0 = *(uint32_t *)&cur_K[swz_byte(br + gid, cl)];
                uint32_t b1 = *(uint32_t *)&cur_K[swz_byte(br + gid, ch)];
#pragma unroll
                for (int mi = 0; mi < M_TILES; mi++)
                {
                    mma_fp8_f16(Sr_p[nt][mi][0], Sr_p[nt][mi][1],
                                Qr[ks][mi][0], Qr[ks][mi][1],
                                Qr[ks][mi][2], Qr[ks][mi][3],
                                b0, b1, Sr_p[nt][mi][0], Sr_p[nt][mi][1]);
                }
            }
        }

        // Sr[nt][mi][r] — float for softmax math
        float Sr[8][M_TILES][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            float fs = scale * qk_descale;
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                __half2 v0 = *reinterpret_cast<__half2 *>(&Sr_p[nt][mi][0]);
                __half2 v1 = *reinterpret_cast<__half2 *>(&Sr_p[nt][mi][1]);
                Sr[nt][mi][0] = __half2float(__low2half(v0)) * fs;
                Sr[nt][mi][1] = __half2float(__high2half(v0)) * fs;
                Sr[nt][mi][2] = __half2float(__low2half(v1)) * fs;
                Sr[nt][mi][3] = __half2float(__high2half(v1)) * fs;
            }
        }

        if (causal)
        {
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                int gq0 = qs + mrb + mi * 16 + gid, gq8 = gq0 + 8;
#pragma unroll
                for (int nt = 0; nt < 8; nt++)
                {
                    int gk0 = kvs + nt * 8 + tid * 2, gk1 = gk0 + 1;
                    if (gk0 > gq0) Sr[nt][mi][0] = -1e30f;
                    if (gk1 > gq0) Sr[nt][mi][1] = -1e30f;
                    if (gk0 > gq8) Sr[nt][mi][2] = -1e30f;
                    if (gk1 > gq8) Sr[nt][mi][3] = -1e30f;
                    if (gq0 >= seq_len) Sr[nt][mi][0] = Sr[nt][mi][1] = -1e30f;
                    if (gq8 >= seq_len) Sr[nt][mi][2] = Sr[nt][mi][3] = -1e30f;
                    if (gk0 >= seq_len) Sr[nt][mi][0] = Sr[nt][mi][2] = -1e30f;
                    if (gk1 >= seq_len) Sr[nt][mi][1] = Sr[nt][mi][3] = -1e30f;
                }
            }
        }

        // Per-tile softmax: max, rescale Or, exp+sum.
        float nm[M_TILES][2];
        float rsc[M_TILES][2];
#pragma unroll
        for (int mi = 0; mi < M_TILES; mi++)
        {
            nm[mi][0] = -1e30f; nm[mi][1] = -1e30f;
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
            {
                nm[mi][0] = fmaxf(nm[mi][0], fmaxf(Sr[nt][mi][0], Sr[nt][mi][1]));
                nm[mi][1] = fmaxf(nm[mi][1], fmaxf(Sr[nt][mi][2], Sr[nt][mi][3]));
            }
            nm[mi][0] = fmaxf(nm[mi][0], __shfl_xor_sync(0xffffffff, nm[mi][0], 1));
            nm[mi][0] = fmaxf(nm[mi][0], __shfl_xor_sync(0xffffffff, nm[mi][0], 2));
            nm[mi][1] = fmaxf(nm[mi][1], __shfl_xor_sync(0xffffffff, nm[mi][1], 1));
            nm[mi][1] = fmaxf(nm[mi][1], __shfl_xor_sync(0xffffffff, nm[mi][1], 2));
            nm[mi][0] = fmaxf(nm[mi][0], rmax[mi][0]);
            nm[mi][1] = fmaxf(nm[mi][1], rmax[mi][1]);
            rsc[mi][0] = __expf(rmax[mi][0] - nm[mi][0]);
            rsc[mi][1] = __expf(rmax[mi][1] - nm[mi][1]);
        }

        // Rescale Or by per-(mi,side) factor.
#pragma unroll
        for (int mi = 0; mi < M_TILES; mi++)
        {
            __half2 h2_rsc0 = __float2half2_rn(rsc[mi][0]);
            __half2 h2_rsc1 = __float2half2_rn(rsc[mi][1]);
#pragma unroll
            for (int t = 0; t < 16; t++)
            {
                __half2 v0 = *reinterpret_cast<__half2 *>(&Or_p[t][mi][0]);
                __half2 v1 = *reinterpret_cast<__half2 *>(&Or_p[t][mi][1]);
                v0 = __hmul2(v0, h2_rsc0);
                v1 = __hmul2(v1, h2_rsc1);
                Or_p[t][mi][0] = *reinterpret_cast<uint32_t *>(&v0);
                Or_p[t][mi][1] = *reinterpret_cast<uint32_t *>(&v1);
            }
        }
#pragma unroll
        for (int mi = 0; mi < M_TILES; mi++) {
            rmax[mi][0] = nm[mi][0];
            rmax[mi][1] = nm[mi][1];
        }

        // Compute P = exp(S - rmax), accumulate row sum.
        float ns[M_TILES][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
        float P_local[8][M_TILES][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                float d0 = Sr[nt][mi][0] - rmax[mi][0], d1 = Sr[nt][mi][1] - rmax[mi][0];
                float d2 = Sr[nt][mi][2] - rmax[mi][1], d3 = Sr[nt][mi][3] - rmax[mi][1];
                float p0 = __expf(d0), p1 = __expf(d1);
                float p2 = __expf(d2), p3 = __expf(d3);
                ns[mi][0] += p0 + p1;
                ns[mi][1] += p2 + p3;
                P_local[nt][mi][0] = p0;
                P_local[nt][mi][1] = p1;
                P_local[nt][mi][2] = p2;
                P_local[nt][mi][3] = p3;
            }
        }
#pragma unroll
        for (int mi = 0; mi < M_TILES; mi++)
        {
            ns[mi][0] += __shfl_xor_sync(0xffffffff, ns[mi][0], 1);
            ns[mi][0] += __shfl_xor_sync(0xffffffff, ns[mi][0], 2);
            ns[mi][1] += __shfl_xor_sync(0xffffffff, ns[mi][1], 1);
            ns[mi][1] += __shfl_xor_sync(0xffffffff, ns[mi][1], 2);
            rsexp[mi][0] = rsexp[mi][0] * rsc[mi][0] + ns[mi][0];
            rsexp[mi][1] = rsexp[mi][1] * rsc[mi][1] + ns[mi][1];
        }

        __syncthreads();

        // Quantize P → smP for both M-tiles.
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            int col0 = nt * 8 + tid * 2, col1 = col0 + 1;
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                int mr = mrb + mi * 16;
                int row0 = mr + gid, row8 = mr + gid + 8;
                __half2 h2_top = __halves2half2(__float2half(P_local[nt][mi][0]),
                                                __float2half(P_local[nt][mi][1]));
                __half2 h2_bot = __halves2half2(__float2half(P_local[nt][mi][2]),
                                                __float2half(P_local[nt][mi][3]));
                uint16_t fp8x2_top = fp16x2_to_e4m3x2(*reinterpret_cast<uint32_t *>(&h2_top));
                uint16_t fp8x2_bot = fp16x2_to_e4m3x2(*reinterpret_cast<uint32_t *>(&h2_bot));
                *(uint16_t *)&smP[swz_byte_bc(row0, col0)] = fp8x2_top;
                *(uint16_t *)&smP[swz_byte_bc(row8, col0)] = fp8x2_bot;
            }
        }
        __syncthreads();

        // O += P · V — V B-operand loaded once per (nt, ks), reused across mi.
#pragma unroll
        for (int ks = 0; ks < 2; ks++)
        {
            int k_off = ks * 32;
            int cl = k_off + tid * 4, ch = cl + 16;
            // Load P A-operand for both M-tiles.
            uint32_t Pr[M_TILES][4];
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                int mr = mrb + mi * 16;
                int g0 = mr + gid, g8 = g0 + 8;
                Pr[mi][0] = *(uint32_t *)&smP[swz_byte_bc(g0, cl)];
                Pr[mi][1] = *(uint32_t *)&smP[swz_byte_bc(g8, cl)];
                Pr[mi][2] = *(uint32_t *)&smP[swz_byte_bc(g0, ch)];
                Pr[mi][3] = *(uint32_t *)&smP[swz_byte_bc(g8, ch)];
            }
#pragma unroll
            for (int nt = 0; nt < 16; nt++)
            {
                int br = nt * 8;
                // v69_s1: byte-gather V from cur_V (no smV_T transpose buffer).
                // b0 = 4 FP8 bytes at V[K=cl..cl+3, N=br+gid].
                int nv = br + gid;
                uint32_t b0 = ((uint32_t)cur_V[swz_byte(cl + 0, nv)])
                            | ((uint32_t)cur_V[swz_byte(cl + 1, nv)] << 8)
                            | ((uint32_t)cur_V[swz_byte(cl + 2, nv)] << 16)
                            | ((uint32_t)cur_V[swz_byte(cl + 3, nv)] << 24);
                uint32_t b1 = ((uint32_t)cur_V[swz_byte(ch + 0, nv)])
                            | ((uint32_t)cur_V[swz_byte(ch + 1, nv)] << 8)
                            | ((uint32_t)cur_V[swz_byte(ch + 2, nv)] << 16)
                            | ((uint32_t)cur_V[swz_byte(ch + 3, nv)] << 24);
#pragma unroll
                for (int mi = 0; mi < M_TILES; mi++)
                {
                    mma_fp8_f16(Or_p[nt][mi][0], Or_p[nt][mi][1],
                                Pr[mi][0], Pr[mi][1], Pr[mi][2], Pr[mi][3],
                                b0, b1, Or_p[nt][mi][0], Or_p[nt][mi][1]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mi = 0; mi < M_TILES; mi++)
    {
        float li0 = (rsexp[mi][0] > 0) ? v_descale / rsexp[mi][0] : 0.0f;
        float li1 = (rsexp[mi][1] > 0) ? v_descale / rsexp[mi][1] : 0.0f;
        int mr = mrb + mi * 16;
        int gr0 = qs + mr + gid, gr8 = gr0 + 8;
#pragma unroll
        for (int nt = 0; nt < 16; nt++)
        {
            int c0 = nt * 8 + tid * 2, c1 = c0 + 1;
            __half2 v0 = *reinterpret_cast<__half2 *>(&Or_p[nt][mi][0]);
            __half2 v1 = *reinterpret_cast<__half2 *>(&Or_p[nt][mi][1]);
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
    printf("=== FA v69_s1 — Stage 1: drop smV_T + byte-gather V ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n", p.name, p.multiProcessorCount, clk / 1000);

    // Report actual kernel attributes (real SMEM use + reg count).
    cudaFuncAttributes attr;
    CK(cudaFuncGetAttributes(&attr, fa69_s1_kernel));
    printf("Kernel attrs: numRegs=%d  binSize=%d  sharedSizeBytes(static)=%zu\n",
           attr.numRegs, attr.binaryVersion, attr.sharedSizeBytes);
    int smem_hd128 = (FA_BR + 2 * FA_BC + 2 * FA_BC) * FA_STRIDE + FA_BR * FA_BC;
    int smem_v68   = (FA_BR + 2 * FA_BC + 2 * FA_BC) * FA_STRIDE + 128 * 68;
    printf("Dynamic SMEM (hd=128): v69_s1=%d B (%.2f KB) vs v68=%d B (delta=%+d B)\n\n",
           smem_hd128, smem_hd128 / 1024.0, smem_v68, smem_hd128 - smem_v68);

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

        // v69_s1: smQ + 2×smK + 2×smV + smP (own slot, no smV_T)
        // For hd=128: 16384 + 16384 + 16384 + (FA_BR * FA_BC = 8192) = 57344 B
        int smem = (FA_BR + 2 * FA_BC + 2 * FA_BC) * FA_STRIDE + FA_BR * FA_BC;
        CK(cudaFuncSetAttribute(fa69_s1_kernel,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        int nqt = (sl + FA_BR - 1) / FA_BR;
        float scale = 1.0f / sqrtf((float)hd);
        fa69_s1_kernel<<<bh * nqt, FA_THREADS, smem>>>(
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

        // v69_s1: smQ + 2×smK + 2×smV + smP (own slot, no smV_T)
        // For hd=128: 16384 + 16384 + 16384 + (FA_BR * FA_BC = 8192) = 57344 B
        int smem = (FA_BR + 2 * FA_BC + 2 * FA_BC) * FA_STRIDE + FA_BR * FA_BC;
        int nqt = (sl + FA_BR - 1) / FA_BR;
        float scale = 1.0f / sqrtf((float)hd);

        for (int i = 0; i < 5; i++)
            fa69_s1_kernel<<<bh * nqt, FA_THREADS, smem>>>(
                Q_d, K_d, V_d, O_d, sl, hd, 0, scale, 1.0f, 1.0f);
        CK(cudaDeviceSynchronize());

        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        int it = 50;
        cudaEventRecord(t0);
        for (int i = 0; i < it; i++)
            fa69_s1_kernel<<<bh * nqt, FA_THREADS, smem>>>(
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
