// =============================================================================
// FlashAttention v72 — EXPERIMENT: 4 blocks/SM via Br=64 + door 2 + single-K
// =============================================================================
// Test of 4 blocks/SM upper bound. Requires:
//   - Br=64 (was 128) → smQ 16 → 8 KB
//   - M_TILES=1 (with Br=64, 4 warps × 16 rows = 64)
//   - Single-K (smK[2] → smK)
//   - Door 2 V direct LDG (no smV slot)
//   - launch_bounds(128, 4) → ptxas reg cap = 128/thread
//
// SMEM 24.5 KB × 4 = 98 ≤ 100 ✓. Reg 128 × 4 × 128 = 65536 ✓ (tight).
// Predicted worse than v71 (-8 to -23%) due to Br=64 amortization loss.
// =============================================================================
// Production default for sm_120a (RTX PRO 6000 Blackwell). Replaces v68 = 220T
// peak with new ceiling 338T (+53%) on production-shape configs (≥256 blocks).
//
// Single 8 KB SMEM save vs v68 — by single-buffering V (drop smV[1]) — enables
// 2 blocks/SM (vs v68's 1). For grids ≥ 188 × 2 = 376 blocks, this halves wave
// count; for 256+ blocks (where waves go 2→1) gain is +51%; for 512+ (3→2)
// gain is +15-23%. For small grids (<188 blocks) v69 = v68 paritet (no harm).
//
// SMEM layout (48.5 KB):
//   smQ:    16 KB
//   smK[2]: 16 KB  (K stays double-buffered)
//   smV:     8 KB  (was 16 KB = double-buffered)
//   smV_T:  8.5 KB (padded stride 68 from v68, breaks 32-way write conflict)
//   smP overlaps smV after transpose_v (cur_V data extracted to smV_T)
//
// Sequencing change vs v68: V prefetch moves to END of iter (after smP read).
// K prefetch stays at MID-iter and still overlaps with compute. Cost: V load
// loses overlap, but v64 datapoint + v68 NCu (mem busy 22%) prove kernel is
// NOT memory-bound → V overlap loss ≤ 2% on all measured shapes.
//
// Why v69 > v68:
//   - Occupancy 8.33% (1 block/SM × 4 warps) → 16.67% (2 × 4 = 8 warps/SM)
//   - Directly hides MMA pipeline latency (was 1.06 cycles/inst floor in v68)
//   - Wave reduction on large grids (typical for batched LLM inference)
//
// Build: nvcc -gencode arch=compute_120a,code=sm_120a
// History: v66 → v68 → v69 (production 338T). v70/v71 dead ends.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define FA_BR 64                  // v72: was 128. 4 warps × M_TILES=1 × 16 rows = 64
#define FA_BC 64
#define FA_THREADS 128            // 4 warps (kept)
#define FA_STRIDE 128
#define M_TILES 1                 // v72: was 2. Halves per-thread reg state arrays.
#define SMV_T_STRIDE 68           // padded smV_T stride breaks 32-way conflict

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

// v68 padded swizzle for smV_T (stride = SMV_T_STRIDE = 68).
// Breaks 32-way write conflict in transpose_v: with stride 64 the
// lane-to-bank period was 128 B → all 32 lanes hit bank 0.
// With stride 68: lane_stride / 4 = 17, gcd(17, 32) = 1 → 32 distinct banks.
__device__ __forceinline__ int swz_byte_smvt(int row, int col_bytes)
{
    int chunk = col_bytes >> 4;
    int within = col_bytes & 15;
    return row * SMV_T_STRIDE + ((chunk ^ (row & 3)) << 4) + within;
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

// Hardware FP16x2 → FP8x2 conversion. cvt.rn.satfinite.e4m3x2.f16x2 is
// available on sm_89+ as a single PTX instruction.
__device__ __forceinline__ uint16_t fp16x2_to_e4m3x2(uint32_t h2)
{
    uint16_t out;
    asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
                 : "=h"(out) : "r"(h2));
    return out;
}

// v80b: restored transpose_v (4×4 byte transposes per thread) from v69 pattern.
// Reads smV[k_row][n_col] → writes smV_T[n_row][k_col] via 4 uint32_t per thread.
__device__ __forceinline__ void transpose_v(
    uint8_t *smV_T, const uint8_t *smV, int head_dim)
{
    constexpr int TILE = 4;
    int tiles_k = FA_BC / TILE;       // 16
    int tiles_n = head_dim / TILE;     // 32 for hd=128
    int total = tiles_k * tiles_n;     // 512
    for (int t = threadIdx.x; t < total; t += FA_THREADS)
    {
        int tk = t / tiles_n;
        int tn = t % tiles_n;
        int k0 = tk * TILE;
        int n0 = tn * TILE;
        uint32_t r0 = *(uint32_t *)&smV[swz_byte(k0 + 0, n0)];
        uint32_t r1 = *(uint32_t *)&smV[swz_byte(k0 + 1, n0)];
        uint32_t r2 = *(uint32_t *)&smV[swz_byte(k0 + 2, n0)];
        uint32_t r3 = *(uint32_t *)&smV[swz_byte(k0 + 3, n0)];
        uint32_t c0 = ((r0 >>  0) & 0xff)
                    | ((r1 <<  8) & 0xff00)
                    | ((r2 << 16) & 0xff0000)
                    | ((r3 << 24) & 0xff000000);
        uint32_t c1 = ((r0 >>  8) & 0xff)
                    | ((r1 <<  0) & 0xff00)
                    | ((r2 <<  8) & 0xff0000)
                    | ((r3 << 16) & 0xff000000);
        uint32_t c2 = ((r0 >> 16) & 0xff)
                    | ((r1 >>  8) & 0xff00)
                    | ((r2 <<  0) & 0xff0000)
                    | ((r3 <<  8) & 0xff000000);
        uint32_t c3 = ((r0 >> 24) & 0xff)
                    | ((r1 >> 16) & 0xff00)
                    | ((r2 >>  8) & 0xff0000)
                    | ((r3 <<  0) & 0xff000000);
        *(uint32_t *)&smV_T[swz_byte_smvt(n0 + 0, k0)] = c0;
        *(uint32_t *)&smV_T[swz_byte_smvt(n0 + 1, k0)] = c1;
        *(uint32_t *)&smV_T[swz_byte_smvt(n0 + 2, k0)] = c2;
        *(uint32_t *)&smV_T[swz_byte_smvt(n0 + 3, k0)] = c3;
    }
}

// v80b: launch_bounds(128, 3) → ptxas reg cap ≤170/thread for 3 blocks/SM.
// SMEM 32.5KB × 3 = 97.5KB ≤ 100KB. Trade 4 blocks/SM → 3 to enable V cp.async overlap.
__global__ void __launch_bounds__(FA_THREADS, 3)
    fa80b_kernel(
        const uint8_t *__restrict__ Q,
        const uint8_t *__restrict__ K,
        const uint8_t *__restrict__ V,
        __half *__restrict__ O,
        int seq_len, int head_dim, int causal, float scale,
        float qk_descale, float v_descale,
        int window)  // 0 = no window. >0 = causal sliding window.
{
    int nqt = (seq_len + FA_BR - 1) / FA_BR;
    int bh = blockIdx.x / nqt, qt = blockIdx.x % nqt, qs = qt * FA_BR;
    if (qs >= seq_len) return;
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    int gid = lane / 4, tid = lane % 4;
    int mrb = wid * 16;  // v72: M_TILES=1 → each warp owns 16 rows (was 32)

    extern __shared__ uint8_t raw[];
    uint8_t *smQ = raw;
    // v80b: single-K + smV (cp.async target) + smV_T (transpose target).
    // smP overlaps smK after Q·K^T compute.
    // Total = smQ(8K) + smK(8K) + smV(8K) + smV_T(8.5K) = 32.5KB. × 3 = 97.5KB ≤ 100KB cap.
    uint8_t *smK = smQ + FA_BR * FA_STRIDE;
    uint8_t *smV = smK + FA_BC * FA_STRIDE;
    uint8_t *smV_T = smV + FA_BC * FA_STRIDE;

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
    float rmax[M_TILES][2];  rmax[0][0] = -1e30f; rmax[0][1] = -1e30f;
    float rsexp[M_TILES][2]; rsexp[0][0] = 0.f;   rsexp[0][1] = 0.f;
    int nkv = (seq_len + FA_BC - 1) / FA_BC;
    int kv_max_blocks = causal ? ((qs + FA_BR - 1) / FA_BC + 1) : nkv;
    if (kv_max_blocks > nkv) kv_max_blocks = nkv;
    // v72+window: sliding-window lower bound.
    int kv_min_blocks = 0;
    if (window > 0 && qs + 1 > window) {
        kv_min_blocks = (qs - window + 1) / FA_BC;
    }

    // v80b: pre-load K + V both via cp.async (V goes to smV, will be transposed at iter start).
    load_tile_fp8(smK, Kh, kv_min_blocks * FA_BC, FA_BC, seq_len, head_dim);
    load_tile_fp8(smV, Vh, kv_min_blocks * FA_BC, FA_BC, seq_len, head_dim);
    cpa_commit();

    for (int kv = kv_min_blocks; kv < kv_max_blocks; kv++)
    {
        int kvs = kv * FA_BC;

        // v80b: wait for K and V cp.async to drain.
        cpa_wait<0>();
        __syncthreads();

        // v80b: transpose V from smV → smV_T (smV is freed after this, can be cp.async'd for kv+1).
        transpose_v(smV_T, smV, head_dim);
        __syncthreads();

        // v72/v80b: smP overlaps smK (smK dead after Q·K^T below).
        uint8_t *smP = smK;

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
                uint32_t b0 = *(uint32_t *)&smK[swz_byte(br + gid, cl)];
                uint32_t b1 = *(uint32_t *)&smK[swz_byte(br + gid, ch)];
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

        // v80b: V[kv+1] cp.async → smV. Overlaps with softmax + smP STS + PV MMA.
        // smV slot is dead after transpose_v completed at start of iter (smV_T holds the data).
        // mma.sync for QK has issued, K LDS data already in b0/b1 registers — smK[buf] not needed for QK anymore.
        // Next iter's transpose_v will read smV (containing V[kv+1]).
        {
            int kv_p = kv + 1;
            int rows_p = (kv_p < kv_max_blocks) ? FA_BC : 0;
            load_tile_fp8(smV, Vh, kv_p * FA_BC, rows_p, seq_len, head_dim);
        }
        cpa_commit();

        // Sr[nt][mi][r] — float for softmax math
        float Sr[8][M_TILES][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            // v80a Lever 4: pre-multiply by log2(e). Sr/rmax/nm in LOG2 space.
            float fs = scale * qk_descale * 1.4426950408889634f;
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
                int kmin0 = (window > 0 && gq0 + 1 > window) ? (gq0 - window + 1) : 0;
                int kmin8 = (window > 0 && gq8 + 1 > window) ? (gq8 - window + 1) : 0;
#pragma unroll
                for (int nt = 0; nt < 8; nt++)
                {
                    int gk0 = kvs + nt * 8 + tid * 2, gk1 = gk0 + 1;
                    if (gk0 > gq0) Sr[nt][mi][0] = -1e30f;
                    if (gk1 > gq0) Sr[nt][mi][1] = -1e30f;
                    if (gk0 > gq8) Sr[nt][mi][2] = -1e30f;
                    if (gk1 > gq8) Sr[nt][mi][3] = -1e30f;
                    // Sliding window lower bound:
                    if (gk0 < kmin0) Sr[nt][mi][0] = -1e30f;
                    if (gk1 < kmin0) Sr[nt][mi][1] = -1e30f;
                    if (gk0 < kmin8) Sr[nt][mi][2] = -1e30f;
                    if (gk1 < kmin8) Sr[nt][mi][3] = -1e30f;
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
            // v80a Lever 4: log2-space → exp2f instead of __expf.
            rsc[mi][0] = exp2f(rmax[mi][0] - nm[mi][0]);
            rsc[mi][1] = exp2f(rmax[mi][1] - nm[mi][1]);
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

        // v80a Step 1: P stored as __half2 (16 b32 vs 64 FP32).
        // Skips float→half conversion in smP STS path.
        float ns[M_TILES][2]; ns[0][0] = 0.f; ns[0][1] = 0.f;
        __half2 P_top[8][M_TILES], P_bot[8][M_TILES];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                float d0 = Sr[nt][mi][0] - rmax[mi][0], d1 = Sr[nt][mi][1] - rmax[mi][0];
                float d2 = Sr[nt][mi][2] - rmax[mi][1], d3 = Sr[nt][mi][3] - rmax[mi][1];
                __half2 d_top = __floats2half2_rn(d0, d1);
                __half2 d_bot = __floats2half2_rn(d2, d3);
                uint32_t p_top_u, p_bot_u;
                asm("ex2.approx.f16x2 %0, %1;"
                    : "=r"(p_top_u) : "r"(*reinterpret_cast<uint32_t *>(&d_top)));
                asm("ex2.approx.f16x2 %0, %1;"
                    : "=r"(p_bot_u) : "r"(*reinterpret_cast<uint32_t *>(&d_bot)));
                P_top[nt][mi] = *reinterpret_cast<__half2 *>(&p_top_u);
                P_bot[nt][mi] = *reinterpret_cast<__half2 *>(&p_bot_u);
                ns[mi][0] += __low2float(P_top[nt][mi]) + __high2float(P_top[nt][mi]);
                ns[mi][1] += __low2float(P_bot[nt][mi]) + __high2float(P_bot[nt][mi]);
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
                // v80a Step 1: P_top/P_bot already __half2 — no float→half conversion.
                uint16_t fp8x2_top = fp16x2_to_e4m3x2(*reinterpret_cast<uint32_t *>(&P_top[nt][mi]));
                uint16_t fp8x2_bot = fp16x2_to_e4m3x2(*reinterpret_cast<uint32_t *>(&P_bot[nt][mi]));
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
                uint32_t b0 = *(uint32_t *)&smV_T[swz_byte_smvt(br + gid, cl)];
                uint32_t b1 = *(uint32_t *)&smV_T[swz_byte_smvt(br + gid, ch)];
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

        // v80b: end-of-iter — K cp.async only. V was prefetched MID-iter (after QK MMA).
        // Branch-free Lever 3: rows_p=0 on last iter → load_tile_fp8 inner loop no-op.
        int kv_p_k = kv + 1;
        int rows_p_k = (kv_p_k < kv_max_blocks) ? FA_BC : 0;
        load_tile_fp8(smK, Kh, kv_p_k * FA_BC, rows_p_k, seq_len, head_dim);
        cpa_commit();
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
    float *O_out, int bh, int sl, int hd, int causal, int window = 0)
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
            int kv_min = (window > 0 && q + 1 > window) ? (q - window + 1) : 0;
            float *P = (float *)malloc(sizeof(float) * sl);
            float rmax = -1e30f;
            for (int k = kv_min; k < kv_max; k++)
            {
                float s = 0;
                for (int d = 0; d < hd; d++)
                    s += e4m3_to_float(Qh[q * hd + d]) * e4m3_to_float(Kh[k * hd + d]);
                P[k] = s * scale;
                if (P[k] > rmax) rmax = P[k];
            }
            float rsum = 0;
            for (int k = kv_min; k < kv_max; k++)
            {
                P[k] = expf(P[k] - rmax);
                rsum += P[k];
            }
            for (int k = kv_min; k < kv_max; k++) P[k] /= rsum;
            for (int d = 0; d < hd; d++)
            {
                float o = 0;
                for (int k = kv_min; k < kv_max; k++)
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
    printf("=== FA v80b — EXPERIMENT: 4 blocks/SM via Br=64 + door 2 + single-K ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n", p.name, p.multiProcessorCount, clk / 1000);

    // Report actual kernel attributes (real SMEM use + reg count).
    cudaFuncAttributes attr;
    CK(cudaFuncGetAttributes(&attr, fa80b_kernel));
    printf("Kernel attrs: numRegs=%d  binSize=%d  sharedSizeBytes(static)=%zu\n",
           attr.numRegs, attr.binaryVersion, attr.sharedSizeBytes);
    // v80b: smQ + smK + smV + smV_T. SMEM = 32.5KB. Target 3 blocks/SM.
    int smem_hd128 = FA_BR * FA_STRIDE + FA_BC * FA_STRIDE + FA_BC * FA_STRIDE + 128 * SMV_T_STRIDE;
    printf("Dynamic SMEM (hd=128): v80b=%d B (%.2f KB)\n",
           smem_hd128, smem_hd128 / 1024.0);
    printf("3 blocks × %.2f KB = %.2f KB. Reg cap = 65536/(3*128) = 170/thread.\n",
           smem_hd128 / 1024.0, 3 * smem_hd128 / 1024.0);
    printf("Per-thread reg check: numRegs * 3 * 128 = %d (≤ 65536 for 3 blocks)\n\n",
           attr.numRegs * 3 * 128);

    printf("--- Correctness vs CPU FP8-roundtripped reference ---\n");
    // 5th col = window (0 = none). Mirror v69 configs.
    int configs[][5] = {
        {1, 64,  128, 0, 0},
        {1, 128, 128, 0, 0},
        {1, 256, 128, 0, 0},
        {1, 512, 128, 0, 0},
        {2, 256, 128, 1, 0},
        // Sliding-window edge cases:
        {1, 256, 128, 1, 64},
        {1, 256, 128, 1, 100},
        {1, 300, 128, 1, 96},
    };
    for (auto &c : configs)
    {
        int bh = c[0], sl = c[1], hd = c[2], ca = c[3], wnd = c[4];
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
        cpu_attention_fp8(Qq, Kq, Vq, O_ref, bh, sl, hd, ca, wnd);

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

        // v72: smQ(Br*Stride=64*128=8K) + smK(8K) + smV_T(128*68=8.5K) = 24.5 KB.
        // 4 blocks × 24.5 = 98 KB ≤ 100 KB cap → 4 blocks/SM target.
        int smem = FA_BR * FA_STRIDE + FA_BC * FA_STRIDE + FA_BC * FA_STRIDE + hd * SMV_T_STRIDE;
        CK(cudaFuncSetAttribute(fa80b_kernel,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        int nqt = (sl + FA_BR - 1) / FA_BR;
        float scale = 1.0f / sqrtf((float)hd);
        fa80b_kernel<<<bh * nqt, FA_THREADS, smem>>>(
            Q_d, K_d, V_d, O_d, sl, hd, ca, scale, 1.0f, 1.0f, wnd);
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
        printf("  bh=%d sl=%d hd=%d ca=%d wnd=%d  max_diff=%.4f errs=%d → %s\n",
               bh, sl, hd, ca, wnd, mx, errs, errs == 0 ? "PASS" : "FAIL");

        free(Qf); free(Kf); free(Vf);
        free(Qq); free(Kq); free(Vq);
        free(O_ref); free(O_cpu);
        cudaFree(Q_d); cudaFree(K_d); cudaFree(V_d); cudaFree(O_d);
    }

    printf("\n--- Performance ---\n");
    int bench_configs[][4] = {
        {4, 1024, 128, 0},
        {4, 2048, 128, 0},
        {8, 2048, 128, 0},
        {4, 4096, 128, 0},
        {8, 4096, 128, 0},
        {16, 2048, 128, 0},
        {16, 4096, 128, 0},
        {32, 2048, 128, 0},
        // Sliding window (Gemma 3 local-layer regime):
        {4, 8192, 128, 1024},
        {4, 4096, 128, 1024},
        {8, 8192, 128, 1024},
    };
    for (auto &c : bench_configs)
    {
        int bh = c[0], sl = c[1], hd = c[2], wnd = c[3];
        int ca_bench = (wnd > 0) ? 1 : 0;
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

        // v72: smQ(Br*Stride=64*128=8K) + smK(8K) + smV_T(128*68=8.5K) = 24.5 KB.
        // 4 blocks × 24.5 = 98 KB ≤ 100 KB cap → 4 blocks/SM target.
        int smem = FA_BR * FA_STRIDE + FA_BC * FA_STRIDE + FA_BC * FA_STRIDE + hd * SMV_T_STRIDE;
        int nqt = (sl + FA_BR - 1) / FA_BR;
        float scale = 1.0f / sqrtf((float)hd);

        for (int i = 0; i < 5; i++)
            fa80b_kernel<<<bh * nqt, FA_THREADS, smem>>>(
                Q_d, K_d, V_d, O_d, sl, hd, ca_bench, scale, 1.0f, 1.0f, wnd);
        CK(cudaDeviceSynchronize());

        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        int it = 50;
        cudaEventRecord(t0);
        for (int i = 0; i < it; i++)
            fa80b_kernel<<<bh * nqt, FA_THREADS, smem>>>(
                Q_d, K_d, V_d, O_d, sl, hd, ca_bench, scale, 1.0f, 1.0f, wnd);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        ms /= it;
        double sl_eff = (wnd > 0) ? (double)wnd : (double)sl;
        if (ca_bench && wnd == 0) sl_eff = (double)sl / 2.0;
        double flops = 4.0 * (double)bh * (double)sl * sl_eff * (double)hd;
        double tf = flops / (ms / 1000.0) / 1e12;
        printf("  bh=%d sl=%d hd=%d ca=%d wnd=%d  time=%.3f ms  perf=%.1f TFLOPS (effective)\n",
               bh, sl, hd, ca_bench, wnd, ms, tf);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaFree(Q_d); cudaFree(K_d); cudaFree(V_d); cudaFree(O_d);
    }

    return 0;
}
