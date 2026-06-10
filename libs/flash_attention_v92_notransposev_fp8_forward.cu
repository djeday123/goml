// =============================================================================
// FlashAttention v87 — FP8 Forward, hd=64 + Option C: explicit ks-batched MMA
// =============================================================================
// v81 base. ONE structural change in both QK and PV:
//   Replace `#pragma unroll for ks { ... }` with two explicit blocks
//   { ks=0 all-nt batch } { ks=1 all-nt batch }
//
// Purpose: force compiler to see "ks=0 issues 16 MMA, then ks=1 issues 16 MMA"
// pattern. Gap between same-accumulator MMA pairs (Sr_p[nt][mi] from ks=0 to
// ks=1) = 15 ops. If MMA latency ≤ 15 cycles, wait should be fully hidden.
//
// If wait stall stays ≈ 28% after this reorder → compiler was already doing
// optimal scheduling, and the wait comes from inherent MMA latency that can't
// be reduced without extra accumulators (Option A/B, +32 regs, breaks LB=3).
//
// Reg cost: ZERO. Same Sr_p/Or_p accumulators. Same Qr. Pr cloned per-batch
// (Pr0, Pr1) — independent SSA, compiler may merge or not.
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
// History: v66 baseline → v68 conflict fix (smV_T padding) → v69 single-V.
//          v67 TMA rejected. v69_singleV experiment kept as reference.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <utility>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define FA_BR 128
#define FA_BC 64
#define FA_THREADS 128   // 4 warps × M_TILES=2 × 16 rows = Br=128
#define FA_STRIDE 64     // hd=64 specialized: smQ/smK/smV row stride = head_dim
#define FA_HD 64         // hardcoded head_dim
#define N_TILES_PV 8     // FA_HD / 8 = output N-tile count (was 16 for hd=128)
#define KS_QK 2          // FA_HD / 32 = QK^T outer ks steps (was 4 for hd=128)
#define M_TILES 2        // each warp owns 2 m16 sub-tiles in M direction
#define SMV_T_STRIDE 68  // padded smV_T row stride to break 32-way conflict
                          // (64 + 4: gcd(17, 32) = 1 → 32 lanes hit 32 banks)

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
    // hd=64: row stride 64 = 4 × 16B chunks → period 4 (was 8 for hd=128)
    int chunk = col_bytes >> 4;
    int within = col_bytes & 15;
    return row * FA_STRIDE + ((chunk ^ (row & 3)) << 4) + within;
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

// Transpose smV [seq_k=Bc, head_dim] → smV_T [head_dim, seq_k=Bc].
// Once per K-block iter. After this, V B-operand reads use the same fast
// scalar uint32_t pattern as K (no byte-gather).
//
// Each thread copies a 4×4 tile: 4 consecutive k-rows × 4 consecutive n-cols.
// Reads as uint32_t (4 contiguous n-bytes per k-row), reassembles via
// bytewise shuffle, writes as uint32_t into the transposed layout.
__device__ __forceinline__ void transpose_v(
    uint8_t *smV_T, const uint8_t *smV, int head_dim)
{
    // Layout: smV[k_row * FA_STRIDE + n_col] (swizzled by swz_byte)
    //         smV_T[n_row * FA_STRIDE + k_col] (swizzled by swz_byte)
    // We do 16 4x4 transposes per thread (covers 64 rows × 64 cols = 4096 elems).
    constexpr int TILE = 4;
    int tiles_k = FA_BC / TILE;       // 16
    int tiles_n = head_dim / TILE;     // 32 for hd=128
    int total = tiles_k * tiles_n;     // 512
    for (int t = threadIdx.x; t < total; t += FA_THREADS)
    {
        int tk = t / tiles_n;          // 0..15
        int tn = t % tiles_n;          // 0..31
        int k0 = tk * TILE;
        int n0 = tn * TILE;
        // Load 4 uint32_t (one per k-row) = 4x4 fp8 block
        uint32_t r0 = *(uint32_t *)&smV[swz_byte(k0 + 0, n0)];
        uint32_t r1 = *(uint32_t *)&smV[swz_byte(k0 + 1, n0)];
        uint32_t r2 = *(uint32_t *)&smV[swz_byte(k0 + 2, n0)];
        uint32_t r3 = *(uint32_t *)&smV[swz_byte(k0 + 3, n0)];
        // Transpose 4x4: write 4 uint32_t (one per n-col, 4 consecutive k bytes)
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

// v81 probe: two instantiations.
//   LB=2 → natural reg count (no pressure) — baseline for hd=64 structure
//   LB=3 → ptxas forced into 170 regs/thread ceiling → see spill cost
// ptxas -v output is the falsifiable answer per template instantiation.
template <int LB>
__global__ void __launch_bounds__(FA_THREADS, LB)
    fa92_kernel(
        const uint8_t *__restrict__ Q,
        const uint8_t *__restrict__ K,
        const uint8_t *__restrict__ V,
        __half *__restrict__ O,
        int seq_len, int head_dim, int causal, float scale,
        float qk_descale, float v_descale,
        int window)  // 0 = no window (full attention or causal). >0 = sliding window.
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
    // v69_singleV: single V buffer (was double). smP overlaps smV after
    // transpose. V prefetch goes to END of iter (cannot overlap with smP use).
    uint8_t *smV = smK[1] + FA_BC * FA_STRIDE;
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

    // Qr[ks][mi][r] — KS_QK k-steps × 2 M-tiles × 4 uint32 (m16k32 A operand)
    uint32_t Qr[KS_QK][M_TILES][4];
#pragma unroll
    for (int ks = 0; ks < KS_QK; ks++)
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

    // Or_p[nt][mi][r] — N_TILES_PV N-tiles × 2 M-tiles × 2 packed uint32 (m16n8 D)
    uint32_t Or_p[N_TILES_PV][M_TILES][2];
#pragma unroll
    for (int t = 0; t < N_TILES_PV; t++)
#pragma unroll
        for (int mi = 0; mi < M_TILES; mi++)
            Or_p[t][mi][0] = Or_p[t][mi][1] = 0u;

    // Per-row state: [mi][side] where side=0 is gid row, side=1 is gid+8 row
    float rmax[M_TILES][2] = {{-1e30f, -1e30f}, {-1e30f, -1e30f}};
    float rsexp[M_TILES][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    int nkv = (seq_len + FA_BC - 1) / FA_BC;
    int kv_max_blocks = causal ? ((qs + FA_BR - 1) / FA_BC + 1) : nkv;
    if (kv_max_blocks > nkv) kv_max_blocks = nkv;
    // v69+window: sliding-window lower bound. For causal sliding window, Q-row q
    // attends to K in [max(0, q - window + 1), q]. Q-block covers qs..qs+Br-1;
    // earliest K needed = max(0, qs - window + 1). Skip K-blocks below that.
    int kv_min_blocks = 0;
    if (window > 0 && qs + 1 > window) {
        kv_min_blocks = (qs - window + 1) / FA_BC;  // floor
    }

    // Pre-load iter kv_min_blocks: K + V. Iter k prefetches iter k+1 → other bank.
    load_tile_fp8(smK[kv_min_blocks & 1], Kh, kv_min_blocks * FA_BC, FA_BC, seq_len, head_dim);
    load_tile_fp8(smV, Vh, kv_min_blocks * FA_BC, FA_BC, seq_len, head_dim);
    cpa_commit();

    // v78: V buffer alternates location. Iter kv_min reads V from smV (pre-loop).
    // Iter kv_min+N (N≥1) reads V from smK[buf_prev] (loaded by prev iter's mid-iter cp.async).
    uint8_t *prev_V_slot = smV;

    for (int kv = kv_min_blocks; kv < kv_max_blocks; kv++)
    {
        int kvs = kv * FA_BC;
        int buf = kv & 1;
        uint8_t *cur_K = smK[buf];
        uint8_t *nxt_K = smK[buf ^ 1];

        // Wait until current iter's K AND V land (V may be in smK[buf_prev] = nxt_K).
        cpa_wait<0>();
        __syncthreads();

        // v92: SKIP transpose_v entirely. PV will read V directly from prev_V_slot
        // (K×N layout) via byte-gather. Saves: 1 __syncthreads + transpose_v function call.
        // Cost: 4× more SMEM read instructions in PV (byte-gather vs uint32-gather).
        // smV_T allocation kept (compiler-DCE'd) for now to minimize SMEM layout changes.
        uint8_t *smV_pv = prev_V_slot;  // direct V read source for PV (K×N layout)

        // v81 hd=64 fix: smP needs Br×Bc = 128×64 = 8KB but smV is only 4KB (FA_STRIDE=64).
        // smQ is 8KB and DEAD after register-load of Qr[][][] before the kv loop → reuse.
        uint8_t *smP = smQ;

        // v79 lever 3: branch-free row count — no `if` guard, ternary clamps for last iter.
        // load_tile_fp8's inner loop runs 0 iters when rows_p=0 → no cp.async issued.
        int kv_p = kv + 1;
        int rows_p = (kv_p < kv_max_blocks) ? FA_BC : 0;
        load_tile_fp8(nxt_K, Kh, kv_p * FA_BC, rows_p, seq_len, head_dim);
        cpa_commit();

        // S = Q · Kᵀ — v87 Option C: explicit two-phase ks batching.
        // ks=0 batch (16 MMAs for all nt,mi) BEFORE ks=1 batch. Forces 15-op
        // gap between same-accumulator MMA pairs across ks. No #pragma unroll
        // on outer ks (no outer ks loop — two literal blocks).
        uint32_t Sr_p[8][M_TILES][2];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
                Sr_p[nt][mi][0] = Sr_p[nt][mi][1] = 0u;

        // === QK ks=0 batch ===
        {
            int cl = tid * 4, ch = cl + 16;
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
                                Qr[0][mi][0], Qr[0][mi][1],
                                Qr[0][mi][2], Qr[0][mi][3],
                                b0, b1, Sr_p[nt][mi][0], Sr_p[nt][mi][1]);
                }
            }
        }
        // === QK ks=1 batch ===
        {
            int cl = 32 + tid * 4, ch = cl + 16;
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
                                Qr[1][mi][0], Qr[1][mi][1],
                                Qr[1][mi][2], Qr[1][mi][3],
                                b0, b1, Sr_p[nt][mi][0], Sr_p[nt][mi][1]);
                }
            }
        }

        // v78: V[kv+1] cp.async → smK[buf] (dead after QK MMA finished). Overlaps
        // with softmax + smP STS + PV MMA (~60% of iter time).
        // v79 lever 3: branch-free. rows_v=0 on last iter → load_tile_fp8 inner loop is no-op.
        // prev_V_slot is unconditionally set to smK[buf]; on the last iter it'll never be
        // read (we exit the loop), so no harm.
        int rows_v = (kv + 1 < kv_max_blocks) ? FA_BC : 0;
        load_tile_fp8(smK[buf], Vh, (kv + 1) * FA_BC, rows_v, seq_len, head_dim);
        prev_V_slot = smK[buf];
        cpa_commit();

        // Sr[nt][mi][r] — float for softmax math
        float Sr[8][M_TILES][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            // v79 lever 4: pre-multiply scale by log2(e). Sr now in LOG2 space.
            // This lets the softmax exp use ex2.approx.f16x2 (2× throughput of ex2.f32).
            // rmax, nm, rsc all flow in log2 space — exp2f replaces __expf below.
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
                // v69+window: sliding-window lower bounds per row.
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
                    // Sliding window: mask K below row's lower bound.
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
            // v79 lever 4: log2-space → exp2f instead of __expf.
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
            for (int t = 0; t < N_TILES_PV; t++)
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

        // v79b: P stored as __half2 (16 b32 regs total vs v79's 64 FP32 = save ~48 regs).
        // ns sum stays FP32 to avoid f16 accumulator drift on 32-element row sums.
        float ns[M_TILES][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
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

        // v89 P-in-regs: skip smP write+sync entirely. Pack P_top/P_bot fp8x2 per thread
        // into Pf_pair[nt][mi]: low16 = fp8x2(P_top), high16 = fp8x2(P_bot).
        // PV ks=0/1 will gather Pr via shfl_sync — no SMEM round-trip, no __syncthreads.
        uint32_t Pf_pair[8][M_TILES];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                uint16_t fp8x2_top = fp16x2_to_e4m3x2(*reinterpret_cast<uint32_t *>(&P_top[nt][mi]));
                uint16_t fp8x2_bot = fp16x2_to_e4m3x2(*reinterpret_cast<uint32_t *>(&P_bot[nt][mi]));
                Pf_pair[nt][mi] = ((uint32_t)fp8x2_top) | (((uint32_t)fp8x2_bot) << 16);
            }
        }
        // Helper for warp-local lane base (same gid, different tid).
        uint32_t gid_lane_base = (threadIdx.x & 0x1c);  // (lane & ~3)

        // O += P · V — v87 Option C: explicit two-phase ks batching.
        // ks=0 batch (16 MMAs) BEFORE ks=1 batch. Same Or_p accumulators across batches.
        // Pr split into Pr0/Pr1 — different ks reads different smP columns.

        // === PV ks=0 batch ===
        {
            int cl = tid * 4, ch = cl + 16;
            uint32_t Pr0[M_TILES][4];
            // v89: gather Pr0 via shfl from Pf_pair (no smP).
            // ks=0: K cols 0..31. tids 0,1 use nt=0 for low (cols 0..7); tids 2,3 use nt=1 (cols 8..15).
            //        Similarly ch range: tids 0,1 use nt=2; tids 2,3 use nt=3.
            uint32_t src_low  = gid_lane_base | ((tid & 1) << 1);       // 0,2,0,2 within group
            uint32_t src_high = gid_lane_base | ((tid & 1) << 1) | 1;   // 1,3,1,3
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                // Pr0[mi][0]/[1] from cols cl..cl+3 of rows g0/g8 → nt=0 or 1
                uint32_t p_nt0_l = __shfl_sync(0xffffffff, Pf_pair[0][mi], src_low);
                uint32_t p_nt0_h = __shfl_sync(0xffffffff, Pf_pair[0][mi], src_high);
                uint32_t p_nt1_l = __shfl_sync(0xffffffff, Pf_pair[1][mi], src_low);
                uint32_t p_nt1_h = __shfl_sync(0xffffffff, Pf_pair[1][mi], src_high);
                uint32_t low_a  = (tid < 2) ? p_nt0_l : p_nt1_l;
                uint32_t high_a = (tid < 2) ? p_nt0_h : p_nt1_h;
                Pr0[mi][0] = (low_a & 0xFFFF) | ((high_a & 0xFFFF) << 16);  // top row, cols cl..cl+3
                Pr0[mi][1] = (low_a >> 16) | ((high_a >> 16) << 16);        // bot row, same cols
                // Pr0[mi][2]/[3] from cols ch..ch+3 → nt=2 or 3
                uint32_t p_nt2_l = __shfl_sync(0xffffffff, Pf_pair[2][mi], src_low);
                uint32_t p_nt2_h = __shfl_sync(0xffffffff, Pf_pair[2][mi], src_high);
                uint32_t p_nt3_l = __shfl_sync(0xffffffff, Pf_pair[3][mi], src_low);
                uint32_t p_nt3_h = __shfl_sync(0xffffffff, Pf_pair[3][mi], src_high);
                uint32_t low_b  = (tid < 2) ? p_nt2_l : p_nt3_l;
                uint32_t high_b = (tid < 2) ? p_nt2_h : p_nt3_h;
                Pr0[mi][2] = (low_b & 0xFFFF) | ((high_b & 0xFFFF) << 16);
                Pr0[mi][3] = (low_b >> 16) | ((high_b >> 16) << 16);
            }
#pragma unroll
            for (int nt = 0; nt < N_TILES_PV; nt++)
            {
                int br = nt * 8;
                int n = br + gid;
                // v92 byte-gather: read 4 sequential k-rows for column n from
                // smV_pv (K×N layout, swizzled by swz_byte). Pack 4 bytes → uint32.
                uint32_t b00 = smV_pv[swz_byte(cl + 0, n)];
                uint32_t b01 = smV_pv[swz_byte(cl + 1, n)];
                uint32_t b02 = smV_pv[swz_byte(cl + 2, n)];
                uint32_t b03 = smV_pv[swz_byte(cl + 3, n)];
                uint32_t b0 = b00 | (b01 << 8) | (b02 << 16) | (b03 << 24);
                uint32_t b10 = smV_pv[swz_byte(ch + 0, n)];
                uint32_t b11 = smV_pv[swz_byte(ch + 1, n)];
                uint32_t b12 = smV_pv[swz_byte(ch + 2, n)];
                uint32_t b13 = smV_pv[swz_byte(ch + 3, n)];
                uint32_t b1 = b10 | (b11 << 8) | (b12 << 16) | (b13 << 24);
#pragma unroll
                for (int mi = 0; mi < M_TILES; mi++)
                {
                    mma_fp8_f16(Or_p[nt][mi][0], Or_p[nt][mi][1],
                                Pr0[mi][0], Pr0[mi][1], Pr0[mi][2], Pr0[mi][3],
                                b0, b1, Or_p[nt][mi][0], Or_p[nt][mi][1]);
                }
            }
        }
        // === PV ks=1 batch ===
        {
            int cl = 32 + tid * 4, ch = cl + 16;
            uint32_t Pr1[M_TILES][4];
            // v89: ks=1 K cols 32..63. tids 0,1 use nt=4 for low; tids 2,3 use nt=5.
            //      ch range: tids 0,1 use nt=6; tids 2,3 use nt=7.
            uint32_t src_low  = gid_lane_base | ((tid & 1) << 1);
            uint32_t src_high = gid_lane_base | ((tid & 1) << 1) | 1;
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                uint32_t p_nt4_l = __shfl_sync(0xffffffff, Pf_pair[4][mi], src_low);
                uint32_t p_nt4_h = __shfl_sync(0xffffffff, Pf_pair[4][mi], src_high);
                uint32_t p_nt5_l = __shfl_sync(0xffffffff, Pf_pair[5][mi], src_low);
                uint32_t p_nt5_h = __shfl_sync(0xffffffff, Pf_pair[5][mi], src_high);
                uint32_t low_a  = (tid < 2) ? p_nt4_l : p_nt5_l;
                uint32_t high_a = (tid < 2) ? p_nt4_h : p_nt5_h;
                Pr1[mi][0] = (low_a & 0xFFFF) | ((high_a & 0xFFFF) << 16);
                Pr1[mi][1] = (low_a >> 16) | ((high_a >> 16) << 16);

                uint32_t p_nt6_l = __shfl_sync(0xffffffff, Pf_pair[6][mi], src_low);
                uint32_t p_nt6_h = __shfl_sync(0xffffffff, Pf_pair[6][mi], src_high);
                uint32_t p_nt7_l = __shfl_sync(0xffffffff, Pf_pair[7][mi], src_low);
                uint32_t p_nt7_h = __shfl_sync(0xffffffff, Pf_pair[7][mi], src_high);
                uint32_t low_b  = (tid < 2) ? p_nt6_l : p_nt7_l;
                uint32_t high_b = (tid < 2) ? p_nt6_h : p_nt7_h;
                Pr1[mi][2] = (low_b & 0xFFFF) | ((high_b & 0xFFFF) << 16);
                Pr1[mi][3] = (low_b >> 16) | ((high_b >> 16) << 16);
            }
#pragma unroll
            for (int nt = 0; nt < N_TILES_PV; nt++)
            {
                int br = nt * 8;
                int n = br + gid;
                // v92 byte-gather: ks=1 reads V[k=32..63][n] in 8 byte loads per thread.
                uint32_t b00 = smV_pv[swz_byte(cl + 0, n)];
                uint32_t b01 = smV_pv[swz_byte(cl + 1, n)];
                uint32_t b02 = smV_pv[swz_byte(cl + 2, n)];
                uint32_t b03 = smV_pv[swz_byte(cl + 3, n)];
                uint32_t b0 = b00 | (b01 << 8) | (b02 << 16) | (b03 << 24);
                uint32_t b10 = smV_pv[swz_byte(ch + 0, n)];
                uint32_t b11 = smV_pv[swz_byte(ch + 1, n)];
                uint32_t b12 = smV_pv[swz_byte(ch + 2, n)];
                uint32_t b13 = smV_pv[swz_byte(ch + 3, n)];
                uint32_t b1 = b10 | (b11 << 8) | (b12 << 16) | (b13 << 24);
#pragma unroll
                for (int mi = 0; mi < M_TILES; mi++)
                {
                    mma_fp8_f16(Or_p[nt][mi][0], Or_p[nt][mi][1],
                                Pr1[mi][0], Pr1[mi][1], Pr1[mi][2], Pr1[mi][3],
                                b0, b1, Or_p[nt][mi][0], Or_p[nt][mi][1]);
                }
            }
        }
        // v79 lever 2: end-of-iter __syncthreads removed. Was needed in v69 to gate the
        // end-of-iter V cp.async vs PV's smP reads, but v78 moved V cp.async to mid-iter
        // → no SMEM writes after PV in this iter. Next iter's cpa_wait + sync at line 278-279
        // synchronizes all warps before transpose_v writes smV_T.
    }

#pragma unroll
    for (int mi = 0; mi < M_TILES; mi++)
    {
        float li0 = (rsexp[mi][0] > 0) ? v_descale / rsexp[mi][0] : 0.0f;
        float li1 = (rsexp[mi][1] > 0) ? v_descale / rsexp[mi][1] : 0.0f;
        int mr = mrb + mi * 16;
        int gr0 = qs + mr + gid, gr8 = gr0 + 8;
#pragma unroll
        for (int nt = 0; nt < N_TILES_PV; nt++)
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
            // Sliding window: K range = [max(0, q - window + 1), q] for causal.
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

static void report_attrs(const char *label, cudaFuncAttributes &attr, int smem_bytes)
{
    int reg_blocks = 65536 / (attr.numRegs * FA_THREADS);
    if (reg_blocks > 16) reg_blocks = 16;
    int smem_blocks = (smem_bytes > 0) ? (100 * 1024 / smem_bytes) : 16;
    int real_blocks = (reg_blocks < smem_blocks) ? reg_blocks : smem_blocks;
    printf("  %s  numRegs=%d  localSpill=%zu B  staticSmem=%zu B\n",
           label, attr.numRegs, attr.localSizeBytes, attr.sharedSizeBytes);
    printf("    reg-bound blocks/SM = %d   smem-bound = %d   ACTUAL = %d\n",
           reg_blocks, smem_blocks, real_blocks);
}

// NCu-friendly launcher: ./fa_v81_hd64_fp8 --ncu <cfg_idx> <LB>
// Runs ONE warmup + ONE measured launch of fa92_kernel<LB> on bench_configs[cfg_idx].
// NCu profiles only the measured launch (use --launch-skip 1 --launch-count 1).
static int ncu_mode(int cfg_idx, int LB)
{
    int bench_configs[][4] = {
        {4, 1024, 64, 0},   {4, 2048, 64, 0},   {8, 2048, 64, 0},
        {4, 4096, 64, 0},   {8, 4096, 64, 0},   {16, 2048, 64, 0},
        {16, 4096, 64, 0},  {32, 2048, 64, 0},
        {4, 8192, 64, 1024}, {4, 4096, 64, 1024}, {8, 8192, 64, 1024},
    };
    int n_bench = sizeof(bench_configs) / sizeof(bench_configs[0]);
    if (cfg_idx < 0 || cfg_idx >= n_bench) {
        fprintf(stderr, "cfg_idx %d out of range [0, %d)\n", cfg_idx, n_bench);
        return 1;
    }
    if (LB != 2 && LB != 3) {
        fprintf(stderr, "LB must be 2 or 3\n");
        return 1;
    }
    int bh = bench_configs[cfg_idx][0], sl = bench_configs[cfg_idx][1];
    int hd = bench_configs[cfg_idx][2], wnd = bench_configs[cfg_idx][3];
    int ca = (wnd > 0) ? 1 : 0;
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

    int smem = FA_BR * FA_STRIDE + 2 * FA_BC * FA_STRIDE + FA_BC * FA_STRIDE
             + hd * SMV_T_STRIDE;
    int nqt = (sl + FA_BR - 1) / FA_BR;
    int grid = bh * nqt;
    float scale = 1.0f / sqrtf((float)hd);

    fprintf(stderr, "NCu config: bh=%d sl=%d hd=%d wnd=%d grid=%d LB=%d\n",
            bh, sl, hd, wnd, grid, LB);

    if (LB == 2) {
        CK(cudaFuncSetAttribute(fa92_kernel<2>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        // warmup (NCu skips with --launch-skip)
        fa92_kernel<2><<<grid, FA_THREADS, smem>>>(Q_d, K_d, V_d, O_d, sl, hd, ca, scale, 1.0f, 1.0f, wnd);
        CK(cudaDeviceSynchronize());
        // measured (NCu profiles this one)
        fa92_kernel<2><<<grid, FA_THREADS, smem>>>(Q_d, K_d, V_d, O_d, sl, hd, ca, scale, 1.0f, 1.0f, wnd);
    } else {
        CK(cudaFuncSetAttribute(fa92_kernel<3>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        fa92_kernel<3><<<grid, FA_THREADS, smem>>>(Q_d, K_d, V_d, O_d, sl, hd, ca, scale, 1.0f, 1.0f, wnd);
        CK(cudaDeviceSynchronize());
        fa92_kernel<3><<<grid, FA_THREADS, smem>>>(Q_d, K_d, V_d, O_d, sl, hd, ca, scale, 1.0f, 1.0f, wnd);
    }
    CK(cudaDeviceSynchronize());

    cudaFree(Q_d); cudaFree(K_d); cudaFree(V_d); cudaFree(O_d);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc >= 4 && strcmp(argv[1], "--ncu") == 0) {
        return ncu_mode(atoi(argv[2]), atoi(argv[3]));
    }
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("=== FA v92 hd=64 + no-transpose_v + byte-gather V (4× SMEM ops in PV) ===\n");
    printf("GPU: %s (%d SMs)\n", p.name, p.multiProcessorCount);

    int smem_hd64 = FA_BR * FA_STRIDE + 2 * FA_BC * FA_STRIDE + FA_BC * FA_STRIDE
                  + FA_HD * SMV_T_STRIDE;
    printf("\nSMEM/block (hd=64): %d B (%.2f KB) — SMEM allows %d blocks/SM (≤ 100 KB cap)\n",
           smem_hd64, smem_hd64 / 1024.0, 100 * 1024 / smem_hd64);
    printf("v79b baseline @ hd=128: 249 regs/thread → reg-bound 2 blocks/SM\n");
    printf("Need ≤ 170 regs/thread for 3 blocks. Expected v81 from arithmetic: ~217 regs (still over)\n\n");

    cudaFuncAttributes a2, a3;
    CK(cudaFuncGetAttributes(&a2, fa92_kernel<2>));
    CK(cudaFuncGetAttributes(&a3, fa92_kernel<3>));
    printf("--- Kernel attrs ---\n");
    report_attrs("LB=2 (natural)  ", a2, smem_hd64);
    report_attrs("LB=3 (forced)   ", a3, smem_hd64);

    // -------------------------------------------------------------------------
    // Phase 1: Full correctness on BOTH LB=2 and LB=3 kernels.
    // Edge cases that v79b's bench covered: causal, sliding window aligned/unaligned,
    // sl-not-multiple-of-Br. Same dataset feeds both kernels → byte-identical Q/K/V.
    // -------------------------------------------------------------------------
    printf("\n--- Phase 1: Correctness on hd=64 (all forms × both LB) ---\n");
    int configs[][5] = {
        {1, 64,  64, 0, 0},     // small full attn
        {1, 128, 64, 0, 0},
        {1, 256, 64, 0, 0},
        {1, 512, 64, 0, 0},
        {2, 256, 64, 1, 0},     // causal no-window
        {1, 256, 64, 1, 64},    // sliding window aligned to FA_BC
        {1, 256, 64, 1, 100},   // window NOT multiple of FA_BC
        {1, 300, 64, 1, 96},    // window mid-block, sl not multiple of FA_BR
    };
    int total_configs = sizeof(configs) / sizeof(configs[0]);
    int pass_lb2 = 0, pass_lb3 = 0;
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

        int smem = FA_BR * FA_STRIDE + 2 * FA_BC * FA_STRIDE + FA_BC * FA_STRIDE
                 + hd * SMV_T_STRIDE;
        int nqt = (sl + FA_BR - 1) / FA_BR;
        float scale = 1.0f / sqrtf((float)hd);
        uint16_t *O_cpu = (uint16_t *)malloc(n_elems * 2);

        // LB=2 run
        CK(cudaFuncSetAttribute(fa92_kernel<2>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        CK(cudaMemset(O_d, 0, n_elems * 2));
        fa92_kernel<2><<<bh * nqt, FA_THREADS, smem>>>(
            Q_d, K_d, V_d, O_d, sl, hd, ca, scale, 1.0f, 1.0f, wnd);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(O_cpu, O_d, n_elems * 2, cudaMemcpyDeviceToHost));
        float mx2 = 0; int errs2 = 0;
        for (size_t i = 0; i < n_elems; i++) {
            float gpu = fp16f(O_cpu[i]); float ref = O_ref[i];
            float ae = fabsf(gpu - ref);
            if (ae > mx2) mx2 = ae;
            if (ae > fmaxf(0.05f, fabsf(ref) * 0.1f)) errs2++;
        }
        if (errs2 == 0) pass_lb2++;

        // LB=3 run
        CK(cudaFuncSetAttribute(fa92_kernel<3>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        CK(cudaMemset(O_d, 0, n_elems * 2));
        fa92_kernel<3><<<bh * nqt, FA_THREADS, smem>>>(
            Q_d, K_d, V_d, O_d, sl, hd, ca, scale, 1.0f, 1.0f, wnd);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(O_cpu, O_d, n_elems * 2, cudaMemcpyDeviceToHost));
        float mx3 = 0; int errs3 = 0;
        for (size_t i = 0; i < n_elems; i++) {
            float gpu = fp16f(O_cpu[i]); float ref = O_ref[i];
            float ae = fabsf(gpu - ref);
            if (ae > mx3) mx3 = ae;
            if (ae > fmaxf(0.05f, fabsf(ref) * 0.1f)) errs3++;
        }
        if (errs3 == 0) pass_lb3++;

        printf("  bh=%d sl=%d hd=%d ca=%d wnd=%-4d  LB2: diff=%.4f errs=%d %s | LB3: diff=%.4f errs=%d %s\n",
               bh, sl, hd, ca, wnd,
               mx2, errs2, errs2 == 0 ? "PASS" : "FAIL",
               mx3, errs3, errs3 == 0 ? "PASS" : "FAIL");

        free(Qf); free(Kf); free(Vf);
        free(Qq); free(Kq); free(Vq);
        free(O_ref); free(O_cpu);
        cudaFree(Q_d); cudaFree(K_d); cudaFree(V_d); cudaFree(O_d);
    }
    printf("  → LB=2: %d/%d PASS   LB=3: %d/%d PASS\n",
           pass_lb2, total_configs, pass_lb3, total_configs);

    // -------------------------------------------------------------------------
    // Phase 2: A/B perf bench — LB=2 (2 blocks/SM) vs LB=3 (3 blocks/SM)
    // on the SAME hd=64 kernel, SAME inputs, SAME shapes. Isolates pure
    // occupancy effect. Variance ×3 (best/median/worst per LB×config).
    // -------------------------------------------------------------------------
    printf("\n--- Phase 2: A/B perf — LB=2 vs LB=3 on hd=64 (variance ×3) ---\n");
    int bench_configs[][4] = {
        // Full-attention configs (window=0, causal=0):
        {4, 1024, 64, 0},
        {4, 2048, 64, 0},
        {8, 2048, 64, 0},
        {4, 4096, 64, 0},
        {8, 4096, 64, 0},
        {16, 2048, 64, 0},
        {16, 4096, 64, 0},
        {32, 2048, 64, 0},
        // v92 large-grid additions — confirm or refute regression at 466T peak zone:
        {64, 4096, 64, 0},
        {64, 8192, 64, 0},    // v89 absolute peak 466.8T lives here
        {128, 2048, 64, 0},
        {128, 4096, 64, 0},
        // Sliding-window configs (causal=1):
        {4, 8192, 64, 1024},
        {4, 4096, 64, 1024},
        {8, 8192, 64, 1024},
    };
    int n_bench = sizeof(bench_configs) / sizeof(bench_configs[0]);
    const int VARIANCE_RUNS = 3;
    const int WARMUP = 5;
    const int ITERS = 50;

    printf("  %-26s | LB=2 (best/med/worst)        | LB=3 (best/med/worst)        | LB3/LB2\n",
           "config");
    printf("  --------------------------|-------------------------------|-------------------------------|--------\n");
    for (int ci = 0; ci < n_bench; ci++)
    {
        int bh = bench_configs[ci][0], sl = bench_configs[ci][1];
        int hd = bench_configs[ci][2], wnd = bench_configs[ci][3];
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

        int smem = FA_BR * FA_STRIDE + 2 * FA_BC * FA_STRIDE + FA_BC * FA_STRIDE
                 + hd * SMV_T_STRIDE;
        CK(cudaFuncSetAttribute(fa92_kernel<2>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        CK(cudaFuncSetAttribute(fa92_kernel<3>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        int nqt = (sl + FA_BR - 1) / FA_BR;
        float scale = 1.0f / sqrtf((float)hd);
        int grid = bh * nqt;

        double sl_eff = (wnd > 0) ? (double)wnd : (double)sl;
        if (ca_bench && wnd == 0) sl_eff = (double)sl / 2.0;
        double flops = 4.0 * (double)bh * (double)sl * sl_eff * (double)hd;

        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

        auto measure = [&](auto kernel_ptr) -> double {
            for (int i = 0; i < WARMUP; i++)
                kernel_ptr<<<grid, FA_THREADS, smem>>>(Q_d, K_d, V_d, O_d, sl, hd, ca_bench, scale, 1.0f, 1.0f, wnd);
            CK(cudaDeviceSynchronize());
            cudaEventRecord(t0);
            for (int i = 0; i < ITERS; i++)
                kernel_ptr<<<grid, FA_THREADS, smem>>>(Q_d, K_d, V_d, O_d, sl, hd, ca_bench, scale, 1.0f, 1.0f, wnd);
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            float ms; cudaEventElapsedTime(&ms, t0, t1);
            return ms / ITERS;
        };

        double r2[VARIANCE_RUNS], r3[VARIANCE_RUNS];
        for (int v = 0; v < VARIANCE_RUNS; v++) {
            r2[v] = measure(fa92_kernel<2>);
            r3[v] = measure(fa92_kernel<3>);
        }
        // sort each
        auto sort3 = [](double *a) {
            if (a[0] > a[1]) std::swap(a[0], a[1]);
            if (a[1] > a[2]) std::swap(a[1], a[2]);
            if (a[0] > a[1]) std::swap(a[0], a[1]);
        };
        sort3(r2); sort3(r3);

        double tf2_best = flops / (r2[0] / 1000.0) / 1e12;
        double tf2_med  = flops / (r2[1] / 1000.0) / 1e12;
        double tf2_worst= flops / (r2[2] / 1000.0) / 1e12;
        double tf3_best = flops / (r3[0] / 1000.0) / 1e12;
        double tf3_med  = flops / (r3[1] / 1000.0) / 1e12;
        double tf3_worst= flops / (r3[2] / 1000.0) / 1e12;
        double ratio    = tf3_med / tf2_med;

        char cfg[32];
        snprintf(cfg, sizeof(cfg), "bh=%d sl=%d wnd=%d", bh, sl, wnd);
        printf("  %-26s | %5.1f / %5.1f / %5.1f T  | %5.1f / %5.1f / %5.1f T  | %5.3f%s\n",
               cfg, tf2_best, tf2_med, tf2_worst,
               tf3_best, tf3_med, tf3_worst, ratio,
               ratio >= 1.02 ? "  ✓" : (ratio <= 0.98 ? "  ✗" : "  ="));

        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaFree(Q_d); cudaFree(K_d); cudaFree(V_d); cudaFree(O_d);
    }

    printf("\n=== Verdict ===\n");
    printf("LB=2 (226 regs, 2 blocks/SM) vs LB=3 (168 regs, 3 blocks/SM) — same hd=64 kernel.\n");
    printf("ratio > 1.02 → 3rd block buys speed (occupancy hides latency)\n");
    printf("ratio ≈ 1.00 → 3rd block neutral (occupancy not the bottleneck)\n");
    printf("ratio < 0.98 → 3rd block hurts (e.g. extra resched cost, cache pressure)\n");

    return 0;
}
