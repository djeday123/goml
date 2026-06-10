// =============================================================================
// FlashAttention v69 — FP8 Forward, single-buffer V → 2 blocks/SM
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
// History: v66 baseline → v68 conflict fix (smV_T padding) → v69 single-V.
//          v67 TMA rejected. v69_singleV experiment kept as reference.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define FA_BR 128
#define FA_BC 64
#define FA_THREADS 128   // 4 warps × M_TILES=2 × 16 rows = Br=128
#define FA_STRIDE 128
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

// v115: SHFL-based cooperative transpose.
// 4 threads in a group cooperatively transpose 1 4x4 tile.
// - Each thread loads 1 row of source (1 uint32 = 4 fp8 bytes)
// - 2 SHFL.XOR steps + byte shuffles → each thread holds 1 transposed row
// - Each thread writes 1 row to smV_T
//
// WRITE PATTERN: 4 threads in group write to 4 ADJACENT rows of smV_T (stride 1).
// Bank stride = 17 (gcd 1 with 32) → potentially NO bank conflict on writes.
// (Original: 4-way conflict due to 4-row stride between adjacent threads.)
//
// Work amount: 16 iters × 32 groups = 512 tiles (same as original 4 iters × 128 threads).
// Per-thread per-iter: 1 LDS read + 2 SHFL + 1 STS write (vs original 4 LDS + 4 STS).
__device__ __forceinline__ void transpose_v(
    uint8_t *smV_T, const uint8_t *smV, int head_dim)
{
    constexpr int TILE = 4;
    int tiles_k = FA_BC / TILE;       // 16
    int tiles_n = head_dim / TILE;     // 32 for hd=128
    int total = tiles_k * tiles_n;     // 512

    int group_id = threadIdx.x / 4;     // 0..31 (32 groups in 128-thread block)
    int in_group = threadIdx.x & 3;     // 0..3 (which row in the 4x4 tile)

    // 32 groups process 32 tiles per iter. Need 512/32 = 16 iters.
    for (int g = group_id; g < total; g += 32) {
        int tk = g / tiles_n;
        int tn = g % tiles_n;
        int k0 = tk * TILE;
        int n0 = tn * TILE;

        // Each thread loads 1 row of source 4x4 tile.
        // Thread in_group=t loads smV[k0+t, n0..n0+3] as uint32 (4 fp8 bytes).
        uint32_t my_row = *(uint32_t *)&smV[swz_byte(k0 + in_group, n0)];

        // Step 1: SHFL.XOR with mask 1 — swap row pairs (0↔1, 2↔3).
        uint32_t r_partner1 = __shfl_xor_sync(0xffffffff, my_row, 1);
        uint32_t s1;
        if ((in_group & 1) == 0) {
            // Even (thread 0, 2): low 16 of self at pos 0,1 + low 16 of partner at pos 2,3
            // Thread 0: s1 = (b00, b01, b10, b11)
            s1 = (my_row & 0x0000FFFFu) | ((r_partner1 & 0x0000FFFFu) << 16);
        } else {
            // Odd (thread 1, 3): high 16 of partner at pos 0,1 + high 16 of self at pos 2,3
            // Thread 1: s1 = (b02, b03, b12, b13)
            // FIX: was wrong direction — bytes from partner1 (>>16) at low, self high at high
            s1 = (my_row & 0xFFFF0000u) | (r_partner1 >> 16);
        }

        // Step 2: SHFL.XOR with mask 2 — swap pair of pairs (0↔2, 1↔3).
        uint32_t r_partner2 = __shfl_xor_sync(0xffffffff, s1, 2);
        uint32_t s2;
        if ((in_group & 2) == 0) {
            // Even (threads 0, 1): extract bytes 0, 2 from both s1 (low half) and r_partner2 (high half)
            // Thread 0 ends up with col 0; Thread 1 ends up with col 2 (NOT col 1!)
            s2 = (s1 & 0x000000FFu)                  // byte 0 of s1 → byte 0
               | ((s1 & 0x00FF0000u) >> 8)           // byte 2 of s1 → byte 1
               | ((r_partner2 & 0x000000FFu) << 16)  // byte 0 of r_partner2 → byte 2
               | ((r_partner2 & 0x00FF0000u) << 8);  // byte 2 of r_partner2 → byte 3
        } else {
            // Odd (threads 2, 3): FIX — bytes from r_partner2 go to LOW (pos 0,1), s1 to HIGH (pos 2,3)
            // Thread 2 ends up with col 1; Thread 3 ends up with col 3
            s2 = ((r_partner2 & 0x0000FF00u) >> 8)   // byte 1 of r_partner2 → byte 0
               | ((r_partner2 & 0xFF000000u) >> 16)  // byte 3 of r_partner2 → byte 1
               | ((s1 & 0x0000FF00u) << 8)           // byte 1 of s1 → byte 2
               | (s1 & 0xFF000000u);                  // byte 3 of s1 → byte 3
        }

        // FIX: After my SHFL pattern, thread t holds column (bit-reversed t):
        //   thread 0 → col 0, thread 1 → col 2, thread 2 → col 1, thread 3 → col 3
        // So write_offset = bit-reverse of low 2 bits of in_group: 0→0, 1→2, 2→1, 3→3
        int write_offset = ((in_group & 1) << 1) | ((in_group & 2) >> 1);
        *(uint32_t *)&smV_T[swz_byte_smvt(n0 + write_offset, k0)] = s2;
    }
}

__global__ void __launch_bounds__(FA_THREADS, 2)
    fa115_kernel(
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

        // v78: read V from wherever prev iter's mid-iter cp.async put it.
        // NOTE on Lever 1 (K-before-transpose) — IT IS UNSAFE in this scheme.
        // prev_V_slot == nxt_K in iter ≥ 1 (V[N+1] and K[N+2] alias the same smK slot).
        // K cp.async before transpose would write the slot WHILE transpose reads it.
        transpose_v(smV_T, prev_V_slot, head_dim);
        __syncthreads();

        uint8_t *smP = smV;  // smV stays the smP scratchpad (only iter kv_min stored real V here)

        // v79 lever 3: branch-free row count — no `if` guard, ternary clamps for last iter.
        // load_tile_fp8's inner loop runs 0 iters when rows_p=0 → no cp.async issued.
        int kv_p = kv + 1;
        int rows_p = (kv_p < kv_max_blocks) ? FA_BC : 0;
        load_tile_fp8(nxt_K, Kh, kv_p * FA_BC, rows_p, seq_len, head_dim);
        cpa_commit();

        // S = Q · Kᵀ — K B-operand loaded once per (nt, ks), reused across mi.
        uint32_t Sr_p[8][M_TILES][2];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
                Sr_p[nt][mi][0] = Sr_p[nt][mi][1] = 0u;
        // v96: Option C ks-batching ported from v87 hd=64.
        // hd=128 has 4 ks-steps in QK (head_dim/32). Explicit batches replace
        // `for ks` outer loop — each batch is a complete (nt, mi) sweep with
        // fixed ks. Scheduler sees explicit phase boundaries.
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
        // === QK ks=2 batch ===
        {
            int cl = 64 + tid * 4, ch = cl + 16;
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
                                Qr[2][mi][0], Qr[2][mi][1],
                                Qr[2][mi][2], Qr[2][mi][3],
                                b0, b1, Sr_p[nt][mi][0], Sr_p[nt][mi][1]);
                }
            }
        }
        // === QK ks=3 batch ===
        {
            int cl = 96 + tid * 4, ch = cl + 16;
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
                                Qr[3][mi][0], Qr[3][mi][1],
                                Qr[3][mi][2], Qr[3][mi][3],
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
                // v79b: P_top/P_bot already __half2 — no float→half conversion needed.
                uint16_t fp8x2_top = fp16x2_to_e4m3x2(*reinterpret_cast<uint32_t *>(&P_top[nt][mi]));
                uint16_t fp8x2_bot = fp16x2_to_e4m3x2(*reinterpret_cast<uint32_t *>(&P_bot[nt][mi]));
                *(uint16_t *)&smP[swz_byte_bc(row0, col0)] = fp8x2_top;
                *(uint16_t *)&smP[swz_byte_bc(row8, col0)] = fp8x2_bot;
            }
        }
        __syncthreads();

        // v96: Option C ks-batching for PV. hd=128 has 2 ks-steps in PV (Bc/32).
        // === PV ks=0 batch ===
        {
            int cl = tid * 4, ch = cl + 16;
            uint32_t Pr0[M_TILES][4];
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                int mr = mrb + mi * 16;
                int g0 = mr + gid, g8 = g0 + 8;
                Pr0[mi][0] = *(uint32_t *)&smP[swz_byte_bc(g0, cl)];
                Pr0[mi][1] = *(uint32_t *)&smP[swz_byte_bc(g8, cl)];
                Pr0[mi][2] = *(uint32_t *)&smP[swz_byte_bc(g0, ch)];
                Pr0[mi][3] = *(uint32_t *)&smP[swz_byte_bc(g8, ch)];
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
                                Pr0[mi][0], Pr0[mi][1], Pr0[mi][2], Pr0[mi][3],
                                b0, b1, Or_p[nt][mi][0], Or_p[nt][mi][1]);
                }
            }
        }
        // === PV ks=1 batch ===
        {
            int cl = 32 + tid * 4, ch = cl + 16;
            uint32_t Pr1[M_TILES][4];
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                int mr = mrb + mi * 16;
                int g0 = mr + gid, g8 = g0 + 8;
                Pr1[mi][0] = *(uint32_t *)&smP[swz_byte_bc(g0, cl)];
                Pr1[mi][1] = *(uint32_t *)&smP[swz_byte_bc(g8, cl)];
                Pr1[mi][2] = *(uint32_t *)&smP[swz_byte_bc(g0, ch)];
                Pr1[mi][3] = *(uint32_t *)&smP[swz_byte_bc(g8, ch)];
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

// NCu-friendly launcher: ./fa_v96_ksbatched --ncu <cfg_idx>
// Runs ONE warmup + ONE measured launch of fa115_kernel on bench_configs[cfg_idx].
// NCu profiles only the measured launch (use --launch-skip 1 --launch-count 1).
static int ncu_mode(int cfg_idx)
{
    int bench_configs[][4] = {
        {4, 1024, 128, 0},  {4, 2048, 128, 0},  {8, 2048, 128, 0},
        {4, 4096, 128, 0},  {8, 4096, 128, 0},  {16, 2048, 128, 0},
        {16, 4096, 128, 0}, {32, 2048, 128, 0},
        {64, 4096, 128, 0}, {64, 8192, 128, 0},   // 8, 9 — large; 9 = PEAK
        {128, 2048, 128, 0}, {128, 4096, 128, 0}, // 10, 11
        {4, 8192, 128, 1024}, {4, 4096, 128, 1024}, {8, 8192, 128, 1024},
        {16, 8192, 128, 1024},
    };
    int n_bench = sizeof(bench_configs) / sizeof(bench_configs[0]);
    if (cfg_idx < 0 || cfg_idx >= n_bench) {
        fprintf(stderr, "cfg_idx %d out of range [0, %d)\n", cfg_idx, n_bench);
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
    fprintf(stderr, "NCu config: bh=%d sl=%d hd=%d wnd=%d grid=%d\n",
            bh, sl, hd, wnd, grid);
    CK(cudaFuncSetAttribute(fa115_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    fa115_kernel<<<grid, FA_THREADS, smem>>>(Q_d, K_d, V_d, O_d, sl, hd, ca, scale, 1.0f, 1.0f, wnd);
    CK(cudaDeviceSynchronize());
    fa115_kernel<<<grid, FA_THREADS, smem>>>(Q_d, K_d, V_d, O_d, sl, hd, ca, scale, 1.0f, 1.0f, wnd);
    CK(cudaDeviceSynchronize());
    cudaFree(Q_d); cudaFree(K_d); cudaFree(V_d); cudaFree(O_d);
    return 0;
}

// Loop mode: run cfg_idx kernel for `seconds` continuously.
// Use case: monitor GPU clocks/temp/throttle in another terminal during sustained run.
// Allocates buffers once, runs kernel in tight loop. No timing, just keep GPU busy.
static int loop_mode(int cfg_idx, int seconds)
{
    int bench_configs[][4] = {
        {4, 1024, 128, 0},  {4, 2048, 128, 0},  {8, 2048, 128, 0},
        {4, 4096, 128, 0},  {8, 4096, 128, 0},  {16, 2048, 128, 0},
        {16, 4096, 128, 0}, {32, 2048, 128, 0},
        {64, 4096, 128, 0}, {64, 8192, 128, 0},
        {128, 2048, 128, 0}, {128, 4096, 128, 0},
        {4, 8192, 128, 1024}, {4, 4096, 128, 1024}, {8, 8192, 128, 1024},
        {16, 8192, 128, 1024},
    };
    int n_bench = sizeof(bench_configs) / sizeof(bench_configs[0]);
    if (cfg_idx < 0 || cfg_idx >= n_bench) {
        fprintf(stderr, "cfg_idx out of range [0, %d)\n", n_bench); return 1;
    }
    int bh = bench_configs[cfg_idx][0], sl = bench_configs[cfg_idx][1];
    int hd = bench_configs[cfg_idx][2], wnd = bench_configs[cfg_idx][3];
    int ca = (wnd > 0) ? 1 : 0;
    size_t n_elems = (size_t)bh * sl * hd;
    uint8_t *Q_d, *K_d, *V_d; __half *O_d;
    CK(cudaMalloc(&Q_d, n_elems)); CK(cudaMalloc(&K_d, n_elems));
    CK(cudaMalloc(&V_d, n_elems)); CK(cudaMalloc(&O_d, n_elems * 2));
    CK(cudaMemset(Q_d, 0x38, n_elems)); CK(cudaMemset(K_d, 0x38, n_elems));
    CK(cudaMemset(V_d, 0x38, n_elems));
    int smem = FA_BR * FA_STRIDE + 2 * FA_BC * FA_STRIDE + FA_BC * FA_STRIDE
             + hd * SMV_T_STRIDE;
    int nqt = (sl + FA_BR - 1) / FA_BR;
    int grid = bh * nqt;
    float scale = 1.0f / sqrtf((float)hd);
    CK(cudaFuncSetAttribute(fa115_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    fprintf(stderr, "Loop: cfg=%d bh=%d sl=%d wnd=%d grid=%d, running for %d sec\n",
            cfg_idx, bh, sl, wnd, grid, seconds);
    fprintf(stderr, "Monitor in another terminal:\n");
    fprintf(stderr, "  nvidia-smi dmon -s pucmt -d 1\n");
    fprintf(stderr, "Or:\n");
    fprintf(stderr, "  watch -n 0.5 'nvidia-smi --query-gpu=clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,utilization.gpu --format=csv'\n");
    // Measure how many launches we can do per second to time the loop
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    int probe = 50;
    for (int i = 0; i < probe; i++)
        fa115_kernel<<<grid, FA_THREADS, smem>>>(Q_d, K_d, V_d, O_d, sl, hd, ca, scale, 1.0f, 1.0f, wnd);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    double per_launch_ms = ms / probe;
    long long total_launches = (long long)((double)seconds * 1000.0 / per_launch_ms);
    fprintf(stderr, "Probe: %.3f ms/launch → %lld launches for %d sec\n",
            per_launch_ms, total_launches, seconds);
    long long batch = 200;  // print progress every batch launches
    auto t_start = std::chrono::steady_clock::now();
    long long launched = 0;
    while (launched < total_launches) {
        for (long long i = 0; i < batch && launched < total_launches; i++, launched++)
            fa115_kernel<<<grid, FA_THREADS, smem>>>(Q_d, K_d, V_d, O_d, sl, hd, ca, scale, 1.0f, 1.0f, wnd);
        CK(cudaDeviceSynchronize());
        auto t_now = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration<double>(t_now - t_start).count();
        fprintf(stderr, "  [%5.1fs] launched %lld / %lld (%.1f%%)\n",
                elapsed_s, launched, total_launches, 100.0 * launched / total_launches);
    }
    auto t_end = std::chrono::steady_clock::now();
    double total_s = std::chrono::duration<double>(t_end - t_start).count();
    fprintf(stderr, "Done. Total wall time: %.2f sec\n", total_s);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(Q_d); cudaFree(K_d); cudaFree(V_d); cudaFree(O_d);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc >= 3 && strcmp(argv[1], "--ncu") == 0) {
        return ncu_mode(atoi(argv[2]));
    }
    if (argc >= 4 && strcmp(argv[1], "--loop") == 0) {
        return loop_mode(atoi(argv[2]), atoi(argv[3]));
    }
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    int clk = 0; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    printf("=== FA v115 = v96 + SHFL cooperative transpose (4-thread groups, no write conflict) ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n", p.name, p.multiProcessorCount, clk / 1000);

    // Report actual kernel attributes (real SMEM use + reg count).
    cudaFuncAttributes attr;
    CK(cudaFuncGetAttributes(&attr, fa115_kernel));
    printf("Kernel attrs: numRegs=%d  binSize=%d  sharedSizeBytes(static)=%zu\n",
           attr.numRegs, attr.binaryVersion, attr.sharedSizeBytes);
    int smem_hd128 = FA_BR * FA_STRIDE + 2 * FA_BC * FA_STRIDE + FA_BC * FA_STRIDE
                   + 128 * SMV_T_STRIDE;
    int smem_v68   = (FA_BR + 2 * FA_BC + 2 * FA_BC) * FA_STRIDE + 128 * SMV_T_STRIDE;
    printf("Dynamic SMEM (hd=128): v69_singleV=%d B (%.2f KB) vs v68=%d B (delta=%+d B)\n",
           smem_hd128, smem_hd128 / 1024.0, smem_v68, smem_hd128 - smem_v68);
    printf("2 blocks × %.2f KB = %.2f KB ≤ 100 KB cap → 2 blocks/SM (occupancy ×2)\n\n",
           smem_hd128 / 1024.0, 2 * smem_hd128 / 1024.0);

    printf("--- Correctness vs CPU FP8-roundtripped reference ---\n");
    // v69+window: 5th column = window (0 = no window). All causal sliding window tests.
    int configs[][5] = {
        {1, 64,  128, 0, 0},     // full attn (small)
        {1, 128, 128, 0, 0},
        {1, 256, 128, 0, 0},
        {1, 512, 128, 0, 0},
        {2, 256, 128, 1, 0},     // causal no-window
        // Sliding-window edge cases — kept SMALL so CPU ref is tractable:
        {1, 256, 128, 1, 64},    // window aligned to FA_BC=64
        {1, 256, 128, 1, 100},   // window NOT multiple of any block boundary
        {1, 300, 128, 1, 96},    // window cuts mid-block, sl not multiple of Br=128
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

        // SMEM: smQ + 2×smK + 2×smV (smP overlaps cur_V) + smV_T
        // v69_singleV: smQ + 2×smK + 1×smV + smV_T(padded).
        // = 16384 + 16384 + 8192 + 128*68(=8704) = 49664 B = 48.5 KB.
        // 2 blocks × 48.5 = 97 KB ≤ 100 KB cap → 2 blocks/SM.
        int smem = FA_BR * FA_STRIDE + 2 * FA_BC * FA_STRIDE + FA_BC * FA_STRIDE
                 + hd * SMV_T_STRIDE;
        CK(cudaFuncSetAttribute(fa115_kernel,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        int nqt = (sl + FA_BR - 1) / FA_BR;
        float scale = 1.0f / sqrtf((float)hd);
        fa115_kernel<<<bh * nqt, FA_THREADS, smem>>>(
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
    // 4th column = window (0 = full causal, >0 = sliding window).
    int bench_configs[][4] = {
        // Full-attention configs (window=0, causal=0 in launch — non-causal full):
        {4, 1024, 128, 0},
        {4, 2048, 128, 0},
        {8, 2048, 128, 0},
        {4, 4096, 128, 0},
        {8, 4096, 128, 0},
        {16, 2048, 128, 0},
        {16, 4096, 128, 0},
        {32, 2048, 128, 0},
        // Large-grid:
        {32, 4096, 128, 0},
        {32, 8192, 128, 0},   // 32-batch long-seq
        {64, 4096, 128, 0},
        {64, 8192, 128, 0},   // PEAK reference
        {128, 2048, 128, 0},
        {128, 4096, 128, 0},
        {128, 8192, 128, 0},  // large bh + long seq
        // XL grids (heavy memory but informative):
        {256, 2048, 128, 0},  // very large bh
        {256, 4096, 128, 0},
        // SLIDING WINDOW configs:
        {4, 8192, 128, 1024},
        {4, 4096, 128, 1024},
        {8, 8192, 128, 1024},
        {16, 8192, 128, 1024},
        {32, 8192, 128, 1024}, // mid-batch sliding
        {64, 8192, 128, 1024}, // large-batch sliding
    };
    for (auto &c : bench_configs)
    {
        int bh = c[0], sl = c[1], hd = c[2], wnd = c[3];
        int ca_bench = (wnd > 0) ? 1 : 0;  // sliding window requires causal=1
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

        // SMEM: smQ + 2×smK + 2×smV (smP overlaps cur_V) + smV_T
        // v69_singleV: smQ + 2×smK + 1×smV + smV_T(padded).
        // = 16384 + 16384 + 8192 + 128*68(=8704) = 49664 B = 48.5 KB.
        // 2 blocks × 48.5 = 97 KB ≤ 100 KB cap → 2 blocks/SM.
        int smem = FA_BR * FA_STRIDE + 2 * FA_BC * FA_STRIDE + FA_BC * FA_STRIDE
                 + hd * SMV_T_STRIDE;
        int nqt = (sl + FA_BR - 1) / FA_BR;
        float scale = 1.0f / sqrtf((float)hd);

        for (int i = 0; i < 5; i++)
            fa115_kernel<<<bh * nqt, FA_THREADS, smem>>>(
                Q_d, K_d, V_d, O_d, sl, hd, ca_bench, scale, 1.0f, 1.0f, wnd);
        CK(cudaDeviceSynchronize());

        // Variance ×3: best/median/worst of 3 independent measurements.
        const int VARIANCE_RUNS = 3;
        double tf_runs[VARIANCE_RUNS];
        int it = 50;
        for (int v = 0; v < VARIANCE_RUNS; v++) {
            cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
            cudaEventRecord(t0);
            for (int i = 0; i < it; i++)
                fa115_kernel<<<bh * nqt, FA_THREADS, smem>>>(
                    Q_d, K_d, V_d, O_d, sl, hd, ca_bench, scale, 1.0f, 1.0f, wnd);
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            float ms; cudaEventElapsedTime(&ms, t0, t1);
            ms /= it;
            double sl_eff = (wnd > 0) ? (double)wnd : (double)sl;
            if (ca_bench && wnd == 0) sl_eff = (double)sl / 2.0;
            double flops = 4.0 * (double)bh * (double)sl * sl_eff * (double)hd;
            tf_runs[v] = flops / (ms / 1000.0) / 1e12;
            cudaEventDestroy(t0); cudaEventDestroy(t1);
        }
        // Sort: tf_runs[0] = min, [VARIANCE_RUNS-1] = max (asc sort)
        for (int a = 0; a < VARIANCE_RUNS-1; a++)
            for (int b = a+1; b < VARIANCE_RUNS; b++)
                if (tf_runs[a] > tf_runs[b]) { double t = tf_runs[a]; tf_runs[a] = tf_runs[b]; tf_runs[b] = t; }
        double sum = 0;
        for (int v = 0; v < VARIANCE_RUNS; v++) sum += tf_runs[v];
        double mean = sum / VARIANCE_RUNS;
        double var = 0;
        for (int v = 0; v < VARIANCE_RUNS; v++) var += (tf_runs[v] - mean) * (tf_runs[v] - mean);
        double sd = (VARIANCE_RUNS > 1) ? sqrt(var / (VARIANCE_RUNS - 1)) : 0.0;
        printf("  bh=%-3d sl=%-4d wnd=%-4d  |  %6.1f / %6.1f / %6.1f T  |  mean=%6.1f sd=%4.1f\n",
               bh, sl, wnd,
               tf_runs[VARIANCE_RUNS-1], tf_runs[VARIANCE_RUNS/2], tf_runs[0],
               mean, sd);
        cudaFree(Q_d); cudaFree(K_d); cudaFree(V_d); cudaFree(O_d);
    }

    return 0;
}
