// =============================================================================
// FlashAttention v110 — STEP 3 of warp-spec probe: ROLE BALANCE 1P+3C @ Br=96
// =============================================================================
// v108/v109 had 2 producer + 2 consumer warps at Br=64. NCu showed math_pipe
// 6.59% (low = MMA undersaturated) and Eligible 26.88% vs v96's 32.90% (−6pp).
// Diagnosis: only 2 consumer warps = half v96's MMA throughput per block.
//
// v110 rebalances to 1 producer + 3 consumer warps:
//   wid=0:      PRODUCER (cp.async K + V via load_tile_fp8_warp)
//   wid=1,2,3:  CONSUMER (3 × M_TILES=2 × 16 = 96 rows = Br=96)
//
// SMEM constraint: Br=96 + double-V doesn't fit at 2 blocks/SM (52.5 KB × 2 = 105K).
// → V_STAGES=1 (single V buffer, v96-style). Producer loses V deep-pipeline,
//   but gains 1 more consumer warp = closer-to-v96 MMA throughput.
//
// SMEM/block = smQ(12K) + 2×smK(16K) + 1×smV(8K) + smV_T(8.5K) = 44.5 KB
//   2 blocks × 44.5 = 89 KB ≤ 100 KB cap → LB(128, 2) preserved
//
// Inherited from v109: bar.sync 1/2 consumer-only (arrival count = 96 now).
//
// Diagnostic interpretation:
//   v110 perf ≥ 450T → role balance works, warp-spec is competitive
//                    → step 4 / deeper buffer (needs ldmatrix.trans probe)
//   v110 perf 400-450 → marginal, warp-spec gives small signal but below v96
//   v110 perf <400   → 3 consumer + 1 producer still loses; CLOSE warp-spec
// =============================================================================

// ---- (original v109 docs)
// =============================================================================
// FlashAttention v109 — STEP 2 of warp-spec probe: NAMED-BAR consumer-only sync
// =============================================================================
// v108 NCu showed barrier stall 19.48% (vs v96's 2.00%, +17.48pp). Cause: producer
// warps idle-waited at block-wide __syncthreads while consumers ran QK/softmax/PV.
// v109 replaces 2 of 4 __syncthreads with NAMED BARRIERS (PTX bar.sync N, 64) that
// only consumer warps participate in. Producer no longer blocks at consumer-only
// barriers → can pipeline cp.async issues ahead.
//
// Replaced barriers:
//   __syncthreads after QK+softmax block  →  bar.sync 1, 64 (consumers only)
//   __syncthreads after smP STS block     →  bar.sync 2, 64 (consumers only)
//
// Kept block-wide (producer participates in cp.async/transpose):
//   __syncthreads after cpa_wait at iter top (block-wide)
//   __syncthreads after transpose_v        (block-wide)
//
// Hypothesis: barrier 19.48% → ~3-5% (producers skip 2 of 4 bars per iter).
//   If wait stays 39% → confirms wait is math-latency, not memory (warp-spec
//                       can't hide MMA-pipe dependencies; close warp-spec path)
//   If wait drops → producer truly running ahead now (continue to step 3 balance)
//
// Same as v108: Br=64, 2P+2C, K=2 V=2, smV_T, 48.5 KB, LB(128, 2).
// =============================================================================

// =============================================================================
// FlashAttention v108 — STEP 1 of warp-specialization probe for hd=128 sm_120a
// =============================================================================
// Goal of step 1: skeleton with producer/consumer warp role split, correctness
// 8/8 PASS. Perf likely below v96 (coordination overhead, fewer MMA warps).
//
// Architecture:
//   Br=64  (frees SMEM for deeper buffer: 4 stages combined vs v96's 3)
//   FA_THREADS=128 = 4 warps
//     wid 0,1: PRODUCER warps (cp.async K, V into double-buffered slots)
//     wid 2,3: CONSUMER warps (transpose_v, QK, softmax, PV MMA)
//   M_TILES=2 → 2 consumer warps × M_TILES=2 × 16 = Br=64 ✓
//   K_STAGES=2, V_STAGES=2 (4 stages combined, +1 over v96's V=1)
//   smP reuses smQ (Qr in regs after pre-loop → smQ free for softmax output)
//   Sync: __syncthreads handshake (mbarrier deferred to step 4)
//
// SMEM/block = smQ(8K) + 2×smK(16K) + 2×smV(16K) + smV_T(8.5K) = 48.5 KB
//   2 blocks × 48.5 = 97 KB ≤ 100 KB cap → LB(128, 2) preserved
//
// Baseline ref: v96 = 568 peak / 564 mean / wait 37.77% / Eligible 32.96%.
// Step 1 success criterion: correctness 8/8 PASS. Perf signal interpretation:
//   - wait↓ AND perf close to v96 → warp-spec mechanism works, iterate deeper
//   - wait↓ but perf <90% v96 → role overhead too high, tune balance (step 3)
//   - wait unchanged → producer not actually overlapping (debug pipeline)
// =============================================================================

// =============================================================================
// FlashAttention v69 — FP8 Forward, single-buffer V → 2 blocks/SM (original)
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

// v110 warp-spec step 3 constants — 1 producer + 3 consumer, Br=96
#define FA_BR 96         // 3 consumer × M_TILES=2 × 16 = 96 ✓
#define FA_BC 64
#define FA_THREADS 128   // 4 warps total: 1 producer + 3 consumer (= 96 threads)
#define FA_STRIDE 128
#define FA_PRODUCERS 1   // wid 0: cp.async K + V (alternating)
#define FA_CONSUMERS 3   // wid 1,2,3: compute. 3 × M_TILES=2 × 16 = Br=96 ✓
#define M_TILES 2        // each consumer warp owns 2 m16 sub-tiles = 32 rows
#define K_STAGES 2       // double-buffered K (producer lookahead 1 iter)
#define V_STAGES 1       // SINGLE V buffer (SMEM constraint; v96-style pattern)
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

// v108: single-warp loader — only `lane` threads (0..31) participate.
// Used by producer warps to load K/V independently. Br=64 K tile = 64×8=512 chunks
// → 16 cp.async issues per thread (manageable).
__device__ __forceinline__ void load_tile_fp8_warp(
    uint8_t *dst, const uint8_t *src, int start, int rows,
    int seq_len, int head_dim, int lane)
{
    constexpr int CHUNK = 16;
    int chunks_per_row = head_dim / CHUNK;
    int total = rows * chunks_per_row;
#pragma unroll
    for (int c = lane; c < total; c += 32)
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
// v111: producer-skip transpose — only consumer warps (96 threads, offset 32).
// Producer warp doesn't participate → producer doesn't wait at post-transpose sync.
// Stride = FA_CONSUMERS * 32 = 96 threads. Local thread id = threadIdx.x - 32.
__device__ __forceinline__ void transpose_v_consumer(
    uint8_t *smV_T, const uint8_t *smV, int head_dim)
{
    constexpr int TILE = 4;
    int tiles_k = FA_BC / TILE;       // 16
    int tiles_n = head_dim / TILE;     // 32 for hd=128
    int total = tiles_k * tiles_n;     // 512
    constexpr int CONSUMER_THREADS = FA_CONSUMERS * 32;  // 96
    constexpr int PRODUCER_THREADS = FA_PRODUCERS * 32;  // 32
    int t_local = threadIdx.x - PRODUCER_THREADS;  // 0..95 for consumer threads
    for (int t = t_local; t < total; t += CONSUMER_THREADS)
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

__global__ void __launch_bounds__(FA_THREADS, 2)
    fa111_kernel(
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
    // v110 warp-spec: wid 0 = producer (cp.async K+V), wid 1,2,3 = consumer (compute)
    bool is_producer = (wid < FA_PRODUCERS);
    int cwid = wid - FA_PRODUCERS;  // consumer index 0,1,2 (undefined for producer)
    int mrb = cwid * 32;  // consumer warps own 32 rows each; M_TILES=2 × 16 = 32

    extern __shared__ uint8_t raw[];
    uint8_t *smQ = raw;
    // v110 SMEM layout: smQ + 2×smK + 1×smV + smV_T = 12K + 16K + 8K + 8.5K = 44.5K
    // 2 blocks × 44.5 = 89 KB ≤ 100 KB cap ✓
    uint8_t *smK[K_STAGES] = {
        smQ + FA_BR * FA_STRIDE,
        smQ + FA_BR * FA_STRIDE + FA_BC * FA_STRIDE,
    };
    // v110: single V buffer (V_STAGES=1). Producer overwrites smV at top of next
    // iter; consumer's transpose_v at iter top extracts to smV_T before any reuse.
    uint8_t *smV = smK[1] + FA_BC * FA_STRIDE;
    uint8_t *smV_T = smV + FA_BC * FA_STRIDE;
    // smP scratchpad reuses smQ (Qr lives in regs after pre-loop → smQ is free
    // for softmax output). smP layout = FA_BR × FA_BC = 96 × 64 = 6 KB ≤ smQ's 12 KB.
    uint8_t *smP = smQ;

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
    // Consumer warps only; producers don't do MMA.
    uint32_t Qr[4][M_TILES][4];
    uint32_t Or_p[16][M_TILES][2];
    float rmax[M_TILES][2] = {{-1e30f, -1e30f}, {-1e30f, -1e30f}};
    float rsexp[M_TILES][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    if (!is_producer)
    {
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
#pragma unroll
        for (int t = 0; t < 16; t++)
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
                Or_p[t][mi][0] = Or_p[t][mi][1] = 0u;
    }
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

    // v110: PRE-LOAD via single producer warp (wid=0). Issues K and V cp.async
    // sequentially within the same warp. cp.async runs async — the issues queue up.
    if (wid == 0) {
        load_tile_fp8_warp(smK[kv_min_blocks & 1], Kh, kv_min_blocks * FA_BC, FA_BC,
                           seq_len, head_dim, lane);
        load_tile_fp8_warp(smV, Vh, kv_min_blocks * FA_BC, FA_BC,
                           seq_len, head_dim, lane);
    }
    cpa_commit();

    for (int kv = kv_min_blocks; kv < kv_max_blocks; kv++)
    {
        int kvs = kv * FA_BC;
        int buf = kv & 1;
        int nxt_buf = (kv + 1) & 1;
        uint8_t *cur_K = smK[buf];

        // Wait until current iter's K AND V load completes (producer issued them
        // either in pre-loop or last iter's mid-cp.async).
        cpa_wait<0>();
        __syncthreads();

        // v111: PRODUCER-SKIP transpose. Only consumer warps (96 threads, offset 32)
        // participate. Producer skips → falls straight to mid-iter cp.async.
        // Post-transpose sync is consumer-only via bar.sync 3, 96 (producer skips).
        // V race RISK: producer's mid-iter cp.async V[kv+1] starts ~same time as
        // consumer transpose. Consumer transpose ~100-200 cycles; cp.async HBM ~500+.
        // Empirically consumer transpose should FINISH before V cp.async LANDS.
        if (!is_producer) {
            transpose_v_consumer(smV_T, smV, head_dim);
            asm volatile("bar.sync 3, 96;");  // consumer-only post-transpose sync
        }

        // v110: producer-only mid-iter prefetch. Single producer warp loads BOTH
        // K[kv+1] AND V[kv+1] sequentially (V single-buf → overwrites smV after
        // transpose_v read it). cp.async issues queue and run in background during
        // consumer's QK+softmax+PV phase.
        int kv_p = kv + 1;
        int rows_p = (kv_p < kv_max_blocks) ? FA_BC : 0;
        if (wid == 0) {
            load_tile_fp8_warp(smK[nxt_buf], Kh, kv_p * FA_BC, rows_p,
                               seq_len, head_dim, lane);
            load_tile_fp8_warp(smV, Vh, kv_p * FA_BC, rows_p,
                               seq_len, head_dim, lane);
        }
        cpa_commit();

        // ===== QK + softmax: CONSUMER WARPS ONLY =====
        // Producers fall through to the smP barrier below (must reach __syncthreads).
        // P_top/P_bot declared in outer scope so they're visible in the smP STS block.
        uint32_t Sr_p[8][M_TILES][2];
        __half2 P_top[8][M_TILES], P_bot[8][M_TILES];
        if (!is_producer) {
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

        // v108: v78's mid-QK V prefetch removed. V[kv+1] is now prefetched by
        // producer warp 1 at top of iter into smV[nxt_buf] (double-buffer).

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
        // v108: P_top/P_bot hoisted to outer scope (before the QK+softmax if-block).
        float ns[M_TILES][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
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
        } // ===== end QK+softmax consumer-only block =====

        // v109: consumer-only named barrier (replaces block-wide __syncthreads).
        // Producer warps SKIP this — they pipeline straight to next iter's cp.async.
        // 64 = 2 consumer warps × 32 = expected arrivals at bar 1.
        if (!is_producer) {
            asm volatile("bar.sync 1, 96;");  // v110: 3 consumer warps × 32 = 96
        }

        // ===== smP STS: CONSUMER WARPS ONLY =====
        if (!is_producer)
        {
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
        }
        // v109: consumer-only named barrier (was block-wide __syncthreads).
        if (!is_producer) {
            asm volatile("bar.sync 2, 96;");  // v110: 3 consumer warps × 32 = 96
        }

        // ===== PV MMA: CONSUMER WARPS ONLY =====
        if (!is_producer)
        {
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
        } // ===== end PV consumer-only block =====
        // v108: end of kv-iter. Producers reach here after the smP barrier (no compute).
        // No end-of-iter __syncthreads — next iter's cpa_wait + sync handles ordering.
    }

    // Consumer-only epilogue: write Or_p / rsexp normalized to global O.
    if (!is_producer)
    {
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
// Runs ONE warmup + ONE measured launch of fa111_kernel on bench_configs[cfg_idx].
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
    CK(cudaFuncSetAttribute(fa111_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    fa111_kernel<<<grid, FA_THREADS, smem>>>(Q_d, K_d, V_d, O_d, sl, hd, ca, scale, 1.0f, 1.0f, wnd);
    CK(cudaDeviceSynchronize());
    fa111_kernel<<<grid, FA_THREADS, smem>>>(Q_d, K_d, V_d, O_d, sl, hd, ca, scale, 1.0f, 1.0f, wnd);
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
    CK(cudaFuncSetAttribute(fa111_kernel,
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
        fa111_kernel<<<grid, FA_THREADS, smem>>>(Q_d, K_d, V_d, O_d, sl, hd, ca, scale, 1.0f, 1.0f, wnd);
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
            fa111_kernel<<<grid, FA_THREADS, smem>>>(Q_d, K_d, V_d, O_d, sl, hd, ca, scale, 1.0f, 1.0f, wnd);
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
    printf("=== FA v111 PRODUCER-SKIP transpose (1P+3C Br=96, consumer-only bar.sync 96) ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n", p.name, p.multiProcessorCount, clk / 1000);

    // Report actual kernel attributes (real SMEM use + reg count).
    cudaFuncAttributes attr;
    CK(cudaFuncGetAttributes(&attr, fa111_kernel));
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
        CK(cudaFuncSetAttribute(fa111_kernel,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        int nqt = (sl + FA_BR - 1) / FA_BR;
        float scale = 1.0f / sqrtf((float)hd);
        fa111_kernel<<<bh * nqt, FA_THREADS, smem>>>(
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
            fa111_kernel<<<bh * nqt, FA_THREADS, smem>>>(
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
                fa111_kernel<<<bh * nqt, FA_THREADS, smem>>>(
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
