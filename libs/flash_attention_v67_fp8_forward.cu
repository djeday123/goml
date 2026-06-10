// =============================================================================
// FlashAttention v67 — FP8 Forward + TMA conveyor (sm_120a)
// =============================================================================
// Same algorithm as v66 (Br=128, Bc=64, 4 warps × M_TILES=2 × 16 rows).
// Same MMA layout (PTX-doc m16n8k32 verified by mma_probe_sm120 → 32/32 lanes).
//
// ONLY change vs v66: replace per-chunk cp.async.cg loads of Q/K/V with one
// cp.async.bulk.tensor.2d (TMA) per tile, with mbarrier-based completion.
// TMA SASS family on sm_120a = UTMALDG (confirmed via nv_isa_solver catalog).
//
// Reasoning: v66 issues ~512 cp.async.cg per Q-tile and ~1024 per K/V double-
// buffer cycle. TMA condenses each tile to 1 issued instruction → frees issue
// slots for MMAs. Expected gain: +5-15% if instruction-issue bound; potentially
// 0% if pure compute-bound.
//
// Build:
//   nvcc -gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 \
//     libs/flash_attention_v67_fp8_forward.cu -o runs/fa_v67_fp8 -lcudart -lcuda
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define FA_BR 128
#define FA_BC 64
#define FA_THREADS 128
#define FA_STRIDE 128
#define M_TILES 2

#define BYTES_Q (FA_BR * FA_STRIDE)        // 16384
#define BYTES_KV (FA_BC * FA_STRIDE)       //  8192

// ----------------------------------------------------------------------------
// TMA & mbarrier PTX helpers
// ----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t s2u32(const void *p) {
    return (uint32_t)__cvta_generic_to_shared(const_cast<void*>(p));
}

__device__ __forceinline__ void mbarrier_init(uint64_t *mbar, int count) {
    uint32_t bar = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t *mbar, int bytes) {
    uint32_t bar = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                 :: "r"(bar), "r"(bytes));
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t *mbar, int phase) {
    uint32_t bar = __cvta_generic_to_shared(mbar);
    asm volatile(
        "{                                                       \n"
        "  .reg .pred p;                                         \n"
        "  WAIT_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "  @p bra DONE_%=;                                       \n"
        "  bra WAIT_%=;                                          \n"
        "  DONE_%=:                                              \n"
        "}" :: "r"(bar), "r"(phase));
}

// 2D TMA load: copy tile [boxRows × boxCols] from gmem (described by tensorMap)
// to SMEM (dst). Issuer must call mbarrier_arrive_expect_tx beforehand and all
// readers must mbarrier_wait_parity afterwards.
//
// coordinate order = (col_index, row_index) — fastest dim first per TMA spec.
__device__ __forceinline__ void tma_load_2d(
    void *dst, const CUtensorMap *tensor_map, int x, int y, uint64_t *mbar)
{
    uint32_t s = __cvta_generic_to_shared(dst);
    uint32_t b = __cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(s), "l"(tensor_map), "r"(x), "r"(y), "r"(b));
}

__device__ __forceinline__ void cp_async_bulk_commit() {
    asm volatile("cp.async.bulk.commit_group;");
}

// ----------------------------------------------------------------------------
// MMA helper (FP8 f16-acc, identical to v66)
// ----------------------------------------------------------------------------
__device__ __forceinline__ void mma_fp8_f16(
    uint32_t &d0, uint32_t &d1,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t c0, uint32_t c1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e4m3.e4m3.f16 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(c0), "r"(c1));
}

// Same swizzle as v66 — must match TMA's 128B swizzle pattern so MMA reads are
// bank-conflict-free. 128B-swizzle on uint8 stride-128 row: chunk ^= (row & 7).
__device__ __forceinline__ int swz_byte(int row, int col_bytes)
{
    int chunk = col_bytes >> 4;
    int within = col_bytes & 15;
    return row * FA_STRIDE + ((chunk ^ (row & 7)) << 4) + within;
}

__device__ __forceinline__ int swz_byte_bc(int row, int col_bytes)
{
    int chunk = col_bytes >> 4;
    int within = col_bytes & 15;
    return row * FA_BC + ((chunk ^ (row & 3)) << 4) + within;
}

__device__ __forceinline__ uint16_t fp16x2_to_e4m3x2(uint32_t h2)
{
    uint16_t out;
    asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(out) : "r"(h2));
    return out;
}

__device__ __forceinline__ void transpose_v(
    uint8_t *smV_T, const uint8_t *smV, int head_dim)
{
    constexpr int TILE = 4;
    int tiles_k = FA_BC / TILE;
    int tiles_n = head_dim / TILE;
    int total = tiles_k * tiles_n;
    for (int t = threadIdx.x; t < total; t += FA_THREADS)
    {
        int tk = t / tiles_n, tn = t % tiles_n;
        int k0 = tk * TILE, n0 = tn * TILE;
        uint32_t r0 = *(uint32_t *)&smV[swz_byte(k0 + 0, n0)];
        uint32_t r1 = *(uint32_t *)&smV[swz_byte(k0 + 1, n0)];
        uint32_t r2 = *(uint32_t *)&smV[swz_byte(k0 + 2, n0)];
        uint32_t r3 = *(uint32_t *)&smV[swz_byte(k0 + 3, n0)];
        uint32_t c0 = ((r0 >>  0) & 0xff) | ((r1 <<  8) & 0xff00)
                    | ((r2 << 16) & 0xff0000) | ((r3 << 24) & 0xff000000);
        uint32_t c1 = ((r0 >>  8) & 0xff) | ((r1 <<  0) & 0xff00)
                    | ((r2 <<  8) & 0xff0000) | ((r3 << 16) & 0xff000000);
        uint32_t c2 = ((r0 >> 16) & 0xff) | ((r1 >>  8) & 0xff00)
                    | ((r2 <<  0) & 0xff0000) | ((r3 <<  8) & 0xff000000);
        uint32_t c3 = ((r0 >> 24) & 0xff) | ((r1 >> 16) & 0xff00)
                    | ((r2 >>  8) & 0xff0000) | ((r3 <<  0) & 0xff000000);
        *(uint32_t *)&smV_T[swz_byte_bc(n0 + 0, k0)] = c0;
        *(uint32_t *)&smV_T[swz_byte_bc(n0 + 1, k0)] = c1;
        *(uint32_t *)&smV_T[swz_byte_bc(n0 + 2, k0)] = c2;
        *(uint32_t *)&smV_T[swz_byte_bc(n0 + 3, k0)] = c3;
    }
}

// ----------------------------------------------------------------------------
// Kernel
// ----------------------------------------------------------------------------
// tensorMap_* are passed by value (CUtensorMap is opaque 128-byte struct).
// All bh*nqt blocks share the same maps.
__global__ void __launch_bounds__(FA_THREADS, 2)
    fa67_kernel(
        const __grid_constant__ CUtensorMap tensorMap_Q,
        const __grid_constant__ CUtensorMap tensorMap_K,
        const __grid_constant__ CUtensorMap tensorMap_V,
        __half *__restrict__ O,
        int seq_len, int head_dim, int causal, float scale,
        float qk_descale, float v_descale)
{
    int nqt = (seq_len + FA_BR - 1) / FA_BR;
    int bh = blockIdx.x / nqt, qt = blockIdx.x % nqt, qs = qt * FA_BR;
    if (qs >= seq_len) return;
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    int gid = lane / 4, tid = lane % 4;
    int mrb = wid * 32;

    extern __shared__ uint8_t raw[];
    // Layout in dynamic smem:
    //   smQ                   (BYTES_Q = 16384)
    //   smK[0], smK[1]        (BYTES_KV each)
    //   smV[0], smV[1]        (BYTES_KV each)
    //   smV_T                 (8192)
    //   --- 8-byte aligned ---
    //   uint64_t mbar[5]      (Q, K[0], K[1], V[0], V[1])
    uint8_t *smQ = raw;
    uint8_t *smK[2] = { smQ + BYTES_Q, smQ + BYTES_Q + BYTES_KV };
    uint8_t *smV[2] = { smK[1] + BYTES_KV, smK[1] + 2 * BYTES_KV };
    uint8_t *smV_T  = smV[1] + BYTES_KV;
    uint64_t *mbar  = reinterpret_cast<uint64_t*>(smV_T + BYTES_Q / 2);  // +8192
    uint64_t *bar_Q  = &mbar[0];
    uint64_t *bar_K[2] = { &mbar[1], &mbar[2] };
    uint64_t *bar_V[2] = { &mbar[3], &mbar[4] };

    __half *Oh = O + bh * seq_len * head_dim;

    // Initialize all mbarriers once (single thread).
    if (threadIdx.x == 0) {
        mbarrier_init(bar_Q, 1);
        mbarrier_init(bar_K[0], 1);
        mbarrier_init(bar_K[1], 1);
        mbarrier_init(bar_V[0], 1);
        mbarrier_init(bar_V[1], 1);
    }
    __syncthreads();

    // Phase trackers — flip after each completed wait.
    int phase_Q = 0;
    int phase_K[2] = {0, 0};
    int phase_V[2] = {0, 0};

    // Issue Q load (lane 0 of warp 0).
    if (threadIdx.x == 0) {
        mbarrier_arrive_expect_tx(bar_Q, BYTES_Q);
        // tensorMap_Q is 3D: (head_dim, seq_len, bh_total). Coords = (col, row, bh).
        // We use cp.async.bulk.tensor.3d? Actually I'll use 2D maps per-tensor
        // but the tensor extends across bh — choose 3D. See host: 3D map.
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cluster.global.tile."
            "mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3, %4}], [%5];"
            :: "r"(s2u32(smQ)), "l"(&tensorMap_Q),
               "r"(0), "r"(qs), "r"(bh),
               "r"(s2u32(bar_Q)));
    }

    // Pre-load K[0], V[0] for iter 0.
    if (threadIdx.x == 0) {
        mbarrier_arrive_expect_tx(bar_K[0], BYTES_KV);
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cluster.global.tile."
            "mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3, %4}], [%5];"
            :: "r"(s2u32(smK[0])), "l"(&tensorMap_K),
               "r"(0), "r"(0), "r"(bh),
               "r"(s2u32(bar_K[0])));
        mbarrier_arrive_expect_tx(bar_V[0], BYTES_KV);
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cluster.global.tile."
            "mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3, %4}], [%5];"
            :: "r"(s2u32(smV[0])), "l"(&tensorMap_V),
               "r"(0), "r"(0), "r"(bh),
               "r"(s2u32(bar_V[0])));
    }

    // Wait Q load.
    mbarrier_wait_parity(bar_Q, phase_Q); phase_Q ^= 1;

    // Pre-stage Q register file (same as v66).
    uint32_t Qr[4][M_TILES][4];
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
        int k_off = ks * 32;
        int cl = k_off + tid * 4, ch = cl + 16;
#pragma unroll
        for (int mi = 0; mi < M_TILES; mi++) {
            int mr = mrb + mi * 16;
            int g0 = mr + gid, g8 = g0 + 8;
            Qr[ks][mi][0] = *(uint32_t *)&smQ[swz_byte(g0, cl)];
            Qr[ks][mi][1] = *(uint32_t *)&smQ[swz_byte(g8, cl)];
            Qr[ks][mi][2] = *(uint32_t *)&smQ[swz_byte(g0, ch)];
            Qr[ks][mi][3] = *(uint32_t *)&smQ[swz_byte(g8, ch)];
        }
    }

    uint32_t Or_p[16][M_TILES][2];
#pragma unroll
    for (int t = 0; t < 16; t++)
#pragma unroll
        for (int mi = 0; mi < M_TILES; mi++)
            Or_p[t][mi][0] = Or_p[t][mi][1] = 0u;

    float rmax[M_TILES][2] = {{-1e30f, -1e30f}, {-1e30f, -1e30f}};
    float rsexp[M_TILES][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    int nkv = (seq_len + FA_BC - 1) / FA_BC;
    int kv_max_blocks = causal ? ((qs + FA_BR - 1) / FA_BC + 1) : nkv;
    if (kv_max_blocks > nkv) kv_max_blocks = nkv;

    for (int kv = 0; kv < kv_max_blocks; kv++)
    {
        int kvs = kv * FA_BC;
        int buf = kv & 1;
        uint8_t *cur_K = smK[buf];
        uint8_t *cur_V = smV[buf];
        int nxt_buf = buf ^ 1;

        // Wait cur_K, cur_V.
        mbarrier_wait_parity(bar_K[buf], phase_K[buf]); phase_K[buf] ^= 1;
        mbarrier_wait_parity(bar_V[buf], phase_V[buf]); phase_V[buf] ^= 1;
        __syncthreads();

        transpose_v(smV_T, cur_V, head_dim);
        __syncthreads();

        uint8_t *smP = cur_V;  // reuse cur_V slot

        // Prefetch iter kv+1 into nxt_buf via TMA.
        if (kv + 1 < kv_max_blocks) {
            int nxt_kvs = (kv + 1) * FA_BC;
            if (threadIdx.x == 0) {
                mbarrier_arrive_expect_tx(bar_K[nxt_buf], BYTES_KV);
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile."
                    "mbarrier::complete_tx::bytes "
                    "[%0], [%1, {%2, %3, %4}], [%5];"
                    :: "r"(s2u32(smK[nxt_buf])),
                       "l"(&tensorMap_K),
                       "r"(0), "r"(nxt_kvs), "r"(bh),
                       "r"(s2u32(bar_K[nxt_buf])));
                mbarrier_arrive_expect_tx(bar_V[nxt_buf], BYTES_KV);
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile."
                    "mbarrier::complete_tx::bytes "
                    "[%0], [%1, {%2, %3, %4}], [%5];"
                    :: "r"(s2u32(smV[nxt_buf])),
                       "l"(&tensorMap_V),
                       "r"(0), "r"(nxt_kvs), "r"(bh),
                       "r"(s2u32(bar_V[nxt_buf])));
            }
        }

        // ----------- Compute (identical to v66) -----------
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
                }
            }
        }

        float lmax[M_TILES][2];
#pragma unroll
        for (int mi = 0; mi < M_TILES; mi++) {
            lmax[mi][0] = -1e30f; lmax[mi][1] = -1e30f;
        }
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++) {
                lmax[mi][0] = fmaxf(lmax[mi][0], fmaxf(Sr[nt][mi][0], Sr[nt][mi][1]));
                lmax[mi][1] = fmaxf(lmax[mi][1], fmaxf(Sr[nt][mi][2], Sr[nt][mi][3]));
            }
#pragma unroll
        for (int mi = 0; mi < M_TILES; mi++) {
            lmax[mi][0] = fmaxf(lmax[mi][0], __shfl_xor_sync(0xffffffff, lmax[mi][0], 1));
            lmax[mi][0] = fmaxf(lmax[mi][0], __shfl_xor_sync(0xffffffff, lmax[mi][0], 2));
            lmax[mi][1] = fmaxf(lmax[mi][1], __shfl_xor_sync(0xffffffff, lmax[mi][1], 1));
            lmax[mi][1] = fmaxf(lmax[mi][1], __shfl_xor_sync(0xffffffff, lmax[mi][1], 2));
        }

        float nmax[M_TILES][2], rsc[M_TILES][2];
#pragma unroll
        for (int mi = 0; mi < M_TILES; mi++)
        {
            nmax[mi][0] = fmaxf(rmax[mi][0], lmax[mi][0]);
            nmax[mi][1] = fmaxf(rmax[mi][1], lmax[mi][1]);
            rsc[mi][0] = (rmax[mi][0] > -1e29f) ? expf(rmax[mi][0] - nmax[mi][0]) : 0.0f;
            rsc[mi][1] = (rmax[mi][1] > -1e29f) ? expf(rmax[mi][1] - nmax[mi][1]) : 0.0f;
        }

        float P_local[8][M_TILES][4];
        float ns[M_TILES][2] = {{0,0},{0,0}};
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                P_local[nt][mi][0] = expf(Sr[nt][mi][0] - nmax[mi][0]);
                P_local[nt][mi][1] = expf(Sr[nt][mi][1] - nmax[mi][0]);
                P_local[nt][mi][2] = expf(Sr[nt][mi][2] - nmax[mi][1]);
                P_local[nt][mi][3] = expf(Sr[nt][mi][3] - nmax[mi][1]);
                ns[mi][0] += P_local[nt][mi][0] + P_local[nt][mi][1];
                ns[mi][1] += P_local[nt][mi][2] + P_local[nt][mi][3];
            }
#pragma unroll
        for (int t = 0; t < 16; t++)
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                __half2 c0 = *reinterpret_cast<__half2 *>(&Or_p[t][mi][0]);
                __half2 c1 = *reinterpret_cast<__half2 *>(&Or_p[t][mi][1]);
                float o0 = __half2float(__low2half(c0))  * rsc[mi][0];
                float o1 = __half2float(__high2half(c0)) * rsc[mi][0];
                float o2 = __half2float(__low2half(c1))  * rsc[mi][1];
                float o3 = __half2float(__high2half(c1)) * rsc[mi][1];
                __half2 r0 = __halves2half2(__float2half(o0), __float2half(o1));
                __half2 r1 = __halves2half2(__float2half(o2), __float2half(o3));
                Or_p[t][mi][0] = *reinterpret_cast<uint32_t*>(&r0);
                Or_p[t][mi][1] = *reinterpret_cast<uint32_t*>(&r1);
            }

#pragma unroll
        for (int mi = 0; mi < M_TILES; mi++)
        {
            rmax[mi][0] = nmax[mi][0]; rmax[mi][1] = nmax[mi][1];
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

#pragma unroll
        for (int ks = 0; ks < 2; ks++)
        {
            int k_off = ks * 32;
            int cl = k_off + tid * 4, ch = cl + 16;
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
                uint32_t b0 = *(uint32_t *)&smV_T[swz_byte_bc(br + gid, cl)];
                uint32_t b1 = *(uint32_t *)&smV_T[swz_byte_bc(br + gid, ch)];
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

// ----------------------------------------------------------------------------
// Host
// ----------------------------------------------------------------------------
#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); }} while(0)
#define CKU(c) do { CUresult r = (c); if (r != CUDA_SUCCESS) { \
    const char *s; cuGetErrorString(r, &s); \
    fprintf(stderr, "CU %s:%d: %s\n", __FILE__, __LINE__, s); exit(1); }} while(0)

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
static inline float fp16f(uint16_t h) { __half hv; memcpy(&hv, &h, 2); return __half2float(hv); }

static void cpu_attention_fp8(
    const uint8_t *Q, const uint8_t *K, const uint8_t *V,
    float *O_out, int bh, int sl, int hd, int causal)
{
    float scale = 1.0f / sqrtf((float)hd);
    int hs = sl * hd;
    for (int h = 0; h < bh; h++) {
        const uint8_t *Qh = Q + h * hs; const uint8_t *Kh = K + h * hs;
        const uint8_t *Vh = V + h * hs; float *Oh = O_out + h * hs;
        for (int q = 0; q < sl; q++) {
            int kv_max = causal ? (q + 1) : sl;
            float *P = (float *)malloc(sizeof(float) * sl);
            float rmax = -1e30f;
            for (int k = 0; k < kv_max; k++) {
                float s = 0;
                for (int d = 0; d < hd; d++)
                    s += e4m3_to_float(Qh[q * hd + d]) * e4m3_to_float(Kh[k * hd + d]);
                P[k] = s * scale; if (P[k] > rmax) rmax = P[k];
            }
            float rsum = 0;
            for (int k = 0; k < kv_max; k++) { P[k] = expf(P[k] - rmax); rsum += P[k]; }
            for (int k = 0; k < kv_max; k++) P[k] /= rsum;
            for (int d = 0; d < hd; d++) {
                float o = 0;
                for (int k = 0; k < kv_max; k++) o += P[k] * e4m3_to_float(Vh[k * hd + d]);
                Oh[q * hd + d] = o;
            }
            free(P);
        }
    }
}

// Build a 3D CUtensorMap for FP8 tensor of shape (bh, sl, hd) with
// box (1, tile_rows, hd) and 128B swizzle.
static void make_tensor_map(
    CUtensorMap *map, void *base, int bh, int sl, int hd, int tile_rows)
{
    // PTX TMA: dim ordering is fastest-varying first.
    // Tensor in memory: bh slowest, sl middle, hd fastest.
    cuuint64_t globalDim[3] = { (cuuint64_t)hd, (cuuint64_t)sl, (cuuint64_t)bh };
    // globalStrides has rank-1 entries, in bytes, for dims 1..rank-1.
    cuuint64_t globalStrides[2] = { (cuuint64_t)hd, (cuuint64_t)hd * sl };
    cuuint32_t boxDim[3] = { (cuuint32_t)hd, (cuuint32_t)tile_rows, 1 };
    cuuint32_t elementStrides[3] = { 1, 1, 1 };
    CKU(cuTensorMapEncodeTiled(
        map,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        3, base, globalDim, globalStrides,
        boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

int main()
{
    cuInit(0);
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    int clk = 0; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    printf("=== FA v67 FP8 forward + TMA conveyor ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n\n", p.name, p.multiProcessorCount, clk / 1000);

    printf("--- Correctness vs CPU FP8-roundtripped reference ---\n");
    int configs[][4] = {
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

        CUtensorMap mapQ, mapK, mapV;
        make_tensor_map(&mapQ, Q_d, bh, sl, hd, FA_BR);
        make_tensor_map(&mapK, K_d, bh, sl, hd, FA_BC);
        make_tensor_map(&mapV, V_d, bh, sl, hd, FA_BC);

        // SMEM: smQ + 2×smK + 2×smV + smV_T + mbar
        int smem = BYTES_Q + 2*BYTES_KV + 2*BYTES_KV + BYTES_Q/2 + 5 * 8;
        CK(cudaFuncSetAttribute(fa67_kernel,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        int nqt = (sl + FA_BR - 1) / FA_BR;
        float scale = 1.0f / sqrtf((float)hd);
        fa67_kernel<<<bh * nqt, FA_THREADS, smem>>>(
            mapQ, mapK, mapV, O_d, sl, hd, ca, scale, 1.0f, 1.0f);
        cudaError_t le = cudaDeviceSynchronize();
        if (le != cudaSuccess) {
            printf("  bh=%d sl=%d hd=%d ca=%d  LAUNCH FAILED: %s\n",
                   bh, sl, hd, ca, cudaGetErrorString(le));
            free(Qf); free(Kf); free(Vf); free(Qq); free(Kq); free(Vq); free(O_ref);
            cudaFree(Q_d); cudaFree(K_d); cudaFree(V_d); cudaFree(O_d);
            continue;
        }

        uint16_t *O_cpu = (uint16_t *)malloc(n_elems * 2);
        CK(cudaMemcpy(O_cpu, O_d, n_elems * 2, cudaMemcpyDeviceToHost));

        float mx = 0; int errs = 0;
        for (size_t i = 0; i < n_elems; i++) {
            float gpu = fp16f(O_cpu[i]); float ref = O_ref[i];
            float ae = fabsf(gpu - ref); if (ae > mx) mx = ae;
            if (ae > fmaxf(0.05f, fabsf(ref) * 0.1f)) errs++;
        }
        printf("  bh=%d sl=%d hd=%d ca=%d  max_diff=%.4f errs=%d → %s\n",
               bh, sl, hd, ca, mx, errs, errs == 0 ? "PASS" : "FAIL");

        free(Qf); free(Kf); free(Vf); free(Qq); free(Kq); free(Vq);
        free(O_ref); free(O_cpu);
        cudaFree(Q_d); cudaFree(K_d); cudaFree(V_d); cudaFree(O_d);
    }

    printf("\n--- Performance ---\n");
    int bench_configs[][3] = {
        {4, 1024, 128}, {4, 2048, 128}, {8, 2048, 128}, {4, 4096, 128},
    };
    for (auto &c : bench_configs)
    {
        int bh = c[0], sl = c[1], hd = c[2];
        size_t n_elems = (size_t)bh * sl * hd;
        uint8_t *Q_d, *K_d, *V_d; __half *O_d;
        CK(cudaMalloc(&Q_d, n_elems)); CK(cudaMalloc(&K_d, n_elems)); CK(cudaMalloc(&V_d, n_elems));
        CK(cudaMalloc(&O_d, n_elems * 2));
        CK(cudaMemset(Q_d, 0x38, n_elems)); CK(cudaMemset(K_d, 0x38, n_elems)); CK(cudaMemset(V_d, 0x38, n_elems));

        CUtensorMap mapQ, mapK, mapV;
        make_tensor_map(&mapQ, Q_d, bh, sl, hd, FA_BR);
        make_tensor_map(&mapK, K_d, bh, sl, hd, FA_BC);
        make_tensor_map(&mapV, V_d, bh, sl, hd, FA_BC);

        int smem = BYTES_Q + 2*BYTES_KV + 2*BYTES_KV + BYTES_Q/2 + 5 * 8;
        CK(cudaFuncSetAttribute(fa67_kernel,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        int nqt = (sl + FA_BR - 1) / FA_BR;
        float scale = 1.0f / sqrtf((float)hd);

        for (int i = 0; i < 5; i++)
            fa67_kernel<<<bh * nqt, FA_THREADS, smem>>>(
                mapQ, mapK, mapV, O_d, sl, hd, 0, scale, 1.0f, 1.0f);
        CK(cudaDeviceSynchronize());

        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        int it = 50;
        cudaEventRecord(t0);
        for (int i = 0; i < it; i++)
            fa67_kernel<<<bh * nqt, FA_THREADS, smem>>>(
                mapQ, mapK, mapV, O_d, sl, hd, 0, scale, 1.0f, 1.0f);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= it;
        double flops = 4.0 * (double)bh * (double)sl * (double)sl * (double)hd;
        double tf = flops / (ms / 1000.0) / 1e12;
        printf("  bh=%d sl=%d hd=%d  time=%.3f ms  perf=%.1f TFLOPS\n",
               bh, sl, hd, ms, tf);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaFree(Q_d); cudaFree(K_d); cudaFree(V_d); cudaFree(O_d);
    }
    return 0;
}
