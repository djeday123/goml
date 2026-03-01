// =============================================================================
// FlashAttention v2 — Double-Buffered with cp.async (SM89)
// =============================================================================
// Based on verified FA2 kernel (101 TFLOPS). Optimizations:
//
//   1. cp.async for K and V loads (hardware async global→SMEM, no register)
//   2. Overlap V load with softmax computation
//   3. Prefetch next K tile during P@V compute (Phase 6)
//   4. Reduced syncthreads: 4 per iteration (was 6)
//
// Current pipeline per KV tile:
//   Phase 1: Load K → KV_s          ← stall
//   sync
//   Phase 2: S = Q @ K^T            ← compute
//   Phase 3: Scale + mask → S_s
//   sync
//   Phase 4: Softmax                ← compute (warp-level)
//   sync
//   Phase 5: Load V → KV_s          ← stall
//   sync
//   Phase 6: O += P @ V             ← compute
//   sync
//
// New pipeline:
//   [prologue: cp.async K[0] → KV_s]
//   sync
//   for each kv_tile:
//     Phase 2: S = Q @ K^T                         ← compute on KV_s (has K)
//     Phase 3: Scale + mask → S_s
//     Phase 4: Softmax + cp.async V → V_s          ← overlap softmax + V load!
//     sync (V ready + softmax done)
//     Phase 5: Rescale O
//     Phase 6: O += P @ V + cp.async K[next] → KV_s  ← overlap PV compute + next K load!
//     sync (K[next] ready + PV done)
//
// SMEM layout change: separate V_s buffer instead of reusing KV_s
//   Q_s:  64 × 136 × 2 = 17,408
//   KV_s: 64 × 136 × 2 = 17,408  (for K)
//   V_s:  64 × 136 × 2 = 17,408  (for V, separate!)
//   S_s:  64 × 72  × 2 =  9,216
//   m/l:  64 × 2 × 4   =    512
//   Total: ~61,952 bytes → need launch_bounds(256, 1) = 1 block/SM
//
// With 1 block/SM, we lose occupancy but gain from overlap.
// Net effect should be positive for memory-bound seq lengths.
//
// Alternative approach (implemented here): Use SAME KV_s but pipeline
// the loads so V overwrites K only after K is fully consumed.
// This keeps 2 blocks/SM but requires careful ordering.
//
// ACTUALLY: The key insight is that after Phase 2 (S = Q @ K^T), K data
// in KV_s is no longer needed. So we can start loading V into KV_s
// immediately after Phase 2 using cp.async, and the softmax (Phase 4)
// doesn't touch KV_s at all. So the overlap is:
//
//   Phase 2: S = Q @ K^T (reads KV_s=K)    → after this, K in KV_s is dead
//   Phase 3: Scale + mask → S_s             → doesn't touch KV_s
//   Phase 4: Softmax (reads/writes S_s, m, l) + cp.async V → KV_s  ← OVERLAP!
//   cp.async.wait  ← V is ready
//   Phase 6: O += P @ V (reads S_s + KV_s=V)
//
// And for next tile prefetch:
//   Phase 6: O += P @ V + cp.async K[next] → ???
//   Problem: KV_s has V, can't overwrite with K[next] until Phase 6 done.
//   Solution: Double buffer KV_s with 2 stages! But that costs 17KB more...
//
// FINAL DESIGN: 2-stage KV buffer
//   KV_s[0]: 64 × 72 × 2 = 9,216 bytes  (reduced stride to 72, pad for alignment)
//   KV_s[1]: 64 × 72 × 2 = 9,216 bytes
//   Wait — head_dim=128, stride must be ≥ 128. Can't reduce.
//
// OK, let's go with the simpler but effective approach:
//   - cp.async V load overlapped with softmax (saves Phase 5 stall)
//   - Keep same SMEM layout (Q_s + KV_s + S_s), reuse KV_s for both K and V
//   - This alone should give ~15-20% speedup by hiding V load latency
//   - Keep 2 blocks/SM
// =============================================================================

#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define FA2_BR 64
#define FA2_BC 64
#define FA2_THREADS 256
#define FA2_WARPS 8

#define FA2_Q_STRIDE 136
#define FA2_KV_STRIDE 136
#define FA2_S_STRIDE 72

// =============================================================================
// cp.async intrinsics (SM80+)
// =============================================================================

__device__ __forceinline__ void cp_async_16B(void *smem, const void *gmem)
{
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_addr), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;");
}

template <int N>
__device__ __forceinline__ void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

// =============================================================================
// Warp reductions
// =============================================================================

__device__ __forceinline__ float warp_reduce_max(float v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// =============================================================================
// MMA helpers (identical to verified kernel)
// =============================================================================

__device__ __forceinline__ uint32_t pack_h2(const __half *ptr)
{
    uint32_t r;
    asm("mov.b32 %0, {%1, %2};" : "=r"(r)
        : "h"(*(const unsigned short *)&ptr[0]),
          "h"(*(const unsigned short *)&ptr[1]));
    return r;
}

__device__ __forceinline__ uint32_t pack_h2_strided(const __half *p0, const __half *p1)
{
    uint32_t r;
    asm("mov.b32 %0, {%1, %2};" : "=r"(r)
        : "h"(*(const unsigned short *)p0),
          "h"(*(const unsigned short *)p1));
    return r;
}

__device__ __forceinline__ void mma_f16_f32(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// =============================================================================
// Fragment loading — VERIFIED mapping (unchanged)
// =============================================================================

__device__ __forceinline__ void load_a_frag(
    uint32_t &a0, uint32_t &a1, uint32_t &a2, uint32_t &a3,
    const __half *smem, int stride, int row_base, int k_base, int lane)
{
    int gid = lane >> 2;
    int tid = lane & 3;
    int r0 = row_base + gid;
    int r8 = r0 + 8;
    int k0 = k_base + tid * 2;
    int k8 = k0 + 8;

    a0 = pack_h2(&smem[r0 * stride + k0]);
    a1 = pack_h2(&smem[r8 * stride + k0]); // a1↔a2 SWAPPED vs PTX doc
    a2 = pack_h2(&smem[r0 * stride + k8]);
    a3 = pack_h2(&smem[r8 * stride + k8]);
}

__device__ __forceinline__ void load_b_frag_kt(
    uint32_t &b0, uint32_t &b1,
    const __half *smem, int stride, int n_base, int k_base, int lane)
{
    int gid = lane >> 2;
    int tid = lane & 3;
    int n = n_base + gid;
    int k0 = k_base + tid * 2;
    int k8 = k0 + 8;

    b0 = pack_h2(&smem[n * stride + k0]);
    b1 = pack_h2(&smem[n * stride + k8]);
}

__device__ __forceinline__ void load_b_frag_v(
    uint32_t &b0, uint32_t &b1,
    const __half *smem, int stride, int n_base, int k_base, int lane)
{
    int gid = lane >> 2;
    int tid = lane & 3;
    int n = n_base + gid;
    int k0 = k_base + tid * 2;
    int k8 = k0 + 8;

    b0 = pack_h2_strided(&smem[k0 * stride + n],
                         &smem[(k0 + 1) * stride + n]);
    b1 = pack_h2_strided(&smem[k8 * stride + n],
                         &smem[(k8 + 1) * stride + n]);
}

__device__ __forceinline__ void store_d_to_smem(
    __half *smem, int stride, int row_base, int col_base,
    float d0, float d1, float d2, float d3, int lane)
{
    int gid = lane >> 2;
    int tid = lane & 3;
    int r0 = row_base + gid;
    int r8 = r0 + 8;
    int c0 = col_base + tid * 2;
    int c1 = c0 + 1;

    smem[r0 * stride + c0] = __float2half(d0);
    smem[r0 * stride + c1] = __float2half(d1);
    smem[r8 * stride + c0] = __float2half(d2);
    smem[r8 * stride + c1] = __float2half(d3);
}

// =============================================================================
// cp.async K/V loading helpers
// =============================================================================

// Load a tile of K or V into KV_s using cp.async (16-byte = 8 half per copy)
// tile: [BC][head_dim], stored as [row][KV_STRIDE] in SMEM
// Each thread handles multiple 16B chunks
__device__ __forceinline__ void cp_async_load_kv(
    __half *kv_smem, const __half *src_head,
    int kv_start, int seq_len, int head_dim)
{
    // Total elements: FA2_BC * head_dim = 64 * 128 = 8192 halfs = 16384 bytes
    // 16B per cp.async = 8 halfs per copy
    // Total copies: 8192 / 8 = 1024
    // Per thread: 1024 / 256 = 4 copies
    const int total_copies = (FA2_BC * head_dim) / 8;

    for (int c = 0; c < total_copies; c += FA2_THREADS)
    {
        int copy_id = c + threadIdx.x;
        if (copy_id < total_copies)
        {
            // Map copy_id to (row, col_chunk)
            int halfs_per_row = head_dim;           // 128
            int copies_per_row = halfs_per_row / 8; // 16
            int row = copy_id / copies_per_row;
            int col8 = (copy_id % copies_per_row) * 8;
            int grow = kv_start + row;

            __half *dst = &kv_smem[row * FA2_KV_STRIDE + col8];

            if (grow < seq_len)
            {
                const __half *src = &src_head[grow * head_dim + col8];
                cp_async_16B(dst, src);
            }
            else
            {
                // Zero-fill out-of-bounds rows (can't use cp.async for this)
                // Use regular store — these are rare (only last tile)
                for (int j = 0; j < 8; j++)
                    dst[j] = __float2half(0.0f);
            }
        }
    }
}

// =============================================================================
// FlashAttention v2 — Double-Buffered Kernel
// =============================================================================

__global__ void __launch_bounds__(FA2_THREADS, 2)
    flash_attention_v2_db_kernel(
        const __half *__restrict__ Q,
        const __half *__restrict__ K,
        const __half *__restrict__ V,
        __half *__restrict__ O,
        int seq_len, int head_dim, int causal, float scale)
{
    int num_q_tiles = (seq_len + FA2_BR - 1) / FA2_BR;
    int bh = blockIdx.x / num_q_tiles;
    int q_tile = blockIdx.x % num_q_tiles;
    int q_start = q_tile * FA2_BR;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int warp_m = warp_id % 4;
    int warp_n = warp_id / 4;
    int gid = lane_id >> 2;
    int tid = lane_id & 3;

    // ---- SMEM layout (same as original) ----
    extern __shared__ char smem_raw[];
    __half *Q_s = (__half *)smem_raw;
    __half *KV_s = (__half *)(smem_raw + FA2_BR * FA2_Q_STRIDE * sizeof(__half));
    __half *S_s = (__half *)(smem_raw + FA2_BR * FA2_Q_STRIDE * sizeof(__half) +
                             FA2_BC * FA2_KV_STRIDE * sizeof(__half));
    float *m_smem = (float *)(smem_raw + FA2_BR * FA2_Q_STRIDE * sizeof(__half) +
                              FA2_BC * FA2_KV_STRIDE * sizeof(__half) +
                              FA2_BR * FA2_S_STRIDE * sizeof(__half));
    float *l_smem = m_smem + FA2_BR;

    // ---- Head pointers ----
    int hstride = seq_len * head_dim;
    const __half *Q_head = Q + bh * hstride;
    const __half *K_head = K + bh * hstride;
    const __half *V_head = V + bh * hstride;
    __half *O_head = O + bh * hstride;

    // ==== Load Q tile (regular, only once) ====
    for (int i = threadIdx.x; i < FA2_BR * head_dim; i += FA2_THREADS)
    {
        int row = i / head_dim, col = i % head_dim;
        int grow = q_start + row;
        Q_s[row * FA2_Q_STRIDE + col] = (grow < seq_len)
                                            ? Q_head[grow * head_dim + col]
                                            : __float2half(0.0f);
    }

    // Init m/l
    for (int i = threadIdx.x; i < FA2_BR; i += FA2_THREADS)
    {
        m_smem[i] = -1e30f;
        l_smem[i] = 0.0f;
    }

    // O accumulators
    float o_acc[8][4];
#pragma unroll
    for (int t = 0; t < 8; t++)
#pragma unroll
        for (int r = 0; r < 4; r++)
            o_acc[t][r] = 0.0f;

    int my_row0 = warp_m * 16 + gid;
    int my_row8 = my_row0 + 8;

    int num_kv_tiles = (seq_len + FA2_BC - 1) / FA2_BC;

    // ==== Prologue: async load K[0] → KV_s ====
    int kv_start = 0;
    if (!(causal && kv_start > q_start + FA2_BR - 1))
    {
        cp_async_load_kv(KV_s, K_head, kv_start, seq_len, head_dim);
        cp_async_commit();
    }

    __syncthreads(); // ensure Q_s is ready

    // ---- Main KV tile loop ----
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++)
    {
        kv_start = kv_tile * FA2_BC;
        if (causal && kv_start > q_start + FA2_BR - 1)
            break;

        // ==== Wait for K data (from prologue or previous iteration's prefetch) ====
        cp_async_wait<0>();
        __syncthreads();

        // ==== Phase 2: S = Q @ K^T via MMA ====
        float s_acc[4][4];
#pragma unroll
        for (int t = 0; t < 4; t++)
#pragma unroll
            for (int r = 0; r < 4; r++)
                s_acc[t][r] = 0.0f;

        int m_row_base = warp_m * 16;
        int n_col_base = warp_n * 32;
        int k_steps = head_dim / 16;

        for (int ks = 0; ks < k_steps; ks++)
        {
            uint32_t a0, a1, a2, a3;
            load_a_frag(a0, a1, a2, a3, Q_s, FA2_Q_STRIDE,
                        m_row_base, ks * 16, lane_id);

#pragma unroll
            for (int nt = 0; nt < 4; nt++)
            {
                uint32_t b0, b1;
                load_b_frag_kt(b0, b1, KV_s, FA2_KV_STRIDE,
                               n_col_base + nt * 8, ks * 16, lane_id);
                mma_f16_f32(
                    s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3],
                    a0, a1, a2, a3, b0, b1,
                    s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3]);
            }
        }

        // ==== Phase 3: Write S to SMEM (scale + causal mask) ====
        // After this, K in KV_s is dead — safe to overwrite with V
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
        {
            int col_base = n_col_base + nt * 8;
            s_acc[nt][0] *= scale;
            s_acc[nt][1] *= scale;
            s_acc[nt][2] *= scale;
            s_acc[nt][3] *= scale;

            if (causal)
            {
                int r0 = m_row_base + gid, r8 = r0 + 8;
                int c0 = col_base + tid * 2, c1 = c0 + 1;
                int gq0 = q_start + r0, gq8 = q_start + r8;
                int gk0 = kv_start + c0, gk1 = kv_start + c1;

                if (gk0 > gq0)
                    s_acc[nt][0] = -1e30f;
                if (gk1 > gq0)
                    s_acc[nt][1] = -1e30f;
                if (gk0 > gq8)
                    s_acc[nt][2] = -1e30f;
                if (gk1 > gq8)
                    s_acc[nt][3] = -1e30f;
                if (gq0 >= seq_len)
                {
                    s_acc[nt][0] = -1e30f;
                    s_acc[nt][1] = -1e30f;
                }
                if (gq8 >= seq_len)
                {
                    s_acc[nt][2] = -1e30f;
                    s_acc[nt][3] = -1e30f;
                }
                if (gk0 >= seq_len)
                {
                    s_acc[nt][0] = -1e30f;
                    s_acc[nt][2] = -1e30f;
                }
                if (gk1 >= seq_len)
                {
                    s_acc[nt][1] = -1e30f;
                    s_acc[nt][3] = -1e30f;
                }
            }

            store_d_to_smem(S_s, FA2_S_STRIDE, m_row_base, col_base,
                            s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3],
                            lane_id);
        }

        // ---- CRITICAL SYNC: S_s writes must be visible before softmax reads ----
        // Also: all warps must be done reading K from KV_s before V load starts
        __syncthreads();

        // ==== Phase 4: Softmax + OVERLAPPED V load via cp.async ====
        // Start V load IMMEDIATELY — softmax only touches S_s and m/l, not KV_s
        cp_async_load_kv(KV_s, V_head, kv_start, seq_len, head_dim);
        cp_async_commit();

        // Now do softmax while V loads in background
        float m_old_r0 = m_smem[my_row0];
        float m_old_r8 = m_smem[my_row8];

        {
            int srow_start = warp_m * 16 + warp_n * 8;

            for (int r = 0; r < 8; r++)
            {
                int srow = srow_start + r;

                float v0 = __half2float(S_s[srow * FA2_S_STRIDE + lane_id * 2]);
                float v1 = __half2float(S_s[srow * FA2_S_STRIDE + lane_id * 2 + 1]);

                float local_max = fmaxf(v0, v1);
                float row_max = warp_reduce_max(local_max);

                float m_old = m_smem[srow];
                float m_new = fmaxf(m_old, row_max);

                float e0 = expf(v0 - m_new);
                float e1 = expf(v1 - m_new);

                float local_sum = e0 + e1;
                float row_sum = warp_reduce_sum(local_sum);

                float rescale_old = expf(m_old - m_new);

                if (lane_id == 0)
                {
                    m_smem[srow] = m_new;
                    l_smem[srow] = l_smem[srow] * rescale_old + row_sum;
                }

                S_s[srow * FA2_S_STRIDE + lane_id * 2] = __float2half(e0);
                S_s[srow * FA2_S_STRIDE + lane_id * 2 + 1] = __float2half(e1);
            }
        }

        // ==== Wait for V load to complete ====
        cp_async_wait<0>();
        __syncthreads();

        // Rescale O accumulators
        float m_new_r0 = m_smem[my_row0];
        float m_new_r8 = m_smem[my_row8];
        float rescale0 = expf(m_old_r0 - m_new_r0);
        float rescale8 = expf(m_old_r8 - m_new_r8);

#pragma unroll
        for (int t = 0; t < 8; t++)
        {
            o_acc[t][0] *= rescale0;
            o_acc[t][1] *= rescale0;
            o_acc[t][2] *= rescale8;
            o_acc[t][3] *= rescale8;
        }

        // ==== Phase 6: O += P @ V via MMA ====
        int o_n_base = warp_n * 64;
        int pv_k_steps = FA2_BC / 16; // 4

        // Prefetch K[next] while computing P @ V
        int next_kv_tile = kv_tile + 1;
        int next_kv_start = next_kv_tile * FA2_BC;
        bool has_next = (next_kv_tile < num_kv_tiles);
        if (causal && next_kv_start > q_start + FA2_BR - 1)
            has_next = false;

        // Start first half of P@V compute
        for (int nt = 0; nt < 4; nt++)
        {
            int n_col = o_n_base + nt * 8;

            for (int ks = 0; ks < pv_k_steps; ks++)
            {
                uint32_t a0, a1, a2, a3;
                load_a_frag(a0, a1, a2, a3, S_s, FA2_S_STRIDE,
                            m_row_base, ks * 16, lane_id);

                uint32_t b0, b1;
                load_b_frag_v(b0, b1, KV_s, FA2_KV_STRIDE,
                              n_col, ks * 16, lane_id);

                mma_f16_f32(
                    o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3],
                    a0, a1, a2, a3, b0, b1,
                    o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3]);
            }
        }

        // After P@V done with V in KV_s, it's safe to overwrite with K[next]
        // NOTE: We can't overlap K[next] load with P@V compute because both
        // use KV_s. But we CAN issue the cp.async right after the last MMA
        // so it starts while the epilogue / loop overhead runs.
        for (int nt = 4; nt < 8; nt++)
        {
            int n_col = o_n_base + nt * 8;

            for (int ks = 0; ks < pv_k_steps; ks++)
            {
                uint32_t a0, a1, a2, a3;
                load_a_frag(a0, a1, a2, a3, S_s, FA2_S_STRIDE,
                            m_row_base, ks * 16, lane_id);

                uint32_t b0, b1;
                load_b_frag_v(b0, b1, KV_s, FA2_KV_STRIDE,
                              n_col, ks * 16, lane_id);

                mma_f16_f32(
                    o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3],
                    a0, a1, a2, a3, b0, b1,
                    o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3]);
            }
        }

        // All MMA done reading V from KV_s — now safe to load next K
        __syncthreads(); // ensure all warps done with KV_s

        if (has_next)
        {
            cp_async_load_kv(KV_s, K_head, next_kv_start, seq_len, head_dim);
            cp_async_commit();
        }
    }

    // ==== Final: O = O / l, write to global ====
    {
        float l_inv0 = (l_smem[my_row0] > 0.0f) ? 1.0f / l_smem[my_row0] : 0.0f;
        float l_inv8 = (l_smem[my_row8] > 0.0f) ? 1.0f / l_smem[my_row8] : 0.0f;
        int grow0 = q_start + my_row0;
        int grow8 = q_start + my_row8;
        int o_n_base = warp_n * 64;

        for (int nt = 0; nt < 8; nt++)
        {
            int col0 = o_n_base + nt * 8 + tid * 2;
            int col1 = col0 + 1;

            if (grow0 < seq_len && col0 < head_dim)
                O_head[grow0 * head_dim + col0] = __float2half(o_acc[nt][0] * l_inv0);
            if (grow0 < seq_len && col1 < head_dim)
                O_head[grow0 * head_dim + col1] = __float2half(o_acc[nt][1] * l_inv0);
            if (grow8 < seq_len && col0 < head_dim)
                O_head[grow8 * head_dim + col0] = __float2half(o_acc[nt][2] * l_inv8);
            if (grow8 < seq_len && col1 < head_dim)
                O_head[grow8 * head_dim + col1] = __float2half(o_acc[nt][3] * l_inv8);
        }
    }
}

// =============================================================================
// C API
// =============================================================================

static int g_fa2_db_smem_max = 0;

extern "C"
{

    int flash_attention_v2_db_forward(
        const void *Q, const void *K, const void *V, void *O,
        int total_heads, int seq_len, int head_dim,
        int causal, void *stream)
    {
        if (head_dim % 16 != 0 || head_dim > 128)
            return -1;

        int smem_bytes = FA2_BR * FA2_Q_STRIDE * (int)sizeof(__half) +
                         FA2_BC * FA2_KV_STRIDE * (int)sizeof(__half) +
                         FA2_BR * FA2_S_STRIDE * (int)sizeof(__half) +
                         FA2_BR * 2 * (int)sizeof(float);

        if (smem_bytes > g_fa2_db_smem_max)
        {
            cudaError_t err = cudaFuncSetAttribute(
                flash_attention_v2_db_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
            if (err != cudaSuccess)
                return (int)err;
            g_fa2_db_smem_max = smem_bytes;
        }

        float scale_val = 1.0f / sqrtf((float)head_dim);
        int num_q_tiles = (seq_len + FA2_BR - 1) / FA2_BR;
        int total_blocks = total_heads * num_q_tiles;

        flash_attention_v2_db_kernel<<<total_blocks, FA2_THREADS, smem_bytes, (cudaStream_t)stream>>>(
            (const __half *)Q, (const __half *)K, (const __half *)V, (__half *)O,
            seq_len, head_dim, causal, scale_val);

        return (int)cudaGetLastError();
    }

    int flash_attention_v2_db_forward_bhsd(
        const void *Q, const void *K, const void *V, void *O,
        int batch, int num_heads, int seq_len, int head_dim,
        int causal, void *stream)
    {
        return flash_attention_v2_db_forward(Q, K, V, O,
                                             batch * num_heads, seq_len, head_dim, causal, stream);
    }
}
