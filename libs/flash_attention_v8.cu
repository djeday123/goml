// =============================================================================
// FlashAttention v8 — Warp Specialization (SM89)
// =============================================================================
// Architecture: 8 warps split into 2 roles:
//   MMA warps  (0-3): QK^T compute, PV compute, O accumulation
//   Soft warps (4-7): Softmax, cp.async K/V memory transfers
//
// Synchronization via named barriers (bar.sync with thread count):
//   bar 1: K data ready in KV_s      (soft → mma)
//   bar 2: S scores ready in S_s     (mma → soft)
//   bar 3: P + V ready               (soft → mma)
//
// MMA warp layout: 4 warps cover 64×128 output
//   warp 0: rows  0-15, all 128 cols (8× m16n8k16)
//   warp 1: rows 16-31
//   warp 2: rows 32-47
//   warp 3: rows 48-63
//
// Soft warp layout: 4 warps each handle 16 softmax rows
//   warp 4: rows  0-15  (8 rows × 2 passes for warp_n=0,1 equivalent)
//   warp 5: rows 16-31
//   warp 6: rows 32-47
//   warp 7: rows 48-63
//
// Same SMEM (44KB), same ldmatrix, same cp.async.
// Target: eliminate 3× __syncthreads → fine-grained named barriers.
// =============================================================================

#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define FA8_BR 64
#define FA8_BC 64
#define FA8_THREADS 256

#define FA8_Q_STRIDE 136
#define FA8_KV_STRIDE 136
#define FA8_S_STRIDE 72

// Number of threads per role
#define FA8_MMA_THREADS 128  // warps 0-3
#define FA8_SOFT_THREADS 128 // warps 4-7

// =============================================================================
// Named barrier helpers (SM80+)
// =============================================================================
// bar.sync N, thread_count — waits until thread_count threads arrive at bar N
// We use bars 1-4 (bar 0 is __syncthreads)

__device__ __forceinline__ void bar_sync(int bar_id, int thread_count)
{
    asm volatile("bar.sync %0, %1;" ::"r"(bar_id), "r"(thread_count));
}

// Full block sync (all 256 threads) — same as __syncthreads but explicit
__device__ __forceinline__ void bar_sync_all()
{
    asm volatile("bar.sync 0, 256;");
}

// =============================================================================
// cp.async
// =============================================================================

__device__ __forceinline__ void cp_async_cg_16(
    void *smem_ptr, const void *gmem_ptr, int src_bytes)
{
    uint32_t sa = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sa), "l"(gmem_ptr), "r"(src_bytes));
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
// ldmatrix
// =============================================================================

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3,
    const void *smem_ptr)
{
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(
    uint32_t &r0, uint32_t &r1, const void *smem_ptr)
{
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];"
        : "=r"(r0), "=r"(r1) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t &r0, uint32_t &r1, const void *smem_ptr)
{
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16.trans {%0,%1}, [%2];"
        : "=r"(r0), "=r"(r1) : "r"(addr));
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
// MMA
// =============================================================================

__device__ __forceinline__ void mma_f16_f32(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// =============================================================================
// Fragment loaders (same as v6)
// =============================================================================

__device__ __forceinline__ void load_a_ldm(
    uint32_t &a0, uint32_t &a1, uint32_t &a2, uint32_t &a3,
    const __half *smem, int stride, int row_base, int k_base, int lane)
{
    int sub = lane / 8;
    int sub_row = lane % 8;
    int row = row_base + (sub & 1) * 8 + sub_row;
    int col = k_base + (sub >> 1) * 8;
    ldmatrix_x4(a0, a1, a2, a3, &smem[row * stride + col]);
}

__device__ __forceinline__ void load_b_kt_ldm(
    uint32_t &b0, uint32_t &b1,
    const __half *smem, int stride, int n_base, int k_base, int lane)
{
    int sub = (lane / 8) % 2;
    int sub_row = lane % 8;
    ldmatrix_x2(b0, b1, &smem[(n_base + sub_row) * stride + k_base + sub * 8]);
}

__device__ __forceinline__ void load_b_v_ldm(
    uint32_t &b0, uint32_t &b1,
    const __half *smem, int stride, int n_base, int k_base, int lane)
{
    int sub = (lane / 8) % 2;
    int sub_row = lane % 8;
    ldmatrix_x2_trans(b0, b1, &smem[(k_base + sub * 8 + sub_row) * stride + n_base]);
}

__device__ __forceinline__ void store_d_to_smem(
    __half *smem, int stride, int row_base, int col_base,
    float d0, float d1, float d2, float d3, int lane)
{
    int gid = lane >> 2, tid = lane & 3;
    int r0 = row_base + gid, r8 = r0 + 8;
    int c0 = col_base + tid * 2, c1 = c0 + 1;
    smem[r0 * stride + c0] = __float2half(d0);
    smem[r0 * stride + c1] = __float2half(d1);
    smem[r8 * stride + c0] = __float2half(d2);
    smem[r8 * stride + c1] = __float2half(d3);
}

// =============================================================================
// Async tile load — only soft warps participate
// Uses local thread index within soft warps (0-127)
// =============================================================================

__device__ __forceinline__ void async_load_tile_soft(
    __half *smem, const __half *src_head,
    int tile_start, int seq_len, int head_dim, int local_tid)
{
    constexpr int CHUNKS_PER_ROW = 16;
    constexpr int TOTAL_CHUNKS = FA8_BC * CHUNKS_PER_ROW; // 1024

    // 128 soft threads, 1024 chunks → 8 per thread
#pragma unroll 8
    for (int c = local_tid; c < TOTAL_CHUNKS; c += FA8_SOFT_THREADS)
    {
        int row = c / CHUNKS_PER_ROW;
        int col8 = (c % CHUNKS_PER_ROW) * 8;
        int grow = tile_start + row;

        cp_async_cg_16(&smem[row * FA8_KV_STRIDE + col8],
                       &src_head[grow * head_dim + col8],
                       (grow < seq_len) ? 16 : 0);
    }
}

// =============================================================================
// FlashAttention v8 Kernel — Warp Specialized
// =============================================================================

__global__ void __launch_bounds__(FA8_THREADS, 2)
    flash_attention_v8_kernel(
        const __half *__restrict__ Q,
        const __half *__restrict__ K,
        const __half *__restrict__ V,
        __half *__restrict__ O,
        int seq_len, int head_dim, int causal, float scale)
{
    int num_q_tiles = (seq_len + FA8_BR - 1) / FA8_BR;
    int bh = blockIdx.x / num_q_tiles;
    int q_tile = blockIdx.x % num_q_tiles;
    int q_start = q_tile * FA8_BR;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int gid = lane_id >> 2;
    int tid = lane_id & 3;

    bool is_mma_warp = (warp_id < 4);
    bool is_soft_warp = (warp_id >= 4);

    // Local warp index within role
    int mma_warp = warp_id;                        // 0-3 for MMA warps
    int soft_warp = warp_id - 4;                   // 0-3 for soft warps
    int soft_local_tid = soft_warp * 32 + lane_id; // 0-127 within soft group

    extern __shared__ char smem_raw[];
    __half *Q_s = (__half *)smem_raw;
    __half *KV_s = (__half *)(smem_raw + FA8_BR * FA8_Q_STRIDE * sizeof(__half));
    __half *S_s = (__half *)(smem_raw + FA8_BR * FA8_Q_STRIDE * sizeof(__half) + FA8_BC * FA8_KV_STRIDE * sizeof(__half));
    float *m_smem = (float *)(smem_raw + FA8_BR * FA8_Q_STRIDE * sizeof(__half) + FA8_BC * FA8_KV_STRIDE * sizeof(__half) + FA8_BR * FA8_S_STRIDE * sizeof(__half));
    float *l_smem = m_smem + FA8_BR;

    int hstride = seq_len * head_dim;
    const __half *Q_head = Q + bh * hstride;
    const __half *K_head = K + bh * hstride;
    const __half *V_head = V + bh * hstride;
    __half *O_head = O + bh * hstride;

    // ==== Init phase (all threads) ====

    // Load Q via cp.async (all 256 threads cooperate)
    {
        constexpr int CHUNKS_PER_ROW = 16;
        constexpr int TOTAL = FA8_BR * CHUNKS_PER_ROW;
#pragma unroll 4
        for (int c = threadIdx.x; c < TOTAL; c += FA8_THREADS)
        {
            int row = c / CHUNKS_PER_ROW;
            int col8 = (c % CHUNKS_PER_ROW) * 8;
            int grow = q_start + row;
            cp_async_cg_16(&Q_s[row * FA8_Q_STRIDE + col8],
                           &Q_head[grow * head_dim + col8],
                           (grow < seq_len) ? 16 : 0);
        }
        cp_async_commit();
        cp_async_wait<0>();
    }

    // Init m/l
    for (int i = threadIdx.x; i < FA8_BR; i += FA8_THREADS)
    {
        m_smem[i] = -1e30f;
        l_smem[i] = 0.0f;
    }

    bar_sync_all(); // ensure Q + m/l init visible

    // ==== Role-specific state ====

    // MMA warps: O accumulators + row indices
    float o_acc[16][4]; // 16 n-tiles × 4 values (covering all 128 cols)
    int mma_row_base;

    if (is_mma_warp)
    {
        mma_row_base = mma_warp * 16; // each MMA warp owns 16 rows
#pragma unroll
        for (int t = 0; t < 16; t++)
#pragma unroll
            for (int r = 0; r < 4; r++)
                o_acc[t][r] = 0.0f;
    }

    // Soft warps: register m/l, row indices
    int soft_row_base;
    float m_reg[16], l_reg[16]; // up to 16 rows per soft warp

    if (is_soft_warp)
    {
        soft_row_base = soft_warp * 16;
#pragma unroll
        for (int r = 0; r < 16; r++)
        {
            m_reg[r] = -1e30f;
            l_reg[r] = 0.0f;
        }
    }

    int num_kv_tiles = (seq_len + FA8_BC - 1) / FA8_BC;
    int k_steps = head_dim / 16;

    // ==== Prologue: soft warps load K[0] ====
    if (is_soft_warp)
    {
        async_load_tile_soft(KV_s, K_head, 0, seq_len, head_dim, soft_local_tid);
        cp_async_commit();
        cp_async_wait<0>();
    }

    // Bar 1: K[0] ready — signal to MMA warps
    bar_sync_all(); // initial sync, then switch to named barriers

    // ==== Main loop ====
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++)
    {
        int kv_start = kv_tile * FA8_BC;
        if (causal && kv_start > q_start + FA8_BR - 1)
            break;

        // ============================================================
        // PHASE 1: MMA warps compute S = Q @ K^T
        //          Soft warps idle (could prefetch V here)
        // ============================================================
        if (is_mma_warp)
        {
            float s_acc[8][4]; // 8 n-tiles of 8 cols = 64 cols
#pragma unroll
            for (int t = 0; t < 8; t++)
#pragma unroll
                for (int r = 0; r < 4; r++)
                    s_acc[t][r] = 0.0f;

            for (int ks = 0; ks < k_steps; ks++)
            {
                uint32_t a0, a1, a2, a3;
                load_a_ldm(a0, a1, a2, a3, Q_s, FA8_Q_STRIDE,
                           mma_row_base, ks * 16, lane_id);

#pragma unroll
                for (int nt = 0; nt < 8; nt++)
                {
                    uint32_t b0, b1;
                    load_b_kt_ldm(b0, b1, KV_s, FA8_KV_STRIDE,
                                  nt * 8, ks * 16, lane_id);
                    mma_f16_f32(
                        s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3],
                        a0, a1, a2, a3, b0, b1,
                        s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3]);
                }
            }

            // Scale + causal mask → S_s
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
            {
                int col_base = nt * 8;
                s_acc[nt][0] *= scale;
                s_acc[nt][1] *= scale;
                s_acc[nt][2] *= scale;
                s_acc[nt][3] *= scale;

                if (causal)
                {
                    int r0 = mma_row_base + gid, r8 = r0 + 8;
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

                store_d_to_smem(S_s, FA8_S_STRIDE, mma_row_base, col_base,
                                s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3],
                                lane_id);
            }
        }

        // Bar 2: S_s ready — MMA done writing, soft can read
        bar_sync_all();

        // ============================================================
        // PHASE 2: Soft warps do softmax + load V
        //          MMA warps read m_old for rescale, then idle
        // ============================================================

        // MMA warps: read m_old BEFORE soft warps overwrite
        float m_old_r0, m_old_r8;
        if (is_mma_warp)
        {
            int my_row0 = mma_row_base + gid;
            int my_row8 = my_row0 + 8;
            m_old_r0 = m_smem[my_row0];
            m_old_r8 = m_smem[my_row8];
        }

        if (is_soft_warp)
        {
            // Start V load (overlapped with softmax)
            async_load_tile_soft(KV_s, V_head, kv_start, seq_len, head_dim, soft_local_tid);
            cp_async_commit();

            // Softmax: each soft warp handles 16 rows
            // 32 lanes cover 64 cols (lane_id * 2, lane_id * 2 + 1)
#pragma unroll
            for (int rr = 0; rr < 16; rr += 2)
            {
                int srow_a = soft_row_base + rr;
                int srow_b = srow_a + 1;

                float va0 = __half2float(S_s[srow_a * FA8_S_STRIDE + lane_id * 2]);
                float vb0 = __half2float(S_s[srow_b * FA8_S_STRIDE + lane_id * 2]);
                float va1 = __half2float(S_s[srow_a * FA8_S_STRIDE + lane_id * 2 + 1]);
                float vb1 = __half2float(S_s[srow_b * FA8_S_STRIDE + lane_id * 2 + 1]);

                float rmax_a = warp_reduce_max(fmaxf(va0, va1));
                float rmax_b = warp_reduce_max(fmaxf(vb0, vb1));

                float m_new_a = fmaxf(m_reg[rr], rmax_a);
                float m_new_b = fmaxf(m_reg[rr + 1], rmax_b);

                float ea0 = __expf(va0 - m_new_a);
                float eb0 = __expf(vb0 - m_new_b);
                float ea1 = __expf(va1 - m_new_a);
                float eb1 = __expf(vb1 - m_new_b);

                float rsum_a = warp_reduce_sum(ea0 + ea1);
                float rsum_b = warp_reduce_sum(eb0 + eb1);

                float rsc_a = __expf(m_reg[rr] - m_new_a);
                float rsc_b = __expf(m_reg[rr + 1] - m_new_b);

                m_reg[rr] = m_new_a;
                m_reg[rr + 1] = m_new_b;
                if (lane_id == 0)
                {
                    l_reg[rr] = l_reg[rr] * rsc_a + rsum_a;
                    l_reg[rr + 1] = l_reg[rr + 1] * rsc_b + rsum_b;
                }

                S_s[srow_a * FA8_S_STRIDE + lane_id * 2] = __float2half(ea0);
                S_s[srow_b * FA8_S_STRIDE + lane_id * 2] = __float2half(eb0);
                S_s[srow_a * FA8_S_STRIDE + lane_id * 2 + 1] = __float2half(ea1);
                S_s[srow_b * FA8_S_STRIDE + lane_id * 2 + 1] = __float2half(eb1);
            }

            // Flush m/l to SMEM
#pragma unroll
            for (int r = 0; r < 16; r++)
                m_smem[soft_row_base + r] = m_reg[r];
            if (lane_id == 0)
            {
#pragma unroll
                for (int r = 0; r < 16; r++)
                    l_smem[soft_row_base + r] = l_reg[r];
            }

            // Wait for V
            cp_async_wait<0>();
        }

        // Bar 3: P in S_s + V in KV_s + m/l updated
        bar_sync_all();

        // ============================================================
        // PHASE 3: MMA warps do rescale + O += P @ V
        //          Soft warps load K[next]
        // ============================================================

        // Soft warps: prefetch K[next] (overlapped with PV compute!)
        int next_tile = kv_tile + 1;
        int next_start = next_tile * FA8_BC;
        bool has_next = (next_tile < num_kv_tiles) &&
                        (!causal || next_start <= q_start + FA8_BR - 1);

        if (is_soft_warp && has_next)
        {
            async_load_tile_soft(KV_s, K_head, next_start, seq_len, head_dim, soft_local_tid);
            cp_async_commit();
        }

        if (is_mma_warp)
        {
            // Rescale O
            int my_row0 = mma_row_base + gid;
            int my_row8 = my_row0 + 8;
            float m_new_r0 = m_smem[my_row0];
            float m_new_r8 = m_smem[my_row8];
            float rescale0 = __expf(m_old_r0 - m_new_r0);
            float rescale8 = __expf(m_old_r8 - m_new_r8);

#pragma unroll
            for (int t = 0; t < 16; t++)
            {
                o_acc[t][0] *= rescale0;
                o_acc[t][1] *= rescale0;
                o_acc[t][2] *= rescale8;
                o_acc[t][3] *= rescale8;
            }

            // O += P @ V
            // Each MMA warp covers 16 rows × 128 cols = 16 n-tiles of 8 cols
#pragma unroll
            for (int nt = 0; nt < 16; nt++)
            {
                int n_col = nt * 8;

#pragma unroll
                for (int ks = 0; ks < 4; ks++)
                {
                    uint32_t a0, a1, a2, a3;
                    load_a_ldm(a0, a1, a2, a3, S_s, FA8_S_STRIDE,
                               mma_row_base, ks * 16, lane_id);

                    uint32_t b0, b1;
                    load_b_v_ldm(b0, b1, KV_s, FA8_KV_STRIDE,
                                 n_col, ks * 16, lane_id);

                    mma_f16_f32(
                        o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3],
                        a0, a1, a2, a3, b0, b1,
                        o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3]);
                }
            }
        }

        // Soft warps: wait for K[next]
        if (is_soft_warp && has_next)
        {
            cp_async_wait<0>();
        }

        // Bar 1 (reuse): K[next] ready + PV done, safe to start next iter
        bar_sync_all();
    }

    // ==== Final: O / l → global (MMA warps only) ====
    if (is_mma_warp)
    {
        int my_row0 = mma_row_base + gid;
        int my_row8 = my_row0 + 8;
        float l_inv0 = (l_smem[my_row0] > 0.0f) ? 1.0f / l_smem[my_row0] : 0.0f;
        float l_inv8 = (l_smem[my_row8] > 0.0f) ? 1.0f / l_smem[my_row8] : 0.0f;
        int grow0 = q_start + my_row0;
        int grow8 = q_start + my_row8;

#pragma unroll
        for (int nt = 0; nt < 16; nt++)
        {
            int col0 = nt * 8 + tid * 2;
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

static int g_fa8_smem_max = 0;

extern "C"
{

    int flash_attention_v8_forward(
        const void *Q, const void *K, const void *V, void *O,
        int total_heads, int seq_len, int head_dim,
        int causal, void *stream)
    {
        if (head_dim % 16 != 0 || head_dim > 128)
            return -1;

        int smem_bytes = FA8_BR * FA8_Q_STRIDE * (int)sizeof(__half) +
                         FA8_BC * FA8_KV_STRIDE * (int)sizeof(__half) +
                         FA8_BR * FA8_S_STRIDE * (int)sizeof(__half) +
                         FA8_BR * 2 * (int)sizeof(float);

        if (smem_bytes > g_fa8_smem_max)
        {
            cudaError_t err = cudaFuncSetAttribute(
                flash_attention_v8_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
            if (err != cudaSuccess)
                return (int)err;
            g_fa8_smem_max = smem_bytes;
        }

        float scale_val = 1.0f / sqrtf((float)head_dim);
        int num_q_tiles = (seq_len + FA8_BR - 1) / FA8_BR;
        int total_blocks = total_heads * num_q_tiles;

        flash_attention_v8_kernel<<<total_blocks, FA8_THREADS, smem_bytes, (cudaStream_t)stream>>>(
            (const __half *)Q, (const __half *)K, (const __half *)V, (__half *)O,
            seq_len, head_dim, causal, scale_val);

        return (int)cudaGetLastError();
    }

    int flash_attention_v8_forward_bhsd(
        const void *Q, const void *K, const void *V, void *O,
        int batch, int num_heads, int seq_len, int head_dim,
        int causal, void *stream)
    {
        return flash_attention_v8_forward(Q, K, V, O,
                                          batch * num_heads, seq_len, head_dim, causal, stream);
    }

} // extern "C"
