// =============================================================================
// FlashAttention v6 — ldmatrix + Optimized Softmax (SM89)
// =============================================================================
// Changes vs v5 (191 TFLOPS):
//   1. __expf() — hardware fast-math exp (~1 vs ~10 cycles)
//   2. Register m/l — no SMEM reads inside softmax inner loop
//   3. 2-row unrolled softmax — ILP on shuffles, expf, SMEM access
//   4. m_old read from SMEM before softmax (cross-warp safe)
//
// Same pipeline, SMEM layout, ldmatrix as v5.
// =============================================================================

#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define FA6_BR 64
#define FA6_BC 64
#define FA6_THREADS 256

#define FA6_Q_STRIDE 136
#define FA6_KV_STRIDE 136
#define FA6_S_STRIDE 72

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
// Fragment loaders (ldmatrix, same as v5)
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
// Async tile load
// =============================================================================

__device__ __forceinline__ void async_load_tile(
    __half *smem, const __half *src_head,
    int tile_start, int seq_len, int head_dim)
{
    constexpr int CHUNKS_PER_ROW = 16;
    constexpr int TOTAL_CHUNKS = FA6_BC * CHUNKS_PER_ROW;

#pragma unroll 4
    for (int c = threadIdx.x; c < TOTAL_CHUNKS; c += FA6_THREADS)
    {
        int row = c / CHUNKS_PER_ROW;
        int col8 = (c % CHUNKS_PER_ROW) * 8;
        int grow = tile_start + row;

        __half *dst = &smem[row * FA6_KV_STRIDE + col8];
        const __half *src = &src_head[grow * head_dim + col8];
        cp_async_cg_16(dst, src, (grow < seq_len) ? 16 : 0);
    }
}

// =============================================================================
// FlashAttention v6 Kernel
// =============================================================================

__global__ void __launch_bounds__(FA6_THREADS, 2)
    flash_attention_v6_kernel(
        const __half *__restrict__ Q,
        const __half *__restrict__ K,
        const __half *__restrict__ V,
        __half *__restrict__ O,
        int seq_len, int head_dim, int causal, float scale)
{
    int num_q_tiles = (seq_len + FA6_BR - 1) / FA6_BR;
    int bh = blockIdx.x / num_q_tiles;
    int q_tile = blockIdx.x % num_q_tiles;
    int q_start = q_tile * FA6_BR;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int warp_m = warp_id % 4;
    int warp_n = warp_id / 4;
    int gid = lane_id >> 2;
    int tid = lane_id & 3;

    extern __shared__ char smem_raw[];
    __half *Q_s = (__half *)smem_raw;
    __half *KV_s = (__half *)(smem_raw + FA6_BR * FA6_Q_STRIDE * sizeof(__half));
    __half *S_s = (__half *)(smem_raw + FA6_BR * FA6_Q_STRIDE * sizeof(__half) + FA6_BC * FA6_KV_STRIDE * sizeof(__half));
    float *m_smem = (float *)(smem_raw + FA6_BR * FA6_Q_STRIDE * sizeof(__half) + FA6_BC * FA6_KV_STRIDE * sizeof(__half) + FA6_BR * FA6_S_STRIDE * sizeof(__half));
    float *l_smem = m_smem + FA6_BR;

    int hstride = seq_len * head_dim;
    const __half *Q_head = Q + bh * hstride;
    const __half *K_head = K + bh * hstride;
    const __half *V_head = V + bh * hstride;
    __half *O_head = O + bh * hstride;

    // Load Q via cp.async
    {
        constexpr int CHUNKS_PER_ROW = 16;
        constexpr int TOTAL = FA6_BR * CHUNKS_PER_ROW;
#pragma unroll 4
        for (int c = threadIdx.x; c < TOTAL; c += FA6_THREADS)
        {
            int row = c / CHUNKS_PER_ROW;
            int col8 = (c % CHUNKS_PER_ROW) * 8;
            int grow = q_start + row;
            cp_async_cg_16(&Q_s[row * FA6_Q_STRIDE + col8],
                           &Q_head[grow * head_dim + col8],
                           (grow < seq_len) ? 16 : 0);
        }
        cp_async_commit();
    }

    // Init m/l SMEM
    for (int i = threadIdx.x; i < FA6_BR; i += FA6_THREADS)
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

    // MMA output rows for this thread
    int my_row0 = warp_m * 16 + gid; // rows 0-7 within warp_m's 16-row block
    int my_row8 = my_row0 + 8;       // rows 8-15

    // Softmax rows for this warp
    int srow_base = warp_m * 16 + warp_n * 8;

    // Register m/l for softmax (each warp owns 8 rows)
    float m_reg[8], l_reg[8];
#pragma unroll
    for (int r = 0; r < 8; r++)
    {
        m_reg[r] = -1e30f;
        l_reg[r] = 0.0f;
    }

    int num_kv_tiles = (seq_len + FA6_BC - 1) / FA6_BC;
    int m_row_base = warp_m * 16;
    int n_col_base = warp_n * 32;
    int k_steps = head_dim / 16;

    // Prologue: K[0]
    async_load_tile(KV_s, K_head, 0, seq_len, head_dim);
    cp_async_commit();

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++)
    {
        int kv_start = kv_tile * FA6_BC;
        if (causal && kv_start > q_start + FA6_BR - 1)
            break;

        cp_async_wait<0>();
        __syncthreads();

        // ---- S = Q @ K^T (ldmatrix) ----
        float s_acc[4][4];
#pragma unroll
        for (int t = 0; t < 4; t++)
#pragma unroll
            for (int r = 0; r < 4; r++)
                s_acc[t][r] = 0.0f;

        for (int ks = 0; ks < k_steps; ks++)
        {
            uint32_t a0, a1, a2, a3;
            load_a_ldm(a0, a1, a2, a3, Q_s, FA6_Q_STRIDE,
                       m_row_base, ks * 16, lane_id);

#pragma unroll
            for (int nt = 0; nt < 4; nt++)
            {
                uint32_t b0, b1;
                load_b_kt_ldm(b0, b1, KV_s, FA6_KV_STRIDE,
                              n_col_base + nt * 8, ks * 16, lane_id);
                mma_f16_f32(
                    s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3],
                    a0, a1, a2, a3, b0, b1,
                    s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3]);
            }
        }

// ---- Scale + causal mask → S_s ----
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

            store_d_to_smem(S_s, FA6_S_STRIDE, m_row_base, col_base,
                            s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3],
                            lane_id);
        }

        __syncthreads();

        // ---- V load overlapped with softmax ----
        async_load_tile(KV_s, V_head, kv_start, seq_len, head_dim);
        cp_async_commit();

        // Read m_old for O rescale from SMEM BEFORE softmax overwrites it
        // m_smem still has PREVIOUS iteration's values here
        // (guaranteed by __syncthreads at end of previous iteration)
        float m_old_r0 = m_smem[my_row0];
        float m_old_r8 = m_smem[my_row8];

// ---- Optimized softmax: 2-row unrolled, register m/l, __expf ----
#pragma unroll
        for (int rr = 0; rr < 8; rr += 2)
        {
            int srow_a = srow_base + rr;
            int srow_b = srow_a + 1;

            // SMEM reads — interleaved for ILP
            float va0 = __half2float(S_s[srow_a * FA6_S_STRIDE + lane_id * 2]);
            float vb0 = __half2float(S_s[srow_b * FA6_S_STRIDE + lane_id * 2]);
            float va1 = __half2float(S_s[srow_a * FA6_S_STRIDE + lane_id * 2 + 1]);
            float vb1 = __half2float(S_s[srow_b * FA6_S_STRIDE + lane_id * 2 + 1]);

            // Local max
            float lmax_a = fmaxf(va0, va1);
            float lmax_b = fmaxf(vb0, vb1);

            // Warp reduce max — two independent chains (ILP on shuffles)
            float rmax_a = warp_reduce_max(lmax_a);
            float rmax_b = warp_reduce_max(lmax_b);

            // New max from registers (no SMEM read!)
            float m_old_a = m_reg[rr];
            float m_old_b = m_reg[rr + 1];
            float m_new_a = fmaxf(m_old_a, rmax_a);
            float m_new_b = fmaxf(m_old_b, rmax_b);

            // Fast exp — __expf hardware approximation
            float ea0 = __expf(va0 - m_new_a);
            float eb0 = __expf(vb0 - m_new_b);
            float ea1 = __expf(va1 - m_new_a);
            float eb1 = __expf(vb1 - m_new_b);

            // Sum reduce — two independent chains
            float rsum_a = warp_reduce_sum(ea0 + ea1);
            float rsum_b = warp_reduce_sum(eb0 + eb1);

            // Rescale factors
            float rsc_a = __expf(m_old_a - m_new_a);
            float rsc_b = __expf(m_old_b - m_new_b);

            // Update register m/l
            m_reg[rr] = m_new_a;
            m_reg[rr + 1] = m_new_b;
            // l update only needed on lane 0 (only lane 0 uses l_reg)
            if (lane_id == 0)
            {
                l_reg[rr] = l_reg[rr] * rsc_a + rsum_a;
                l_reg[rr + 1] = l_reg[rr + 1] * rsc_b + rsum_b;
            }

            // Write P to S_s — interleaved stores
            S_s[srow_a * FA6_S_STRIDE + lane_id * 2] = __float2half(ea0);
            S_s[srow_b * FA6_S_STRIDE + lane_id * 2] = __float2half(eb0);
            S_s[srow_a * FA6_S_STRIDE + lane_id * 2 + 1] = __float2half(ea1);
            S_s[srow_b * FA6_S_STRIDE + lane_id * 2 + 1] = __float2half(eb1);
        }

// Flush m/l registers → SMEM (needed by partner warp for rescale)
#pragma unroll
        for (int r = 0; r < 8; r++)
        {
            m_smem[srow_base + r] = m_reg[r];
        }
        if (lane_id == 0)
        {
#pragma unroll
            for (int r = 0; r < 8; r++)
            {
                l_smem[srow_base + r] = l_reg[r];
            }
        }

        // Wait for V + ensure m/l SMEM visible to all warps
        cp_async_wait<0>();
        __syncthreads();

        // ---- Rescale O ----
        // m_old: read before softmax (previous iteration values)
        // m_new: read from m_smem (just written by this iteration's softmax)
        float m_new_r0 = m_smem[my_row0];
        float m_new_r8 = m_smem[my_row8];
        float rescale0 = __expf(m_old_r0 - m_new_r0);
        float rescale8 = __expf(m_old_r8 - m_new_r8);

#pragma unroll
        for (int t = 0; t < 8; t++)
        {
            o_acc[t][0] *= rescale0;
            o_acc[t][1] *= rescale0;
            o_acc[t][2] *= rescale8;
            o_acc[t][3] *= rescale8;
        }

        // ---- O += P @ V (ldmatrix + ldmatrix.trans) ----
        int o_n_base = warp_n * 64;

#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            int n_col = o_n_base + nt * 8;

#pragma unroll
            for (int ks = 0; ks < 4; ks++)
            {
                uint32_t a0, a1, a2, a3;
                load_a_ldm(a0, a1, a2, a3, S_s, FA6_S_STRIDE,
                           m_row_base, ks * 16, lane_id);

                uint32_t b0, b1;
                load_b_v_ldm(b0, b1, KV_s, FA6_KV_STRIDE,
                             n_col, ks * 16, lane_id);

                mma_f16_f32(
                    o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3],
                    a0, a1, a2, a3, b0, b1,
                    o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3]);
            }
        }

        // ---- Prefetch K[next] ----
        __syncthreads();

        int next_tile = kv_tile + 1;
        int next_start = next_tile * FA6_BC;
        bool has_next = (next_tile < num_kv_tiles) &&
                        (!causal || next_start <= q_start + FA6_BR - 1);
        if (has_next)
        {
            async_load_tile(KV_s, K_head, next_start, seq_len, head_dim);
            cp_async_commit();
        }
    }

    // ---- Final: O / l → global ----
    {
        float l_inv0 = (l_smem[my_row0] > 0.0f) ? 1.0f / l_smem[my_row0] : 0.0f;
        float l_inv8 = (l_smem[my_row8] > 0.0f) ? 1.0f / l_smem[my_row8] : 0.0f;
        int grow0 = q_start + my_row0;
        int grow8 = q_start + my_row8;
        int o_n_base = warp_n * 64;

#pragma unroll
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

static int g_fa6_smem_max = 0;

extern "C"
{

    int flash_attention_v6_forward(
        const void *Q, const void *K, const void *V, void *O,
        int total_heads, int seq_len, int head_dim,
        int causal, void *stream)
    {
        if (head_dim % 16 != 0 || head_dim > 128)
            return -1;

        int smem_bytes = FA6_BR * FA6_Q_STRIDE * (int)sizeof(__half) + FA6_BC * FA6_KV_STRIDE * (int)sizeof(__half) + FA6_BR * FA6_S_STRIDE * (int)sizeof(__half) + FA6_BR * 2 * (int)sizeof(float);

        if (smem_bytes > g_fa6_smem_max)
        {
            cudaError_t err = cudaFuncSetAttribute(
                flash_attention_v6_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
            if (err != cudaSuccess)
                return (int)err;
            g_fa6_smem_max = smem_bytes;
        }

        float scale_val = 1.0f / sqrtf((float)head_dim);
        int num_q_tiles = (seq_len + FA6_BR - 1) / FA6_BR;
        int total_blocks = total_heads * num_q_tiles;

        flash_attention_v6_kernel<<<total_blocks, FA6_THREADS, smem_bytes, (cudaStream_t)stream>>>(
            (const __half *)Q, (const __half *)K, (const __half *)V, (__half *)O,
            seq_len, head_dim, causal, scale_val);

        return (int)cudaGetLastError();
    }

    int flash_attention_v6_forward_bhsd(
        const void *Q, const void *K, const void *V, void *O,
        int batch, int num_heads, int seq_len, int head_dim,
        int causal, void *stream)
    {
        return flash_attention_v6_forward(Q, K, V, O,
                                          batch * num_heads, seq_len, head_dim, causal, stream);
    }

} // extern "C"
