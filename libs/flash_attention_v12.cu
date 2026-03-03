// =============================================================================
// FlashAttention v8 — Persistent Kernel + Dynamic Tile Scheduling (SM89)
// =============================================================================
// Based on v7 (109 TFLOPS, 66% util). Changes:
//   1. Persistent kernel: launch num_sms*2 blocks, loop grabbing work
//   2. Atomic tile counter: blocks pick next tile dynamically
//   3. Reversed tile order: heavy Q tiles (many KV tiles) first
//      → better load balancing for causal triangular workload
//
// Core compute path identical to v7 (ldmatrix, __expf, reg m/l, swizzle).
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
// Fragment loaders (identical to v7)
// =============================================================================

__device__ __forceinline__ void load_a_ldm(
    uint32_t &a0, uint32_t &a1, uint32_t &a2, uint32_t &a3,
    const __half *smem, int stride, int row_base, int k_base, int lane)
{
    int sub = lane / 8, sub_row = lane % 8;
    int row = row_base + (sub & 1) * 8 + sub_row;
    int col = k_base + (sub >> 1) * 8;
    ldmatrix_x4(a0, a1, a2, a3, &smem[row * stride + col]);
}

__device__ __forceinline__ void load_b_kt_ldm(
    uint32_t &b0, uint32_t &b1,
    const __half *smem, int stride, int n_base, int k_base, int lane)
{
    int sub = (lane / 8) % 2, sub_row = lane % 8;
    ldmatrix_x2(b0, b1, &smem[(n_base + sub_row) * stride + k_base + sub * 8]);
}

__device__ __forceinline__ void load_b_v_ldm(
    uint32_t &b0, uint32_t &b1,
    const __half *smem, int stride, int n_base, int k_base, int lane)
{
    int sub = (lane / 8) % 2, sub_row = lane % 8;
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
    constexpr int TOTAL_CHUNKS = FA8_BC * CHUNKS_PER_ROW;

#pragma unroll 4
    for (int c = threadIdx.x; c < TOTAL_CHUNKS; c += FA8_THREADS)
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
// Persistent Kernel
// =============================================================================

__global__ void __launch_bounds__(FA8_THREADS, 2)
    flash_attention_v8_kernel(
        const __half *__restrict__ Q,
        const __half *__restrict__ K,
        const __half *__restrict__ V,
        __half *__restrict__ O,
        int seq_len, int head_dim, int total_heads,
        int causal, float scale,
        int total_tiles, int num_q_tiles,
        int *__restrict__ tile_counter)
{
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int warp_m = warp_id % 4;
    int warp_n = warp_id / 4;
    int gid = lane_id >> 2;
    int tid = lane_id & 3;

    extern __shared__ char smem_raw[];
    __half *Q_s = (__half *)smem_raw;
    __half *KV_s = (__half *)(smem_raw + FA8_BR * FA8_Q_STRIDE * sizeof(__half));
    __half *S_s = (__half *)(smem_raw + FA8_BR * FA8_Q_STRIDE * sizeof(__half) +
                             FA8_BC * FA8_KV_STRIDE * sizeof(__half));
    float *m_smem = (float *)(smem_raw + FA8_BR * FA8_Q_STRIDE * sizeof(__half) +
                              FA8_BC * FA8_KV_STRIDE * sizeof(__half) +
                              FA8_BR * FA8_S_STRIDE * sizeof(__half));
    float *l_smem = m_smem + FA8_BR;

    int hstride = seq_len * head_dim;
    int k_steps = head_dim / 16;
    int m_row_base = warp_m * 16;
    int n_col_base = warp_n * 32;
    int my_row0 = warp_m * 16 + gid;
    int my_row8 = my_row0 + 8;
    int srow_base = warp_m * 16 + warp_n * 8;

    // Persistent loop
    while (true)
    {
        // Grab next tile
        __shared__ int s_tile_idx;
        if (threadIdx.x == 0)
            s_tile_idx = atomicAdd(tile_counter, 1);
        __syncthreads();

        int tile_idx = s_tile_idx;
        if (tile_idx >= total_tiles)
            return;

        // Reversed: heavy tiles first
        int rev_idx = total_tiles - 1 - tile_idx;
        int bh = rev_idx / num_q_tiles;
        int q_tile = rev_idx % num_q_tiles;
        int q_start = q_tile * FA8_BR;

        const __half *Q_head = Q + bh * hstride;
        const __half *K_head = K + bh * hstride;
        const __half *V_head = V + bh * hstride;
        __half *O_head = O + bh * hstride;

        // ---- Load Q ----
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
        }

        // Init m/l
        for (int i = threadIdx.x; i < FA8_BR; i += FA8_THREADS)
        {
            m_smem[i] = -1e30f;
            l_smem[i] = 0.0f;
        }

        // O accumulators (reset per tile!)
        float o_acc[8][4];
#pragma unroll
        for (int t = 0; t < 8; t++)
#pragma unroll
            for (int r = 0; r < 4; r++)
                o_acc[t][r] = 0.0f;

        // Register m/l
        float m_reg[8], l_reg[8];
#pragma unroll
        for (int r = 0; r < 8; r++)
        {
            m_reg[r] = -1e30f;
            l_reg[r] = 0.0f;
        }

        int num_kv_tiles = (seq_len + FA8_BC - 1) / FA8_BC;

        // Prologue: K[0]
        async_load_tile(KV_s, K_head, 0, seq_len, head_dim);
        cp_async_commit();

        // Wait for Q + K[0]
        cp_async_wait<0>();
        __syncthreads();

        // ---- KV tile loop ----
        for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++)
        {
            int kv_start = kv_tile * FA8_BC;
            if (causal && kv_start > q_start + FA8_BR - 1)
                break;

            if (kv_tile > 0)
            {
                cp_async_wait<0>();
                __syncthreads();
            }

            // ---- S = Q @ K^T ----
            float s_acc[4][4];
#pragma unroll
            for (int t = 0; t < 4; t++)
#pragma unroll
                for (int r = 0; r < 4; r++)
                    s_acc[t][r] = 0.0f;

            for (int ks = 0; ks < k_steps; ks++)
            {
                uint32_t a0, a1, a2, a3;
                load_a_ldm(a0, a1, a2, a3, Q_s, FA8_Q_STRIDE,
                           m_row_base, ks * 16, lane_id);
#pragma unroll
                for (int nt = 0; nt < 4; nt++)
                {
                    uint32_t b0, b1;
                    load_b_kt_ldm(b0, b1, KV_s, FA8_KV_STRIDE,
                                  n_col_base + nt * 8, ks * 16, lane_id);
                    mma_f16_f32(
                        s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3],
                        a0, a1, a2, a3, b0, b1,
                        s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3]);
                }
            }

            // Scale + mask → S_s
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

                store_d_to_smem(S_s, FA8_S_STRIDE, m_row_base, col_base,
                                s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3],
                                lane_id);
            }

            __syncthreads();

            // V load + softmax
            async_load_tile(KV_s, V_head, kv_start, seq_len, head_dim);
            cp_async_commit();

            float m_old_r0 = m_smem[my_row0];
            float m_old_r8 = m_smem[my_row8];

#pragma unroll
            for (int rr = 0; rr < 8; rr += 2)
            {
                int srow_a = srow_base + rr, srow_b = srow_a + 1;

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

#pragma unroll
            for (int r = 0; r < 8; r++)
                m_smem[srow_base + r] = m_reg[r];
            if (lane_id == 0)
            {
#pragma unroll
                for (int r = 0; r < 8; r++)
                    l_smem[srow_base + r] = l_reg[r];
            }

            cp_async_wait<0>();
            __syncthreads();

            // Rescale O
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

            // O += P @ V
            int o_n_base = warp_n * 64;
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
            {
                int n_col = o_n_base + nt * 8;
#pragma unroll
                for (int ks = 0; ks < 4; ks++)
                {
                    uint32_t a0, a1, a2, a3;
                    load_a_ldm(a0, a1, a2, a3, S_s, FA8_S_STRIDE,
                               m_row_base, ks * 16, lane_id);
                    uint32_t b0, b1;
                    load_b_v_ldm(b0, b1, KV_s, FA8_KV_STRIDE,
                                 n_col, ks * 16, lane_id);
                    mma_f16_f32(
                        o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3],
                        a0, a1, a2, a3, b0, b1,
                        o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3]);
                }
            }

            __syncthreads();

            // Prefetch K[next]
            int next_tile = kv_tile + 1;
            int next_start = next_tile * FA8_BC;
            bool has_next = (next_tile < num_kv_tiles) &&
                            (!causal || next_start <= q_start + FA8_BR - 1);
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
            int grow0 = q_start + my_row0, grow8 = q_start + my_row8;
            int o_n_base = warp_n * 64;

#pragma unroll
            for (int nt = 0; nt < 8; nt++)
            {
                int col0 = o_n_base + nt * 8 + tid * 2, col1 = col0 + 1;
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

        __syncthreads(); // SMEM clean for next tile
    }
}

// =============================================================================
// C API
// =============================================================================

static int g_fa8_smem_max = 0;
static int *d_tile_counter = nullptr;

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

        if (!d_tile_counter)
        {
            cudaError_t err = cudaMalloc(&d_tile_counter, sizeof(int));
            if (err != cudaSuccess)
                return (int)err;
        }
        cudaMemsetAsync(d_tile_counter, 0, sizeof(int), (cudaStream_t)stream);

        float scale_val = 1.0f / sqrtf((float)head_dim);
        int num_q_tiles = (seq_len + FA8_BR - 1) / FA8_BR;
        int total_tiles = total_heads * num_q_tiles;

        // Persistent: num_sms * occupancy
        int num_sms;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        int persistent_blocks = num_sms * 2;
        if (persistent_blocks > total_tiles)
            persistent_blocks = total_tiles;

        flash_attention_v8_kernel<<<persistent_blocks, FA8_THREADS, smem_bytes, (cudaStream_t)stream>>>(
            (const __half *)Q, (const __half *)K, (const __half *)V, (__half *)O,
            seq_len, head_dim, total_heads, causal, scale_val,
            total_tiles, num_q_tiles, d_tile_counter);

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
