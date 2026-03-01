// =============================================================================
// FlashAttention v2 — DB + SMEM Swizzle v2 (CORRECTED) (SM89)
// =============================================================================
// Fix: swizzle at 32-half (64-byte) sub-sector granularity.
// This keeps pack_h2 pairs contiguous and cp.async 16B chunks intact.
//
// Swizzle: 4 sub-sectors per row (cols 0-31, 32-63, 64-95, 96-127)
//   sub = (half_col >> 5) & 3
//   sw_sub = sub ^ (row & 3)
//   sw_col = (sw_sub << 5) | (half_col & 31)
//
// cp.async writes 8 halfs (1 chunk) — fits within 32-half sub-sector ✓
// pack_h2 reads 2 contiguous halfs — stays within sub-sector ✓
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
// Swizzle: 32-half sub-sector XOR
// =============================================================================

__device__ __forceinline__ int kv_swizzle(int row, int half_col)
{
    int sub = (half_col >> 5) & 3;
    int local = half_col & 31;
    int sw_sub = sub ^ (row & 3);
    return (sw_sub << 5) | local;
}

__device__ __forceinline__ int kv_swizzle_chunk(int row, int chunk_id)
{
    // chunk_id 0..15 for d=128 (8 halfs each)
    // sub-sector = chunk_id / 4 (each sub has 4 chunks of 8 halfs = 32 halfs)
    int sub = (chunk_id >> 2) & 3;
    int local_chunk = chunk_id & 3;
    int sw_sub = sub ^ (row & 3);
    return (sw_sub << 2) | local_chunk;
}

// =============================================================================
// cp.async
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
// MMA helpers
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
// Fragment loading
// =============================================================================

// Q: no swizzle
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
    a1 = pack_h2(&smem[r8 * stride + k0]);
    a2 = pack_h2(&smem[r0 * stride + k8]);
    a3 = pack_h2(&smem[r8 * stride + k8]);
}

// K^T: SMEM row=n, col=k → swizzle col k based on row n
__device__ __forceinline__ void load_b_frag_kt_sw(
    uint32_t &b0, uint32_t &b1,
    const __half *smem, int stride, int n_base, int k_base, int lane)
{
    int gid = lane >> 2;
    int tid = lane & 3;
    int n = n_base + gid;
    int k0 = k_base + tid * 2;
    int k8 = k0 + 8;

    int sk0 = kv_swizzle(n, k0);
    int sk8 = kv_swizzle(n, k8);

    b0 = pack_h2(&smem[n * stride + sk0]);
    b1 = pack_h2(&smem[n * stride + sk8]);
}

// V: SMEM row=k, col=n → swizzle col n based on row k
__device__ __forceinline__ void load_b_frag_v_sw(
    uint32_t &b0, uint32_t &b1,
    const __half *smem, int stride, int n_base, int k_base, int lane)
{
    int gid = lane >> 2;
    int tid = lane & 3;
    int n = n_base + gid;
    int k0 = k_base + tid * 2;
    int k8 = k0 + 8;

    int sn_k0 = kv_swizzle(k0, n);
    int sn_k0p1 = kv_swizzle(k0 + 1, n);
    int sn_k8 = kv_swizzle(k8, n);
    int sn_k8p1 = kv_swizzle(k8 + 1, n);

    b0 = pack_h2_strided(&smem[k0 * stride + sn_k0],
                         &smem[(k0 + 1) * stride + sn_k0p1]);
    b1 = pack_h2_strided(&smem[k8 * stride + sn_k8],
                         &smem[(k8 + 1) * stride + sn_k8p1]);
}

// S: no swizzle
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
// Swizzled cp.async KV load
// =============================================================================

__device__ __forceinline__ void cp_async_load_kv_sw(
    __half *kv_smem, const __half *src_head,
    int kv_start, int seq_len, int head_dim)
{
    const int chunks_per_row = head_dim / 8;
    const int total_chunks = FA2_BC * chunks_per_row;

    for (int c = 0; c < total_chunks; c += FA2_THREADS)
    {
        int copy_id = c + threadIdx.x;
        if (copy_id < total_chunks)
        {
            int row = copy_id / chunks_per_row;
            int chunk_id = copy_id % chunks_per_row;
            int grow = kv_start + row;

            int sw_chunk = kv_swizzle_chunk(row, chunk_id);
            int sw_col8 = sw_chunk * 8;

            __half *dst = &kv_smem[row * FA2_KV_STRIDE + sw_col8];

            if (grow < seq_len)
            {
                int col8 = chunk_id * 8;
                const __half *src = &src_head[grow * head_dim + col8];
                cp_async_16B(dst, src);
            }
            else
            {
                for (int j = 0; j < 8; j++)
                    dst[j] = __float2half(0.0f);
            }
        }
    }
}

// =============================================================================
// Kernel
// =============================================================================

__global__ void __launch_bounds__(FA2_THREADS, 2)
    flash_attention_v2_dbsw_kernel(
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

    extern __shared__ char smem_raw[];
    __half *Q_s = (__half *)smem_raw;
    __half *KV_s = (__half *)(smem_raw + FA2_BR * FA2_Q_STRIDE * sizeof(__half));
    __half *S_s = (__half *)(smem_raw + FA2_BR * FA2_Q_STRIDE * sizeof(__half) +
                             FA2_BC * FA2_KV_STRIDE * sizeof(__half));
    float *m_smem = (float *)(smem_raw + FA2_BR * FA2_Q_STRIDE * sizeof(__half) +
                              FA2_BC * FA2_KV_STRIDE * sizeof(__half) +
                              FA2_BR * FA2_S_STRIDE * sizeof(__half));
    float *l_smem = m_smem + FA2_BR;

    int hstride = seq_len * head_dim;
    const __half *Q_head = Q + bh * hstride;
    const __half *K_head = K + bh * hstride;
    const __half *V_head = V + bh * hstride;
    __half *O_head = O + bh * hstride;

    // Load Q (no swizzle)
    for (int i = threadIdx.x; i < FA2_BR * head_dim; i += FA2_THREADS)
    {
        int row = i / head_dim, col = i % head_dim;
        int grow = q_start + row;
        Q_s[row * FA2_Q_STRIDE + col] = (grow < seq_len)
                                            ? Q_head[grow * head_dim + col]
                                            : __float2half(0.0f);
    }

    for (int i = threadIdx.x; i < FA2_BR; i += FA2_THREADS)
    {
        m_smem[i] = -1e30f;
        l_smem[i] = 0.0f;
    }

    float o_acc[8][4];
#pragma unroll
    for (int t = 0; t < 8; t++)
#pragma unroll
        for (int r = 0; r < 4; r++)
            o_acc[t][r] = 0.0f;

    int my_row0 = warp_m * 16 + gid;
    int my_row8 = my_row0 + 8;
    int num_kv_tiles = (seq_len + FA2_BC - 1) / FA2_BC;

    // Prologue
    int kv_start = 0;
    if (!(causal && kv_start > q_start + FA2_BR - 1))
    {
        cp_async_load_kv_sw(KV_s, K_head, kv_start, seq_len, head_dim);
        cp_async_commit();
    }
    __syncthreads();

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++)
    {
        kv_start = kv_tile * FA2_BC;
        if (causal && kv_start > q_start + FA2_BR - 1)
            break;

        cp_async_wait<0>();
        __syncthreads();

        // S = Q @ K^T
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
                load_b_frag_kt_sw(b0, b1, KV_s, FA2_KV_STRIDE,
                                  n_col_base + nt * 8, ks * 16, lane_id);
                mma_f16_f32(s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3],
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
            store_d_to_smem(S_s, FA2_S_STRIDE, m_row_base, col_base,
                            s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3], lane_id);
        }
        __syncthreads();

        // Softmax + async V
        cp_async_load_kv_sw(KV_s, V_head, kv_start, seq_len, head_dim);
        cp_async_commit();

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
                float row_sum = warp_reduce_sum(e0 + e1);
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

        cp_async_wait<0>();
        __syncthreads();

        // Rescale O
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

        // O += P @ V
        int o_n_base = warp_n * 64;
        int pv_k_steps = FA2_BC / 16;
        int next_kv_tile = kv_tile + 1;
        int next_kv_start = next_kv_tile * FA2_BC;
        bool has_next = (next_kv_tile < num_kv_tiles);
        if (causal && next_kv_start > q_start + FA2_BR - 1)
            has_next = false;

        for (int nt = 0; nt < 8; nt++)
        {
            int n_col = o_n_base + nt * 8;
            for (int ks = 0; ks < pv_k_steps; ks++)
            {
                uint32_t a0, a1, a2, a3;
                load_a_frag(a0, a1, a2, a3, S_s, FA2_S_STRIDE,
                            m_row_base, ks * 16, lane_id);
                uint32_t b0, b1;
                load_b_frag_v_sw(b0, b1, KV_s, FA2_KV_STRIDE,
                                 n_col, ks * 16, lane_id);
                mma_f16_f32(o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3],
                            a0, a1, a2, a3, b0, b1,
                            o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3]);
            }
        }

        __syncthreads();
        if (has_next)
        {
            cp_async_load_kv_sw(KV_s, K_head, next_kv_start, seq_len, head_dim);
            cp_async_commit();
        }
    }

    // Final
    {
        float l_inv0 = (l_smem[my_row0] > 0.0f) ? 1.0f / l_smem[my_row0] : 0.0f;
        float l_inv8 = (l_smem[my_row8] > 0.0f) ? 1.0f / l_smem[my_row8] : 0.0f;
        int grow0 = q_start + my_row0, grow8 = q_start + my_row8;
        int o_n_base = warp_n * 64;
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
}

// =============================================================================
// C API
// =============================================================================

static int g_fa2_dbsw_smem_max = 0;

extern "C"
{

    int flash_attention_v2_dbsw_forward(
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

        if (smem_bytes > g_fa2_dbsw_smem_max)
        {
            cudaError_t err = cudaFuncSetAttribute(
                flash_attention_v2_dbsw_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
            if (err != cudaSuccess)
                return (int)err;
            g_fa2_dbsw_smem_max = smem_bytes;
        }

        float scale_val = 1.0f / sqrtf((float)head_dim);
        int num_q_tiles = (seq_len + FA2_BR - 1) / FA2_BR;
        int total_blocks = total_heads * num_q_tiles;

        flash_attention_v2_dbsw_kernel<<<total_blocks, FA2_THREADS, smem_bytes, (cudaStream_t)stream>>>(
            (const __half *)Q, (const __half *)K, (const __half *)V, (__half *)O,
            seq_len, head_dim, causal, scale_val);

        return (int)cudaGetLastError();
    }
}
