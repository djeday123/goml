// =============================================================================
// FlashAttention v4 — Based on v2-DB (159 TFLOPS) + Quick Optimizations
// =============================================================================
// Changes vs v2-DB:
//   1. Branchless cp.async with src_bytes=0 zero-fill (no scalar fallback)
//   2. Merged PV loop (was split nt 0-3 / 4-7 for no reason)
//   3. Q load also via cp.async (bypass registers)
//   4. Tighter loop: cp.async K[next] issued right after last V read
//
// Same SMEM layout as v2-DB (single KV_s → 44KB → 2 blocks/SM):
//   Q_s:  64 × 136 × 2 = 17,408 B
//   KV_s: 64 × 136 × 2 = 17,408 B
//   S_s:  64 × 72  × 2 =  9,216 B
//   m+l:  64 × 2   × 4 =    512 B
//   Total: ~44,544 B → launch_bounds(256, 2)
//
// Build: nvcc -O3 -arch=sm_89 -std=c++17 libs/flash_attention_v4.cu ...
// =============================================================================

#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define FA4_BR 64
#define FA4_BC 64
#define FA4_THREADS 256

#define FA4_Q_STRIDE 136
#define FA4_KV_STRIDE 136
#define FA4_S_STRIDE 72

// =============================================================================
// cp.async (SM80+) — branchless with src_bytes for zero-fill
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
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// =============================================================================
// Fragment loading — VERIFIED (a1↔a2 swapped for SM89)
// =============================================================================

__device__ __forceinline__ void load_a_frag(
    uint32_t &a0, uint32_t &a1, uint32_t &a2, uint32_t &a3,
    const __half *smem, int stride, int row_base, int k_base, int lane)
{
    int gid = lane >> 2, tid = lane & 3;
    int r0 = row_base + gid, r8 = r0 + 8;
    int k0 = k_base + tid * 2, k8 = k0 + 8;

    a0 = pack_h2(&smem[r0 * stride + k0]);
    a1 = pack_h2(&smem[r8 * stride + k0]); // a1↔a2 SWAPPED
    a2 = pack_h2(&smem[r0 * stride + k8]);
    a3 = pack_h2(&smem[r8 * stride + k8]);
}

__device__ __forceinline__ void load_b_frag_kt(
    uint32_t &b0, uint32_t &b1,
    const __half *smem, int stride, int n_base, int k_base, int lane)
{
    int gid = lane >> 2, tid = lane & 3;
    int n = n_base + gid;
    int k0 = k_base + tid * 2;

    b0 = pack_h2(&smem[n * stride + k0]);
    b1 = pack_h2(&smem[n * stride + k0 + 8]);
}

__device__ __forceinline__ void load_b_frag_v(
    uint32_t &b0, uint32_t &b1,
    const __half *smem, int stride, int n_base, int k_base, int lane)
{
    int gid = lane >> 2, tid = lane & 3;
    int n = n_base + gid;
    int k0 = k_base + tid * 2;

    b0 = pack_h2_strided(&smem[k0 * stride + n],
                         &smem[(k0 + 1) * stride + n]);
    b1 = pack_h2_strided(&smem[(k0 + 8) * stride + n],
                         &smem[(k0 + 9) * stride + n]);
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
// Branchless async tile load: 64×128 halves = 1024 × 16B cp.async ops
// OOB rows auto zero-filled by src_bytes=0 (no scalar fallback!)
// =============================================================================

__device__ __forceinline__ void async_load_tile(
    __half *smem, const __half *src_head,
    int tile_start, int seq_len, int head_dim)
{
    // head_dim=128 → 16 chunks of 8 halves (16B) per row
    // 64 rows × 16 chunks = 1024 total, 4 per thread
    constexpr int CHUNKS_PER_ROW = 16;                    // 128/8
    constexpr int TOTAL_CHUNKS = FA4_BC * CHUNKS_PER_ROW; // 1024

#pragma unroll 4
    for (int c = threadIdx.x; c < TOTAL_CHUNKS; c += FA4_THREADS)
    {
        int row = c / CHUNKS_PER_ROW;
        int col8 = (c % CHUNKS_PER_ROW) * 8;
        int grow = tile_start + row;

        __half *dst = &smem[row * FA4_KV_STRIDE + col8];
        const __half *src = &src_head[grow * head_dim + col8];
        int src_bytes = (grow < seq_len) ? 16 : 0;

        cp_async_cg_16(dst, src, src_bytes);
    }
}

// =============================================================================
// FlashAttention v4 Kernel
// =============================================================================

__global__ void __launch_bounds__(FA4_THREADS, 2)
    flash_attention_v4_kernel(
        const __half *__restrict__ Q,
        const __half *__restrict__ K,
        const __half *__restrict__ V,
        __half *__restrict__ O,
        int seq_len, int head_dim, int causal, float scale)
{
    int num_q_tiles = (seq_len + FA4_BR - 1) / FA4_BR;
    int bh = blockIdx.x / num_q_tiles;
    int q_tile = blockIdx.x % num_q_tiles;
    int q_start = q_tile * FA4_BR;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int warp_m = warp_id % 4;
    int warp_n = warp_id / 4;
    int gid = lane_id >> 2;
    int tid = lane_id & 3;

    // ---- SMEM ----
    extern __shared__ char smem_raw[];
    __half *Q_s = (__half *)smem_raw;
    __half *KV_s = (__half *)(smem_raw + FA4_BR * FA4_Q_STRIDE * sizeof(__half));
    __half *S_s = (__half *)(smem_raw + FA4_BR * FA4_Q_STRIDE * sizeof(__half) + FA4_BC * FA4_KV_STRIDE * sizeof(__half));
    float *m_smem = (float *)(smem_raw + FA4_BR * FA4_Q_STRIDE * sizeof(__half) + FA4_BC * FA4_KV_STRIDE * sizeof(__half) + FA4_BR * FA4_S_STRIDE * sizeof(__half));
    float *l_smem = m_smem + FA4_BR;

    // ---- Head pointers ----
    int hstride = seq_len * head_dim;
    const __half *Q_head = Q + bh * hstride;
    const __half *K_head = K + bh * hstride;
    const __half *V_head = V + bh * hstride;
    __half *O_head = O + bh * hstride;

    // ==== Load Q via cp.async (branchless) ====
    {
        constexpr int CHUNKS_PER_ROW = 16;
        constexpr int TOTAL = FA4_BR * CHUNKS_PER_ROW;
#pragma unroll 4
        for (int c = threadIdx.x; c < TOTAL; c += FA4_THREADS)
        {
            int row = c / CHUNKS_PER_ROW;
            int col8 = (c % CHUNKS_PER_ROW) * 8;
            int grow = q_start + row;
            __half *dst = &Q_s[row * FA4_Q_STRIDE + col8];
            const __half *src = &Q_head[grow * head_dim + col8];
            int sb = (grow < seq_len) ? 16 : 0;
            cp_async_cg_16(dst, src, sb);
        }
        cp_async_commit();
    }

    // Init m/l
    for (int i = threadIdx.x; i < FA4_BR; i += FA4_THREADS)
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

    int num_kv_tiles = (seq_len + FA4_BC - 1) / FA4_BC;

    // Prologue: K[0] → KV_s (will be awaited at loop top)
    async_load_tile(KV_s, K_head, 0, seq_len, head_dim);
    cp_async_commit();

    // ---- Main loop ----
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++)
    {
        int kv_start = kv_tile * FA4_BC;
        if (causal && kv_start > q_start + FA4_BR - 1)
            break;

        // ==== Wait for K (from prologue or previous iteration's prefetch) ====
        cp_async_wait<0>();
        __syncthreads();

        // ==== S = Q @ K^T (K in KV_s) ====
        float s_acc[4][4];
#pragma unroll
        for (int t = 0; t < 4; t++)
#pragma unroll
            for (int r = 0; r < 4; r++)
                s_acc[t][r] = 0.0f;

        int m_row_base = warp_m * 16;
        int n_col_base = warp_n * 32;
        int k_steps = head_dim / 16; // 8 for d=128

        for (int ks = 0; ks < k_steps; ks++)
        {
            uint32_t a0, a1, a2, a3;
            load_a_frag(a0, a1, a2, a3, Q_s, FA4_Q_STRIDE,
                        m_row_base, ks * 16, lane_id);

#pragma unroll
            for (int nt = 0; nt < 4; nt++)
            {
                uint32_t b0, b1;
                load_b_frag_kt(b0, b1, KV_s, FA4_KV_STRIDE,
                               n_col_base + nt * 8, ks * 16, lane_id);
                mma_f16_f32(
                    s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3],
                    a0, a1, a2, a3, b0, b1,
                    s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3]);
            }
        }

// ==== Scale + causal mask → S_s ====
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

            store_d_to_smem(S_s, FA4_S_STRIDE, m_row_base, col_base,
                            s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3],
                            lane_id);
        }

        // Sync: S_s visible + all warps done with K in KV_s
        __syncthreads();

        // ==== Softmax + OVERLAPPED V load ====
        // V → KV_s via cp.async (K is dead after QK^T)
        async_load_tile(KV_s, V_head, kv_start, seq_len, head_dim);
        cp_async_commit();

        // Softmax (only S_s + m/l, doesn't touch KV_s)
        float m_old_r0 = m_smem[my_row0];
        float m_old_r8 = m_smem[my_row8];

        {
            int srow_start = warp_m * 16 + warp_n * 8;

            for (int r = 0; r < 8; r++)
            {
                int srow = srow_start + r;

                float v0 = __half2float(S_s[srow * FA4_S_STRIDE + lane_id * 2]);
                float v1 = __half2float(S_s[srow * FA4_S_STRIDE + lane_id * 2 + 1]);

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

                S_s[srow * FA4_S_STRIDE + lane_id * 2] = __float2half(e0);
                S_s[srow * FA4_S_STRIDE + lane_id * 2 + 1] = __float2half(e1);
            }
        }

        // Wait for V
        cp_async_wait<0>();
        __syncthreads();

        // ==== Rescale O ====
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

        // ==== O += P @ V (single merged loop) ====
        int o_n_base = warp_n * 64;
        int pv_k_steps = FA4_BC / 16; // 4

#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            int n_col = o_n_base + nt * 8;

#pragma unroll
            for (int ks = 0; ks < pv_k_steps; ks++)
            {
                uint32_t a0, a1, a2, a3;
                load_a_frag(a0, a1, a2, a3, S_s, FA4_S_STRIDE,
                            m_row_base, ks * 16, lane_id);

                uint32_t b0, b1;
                load_b_frag_v(b0, b1, KV_s, FA4_KV_STRIDE,
                              n_col, ks * 16, lane_id);

                mma_f16_f32(
                    o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3],
                    a0, a1, a2, a3, b0, b1,
                    o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3]);
            }
        }

        // ==== All done with V in KV_s — prefetch K[next] ====
        __syncthreads();

        int next_tile = kv_tile + 1;
        int next_start = next_tile * FA4_BC;
        bool has_next = (next_tile < num_kv_tiles) &&
                        (!causal || next_start <= q_start + FA4_BR - 1);
        if (has_next)
        {
            async_load_tile(KV_s, K_head, next_start, seq_len, head_dim);
            cp_async_commit();
        }
    }

    // ==== Final: O / l → global ====
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

static int g_fa4_smem_max = 0;

extern "C"
{

    int flash_attention_v4_forward(
        const void *Q, const void *K, const void *V, void *O,
        int total_heads, int seq_len, int head_dim,
        int causal, void *stream)
    {
        if (head_dim % 16 != 0 || head_dim > 128)
            return -1;

        int smem_bytes = FA4_BR * FA4_Q_STRIDE * (int)sizeof(__half) + FA4_BC * FA4_KV_STRIDE * (int)sizeof(__half) + FA4_BR * FA4_S_STRIDE * (int)sizeof(__half) + FA4_BR * 2 * (int)sizeof(float);

        if (smem_bytes > g_fa4_smem_max)
        {
            cudaError_t err = cudaFuncSetAttribute(
                flash_attention_v4_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
            if (err != cudaSuccess)
                return (int)err;
            g_fa4_smem_max = smem_bytes;
        }

        float scale_val = 1.0f / sqrtf((float)head_dim);
        int num_q_tiles = (seq_len + FA4_BR - 1) / FA4_BR;
        int total_blocks = total_heads * num_q_tiles;

        flash_attention_v4_kernel<<<total_blocks, FA4_THREADS, smem_bytes, (cudaStream_t)stream>>>(
            (const __half *)Q, (const __half *)K, (const __half *)V, (__half *)O,
            seq_len, head_dim, causal, scale_val);

        return (int)cudaGetLastError();
    }

    int flash_attention_v4_forward_bhsd(
        const void *Q, const void *K, const void *V, void *O,
        int batch, int num_heads, int seq_len, int head_dim,
        int causal, void *stream)
    {
        return flash_attention_v4_forward(Q, K, V, O,
                                          batch * num_heads, seq_len, head_dim, causal, stream);
    }

} // extern "C"
