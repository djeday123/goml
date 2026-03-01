// =============================================================================
// FlashAttention v2 — MMA Tensor Core (SM89) — VERIFIED CORRECT
// =============================================================================
// Fragment mapping verified by mma_probe (all 32 lanes OK):
//   gid = lane >> 2 (0..7),  tid = lane & 3 (0..3)
//   A: a0={gid, k_lo}, a1={gid+8, k_lo}, a2={gid, k_hi}, a3={gid+8, k_hi}
//   B: b0={k_lo, gid},  b1={k_hi, gid}
//   D: d0={gid, col_lo}, d1={gid, col_hi}, d2={gid+8, col_lo}, d3={gid+8, col_hi}
//
// Tiles: Br=64, Bc=64, d=128
// Threads: 256 (8 warps), warp_m = warp_id % 4, warp_n = warp_id / 4
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
// Fragment loading — VERIFIED mapping
// =============================================================================

// A from row-major SMEM: gid=lane>>2, tid=lane&3
// a0={row_base+gid, k_base+tid*2},      a1={row_base+gid+8, k_base+tid*2}  (SWAPPED)
// a2={row_base+gid, k_base+tid*2+8},    a3={row_base+gid+8, k_base+tid*2+8}
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

// B from K^T: K stored row-major as K[n][k], we need B[k][n] col-major
// For QK^T: A=Q, B=K^T → B[k][n]=K[n][k]
// K is in KV_s as row-major: KV_s[n * stride + k]
// b0: k=tid*2, n=gid → strided load from KV_s[gid][tid*2], KV_s[gid][tid*2+1]
// Wait — B expects col-major. n from gid, k from tid.
// b0 packs {B[k0][n], B[k0+1][n]} = {K[n][k0], K[n][k0+1]}
// With K row-major: K[n * stride + k0] and K[n * stride + k0+1] are contiguous!
__device__ __forceinline__ void load_b_frag_kt(
    uint32_t &b0, uint32_t &b1,
    const __half *smem, int stride, int n_base, int k_base, int lane)
{
    int gid = lane >> 2;
    int tid = lane & 3;
    int n = n_base + gid;
    int k0 = k_base + tid * 2;
    int k8 = k0 + 8;

    // K[n][k0..k0+1] contiguous in row-major
    b0 = pack_h2(&smem[n * stride + k0]);
    b1 = pack_h2(&smem[n * stride + k8]);
}

// B from V: V stored row-major as V[k][d], need B[k][n] col-major
// n from gid, k from tid → B[k][n] = V[k][n]
// V row-major: V[k * stride + n], strided access
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

// Store D fragments to SMEM (f32 → f16)
// d0={row_base+gid, col_base+tid*2}, d1={.., col+1}
// d2={row_base+gid+8, col_base+tid*2}, d3={.., col+1}
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
// FlashAttention v2 Kernel
// =============================================================================

__global__ void __launch_bounds__(FA2_THREADS, 2)
    flash_attention_v2_kernel(
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

    // ---- SMEM ----
    extern __shared__ char smem_raw[];
    __half *Q_s = (__half *)smem_raw;
    __half *KV_s = (__half *)(smem_raw + FA2_BR * FA2_Q_STRIDE * sizeof(__half));
    __half *S_s = (__half *)(smem_raw + FA2_BR * FA2_Q_STRIDE * sizeof(__half) + FA2_BC * FA2_KV_STRIDE * sizeof(__half));
    float *m_smem = (float *)(smem_raw + FA2_BR * FA2_Q_STRIDE * sizeof(__half) + FA2_BC * FA2_KV_STRIDE * sizeof(__half) + FA2_BR * FA2_S_STRIDE * sizeof(__half));
    float *l_smem = m_smem + FA2_BR;

    // ---- Head pointers ----
    int hstride = seq_len * head_dim;
    const __half *Q_head = Q + bh * hstride;
    const __half *K_head = K + bh * hstride;
    const __half *V_head = V + bh * hstride;
    __half *O_head = O + bh * hstride;

    // Load Q tile
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

    // O accumulators: each warp covers 16 M-rows, warp_n splits 128 d-cols
    // warp_n=0 → cols 0..63, warp_n=1 → cols 64..127
    // 8 N-tiles of 8 cols each → 8 tiles, 4 regs each
    float o_acc[8][4];
#pragma unroll
    for (int t = 0; t < 8; t++)
#pragma unroll
        for (int r = 0; r < 4; r++)
            o_acc[t][r] = 0.0f;

    __syncthreads();

    // This thread's MMA output rows (for O rescaling)
    int my_row0 = warp_m * 16 + gid; // row in [0..63]
    int my_row8 = my_row0 + 8;

    // ---- Main KV tile loop ----
    int num_kv_tiles = (seq_len + FA2_BC - 1) / FA2_BC;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++)
    {
        int kv_start = kv_tile * FA2_BC;
        if (causal && kv_start > q_start + FA2_BR - 1)
            break;

        // ==== Phase 1: Load K → KV_s ====
        for (int i = threadIdx.x; i < FA2_BC * head_dim; i += FA2_THREADS)
        {
            int row = i / head_dim, col = i % head_dim;
            int grow = kv_start + row;
            KV_s[row * FA2_KV_STRIDE + col] = (grow < seq_len)
                                                  ? K_head[grow * head_dim + col]
                                                  : __float2half(0.0f);
        }
        __syncthreads();

        // ==== Phase 2: S = Q @ K^T via MMA ====
        // Each warp: 16 M-rows × 32 N-cols (warp_n selects which 32 of 64)
        // 4 N-tiles × (d/16) K-steps MMA operations
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
        __syncthreads();

        // ==== Phase 4: Softmax ====
        // Save old m for rescaling O
        float m_old_r0 = m_smem[my_row0];
        float m_old_r8 = m_smem[my_row8];

        // Each warp processes 8 rows. All 32 lanes on same row.
        // 32 lanes × 2 elements = 64 = FA2_BC
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
        __syncthreads();

        // Rescale O accumulators for updated max
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

        // ==== Phase 5: Load V → KV_s ====
        for (int i = threadIdx.x; i < FA2_BC * head_dim; i += FA2_THREADS)
        {
            int row = i / head_dim, col = i % head_dim;
            int grow = kv_start + row;
            KV_s[row * FA2_KV_STRIDE + col] = (grow < seq_len)
                                                  ? V_head[grow * head_dim + col]
                                                  : __float2half(0.0f);
        }
        __syncthreads();

        // ==== Phase 6: O += P @ V via MMA ====
        int o_n_base = warp_n * 64;
        int pv_k_steps = FA2_BC / 16; // 4

        for (int nt = 0; nt < 8; nt++)
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
        __syncthreads();
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

static int g_fa2_smem_max = 0;

extern "C"
{

    int flash_attention_v2_forward(
        const void *Q, const void *K, const void *V, void *O,
        int total_heads, int seq_len, int head_dim,
        int causal, void *stream)
    {
        if (head_dim % 16 != 0 || head_dim > 128)
            return -1;

        int smem_bytes = FA2_BR * FA2_Q_STRIDE * (int)sizeof(__half) + FA2_BC * FA2_KV_STRIDE * (int)sizeof(__half) + FA2_BR * FA2_S_STRIDE * (int)sizeof(__half) + FA2_BR * 2 * (int)sizeof(float);

        if (smem_bytes > g_fa2_smem_max)
        {
            cudaError_t err = cudaFuncSetAttribute(
                flash_attention_v2_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
            if (err != cudaSuccess)
                return (int)err;
            g_fa2_smem_max = smem_bytes;
        }

        float scale_val = 1.0f / sqrtf((float)head_dim);
        int num_q_tiles = (seq_len + FA2_BR - 1) / FA2_BR;
        int total_blocks = total_heads * num_q_tiles;

        flash_attention_v2_kernel<<<total_blocks, FA2_THREADS, smem_bytes, (cudaStream_t)stream>>>(
            (const __half *)Q, (const __half *)K, (const __half *)V, (__half *)O,
            seq_len, head_dim, causal, scale_val);

        return (int)cudaGetLastError();
    }

    int flash_attention_v2_forward_bhsd(
        const void *Q, const void *K, const void *V, void *O,
        int batch, int num_heads, int seq_len, int head_dim,
        int causal, void *stream)
    {
        return flash_attention_v2_forward(Q, K, V, O,
                                          batch * num_heads, seq_len, head_dim, causal, stream);
    }

} // extern "C"
