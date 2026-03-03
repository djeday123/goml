// =============================================================================
// FlashAttention v21 — K-first loop reorder + software exp2f (SM89)
// =============================================================================
// Over v2 (151 TFLOPS on 7B-8K):
//
//   1. K-first loop reorder in QK^T and P@V:
//      v2:  for(nt) for(ks) { load A, load B, MMA }
//      v21: for(ks) { load A ONCE, for(nt) { load B, MMA }}
//      → A fragments stay in registers across all N tiles
//      QK^T: 4× fewer A loads (8 vs 32)
//      P@V:  8× fewer A loads (4 vs 32)
//
//   2. Software exp2f: expf(x) = exp2f(x * LOG2E)
//      exp2f maps to single PTX instruction ex2.approx.f32
//      vs expf which is a multi-instruction sequence
//
// NOTE: ldmatrix optimization prepared but deferred to v21b.
//       One variable at a time — establish reorder+exp2f delta first.
//
// Target: 155-160 TFLOPS (from 151T baseline)
//
// Build:
//   nvcc -O3 -arch=sm_89 -std=c++17 flash_attention_v21.cu -o fa_v21 -lcudart
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

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
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// Software exp2f: single PTX instruction ex2.approx.f32
#define LOG2E 1.4426950408889634f
__device__ __forceinline__ float fast_expf(float x)
{
    return exp2f(x * LOG2E);
}

// =============================================================================
// Fragment loaders (scalar — same as v2)
// =============================================================================

__device__ __forceinline__ void load_a_frag(
    uint32_t &a0, uint32_t &a1, uint32_t &a2, uint32_t &a3,
    const __half *smem, int stride, int m_base, int k_base, int lane)
{
    int gid = lane >> 2, tid = lane & 3;
    int r0 = m_base + gid, r8 = r0 + 8;
    int k0 = k_base + tid * 2, k8 = k0 + 8;
    a0 = pack_h2(&smem[r0 * stride + k0]);
    a1 = pack_h2(&smem[r8 * stride + k0]);
    a2 = pack_h2(&smem[r0 * stride + k8]);
    a3 = pack_h2(&smem[r8 * stride + k8]);
}

__device__ __forceinline__ void load_b_frag_kt(
    uint32_t &b0, uint32_t &b1,
    const __half *smem, int stride, int n_base, int k_base, int lane)
{
    int gid = lane >> 2, tid = lane & 3;
    int n = n_base + gid;
    int k0 = k_base + tid * 2, k8 = k0 + 8;
    b0 = pack_h2(&smem[n * stride + k0]);
    b1 = pack_h2(&smem[n * stride + k8]);
}

__device__ __forceinline__ void load_b_frag_v(
    uint32_t &b0, uint32_t &b1,
    const __half *smem, int stride, int n_base, int k_base, int lane)
{
    int gid = lane >> 2, tid = lane & 3;
    int n = n_base + gid;
    int k0 = k_base + tid * 2, k8 = k0 + 8;
    b0 = pack_h2_strided(&smem[k0 * stride + n], &smem[(k0 + 1) * stride + n]);
    b1 = pack_h2_strided(&smem[k8 * stride + n], &smem[(k8 + 1) * stride + n]);
}

// =============================================================================
// Store S tile to SMEM with scale and causal mask
// =============================================================================

__device__ __forceinline__ void store_s_tile(
    __half *S_s, int s_stride, float s_acc[4][4], float scale,
    int m_row_base, int n_base_start, int q_start, int kv_start,
    int gid, int tid, int causal)
{
#pragma unroll
    for (int nt = 0; nt < 4; nt++)
    {
        int col0 = n_base_start + nt * 8 + tid * 2;
        int col1 = col0 + 1;
        int r0 = m_row_base + gid, r8 = r0 + 8;
        float v0 = s_acc[nt][0] * scale, v1 = s_acc[nt][1] * scale;
        float v2 = s_acc[nt][2] * scale, v3 = s_acc[nt][3] * scale;
        if (causal)
        {
            int gq0 = q_start + r0, gq8 = q_start + r8;
            int gk0 = kv_start + col0, gk1 = kv_start + col1;
            if (gk0 > gq0)
                v0 = -1e30f;
            if (gk1 > gq0)
                v1 = -1e30f;
            if (gk0 > gq8)
                v2 = -1e30f;
            if (gk1 > gq8)
                v3 = -1e30f;
        }
        S_s[r0 * s_stride + col0] = __float2half(v0);
        S_s[r0 * s_stride + col1] = __float2half(v1);
        S_s[r8 * s_stride + col0] = __float2half(v2);
        S_s[r8 * s_stride + col1] = __float2half(v3);
    }
}

// =============================================================================
// v2 baseline kernel (reference — original loop order, expf)
// =============================================================================

__global__ void __launch_bounds__(FA2_THREADS, 2)
    flash_attn_v2_kernel(
        const __half *__restrict__ Q, const __half *__restrict__ K,
        const __half *__restrict__ V, __half *__restrict__ O,
        int seq_len, int head_dim, int causal, float scale)
{
    int num_q_tiles = (seq_len + FA2_BR - 1) / FA2_BR;
    int bh = blockIdx.x / num_q_tiles;
    int q_tile = blockIdx.x % num_q_tiles;
    int q_start = q_tile * FA2_BR;
    int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32;
    int warp_m = warp_id % 4, warp_n = warp_id / 4;
    int gid = lane_id >> 2, tid = lane_id & 3;

    extern __shared__ char smem_raw[];
    __half *Q_s = (__half *)smem_raw;
    __half *KV_s = (__half *)(smem_raw + FA2_BR * FA2_Q_STRIDE * sizeof(__half));
    __half *S_s = (__half *)(smem_raw + FA2_BR * FA2_Q_STRIDE * sizeof(__half) + FA2_BC * FA2_KV_STRIDE * sizeof(__half));
    float *m_smem = (float *)(smem_raw + FA2_BR * FA2_Q_STRIDE * sizeof(__half) + FA2_BC * FA2_KV_STRIDE * sizeof(__half) + FA2_BR * FA2_S_STRIDE * sizeof(__half));
    float *l_smem = m_smem + FA2_BR;

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

    __syncthreads();

    int my_row0 = warp_m * 16 + gid, my_row8 = my_row0 + 8;
    int m_row_base = warp_m * 16;
    int num_kv_tiles = (seq_len + FA2_BC - 1) / FA2_BC;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++)
    {
        int kv_start = kv_tile * FA2_BC;
        if (causal && kv_start > q_start + FA2_BR - 1)
            break;

        // Load K
        for (int i = threadIdx.x; i < FA2_BC * head_dim; i += FA2_THREADS)
        {
            int row = i / head_dim, col = i % head_dim;
            int grow = kv_start + row;
            KV_s[row * FA2_KV_STRIDE + col] = (grow < seq_len)
                                                  ? K_head[grow * head_dim + col]
                                                  : __float2half(0.0f);
        }
        __syncthreads();

        // QK^T — v2 ORIGINAL: for each N-tile, reload A each K-step
        {
            int qk_k_steps = head_dim / 16;
            int n_base_start = warp_n * 32;
            float s_acc[4][4];
#pragma unroll
            for (int nt = 0; nt < 4; nt++)
                s_acc[nt][0] = s_acc[nt][1] = s_acc[nt][2] = s_acc[nt][3] = 0.0f;

            for (int nt = 0; nt < 4; nt++)
            {
                int n_col = n_base_start + nt * 8;
                for (int ks = 0; ks < qk_k_steps; ks++)
                {
                    uint32_t a0, a1, a2, a3;
                    load_a_frag(a0, a1, a2, a3, Q_s, FA2_Q_STRIDE,
                                m_row_base, ks * 16, lane_id);
                    uint32_t b0, b1;
                    load_b_frag_kt(b0, b1, KV_s, FA2_KV_STRIDE,
                                   n_col, ks * 16, lane_id);
                    mma_f16_f32(s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3],
                                a0, a1, a2, a3, b0, b1,
                                s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3]);
                }
            }
            store_s_tile(S_s, FA2_S_STRIDE, s_acc, scale, m_row_base,
                         n_base_start, q_start, kv_start, gid, tid, causal);
        }
        __syncthreads();

        // Online Softmax — v2: standard expf
        float m_old_r0 = m_smem[my_row0], m_old_r8 = m_smem[my_row8];
        {
            int srow_start = warp_m * 16 + warp_n * 8;
            for (int r = 0; r < 8; r++)
            {
                int srow = srow_start + r;
                float v0 = __half2float(S_s[srow * FA2_S_STRIDE + lane_id * 2]);
                float v1 = __half2float(S_s[srow * FA2_S_STRIDE + lane_id * 2 + 1]);
                float row_max = warp_reduce_max(fmaxf(v0, v1));
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
        __syncthreads();

        // Rescale O
        float m_new_r0 = m_smem[my_row0], m_new_r8 = m_smem[my_row8];
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

        // Load V
        for (int i = threadIdx.x; i < FA2_BC * head_dim; i += FA2_THREADS)
        {
            int row = i / head_dim, col = i % head_dim;
            int grow = kv_start + row;
            KV_s[row * FA2_KV_STRIDE + col] = (grow < seq_len)
                                                  ? V_head[grow * head_dim + col]
                                                  : __float2half(0.0f);
        }
        __syncthreads();

        // P@V — v2 ORIGINAL: for each N-tile, reload A each K-step
        {
            int o_n_base = warp_n * 64;
            int pv_k_steps = FA2_BC / 16;
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
                    mma_f16_f32(o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3],
                                a0, a1, a2, a3, b0, b1,
                                o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3]);
                }
            }
        }
        __syncthreads();
    }

    // Final: O / l
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

// =============================================================================
// v21 kernel — K-first loop reorder + fast_expf
// =============================================================================
// Changes from v2:
//   QK^T: for(ks) { load_A_once; for(nt) { load_B, MMA } }
//   P@V:  for(ks) { load_A_once; for(nt) { load_B, MMA } }
//   Softmax: fast_expf (exp2f * LOG2E) instead of expf
// =============================================================================

__global__ void __launch_bounds__(FA2_THREADS, 2)
    flash_attn_v21_kernel(
        const __half *__restrict__ Q, const __half *__restrict__ K,
        const __half *__restrict__ V, __half *__restrict__ O,
        int seq_len, int head_dim, int causal, float scale)
{
    int num_q_tiles = (seq_len + FA2_BR - 1) / FA2_BR;
    int bh = blockIdx.x / num_q_tiles;
    int q_tile = blockIdx.x % num_q_tiles;
    int q_start = q_tile * FA2_BR;
    int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32;
    int warp_m = warp_id % 4, warp_n = warp_id / 4;
    int gid = lane_id >> 2, tid = lane_id & 3;

    extern __shared__ char smem_raw[];
    __half *Q_s = (__half *)smem_raw;
    __half *KV_s = (__half *)(smem_raw + FA2_BR * FA2_Q_STRIDE * sizeof(__half));
    __half *S_s = (__half *)(smem_raw + FA2_BR * FA2_Q_STRIDE * sizeof(__half) + FA2_BC * FA2_KV_STRIDE * sizeof(__half));
    float *m_smem = (float *)(smem_raw + FA2_BR * FA2_Q_STRIDE * sizeof(__half) + FA2_BC * FA2_KV_STRIDE * sizeof(__half) + FA2_BR * FA2_S_STRIDE * sizeof(__half));
    float *l_smem = m_smem + FA2_BR;

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

    __syncthreads();

    int my_row0 = warp_m * 16 + gid, my_row8 = my_row0 + 8;
    int m_row_base = warp_m * 16;
    int num_kv_tiles = (seq_len + FA2_BC - 1) / FA2_BC;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++)
    {
        int kv_start = kv_tile * FA2_BC;
        if (causal && kv_start > q_start + FA2_BR - 1)
            break;

        // ==== Load K ====
        for (int i = threadIdx.x; i < FA2_BC * head_dim; i += FA2_THREADS)
        {
            int row = i / head_dim, col = i % head_dim;
            int grow = kv_start + row;
            KV_s[row * FA2_KV_STRIDE + col] = (grow < seq_len)
                                                  ? K_head[grow * head_dim + col]
                                                  : __float2half(0.0f);
        }
        __syncthreads();

        // ==== QK^T — v21: K-first loop, A loaded ONCE per K-step ====
        {
            int qk_k_steps = head_dim / 16;
            int n_base_start = warp_n * 32;

            float s_acc[4][4];
#pragma unroll
            for (int nt = 0; nt < 4; nt++)
                s_acc[nt][0] = s_acc[nt][1] = s_acc[nt][2] = s_acc[nt][3] = 0.0f;

            for (int ks = 0; ks < qk_k_steps; ks++)
            {
                // Load A (Q) ONCE — reused across all 4 N-tiles
                uint32_t a0, a1, a2, a3;
                load_a_frag(a0, a1, a2, a3, Q_s, FA2_Q_STRIDE,
                            m_row_base, ks * 16, lane_id);

#pragma unroll
                for (int nt = 0; nt < 4; nt++)
                {
                    int n_col = n_base_start + nt * 8;
                    uint32_t b0, b1;
                    load_b_frag_kt(b0, b1, KV_s, FA2_KV_STRIDE,
                                   n_col, ks * 16, lane_id);
                    mma_f16_f32(
                        s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3],
                        a0, a1, a2, a3, b0, b1,
                        s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3]);
                }
            }

            store_s_tile(S_s, FA2_S_STRIDE, s_acc, scale, m_row_base,
                         n_base_start, q_start, kv_start, gid, tid, causal);
        }
        __syncthreads();

        // ==== Online Softmax — v21: fast_expf ====
        float m_old_r0 = m_smem[my_row0], m_old_r8 = m_smem[my_row8];
        {
            int srow_start = warp_m * 16 + warp_n * 8;
            for (int r = 0; r < 8; r++)
            {
                int srow = srow_start + r;
                float v0 = __half2float(S_s[srow * FA2_S_STRIDE + lane_id * 2]);
                float v1 = __half2float(S_s[srow * FA2_S_STRIDE + lane_id * 2 + 1]);
                float row_max = warp_reduce_max(fmaxf(v0, v1));
                float m_old = m_smem[srow];
                float m_new = fmaxf(m_old, row_max);

                // >>> v21: fast_expf — single ex2.approx.f32 PTX instruction <<<
                float e0 = fast_expf(v0 - m_new);
                float e1 = fast_expf(v1 - m_new);
                float row_sum = warp_reduce_sum(e0 + e1);
                float rescale_old = fast_expf(m_old - m_new);

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

        // Rescale O with fast_expf
        float m_new_r0 = m_smem[my_row0], m_new_r8 = m_smem[my_row8];
        float rescale0 = fast_expf(m_old_r0 - m_new_r0);
        float rescale8 = fast_expf(m_old_r8 - m_new_r8);
#pragma unroll
        for (int t = 0; t < 8; t++)
        {
            o_acc[t][0] *= rescale0;
            o_acc[t][1] *= rescale0;
            o_acc[t][2] *= rescale8;
            o_acc[t][3] *= rescale8;
        }

        // ==== Load V ====
        for (int i = threadIdx.x; i < FA2_BC * head_dim; i += FA2_THREADS)
        {
            int row = i / head_dim, col = i % head_dim;
            int grow = kv_start + row;
            KV_s[row * FA2_KV_STRIDE + col] = (grow < seq_len)
                                                  ? V_head[grow * head_dim + col]
                                                  : __float2half(0.0f);
        }
        __syncthreads();

        // ==== P@V — v21: K-first loop, A loaded ONCE per K-step ====
        {
            int o_n_base = warp_n * 64;
            int pv_k_steps = FA2_BC / 16; // 4

            for (int ks = 0; ks < pv_k_steps; ks++)
            {
                // Load A (S/P row) ONCE — reused across all 8 N-tiles
                uint32_t a0, a1, a2, a3;
                load_a_frag(a0, a1, a2, a3, S_s, FA2_S_STRIDE,
                            m_row_base, ks * 16, lane_id);

#pragma unroll
                for (int nt = 0; nt < 8; nt++)
                {
                    int n_col = o_n_base + nt * 8;
                    uint32_t b0, b1;
                    load_b_frag_v(b0, b1, KV_s, FA2_KV_STRIDE,
                                  n_col, ks * 16, lane_id);
                    mma_f16_f32(
                        o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3],
                        a0, a1, a2, a3, b0, b1,
                        o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3]);
                }
            }
        }
        __syncthreads();
    }

    // ==== Final: O / l ====
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

// =============================================================================
// C API + Benchmark harness
// =============================================================================

#define CK(c)                                                       \
    do                                                              \
    {                                                               \
        cudaError_t e = (c);                                        \
        if (e != cudaSuccess)                                       \
        {                                                           \
            fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(e));                         \
            exit(1);                                                \
        }                                                           \
    } while (0)

static inline float h2f(uint16_t h)
{
    __half hv;
    memcpy(&hv, &h, 2);
    return __half2float(hv);
}
static inline uint16_t f2h(float f)
{
    __half hv = __float2half(f);
    uint16_t r;
    memcpy(&r, &hv, 2);
    return r;
}

void fill_random_fp16(uint16_t *d_ptr, int n)
{
    uint16_t *h = (uint16_t *)malloc(n * 2);
    for (int i = 0; i < n; i++)
        h[i] = f2h(((float)(rand() % 2001) - 1000.0f) / 1000.0f);
    CK(cudaMemcpy(d_ptr, h, n * 2, cudaMemcpyHostToDevice));
    free(h);
}

struct Timer
{
    cudaEvent_t t0, t1;
    Timer()
    {
        CK(cudaEventCreate(&t0));
        CK(cudaEventCreate(&t1));
    }
    ~Timer()
    {
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }
    void start() { CK(cudaEventRecord(t0)); }
    float stop()
    {
        CK(cudaEventRecord(t1));
        CK(cudaEventSynchronize(t1));
        float ms;
        CK(cudaEventElapsedTime(&ms, t0, t1));
        return ms;
    }
};

static int g_smem_v2 = 0, g_smem_v21 = 0;

void launch_v2(const __half *Q, const __half *K, const __half *V, __half *O,
               int total_heads, int seq_len, int head_dim, int causal)
{
    int smem = FA2_BR * FA2_Q_STRIDE * 2 + FA2_BC * FA2_KV_STRIDE * 2 + FA2_BR * FA2_S_STRIDE * 2 + FA2_BR * 2 * 4;
    if (smem > g_smem_v2)
    {
        CK(cudaFuncSetAttribute(flash_attn_v2_kernel,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        g_smem_v2 = smem;
    }
    float scale = 1.0f / sqrtf((float)head_dim);
    int ntiles = (seq_len + FA2_BR - 1) / FA2_BR;
    flash_attn_v2_kernel<<<total_heads * ntiles, FA2_THREADS, smem>>>(
        Q, K, V, O, seq_len, head_dim, causal, scale);
}

void launch_v21(const __half *Q, const __half *K, const __half *V, __half *O,
                int total_heads, int seq_len, int head_dim, int causal)
{
    int smem = FA2_BR * FA2_Q_STRIDE * 2 + FA2_BC * FA2_KV_STRIDE * 2 + FA2_BR * FA2_S_STRIDE * 2 + FA2_BR * 2 * 4;
    if (smem > g_smem_v21)
    {
        CK(cudaFuncSetAttribute(flash_attn_v21_kernel,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        g_smem_v21 = smem;
    }
    float scale = 1.0f / sqrtf((float)head_dim);
    int ntiles = (seq_len + FA2_BR - 1) / FA2_BR;
    flash_attn_v21_kernel<<<total_heads * ntiles, FA2_THREADS, smem>>>(
        Q, K, V, O, seq_len, head_dim, causal, scale);
}

// =============================================================================
// Correctness: v2 vs v21 vs CPU reference
// =============================================================================

void test_correctness()
{
    printf("--- Correctness: v2 vs v21 vs CPU ---\n");

    int configs[][3] = {
        {1, 32, 128}, {1, 64, 128}, {2, 128, 128}, {1, 256, 128}, {1, 512, 128}};

    for (auto &c : configs)
    {
        int heads = c[0], seq = c[1], dim = c[2];
        int n = heads * seq * dim;
        size_t sz = (size_t)n * 2;

        uint16_t *hQ = (uint16_t *)malloc(sz);
        uint16_t *hK = (uint16_t *)malloc(sz);
        uint16_t *hV = (uint16_t *)malloc(sz);
        float *ref = (float *)calloc(n, 4);

        srand(42);
        for (int i = 0; i < n; i++)
            hQ[i] = f2h(((float)(rand() % 2001) - 1000.0f) / 1000.0f);
        for (int i = 0; i < n; i++)
            hK[i] = f2h(((float)(rand() % 2001) - 1000.0f) / 1000.0f);
        for (int i = 0; i < n; i++)
            hV[i] = f2h(((float)(rand() % 2001) - 1000.0f) / 1000.0f);

        // CPU reference (causal)
        float scale = 1.0f / sqrtf((float)dim);
        for (int h = 0; h < heads; h++)
        {
            int off = h * seq * dim;
            for (int q = 0; q < seq; q++)
            {
                float row_max = -1e30f;
                float *scores = (float *)calloc(seq, 4);
                for (int k = 0; k <= q; k++)
                {
                    float dot = 0;
                    for (int d = 0; d < dim; d++)
                        dot += h2f(hQ[off + q * dim + d]) * h2f(hK[off + k * dim + d]);
                    scores[k] = dot * scale;
                    row_max = fmaxf(row_max, scores[k]);
                }
                float sum = 0;
                for (int k = 0; k <= q; k++)
                {
                    scores[k] = expf(scores[k] - row_max);
                    sum += scores[k];
                }
                for (int d = 0; d < dim; d++)
                {
                    float acc = 0;
                    for (int k = 0; k <= q; k++)
                        acc += (scores[k] / sum) * h2f(hV[off + k * dim + d]);
                    ref[off + q * dim + d] = acc;
                }
                free(scores);
            }
        }

        void *dQ, *dK, *dV;
        __half *dO_v2, *dO_v21;
        CK(cudaMalloc(&dQ, sz));
        CK(cudaMalloc(&dK, sz));
        CK(cudaMalloc(&dV, sz));
        CK(cudaMalloc(&dO_v2, sz));
        CK(cudaMalloc(&dO_v21, sz));
        CK(cudaMemcpy(dQ, hQ, sz, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dK, hK, sz, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dV, hV, sz, cudaMemcpyHostToDevice));
        CK(cudaMemset(dO_v2, 0, sz));
        CK(cudaMemset(dO_v21, 0, sz));

        launch_v2((const __half *)dQ, (const __half *)dK, (const __half *)dV,
                  dO_v2, heads, seq, dim, 1);
        launch_v21((const __half *)dQ, (const __half *)dK, (const __half *)dV,
                   dO_v21, heads, seq, dim, 1);
        CK(cudaDeviceSynchronize());

        uint16_t *hO_v2 = (uint16_t *)malloc(sz);
        uint16_t *hO_v21 = (uint16_t *)malloc(sz);
        CK(cudaMemcpy(hO_v2, dO_v2, sz, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hO_v21, dO_v21, sz, cudaMemcpyDeviceToHost));

        float mx2 = 0, mx21 = 0;
        int e2 = 0, e21 = 0;
        for (int i = 0; i < n; i++)
        {
            float ae2 = fabsf(h2f(hO_v2[i]) - ref[i]);
            float ae21 = fabsf(h2f(hO_v21[i]) - ref[i]);
            if (ae2 > mx2)
                mx2 = ae2;
            if (ae21 > mx21)
                mx21 = ae21;
            float thr = fmaxf(0.002f, fabsf(ref[i]) * 0.05f);
            if (ae2 > thr)
                e2++;
            if (ae21 > thr)
                e21++;
        }

        printf("  h=%d s=%3d d=%d  v2: max=%.4f err=%d %s  "
               "v21: max=%.4f err=%d %s\n",
               heads, seq, dim,
               mx2, e2, e2 == 0 ? "PASS" : "FAIL",
               mx21, e21, e21 == 0 ? "PASS" : "FAIL");

        cudaFree(dQ);
        cudaFree(dK);
        cudaFree(dV);
        cudaFree(dO_v2);
        cudaFree(dO_v21);
        free(hQ);
        free(hK);
        free(hV);
        free(ref);
        free(hO_v2);
        free(hO_v21);
    }
}

// =============================================================================
// Performance benchmark
// =============================================================================

void bench()
{
    printf("\n--- Performance: v2 (baseline) vs v21 (reorder + exp2f) ---\n");
    printf("%-12s %10s %10s %8s\n", "Config", "v2 (T)", "v21 (T)", "delta");
    printf("------------------------------------------------\n");

    struct
    {
        int b, h, s, d;
        const char *label;
    } configs[] = {
        {1, 32, 256, 128, "7B-256"},
        {1, 32, 512, 128, "7B-512"},
        {1, 32, 1024, 128, "7B-1K"},
        {1, 32, 2048, 128, "7B-2K"},
        {1, 32, 4096, 128, "7B-4K"},
        {1, 32, 8192, 128, "7B-8K"},
        {1, 40, 512, 128, "13B-512"},
        {1, 64, 512, 128, "70B-512"},
        {1, 64, 2048, 128, "70B-2K"},
        {1, 64, 4096, 128, "70B-4K"},
    };

    Timer t;

    for (auto &c : configs)
    {
        int n = c.b * c.h * c.s * c.d;
        int total_heads = c.b * c.h;
        // Causal attention FLOPs: 4 * s^2 * d / 2 per head (triangle)
        double flops = 4.0 * c.b * c.h * (double)c.s * c.s * c.d;

        void *dQ, *dK, *dV;
        __half *dO;
        CK(cudaMalloc(&dQ, (size_t)n * 2));
        CK(cudaMalloc(&dK, (size_t)n * 2));
        CK(cudaMalloc(&dV, (size_t)n * 2));
        CK(cudaMalloc(&dO, (size_t)n * 2));
        fill_random_fp16((uint16_t *)dQ, n);
        fill_random_fp16((uint16_t *)dK, n);
        fill_random_fp16((uint16_t *)dV, n);

        // ---- v2 ----
        for (int i = 0; i < 3; i++)
            launch_v2((const __half *)dQ, (const __half *)dK,
                      (const __half *)dV, dO, total_heads, c.s, c.d, 1);
        CK(cudaDeviceSynchronize());
        int it = (c.s <= 1024) ? 100 : 20;
        t.start();
        for (int i = 0; i < it; i++)
            launch_v2((const __half *)dQ, (const __half *)dK,
                      (const __half *)dV, dO, total_heads, c.s, c.d, 1);
        float ms_v2 = t.stop();
        double tf_v2 = flops / (ms_v2 / it / 1000.0) / 1e12;

        // ---- v21 ----
        CK(cudaMemset(dO, 0, (size_t)n * 2));
        for (int i = 0; i < 3; i++)
            launch_v21((const __half *)dQ, (const __half *)dK,
                       (const __half *)dV, dO, total_heads, c.s, c.d, 1);
        CK(cudaDeviceSynchronize());
        t.start();
        for (int i = 0; i < it; i++)
            launch_v21((const __half *)dQ, (const __half *)dK,
                       (const __half *)dV, dO, total_heads, c.s, c.d, 1);
        float ms_v21 = t.stop();
        double tf_v21 = flops / (ms_v21 / it / 1000.0) / 1e12;

        printf("%-12s %10.1f %10.1f %+8.1f", c.label, tf_v2, tf_v21,
               tf_v21 - tf_v2);
        if (tf_v21 > tf_v2 * 1.02)
            printf("  ** WIN **");
        if (tf_v21 < tf_v2 * 0.98)
            printf("  !! LOSS !!");
        printf("\n");

        cudaFree(dQ);
        cudaFree(dK);
        cudaFree(dV);
        cudaFree(dO);
    }

    printf("\nv2 baseline: ~151T on 7B-8K @ OC\n");
    printf("Target v21: 155-160T (reorder + exp2f)\n");
    printf("Next: v21b (+ ldmatrix) → 159-163T\n");
}

// =============================================================================
// Main
// =============================================================================

int main()
{
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("=== FlashAttention v21 — K-first reorder + fast_expf ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n\n",
           p.name, p.multiProcessorCount, p.clockRate / 1000);

    test_correctness();
    bench();

    return 0;
}
