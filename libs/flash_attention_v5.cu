// =============================================================================
// FlashAttention v5 — ldmatrix + cp.async (SM89)
// =============================================================================
// Changes vs v4 (168 TFLOPS):
//   1. ldmatrix.x4 for A fragments (Q, P) — 1 instruction vs 4× pack_h2
//   2. ldmatrix.x2 for B fragments from K — 1 instruction vs 2× pack_h2
//   3. ldmatrix.x2.trans for B fragments from V — hardware transpose!
//      Eliminates strided access (was 4 separate SMEM loads per fragment)
//
// ldmatrix loads MMA fragments in hardware-native layout.
// No a1↔a2 swap needed — ldmatrix produces what mma.sync expects.
//
// Thread address mapping for ldmatrix:
//   Each thread provides a 16B-aligned SMEM address.
//   .x4: sub = lane/8 (0-3), sub_row = lane%8 (0-7)
//   .x2: sub = (lane/8)%2 (0-1), sub_row = lane%8 (0-7)
//
// Same SMEM layout (44KB → 2 blocks/SM), same pipeline as v4.
// =============================================================================

#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define FA5_BR 64
#define FA5_BC 64
#define FA5_THREADS 256

#define FA5_Q_STRIDE 136
#define FA5_KV_STRIDE 136
#define FA5_S_STRIDE 72

// =============================================================================
// cp.async (SM80+)
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
// ldmatrix intrinsics (SM75+)
// =============================================================================

// Load 4×(8×8) matrices for A operand (m16n8k16)
__device__ __forceinline__ void ldmatrix_x4(
    uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3,
    const void *smem_ptr)
{
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}

// Load 2×(8×8) matrices for B operand (contiguous K)
__device__ __forceinline__ void ldmatrix_x2(
    uint32_t &r0, uint32_t &r1,
    const void *smem_ptr)
{
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];"
        : "=r"(r0), "=r"(r1)
        : "r"(addr));
}

// Load 2×(8×8) with transpose for B operand (V — row-major → col-major)
__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t &r0, uint32_t &r1,
    const void *smem_ptr)
{
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16.trans {%0,%1}, [%2];"
        : "=r"(r0), "=r"(r1)
        : "r"(addr));
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
// pack_h2 — only needed for S_s store (store_d_to_smem)
// =============================================================================

__device__ __forceinline__ uint32_t pack_h2(const __half *ptr)
{
    uint32_t r;
    asm("mov.b32 %0, {%1, %2};" : "=r"(r)
        : "h"(*(const unsigned short *)&ptr[0]),
          "h"(*(const unsigned short *)&ptr[1]));
    return r;
}

// =============================================================================
// Fragment loading via ldmatrix
// =============================================================================

// A from row-major SMEM (Q_s or S_s): ldmatrix.x4
// Loads 16×16 block at [row_base..+15][k_base..+15]
// Sub-matrix layout:
//   sub 0 (threads 0-7):   rows 0-7,  k 0-7   → r0
//   sub 1 (threads 8-15):  rows 8-15, k 0-7   → r1
//   sub 2 (threads 16-23): rows 0-7,  k 8-15  → r2
//   sub 3 (threads 24-31): rows 8-15, k 8-15  → r3
__device__ __forceinline__ void load_a_ldm(
    uint32_t &a0, uint32_t &a1, uint32_t &a2, uint32_t &a3,
    const __half *smem, int stride, int row_base, int k_base, int lane)
{
    int sub = lane / 8;     // 0,1,2,3
    int sub_row = lane % 8; // 0-7

    int row = row_base + (sub & 1) * 8 + sub_row; // sub 0,2 → rows 0-7; sub 1,3 → rows 8-15
    int col = k_base + (sub >> 1) * 8;            // sub 0,1 → k_lo; sub 2,3 → k_hi

    ldmatrix_x4(a0, a1, a2, a3, &smem[row * stride + col]);
}

// B from K (row-major K[n][k]): ldmatrix.x2
// Loads 8n × 16k block: K[n_base..+7][k_base..+15]
// This IS col-major B[k][n] for MMA (K's rows are contiguous in k)
//   sub 0: n 0-7, k 0-7   → r0 (b0)
//   sub 1: n 0-7, k 8-15  → r1 (b1)
__device__ __forceinline__ void load_b_kt_ldm(
    uint32_t &b0, uint32_t &b1,
    const __half *smem, int stride, int n_base, int k_base, int lane)
{
    int sub = (lane / 8) % 2; // 0 or 1
    int sub_row = lane % 8;   // 0-7

    int row = n_base + sub_row;
    int col = k_base + sub * 8;

    ldmatrix_x2(b0, b1, &smem[row * stride + col]);
}

// B from V (row-major V[k][d]): ldmatrix.x2.trans
// V[k][d] is row-major → contiguous in d dimension.
// For MMA B[k][n]: need col-major (k varying fast).
// .trans transposes each 8×8 sub-matrix during load!
//
// We load V[k_base..+15][n_base..+7] as two 8-row strips:
//   sub 0: V rows k_base..k_base+7, cols n_base..n_base+7 → transpose → b0
//   sub 1: V rows k_base+8..k_base+15, cols n_base..n_base+7 → transpose → b1
__device__ __forceinline__ void load_b_v_ldm(
    uint32_t &b0, uint32_t &b1,
    const __half *smem, int stride, int n_base, int k_base, int lane)
{
    int sub = (lane / 8) % 2;
    int sub_row = lane % 8;

    int k = k_base + sub * 8 + sub_row;

    ldmatrix_x2_trans(b0, b1, &smem[k * stride + n_base]);
}

// Store S (D fragments) to SMEM — no ldmatrix for stores
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
// Async tile load (branchless cp.async)
// =============================================================================

__device__ __forceinline__ void async_load_tile(
    __half *smem, const __half *src_head,
    int tile_start, int seq_len, int head_dim)
{
    constexpr int CHUNKS_PER_ROW = 16; // 128/8
    constexpr int TOTAL_CHUNKS = FA5_BC * CHUNKS_PER_ROW;

#pragma unroll 4
    for (int c = threadIdx.x; c < TOTAL_CHUNKS; c += FA5_THREADS)
    {
        int row = c / CHUNKS_PER_ROW;
        int col8 = (c % CHUNKS_PER_ROW) * 8;
        int grow = tile_start + row;

        __half *dst = &smem[row * FA5_KV_STRIDE + col8];
        const __half *src = &src_head[grow * head_dim + col8];
        int src_bytes = (grow < seq_len) ? 16 : 0;

        cp_async_cg_16(dst, src, src_bytes);
    }
}

// =============================================================================
// FlashAttention v5 Kernel
// =============================================================================

__global__ void __launch_bounds__(FA5_THREADS, 2)
    flash_attention_v5_kernel(
        const __half *__restrict__ Q,
        const __half *__restrict__ K,
        const __half *__restrict__ V,
        __half *__restrict__ O,
        int seq_len, int head_dim, int causal, float scale)
{
    int num_q_tiles = (seq_len + FA5_BR - 1) / FA5_BR;
    int bh = blockIdx.x / num_q_tiles;
    int q_tile = blockIdx.x % num_q_tiles;
    int q_start = q_tile * FA5_BR;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int warp_m = warp_id % 4;
    int warp_n = warp_id / 4;
    int gid = lane_id >> 2;
    int tid = lane_id & 3;

    // ---- SMEM ----
    extern __shared__ char smem_raw[];
    __half *Q_s = (__half *)smem_raw;
    __half *KV_s = (__half *)(smem_raw + FA5_BR * FA5_Q_STRIDE * sizeof(__half));
    __half *S_s = (__half *)(smem_raw + FA5_BR * FA5_Q_STRIDE * sizeof(__half) + FA5_BC * FA5_KV_STRIDE * sizeof(__half));
    float *m_smem = (float *)(smem_raw + FA5_BR * FA5_Q_STRIDE * sizeof(__half) + FA5_BC * FA5_KV_STRIDE * sizeof(__half) + FA5_BR * FA5_S_STRIDE * sizeof(__half));
    float *l_smem = m_smem + FA5_BR;

    // ---- Head pointers ----
    int hstride = seq_len * head_dim;
    const __half *Q_head = Q + bh * hstride;
    const __half *K_head = K + bh * hstride;
    const __half *V_head = V + bh * hstride;
    __half *O_head = O + bh * hstride;

    // ==== Load Q via cp.async ====
    {
        constexpr int CHUNKS_PER_ROW = 16;
        constexpr int TOTAL = FA5_BR * CHUNKS_PER_ROW;
#pragma unroll 4
        for (int c = threadIdx.x; c < TOTAL; c += FA5_THREADS)
        {
            int row = c / CHUNKS_PER_ROW;
            int col8 = (c % CHUNKS_PER_ROW) * 8;
            int grow = q_start + row;
            __half *dst = &Q_s[row * FA5_Q_STRIDE + col8];
            const __half *src = &Q_head[grow * head_dim + col8];
            int sb = (grow < seq_len) ? 16 : 0;
            cp_async_cg_16(dst, src, sb);
        }
        cp_async_commit();
    }

    // Init m/l
    for (int i = threadIdx.x; i < FA5_BR; i += FA5_THREADS)
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

    int num_kv_tiles = (seq_len + FA5_BC - 1) / FA5_BC;

    // Prologue: K[0] → KV_s
    async_load_tile(KV_s, K_head, 0, seq_len, head_dim);
    cp_async_commit();

    // ---- Main loop ----
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++)
    {
        int kv_start = kv_tile * FA5_BC;
        if (causal && kv_start > q_start + FA5_BR - 1)
            break;

        // Wait for K
        cp_async_wait<0>();
        __syncthreads();

        // ==== S = Q @ K^T via MMA (ldmatrix) ====
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
            load_a_ldm(a0, a1, a2, a3, Q_s, FA5_Q_STRIDE,
                       m_row_base, ks * 16, lane_id);

#pragma unroll
            for (int nt = 0; nt < 4; nt++)
            {
                uint32_t b0, b1;
                load_b_kt_ldm(b0, b1, KV_s, FA5_KV_STRIDE,
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

            store_d_to_smem(S_s, FA5_S_STRIDE, m_row_base, col_base,
                            s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3],
                            lane_id);
        }

        __syncthreads();

        // ==== Softmax + overlapped V load ====
        async_load_tile(KV_s, V_head, kv_start, seq_len, head_dim);
        cp_async_commit();

        float m_old_r0 = m_smem[my_row0];
        float m_old_r8 = m_smem[my_row8];

        {
            int srow_start = warp_m * 16 + warp_n * 8;

            for (int r = 0; r < 8; r++)
            {
                int srow = srow_start + r;

                float v0 = __half2float(S_s[srow * FA5_S_STRIDE + lane_id * 2]);
                float v1 = __half2float(S_s[srow * FA5_S_STRIDE + lane_id * 2 + 1]);

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

                S_s[srow * FA5_S_STRIDE + lane_id * 2] = __float2half(e0);
                S_s[srow * FA5_S_STRIDE + lane_id * 2 + 1] = __float2half(e1);
            }
        }

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

        // ==== O += P @ V (P from S_s via ldmatrix, V from KV_s via ldmatrix.trans) ====
        int o_n_base = warp_n * 64;
        int pv_k_steps = FA5_BC / 16; // 4

#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            int n_col = o_n_base + nt * 8;

#pragma unroll
            for (int ks = 0; ks < pv_k_steps; ks++)
            {
                uint32_t a0, a1, a2, a3;
                load_a_ldm(a0, a1, a2, a3, S_s, FA5_S_STRIDE,
                           m_row_base, ks * 16, lane_id);

                uint32_t b0, b1;
                load_b_v_ldm(b0, b1, KV_s, FA5_KV_STRIDE,
                             n_col, ks * 16, lane_id);

                mma_f16_f32(
                    o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3],
                    a0, a1, a2, a3, b0, b1,
                    o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3]);
            }
        }

        // ==== Prefetch K[next] ====
        __syncthreads();

        int next_tile = kv_tile + 1;
        int next_start = next_tile * FA5_BC;
        bool has_next = (next_tile < num_kv_tiles) &&
                        (!causal || next_start <= q_start + FA5_BR - 1);
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

static int g_fa5_smem_max = 0;

extern "C"
{

    int flash_attention_v5_forward(
        const void *Q, const void *K, const void *V, void *O,
        int total_heads, int seq_len, int head_dim,
        int causal, void *stream)
    {
        if (head_dim % 16 != 0 || head_dim > 128)
            return -1;

        int smem_bytes = FA5_BR * FA5_Q_STRIDE * (int)sizeof(__half) + FA5_BC * FA5_KV_STRIDE * (int)sizeof(__half) + FA5_BR * FA5_S_STRIDE * (int)sizeof(__half) + FA5_BR * 2 * (int)sizeof(float);

        if (smem_bytes > g_fa5_smem_max)
        {
            cudaError_t err = cudaFuncSetAttribute(
                flash_attention_v5_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
            if (err != cudaSuccess)
                return (int)err;
            g_fa5_smem_max = smem_bytes;
        }

        float scale_val = 1.0f / sqrtf((float)head_dim);
        int num_q_tiles = (seq_len + FA5_BR - 1) / FA5_BR;
        int total_blocks = total_heads * num_q_tiles;

        flash_attention_v5_kernel<<<total_blocks, FA5_THREADS, smem_bytes, (cudaStream_t)stream>>>(
            (const __half *)Q, (const __half *)K, (const __half *)V, (__half *)O,
            seq_len, head_dim, causal, scale_val);

        return (int)cudaGetLastError();
    }

    int flash_attention_v5_forward_bhsd(
        const void *Q, const void *K, const void *V, void *O,
        int batch, int num_heads, int seq_len, int head_dim,
        int causal, void *stream)
    {
        return flash_attention_v5_forward(Q, K, V, O,
                                          batch * num_heads, seq_len, head_dim, causal, stream);
    }

} // extern "C"
