// =============================================================================
// FlashAttention v7 — stride=128 + XOR Swizzle + ldmatrix + Optimized Softmax
// =============================================================================
// Changes vs v6 (107 TFLOPS causal-corrected):
//   Reduce stride from 136 → 128 (no padding waste)
//   XOR chunk swizzle to avoid bank conflicts with stride=128
//
//   Why stride=128 failed before: with pack_h2, needed 4 swizzle ops per
//   fragment → ALU overhead > bank conflict savings.
//   Why it works now: ldmatrix takes 1 address per call → 1 swizzle op.
//
//   Swizzle: physical_chunk = logical_chunk ^ (row & 7)
//   Applied in: cp.async store (load phase), ldmatrix address (read phase)
//
// SMEM savings vs stride=136:
//   Q_s:  64 × 128 × 2 = 16,384 (was 17,408, -1024)
//   KV_s: 64 × 128 × 2 = 16,384 (was 17,408, -1024)
//   S_s:  64 × 72  × 2 =  9,216 (unchanged, not swizzled)
//   m+l:  64 × 2   × 4 =    512
//   Total: 42,496 B (was 44,544) → still 2 blocks/SM
// =============================================================================

#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define FA7_BR 64
#define FA7_BC 64
#define FA7_THREADS 256

#define FA7_Q_STRIDE 128  // was 136 — no padding!
#define FA7_KV_STRIDE 128 // was 136
#define FA7_S_STRIDE 72   // unchanged

// =============================================================================
// XOR chunk swizzle
// =============================================================================

__device__ __forceinline__ int swizzle_col(int row, int col)
{
    // Permute 16B chunks (8 halves) based on row index
    int chunk = col >> 3;
    int within = col & 7;
    return ((chunk ^ (row & 7)) << 3) | within;
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
// SWIZZLED fragment loaders via ldmatrix
// =============================================================================

// A from swizzled Q_s or S_s-as-P (but S_s is NOT swizzled, see load_a_ldm_plain)
__device__ __forceinline__ void load_a_ldm_sw(
    uint32_t &a0, uint32_t &a1, uint32_t &a2, uint32_t &a3,
    const __half *smem, int stride, int row_base, int k_base, int lane)
{
    int sub = lane / 8;
    int sub_row = lane % 8;
    int row = row_base + (sub & 1) * 8 + sub_row;
    int logical_col = k_base + (sub >> 1) * 8;
    int physical_col = swizzle_col(row, logical_col);
    ldmatrix_x4(a0, a1, a2, a3, &smem[row * stride + physical_col]);
}

// B from swizzled KV_s (K stored row-major K[n][k])
__device__ __forceinline__ void load_b_kt_ldm_sw(
    uint32_t &b0, uint32_t &b1,
    const __half *smem, int stride, int n_base, int k_base, int lane)
{
    int sub = (lane / 8) % 2;
    int sub_row = lane % 8;
    int row = n_base + sub_row;
    int logical_col = k_base + sub * 8;
    int physical_col = swizzle_col(row, logical_col);
    ldmatrix_x2(b0, b1, &smem[row * stride + physical_col]);
}

// B from swizzled KV_s with transpose (V stored row-major V[k][d])
__device__ __forceinline__ void load_b_v_ldm_sw(
    uint32_t &b0, uint32_t &b1,
    const __half *smem, int stride, int n_base, int k_base, int lane)
{
    int sub = (lane / 8) % 2;
    int sub_row = lane % 8;
    int k = k_base + sub * 8 + sub_row;
    int physical_n = swizzle_col(k, n_base); // row=k, col=n
    ldmatrix_x2_trans(b0, b1, &smem[k * stride + physical_n]);
}

// A from UN-swizzled S_s (for P @ V)
__device__ __forceinline__ void load_a_ldm_plain(
    uint32_t &a0, uint32_t &a1, uint32_t &a2, uint32_t &a3,
    const __half *smem, int stride, int row_base, int k_base, int lane)
{
    int sub = lane / 8;
    int sub_row = lane % 8;
    int row = row_base + (sub & 1) * 8 + sub_row;
    int col = k_base + (sub >> 1) * 8;
    ldmatrix_x4(a0, a1, a2, a3, &smem[row * stride + col]);
}

// Store D fragments to S_s (NOT swizzled)
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
// Swizzled async tile load
// =============================================================================

__device__ __forceinline__ void async_load_tile_sw(
    __half *smem, const __half *src_head,
    int tile_start, int seq_len, int head_dim)
{
    constexpr int CHUNKS_PER_ROW = 16; // 128/8
    constexpr int TOTAL_CHUNKS = FA7_BC * CHUNKS_PER_ROW;

#pragma unroll 4
    for (int c = threadIdx.x; c < TOTAL_CHUNKS; c += FA7_THREADS)
    {
        int row = c / CHUNKS_PER_ROW;
        int logical_chunk = c % CHUNKS_PER_ROW;
        int grow = tile_start + row;

        // Swizzle destination chunk
        int physical_chunk = logical_chunk ^ (row & 7);
        __half *dst = &smem[row * FA7_KV_STRIDE + physical_chunk * 8];
        const __half *src = &src_head[grow * head_dim + logical_chunk * 8];
        cp_async_cg_16(dst, src, (grow < seq_len) ? 16 : 0);
    }
}

// =============================================================================
// FlashAttention v7 Kernel
// =============================================================================

__global__ void __launch_bounds__(FA7_THREADS, 2)
    flash_attention_v7_kernel(
        const __half *__restrict__ Q,
        const __half *__restrict__ K,
        const __half *__restrict__ V,
        __half *__restrict__ O,
        int seq_len, int head_dim, int causal, float scale)
{
    int num_q_tiles = (seq_len + FA7_BR - 1) / FA7_BR;
    int bh = blockIdx.x / num_q_tiles;
    int q_tile = blockIdx.x % num_q_tiles;
    int q_start = q_tile * FA7_BR;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int warp_m = warp_id % 4;
    int warp_n = warp_id / 4;
    int gid = lane_id >> 2;
    int tid = lane_id & 3;

    extern __shared__ char smem_raw[];
    __half *Q_s = (__half *)smem_raw;
    __half *KV_s = (__half *)(smem_raw + FA7_BR * FA7_Q_STRIDE * sizeof(__half));
    __half *S_s = (__half *)(smem_raw + FA7_BR * FA7_Q_STRIDE * sizeof(__half) + FA7_BC * FA7_KV_STRIDE * sizeof(__half));
    float *m_smem = (float *)(smem_raw + FA7_BR * FA7_Q_STRIDE * sizeof(__half) + FA7_BC * FA7_KV_STRIDE * sizeof(__half) + FA7_BR * FA7_S_STRIDE * sizeof(__half));
    float *l_smem = m_smem + FA7_BR;

    int hstride = seq_len * head_dim;
    const __half *Q_head = Q + bh * hstride;
    const __half *K_head = K + bh * hstride;
    const __half *V_head = V + bh * hstride;
    __half *O_head = O + bh * hstride;

    // Load Q via swizzled cp.async
    {
        constexpr int CHUNKS_PER_ROW = 16;
        constexpr int TOTAL = FA7_BR * CHUNKS_PER_ROW;
#pragma unroll 4
        for (int c = threadIdx.x; c < TOTAL; c += FA7_THREADS)
        {
            int row = c / CHUNKS_PER_ROW;
            int logical_chunk = c % CHUNKS_PER_ROW;
            int grow = q_start + row;
            int physical_chunk = logical_chunk ^ (row & 7);
            cp_async_cg_16(&Q_s[row * FA7_Q_STRIDE + physical_chunk * 8],
                           &Q_head[grow * head_dim + logical_chunk * 8],
                           (grow < seq_len) ? 16 : 0);
        }
        cp_async_commit();
    }

    // Init m/l
    for (int i = threadIdx.x; i < FA7_BR; i += FA7_THREADS)
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
    int srow_base = warp_m * 16 + warp_n * 8;

    float m_reg[8], l_reg[8];
#pragma unroll
    for (int r = 0; r < 8; r++)
    {
        m_reg[r] = -1e30f;
        l_reg[r] = 0.0f;
    }

    int num_kv_tiles = (seq_len + FA7_BC - 1) / FA7_BC;
    int m_row_base = warp_m * 16;
    int n_col_base = warp_n * 32;
    int k_steps = head_dim / 16;

    // Prologue: K[0] swizzled
    async_load_tile_sw(KV_s, K_head, 0, seq_len, head_dim);
    cp_async_commit();

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++)
    {
        int kv_start = kv_tile * FA7_BC;
        if (causal && kv_start > q_start + FA7_BR - 1)
            break;

        cp_async_wait<0>();
        __syncthreads();

        // ---- S = Q @ K^T (both swizzled) ----
        float s_acc[4][4];
#pragma unroll
        for (int t = 0; t < 4; t++)
#pragma unroll
            for (int r = 0; r < 4; r++)
                s_acc[t][r] = 0.0f;

        for (int ks = 0; ks < k_steps; ks++)
        {
            uint32_t a0, a1, a2, a3;
            load_a_ldm_sw(a0, a1, a2, a3, Q_s, FA7_Q_STRIDE,
                          m_row_base, ks * 16, lane_id);

#pragma unroll
            for (int nt = 0; nt < 4; nt++)
            {
                uint32_t b0, b1;
                load_b_kt_ldm_sw(b0, b1, KV_s, FA7_KV_STRIDE,
                                 n_col_base + nt * 8, ks * 16, lane_id);
                mma_f16_f32(
                    s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3],
                    a0, a1, a2, a3, b0, b1,
                    s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3]);
            }
        }

// ---- Scale + causal mask → S_s (NOT swizzled) ----
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

            store_d_to_smem(S_s, FA7_S_STRIDE, m_row_base, col_base,
                            s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3],
                            lane_id);
        }

        __syncthreads();

        // ---- V load (swizzled) overlapped with softmax ----
        async_load_tile_sw(KV_s, V_head, kv_start, seq_len, head_dim);
        cp_async_commit();

        float m_old_r0 = m_smem[my_row0];
        float m_old_r8 = m_smem[my_row8];

// ---- Optimized softmax (same as v6) ----
#pragma unroll
        for (int rr = 0; rr < 8; rr += 2)
        {
            int srow_a = srow_base + rr;
            int srow_b = srow_a + 1;

            float va0 = __half2float(S_s[srow_a * FA7_S_STRIDE + lane_id * 2]);
            float vb0 = __half2float(S_s[srow_b * FA7_S_STRIDE + lane_id * 2]);
            float va1 = __half2float(S_s[srow_a * FA7_S_STRIDE + lane_id * 2 + 1]);
            float vb1 = __half2float(S_s[srow_b * FA7_S_STRIDE + lane_id * 2 + 1]);

            float lmax_a = fmaxf(va0, va1);
            float lmax_b = fmaxf(vb0, vb1);
            float rmax_a = warp_reduce_max(lmax_a);
            float rmax_b = warp_reduce_max(lmax_b);

            float m_old_a = m_reg[rr];
            float m_old_b = m_reg[rr + 1];
            float m_new_a = fmaxf(m_old_a, rmax_a);
            float m_new_b = fmaxf(m_old_b, rmax_b);

            float ea0 = __expf(va0 - m_new_a);
            float eb0 = __expf(vb0 - m_new_b);
            float ea1 = __expf(va1 - m_new_a);
            float eb1 = __expf(vb1 - m_new_b);

            float rsum_a = warp_reduce_sum(ea0 + ea1);
            float rsum_b = warp_reduce_sum(eb0 + eb1);

            float rsc_a = __expf(m_old_a - m_new_a);
            float rsc_b = __expf(m_old_b - m_new_b);

            m_reg[rr] = m_new_a;
            m_reg[rr + 1] = m_new_b;
            if (lane_id == 0)
            {
                l_reg[rr] = l_reg[rr] * rsc_a + rsum_a;
                l_reg[rr + 1] = l_reg[rr + 1] * rsc_b + rsum_b;
            }

            S_s[srow_a * FA7_S_STRIDE + lane_id * 2] = __float2half(ea0);
            S_s[srow_b * FA7_S_STRIDE + lane_id * 2] = __float2half(eb0);
            S_s[srow_a * FA7_S_STRIDE + lane_id * 2 + 1] = __float2half(ea1);
            S_s[srow_b * FA7_S_STRIDE + lane_id * 2 + 1] = __float2half(eb1);
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

        // ---- Rescale O ----
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

        // ---- O += P @ V (S_s plain, KV_s swizzled) ----
        int o_n_base = warp_n * 64;

#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            int n_col = o_n_base + nt * 8;

#pragma unroll
            for (int ks = 0; ks < 4; ks++)
            {
                uint32_t a0, a1, a2, a3;
                load_a_ldm_plain(a0, a1, a2, a3, S_s, FA7_S_STRIDE,
                                 m_row_base, ks * 16, lane_id);

                uint32_t b0, b1;
                load_b_v_ldm_sw(b0, b1, KV_s, FA7_KV_STRIDE,
                                n_col, ks * 16, lane_id);

                mma_f16_f32(
                    o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3],
                    a0, a1, a2, a3, b0, b1,
                    o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3]);
            }
        }

        // ---- Prefetch K[next] (swizzled) ----
        __syncthreads();

        int next_tile = kv_tile + 1;
        int next_start = next_tile * FA7_BC;
        bool has_next = (next_tile < num_kv_tiles) &&
                        (!causal || next_start <= q_start + FA7_BR - 1);
        if (has_next)
        {
            async_load_tile_sw(KV_s, K_head, next_start, seq_len, head_dim);
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

static int g_fa7_smem_max = 0;

extern "C"
{

    int flash_attention_v7_forward(
        const void *Q, const void *K, const void *V, void *O,
        int total_heads, int seq_len, int head_dim,
        int causal, void *stream)
    {
        if (head_dim % 16 != 0 || head_dim > 128)
            return -1;

        int smem_bytes = FA7_BR * FA7_Q_STRIDE * (int)sizeof(__half) + FA7_BC * FA7_KV_STRIDE * (int)sizeof(__half) + FA7_BR * FA7_S_STRIDE * (int)sizeof(__half) + FA7_BR * 2 * (int)sizeof(float);

        if (smem_bytes > g_fa7_smem_max)
        {
            cudaError_t err = cudaFuncSetAttribute(
                flash_attention_v7_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
            if (err != cudaSuccess)
                return (int)err;
            g_fa7_smem_max = smem_bytes;
        }

        float scale_val = 1.0f / sqrtf((float)head_dim);
        int num_q_tiles = (seq_len + FA7_BR - 1) / FA7_BR;
        int total_blocks = total_heads * num_q_tiles;

        flash_attention_v7_kernel<<<total_blocks, FA7_THREADS, smem_bytes, (cudaStream_t)stream>>>(
            (const __half *)Q, (const __half *)K, (const __half *)V, (__half *)O,
            seq_len, head_dim, causal, scale_val);

        return (int)cudaGetLastError();
    }

    int flash_attention_v7_forward_bhsd(
        const void *Q, const void *K, const void *V, void *O,
        int batch, int num_heads, int seq_len, int head_dim,
        int causal, void *stream)
    {
        return flash_attention_v7_forward(Q, K, V, O,
                                          batch * num_heads, seq_len, head_dim, causal, stream);
    }

} // extern "C"
