// =============================================================================
// FlashAttention — Tiled Fused Attention for SM89
// =============================================================================
// O(N) memory instead of O(N²), online softmax, causal mask support.
//
// Algorithm (FlashAttention-2 style):
//   For each Q tile (Br rows):
//     Load Q into SMEM (once)
//     For each KV tile (Bc rows):
//       1. Load K → compute S = Q @ K^T / √d
//       2. Causal mask
//       3. Online softmax: track running max (m) and sum (l)
//       4. Load V → accumulate O += P @ V with rescaling
//     Final: O /= l
//
// Tiles: Br=64, Bc=64, d=128 (standard LLaMA/Qwen head dim)
// SMEM: 48KB (Q 16K + KV 16K + S 16K) → 2 blocks/SM
// Threads: 256 (8 warps × 32 lanes)
// Each warp handles 8 Q rows, each lane handles d/32 = 4 output dims
//
// Performance target: 20-40 TFLOPS (10-15× over naive 2.58 TFLOPS)
// Future: upgrade to FP16 MMA for 50-80 TFLOPS
//
// Build: nvcc -O3 -arch=sm_89 -std=c++17 --shared -Xcompiler -fPIC \
//        flash_attention.cu -o libflashattn.so
// =============================================================================

#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define FA_BR 64       // Q tile rows
#define FA_BC 64       // KV tile rows
#define FA_THREADS 256 // threads per block
#define FA_WARPS 8     // warps per block
#define FA_RPW 8       // rows per warp (FA_BR / FA_WARPS)

// Max dims per lane: head_dim / 32. Supports d=64,96,128
#define FA_MAX_DPL 4 // 128/32 = 4

// SMEM sizes (computed for d=128)
// Q:  FA_BR * d * 2          = 16384
// KV: FA_BC * d * 2          = 16384
// S:  FA_BR * FA_BC * 4      = 16384
// Total:                       49152 = 48KB

// =============================================================================
// Warp reductions
// =============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

// =============================================================================
// FlashAttention Kernel
// =============================================================================

__global__ void __launch_bounds__(FA_THREADS, 2)
    flash_attention_kernel(
        const __half *__restrict__ Q,
        const __half *__restrict__ K,
        const __half *__restrict__ V,
        __half *__restrict__ O,
        int seq_len, int head_dim, int causal, float scale)
{
    // ---- Decode block ID → (batch*head, q_tile) ----
    int num_q_tiles = (seq_len + FA_BR - 1) / FA_BR;
    int bh = blockIdx.x / num_q_tiles; // batch*head index
    int q_tile = blockIdx.x % num_q_tiles;
    int q_start = q_tile * FA_BR;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int dpl = head_dim / 32; // dims per lane

    // ---- SMEM layout ----
    extern __shared__ char smem_raw[];
    __half *Q_s = (__half *)smem_raw;                                               // [FA_BR][head_dim]
    __half *KV_s = (__half *)(smem_raw + FA_BR * head_dim * sizeof(__half));        // [FA_BC][head_dim]
    float *S_s = (float *)(smem_raw + (FA_BR + FA_BC) * head_dim * sizeof(__half)); // [FA_BR][FA_BC]

    // ---- Head data pointers ----
    int head_stride = seq_len * head_dim;
    const __half *Q_head = Q + bh * head_stride;
    const __half *K_head = K + bh * head_stride;
    const __half *V_head = V + bh * head_stride;
    __half *O_head = O + bh * head_stride;

    // ---- Load Q tile into SMEM ----
    int q_load_total = FA_BR * head_dim;
    for (int i = threadIdx.x; i < q_load_total; i += FA_THREADS)
    {
        int row = i / head_dim;
        int col = i % head_dim;
        int grow = q_start + row;
        Q_s[i] = (grow < seq_len) ? Q_head[grow * head_dim + col] : __float2half(0.0f);
    }

    // ---- Per-thread accumulators (in registers) ----
    float o_acc[FA_RPW][FA_MAX_DPL]; // output accumulator
    float m_row[FA_RPW];             // running max per row
    float l_row[FA_RPW];             // running exp sum per row

#pragma unroll
    for (int r = 0; r < FA_RPW; r++)
    {
#pragma unroll
        for (int dd = 0; dd < FA_MAX_DPL; dd++)
            o_acc[r][dd] = 0.0f;
        m_row[r] = -1e30f;
        l_row[r] = 0.0f;
    }

    __syncthreads();

    // ---- Main loop over KV tiles ----
    int num_kv_tiles = (seq_len + FA_BC - 1) / FA_BC;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++)
    {
        int kv_start = kv_tile * FA_BC;

        // Early exit for causal: skip KV tiles entirely past Q range
        if (causal && kv_start > q_start + FA_BR - 1)
            break;

        // ---- Phase 1: Load K tile ----
        int kv_load_total = FA_BC * head_dim;
        for (int i = threadIdx.x; i < kv_load_total; i += FA_THREADS)
        {
            int row = i / head_dim;
            int col = i % head_dim;
            int grow = kv_start + row;
            KV_s[i] = (grow < seq_len) ? K_head[grow * head_dim + col] : __float2half(0.0f);
        }
        __syncthreads();

        // ---- Phase 2: Compute S = Q @ K^T × scale ----
        // Each warp handles FA_RPW Q rows
        // Each lane computes partial dot product (dpl elements of d)
        // Warp reduce → full dot product
        // Lane 0 writes S to SMEM

        for (int r = 0; r < FA_RPW; r++)
        {
            int qi = warp_id * FA_RPW + r; // local Q row index

            for (int j = 0; j < FA_BC; j++)
            {
                // Dot product Q[qi, :] · K[j, :] over head_dim
                float partial = 0.0f;
#pragma unroll
                for (int dd = 0; dd < dpl; dd++)
                {
                    int d_idx = lane_id * dpl + dd;
                    partial += __half2float(Q_s[qi * head_dim + d_idx]) * __half2float(KV_s[j * head_dim + d_idx]);
                }
                partial = warp_reduce_sum(partial);

                if (lane_id == 0)
                {
                    float s_val = partial * scale;
                    // Causal mask
                    int global_qi = q_start + qi;
                    int global_kj = kv_start + j;
                    if (causal && global_kj > global_qi)
                        s_val = -1e30f;
                    if (global_qi >= seq_len)
                        s_val = -1e30f;
                    S_s[qi * FA_BC + j] = s_val;
                }
            }
        }
        __syncthreads(); // S_s ready for all warps

        // ---- Phase 3: Online softmax ----
        // Each warp processes its own rows of S
        for (int r = 0; r < FA_RPW; r++)
        {
            int qi = warp_id * FA_RPW + r;

            // Row max
            float local_max = -1e30f;
            for (int j = lane_id; j < FA_BC; j += 32)
                local_max = fmaxf(local_max, S_s[qi * FA_BC + j]);
            float row_max = warp_reduce_max(local_max);

            float m_new = fmaxf(m_row[r], row_max);

            // Exp and sum
            float local_sum = 0.0f;
            for (int j = lane_id; j < FA_BC; j += 32)
            {
                float p = expf(S_s[qi * FA_BC + j] - m_new);
                S_s[qi * FA_BC + j] = p; // P values in-place
                local_sum += p;
            }
            __syncwarp();
            float row_sum = warp_reduce_sum(local_sum);

            // Rescale previous accumulator for new max
            float rescale = expf(m_row[r] - m_new);
#pragma unroll
            for (int dd = 0; dd < dpl; dd++)
                o_acc[r][dd] *= rescale;

            l_row[r] = l_row[r] * rescale + row_sum;
            m_row[r] = m_new;
        }
        __syncthreads(); // All warps done writing P to S_s

        // ---- Phase 4: Load V tile (reuse KV_s, overwrite K) ----
        for (int i = threadIdx.x; i < kv_load_total; i += FA_THREADS)
        {
            int row = i / head_dim;
            int col = i % head_dim;
            int grow = kv_start + row;
            KV_s[i] = (grow < seq_len) ? V_head[grow * head_dim + col] : __float2half(0.0f);
        }
        __syncthreads();

        // ---- Phase 5: O += P @ V ----
        // Each lane accumulates its dpl output dimensions
        for (int r = 0; r < FA_RPW; r++)
        {
            int qi = warp_id * FA_RPW + r;
#pragma unroll
            for (int dd = 0; dd < dpl; dd++)
            {
                int d_idx = lane_id * dpl + dd;
                float acc = 0.0f;
                for (int j = 0; j < FA_BC; j++)
                {
                    acc += S_s[qi * FA_BC + j] * __half2float(KV_s[j * head_dim + d_idx]);
                }
                o_acc[r][dd] += acc;
            }
        }
        __syncthreads();
    }

    // ---- Final: O = O / l ----
    for (int r = 0; r < FA_RPW; r++)
    {
        int qi = warp_id * FA_RPW + r;
        int global_qi = q_start + qi;
        if (global_qi >= seq_len)
            continue;

        float l_inv = (l_row[r] > 0.0f) ? (1.0f / l_row[r]) : 0.0f;
        for (int dd = 0; dd < dpl; dd++)
        {
            int d_idx = lane_id * dpl + dd;
            if (d_idx < head_dim)
            {
                O_head[global_qi * head_dim + d_idx] =
                    __float2half(o_acc[r][dd] * l_inv);
            }
        }
    }
}

// =============================================================================
// C API — purego compatible
// =============================================================================

static bool g_flash_smem_configured = false;

extern "C"
{

    // flash_attention_forward computes scaled dot-product attention with tiling.
    //
    // Q, K, V: [batch * num_heads, seq_len, head_dim] FP16 device memory
    // O:       [batch * num_heads, seq_len, head_dim] FP16 device memory
    //
    // total_heads = batch * num_heads (pre-merged by caller)
    // head_dim must be multiple of 32 (64, 96, 128)
    // causal: 1 for autoregressive mask, 0 for full attention
    //
    // Returns: 0 on success, CUDA error code on failure
    int flash_attention_forward(
        const void *Q, const void *K, const void *V, void *O,
        int total_heads, int seq_len, int head_dim,
        int causal, void *stream)
    {
        if (head_dim % 32 != 0 || head_dim > 128)
            return -1; // unsupported head_dim

        int smem_bytes = (FA_BR + FA_BC) * head_dim * sizeof(__half) + FA_BR * FA_BC * sizeof(float);

        if (!g_flash_smem_configured)
        {
            cudaError_t err = cudaFuncSetAttribute(
                flash_attention_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_bytes);
            if (err != cudaSuccess)
                return (int)err;
            g_flash_smem_configured = true;
        }

        float scale = 1.0f / sqrtf((float)head_dim);
        int num_q_tiles = (seq_len + FA_BR - 1) / FA_BR;
        int total_blocks = total_heads * num_q_tiles;

        flash_attention_kernel<<<total_blocks, FA_THREADS, smem_bytes, (cudaStream_t)stream>>>(
            (const __half *)Q, (const __half *)K, (const __half *)V, (__half *)O,
            seq_len, head_dim, causal, scale);

        return (int)cudaGetLastError();
    }

    // Wrapper with separate batch/heads for convenience
    int flash_attention_forward_bhsd(
        const void *Q, const void *K, const void *V, void *O,
        int batch, int num_heads, int seq_len, int head_dim,
        int causal, void *stream)
    {
        return flash_attention_forward(Q, K, V, O,
                                       batch * num_heads, seq_len, head_dim, causal, stream);
    }

} // extern "C"
