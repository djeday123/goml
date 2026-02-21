// =============================================================================
// Transformer Kernels — FP16 building blocks for GoML v3
// =============================================================================
// RMSNorm, SwiGLU, RoPE, Scaled Dot-Product Attention (causal)
//
// All operate on FP16 data (compatible with FP8 GEMM output)
// Designed for LLaMA/Qwen-style architectures
//
// Build shared library:
//   nvcc -O3 -arch=sm_89 -std=c++17 --shared -Xcompiler -fPIC \
//        transformer_kernels.cu -o libtransformer.so
// =============================================================================

#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// =============================================================================
// Helper: warp reduction
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

// Block reduction using shared memory + warp shuffle
// Assumes blockDim.x threads, returns result in thread 0
__device__ __forceinline__ float block_reduce_sum(float val, float *smem)
{
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    val = warp_reduce_sum(val);
    if (lane == 0)
        smem[warp] = val;
    __syncthreads();
    int nwarps = blockDim.x / 32;
    val = (threadIdx.x < nwarps) ? smem[threadIdx.x] : 0.0f;
    if (warp == 0)
        val = warp_reduce_sum(val);
    return val; // valid in thread 0
}

__device__ __forceinline__ float block_reduce_max(float val, float *smem)
{
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    val = warp_reduce_max(val);
    if (lane == 0)
        smem[warp] = val;
    __syncthreads();
    int nwarps = blockDim.x / 32;
    val = (threadIdx.x < nwarps) ? smem[threadIdx.x] : -1e30f;
    if (warp == 0)
        val = warp_reduce_max(val);
    return val;
}

// =============================================================================
// 1. RMSNorm
// =============================================================================
// y[i] = x[i] * weight[i] / sqrt(mean(x²) + eps)
//
// Input:  x      [rows, hidden]  FP16
//         weight [hidden]        FP16
// Output: y      [rows, hidden]  FP16
// eps: typically 1e-6
//
// One block per row.  Block size = min(hidden, 1024).
// Threads stride over hidden dim for large hidden sizes.

__global__ void rmsnorm_kernel(
    const __half *__restrict__ x,
    const __half *__restrict__ weight,
    __half *__restrict__ y,
    int rows, int hidden, float eps)
{
    extern __shared__ float smem[];

    int row = blockIdx.x;
    if (row >= rows)
        return;

    const __half *x_row = x + row * hidden;
    __half *y_row = y + row * hidden;

    // Pass 1: compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
    {
        float v = __half2float(x_row[i]);
        sum_sq += v * v;
    }

    sum_sq = block_reduce_sum(sum_sq, smem);
    __shared__ float s_rms_inv;
    if (threadIdx.x == 0)
        s_rms_inv = rsqrtf(sum_sq / (float)hidden + eps);
    __syncthreads();

    float rms_inv = s_rms_inv;

    // Pass 2: normalize and scale
    // Use half2 for 2x throughput when hidden is even
    if (hidden % 2 == 0)
    {
        const __half2 *x2 = (const __half2 *)x_row;
        const __half2 *w2 = (const __half2 *)weight;
        __half2 *y2 = (__half2 *)y_row;
        int half_hidden = hidden / 2;
        for (int i = threadIdx.x; i < half_hidden; i += blockDim.x)
        {
            __half2 xv = x2[i];
            __half2 wv = w2[i];
            float2 xf = __half22float2(xv);
            float2 wf = __half22float2(wv);
            xf.x = xf.x * rms_inv * wf.x;
            xf.y = xf.y * rms_inv * wf.y;
            y2[i] = __float22half2_rn(xf);
        }
    }
    else
    {
        for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        {
            float v = __half2float(x_row[i]);
            float w = __half2float(weight[i]);
            y_row[i] = __float2half(v * rms_inv * w);
        }
    }
}

// =============================================================================
// 2. SwiGLU activation
// =============================================================================
// y = SiLU(gate) * up
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// Input:  gate [n]  FP16
//         up   [n]  FP16
// Output: y    [n]  FP16
//
// Fused: avoids 3 separate kernels (sigmoid, mul, mul)
// Used in LLaMA FFN: y = down_proj(SwiGLU(gate_proj(x), up_proj(x)))

__global__ void swiglu_kernel(
    const __half *__restrict__ gate,
    const __half *__restrict__ up,
    __half *__restrict__ y,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process pairs with half2
    int n2 = n / 2;
    const __half2 *gate2 = (const __half2 *)gate;
    const __half2 *up2 = (const __half2 *)up;
    __half2 *y2 = (__half2 *)y;

    for (int i = idx; i < n2; i += stride)
    {
        __half2 g = gate2[i];
        __half2 u = up2[i];
        float2 gf = __half22float2(g);
        float2 uf = __half22float2(u);

        // SiLU(x) = x / (1 + exp(-x))
        gf.x = gf.x / (1.0f + expf(-gf.x));
        gf.y = gf.y / (1.0f + expf(-gf.y));

        // SwiGLU = SiLU(gate) * up
        float2 out;
        out.x = gf.x * uf.x;
        out.y = gf.y * uf.y;

        y2[i] = __float22half2_rn(out);
    }

    // Handle odd element
    if (n % 2 != 0 && idx == 0)
    {
        float g = __half2float(gate[n - 1]);
        float u = __half2float(up[n - 1]);
        g = g / (1.0f + expf(-g));
        y[n - 1] = __float2half(g * u);
    }
}

// =============================================================================
// 3. RoPE (Rotary Positional Embeddings)
// =============================================================================
// For each pair (x[2i], x[2i+1]) at sequence position pos:
//   theta = pos * base^(-2i/head_dim)
//   y[2i]   = x[2i]*cos(theta) - x[2i+1]*sin(theta)
//   y[2i+1] = x[2i]*sin(theta) + x[2i+1]*cos(theta)
//
// Input:  x   [batch, seq_len, num_heads, head_dim]  FP16
//         pos [seq_len]                               int32 (position indices)
// Output: y   [batch, seq_len, num_heads, head_dim]  FP16
// base: typically 10000.0 (or 500000.0 for extended context)
//
// In-place supported: y can alias x.
// Each thread handles one (cos,sin) rotation pair.

__global__ void rope_kernel(
    const __half *__restrict__ x,
    const int *__restrict__ pos,
    __half *__restrict__ y,
    int batch, int seq_len, int num_heads, int head_dim,
    float theta_base)
{
    // Total pairs: batch * seq_len * num_heads * (head_dim/2)
    int pairs_per_head = head_dim / 2;
    int pairs_per_seq = num_heads * pairs_per_head;
    int pairs_per_batch = seq_len * pairs_per_seq;
    int total_pairs = batch * pairs_per_batch;

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_pairs)
        return;

    // Decode indices
    int b = gid / pairs_per_batch;
    int rem = gid % pairs_per_batch;
    int s = rem / pairs_per_seq;
    int rem2 = rem % pairs_per_seq;
    int h = rem2 / pairs_per_head;
    int i = rem2 % pairs_per_head; // pair index within head

    // Compute theta
    float freq = 1.0f / powf(theta_base, (float)(2 * i) / (float)head_dim);
    float theta = (float)pos[s] * freq;
    float cos_t, sin_t;
    sincosf(theta, &sin_t, &cos_t);

    // Load pair
    int offset = ((b * seq_len + s) * num_heads + h) * head_dim + 2 * i;
    float x0 = __half2float(x[offset]);
    float x1 = __half2float(x[offset + 1]);

    // Rotate
    y[offset] = __float2half(x0 * cos_t - x1 * sin_t);
    y[offset + 1] = __float2half(x0 * sin_t + x1 * cos_t);
}

// =============================================================================
// 4. Scaled Dot-Product Attention (with causal mask)
// =============================================================================
// scores = Q @ K^T / sqrt(head_dim)
// if causal: scores[i][j] = -inf where j > i
// attn = softmax(scores, dim=-1)
// output = attn @ V
//
// Input:  Q [batch, num_heads, seq_len, head_dim]  FP16
//         K [batch, num_heads, seq_len, head_dim]  FP16
//         V [batch, num_heads, seq_len, head_dim]  FP16
// Output: O [batch, num_heads, seq_len, head_dim]  FP16
//
// This is a BASIC fused implementation (not FlashAttention).
// Suitable for seq_len <= 2048. For longer sequences, FlashAttention needed.
//
// Layout: one block per (batch, head, query_row)
// Each block computes one row of the output.
// Block threads stride over key positions and head_dim.

__global__ void attention_kernel(
    const __half *__restrict__ Q,
    const __half *__restrict__ K,
    const __half *__restrict__ V,
    __half *__restrict__ O,
    int batch, int num_heads, int seq_len, int head_dim,
    int causal, float scale)
{
    extern __shared__ float smem[];
    // smem layout: [0..nwarps-1] for reduction, [nwarps..nwarps+seq_len-1] for scores

    int nwarps = blockDim.x / 32;
    float *reduce_smem = smem;
    float *scores = smem + nwarps; // [seq_len]

    // Decode block → (batch_idx, head_idx, query_pos)
    int bh = blockIdx.x / seq_len;
    int q_pos = blockIdx.x % seq_len;
    int b = bh / num_heads;
    int h = bh % num_heads;

    if (b >= batch)
        return;

    // Pointers to this head's Q, K, V, O
    int head_offset = ((b * num_heads + h) * seq_len) * head_dim;
    const __half *q_row = Q + head_offset + q_pos * head_dim;
    const __half *k_base = K + head_offset;
    const __half *v_base = V + head_offset;
    __half *o_row = O + head_offset + q_pos * head_dim;

    // Step 1: Compute scores = Q[q_pos] · K[j]^T * scale for all j
    int max_j = causal ? (q_pos + 1) : seq_len;

    for (int j = threadIdx.x; j < seq_len; j += blockDim.x)
    {
        if (j < max_j)
        {
            const __half *k_row = k_base + j * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++)
            {
                dot += __half2float(q_row[d]) * __half2float(k_row[d]);
            }
            scores[j] = dot * scale;
        }
        else
        {
            scores[j] = -1e30f; // causal mask
        }
    }
    __syncthreads();

    // Step 2: Online softmax — find max
    float local_max = -1e30f;
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x)
        local_max = fmaxf(local_max, scores[j]);
    float global_max = block_reduce_max(local_max, reduce_smem);
    __syncthreads();

    // Broadcast max (thread 0 has it from reduction)
    __shared__ float s_max;
    if (threadIdx.x == 0)
        s_max = global_max;
    __syncthreads();
    global_max = s_max;

    // Step 3: exp(score - max) and sum
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x)
    {
        float e = expf(scores[j] - global_max);
        scores[j] = e;
        local_sum += e;
    }
    float global_sum = block_reduce_sum(local_sum, reduce_smem);
    __syncthreads();

    __shared__ float s_sum_inv;
    if (threadIdx.x == 0)
        s_sum_inv = 1.0f / (global_sum + 1e-8f);
    __syncthreads();
    float sum_inv = s_sum_inv;

    // Step 4: Normalize scores in place
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x)
        scores[j] *= sum_inv;
    __syncthreads();

    // Step 5: Output = attn_weights @ V
    // Each thread computes a subset of output dimensions
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
    {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; j++)
        {
            acc += scores[j] * __half2float(v_base[j * head_dim + d]);
        }
        o_row[d] = __float2half(acc);
    }
}

// =============================================================================
// C API — purego compatible
// =============================================================================

extern "C"
{

    // --- RMSNorm ---
    // x, y: [rows, hidden] FP16 device memory
    // weight: [hidden] FP16 device memory
    int rmsnorm_forward(
        const void *x, const void *weight, void *y,
        int rows, int hidden, float eps, void *stream)
    {
        int threads = (hidden < 1024) ? hidden : 1024;
        // Round up to warp size
        threads = ((threads + 31) / 32) * 32;
        int smem = (threads / 32) * sizeof(float);

        rmsnorm_kernel<<<rows, threads, smem, (cudaStream_t)stream>>>(
            (const __half *)x, (const __half *)weight, (__half *)y,
            rows, hidden, eps);
        return (int)cudaGetLastError();
    }

    // --- SwiGLU ---
    // gate, up, y: [n] FP16 device memory
    int swiglu_forward(
        const void *gate, const void *up, void *y,
        int n, void *stream)
    {
        int threads = 256;
        int blocks = (n / 2 + threads - 1) / threads;
        if (blocks < 1)
            blocks = 1;

        swiglu_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
            (const __half *)gate, (const __half *)up, (__half *)y, n);
        return (int)cudaGetLastError();
    }

    // --- RoPE ---
    // x, y: [batch, seq_len, num_heads, head_dim] FP16 device memory
    // pos: [seq_len] int32 device memory (position indices)
    // y can alias x for in-place operation
    int rope_forward(
        const void *x, const void *pos, void *y,
        int batch, int seq_len, int num_heads, int head_dim,
        float theta_base, void *stream)
    {
        int total_pairs = batch * seq_len * num_heads * (head_dim / 2);
        int threads = 256;
        int blocks = (total_pairs + threads - 1) / threads;

        rope_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
            (const __half *)x, (const int *)pos, (__half *)y,
            batch, seq_len, num_heads, head_dim, theta_base);
        return (int)cudaGetLastError();
    }

    // --- Attention ---
    // Q, K, V, O: [batch, num_heads, seq_len, head_dim] FP16 device memory
    // causal: 1 for causal mask, 0 for full attention
    int attention_forward(
        const void *Q, const void *K, const void *V, void *O,
        int batch, int num_heads, int seq_len, int head_dim,
        int causal, void *stream)
    {
        float scale = 1.0f / sqrtf((float)head_dim);
        int total_rows = batch * num_heads * seq_len;

        int threads = 256;
        // Shared memory: nwarps floats for reduction + seq_len floats for scores
        int nwarps = threads / 32;
        int smem = (nwarps + seq_len) * sizeof(float);

        // Check shared memory limit (100KB on SM89)
        if (smem > 100 * 1024)
        {
            // seq_len too large for this kernel — need FlashAttention
            return -1;
        }

        attention_kernel<<<total_rows, threads, smem, (cudaStream_t)stream>>>(
            (const __half *)Q, (const __half *)K, (const __half *)V, (__half *)O,
            batch, num_heads, seq_len, head_dim, causal, scale);
        return (int)cudaGetLastError();
    }

} // extern "C"
