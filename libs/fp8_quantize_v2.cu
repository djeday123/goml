#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>

#define FP8_E4M3_MAX 448.0f
#define BLOCK 256

// ============================================================
// absmax reduction: FP16 input, vectorized half2 loads
// ============================================================
__global__ void absmax_fp16_kernel(const half *__restrict__ in, float *__restrict__ out, int n)
{
    extern __shared__ float sm[];
    int tid = threadIdx.x;
    int i = (blockIdx.x * blockDim.x * 2 + tid) * 2; // each thread handles 4 elements

    float v = 0.0f;
    if (i + 1 < n)
    {
        half2 h2 = *(const half2 *)&in[i];
        v = fmaxf(fabsf(__half2float(h2.x)), fabsf(__half2float(h2.y)));
    }
    else if (i < n)
    {
        v = fabsf(__half2float(in[i]));
    }

    int j = i + blockDim.x * 2;
    if (j + 1 < n)
    {
        half2 h2 = *(const half2 *)&in[j];
        v = fmaxf(v, fmaxf(fabsf(__half2float(h2.x)), fabsf(__half2float(h2.y))));
    }
    else if (j < n)
    {
        v = fmaxf(v, fabsf(__half2float(in[j])));
    }

    sm[tid] = v;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sm[tid] = fmaxf(sm[tid], sm[tid + s]);
        __syncthreads();
    }
    if (tid == 0)
        out[blockIdx.x] = sm[0];
}

// absmax reduction: float input (second pass)
__global__ void absmax_f32_kernel(const float *__restrict__ in, float *__restrict__ out, int n)
{
    extern __shared__ float sm[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;

    float v = 0.0f;
    if (i < n)
        v = fabsf(in[i]);
    if (i + blockDim.x < n)
        v = fmaxf(v, fabsf(in[i + blockDim.x]));
    sm[tid] = v;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sm[tid] = fmaxf(sm[tid], sm[tid + s]);
        __syncthreads();
    }
    if (tid == 0)
        out[blockIdx.x] = sm[0];
}

// ============================================================
// Quantize FP16 -> FP8 E4M3, vectorized half2 + uchar2
// ============================================================
__global__ void quant_kernel(
    const half *__restrict__ in,
    uint8_t *__restrict__ out,
    const float *__restrict__ absmax_ptr,
    float *__restrict__ scale_out,
    int n)
{
    __shared__ float inv_scale;
    if (threadIdx.x == 0)
    {
        float amax = *absmax_ptr;
        float scale = (amax > 1e-12f) ? (amax / FP8_E4M3_MAX) : 1.0f;
        if (blockIdx.x == 0)
            *scale_out = scale;
        inv_scale = 1.0f / scale;
    }
    __syncthreads();

    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n)
    {
        half2 h2 = *(const half2 *)&in[idx];
        float v0 = __half2float(h2.x) * inv_scale;
        float v1 = __half2float(h2.y) * inv_scale;
        v0 = fminf(fmaxf(v0, -FP8_E4M3_MAX), FP8_E4M3_MAX);
        v1 = fminf(fmaxf(v1, -FP8_E4M3_MAX), FP8_E4M3_MAX);
        __nv_fp8_e4m3 f0 = __nv_fp8_e4m3(v0);
        __nv_fp8_e4m3 f1 = __nv_fp8_e4m3(v1);
        uchar2 pair;
        pair.x = *(uint8_t *)&f0;
        pair.y = *(uint8_t *)&f1;
        *(uchar2 *)&out[idx] = pair;
    }
    else if (idx < n)
    {
        float v = __half2float(in[idx]) * inv_scale;
        v = fminf(fmaxf(v, -FP8_E4M3_MAX), FP8_E4M3_MAX);
        __nv_fp8_e4m3 f = __nv_fp8_e4m3(v);
        out[idx] = *(uint8_t *)&f;
    }
}

// ============================================================
// Dequantize FP8 -> FP16, vectorized uchar2 + half2
// ============================================================
__global__ void dequant_kernel(
    const uint8_t *__restrict__ in,
    half *__restrict__ out,
    const float *__restrict__ scale_ptr,
    int n)
{
    __shared__ float scale;
    if (threadIdx.x == 0)
        scale = *scale_ptr;
    __syncthreads();

    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n)
    {
        uchar2 pair = *(const uchar2 *)&in[idx];
        __nv_fp8_e4m3 f0 = *(__nv_fp8_e4m3 *)&pair.x;
        __nv_fp8_e4m3 f1 = *(__nv_fp8_e4m3 *)&pair.y;
        half2 h2;
        h2.x = __float2half(float(f0) * scale);
        h2.y = __float2half(float(f1) * scale);
        *(half2 *)&out[idx] = h2;
    }
    else if (idx < n)
    {
        __nv_fp8_e4m3 f = *(__nv_fp8_e4m3 *)&in[idx];
        out[idx] = __float2half(float(f) * scale);
    }
}

// ============================================================
// C API
// ============================================================
extern "C"
{

    static float *d_reduce = nullptr;
    static int d_reduce_cap = 0;

    static void ensure_reduce(int n)
    {
        if (n > d_reduce_cap)
        {
            if (d_reduce)
                cudaFree(d_reduce);
            cudaMalloc(&d_reduce, n * sizeof(float));
            d_reduce_cap = n;
        }
    }

    int fp8_quantize_fp16(const void *input_fp16, void *output_fp8,
                          float *scale_out, int n)
    {
        // Pass 1: absmax (each thread handles 4 elements)
        int grid1 = (n + BLOCK * 4 - 1) / (BLOCK * 4);
        ensure_reduce(grid1 + 256);
        absmax_fp16_kernel<<<grid1, BLOCK, BLOCK * sizeof(float)>>>(
            (const half *)input_fp16, d_reduce, n);

        // Multi-pass reduce
        float *src = d_reduce;
        int cnt = grid1;
        while (cnt > 1)
        {
            int grid2 = (cnt + BLOCK * 2 - 1) / (BLOCK * 2);
            absmax_f32_kernel<<<grid2, BLOCK, BLOCK * sizeof(float)>>>(
                src, src + cnt, cnt);
            src = src + cnt;
            cnt = grid2;
        }

        // Quantize (each thread handles 2 elements)
        int grid3 = (n + BLOCK * 2 - 1) / (BLOCK * 2);
        quant_kernel<<<grid3, BLOCK>>>(
            (const half *)input_fp16, (uint8_t *)output_fp8,
            src, scale_out, n);

        return 0;
    }

    int fp8_dequantize_to_fp16(const void *input_fp8, void *output_fp16,
                               const float *scale_in, int n)
    {
        int grid = (n + BLOCK * 2 - 1) / (BLOCK * 2);
        dequant_kernel<<<grid, BLOCK>>>(
            (const uint8_t *)input_fp8, (half *)output_fp16,
            scale_in, n);
        return 0;
    }

    int cuda_device_sync2(void) { return (int)cudaDeviceSynchronize(); }
}