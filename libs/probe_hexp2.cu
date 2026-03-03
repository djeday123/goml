// Probe: ex2.approx.f16x2 on SM89
// nvcc -O3 -arch=sm_89 -std=c++17 probe_hexp2.cu -o probe_hexp2 -lcudart
// If compile fails → SM89 doesn't support it in PTX
// If runs → check SASS with: nvcc -O3 -arch=sm_89 --ptxas-options=-v -cubin probe_hexp2.cu && cuobjdump -sass probe_hexp2.cubin

#include <stdint.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Method 1: inline PTX ex2.approx.f16x2
__device__ __forceinline__ uint32_t hexp2_ptx(uint32_t in)
{
    uint32_t out;
    asm("ex2.approx.f16x2 %0, %1;" : "=r"(out) : "r"(in));
    return out;
}

// Method 2: inline PTX ex2.approx.f16 (scalar)
__device__ __forceinline__ unsigned short hexp2_scalar_ptx(unsigned short in)
{
    unsigned short out;
    asm("ex2.approx.f16 %0, %1;" : "=h"(out) : "h"(in));
    return out;
}

__global__ void test_hexp2(float *out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    // Test value: exp2(1.0) = 2.0, exp2(0.0) = 1.0
    __half h_one = __float2half(1.0f);
    __half h_zero = __float2half(0.0f);
    __half h_neg = __float2half(-1.0f);

    // Pack into half2: {1.0, 0.0}
    uint32_t packed;
    __half2 h2 = __halves2half2(h_one, h_zero);
    packed = *reinterpret_cast<uint32_t *>(&h2);

    uint32_t result = hexp2_ptx(packed);
    __half2 h2_result = *reinterpret_cast<__half2 *>(&result);

    __half lo = __low2half(h2_result);
    __half hi = __high2half(h2_result);

    // exp2(1.0) = 2.0, exp2(0.0) = 1.0
    out[idx * 4 + 0] = __half2float(lo); // should be 2.0
    out[idx * 4 + 1] = __half2float(hi); // should be 1.0

    // Also test scalar
    unsigned short s_neg = __half_as_ushort(h_neg);
    unsigned short s_res = hexp2_scalar_ptx(s_neg);
    out[idx * 4 + 2] = __half2float(__ushort_as_half(s_res)); // exp2(-1) = 0.5

    // Reference: FP32
    out[idx * 4 + 3] = exp2f(1.0f); // 2.0
}

int main()
{
    float *d_out, h_out[16];
    cudaMalloc(&d_out, 64);
    test_hexp2<<<1, 1>>>(d_out, 1);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        printf("ex2.approx.f16x2 NOT supported on this GPU\n");
        cudaFree(d_out);
        return 1;
    }
    cudaMemcpy(h_out, d_out, 64, cudaMemcpyDeviceToHost);

    printf("ex2.approx.f16x2 SUPPORTED on SM89!\n");
    printf("  exp2(1.0)  = %.4f (expect 2.0)\n", h_out[0]);
    printf("  exp2(0.0)  = %.4f (expect 1.0)\n", h_out[1]);
    printf("  exp2(-1.0) = %.4f (expect 0.5)\n", h_out[2]);
    printf("  FP32 ref   = %.4f\n", h_out[3]);

    cudaFree(d_out);
    return 0;
}
