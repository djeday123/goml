// wgmma_probe.cu — minimal "does it compile + run?" test for WGMMA on sm_120
//
// Tries wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16. If sm_120 doesn't
// support this PTX form, nvcc will emit an error at compile time, telling us
// to fall back (e.g. to tcgen05) or stick with the m16n8k16 family.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); }} while(0)

// SMEM descriptor builder per Hopper PTX ISA section 9.7.13.4
// Layout: stride offset(14b) | leading offset(14b) | base(14b) | 0(1b)
//         + swizzle (3b at bit 61..63)
__device__ __forceinline__ uint64_t make_smem_desc(const void *p, uint32_t leading_off, uint32_t stride_off, uint32_t swizzle)
{
    uint32_t base = __cvta_generic_to_shared(p);
    uint64_t desc = ((uint64_t)(base >> 4) & 0x3FFF) |
                    (((uint64_t)(leading_off >> 4) & 0x3FFF) << 16) |
                    (((uint64_t)(stride_off >> 4) & 0x3FFF) << 32) |
                    (((uint64_t)swizzle & 0x7) << 61);
    return desc;
}

__global__ void wgmma_test_kernel(__half *A, __half *B, float *D, int *ok)
{
    // 128 threads = 1 warpgroup
    __shared__ __half smA[64 * 16];
    __shared__ __half smB[16 * 8];
    int tid = threadIdx.x;
    // Cooperative load
    for (int i = tid; i < 64 * 16; i += 128) smA[i] = A[i];
    for (int i = tid; i < 16 * 8; i += 128) smB[i] = B[i];
    __syncthreads();

    // Build SMEM descriptors. For row-major, leading_off = stride between
    // "leading dimension" rows in bytes (k), stride_off = row stride (m).
    // For a 64x16 fp16 row-major matrix: row stride = 32B, leading stride = 8B per swizzle blk.
    uint64_t descA = make_smem_desc(smA, 16, 32, 0);  // no swizzle
    uint64_t descB = make_smem_desc(smB, 16, 16, 0);

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    asm volatile(
        "{\n"
        "  wgmma.fence.sync.aligned;\n"
        "  wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
        "    {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 1, 1;\n"
        "  wgmma.commit_group.sync.aligned;\n"
        "  wgmma.wait_group.sync.aligned 0;\n"
        "}\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "l"(descA), "l"(descB));

    if (tid == 0) {
        D[0] = d0;
        D[1] = d1;
        D[2] = d2;
        D[3] = d3;
        *ok = 1;
    }
}

int main()
{
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("Device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    __half *A, *B;
    float *D;
    int *ok;
    CK(cudaMallocManaged(&A, 64 * 16 * 2));
    CK(cudaMallocManaged(&B, 16 * 8 * 2));
    CK(cudaMallocManaged(&D, 4 * 4));
    CK(cudaMallocManaged(&ok, 4));
    for (int i = 0; i < 64 * 16; i++) A[i] = __float2half(1.0f);
    for (int i = 0; i < 16 * 8; i++) B[i] = __float2half(1.0f);
    *ok = 0;
    wgmma_test_kernel<<<1, 128>>>(A, B, D, ok);
    CK(cudaDeviceSynchronize());
    printf("ok=%d  D[0..3]=%.1f %.1f %.1f %.1f  (expected ~16 each if A,B all-ones, but layout differs)\n",
           *ok, D[0], D[1], D[2], D[3]);
    return 0;
}
