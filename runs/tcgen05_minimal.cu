// tcgen05_minimal.cu — minimal valid tcgen05 kernel.
// Compiles for sm_100a. We'll then:
//   (a) cuobjdump -sass to see what SASS opcodes ptxas emits for tcgen05.
//   (b) attempt to load this sm_100a cubin on sm_120 hardware and record
//       the exact CUDA error code the driver returns.

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void tcgen05_kernel() {
    __shared__ alignas(16) uint32_t tmem_addr;
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&tmem_addr)));
    asm volatile(
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;\n"
        :: "r"(tmem_addr));
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("Device: %s (sm_%d%d)\n", p.name, p.major, p.minor);
    tcgen05_kernel<<<1, 32>>>();
    cudaError_t e = cudaDeviceSynchronize();
    printf("launch+sync err = %d (%s)\n", (int)e, cudaGetErrorString(e));
    return 0;
}
