// tma_probe.cu — check cp.async.bulk.tensor availability on sm_120
#include <cstdio>
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void k() {
    __shared__ alignas(128) char smem[1024];
    asm volatile("cp.async.bulk.shared::cta.global [%0], [%1], 1024, [%2];\n"
                 :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
                    "l"((uint64_t)0), "r"(0u));
}
int main() { k<<<1,1>>>(); cudaDeviceSynchronize(); return 0; }
