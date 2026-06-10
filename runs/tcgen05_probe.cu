// tcgen05_probe.cu — minimal test for tcgen05.mma availability on sm_120
#include <cstdio>
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void k() {
    // tcgen05.mma needs tcgen05.alloc, etc.  Just try one PTX line to see
    // if ptxas accepts the family on sm_120.
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [0];\n");
}

int main() {
    k<<<1, 32>>>();
    cudaDeviceSynchronize();
    printf("tcgen05 family available on this build target\n");
    return 0;
}
