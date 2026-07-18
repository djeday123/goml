#include <cstdio>
#include <cuda_runtime.h>
int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("name=%s cc=%d.%d SMs=%d\n", p.name, p.major, p.minor, p.multiProcessorCount);
    printf("regsPerBlock=%d regsPerSM=%d\n", p.regsPerBlock, p.regsPerMultiprocessor);
    printf("sharedPerBlock=%zu sharedPerSM=%zu sharedOptin=%zu\n",
           p.sharedMemPerBlock, p.sharedMemPerMultiprocessor, p.sharedMemPerBlockOptin);
    printf("maxThreadsPerSM=%d maxThreadsPerBlock=%d\n",
           p.maxThreadsPerMultiProcessor, p.maxThreadsPerBlock);
    printf("warpSize=%d L2=%d KB\n", p.warpSize, (int)(p.l2CacheSize / 1024));
    return 0;
}
