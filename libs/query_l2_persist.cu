// 044 I.1: cudaDevAttrMaxPersistingL2CacheSize runtime query
#include <cstdio>
#include <cuda_runtime.h>
int main() {
    int dev; cudaGetDevice(&dev);
    int max_persist = 0, l2_size = 0, l2_frac = 0;
    cudaDeviceGetAttribute(&max_persist, cudaDevAttrMaxPersistingL2CacheSize, dev);
    cudaDeviceGetAttribute(&l2_size,     cudaDevAttrL2CacheSize,             dev);
    // cudaDevAttrPersistingL2CacheMaxSize -- недоступен в CUDA 13.1 headers
    printf("cudaDevAttrL2CacheSize                = %d B = %.2f MiB\n",
           l2_size, l2_size / (1024.0*1024.0));
    printf("cudaDevAttrMaxPersistingL2CacheSize   = %d B = %.2f MiB\n",
           max_persist, max_persist / (1024.0*1024.0));
    printf("cudaDevAttrPersistingL2CacheMaxSize   = %d B = %.2f MiB\n",
           l2_frac, l2_frac / (1024.0*1024.0));
    return 0;
}
