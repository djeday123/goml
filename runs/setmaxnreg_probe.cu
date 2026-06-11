// Bonus M8: setmaxnreg.dec.sync.aligned.u32 N — есть ли на sm_120a?
// Hopper-style warp-group реg redistribution. Если работает — открывает FA3-style
// producer/consumer reg balance. На consumer-Blackwell (sm_120) официально не задокументировано.
#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);}}while(0)

__global__ void probe_kernel(int *out)
{
    // Decrease our register budget to 80 (down from default).
    asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
    // Increase к 240.
    asm volatile("setmaxnreg.inc.sync.aligned.u32 240;");
    if (threadIdx.x == 0) out[0] = 0xC0DE;
}

int main() {
    int *d, h = 0;
    CK(cudaMalloc(&d, sizeof(int)));
    CK(cudaMemset(d, 0, sizeof(int)));
    probe_kernel<<<1, 128>>>(d);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("EXEC FAIL: %s\n", cudaGetErrorString(e));
        return 1;
    }
    CK(cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost));
    printf("OK marker=0x%X\n", h);
    return 0;
}
