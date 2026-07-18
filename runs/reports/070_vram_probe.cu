// 070 VRAM probe — measures alloc footprint by cudaMemGetInfo before/after malloc.
// Reproduces bench_r2c_e2e wall alloc pattern (with/without dS_T alloc).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

#define CKR(c) do{auto e=(c); if(e){printf("CUDA err %d\n",e); std::exit(1);}}while(0)

int main(int argc, char **argv) {
    int with_dST = (argc>=2 && std::string("with_dST")==argv[1]) ? 1 : 0;
    printf("mode: %s\n", with_dST ? "with_dST (pre-070)" : "no_dST (070 applied)");

    size_t bh=128, sl=8192, hd=128;
    size_t sz = bh*sl*hd;
    size_t lsz = bh*sl;
    size_t stride_ds = (sl+15)&~15;
    size_t dsz = bh*sl*stride_ds;

    size_t free0, total; cudaMemGetInfo(&free0, &total);
    printf("baseline: free=%zu MB, total=%zu MB\n", free0>>20, total>>20);

    // Same buffers bench_r2c_e2e allocates
    uint8_t *dQ, *dK, *dV8; __half *dOG, *dOO;
    float *dL, *dD, *ddV, *ddK, *ddQ;
    uint8_t *dS_nat, *dS_T;

    CKR(cudaMalloc(&dQ, sz));
    CKR(cudaMalloc(&dK, sz));
    CKR(cudaMalloc(&dV8, sz));
    CKR(cudaMalloc(&dOO, sz*2));
    CKR(cudaMalloc(&dOG, sz*2));
    CKR(cudaMalloc(&dL, lsz*4));
    CKR(cudaMalloc(&dD, lsz*4));
    CKR(cudaMalloc(&ddV, sz*4));
    CKR(cudaMalloc(&ddK, sz*4));
    CKR(cudaMalloc(&ddQ, sz*4));
    CKR(cudaMalloc(&dS_nat, dsz));
    if (with_dST) { CKR(cudaMalloc(&dS_T, dsz)); } else { dS_T = nullptr; }

    // Force commit by touching every page with cudaMemset (kilobytes).
    // Otherwise nvidia-smi may not report until first touch.
    CKR(cudaMemset(dS_nat, 0x37, dsz));
    if (dS_T) CKR(cudaMemset(dS_T, 0x39, dsz));

    size_t free1;
    cudaMemGetInfo(&free1, &total);
    size_t used = free0 - free1;
    printf("after alloc + touch: free=%zu MB, used=%zu MB\n", free1>>20, used>>20);
    printf("dsz per buffer: %zu MB (~%.2f GB)\n", dsz>>20, dsz/1073741824.0);

    // Report delta vs no_dST baseline
    printf("[MARKER] mode=%s used_MB=%zu\n", with_dST ? "with_dST" : "no_dST", used>>20);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV8);
    cudaFree(dOO); cudaFree(dOG);
    cudaFree(dL); cudaFree(dD);
    cudaFree(ddV); cudaFree(ddK); cudaFree(ddQ);
    cudaFree(dS_nat);
    if (dS_T) cudaFree(dS_T);
    return 0;
}
