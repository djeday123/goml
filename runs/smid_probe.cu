// Probe co-residency: записать (blockIdx, smid) для каждого блока.
// Запускаем тот же grid что v96b cfg=9 (bh=64 sl=8192 → grid=5504, FA_THREADS=128).
// Хост анализирует: какие blockIdx делят SM (для launch_bounds(_, 2) ожидаем pairs).
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);}}while(0)

#define FA_THREADS 128
#define FA_BR 96
#define HD 128
#define FA_BC 64
#define FA_STRIDE 128
#define SMV_T_STRIDE 68

// Эмулируем v96b SMEM footprint (44.52 KB) для realistic occupancy
__global__ void __launch_bounds__(FA_THREADS, 2)
    probe_kernel(int *blk2smid)
{
    extern __shared__ uint8_t raw[];
    (void)raw;
    if (threadIdx.x == 0) {
        unsigned int smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        blk2smid[blockIdx.x] = (int)smid;
    }
}

int main() {
    int bh = 64, sl = 8192;
    int nqt = (sl + FA_BR - 1) / FA_BR;
    int grid = bh * nqt;
    printf("grid=%d (bh=%d × nqt=%d), threads=%d\n", grid, bh, nqt, FA_THREADS);
    int *d, *h;
    CK(cudaMalloc(&d, grid * sizeof(int)));
    CK(cudaMemset(d, 0xff, grid * sizeof(int)));
    h = (int*)malloc(grid * sizeof(int));

    int smem = FA_BR * FA_STRIDE + 2 * FA_BC * FA_STRIDE + FA_BC * FA_STRIDE + 16
             + HD * SMV_T_STRIDE;
    CK(cudaFuncSetAttribute(probe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    probe_kernel<<<grid, FA_THREADS, smem>>>(d);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h, d, grid * sizeof(int), cudaMemcpyDeviceToHost));

    // Кол-во SMs, сколько блоков на каждом SM.
    int max_smid = 0;
    for (int i = 0; i < grid; i++) if (h[i] > max_smid) max_smid = h[i];
    int n_sm = max_smid + 1;
    int *count = (int*)calloc(n_sm, sizeof(int));
    for (int i = 0; i < grid; i++) count[h[i]]++;

    int max_per_sm = 0;
    for (int s = 0; s < n_sm; s++) if (count[s] > max_per_sm) max_per_sm = count[s];

    printf("n_sm=%d, max blocks per SM in this launch=%d\n", n_sm, max_per_sm);

    // Первые 40 блоков: (blockIdx, smid)
    printf("Первые 40 (blockIdx → smid):\n");
    for (int i = 0; i < 40 && i < grid; i++) {
        printf("  block %4d → SM %3d\n", i, h[i]);
    }

    // Сколько блоков попадает на SM 0, SM 1, SM 2 — первые тройки
    printf("\nДля каждого из первых 8 SMs — какие блоки на нём:\n");
    for (int s = 0; s < 8 && s < n_sm; s++) {
        printf("  SM %3d:", s);
        int cnt = 0;
        for (int i = 0; i < grid; i++) {
            if (h[i] == s) {
                printf(" %d", i);
                cnt++;
                if (cnt >= 8) { printf(" ..."); break; }
            }
        }
        printf("\n");
    }

    // Закономерность: для launch_bounds(_, 2) co-resident pair structure
    // Anyone block has co-resident peer with same SM. Check delta.
    printf("\nДельты внутри SM (block_i+1 - block_i на том же SM, первые 20 SMs):\n");
    for (int s = 0; s < 20 && s < n_sm; s++) {
        int prev = -1; int delta_print = 0;
        printf("  SM %3d:", s);
        for (int i = 0; i < grid && delta_print < 6; i++) {
            if (h[i] == s) {
                if (prev >= 0) printf(" Δ=%d", i - prev);
                prev = i;
                delta_print++;
            }
        }
        printf("\n");
    }

    free(count);
    free(h);
    cudaFree(d);
    return 0;
}
