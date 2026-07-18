#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void tuned128() { __shared__ int s[1]; s[0] = threadIdx.x; }

int main() {
    // Query current merged occupancy
    // Use existing binary approach: just print device caps + do arithmetic
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("SM regs=%d threadsPerBlock=%d cap=%d.%d\n",
           p.regsPerMultiprocessor, p.maxThreadsPerBlock, p.major, p.minor);
    printf("\n--- Merged 253r, 46592B smem, 128 threads/block ---\n");
    int th = 128;
    int r = 253;
    int reg_block = r * th;
    int blk_by_reg = p.regsPerMultiprocessor / reg_block;
    int smem_bytes = 46592;
    int smem_slot = ((smem_bytes + 1023) & ~1023);
    int blk_by_smem = p.sharedMemPerMultiprocessor / smem_slot;
    printf("reg/block=%d → blk_by_reg=%d\n", reg_block, blk_by_reg);
    printf("smem/block=%d (slot %d) → blk_by_smem=%d\n", smem_bytes, smem_slot, blk_by_smem);

    printf("\n--- Find max R for 3 blocks by regs ---\n");
    for (int R : {150, 160, 168, 170, 176, 180, 200, 213}) {
        int rb = R * th;
        int b = p.regsPerMultiprocessor / rb;
        printf("  R=%d → reg/block=%d → blk_by_reg=%d\n", R, rb, b);
    }

    printf("\n--- Find max smem for 3 blocks ---\n");
    for (int S : {33792, 34133, 34816, 40960, 41472, 46592}) {
        int slot = ((S + 1023) & ~1023);
        int b = p.sharedMemPerMultiprocessor / slot;
        printf("  smem=%d (slot %d) → blk=%d\n", S, slot, b);
    }
    return 0;
}
