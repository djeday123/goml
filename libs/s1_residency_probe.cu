// S1 residency probe:
//   1) cudaDevAttrReservedSharedMemoryPerBlock (X) on sm_120a
//   2) cudaDevAttrMaxSharedMemoryPerBlockOptin (SMEM cap)
//   3) cudaDevAttrMaxSharedMemoryPerMultiprocessor
//   4) cudaFuncGetAttributes(kernel_dq): numRegs, maxThreadsPerBlock
//   5) cudaOccupancyMaxActiveBlocksPerMultiprocessor with dynamic smem
//      (a)-layout 46336 B and (c)-layout 33792 B
//
// Compile against the current fa_bwd_dq.cu (holds either (a) or (c) kernel_dq).
// Rebuild the source once with __launch_bounds__ added, rebuild the probe,
// and rerun this binary — API answer will reflect the compiler hint.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace fa_bwd_dq {
    __global__ void kernel_dq(
        const uint8_t *, const uint8_t *, const uint8_t *,
        const __half *, const float *, const float *, float *,
        int, int, int, int, int, float);
}

static void must(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA fail: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main(int argc, char **argv) {
    int dev = 0;
    must(cudaSetDevice(dev), "setDevice");

    cudaDeviceProp p;
    must(cudaGetDeviceProperties(&p, dev), "getDeviceProperties");
    printf("device: %s  CC=%d.%d\n", p.name, p.major, p.minor);

    int reserved = 0;
    must(cudaDeviceGetAttribute(&reserved,
         cudaDevAttrReservedSharedMemoryPerBlock, dev), "reservedSMEM");
    int max_optin = 0;
    must(cudaDeviceGetAttribute(&max_optin,
         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev), "maxOptin");
    int max_sm = 0;
    must(cudaDeviceGetAttribute(&max_sm,
         cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev), "maxSM");
    int max_blocks = 0;
    must(cudaDeviceGetAttribute(&max_blocks,
         cudaDevAttrMaxBlocksPerMultiprocessor, dev), "maxBlocks");
    int max_warps_hw = 0;
    must(cudaDeviceGetAttribute(&max_warps_hw,
         cudaDevAttrMaxThreadsPerMultiProcessor, dev), "maxThreadsSM");

    printf("\n---- device attrs ----\n");
    printf("  reservedSharedMemoryPerBlock (X)       = %d bytes\n", reserved);
    printf("  maxSharedMemoryPerBlockOptin           = %d bytes\n", max_optin);
    printf("  maxSharedMemoryPerMultiprocessor       = %d bytes\n", max_sm);
    printf("  maxBlocksPerMultiprocessor             = %d\n", max_blocks);
    printf("  maxThreadsPerMultiProcessor            = %d  (=%d warps)\n",
           max_warps_hw, max_warps_hw / 32);

    cudaFuncAttributes fa;
    must(cudaFuncGetAttributes(&fa, fa_bwd_dq::kernel_dq), "funcAttr");
    printf("\n---- kernel_dq attrs ----\n");
    printf("  numRegs                                = %d\n", fa.numRegs);
    printf("  maxThreadsPerBlock (launch_bounds cap) = %d\n", fa.maxThreadsPerBlock);
    printf("  sharedSizeBytes (static)               = %zu\n", fa.sharedSizeBytes);
    printf("  localSizeBytes  (stack frame per thr)  = %zu\n", fa.localSizeBytes);
    printf("  preferredShmemCarveout                 = %d\n", fa.preferredShmemCarveout);

    // Raise dynamic smem cap so opt-in allocation succeeds
    must(cudaFuncSetAttribute(fa_bwd_dq::kernel_dq,
         cudaFuncAttributeMaxDynamicSharedMemorySize, max_optin),
         "setMaxDynamic");

    printf("\n---- cudaOccupancyMaxActiveBlocksPerMultiprocessor ----\n");
    const int block_size = 128;
    int layouts_bytes[]  = { 32768, 33280, 33792, 46336 };
    const char* layouts_lbl[] = { "S2-diet (d)     32768B",
                                   "Y0 P1a-smL/D    33280B",
                                   "P1a (sealed)    33792B",
                                   "L3-layout (a)   46336B" };
    for (int i = 0; i < 4; ++i) {
        int n = -1;
        must(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
             &n, fa_bwd_dq::kernel_dq, block_size, layouts_bytes[i]),
             "occAPI");
        printf("  smem=%5d B, block=%d threads -> %d blocks/SM  [%s]\n",
               layouts_bytes[i], block_size, n, layouts_lbl[i]);
    }

    // Y0 binary search: find the exact threshold where API flips 2 -> 3 blocks.
    printf("\n---- Y0 binary search: exact 3-block threshold ----\n");
    int lo = 0, hi = 46336, best = -1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int n = 0;
        must(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
             &n, fa_bwd_dq::kernel_dq, block_size, mid), "occSearch");
        if (n >= 3) { best = mid; lo = mid + 1; }
        else        { hi = mid - 1; }
    }
    printf("  max dyn smem giving >=3 blocks/SM = %d B\n", best);
    printf("  P1a-smL/D (33280 B) delta vs threshold = %d B\n", 33280 - best);

    // Confirm boundary with 128-byte-step probe around threshold.
    printf("\n---- fine scan around threshold ----\n");
    for (int d = best - 512; d <= best + 512; d += 128) {
        int n = 0;
        must(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
             &n, fa_bwd_dq::kernel_dq, block_size, d), "occFine");
        printf("  smem=%5d B -> %d blocks/SM\n", d, n);
    }

    // Deterministic bookkeeping:
    //   per_block_smem_effective = ceil( (X + dynamic_smem) / granularity ) * granularity
    // Unknown granularity; report the naive floor for user cross-check.
    printf("\n---- naive floor (no granularity) ----\n");
    for (int i = 0; i < 3; ++i) {
        int dyn = layouts_bytes[i];
        int per = reserved + dyn;
        int fl  = max_sm / per;
        printf("  layout %5d B: per_block(reserved+dyn) = %d B, floor(SM/per) = %d\n",
               dyn, per, fl);
    }
    printf("  target 3 blocks/SM: need dyn <= (max_sm/3 - X) = %d B\n",
           max_sm / 3 - reserved);
    printf("  P1-layout   deficit vs target = %d B\n",
           33792 - (max_sm / 3 - reserved));
    printf("  S2-diet (d) margin  vs target = %d B\n",
           (max_sm / 3 - reserved) - 32768);
    return 0;
}
