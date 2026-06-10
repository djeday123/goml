// Step 2b: query the per-block SMEM opt-in cap and current default.
// Prints whether we can bump cudaFuncAttributeMaxDynamicSharedMemorySize
// to ~100KB on this RTX PRO 6000 Blackwell to allow 2 blocks/SM at 49KB each.

#include <cuda_runtime.h>
#include <cstdio>

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); return 1; }} while(0)

int main()
{
    int dev = 0;
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s  compute %d.%d  SMs=%d\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    int default_smem, optin_smem, smem_per_sm, reserved;
    CK(cudaDeviceGetAttribute(&default_smem,
        cudaDevAttrMaxSharedMemoryPerBlock, dev));
    CK(cudaDeviceGetAttribute(&optin_smem,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    CK(cudaDeviceGetAttribute(&smem_per_sm,
        cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
    CK(cudaDeviceGetAttribute(&reserved,
        cudaDevAttrReservedSharedMemoryPerBlock, dev));

    printf("\nShared memory limits:\n");
    printf("  Default per-block max:        %d KB\n", default_smem / 1024);
    printf("  Opt-in per-block max:         %d KB\n", optin_smem / 1024);
    printf("  Per-SM total:                 %d KB\n", smem_per_sm / 1024);
    printf("  Reserved (driver) per block:  %d B\n",  reserved);

    printf("\nProjections for v68:\n");
    int v66_dynsmem = 57344;    // measured from NCu = 57.34 KB
    int v68_dynsmem = v66_dynsmem - 8192;  // - smV_T = 49152
    printf("  v66 dynamic SMEM (measured):  %d B = %.1f KB\n",
        v66_dynsmem, v66_dynsmem / 1024.0);
    printf("  v68 projected (- smV_T 8KB):  %d B = %.1f KB\n",
        v68_dynsmem, v68_dynsmem / 1024.0);
    printf("  Two blocks/SM need:           %d B = %.1f KB\n",
        2 * v68_dynsmem, 2 * v68_dynsmem / 1024.0);
    printf("  Per-SM cap:                   %d B = %.1f KB\n",
        smem_per_sm, smem_per_sm / 1024.0);

    bool two_blocks_fit_per_sm = (2 * v68_dynsmem) <= smem_per_sm;
    bool single_block_above_default = v68_dynsmem > default_smem;
    bool single_block_within_optin = v68_dynsmem <= optin_smem;

    printf("\nVerdict:\n");
    printf("  Per-SM capacity for 2 blocks:    %s\n",
        two_blocks_fit_per_sm ? "YES" : "NO (would exceed per-SM total)");
    printf("  v68 per-block above default:     %s (needs opt-in if YES)\n",
        single_block_above_default ? "YES" : "NO");
    printf("  v68 per-block within opt-in cap: %s\n",
        single_block_within_optin ? "YES" : "NO");

    if (two_blocks_fit_per_sm && single_block_within_optin) {
        printf("\n  → GO: cudaFuncSetAttribute(MaxDynamicSharedMemorySize, %d)\n",
            v68_dynsmem);
        printf("  → Expected occupancy: 2 blocks/SM × 4 warps = 8 warps/SM (16.67%%)\n");
    } else {
        printf("\n  → STOP: SMEM cap insufficient. Re-plan.\n");
    }
    return 0;
}
