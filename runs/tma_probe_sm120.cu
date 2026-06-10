// tma_probe_sm120.cu — full TMA probe for sm_120a (Blackwell consumer)
// Tests three things, each independently:
//   A) cp.async.bulk.shared::cluster.global (simple bulk, sm_90+)
//   B) cp.async.bulk.tensor.2d.shared::cluster.global.tile (real TMA, needs tensormap)
//   C) mbarrier.arrive.expect_tx (TMA completion signaling)
// If A compiles+runs but B fails → bulk works but TMA descriptor path absent.
// If B compiles and runs without illegal-instruction → TMA available on sm_120a.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>

#define CK(x) do { cudaError_t e = (x); \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA err %d: %s @ %s:%d\n", e, cudaGetErrorString(e), __FILE__, __LINE__); return -1; } } while(0)

// ---------- A) plain bulk copy with mbarrier (sm_90 primitive) ----------
__global__ void k_bulk(uint8_t *g)
{
    __shared__ alignas(128) uint8_t smem[1024];
    __shared__ alignas(8) uint64_t bar;

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar)));
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
                     "[%0], [%1], 1024, [%2];\n"
                     :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
                        "l"(g),
                        "r"((uint32_t)__cvta_generic_to_shared(&bar)));
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], 1024;"
                     :: "r"((uint32_t)__cvta_generic_to_shared(&bar)));
        asm volatile("{ .reg .pred P; LAB_WAIT: mbarrier.try_wait.parity.shared.b64 P, [%0], 0; @P bra DONE; bra LAB_WAIT; DONE: }"
                     :: "r"((uint32_t)__cvta_generic_to_shared(&bar)));
        // simple use of smem to prevent DCE
        g[0] = smem[0];
    }
}

// ---------- B) TMA tensor.2d copy using tensormap descriptor ----------
__global__ void k_tma_2d(const __grid_constant__ CUtensorMap tmap, uint8_t *out)
{
    __shared__ alignas(128) uint8_t smem[1024];
    __shared__ alignas(8) uint64_t bar;

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar)));
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3}], [%4];\n"
            :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
               "l"(&tmap),
               "r"(0), "r"(0),
               "r"((uint32_t)__cvta_generic_to_shared(&bar)));
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], 1024;"
                     :: "r"((uint32_t)__cvta_generic_to_shared(&bar)));
        asm volatile("{ .reg .pred P; LAB2: mbarrier.try_wait.parity.shared.b64 P, [%0], 0; @P bra END2; bra LAB2; END2: }"
                     :: "r"((uint32_t)__cvta_generic_to_shared(&bar)));
        out[0] = smem[0];
    }
}

int main()
{
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Device: %s  sm_%d%d  L2=%d KB  SMs=%d\n",
           prop.name, prop.major, prop.minor, (int)(prop.l2CacheSize / 1024), prop.multiProcessorCount);

    int driverVer = 0, runtimeVer = 0;
    cudaDriverGetVersion(&driverVer);
    cudaRuntimeGetVersion(&runtimeVer);
    printf("Driver: %d  Runtime: %d\n\n", driverVer, runtimeVer);

    // Test A: plain bulk copy
    uint8_t *d_g;
    CK(cudaMalloc(&d_g, 4096));
    CK(cudaMemset(d_g, 0xAB, 4096));

    printf("[A] cp.async.bulk + mbarrier: ");
    fflush(stdout);
    k_bulk<<<1, 32>>>(d_g);
    cudaError_t errA = cudaDeviceSynchronize();
    if (errA == cudaSuccess) printf("OK (bulk + mbarrier work on sm_120a)\n");
    else printf("FAIL: %s\n", cudaGetErrorString(errA));

    // Test B: real TMA via tensormap
    printf("[B] cp.async.bulk.tensor.2d (real TMA): ");
    fflush(stdout);

    // Build a tensormap for a 128x32 byte tile from global memory.
    CUtensorMap tmap;
    void *globalAddr = d_g;
    cuuint64_t globalDim[2] = {32, 128};        // x=cols (bytes), y=rows
    cuuint64_t globalStride[1] = {32};           // stride between rows
    cuuint32_t boxDim[2] = {32, 32};             // tile shape
    cuuint32_t elementStride[2] = {1, 1};

    CUresult res = cuTensorMapEncodeTiled(
        &tmap,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2,                                   // tensorRank
        globalAddr,
        globalDim,
        globalStride,
        boxDim,
        elementStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (res != CUDA_SUCCESS) {
        const char *errstr = nullptr;
        cuGetErrorString(res, &errstr);
        printf("FAIL @ cuTensorMapEncodeTiled: %s\n", errstr ? errstr : "?");
    } else {
        k_tma_2d<<<1, 32>>>(tmap, d_g);
        cudaError_t errB = cudaDeviceSynchronize();
        if (errB == cudaSuccess) printf("OK (TMA tensor.2d works on sm_120a)\n");
        else printf("FAIL: %s\n", cudaGetErrorString(errB));
    }

    cudaFree(d_g);
    return 0;
}
