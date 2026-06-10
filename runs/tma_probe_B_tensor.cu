// tma_probe_B_tensor.cu — [B] cp.async.bulk.tensor.2d (real TMA) on sm_120a
// Isolated test: real TMA via tensormap descriptor (cuTensorMapEncodeTiled).
// Independent of probe [A] so an illegal-instruction kill in A doesn't poison
// this one. If A=PASS but B=FAIL → bulk primitive exists, tensor-descriptor
// path does NOT exist on Blackwell consumer.
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>

__global__ void k(const __grid_constant__ CUtensorMap tmap, uint8_t *out)
{
    __shared__ alignas(128) uint8_t smem[2048];   // tile = 32×32 bytes = 1024 B, double for safety
    __shared__ alignas(8) uint64_t bar;

    if (threadIdx.x == 0) {
        uint32_t bar_s  = __cvta_generic_to_shared(&bar);
        uint32_t smem_s = __cvta_generic_to_shared(smem);

        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(bar_s));
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], 1024;" :: "r"(bar_s));
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3}], [%4];"
            :: "r"(smem_s), "l"(&tmap), "r"(0), "r"(0), "r"(bar_s));
        asm volatile(
            "{ .reg .pred P;\n"
            "  WAITB: mbarrier.try_wait.parity.shared.b64 P, [%0], 0;\n"
            "  @P bra DONEB;\n"
            "  bra WAITB;\n"
            "  DONEB: }\n"
            :: "r"(bar_s));
        out[0] = smem[0];  // prevent DCE
    }
}

int main()
{
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("[B] cp.async.bulk.tensor.2d on %s sm_%d%d\n", p.name, p.major, p.minor);

    // 32x32-byte tile from a 128x128-byte source.
    uint8_t *d; cudaMalloc(&d, 128 * 128); cudaMemset(d, 0xCD, 128 * 128);

    CUtensorMap tmap;
    memset(&tmap, 0, sizeof(tmap));
    cuuint64_t globalDim[2]    = {128, 128};
    cuuint64_t globalStride[1] = {128};
    cuuint32_t boxDim[2]       = {32, 32};
    cuuint32_t elementStride[2]= {1, 1};

    CUresult res = cuTensorMapEncodeTiled(
        &tmap,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2, d,
        globalDim, globalStride,
        boxDim, elementStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (res != CUDA_SUCCESS) {
        const char *errstr = nullptr; cuGetErrorString(res, &errstr);
        printf("[B] RESULT: FAIL @ cuTensorMapEncodeTiled: %s\n", errstr ? errstr : "?");
        cudaFree(d);
        return 2;
    }
    printf("[B] cuTensorMapEncodeTiled: OK (driver API accepts the descriptor)\n");

    k<<<1, 32>>>(tmap, d);
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) printf("[B] RESULT: PASS — TMA tensor.2d works on sm_120a\n");
    else                    printf("[B] RESULT: FAIL @ runtime — %s\n", cudaGetErrorString(err));
    cudaFree(d);
    return (err == cudaSuccess) ? 0 : 1;
}
