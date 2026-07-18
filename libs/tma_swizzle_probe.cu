// TMA swizzle mode availability probe on sm_120a.
// cuTensorMapEncodeTiled with SWIZZLE_128B — accepted at runtime on our GPU?

#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

int main() {
    // Create runtime context implicitly
    cudaFree(0);

    // Try each swizzle mode
    const char* mode_names[] = {"NONE", "32B", "64B", "128B"};
    CUtensorMapSwizzle modes[] = {
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_SWIZZLE_32B,
        CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_SWIZZLE_128B
    };

    // Allocate dummy global tensor (128x128 uint8, 16 KB, 128-aligned)
    uint8_t* g_ptr;
    cudaMalloc(&g_ptr, 128 * 128);

    uint64_t global_dims[2]     = {128, 128};   // 128 cols, 128 rows (fastest-changing first)
    uint64_t global_strides[1]  = {128};        // byte stride between rows
    uint32_t box_dims[2]        = {64, 64};     // tile 64x64
    uint32_t elem_strides[2]    = {1, 1};

    for (int i = 0; i < 4; i++) {
        CUtensorMap tm;
        CUresult r = cuTensorMapEncodeTiled(
            &tm,
            CU_TENSOR_MAP_DATA_TYPE_UINT8,
            2,
            g_ptr,
            global_dims,
            global_strides,
            box_dims,
            elem_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            modes[i],
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        const char* err = "?";
        if (r == CUDA_SUCCESS) err = "OK";
        else {
            const char* estr = nullptr;
            cuGetErrorString(r, &estr);
            err = estr ? estr : "unknown";
        }
        printf("SWIZZLE_%-5s: r=%d (%s)\n", mode_names[i], (int)r, err);
    }

    cudaFree(g_ptr);
    return 0;
}
