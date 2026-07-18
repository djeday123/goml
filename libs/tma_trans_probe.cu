// N1: TMA-transpose quick-kill probe (3 checks):
//   (a) cuTensorMapEncodeTiled with transposed strides for FP8 rank-2 — accepts/rejects?
//   (b) compile check for cp.async.bulk.tensor.2d.trans (does .trans modifier exist?)
//   (c) if enrolled — paper estimate of L2 amplification for column reads

#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

// (b) DEAD — PTX rejected .trans modifier. See earlier compile output.

int main() {
    cudaFree(0);

    // (a) Test: cuTensorMapEncodeTiled with TRANSPOSED strides
    // K natural in global: [sl][hd] = [8192][128], stride_row=128 bytes (contiguous cols)
    // K^T view: swap dims -> [hd][sl] = [128][8192], stride_row=1 (impossible! elem_size mismatch)
    // Cleaner: encode K as [hd, sl] with strides (col-major access) -- innermost = sl, stride would be sl*1=8192

    // Test A: normal K layout
    uint8_t* g_ptr;
    cudaMalloc(&g_ptr, 128 * 8192);

    CUtensorMap tm_normal;
    {
        uint64_t global_dims[2]    = {128, 8192};   // {cols=hd, rows=sl}
        uint64_t global_strides[1] = {128};         // byte stride between rows
        uint32_t box_dims[2]       = {128, 64};     // {cols=hd, rows=Bc}
        uint32_t elem_strides[2]   = {1, 1};
        CUresult r = cuTensorMapEncodeTiled(&tm_normal, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
            g_ptr, global_dims, global_strides, box_dims, elem_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        const char* estr; if (r != CUDA_SUCCESS) cuGetErrorString(r, &estr);
        printf("Normal K [8192][128] rank-2 UINT8 SWIZZLE_128B: r=%d (%s)\n", (int)r, r==0?"OK":estr);
    }

    // Test B: TRANSPOSED strides — K^T view [128][8192] with innermost=sl (stride 8192)
    CUtensorMap tm_trans;
    {
        // K^T view: outer dim = hd (128), inner dim = sl (8192)
        // If we tell TMA that innermost is "sl" with stride 8192 elements between "hd" rows
        // But TMA innermost MUST be contiguous (stride = 1 element)
        uint64_t global_dims[2]    = {8192, 128};   // {inner=sl, outer=hd}
        uint64_t global_strides[1] = {8192};        // byte stride between "hd" rows = sl bytes
        uint32_t box_dims[2]       = {64, 128};     // tile = 64 rows_sl × 128 cols_hd
        uint32_t elem_strides[2]   = {1, 1};
        CUresult r = cuTensorMapEncodeTiled(&tm_trans, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
            g_ptr, global_dims, global_strides, box_dims, elem_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        const char* estr; if (r != CUDA_SUCCESS) cuGetErrorString(r, &estr);
        printf("Transposed K^T view (outer=hd, inner=sl stride 8192): r=%d (%s)\n", (int)r, r==0?"OK":estr);
    }

    // Test C: TRULY transposed (K natural but access col-major) — stride 1 for outer, N for inner
    // This is what would be needed for "K^T from natural K"
    CUtensorMap tm_impossible;
    {
        uint64_t global_dims[2]    = {8192, 128};   // {sl, hd}
        uint64_t global_strides[1] = {1};           // 1 byte stride between rows -- innermost stride
        uint32_t box_dims[2]       = {64, 128};
        uint32_t elem_strides[2]   = {1, 128};       // outer stride = 128
        CUresult r = cuTensorMapEncodeTiled(&tm_impossible, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
            g_ptr, global_dims, global_strides, box_dims, elem_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        const char* estr; if (r != CUDA_SUCCESS) cuGetErrorString(r, &estr);
        printf("Column-major access (stride=1 innermost): r=%d (%s)\n", (int)r, r==0?"OK":estr);
    }

    cudaFree(g_ptr);
    return 0;
}
