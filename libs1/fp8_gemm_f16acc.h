// fp8_gemm_f16acc.h — C interface for FP8 GEMM with FP16 accumulator
//
// Usage from Go (cgo):
//   // #cgo LDFLAGS: -L. -lfp8gemm -lcudart
//   // #include "fp8_gemm_f16acc.h"
//   import "C"
//   C.fp8_gemm_f16acc(M, N, K, A_ptr, B_ptr, C_ptr)
//
// Layout:
//   A: [M, K] row-major, FP8 e4m3 (device pointer)
//   B: [N, K] row-major, FP8 e4m3 (device pointer, pre-transposed)
//   C: [M, N] row-major, FP16     (device pointer)
//
// Returns: 0 on success, CUDA error code on failure

#ifndef FP8_GEMM_F16ACC_H
#define FP8_GEMM_F16ACC_H

#ifdef __cplusplus
extern "C"
{
#endif

    // FP8 GEMM: C (fp16) = A (fp8 e4m3) × B^T (fp8 e4m3)
    // Uses FP16 accumulator for 2x throughput over FP32 acc on RTX 4090
    int fp8_gemm_f16acc(
        int M, int N, int K,
        const void *A, // device ptr, FP8 e4m3, [M, K] row-major
        const void *B, // device ptr, FP8 e4m3, [N, K] row-major (transposed)
        void *C);      // device ptr, FP16,     [M, N] row-major

#ifdef __cplusplus
}
#endif

#endif // FP8_GEMM_F16ACC_H