// cublas_wrapper.c — Thin wrapper around cublasGemmEx for purego (avoids 19-arg limit).
// Compile: nvcc -shared -o libcublas_wrapper.so cublas_wrapper.c -lcublas
// Or: gcc -shared -fPIC -o libcublas_wrapper.so cublas_wrapper.c -lcublas -L/usr/local/cuda/lib64 -I/usr/local/cuda/include

#include <cublas_v2.h>

// Packed args struct — matches Go side exactly
typedef struct
{
    cublasHandle_t handle;
    cublasOperation_t transa;
    cublasOperation_t transb;
    int m, n, k;
    const void *alpha;
    const void *A;
    cudaDataType Atype;
    int lda;
    const void *B;
    cudaDataType Btype;
    int ldb;
    const void *beta;
    void *C;
    cudaDataType Ctype;
    int ldc;
    cublasComputeType_t computeType;
    cublasGemmAlgo_t algo;
} GemmExArgs;

// Single-pointer entry point: purego calls this with 1 arg
cublasStatus_t gemmex_wrapper(GemmExArgs *a)
{
    return cublasGemmEx(
        a->handle,
        a->transa, a->transb,
        a->m, a->n, a->k,
        a->alpha,
        a->A, a->Atype, a->lda,
        a->B, a->Btype, a->ldb,
        a->beta,
        a->C, a->Ctype, a->ldc,
        a->computeType, a->algo);
}

// Batched version: cublasGemmStridedBatchedEx has even more args
typedef struct
{
    cublasHandle_t handle;
    cublasOperation_t transa;
    cublasOperation_t transb;
    int m, n, k;
    const void *alpha;
    const void *A;
    cudaDataType Atype;
    int lda;
    long long strideA;
    const void *B;
    cudaDataType Btype;
    int ldb;
    long long strideB;
    const void *beta;
    void *C;
    cudaDataType Ctype;
    int ldc;
    long long strideC;
    int batchCount;
    cublasComputeType_t computeType;
    cublasGemmAlgo_t algo;
} GemmStridedBatchedExArgs;

cublasStatus_t gemm_strided_batched_ex_wrapper(GemmStridedBatchedExArgs *a)
{
    return cublasGemmStridedBatchedEx(
        a->handle,
        a->transa, a->transb,
        a->m, a->n, a->k,
        a->alpha,
        a->A, a->Atype, a->lda, a->strideA,
        a->B, a->Btype, a->ldb, a->strideB,
        a->beta,
        a->C, a->Ctype, a->ldc, a->strideC,
        a->batchCount,
        a->computeType, a->algo);
}