#include <cublasLt.h>
#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>

typedef struct
{
    cublasLtHandle_t handle;
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t _pad;
    const void *A;
    const void *B;
    void *C;
    const float *alpha;
    const float *beta;
} Fp8MatmulArgs;

static float *d_one = NULL;

// Cached state for repeated calls with same dimensions
static cublasLtMatmulDesc_t s_opDesc = NULL;
static cublasLtMatrixLayout_t s_Adesc = NULL, s_Bdesc = NULL, s_Cdesc = NULL;
static cublasLtMatmulAlgo_t s_algo;
static void *s_workspace = NULL;
static size_t s_wsSize = 0;
static int32_t s_M = 0, s_N = 0, s_K = 0;

static void cleanup_cached(void)
{
    if (s_opDesc)
    {
        cublasLtMatmulDescDestroy(s_opDesc);
        s_opDesc = NULL;
    }
    if (s_Adesc)
    {
        cublasLtMatrixLayoutDestroy(s_Adesc);
        s_Adesc = NULL;
    }
    if (s_Bdesc)
    {
        cublasLtMatrixLayoutDestroy(s_Bdesc);
        s_Bdesc = NULL;
    }
    if (s_Cdesc)
    {
        cublasLtMatrixLayoutDestroy(s_Cdesc);
        s_Cdesc = NULL;
    }
    if (s_workspace)
    {
        cudaFree(s_workspace);
        s_workspace = NULL;
    }
    s_M = s_N = s_K = 0;
}

static int setup_cached(Fp8MatmulArgs *a)
{
    cublasLtMatmulPreference_t pref = NULL;
    cublasLtMatmulHeuristicResult_t heur;
    int found = 0;
    cublasStatus_t st;

    cleanup_cached();

    if (!d_one)
    {
        cudaMalloc((void **)&d_one, sizeof(float));
        float h = 1.0f;
        cudaMemcpy(d_one, &h, sizeof(float), cudaMemcpyHostToDevice);
    }

    int M = a->M, N = a->N, K = a->K;

    st = cublasLtMatmulDescCreate(&s_opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (st != 0)
        return (int)st;

    cublasOperation_t opA = CUBLAS_OP_T;
    cublasOperation_t opB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(s_opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    cublasLtMatmulDescSetAttribute(s_opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

    const void *sp = (const void *)d_one;
    cublasLtMatmulDescSetAttribute(s_opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sp, sizeof(sp));
    cublasLtMatmulDescSetAttribute(s_opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sp, sizeof(sp));
    cublasLtMatmulDescSetAttribute(s_opDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &sp, sizeof(sp));

    // transA=T: A stored [K,N], transposed to [N,K]. transB=N: B stored [K,M]
    cublasLtMatrixLayoutCreate(&s_Adesc, CUDA_R_8F_E4M3, K, N, K);
    cublasLtMatrixLayoutCreate(&s_Bdesc, CUDA_R_8F_E4M3, K, M, K);
    cublasLtMatrixLayoutCreate(&s_Cdesc, CUDA_R_16F, N, M, N);

    cublasLtMatmulPreferenceCreate(&pref);
    size_t maxWs = 32 * 1024 * 1024;
    cublasLtMatmulPreferenceSetAttribute(pref,
                                         CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs));

    st = cublasLtMatmulAlgoGetHeuristic(
        a->handle, s_opDesc, s_Adesc, s_Bdesc, s_Cdesc, s_Cdesc,
        pref, 1, &heur, &found);

    cublasLtMatmulPreferenceDestroy(pref);

    if (st != 0 || found == 0)
    {
        fprintf(stderr, "[FP8] heuristic: st=%d found=%d\n", st, found);
        cleanup_cached();
        return st != 0 ? (int)st : 15;
    }

    s_algo = heur.algo;
    s_wsSize = heur.workspaceSize;
    fprintf(stderr, "[FP8] setup M=%d N=%d K=%d workspace=%zu\n", M, N, K, s_wsSize);

    if (s_wsSize > 0)
        cudaMalloc(&s_workspace, s_wsSize);

    s_M = M;
    s_N = N;
    s_K = K;
    return 0;
}

int fp8_matmul_wrapper(Fp8MatmulArgs *a)
{
    // Re-setup if dimensions changed
    if (a->M != s_M || a->N != s_N || a->K != s_K)
    {
        int st = setup_cached(a);
        if (st != 0)
            return st;
    }

    // A_cublas = B_user, B_cublas = A_user (row-major swap)
    cublasStatus_t st = cublasLtMatmul(
        a->handle, s_opDesc,
        a->alpha,
        a->B, s_Adesc,
        a->A, s_Bdesc,
        a->beta,
        a->C, s_Cdesc,
        a->C, s_Cdesc,
        &s_algo,
        s_workspace, s_wsSize,
        0);

    return (int)st;
}

int cuda_device_sync(void)
{
    return (int)cudaDeviceSynchronize();
}