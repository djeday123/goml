#include <cublasLt.h>
#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>

// Аргументы для существующего FP16 режима
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

// Расширенные аргументы для FP8 Output режима
typedef struct
{
    cublasLtHandle_t handle;
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t _pad;
    const void *A; // FP8 input
    const void *B; // FP8 input
    void *C;       // Bias (FP16)
    void *D;       // Output (FP8 или FP16)
    const float *alpha;
    const float *beta;
    const float *scaleA; // device ptr
    const float *scaleB; // device ptr
    const float *scaleD; // device ptr
    float *amaxD;        // device ptr (absmax output)
} Fp8MatmulArgsEx;

static float *d_one = NULL;

// Глобальные кэшированные дескрипторы
static cublasLtMatmulDesc_t s_opDesc = NULL;
static cublasLtMatrixLayout_t s_Adesc = NULL, s_Bdesc = NULL, s_Cdesc = NULL;
static cublasLtMatmulAlgo_t s_algo;
static void *s_workspace = NULL;
static size_t s_wsSize = 0;
static int32_t s_M = 0, s_N = 0, s_K = 0;

static cublasLtMatmulDesc_t s_opDesc8 = NULL;
static cublasLtMatrixLayout_t s_Adesc8 = NULL, s_Bdesc8 = NULL, s_Cdesc8 = NULL, s_Ddesc8 = NULL;
static cublasLtMatmulAlgo_t s_algo8;
static void *s_workspace8 = NULL;
static size_t s_wsSize8 = 0;
static int32_t s_M8 = 0, s_N8 = 0, s_K8 = 0;

static void cleanup_fp16(void)
{
    if (s_opDesc)
        cublasLtMatmulDescDestroy(s_opDesc);
    if (s_Adesc)
        cublasLtMatrixLayoutDestroy(s_Adesc);
    if (s_Bdesc)
        cublasLtMatrixLayoutDestroy(s_Bdesc);
    if (s_Cdesc)
        cublasLtMatrixLayoutDestroy(s_Cdesc);
    if (s_workspace)
        cudaFree(s_workspace);
    s_opDesc = NULL;
    s_Adesc = NULL;
    s_Bdesc = NULL;
    s_Cdesc = NULL;
    s_workspace = NULL;
    s_M = s_N = s_K = 0;
}

static void cleanup_fp8out(void)
{
    if (s_opDesc8)
        cublasLtMatmulDescDestroy(s_opDesc8);
    if (s_Adesc8)
        cublasLtMatrixLayoutDestroy(s_Adesc8);
    if (s_Bdesc8)
        cublasLtMatrixLayoutDestroy(s_Bdesc8);
    if (s_Cdesc8)
        cublasLtMatrixLayoutDestroy(s_Cdesc8);
    if (s_Ddesc8)
        cublasLtMatrixLayoutDestroy(s_Ddesc8);
    if (s_workspace8)
        cudaFree(s_workspace8);
    s_opDesc8 = NULL;
    s_Adesc8 = NULL;
    s_Bdesc8 = NULL;
    s_Cdesc8 = NULL;
    s_Ddesc8 = NULL;
    s_workspace8 = NULL;
    s_M8 = s_N8 = s_K8 = 0;
}

static int setup_fp16(Fp8MatmulArgs *a)
{
    cublasLtMatmulPreference_t pref = NULL;
    cublasLtMatmulHeuristicResult_t heurResults[16];
    int found = 0;
    cublasStatus_t st;

    cleanup_fp16();
    if (!d_one)
    {
        cudaMalloc((void **)&d_one, sizeof(float));
        float h = 1.0f;
        cudaMemcpy(d_one, &h, sizeof(float), cudaMemcpyHostToDevice);
    }

    // ТУРБО-РЕЖИМ: Используем FP16 Accumulator вместо FP32
    st = cublasLtMatmulDescCreate(&s_opDesc, CUBLAS_COMPUTE_16F, CUDA_R_16F);
    if (st != 0)
        return (int)st;

    cublasOperation_t opA = CUBLAS_OP_T;
    cublasOperation_t opB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(s_opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    cublasLtMatmulDescSetAttribute(s_opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

    const void *sp = (const void *)d_one;
    cublasLtMatmulDescSetAttribute(s_opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sp, sizeof(sp));
    cublasLtMatmulDescSetAttribute(s_opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sp, sizeof(sp));

    int8_t fastAccum = 1;
    cublasLtMatmulDescSetAttribute(s_opDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccum, sizeof(fastAccum));

    cublasLtMatrixLayoutCreate(&s_Adesc, CUDA_R_8F_E4M3, a->K, a->N, a->K);
    cublasLtMatrixLayoutCreate(&s_Bdesc, CUDA_R_8F_E4M3, a->K, a->M, a->K);
    cublasLtMatrixLayoutCreate(&s_Cdesc, CUDA_R_16F, a->N, a->M, a->N);

    cublasLtMatmulPreferenceCreate(&pref);
    size_t maxWs = 256 * 1024 * 1024;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs));

    st = cublasLtMatmulAlgoGetHeuristic(a->handle, s_opDesc, s_Adesc, s_Bdesc, s_Cdesc, s_Cdesc, pref, 16, heurResults, &found);
    cublasLtMatmulPreferenceDestroy(pref);

    if (st != 0 || found == 0)
        return st != 0 ? (int)st : 15;

    s_algo = heurResults[0].algo;
    s_wsSize = heurResults[0].workspaceSize;
    if (s_wsSize > 0)
        cudaMalloc(&s_workspace, s_wsSize);
    s_M = a->M;
    s_N = a->N;
    s_K = a->K;
    return 0;
}

static int setup_fp8out(Fp8MatmulArgsEx *a)
{
    cublasLtMatmulPreference_t pref = NULL;
    cublasLtMatmulHeuristicResult_t heurResults[16];
    int found = 0;
    cublasStatus_t st;

    cleanup_fp8out();

    // ТУРБО-РЕЖИМ: Используем FP16 Accumulator для режима с FP8 выходом
    st = cublasLtMatmulDescCreate(&s_opDesc8, CUBLAS_COMPUTE_16F, CUDA_R_16F);
    if (st != 0)
        return (int)st;

    cublasOperation_t opA = CUBLAS_OP_T;
    cublasOperation_t opB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(s_opDesc8, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    cublasLtMatmulDescSetAttribute(s_opDesc8, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

    cublasLtMatmulDescSetAttribute(s_opDesc8, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a->scaleA, sizeof(void *));
    cublasLtMatmulDescSetAttribute(s_opDesc8, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &a->scaleB, sizeof(void *));
    cublasLtMatmulDescSetAttribute(s_opDesc8, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &a->scaleD, sizeof(void *));
    cublasLtMatmulDescSetAttribute(s_opDesc8, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &a->amaxD, sizeof(void *));

    int8_t fastAccum = 1;
    cublasLtMatmulDescSetAttribute(s_opDesc8, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccum, sizeof(fastAccum));

    cublasLtMatrixLayoutCreate(&s_Adesc8, CUDA_R_8F_E4M3, a->K, a->N, a->K);
    cublasLtMatrixLayoutCreate(&s_Bdesc8, CUDA_R_8F_E4M3, a->K, a->M, a->K);
    cublasLtMatrixLayoutCreate(&s_Cdesc8, CUDA_R_16F, a->N, a->M, a->N);
    cublasLtMatrixLayoutCreate(&s_Ddesc8, CUDA_R_8F_E4M3, a->N, a->M, a->N);

    cublasLtMatmulPreferenceCreate(&pref);
    size_t maxWs = 256 * 1024 * 1024;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs));

    st = cublasLtMatmulAlgoGetHeuristic(a->handle, s_opDesc8, s_Adesc8, s_Bdesc8, s_Cdesc8, s_Ddesc8, pref, 16, heurResults, &found);
    cublasLtMatmulPreferenceDestroy(pref);

    if (st != 0 || found == 0)
        return st != 0 ? (int)st : 15;

    s_algo8 = heurResults[0].algo;
    s_wsSize8 = heurResults[0].workspaceSize;
    if (s_wsSize8 > 0)
        cudaMalloc(&s_workspace8, s_wsSize8);
    s_M8 = a->M;
    s_N8 = a->N;
    s_K8 = a->K;
    return 0;
}

int fp8_matmul_wrapper(Fp8MatmulArgs *a)
{
    if (a->M != s_M || a->N != s_N || a->K != s_K)
    {
        int st = setup_fp16(a);
        if (st != 0)
            return st;
    }
    return (int)cublasLtMatmul(a->handle, s_opDesc, a->alpha, a->B, s_Adesc, a->A, s_Bdesc, a->beta, a->C, s_Cdesc, a->C, s_Cdesc, &s_algo, s_workspace, s_wsSize, 0);
}

int fp8_matmul_fp8out(Fp8MatmulArgsEx *a)
{
    if (a->M != s_M8 || a->N != s_N8 || a->K != s_K8)
    {
        int st = setup_fp8out(a);
        if (st != 0)
            return st;
    }
    return (int)cublasLtMatmul(a->handle, s_opDesc8, a->alpha, a->B, s_Adesc8, a->A, s_Bdesc8, a->beta, a->C, s_Cdesc8, a->D, s_Ddesc8, &s_algo8, s_workspace8, s_wsSize8, 0);
}

int cuda_device_sync(void)
{
    return (int)cudaDeviceSynchronize();
}