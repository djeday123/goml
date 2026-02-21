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

// Extended args for FP8 output mode
typedef struct
{
    cublasLtHandle_t handle;
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t _pad;
    const void *A; // FP8 input
    const void *B; // FP8 input
    void *C;       // FP8 output (when fp8_output=1) or FP16
    const float *alpha;
    const float *beta;
    const float *scaleA; // device ptr: scale for A
    const float *scaleB; // device ptr: scale for B
    float *scaleD;       // device ptr: scale for output D
    float *amaxD;        // device ptr: cuBLASLt writes absmax of D here
} Fp8MatmulArgsEx;

static float *d_one = NULL;

// === Config 1: FP16 output (existing) ===
static cublasLtMatmulDesc_t s_opDesc = NULL;
static cublasLtMatrixLayout_t s_Adesc = NULL, s_Bdesc = NULL, s_Cdesc = NULL;
static cublasLtMatmulAlgo_t s_algo;
static void *s_workspace = NULL;
static size_t s_wsSize = 0;
static int32_t s_M = 0, s_N = 0, s_K = 0;

// === Config 2: FP8 output ===
static cublasLtMatmulDesc_t s_opDesc8 = NULL;
static cublasLtMatrixLayout_t s_Adesc8 = NULL, s_Bdesc8 = NULL, s_Cdesc8 = NULL, s_Ddesc8 = NULL;
static cublasLtMatmulAlgo_t s_algo8;
static void *s_workspace8 = NULL;
static size_t s_wsSize8 = 0;
static int32_t s_M8 = 0, s_N8 = 0, s_K8 = 0;

static void cleanup_fp16(void)
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

static void cleanup_fp8out(void)
{
    if (s_opDesc8)
    {
        cublasLtMatmulDescDestroy(s_opDesc8);
        s_opDesc8 = NULL;
    }
    if (s_Adesc8)
    {
        cublasLtMatrixLayoutDestroy(s_Adesc8);
        s_Adesc8 = NULL;
    }
    if (s_Bdesc8)
    {
        cublasLtMatrixLayoutDestroy(s_Bdesc8);
        s_Bdesc8 = NULL;
    }
    if (s_Cdesc8)
    {
        cublasLtMatrixLayoutDestroy(s_Cdesc8);
        s_Cdesc8 = NULL;
    }
    if (s_Ddesc8)
    {
        cublasLtMatrixLayoutDestroy(s_Ddesc8);
        s_Ddesc8 = NULL;
    }
    if (s_workspace8)
    {
        cudaFree(s_workspace8);
        s_workspace8 = NULL;
    }
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

    int M = a->M, N = a->N, K = a->K;

    st = cublasLtMatmulDescCreate(&s_opDesc, CUBLAS_COMPUTE_32F_FAST_16F, CUDA_R_32F);
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

    int8_t fastAccum = 1;
    cublasLtMatmulDescSetAttribute(s_opDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccum, sizeof(fastAccum));

    cublasLtMatrixLayoutCreate(&s_Adesc, CUDA_R_8F_E4M3, K, N, K);
    cublasLtMatrixLayoutCreate(&s_Bdesc, CUDA_R_8F_E4M3, K, M, K);
    cublasLtMatrixLayoutCreate(&s_Cdesc, CUDA_R_16F, N, M, N);

    cublasLtMatmulPreferenceCreate(&pref);
    size_t maxWs = 256 * 1024 * 1024;
    cublasLtMatmulPreferenceSetAttribute(pref,
                                         CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs));

    st = cublasLtMatmulAlgoGetHeuristic(
        a->handle, s_opDesc, s_Adesc, s_Bdesc, s_Cdesc, s_Cdesc,
        pref, 16, heurResults, &found);
    cublasLtMatmulPreferenceDestroy(pref);

    if (st != 0 || found == 0)
    {
        fprintf(stderr, "[FP8->FP16] heuristic: st=%d found=%d\n", st, found);
        cleanup_fp16();
        return st != 0 ? (int)st : 15;
    }

    // Benchmark algos
    size_t bestWs = heurResults[0].workspaceSize;
    int bestIdx = 0;
    if (found > 1)
    {
        size_t tempMaxWs = 0;
        for (int i = 0; i < found; i++)
            if (heurResults[i].workspaceSize > tempMaxWs)
                tempMaxWs = heurResults[i].workspaceSize;
        void *tempWs = NULL;
        if (tempMaxWs > 0)
            cudaMalloc(&tempWs, tempMaxWs);
        float bestTime = 1e30f, alpha = 1.0f, beta = 0.0f;
        cublasLtMatmul(a->handle, s_opDesc, &alpha, a->B, s_Adesc, a->A, s_Bdesc, &beta,
                       a->C, s_Cdesc, a->C, s_Cdesc, &heurResults[0].algo, tempWs, heurResults[0].workspaceSize, 0);
        cudaDeviceSynchronize();
        for (int i = 0; i < found; i++)
        {
            st = cublasLtMatmul(a->handle, s_opDesc, &alpha, a->B, s_Adesc, a->A, s_Bdesc, &beta,
                                a->C, s_Cdesc, a->C, s_Cdesc, &heurResults[i].algo, tempWs, heurResults[i].workspaceSize, 0);
            if (st != 0)
                continue;
            cudaDeviceSynchronize();
            cudaEvent_t t0, t1;
            cudaEventCreate(&t0);
            cudaEventCreate(&t1);
            cudaEventRecord(t0, 0);
            for (int r = 0; r < 10; r++)
                cublasLtMatmul(a->handle, s_opDesc, &alpha, a->B, s_Adesc, a->A, s_Bdesc, &beta,
                               a->C, s_Cdesc, a->C, s_Cdesc, &heurResults[i].algo, tempWs, heurResults[i].workspaceSize, 0);
            cudaEventRecord(t1, 0);
            cudaEventSynchronize(t1);
            float ms;
            cudaEventElapsedTime(&ms, t0, t1);
            cudaEventDestroy(t0);
            cudaEventDestroy(t1);
            if (ms < bestTime)
            {
                bestTime = ms;
                bestIdx = i;
                bestWs = heurResults[i].workspaceSize;
            }
        }
        if (tempWs)
            cudaFree(tempWs);
    }

    s_algo = heurResults[bestIdx].algo;
    s_wsSize = bestWs;
    if (s_wsSize > 0)
        cudaMalloc(&s_workspace, s_wsSize);
    s_M = M;
    s_N = N;
    s_K = K;
    fprintf(stderr, "[FP8->FP16] setup M=%d N=%d K=%d algos=%d\n", M, N, K, found);
    return 0;
}

static int setup_fp8out(Fp8MatmulArgsEx *a)
{
    cublasLtMatmulPreference_t pref = NULL;
    cublasLtMatmulHeuristicResult_t heurResults[16];
    int found = 0;
    cublasStatus_t st;

    cleanup_fp8out();

    if (!d_one)
    {
        cudaMalloc((void **)&d_one, sizeof(float));
        float h = 1.0f;
        cudaMemcpy(d_one, &h, sizeof(float), cudaMemcpyHostToDevice);
    }

    int M = a->M, N = a->N, K = a->K;

    st = cublasLtMatmulDescCreate(&s_opDesc8, CUBLAS_COMPUTE_32F_FAST_16F, CUDA_R_32F);
    if (st != 0)
        return (int)st;

    cublasOperation_t opA = CUBLAS_OP_T;
    cublasOperation_t opB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(s_opDesc8, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    cublasLtMatmulDescSetAttribute(s_opDesc8, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

    // Scale pointers
    cublasLtMatmulDescSetAttribute(s_opDesc8, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a->scaleA, sizeof(void *));
    cublasLtMatmulDescSetAttribute(s_opDesc8, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &a->scaleB, sizeof(void *));
    cublasLtMatmulDescSetAttribute(s_opDesc8, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &a->scaleD, sizeof(void *));
    cublasLtMatmulDescSetAttribute(s_opDesc8, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &a->amaxD, sizeof(void *));

    int8_t fastAccum = 1;
    cublasLtMatmulDescSetAttribute(s_opDesc8, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccum, sizeof(fastAccum));

    // A, B: FP8 E4M3. C (bias): FP16. D (output): FP8 E4M3
    cublasLtMatrixLayoutCreate(&s_Adesc8, CUDA_R_8F_E4M3, K, N, K);
    cublasLtMatrixLayoutCreate(&s_Bdesc8, CUDA_R_8F_E4M3, K, M, K);
    cublasLtMatrixLayoutCreate(&s_Cdesc8, CUDA_R_16F, N, M, N);     // C (bias) stays FP16
    cublasLtMatrixLayoutCreate(&s_Ddesc8, CUDA_R_8F_E4M3, N, M, N); // D output is FP8

    cublasLtMatmulPreferenceCreate(&pref);
    size_t maxWs = 256 * 1024 * 1024;
    cublasLtMatmulPreferenceSetAttribute(pref,
                                         CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs));

    st = cublasLtMatmulAlgoGetHeuristic(
        a->handle, s_opDesc8, s_Adesc8, s_Bdesc8, s_Cdesc8, s_Ddesc8,
        pref, 16, heurResults, &found);
    cublasLtMatmulPreferenceDestroy(pref);

    if (st != 0 || found == 0)
    {
        fprintf(stderr, "[FP8->FP8] heuristic: st=%d found=%d\n", st, found);
        cleanup_fp8out();
        return st != 0 ? (int)st : 15;
    }

    // Pick first algo (skip benchmark for setup speed)
    s_algo8 = heurResults[0].algo;
    s_wsSize8 = heurResults[0].workspaceSize;
    if (s_wsSize8 > 0)
        cudaMalloc(&s_workspace8, s_wsSize8);
    s_M8 = M;
    s_N8 = N;
    s_K8 = K;
    fprintf(stderr, "[FP8->FP8] setup M=%d N=%d K=%d algos=%d ws=%zu\n", M, N, K, found, s_wsSize8);
    return 0;
}

// === FP16 output (existing API) ===
int fp8_matmul_wrapper(Fp8MatmulArgs *a)
{
    if (a->M != s_M || a->N != s_N || a->K != s_K)
    {
        int st = setup_fp16(a);
        if (st != 0)
            return st;
    }
    cublasStatus_t st = cublasLtMatmul(
        a->handle, s_opDesc, a->alpha,
        a->B, s_Adesc, a->A, s_Bdesc,
        a->beta, a->C, s_Cdesc, a->C, s_Cdesc,
        &s_algo, s_workspace, s_wsSize, 0);
    return (int)st;
}

// === FP8 output (new API) ===
int fp8_matmul_fp8out(Fp8MatmulArgsEx *a)
{
    if (a->M != s_M8 || a->N != s_N8 || a->K != s_K8)
    {
        int st = setup_fp8out(a);
        if (st != 0)
            return st;
    }
    cublasStatus_t st = cublasLtMatmul(
        a->handle, s_opDesc8, a->alpha,
        a->B, s_Adesc8, a->A, s_Bdesc8,
        a->beta, a->C, s_Cdesc8, a->C, s_Ddesc8,
        &s_algo8, s_workspace8, s_wsSize8, 0);
    return (int)st;
}

// Добавь в libs/cublas_lt_wrapper_v2.c после существующих структур:

// Fused epilogue args
typedef struct
{
    cublasLtHandle_t handle;
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t epilogue; // 0=none, 1=relu, 2=gelu, 3=bias, 4=relu+bias, 5=gelu+bias
    const void *A;
    const void *B;
    void *C;
    const float *alpha;
    const float *beta;
    const void *bias; // device ptr, FP16, size N (or M depending on layout)
} Fp8FusedArgs;

static cublasLtMatmulDesc_t s_fDesc = NULL;
static cublasLtMatrixLayout_t s_fAdesc = NULL, s_fBdesc = NULL, s_fCdesc = NULL;
static cublasLtMatmulAlgo_t s_fAlgo;
static void *s_fWorkspace = NULL;
static size_t s_fWsSize = 0;
static int32_t s_fM = 0, s_fN = 0, s_fK = 0, s_fEpi = -1;

static void cleanup_fused(void)
{
    if (s_fDesc)
    {
        cublasLtMatmulDescDestroy(s_fDesc);
        s_fDesc = NULL;
    }
    if (s_fAdesc)
    {
        cublasLtMatrixLayoutDestroy(s_fAdesc);
        s_fAdesc = NULL;
    }
    if (s_fBdesc)
    {
        cublasLtMatrixLayoutDestroy(s_fBdesc);
        s_fBdesc = NULL;
    }
    if (s_fCdesc)
    {
        cublasLtMatrixLayoutDestroy(s_fCdesc);
        s_fCdesc = NULL;
    }
    if (s_fWorkspace)
    {
        cudaFree(s_fWorkspace);
        s_fWorkspace = NULL;
    }
    s_fM = s_fN = s_fK = 0;
    s_fEpi = -1;
}

static cublasLtEpilogue_t get_epilogue(int epi)
{
    switch (epi)
    {
    case 1:
        return CUBLASLT_EPILOGUE_RELU;
    case 2:
        return CUBLASLT_EPILOGUE_GELU;
    case 3:
        return CUBLASLT_EPILOGUE_BIAS;
    case 4:
        return CUBLASLT_EPILOGUE_RELU_BIAS;
    case 5:
        return CUBLASLT_EPILOGUE_GELU_BIAS;
    default:
        return CUBLASLT_EPILOGUE_DEFAULT;
    }
}

static int setup_fused(Fp8FusedArgs *a)
{
    cublasLtMatmulPreference_t pref = NULL;
    cublasLtMatmulHeuristicResult_t heurResults[16];
    int found = 0;
    cublasStatus_t st;

    cleanup_fused();

    if (!d_one)
    {
        cudaMalloc((void **)&d_one, sizeof(float));
        float h = 1.0f;
        cudaMemcpy(d_one, &h, sizeof(float), cudaMemcpyHostToDevice);
    }

    int M = a->M, N = a->N, K = a->K;

    st = cublasLtMatmulDescCreate(&s_fDesc, CUBLAS_COMPUTE_32F_FAST_16F, CUDA_R_32F);
    if (st != 0)
        return (int)st;

    cublasOperation_t opA = CUBLAS_OP_T;
    cublasOperation_t opB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(s_fDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    cublasLtMatmulDescSetAttribute(s_fDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

    const void *sp = (const void *)d_one;
    cublasLtMatmulDescSetAttribute(s_fDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sp, sizeof(sp));
    cublasLtMatmulDescSetAttribute(s_fDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sp, sizeof(sp));
    cublasLtMatmulDescSetAttribute(s_fDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &sp, sizeof(sp));

    int8_t fastAccum = 1;
    cublasLtMatmulDescSetAttribute(s_fDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccum, sizeof(fastAccum));

    // Set epilogue
    cublasLtEpilogue_t epilogue = get_epilogue(a->epilogue);
    cublasLtMatmulDescSetAttribute(s_fDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));

    // Set bias pointer if needed
    if (a->epilogue >= 3 && a->bias != NULL)
    {
        cublasLtMatmulDescSetAttribute(s_fDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &a->bias, sizeof(void *));
        cudaDataType_t biasType = CUDA_R_16F;
        cublasLtMatmulDescSetAttribute(s_fDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &biasType, sizeof(biasType));
    }

    cublasLtMatrixLayoutCreate(&s_fAdesc, CUDA_R_8F_E4M3, K, N, K);
    cublasLtMatrixLayoutCreate(&s_fBdesc, CUDA_R_8F_E4M3, K, M, K);
    cublasLtMatrixLayoutCreate(&s_fCdesc, CUDA_R_16F, N, M, N);

    cublasLtMatmulPreferenceCreate(&pref);
    size_t maxWs = 256 * 1024 * 1024;
    cublasLtMatmulPreferenceSetAttribute(pref,
                                         CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs));

    st = cublasLtMatmulAlgoGetHeuristic(
        a->handle, s_fDesc, s_fAdesc, s_fBdesc, s_fCdesc, s_fCdesc,
        pref, 16, heurResults, &found);
    cublasLtMatmulPreferenceDestroy(pref);

    if (st != 0 || found == 0)
    {
        fprintf(stderr, "[FP8-fused] heuristic: st=%d found=%d epi=%d\n", st, found, a->epilogue);
        cleanup_fused();
        return st != 0 ? (int)st : 15;
    }

    s_fAlgo = heurResults[0].algo;
    s_fWsSize = heurResults[0].workspaceSize;
    if (s_fWsSize > 0)
        cudaMalloc(&s_fWorkspace, s_fWsSize);
    s_fM = M;
    s_fN = N;
    s_fK = K;
    s_fEpi = a->epilogue;
    fprintf(stderr, "[FP8-fused] setup M=%d N=%d K=%d epi=%d algos=%d\n", M, N, K, a->epilogue, found);
    return 0;
}

int fp8_matmul_fused(Fp8FusedArgs *a)
{
    if (a->M != s_fM || a->N != s_fN || a->K != s_fK || a->epilogue != s_fEpi)
    {
        int st = setup_fused(a);
        if (st != 0)
            return st;
    }
    cublasStatus_t st = cublasLtMatmul(
        a->handle, s_fDesc, a->alpha,
        a->B, s_fAdesc, a->A, s_fBdesc,
        a->beta, a->C, s_fCdesc, a->C, s_fCdesc,
        &s_fAlgo, s_fWorkspace, s_fWsSize, 0);
    return (int)st;
}

int cuda_device_sync(void)
{
    return (int)cudaDeviceSynchronize();
}