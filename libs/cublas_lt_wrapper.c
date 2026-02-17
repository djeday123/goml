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
    cublasLtMatmulHeuristicResult_t heurResults[16];
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

    // Try compute types in order of expected speed
    cublasComputeType_t tryCompute[] = {
        CUBLAS_COMPUTE_32F_FAST_16F,  // FP32 output, FP16 internal math
        CUBLAS_COMPUTE_32F_FAST_TF32, // FP32 output, TF32 internal
        CUBLAS_COMPUTE_32F            // FP32 baseline
    };
    cudaDataType_t tryScale[] = {CUDA_R_32F, CUDA_R_32F, CUDA_R_32F};
    const char *tryNames[] = {"32F_FAST_16F", "32F_FAST_TF32", "32F"};

    int computeIdx = -1;

    for (int ci = 0; ci < 3; ci++)
    {
        if (s_opDesc)
        {
            cublasLtMatmulDescDestroy(s_opDesc);
            s_opDesc = NULL;
        }

        st = cublasLtMatmulDescCreate(&s_opDesc, tryCompute[ci], tryScale[ci]);
        if (st != 0)
            continue;

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

        if (!s_Adesc)
        {
            cublasLtMatrixLayoutCreate(&s_Adesc, CUDA_R_8F_E4M3, K, N, K);
            cublasLtMatrixLayoutCreate(&s_Bdesc, CUDA_R_8F_E4M3, K, M, K);
            cublasLtMatrixLayoutCreate(&s_Cdesc, CUDA_R_16F, N, M, N);
        }

        cublasLtMatmulPreferenceCreate(&pref);
        size_t maxWs = 256 * 1024 * 1024;
        cublasLtMatmulPreferenceSetAttribute(pref,
                                             CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs));

        st = cublasLtMatmulAlgoGetHeuristic(
            a->handle, s_opDesc, s_Adesc, s_Bdesc, s_Cdesc, s_Cdesc,
            pref, 16, heurResults, &found);

        cublasLtMatmulPreferenceDestroy(pref);
        pref = NULL;

        if (st == 0 && found > 0)
        {
            computeIdx = ci;
            fprintf(stderr, "[FP8] %s: found %d algos\n", tryNames[ci], found);
            break;
        }
        fprintf(stderr, "[FP8] %s: no algos (st=%d)\n", tryNames[ci], st);
    }

    if (computeIdx < 0)
    {
        fprintf(stderr, "[FP8] no compute type worked!\n");
        cleanup_cached();
        return 15;
    }

    // Benchmark all returned algorithms
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

        float bestTime = 1e30f;
        float alpha = 1.0f, beta = 0.0f;

        cublasLtMatmul(a->handle, s_opDesc, &alpha,
                       a->B, s_Adesc, a->A, s_Bdesc, &beta,
                       a->C, s_Cdesc, a->C, s_Cdesc,
                       &heurResults[0].algo, tempWs, heurResults[0].workspaceSize, 0);
        cudaDeviceSynchronize();

        for (int i = 0; i < found; i++)
        {
            st = cublasLtMatmul(a->handle, s_opDesc, &alpha,
                                a->B, s_Adesc, a->A, s_Bdesc, &beta,
                                a->C, s_Cdesc, a->C, s_Cdesc,
                                &heurResults[i].algo, tempWs, heurResults[i].workspaceSize, 0);

            if (st != 0)
                continue;
            cudaDeviceSynchronize();

            cudaEvent_t t0, t1;
            cudaEventCreate(&t0);
            cudaEventCreate(&t1);

            cudaEventRecord(t0, 0);
            for (int r = 0; r < 10; r++)
            {
                cublasLtMatmul(a->handle, s_opDesc, &alpha,
                               a->B, s_Adesc, a->A, s_Bdesc, &beta,
                               a->C, s_Cdesc, a->C, s_Cdesc,
                               &heurResults[i].algo, tempWs, heurResults[i].workspaceSize, 0);
            }
            cudaEventRecord(t1, 0);
            cudaEventSynchronize(t1);

            float ms = 0;
            cudaEventElapsedTime(&ms, t0, t1);
            cudaEventDestroy(t0);
            cudaEventDestroy(t1);

            fprintf(stderr, "[FP8] algo %d/%d: %.3f ms (ws=%zu)\n", i, found, ms / 10, heurResults[i].workspaceSize);

            if (ms < bestTime)
            {
                bestTime = ms;
                bestIdx = i;
                bestWs = heurResults[i].workspaceSize;
            }
        }

        if (tempWs)
            cudaFree(tempWs);
        double flops = 2.0 * (double)M * (double)N * (double)K;
        double tflops = (flops / ((bestTime / 10) / 1000.0)) / 1e12;
        fprintf(stderr, "[FP8] best algo: %d (%.3f ms, %.1f TFLOPS)\n", bestIdx, bestTime / 10, tflops);
    }

    s_algo = heurResults[bestIdx].algo;
    s_wsSize = bestWs;
    fprintf(stderr, "[FP8] setup M=%d N=%d K=%d compute=%s algos=%d workspace=%zu\n",
            M, N, K, tryNames[computeIdx], found, s_wsSize);

    if (s_wsSize > 0)
        cudaMalloc(&s_workspace, s_wsSize);

    s_M = M;
    s_N = N;
    s_K = K;
    return 0;
}

int fp8_matmul_wrapper(Fp8MatmulArgs *a)
{
    if (a->M != s_M || a->N != s_N || a->K != s_K)
    {
        int st = setup_cached(a);
        if (st != 0)
            return st;
    }

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