// =============================================================================
// FlashAttention v2 Diagnostic — isolate QK^T, softmax, P@V
// =============================================================================
// Tests each phase independently to find where the bug is.
//
// Build: nvcc -O3 -arch=sm_89 -std=c++17 \
//        libs/flash_attention_v2.cu libs/flash_attention.cu libs/transformer_kernels.cu \
//        libs/fa2_diag.cu -o runs/fa2_diag -lcudart
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C"
{
    int flash_attention_v2_forward(
        const void *Q, const void *K, const void *V, void *O,
        int total_heads, int seq_len, int head_dim,
        int causal, void *stream);
}

#define CK(c)                                                                               \
    do                                                                                      \
    {                                                                                       \
        cudaError_t e = (c);                                                                \
        if (e != cudaSuccess)                                                               \
        {                                                                                   \
            fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(1);                                                                        \
        }                                                                                   \
    } while (0)

static float h2f(uint16_t h)
{
    __half hv;
    memcpy(&hv, &h, 2);
    return __half2float(hv);
}
static uint16_t f2h(float f)
{
    __half hv = __float2half(f);
    uint16_t r;
    memcpy(&r, &hv, 2);
    return r;
}

// =============================================================================
// Test 1: Identity attention (Q=K=I, V=known, no causal)
// If Q[i] = K[i] = one-hot(i), then S = I, softmax(I/scale) should concentrate
// on diagonal, O ≈ V (with some softmax spreading)
// =============================================================================
void test_identity()
{
    printf("=== Test 1: Identity Q=K ===\n");
    int heads = 1, seq = 16, dim = 128;
    int n = heads * seq * dim;

    uint16_t *hQ = (uint16_t *)calloc(n, 2);
    uint16_t *hK = (uint16_t *)calloc(n, 2);
    uint16_t *hV = (uint16_t *)calloc(n, 2);
    uint16_t *hO = (uint16_t *)calloc(n, 2);

    // Q[i][i] = 1.0, K[i][i] = 1.0 (one-hot rows, truncated to dim)
    // This makes S[i][j] = Q[i]·K[j] = delta(i,j)
    for (int i = 0; i < seq; i++)
    {
        if (i < dim)
        {
            hQ[i * dim + i] = f2h(1.0f);
            hK[i * dim + i] = f2h(1.0f);
        }
    }
    // V = sequential values for easy checking
    for (int i = 0; i < seq; i++)
        for (int d = 0; d < dim; d++)
            hV[i * dim + d] = f2h((float)(i * dim + d) / (seq * dim));

    void *dQ, *dK, *dV, *dO;
    CK(cudaMalloc(&dQ, n * 2));
    CK(cudaMalloc(&dK, n * 2));
    CK(cudaMalloc(&dV, n * 2));
    CK(cudaMalloc(&dO, n * 2));
    CK(cudaMemcpy(dQ, hQ, n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, hK, n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, hV, n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemset(dO, 0, n * 2));

    int rc = flash_attention_v2_forward(dQ, dK, dV, dO, heads, seq, dim, 0, nullptr);
    CK(cudaDeviceSynchronize());
    printf("  launch rc=%d\n", rc);

    CK(cudaMemcpy(hO, dO, n * 2, cudaMemcpyDeviceToHost));

    // Without causal mask and with identity S:
    // S[i][j] = delta(i,j) (before scale)
    // After scale: S[i][i] = 1/sqrt(128) ≈ 0.088, S[i][j!=i] = 0
    // softmax: P[i][i] = exp(0.088) / (exp(0.088) + 15*exp(0)) ≈ 1.092 / 16.092 ≈ 0.068
    //          P[i][j] = 1/16.092 ≈ 0.062 for j != i
    // O[i] ≈ 0.068 * V[i] + 0.062 * sum(V[j], j!=i)
    // Almost uniform averaging with slight bias toward V[i]

    // Print first row of O
    printf("  O[0][0:8] = ");
    for (int d = 0; d < 8; d++)
        printf("%.4f ", h2f(hO[d]));
    printf("\n");

    // Compute CPU reference
    float scale = 1.0f / sqrtf(128.0f);
    for (int i = 0; i < 2; i++)
    {
        float scores[16];
        float mx = -1e30f;
        for (int j = 0; j < seq; j++)
        {
            float dot = 0;
            for (int d = 0; d < dim; d++)
                dot += h2f(hQ[i * dim + d]) * h2f(hK[j * dim + d]);
            scores[j] = dot * scale;
            if (scores[j] > mx)
                mx = scores[j];
        }
        float sum = 0;
        for (int j = 0; j < seq; j++)
        {
            scores[j] = expf(scores[j] - mx);
            sum += scores[j];
        }
        float ref[8];
        for (int d = 0; d < 8; d++)
        {
            ref[d] = 0;
            for (int j = 0; j < seq; j++)
                ref[d] += (scores[j] / sum) * h2f(hV[j * dim + d]);
        }
        printf("  ref[%d][0:8] = ", i);
        for (int d = 0; d < 8; d++)
            printf("%.4f ", ref[d]);
        printf("\n");
    }

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
    free(hQ);
    free(hK);
    free(hV);
    free(hO);
}

// =============================================================================
// Test 2: Uniform Q=K=1 (all scores equal), V = row index
// S[i][j] = dim * scale = sqrt(dim). All rows same after softmax → O = mean(V)
// =============================================================================
void test_uniform()
{
    printf("\n=== Test 2: Uniform Q=K=1, no causal ===\n");
    int heads = 1, seq = 16, dim = 128;
    int n = heads * seq * dim;

    uint16_t *hQ = (uint16_t *)malloc(n * 2);
    uint16_t *hK = (uint16_t *)malloc(n * 2);
    uint16_t *hV = (uint16_t *)malloc(n * 2);
    uint16_t *hO = (uint16_t *)calloc(n, 2);

    for (int i = 0; i < n; i++)
        hQ[i] = f2h(1.0f);
    for (int i = 0; i < n; i++)
        hK[i] = f2h(1.0f);
    for (int i = 0; i < seq; i++)
        for (int d = 0; d < dim; d++)
            hV[i * dim + d] = f2h((float)i); // V[i][*] = i

    void *dQ, *dK, *dV, *dO;
    CK(cudaMalloc(&dQ, n * 2));
    CK(cudaMalloc(&dK, n * 2));
    CK(cudaMalloc(&dV, n * 2));
    CK(cudaMalloc(&dO, n * 2));
    CK(cudaMemcpy(dQ, hQ, n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, hK, n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, hV, n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemset(dO, 0, n * 2));

    int rc = flash_attention_v2_forward(dQ, dK, dV, dO, heads, seq, dim, 0, nullptr);
    CK(cudaDeviceSynchronize());
    printf("  launch rc=%d\n", rc);

    CK(cudaMemcpy(hO, dO, n * 2, cudaMemcpyDeviceToHost));

    // All S[i][j] = 128 * (1/sqrt(128)) = sqrt(128) ≈ 11.31
    // After softmax: P[i][j] = 1/16 for all (equal scores, 16 valid positions)
    // O[i][d] = mean(V[*][d]) = mean(0,1,...,15) = 7.5 for all i,d

    printf("  O[0][0] = %.4f (expect ~7.5)\n", h2f(hO[0]));
    printf("  O[0][1] = %.4f (expect ~7.5)\n", h2f(hO[1]));
    printf("  O[8][0] = %.4f (expect ~7.5)\n", h2f(hO[8 * dim]));
    printf("  O[15][0] = %.4f (expect ~7.5)\n", h2f(hO[15 * dim]));

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
    free(hQ);
    free(hK);
    free(hV);
    free(hO);
}

// =============================================================================
// Test 3: Small random with causal, dump full S, P, O for row 0
// =============================================================================
void test_small_dump()
{
    printf("\n=== Test 3: 1h×4s×128d causal, full dump ===\n");
    // Use only 4 seq positions for easy manual verification
    int heads = 1, seq = 4, dim = 128;
    int n = heads * seq * dim;

    uint16_t *hQ = (uint16_t *)malloc(n * 2);
    uint16_t *hK = (uint16_t *)malloc(n * 2);
    uint16_t *hV = (uint16_t *)malloc(n * 2);
    uint16_t *hO = (uint16_t *)calloc(n, 2);

    srand(42);
    for (int i = 0; i < n; i++)
    {
        hQ[i] = f2h(((float)(rand() % 201) - 100.0f) / 100.0f);
        hK[i] = f2h(((float)(rand() % 201) - 100.0f) / 100.0f);
        hV[i] = f2h(((float)(rand() % 201) - 100.0f) / 100.0f);
    }

    // CPU reference
    float scale = 1.0f / sqrtf(128.0f);
    printf("  scale = %.6f\n", scale);

    printf("  CPU S (scaled):\n");
    for (int i = 0; i < seq; i++)
    {
        printf("    row %d: ", i);
        for (int j = 0; j < seq; j++)
        {
            float dot = 0;
            for (int d = 0; d < dim; d++)
                dot += h2f(hQ[i * dim + d]) * h2f(hK[j * dim + d]);
            float s = dot * scale;
            if (j > i)
                s = -1e30f; // causal
            printf("%.4f ", s);
        }
        printf("\n");
    }

    printf("  CPU P (softmax):\n");
    float P_cpu[4][4];
    for (int i = 0; i < seq; i++)
    {
        float scores[4], mx = -1e30f;
        for (int j = 0; j <= i; j++)
        {
            float dot = 0;
            for (int d = 0; d < dim; d++)
                dot += h2f(hQ[i * dim + d]) * h2f(hK[j * dim + d]);
            scores[j] = dot * scale;
            if (scores[j] > mx)
                mx = scores[j];
        }
        float sum = 0;
        printf("    row %d: ", i);
        for (int j = 0; j <= i; j++)
        {
            scores[j] = expf(scores[j] - mx);
            sum += scores[j];
        }
        for (int j = 0; j <= i; j++)
        {
            P_cpu[i][j] = scores[j] / sum;
            printf("%.4f ", P_cpu[i][j]);
        }
        printf("\n");
    }

    // CPU O
    float O_cpu[4][8];
    for (int i = 0; i < seq; i++)
        for (int d = 0; d < 8; d++)
        {
            O_cpu[i][d] = 0;
            for (int j = 0; j <= i; j++)
                O_cpu[i][d] += P_cpu[i][j] * h2f(hV[j * dim + d]);
        }
    printf("  CPU O[0][0:8]: ");
    for (int d = 0; d < 8; d++)
        printf("%.4f ", O_cpu[0][d]);
    printf("\n");
    printf("  CPU O[3][0:8]: ");
    for (int d = 0; d < 8; d++)
        printf("%.4f ", O_cpu[3][d]);
    printf("\n");

    // GPU
    void *dQ, *dK, *dV, *dO;
    CK(cudaMalloc(&dQ, n * 2));
    CK(cudaMalloc(&dK, n * 2));
    CK(cudaMalloc(&dV, n * 2));
    CK(cudaMalloc(&dO, n * 2));
    CK(cudaMemcpy(dQ, hQ, n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, hK, n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, hV, n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemset(dO, 0, n * 2));

    int rc = flash_attention_v2_forward(dQ, dK, dV, dO, heads, seq, dim, 1, nullptr);
    CK(cudaDeviceSynchronize());
    printf("  GPU launch rc=%d\n", rc);

    CK(cudaMemcpy(hO, dO, n * 2, cudaMemcpyDeviceToHost));

    printf("  GPU O[0][0:8]: ");
    for (int d = 0; d < 8; d++)
        printf("%.4f ", h2f(hO[d]));
    printf("\n");
    printf("  GPU O[3][0:8]: ");
    for (int d = 0; d < 8; d++)
        printf("%.4f ", h2f(hO[3 * dim + d]));
    printf("\n");

    // Detailed error for row 0
    float max_err = 0;
    for (int d = 0; d < dim; d++)
    {
        float got = h2f(hO[d]);
        float ref = 0;
        for (int j = 0; j < 1; j++)             // row 0, causal: only j=0
            ref += 1.0f * h2f(hV[j * dim + d]); // P[0][0] = 1.0
        float err = fabsf(got - ref);
        if (err > max_err)
            max_err = err;
    }
    printf("  Row 0 max_err vs V[0]: %.6f (row 0 causal → O[0]=V[0])\n", max_err);

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
    free(hQ);
    free(hK);
    free(hV);
    free(hO);
}

// =============================================================================
// Test 4: V = I, P = uniform → O should be uniform
// Skip QK^T by making Q=K=constant (uniform scores)
// With causal: P[i][j] = 1/(i+1) for j<=i, 0 otherwise
// O[i] = sum(V[j]/(i+1), j<=i) = sum(one_hot(j)/(i+1), j<=i)
// =============================================================================
void test_v_identity()
{
    printf("\n=== Test 4: Q=K=1 causal, V=sequential ===\n");
    int heads = 1, seq = 8, dim = 128;
    int n = heads * seq * dim;

    uint16_t *hQ = (uint16_t *)malloc(n * 2);
    uint16_t *hK = (uint16_t *)malloc(n * 2);
    uint16_t *hV = (uint16_t *)calloc(n, 2);
    uint16_t *hO = (uint16_t *)calloc(n, 2);

    // Q = K = 0.1 (small to avoid overflow)
    for (int i = 0; i < n; i++)
        hQ[i] = f2h(0.1f);
    for (int i = 0; i < n; i++)
        hK[i] = f2h(0.1f);
    // V[i] = one-hot at position i (if i < dim)
    for (int i = 0; i < seq && i < dim; i++)
        hV[i * dim + i] = f2h(1.0f);

    void *dQ, *dK, *dV, *dO;
    CK(cudaMalloc(&dQ, n * 2));
    CK(cudaMalloc(&dK, n * 2));
    CK(cudaMalloc(&dV, n * 2));
    CK(cudaMalloc(&dO, n * 2));
    CK(cudaMemcpy(dQ, hQ, n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, hK, n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, hV, n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemset(dO, 0, n * 2));

    int rc = flash_attention_v2_forward(dQ, dK, dV, dO, heads, seq, dim, 1, nullptr);
    CK(cudaDeviceSynchronize());
    printf("  launch rc=%d\n", rc);

    CK(cudaMemcpy(hO, dO, n * 2, cudaMemcpyDeviceToHost));

    // With causal + uniform scores:
    // P[i][j] = 1/(i+1) for j <= i
    // O[i][d] = sum(P[i][j] * V[j][d], j<=i) = V[d_row][d] * (1/(i+1)) if d < seq
    // O[0] = V[0] = [1, 0, 0, ...]
    // O[1] = 0.5 * V[0] + 0.5 * V[1] = [0.5, 0.5, 0, ...]
    // O[7] = 0.125 * each → [0.125, 0.125, ..., 0.125, 0, ...]

    for (int i = 0; i < seq; i++)
    {
        float expected = 1.0f / (i + 1);
        float got_diag = h2f(hO[i * dim + 0]);                                    // O[i][0]
        float got_self = (i < dim) ? h2f(hO[i * dim + i]) : 0;                    // O[i][i]
        float got_past = (i > 0 && i + 1 < dim) ? h2f(hO[i * dim + (i + 1)]) : 0; // O[i][i+1] should be 0
        printf("  O[%d][0]=%.4f O[%d][%d]=%.4f O[%d][%d]=%.4f  (expect=%.4f)\n",
               i, got_diag, i, i, got_self, i, i + 1, got_past, expected);
    }

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
    free(hQ);
    free(hK);
    free(hV);
    free(hO);
}

int main()
{
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("GPU: %s\n\n", p.name);

    test_identity();
    test_uniform();
    test_small_dump();
    test_v_identity();

    return 0;
}
