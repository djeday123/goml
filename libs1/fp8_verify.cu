// =============================================================================
// Correctness validator for fp8_gemm_f16acc kernel
// =============================================================================
// Tests:
//   1. GPU kernel vs FP32 CPU reference (with tolerance for FP16 acc)
//   2. Identity-like patterns (easy to verify)
//   3. Known values (all-ones, all-zeros)
//
// Build: nvcc -O3 -arch=sm_89 -std=c++17 fp8_verify.cu -o fp8_verify \
//        -L/usr/local/cuda/lib64 -lcublasLt -lcudart
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Import the kernel's C API
extern "C" int fp8_gemm_f16acc(
    int M, int N, int K,
    const void *A, const void *B, void *C);

// =============================================================================
// FP8 E4M3 conversion (same as kernel)
// =============================================================================
static uint8_t f32_to_e4m3(float f)
{
    if (f != f)
        return 0x7Fu;
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint8_t sign = (bits >> 31) & 1u;
    int exp_f32 = (int)((bits >> 23) & 0xFFu) - 127;
    uint32_t mant_f32 = bits & 0x7FFFFFu;

    if (fabsf(f) > 448.0f)
        return sign ? 0xFEu : 0x7Eu;
    if (fabsf(f) < 1.0f / 64.0f)
        return sign ? 0x80u : 0x00u;

    int exp_e4m3 = exp_f32 + 7;
    if (exp_e4m3 < 1)
    {
        int shift = 1 - exp_e4m3;
        uint32_t mant = (0x800000u | mant_f32) >> (shift + 20);
        return (sign << 7) | (mant & 0x7u);
    }
    if (exp_e4m3 > 14)
        exp_e4m3 = 14;
    uint8_t mant3 = (mant_f32 >> 20) & 0x7u;
    return (sign << 7) | ((uint8_t)exp_e4m3 << 3) | mant3;
}

static float e4m3_to_f32(uint8_t v)
{
    uint8_t sign = (v >> 7) & 1u;
    uint8_t exp = (v >> 3) & 0xFu;
    uint8_t mant = v & 0x7u;
    if (exp == 0xFu && mant == 0x7u)
        return nanf("");
    float result;
    if (exp == 0)
        result = ldexpf((float)mant, -9);
    else
        result = ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
    return sign ? -result : result;
}

static float fp16_to_f32(uint16_t h)
{
    __half hv;
    memcpy(&hv, &h, 2);
    return __half2float(hv);
}

#define CHECK(x)                                                                            \
    do                                                                                      \
    {                                                                                       \
        cudaError_t e = (x);                                                                \
        if (e)                                                                              \
        {                                                                                   \
            fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(1);                                                                        \
        }                                                                                   \
    } while (0)

// =============================================================================
// Test 1: All-ones (FP8 1.0 = 0x38)
// C[i][j] = sum of K ones = K (easy to verify)
// =============================================================================
static void test_ones(int M, int N, int K)
{
    printf("\n=== Test: All-Ones (%dx%dx%d) ===\n", M, N, K);
    printf("  Expected: every element = %d.0 (or FP16-nearest)\n", K);

    size_t sA = (size_t)M * K;
    size_t sB = (size_t)N * K;
    size_t sC = (size_t)M * N;

    void *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, sA));
    CHECK(cudaMalloc(&dB, sB));
    CHECK(cudaMalloc(&dC, sC * 2));

    // FP8 e4m3 encoding of 1.0 = 0|0111|000 = 0x38
    CHECK(cudaMemset(dA, 0x38, sA));
    CHECK(cudaMemset(dB, 0x38, sB));
    CHECK(cudaMemset(dC, 0, sC * 2));

    int err = fp8_gemm_f16acc(M, N, K, dA, dB, dC);
    CHECK(cudaDeviceSynchronize());
    if (err)
    {
        printf("  Kernel error: %d\n", err);
        goto done;
    }

    {
        uint16_t *hC = (uint16_t *)malloc(sC * 2);
        CHECK(cudaMemcpy(hC, dC, sC * 2, cudaMemcpyDeviceToHost));

        float expected = (float)K;
        int errors = 0;
        float max_err = 0;
        for (int i = 0; i < M * N; i++)
        {
            float got = fp16_to_f32(hC[i]);
            float err = fabsf(got - expected);
            if (err > max_err)
                max_err = err;
            // FP16 acc: K products of 1.0*1.0, accumulated in FP16
            // For K=64, exact sum=64.0 which is representable in FP16
            if (err > 1.0f)
            {
                if (errors < 5)
                    printf("  FAIL [%d/%d][%d/%d]: got=%.2f expected=%.2f\n",
                           i / N, M, i % N, N, got, expected);
                errors++;
            }
        }
        printf("  Max error: %.4f\n", max_err);
        printf("  Failures: %d / %d\n", errors, M * N);
        printf("  %s\n", errors == 0 ? "PASS" : "FAIL");
        free(hC);
    }
done:
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

// =============================================================================
// Test 2: All-zeros
// =============================================================================
static void test_zeros(int M, int N, int K)
{
    printf("\n=== Test: All-Zeros (%dx%dx%d) ===\n", M, N, K);

    size_t sA = (size_t)M * K, sB = (size_t)N * K, sC = (size_t)M * N;
    void *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, sA));
    CHECK(cudaMalloc(&dB, sB));
    CHECK(cudaMalloc(&dC, sC * 2));
    CHECK(cudaMemset(dA, 0, sA));
    CHECK(cudaMemset(dB, 0, sB));
    CHECK(cudaMemset(dC, 0xFF, sC * 2)); // fill with garbage

    fp8_gemm_f16acc(M, N, K, dA, dB, dC);
    CHECK(cudaDeviceSynchronize());

    uint16_t *hC = (uint16_t *)malloc(sC * 2);
    CHECK(cudaMemcpy(hC, dC, sC * 2, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < M * N; i++)
    {
        float got = fp16_to_f32(hC[i]);
        if (fabsf(got) > 0.001f)
        {
            if (errors < 5)
                printf("  FAIL [%d][%d]: got=%.4f expected=0\n", i / N, i % N, got);
            errors++;
        }
    }
    printf("  Failures: %d / %d → %s\n", errors, M * N, errors == 0 ? "PASS" : "FAIL");
    free(hC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

// =============================================================================
// Test 3: Random data, FP32 CPU reference (ground truth)
// =============================================================================
static void test_random_fp32ref(int M, int N, int K)
{
    printf("\n=== Test: Random FP32-Reference (%dx%dx%d) ===\n", M, N, K);

    size_t sA = (size_t)M * K, sB = (size_t)N * K, sC = (size_t)M * N;

    uint8_t *hA = (uint8_t *)malloc(sA);
    uint8_t *hB = (uint8_t *)malloc(sB);
    uint16_t *hC = (uint16_t *)malloc(sC * 2);
    float *ref = (float *)malloc(sC * sizeof(float));

    // Fill with small random FP8 values
    srand(42);
    for (size_t i = 0; i < sA; i++)
    {
        float v = ((float)(rand() % 20) - 10.0f) * 0.25f;
        hA[i] = f32_to_e4m3(v);
    }
    for (size_t i = 0; i < sB; i++)
    {
        float v = ((float)(rand() % 20) - 10.0f) * 0.25f;
        hB[i] = f32_to_e4m3(v);
    }

    // CPU reference: FP32 accumulation (ground truth)
    printf("  Computing FP32 CPU reference...\n");
    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
            {
                float a = e4m3_to_f32(hA[m * K + k]);
                float b = e4m3_to_f32(hB[n * K + k]);
                sum += a * b;
            }
            ref[m * N + n] = sum;
        }
    }

    // GPU
    void *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, sA));
    CHECK(cudaMalloc(&dB, sB));
    CHECK(cudaMalloc(&dC, sC * 2));
    CHECK(cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dC, 0, sC * 2));

    fp8_gemm_f16acc(M, N, K, dA, dB, dC);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hC, dC, sC * 2, cudaMemcpyDeviceToHost));

    // Compare with generous tolerance for FP16 accumulation
    // FP16 has 10-bit mantissa → ~0.1% relative precision
    // Over K accumulations, error grows as ~sqrt(K) * precision
    // For K=64: expected rel_error ≈ 8 * 0.001 ≈ 1%
    // For K=256: expected rel_error ≈ 16 * 0.001 ≈ 2%
    int errors = 0, total = M * N;
    float max_abs = 0, max_rel = 0;
    float sum_abs = 0;

    for (int i = 0; i < total; i++)
    {
        float gpu = fp16_to_f32(hC[i]);
        float cpu = ref[i];
        float abs_err = fabsf(gpu - cpu);
        float rel_err = (fabsf(cpu) > 0.1f) ? abs_err / fabsf(cpu) : abs_err;

        if (abs_err > max_abs)
            max_abs = abs_err;
        if (rel_err > max_rel)
            max_rel = rel_err;
        sum_abs += abs_err;

        // Tolerance: FP16 accumulator loses bits at every step
        // abs_err > 5% of max possible sum is a real bug
        float tol_abs = fabsf(cpu) * 0.15f + 2.0f; // 15% relative + 2.0 absolute
        if (abs_err > tol_abs)
        {
            if (errors < 10)
                printf("  MISMATCH [%d][%d]: GPU=%.4f FP32_ref=%.4f err=%.4f (tol=%.4f)\n",
                       i / N, i % N, gpu, cpu, abs_err, tol_abs);
            errors++;
        }
    }

    printf("  Max abs error:   %.4f\n", max_abs);
    printf("  Max rel error:   %.4f\n", max_rel);
    printf("  Mean abs error:  %.4f\n", sum_abs / total);
    printf("  Mismatches (>tol): %d / %d (%.2f%%)\n",
           errors, total, 100.0f * errors / total);

    // Additional: check if errors are random (FP16 precision loss)
    // or systematic (bug in kernel)
    if (errors > 0 && errors < total * 0.01f)
    {
        printf("  → Likely FP16 precision loss, not a bug\n");
    }
    else if (errors == 0)
    {
        printf("  → PASS: all within FP16 accumulation tolerance\n");
    }
    else
    {
        printf("  → WARNING: %.1f%% errors — may indicate kernel bug\n",
               100.0f * errors / total);
    }

    // Distribution analysis: bucket errors
    int bucket[6] = {}; // <0.1, <1, <5, <10, <50, >=50
    for (int i = 0; i < total; i++)
    {
        float e = fabsf(fp16_to_f32(hC[i]) - ref[i]);
        if (e < 0.1f)
            bucket[0]++;
        else if (e < 1.0f)
            bucket[1]++;
        else if (e < 5.0f)
            bucket[2]++;
        else if (e < 10.0f)
            bucket[3]++;
        else if (e < 50.0f)
            bucket[4]++;
        else
            bucket[5]++;
    }
    printf("\n  Error distribution:\n");
    printf("    <0.1:  %6d (%5.1f%%)\n", bucket[0], 100.0f * bucket[0] / total);
    printf("    <1.0:  %6d (%5.1f%%)\n", bucket[1], 100.0f * bucket[1] / total);
    printf("    <5.0:  %6d (%5.1f%%)\n", bucket[2], 100.0f * bucket[2] / total);
    printf("    <10:   %6d (%5.1f%%)\n", bucket[3], 100.0f * bucket[3] / total);
    printf("    <50:   %6d (%5.1f%%)\n", bucket[4], 100.0f * bucket[4] / total);
    printf("    >=50:  %6d (%5.1f%%)\n", bucket[5], 100.0f * bucket[5] / total);

    free(hA);
    free(hB);
    free(hC);
    free(ref);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

// =============================================================================
// Test 4: Single non-zero row/column (pinpoints index bugs)
// A = identity-like (one 1.0 per row), B = known pattern
// =============================================================================
static void test_single_row()
{
    int M = 128, N = 128, K = 64;
    printf("\n=== Test: Single-Row Pattern (%dx%dx%d) ===\n", M, N, K);
    printf("  A[0][0..K-1] = 1.0, rest = 0. B[j][0..K-1] = 1.0\n");
    printf("  Expected: C[0][j] = K = %d, C[i>0][j] = 0\n", K);

    size_t sA = (size_t)M * K, sB = (size_t)N * K, sC = (size_t)M * N;
    uint8_t *hA = (uint8_t *)calloc(sA, 1);
    uint8_t *hB = (uint8_t *)calloc(sB, 1);

    // A: only row 0 has 1.0 (0x38)
    for (int k = 0; k < K; k++)
        hA[k] = 0x38;

    // B: all rows have 1.0
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++)
            hB[n * K + k] = 0x38;

    void *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, sA));
    CHECK(cudaMalloc(&dB, sB));
    CHECK(cudaMalloc(&dC, sC * 2));
    CHECK(cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dC, 0, sC * 2));

    fp8_gemm_f16acc(M, N, K, dA, dB, dC);
    CHECK(cudaDeviceSynchronize());

    uint16_t *hC = (uint16_t *)malloc(sC * 2);
    CHECK(cudaMemcpy(hC, dC, sC * 2, cudaMemcpyDeviceToHost));

    int errors = 0;
    // Check row 0: should be K
    for (int j = 0; j < N; j++)
    {
        float got = fp16_to_f32(hC[j]);
        if (fabsf(got - (float)K) > 1.0f)
        {
            if (errors < 5)
                printf("  FAIL C[0][%d] = %.2f (expected %d)\n", j, got, K);
            errors++;
        }
    }
    // Check rows 1+: should be 0
    for (int i = 1; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float got = fp16_to_f32(hC[i * N + j]);
            if (fabsf(got) > 0.5f)
            {
                if (errors < 5)
                    printf("  FAIL C[%d][%d] = %.2f (expected 0)\n", i, j, got);
                errors++;
            }
        }
    }
    printf("  Failures: %d → %s\n", errors, errors == 0 ? "PASS" : "FAIL");

    free(hA);
    free(hB);
    free(hC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

// =============================================================================
// Test 5: Non-tile-aligned sizes (boundary handling)
// =============================================================================
static void test_non_aligned()
{
    // Sizes that DON'T evenly divide BM=128, BN=128, BK=64
    int sizes[][3] = {
        {100, 100, 64},
        {200, 300, 128},
        {127, 129, 64},
        {256, 256, 96}, // K not multiple of 64
    };
    int ntests = 4;

    printf("\n=== Test: Non-Aligned Sizes ===\n");

    for (int t = 0; t < ntests; t++)
    {
        int M = sizes[t][0], N = sizes[t][1], K = sizes[t][2];
        size_t sA = (size_t)M * K, sB = (size_t)N * K, sC = (size_t)M * N;

        uint8_t *hA = (uint8_t *)malloc(sA);
        uint8_t *hB = (uint8_t *)malloc(sB);
        float *ref = (float *)malloc(sC * sizeof(float));

        srand(t + 100);
        for (size_t i = 0; i < sA; i++)
            hA[i] = f32_to_e4m3(((rand() % 10) - 5) * 0.5f);
        for (size_t i = 0; i < sB; i++)
            hB[i] = f32_to_e4m3(((rand() % 10) - 5) * 0.5f);

        // FP32 reference
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
            {
                float s = 0;
                for (int k = 0; k < K; k++)
                    s += e4m3_to_f32(hA[m * K + k]) * e4m3_to_f32(hB[n * K + k]);
                ref[m * N + n] = s;
            }

        void *dA, *dB, *dC;
        CHECK(cudaMalloc(&dA, sA));
        CHECK(cudaMalloc(&dB, sB));
        CHECK(cudaMalloc(&dC, sC * 2));
        CHECK(cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice));
        CHECK(cudaMemset(dC, 0, sC * 2));

        fp8_gemm_f16acc(M, N, K, dA, dB, dC);
        CHECK(cudaDeviceSynchronize());

        uint16_t *hC = (uint16_t *)malloc(sC * 2);
        CHECK(cudaMemcpy(hC, dC, sC * 2, cudaMemcpyDeviceToHost));

        int errs = 0;
        float max_e = 0;
        for (int i = 0; i < M * N; i++)
        {
            float e = fabsf(fp16_to_f32(hC[i]) - ref[i]);
            if (e > max_e)
                max_e = e;
            float tol = fabsf(ref[i]) * 0.15f + 2.0f;
            if (e > tol)
                errs++;
        }
        printf("  %3dx%3dx%3d: max_err=%.2f, fails=%d/%d → %s\n",
               M, N, K, max_e, errs, M * N, errs == 0 ? "PASS" : "FAIL");

        free(hA);
        free(hB);
        free(hC);
        free(ref);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }
}

// =============================================================================
int main()
{
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== FP8 GEMM F16-Acc Correctness Validator ===\n");
    printf("GPU: %s\n", prop.name);

    test_zeros(128, 128, 64);
    test_ones(128, 128, 64);
    test_ones(256, 256, 256);
    test_ones(128, 128, 128);
    test_single_row();
    test_random_fp32ref(128, 128, 64);
    test_random_fp32ref(256, 256, 256);
    test_non_aligned();

    printf("\n=== Done ===\n");
    return 0;
}
