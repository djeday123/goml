// =============================================================================
// FlashAttention Benchmark
// =============================================================================
// Tests correctness against CPU reference and benchmarks vs naive kernel.
//
// Build: nvcc -O3 -arch=sm_89 -std=c++17 \
//        flash_attention.cu transformer_kernels.cu flash_attention_bench.cu \
//        -o flash_attention_bench -lcudart
// =============================================================================

#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C"
{
    int flash_attention_forward(
        const void *Q, const void *K, const void *V, void *O,
        int total_heads, int seq_len, int head_dim,
        int causal, void *stream);

    // Naive attention from transformer_kernels.cu
    int attention_forward(
        const void *Q, const void *K, const void *V, void *O,
        int batch, int num_heads, int seq_len, int head_dim,
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

static inline float fp16f(uint16_t h)
{
    __half hv;
    memcpy(&hv, &h, 2);
    return __half2float(hv);
}
static inline uint16_t f2h(float f)
{
    __half hv = __float2half(f);
    uint16_t r;
    memcpy(&r, &hv, 2);
    return r;
}

void fill_random_fp16(uint16_t *d_ptr, int n)
{
    uint16_t *h = (uint16_t *)malloc(n * 2);
    for (int i = 0; i < n; i++)
        h[i] = f2h(((float)(rand() % 2001) - 1000.0f) / 1000.0f);
    CK(cudaMemcpy(d_ptr, h, n * 2, cudaMemcpyHostToDevice));
    free(h);
}

struct Timer
{
    cudaEvent_t t0, t1;
    Timer()
    {
        CK(cudaEventCreate(&t0));
        CK(cudaEventCreate(&t1));
    }
    ~Timer()
    {
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }
    void start() { CK(cudaEventRecord(t0)); }
    float stop()
    {
        CK(cudaEventRecord(t1));
        CK(cudaEventSynchronize(t1));
        float ms;
        CK(cudaEventElapsedTime(&ms, t0, t1));
        return ms;
    }
};

// =============================================================================
// CPU reference (causal attention)
// =============================================================================
void cpu_attention(
    const uint16_t *Q, const uint16_t *K, const uint16_t *V, float *O,
    int total_heads, int seq_len, int head_dim, int causal)
{
    float scale = 1.0f / sqrtf((float)head_dim);
    int hstride = seq_len * head_dim;

    for (int bh = 0; bh < total_heads; bh++)
    {
        int base = bh * hstride;
        for (int i = 0; i < seq_len; i++)
        {
            // Compute scores
            int max_j = causal ? (i + 1) : seq_len;
            float *scores = (float *)alloca(seq_len * sizeof(float));

            float mx = -1e30f;
            for (int j = 0; j < max_j; j++)
            {
                float dot = 0;
                for (int d = 0; d < head_dim; d++)
                    dot += fp16f(Q[base + i * head_dim + d]) * fp16f(K[base + j * head_dim + d]);
                scores[j] = dot * scale;
                if (scores[j] > mx)
                    mx = scores[j];
            }
            // Softmax
            float sum = 0;
            for (int j = 0; j < max_j; j++)
            {
                scores[j] = expf(scores[j] - mx);
                sum += scores[j];
            }
            float inv = 1.0f / (sum + 1e-8f);
            // Output
            for (int d = 0; d < head_dim; d++)
            {
                float acc = 0;
                for (int j = 0; j < max_j; j++)
                    acc += scores[j] * inv * fp16f(V[base + j * head_dim + d]);
                O[base + i * head_dim + d] = acc;
            }
        }
    }
}

// =============================================================================
// Correctness test
// =============================================================================
void test_correctness()
{
    printf("--- Correctness ---\n");

    struct
    {
        int heads, seq, dim;
        const char *name;
    } tests[] = {
        {1, 16, 64, "tiny 1h×16s×64d"},
        {2, 32, 128, "small 2h×32s×128d"},
        {2, 64, 128, "med 2h×64s×128d"},
        {4, 128, 128, "4h×128s×128d"},
        {2, 256, 128, "2h×256s×128d"},
    };

    for (auto &t : tests)
    {
        int n = t.heads * t.seq * t.dim;

        uint16_t *hQ = (uint16_t *)malloc(n * 2);
        uint16_t *hK = (uint16_t *)malloc(n * 2);
        uint16_t *hV = (uint16_t *)malloc(n * 2);
        uint16_t *hO = (uint16_t *)malloc(n * 2);
        float *ref = (float *)malloc(n * 4);

        srand(42);
        for (int i = 0; i < n; i++)
        {
            hQ[i] = f2h(((float)(rand() % 201) - 100.0f) / 100.0f);
            hK[i] = f2h(((float)(rand() % 201) - 100.0f) / 100.0f);
            hV[i] = f2h(((float)(rand() % 201) - 100.0f) / 100.0f);
        }

        // CPU reference
        cpu_attention(hQ, hK, hV, ref, t.heads, t.seq, t.dim, 1);

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

        int rc = flash_attention_forward(dQ, dK, dV, dO,
                                         t.heads, t.seq, t.dim, 1, nullptr);
        CK(cudaDeviceSynchronize());

        if (rc != 0)
        {
            printf("  %-22s LAUNCH FAIL (rc=%d)\n", t.name, rc);
        }
        else
        {
            CK(cudaMemcpy(hO, dO, n * 2, cudaMemcpyDeviceToHost));
            float max_err = 0, max_rel = 0;
            int errors = 0;
            for (int i = 0; i < n; i++)
            {
                float got = fp16f(hO[i]);
                float exp_val = ref[i];
                float ae = fabsf(got - exp_val);
                float re = (fabsf(exp_val) > 1e-4f) ? ae / fabsf(exp_val) : ae;
                if (ae > max_err)
                    max_err = ae;
                if (re > max_rel)
                    max_rel = re;
                if (re > 0.05f)
                    errors++; // 5% relative tolerance
            }
            const char *status = (max_rel < 0.05f && errors == 0) ? "PASS" : "FAIL";
            printf("  %-22s max_abs=%.4f max_rel=%.4f errors=%d → %s\n",
                   t.name, max_err, max_rel, errors, status);
        }

        cudaFree(dQ);
        cudaFree(dK);
        cudaFree(dV);
        cudaFree(dO);
        free(hQ);
        free(hK);
        free(hV);
        free(hO);
        free(ref);
    }
}

// =============================================================================
// Performance benchmark
// =============================================================================
void bench()
{
    printf("\n--- Performance: Flash vs Naive ---\n");
    printf("%-12s %10s %10s %8s\n", "Config", "Naive", "Flash", "Speedup");
    printf("---------------------------------------------------\n");

    struct
    {
        int b, h, s, d;
        const char *label;
    } configs[] = {
        {1, 32, 256, 128, "7B-256"},
        {1, 32, 512, 128, "7B-512"},
        {1, 32, 1024, 128, "7B-1K"},
        {1, 32, 2048, 128, "7B-2K"},
        {1, 32, 4096, 128, "7B-4K"},
        {1, 40, 512, 128, "13B-512"},
        {1, 64, 512, 128, "70B-512"},
        {1, 64, 2048, 128, "70B-2K"},
    };

    Timer t;

    for (auto &c : configs)
    {
        int n = c.b * c.h * c.s * c.d;
        int total_heads = c.b * c.h;

        void *dQ, *dK, *dV, *dO_naive, *dO_flash;
        CK(cudaMalloc(&dQ, (size_t)n * 2));
        CK(cudaMalloc(&dK, (size_t)n * 2));
        CK(cudaMalloc(&dV, (size_t)n * 2));
        CK(cudaMalloc(&dO_naive, (size_t)n * 2));
        CK(cudaMalloc(&dO_flash, (size_t)n * 2));
        fill_random_fp16((uint16_t *)dQ, n);
        fill_random_fp16((uint16_t *)dK, n);
        fill_random_fp16((uint16_t *)dV, n);

        // FLOPS: 2 * b * h * s² * d (QK^T) + 2 * b * h * s² * d (PV) = 4*b*h*s²*d
        double flops = 4.0 * c.b * c.h * (double)c.s * c.s * c.d;

        // ---- Naive ----
        double naive_tflops = 0;
        int naive_smem = (256 / 32 + c.s) * 4;
        if (naive_smem <= 100 * 1024)
        {
            for (int i = 0; i < 3; i++)
                attention_forward(dQ, dK, dV, dO_naive, c.b, c.h, c.s, c.d, 1, nullptr);
            CK(cudaDeviceSynchronize());
            int it = (c.s <= 1024) ? 50 : 10;
            t.start();
            for (int i = 0; i < it; i++)
                attention_forward(dQ, dK, dV, dO_naive, c.b, c.h, c.s, c.d, 1, nullptr);
            float ms = t.stop();
            naive_tflops = flops / (ms / it / 1000.0) / 1e12;
        }

        // ---- Flash ----
        for (int i = 0; i < 3; i++)
            flash_attention_forward(dQ, dK, dV, dO_flash, total_heads, c.s, c.d, 1, nullptr);
        CK(cudaDeviceSynchronize());
        int it = (c.s <= 1024) ? 100 : 20;
        t.start();
        for (int i = 0; i < it; i++)
            flash_attention_forward(dQ, dK, dV, dO_flash, total_heads, c.s, c.d, 1, nullptr);
        float ms = t.stop();
        double flash_tflops = flops / (ms / it / 1000.0) / 1e12;

        if (naive_tflops > 0)
        {
            printf("%-12s %8.2f T %8.2f T  %5.1f×\n",
                   c.label, naive_tflops, flash_tflops, flash_tflops / naive_tflops);
        }
        else
        {
            printf("%-12s %10s %8.2f T    -\n", c.label, "OOM/skip", flash_tflops);
        }

        cudaFree(dQ);
        cudaFree(dK);
        cudaFree(dV);
        cudaFree(dO_naive);
        cudaFree(dO_flash);
    }

    printf("\nFlash: O(N) memory, no seq_len limit | Naive: O(N²), max ~25K\n");
}

// =============================================================================
int main()
{
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("=== FlashAttention Benchmark ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n\n", p.name, p.multiProcessorCount, p.clockRate / 1000);

    test_correctness();
    bench();

    return 0;
}
