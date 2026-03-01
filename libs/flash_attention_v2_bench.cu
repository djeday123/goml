// =============================================================================
// FlashAttention v2 (MMA) Benchmark
// =============================================================================
// Tests correctness vs CPU reference, benchmarks v1 (scalar) vs v2 (MMA)
//
// Build: nvcc -O3 -arch=sm_89 -std=c++17 \
//        flash_attention_v2.cu flash_attention.cu transformer_kernels.cu \
//        flash_attention_v2_bench.cu -o flash_v2_bench -lcudart
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
    int flash_attention_v2_forward(
        const void *Q, const void *K, const void *V, void *O,
        int total_heads, int seq_len, int head_dim,
        int causal, void *stream);

    int flash_attention_forward(
        const void *Q, const void *K, const void *V, void *O,
        int total_heads, int seq_len, int head_dim,
        int causal, void *stream);

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

// CPU reference
void cpu_attention(
    const uint16_t *Q, const uint16_t *K, const uint16_t *V, float *O,
    int total_heads, int seq_len, int head_dim, int causal)
{
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int bh = 0; bh < total_heads; bh++)
    {
        int base = bh * seq_len * head_dim;
        for (int i = 0; i < seq_len; i++)
        {
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
            float sum = 0;
            for (int j = 0; j < max_j; j++)
            {
                scores[j] = expf(scores[j] - mx);
                sum += scores[j];
            }
            float inv = 1.0f / (sum + 1e-8f);
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
void test_correctness()
{
    printf("--- Correctness (v2 MMA vs CPU) ---\n");

    struct
    {
        int heads, seq, dim;
        const char *name;
    } tests[] = {
        {1, 16, 128, "1h×16s×128d"},
        {2, 32, 128, "2h×32s×128d"},
        {2, 64, 128, "2h×64s×128d"},
        {4, 128, 128, "4h×128s×128d"},
        {2, 256, 128, "2h×256s×128d"},
        {4, 512, 128, "4h×512s×128d"},
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

        cpu_attention(hQ, hK, hV, ref, t.heads, t.seq, t.dim, 1);

        void *dQ, *dK, *dV, *dO;
        CK(cudaMalloc(&dQ, n * 2));
        CK(cudaMalloc(&dK, n * 2));
        CK(cudaMalloc(&dV, n * 2));
        CK(cudaMalloc(&dO, n * 2));
        CK(cudaMemcpy(dQ, hQ, n * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dK, hK, n * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dV, hV, n * 2, cudaMemcpyHostToDevice));
        CK(cudaMemset(dO, 0, n * 2));

        int rc = flash_attention_v2_forward(dQ, dK, dV, dO,
                                            t.heads, t.seq, t.dim, 1, nullptr);
        CK(cudaDeviceSynchronize());

        if (rc != 0)
        {
            printf("  %-20s LAUNCH FAIL (rc=%d)\n", t.name, rc);
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
                if (ae > 0.002f && re > 0.1f)
                    errors++; // FP16: abs~1e-3 normal
            }
            const char *status = (errors == 0) ? "PASS" : "FAIL";
            printf("  %-20s abs=%.4f rel=%.4f err=%d → %s\n",
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
void bench()
{
    printf("\n--- Performance: Naive vs Flash-v1 vs Flash-v2-MMA ---\n");
    printf("%-12s %10s %10s %10s %8s\n",
           "Config", "Naive", "Flash-v1", "v2-MMA", "v2/v1");
    printf("--------------------------------------------------------------\n");

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

        void *dQ, *dK, *dV, *dO;
        CK(cudaMalloc(&dQ, (size_t)n * 2));
        CK(cudaMalloc(&dK, (size_t)n * 2));
        CK(cudaMalloc(&dV, (size_t)n * 2));
        CK(cudaMalloc(&dO, (size_t)n * 2));
        fill_random_fp16((uint16_t *)dQ, n);
        fill_random_fp16((uint16_t *)dK, n);
        fill_random_fp16((uint16_t *)dV, n);

        double flops = 4.0 * c.b * c.h * (double)c.s * c.s * c.d;

        // ---- Naive ----
        double naive_tflops = 0;
        int naive_smem = (256 / 32 + c.s) * 4;
        if (naive_smem <= 100 * 1024)
        {
            for (int i = 0; i < 3; i++)
                attention_forward(dQ, dK, dV, dO, c.b, c.h, c.s, c.d, 1, nullptr);
            CK(cudaDeviceSynchronize());
            int it = (c.s <= 1024) ? 50 : 10;
            t.start();
            for (int i = 0; i < it; i++)
                attention_forward(dQ, dK, dV, dO, c.b, c.h, c.s, c.d, 1, nullptr);
            float ms = t.stop();
            CK(cudaGetLastError());
            naive_tflops = flops / (ms / it / 1000.0) / 1e12;
        }

        // ---- Flash v1 (scalar) ----
        for (int i = 0; i < 3; i++)
            flash_attention_forward(dQ, dK, dV, dO, total_heads, c.s, c.d, 1, nullptr);
        CK(cudaDeviceSynchronize());
        int it1 = (c.s <= 1024) ? 100 : 20;
        t.start();
        for (int i = 0; i < it1; i++)
            flash_attention_forward(dQ, dK, dV, dO, total_heads, c.s, c.d, 1, nullptr);
        float ms1 = t.stop();
        CK(cudaGetLastError());
        double v1_tflops = flops / (ms1 / it1 / 1000.0) / 1e12;

        // ---- Flash v2 (MMA) ----
        CK(cudaMemset(dO, 0, (size_t)n * 2));
        for (int i = 0; i < 3; i++)
            flash_attention_v2_forward(dQ, dK, dV, dO, total_heads, c.s, c.d, 1, nullptr);
        CK(cudaDeviceSynchronize());
        int it2 = (c.s <= 1024) ? 100 : 20;
        t.start();
        for (int i = 0; i < it2; i++)
            flash_attention_v2_forward(dQ, dK, dV, dO, total_heads, c.s, c.d, 1, nullptr);
        float ms2 = t.stop();
        CK(cudaGetLastError());
        double v2_tflops = flops / (ms2 / it2 / 1000.0) / 1e12;

        if (naive_tflops > 0)
        {
            printf("%-12s %8.2f T %8.2f T %8.2f T  %5.1f×\n",
                   c.label, naive_tflops, v1_tflops, v2_tflops, v2_tflops / v1_tflops);
        }
        else
        {
            printf("%-12s %10s %8.2f T %8.2f T  %5.1f×\n",
                   c.label, "skip", v1_tflops, v2_tflops, v2_tflops / v1_tflops);
        }

        cudaFree(dQ);
        cudaFree(dK);
        cudaFree(dV);
        cudaFree(dO);
    }
}

// =============================================================================
int main()
{
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("=== FlashAttention v2 (MMA) Benchmark ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n\n", p.name, p.multiProcessorCount, p.clockRate / 1000);

    test_correctness();
    bench();

    return 0;
}
