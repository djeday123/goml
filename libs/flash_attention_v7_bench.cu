// flash_attention_v7_bench.cu — CORRECTED causal FLOPS (÷2 per Tri Dao)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C"
{
    int flash_attention_forward(const void *, const void *, const void *, void *,
                                int, int, int, int, void *);
    int flash_attention_v6_forward(const void *, const void *, const void *, void *,
                                   int, int, int, int, void *);
    int flash_attention_v7_forward(const void *, const void *, const void *, void *,
                                   int, int, int, int, void *);
}

#define CK(c)                                                                      \
    do                                                                             \
    {                                                                              \
        cudaError_t e = (c);                                                       \
        if (e != cudaSuccess)                                                      \
        {                                                                          \
            printf("CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(1);                                                               \
        }                                                                          \
    } while (0)

struct Timer
{
    cudaEvent_t t0, t1;
    Timer()
    {
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
    }
    void start() { cudaEventRecord(t0); }
    float stop()
    {
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms;
        cudaEventElapsedTime(&ms, t0, t1);
        return ms;
    }
    ~Timer()
    {
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }
};

void test_correctness()
{
    printf("--- Correctness (v7 vs v6) ---\n");
    struct
    {
        int h, s, d;
    } cfgs[] = {
        {1, 16, 128},
        {2, 32, 128},
        {2, 64, 128},
        {4, 128, 128},
        {2, 256, 128},
        {4, 512, 128},
        {32, 1024, 128},
        {32, 2048, 128},
    };
    for (auto &c : cfgs)
    {
        size_t n = (size_t)c.h * c.s * c.d;
        size_t bytes = n * 2;
        __half *dQ, *dK, *dV, *dO_ref, *dO_v7;
        CK(cudaMalloc(&dQ, bytes));
        CK(cudaMalloc(&dK, bytes));
        CK(cudaMalloc(&dV, bytes));
        CK(cudaMalloc(&dO_ref, bytes));
        CK(cudaMalloc(&dO_v7, bytes));

        std::vector<__half> h(n);
        srand(42);
        for (size_t i = 0; i < n; i++)
            h[i] = __float2half((float)(rand() % 1000 - 500) / 1000.f);
        CK(cudaMemcpy(dQ, h.data(), bytes, cudaMemcpyHostToDevice));
        for (size_t i = 0; i < n; i++)
            h[i] = __float2half((float)(rand() % 1000 - 500) / 1000.f);
        CK(cudaMemcpy(dK, h.data(), bytes, cudaMemcpyHostToDevice));
        for (size_t i = 0; i < n; i++)
            h[i] = __float2half((float)(rand() % 1000 - 500) / 1000.f);
        CK(cudaMemcpy(dV, h.data(), bytes, cudaMemcpyHostToDevice));

        CK(cudaMemset(dO_ref, 0, bytes));
        flash_attention_v6_forward(dQ, dK, dV, dO_ref, c.h, c.s, c.d, 1, nullptr);
        CK(cudaDeviceSynchronize());

        CK(cudaMemset(dO_v7, 0, bytes));
        flash_attention_v7_forward(dQ, dK, dV, dO_v7, c.h, c.s, c.d, 1, nullptr);
        CK(cudaDeviceSynchronize());

        std::vector<__half> hr(n), hv(n);
        CK(cudaMemcpy(hr.data(), dO_ref, bytes, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hv.data(), dO_v7, bytes, cudaMemcpyDeviceToHost));

        float max_ae = 0, max_re = 0;
        int errors = 0;
        for (size_t i = 0; i < n; i++)
        {
            float rv = __half2float(hr[i]), vv = __half2float(hv[i]);
            float ae = fabsf(rv - vv);
            float re = (fabsf(rv) > 1e-6f) ? ae / fabsf(rv) : 0;
            if (ae > max_ae)
                max_ae = ae;
            if (re > max_re)
                max_re = re;
            if (ae > 0.01f && re > 0.1f)
                errors++;
        }
        printf("  %dh×%ds×%dd  abs=%.4f rel=%.4f err=%d → %s\n",
               c.h, c.s, c.d, max_ae, max_re, errors, errors == 0 ? "PASS" : "FAIL");

        cudaFree(dQ);
        cudaFree(dK);
        cudaFree(dV);
        cudaFree(dO_ref);
        cudaFree(dO_v7);
    }
}

void bench()
{
    printf("\n--- Performance (causal, CORRECTED FLOPS = 4·s²·d·h / 2) ---\n");
    printf("%-12s %10s %10s %10s  %6s  %6s\n",
           "Config", "v1", "v6", "v7-sw", "v7/v6", "%%peak");
    printf("---------------------------------------------------------------------\n");

    // RTX 4090 FP16 tensor core peak with FP32 accum = 165.2 TFLOPS
    const double gpu_peak = 165.2;

    struct
    {
        const char *l;
        int b, h, s, d;
    } cfgs[] = {
        {"7B-256", 1, 32, 256, 128},
        {"7B-512", 1, 32, 512, 128},
        {"7B-1K", 1, 32, 1024, 128},
        {"7B-2K", 1, 32, 2048, 128},
        {"7B-4K", 1, 32, 4096, 128},
        {"13B-512", 1, 40, 512, 128},
        {"70B-512", 1, 64, 512, 128},
        {"70B-2K", 1, 64, 2048, 128},
    };
    Timer t;

    for (auto &c : cfgs)
    {
        int heads = c.b * c.h;
        size_t n = (size_t)heads * c.s * c.d;
        size_t bytes = n * 2;
        // CORRECTED: divide by 2 for causal (per Tri Dao FA2 paper)
        double flops = (double)heads * (4.0 * c.s * c.s * c.d) / 2.0;
        int it = (c.s <= 1024) ? 100 : 20;

        __half *dQ, *dK, *dV, *dO;
        CK(cudaMalloc(&dQ, bytes));
        CK(cudaMalloc(&dK, bytes));
        CK(cudaMalloc(&dV, bytes));
        CK(cudaMalloc(&dO, bytes));
        CK(cudaMemset(dQ, 0x3C, bytes));
        CK(cudaMemset(dK, 0x3C, bytes));
        CK(cudaMemset(dV, 0x3C, bytes));

        // v1
        double v1_t = 0;
        if (c.s <= 2048)
        {
            for (int i = 0; i < 3; i++)
                flash_attention_forward(dQ, dK, dV, dO, heads, c.s, c.d, 1, nullptr);
            CK(cudaDeviceSynchronize());
            int it1 = (c.s <= 512) ? 100 : 20;
            t.start();
            for (int i = 0; i < it1; i++)
                flash_attention_forward(dQ, dK, dV, dO, heads, c.s, c.d, 1, nullptr);
            float ms = t.stop();
            CK(cudaGetLastError());
            v1_t = flops / (ms / it1 / 1000.0) / 1e12;
        }

        // v6
        CK(cudaMemset(dO, 0, bytes));
        for (int i = 0; i < 3; i++)
            flash_attention_v6_forward(dQ, dK, dV, dO, heads, c.s, c.d, 1, nullptr);
        CK(cudaDeviceSynchronize());
        t.start();
        for (int i = 0; i < it; i++)
            flash_attention_v6_forward(dQ, dK, dV, dO, heads, c.s, c.d, 1, nullptr);
        float ms6 = t.stop();
        CK(cudaGetLastError());
        double v6_t = flops / (ms6 / it / 1000.0) / 1e12;

        // v7
        CK(cudaMemset(dO, 0, bytes));
        for (int i = 0; i < 3; i++)
            flash_attention_v7_forward(dQ, dK, dV, dO, heads, c.s, c.d, 1, nullptr);
        CK(cudaDeviceSynchronize());
        t.start();
        for (int i = 0; i < it; i++)
            flash_attention_v7_forward(dQ, dK, dV, dO, heads, c.s, c.d, 1, nullptr);
        float ms7 = t.stop();
        CK(cudaGetLastError());
        double v7_t = flops / (ms7 / it / 1000.0) / 1e12;

        double pct = v7_t / gpu_peak * 100.0;

        if (v1_t > 0)
            printf("%-12s %8.2f T %8.2f T %8.2f T  %5.2fx  %5.1f%%\n",
                   c.l, v1_t, v6_t, v7_t, v7_t / v6_t, pct);
        else
            printf("%-12s %10s %8.2f T %8.2f T  %5.2fx  %5.1f%%\n",
                   c.l, "skip", v6_t, v7_t, v7_t / v6_t, pct);

        cudaFree(dQ);
        cudaFree(dK);
        cudaFree(dV);
        cudaFree(dO);
    }

    printf("\nNote: FP16 HMMA peak (FP32 accum) = %.1f TFLOPS on RTX 4090\n", gpu_peak);
    printf("Tri Dao FA2 on A100: ~230 TFLOPS (74%% of 312T peak)\n");
}

int main()
{
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("=== FlashAttention v7 Benchmark (Corrected FLOPS) ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n\n", p.name, p.multiProcessorCount, p.clockRate / 1000);
    test_correctness();
    bench();
    return 0;
}
