// flash_attention_v31_bench.cu
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
    int flash_attention_v20_forward(const void *, const void *, const void *, void *,
                                    int, int, int, int, void *);
    int flash_attention_v31_forward(const void *, const void *, const void *, void *,
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
    printf("--- Correctness (v31 vs v1-scalar) ---\n");
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
        size_t n = (size_t)c.h * c.s * c.d, bytes = n * 2;
        __half *dQ, *dK, *dV, *dR, *dT;
        CK(cudaMalloc(&dQ, bytes));
        CK(cudaMalloc(&dK, bytes));
        CK(cudaMalloc(&dV, bytes));
        CK(cudaMalloc(&dR, bytes));
        CK(cudaMalloc(&dT, bytes));
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
        CK(cudaMemset(dR, 0, bytes));
        flash_attention_forward(dQ, dK, dV, dR, c.h, c.s, c.d, 1, nullptr);
        CK(cudaDeviceSynchronize());
        CK(cudaMemset(dT, 0, bytes));
        flash_attention_v31_forward(dQ, dK, dV, dT, c.h, c.s, c.d, 1, nullptr);
        CK(cudaDeviceSynchronize());
        std::vector<__half> hr(n), ht(n);
        CK(cudaMemcpy(hr.data(), dR, bytes, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ht.data(), dT, bytes, cudaMemcpyDeviceToHost));
        float mae = 0;
        int errs = 0;
        for (size_t i = 0; i < n; i++)
        {
            float ae = fabsf(__half2float(hr[i]) - __half2float(ht[i]));
            if (ae > mae)
                mae = ae;
            if (ae > 0.01f)
                errs++;
        }
        printf("  %dh x %ds x %dd  maxabs=%.4f err=%d -> %s\n",
               c.h, c.s, c.d, mae, errs, errs == 0 ? "PASS" : "FAIL");
        cudaFree(dQ);
        cudaFree(dK);
        cudaFree(dV);
        cudaFree(dR);
        cudaFree(dT);
    }
}

void bench()
{
    printf("\n--- Performance: v20 vs v31 (ldm4 + N-first + 3buf) ---\n");
    printf("%-12s %10s %10s  %6s  %6s %6s\n",
           "Config", "v20", "v31", "ratio", "v20%", "v31%");
    printf("--------------------------------------------------------------\n");
    const double pk = 165.2;
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
        {"7B-8K", 1, 32, 8192, 128},
        {"70B-512", 1, 64, 512, 128},
        {"70B-2K", 1, 64, 2048, 128},
        {"hi-512", 32, 16, 512, 128},
        {"hi-1K", 16, 16, 1024, 128},
    };
    Timer t;
    for (auto &c : cfgs)
    {
        int hds = c.b * c.h;
        size_t n = (size_t)hds * c.s * c.d, bytes = n * 2;
        double flops = (double)hds * (4.0 * c.s * c.s * c.d) / 2.0;
        int it = (c.s <= 1024) ? 100 : (c.s <= 4096 ? 20 : 10);
        __half *dQ, *dK, *dV, *dO;
        CK(cudaMalloc(&dQ, bytes));
        CK(cudaMalloc(&dK, bytes));
        CK(cudaMalloc(&dV, bytes));
        CK(cudaMalloc(&dO, bytes));
        CK(cudaMemset(dQ, 0x3C, bytes));
        CK(cudaMemset(dK, 0x3C, bytes));
        CK(cudaMemset(dV, 0x3C, bytes));

        // v20
        for (int i = 0; i < 3; i++)
            flash_attention_v20_forward(dQ, dK, dV, dO, hds, c.s, c.d, 1, nullptr);
        CK(cudaDeviceSynchronize());
        t.start();
        for (int i = 0; i < it; i++)
            flash_attention_v20_forward(dQ, dK, dV, dO, hds, c.s, c.d, 1, nullptr);
        float m20 = t.stop();
        double t20 = flops / (m20 / it / 1e3) / 1e12;

        // v31
        CK(cudaMemset(dO, 0, bytes));
        for (int i = 0; i < 3; i++)
            flash_attention_v31_forward(dQ, dK, dV, dO, hds, c.s, c.d, 1, nullptr);
        CK(cudaDeviceSynchronize());
        t.start();
        for (int i = 0; i < it; i++)
            flash_attention_v31_forward(dQ, dK, dV, dO, hds, c.s, c.d, 1, nullptr);
        float m21 = t.stop();
        double t21 = flops / (m21 / it / 1e3) / 1e12;

        printf("%-12s %8.2f T %8.2f T  %5.2fx  %5.1f%% %5.1f%%\n",
               c.l, t20, t21, t21 / t20, t20 / pk * 100, t21 / pk * 100);
        cudaFree(dQ);
        cudaFree(dK);
        cudaFree(dV);
        cudaFree(dO);
    }
    printf("\nPeak = %.1f TFLOPS\n", pk);
}

int main()
{
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("=== FlashAttention v31 (ldm4 + N-first + 3buf V-overlap) ===\n");
    printf("GPU: %s (%d SMs)\n\n", p.name, p.multiProcessorCount);
    test_correctness();
    bench();
    return 0;
}
