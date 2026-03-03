// flash_attention_v43_bench.cu — v43 (sw pipeline) vs v20

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C"
{
    int flash_attention_v20_forward(const void *, const void *, const void *, void *,
                                    int, int, int, int, void *);
    int flash_attention_v43_forward(const void *, const void *, const void *, void *,
                                    int, int, int, int, void *);
    int flash_attention_forward(const void *, const void *, const void *, void *,
                                int, int, int, int, void *);
}

#define CK(c)                                                                      \
    do                                                                             \
    {                                                                              \
        cudaError_t e = (c);                                                       \
        if (e)                                                                     \
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
    printf("--- Correctness (v43 vs v1-scalar) ---\n");
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
        __half *dQ, *dK, *dV, *dO_ref, *dO;
        CK(cudaMalloc(&dQ, bytes));
        CK(cudaMalloc(&dK, bytes));
        CK(cudaMalloc(&dV, bytes));
        CK(cudaMalloc(&dO_ref, bytes));
        CK(cudaMalloc(&dO, bytes));

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
        flash_attention_forward(dQ, dK, dV, dO_ref, c.h, c.s, c.d, 1, nullptr);
        CK(cudaDeviceSynchronize());

        CK(cudaMemset(dO, 0, bytes));
        flash_attention_v43_forward(dQ, dK, dV, dO, c.h, c.s, c.d, 1, nullptr);
        CK(cudaDeviceSynchronize());

        std::vector<__half> hr(n), hv(n);
        CK(cudaMemcpy(hr.data(), dO_ref, bytes, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hv.data(), dO, bytes, cudaMemcpyDeviceToHost));

        float max_ae = 0;
        int errors = 0;
        for (size_t i = 0; i < n; i++)
        {
            float rv = __half2float(hr[i]), vv = __half2float(hv[i]);
            float ae = fabsf(rv - vv);
            float re = (fabsf(rv) > 1e-6f) ? ae / fabsf(rv) : 0;
            if (ae > max_ae)
                max_ae = ae;
            if (ae > 0.01f && re > 0.1f)
                errors++;
        }
        printf("  %dh x %ds x %dd  maxabs=%.4f err=%d -> %s\n",
               c.h, c.s, c.d, max_ae, errors, errors == 0 ? "PASS" : "FAIL");

        cudaFree(dQ);
        cudaFree(dK);
        cudaFree(dV);
        cudaFree(dO_ref);
        cudaFree(dO);
    }
}

void bench()
{
    printf("\n--- Performance: v20 vs v43 (sw pipeline) ---\n");
    printf("%-12s %10s %10s  %6s  %6s\n", "Config", "v20", "v43", "ratio", "peak%%");
    printf("---------------------------------------------------------\n");

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
        {"hi-512", 1, 128, 512, 128},
        {"hi-1K", 1, 128, 1024, 128},
    };
    Timer t;
    float peak = 165.2f;

    for (auto &c : cfgs)
    {
        int heads = c.b * c.h;
        size_t n = (size_t)heads * c.s * c.d;
        size_t bytes = n * 2;
        double flops = (double)heads * (4.0 * c.s * c.s * c.d) / 2.0;

        __half *dQ, *dK, *dV, *dO;
        CK(cudaMalloc(&dQ, bytes));
        CK(cudaMalloc(&dK, bytes));
        CK(cudaMalloc(&dV, bytes));
        CK(cudaMalloc(&dO, bytes));
        CK(cudaMemset(dQ, 0x3C, bytes));
        CK(cudaMemset(dK, 0x3C, bytes));
        CK(cudaMemset(dV, 0x3C, bytes));

        int it = (c.s <= 1024) ? 100 : 20;

        for (int i = 0; i < 3; i++)
            flash_attention_v20_forward(dQ, dK, dV, dO, heads, c.s, c.d, 1, nullptr);
        CK(cudaDeviceSynchronize());
        t.start();
        for (int i = 0; i < it; i++)
            flash_attention_v20_forward(dQ, dK, dV, dO, heads, c.s, c.d, 1, nullptr);
        float ms20 = t.stop();
        CK(cudaGetLastError());
        double t20 = flops / (ms20 / it / 1000.0) / 1e12;

        for (int i = 0; i < 3; i++)
            flash_attention_v43_forward(dQ, dK, dV, dO, heads, c.s, c.d, 1, nullptr);
        CK(cudaDeviceSynchronize());
        t.start();
        for (int i = 0; i < it; i++)
            flash_attention_v43_forward(dQ, dK, dV, dO, heads, c.s, c.d, 1, nullptr);
        float ms43 = t.stop();
        CK(cudaGetLastError());
        double t43 = flops / (ms43 / it / 1000.0) / 1e12;

        printf("%-12s %8.2f T %8.2f T  %5.2fx  %5.1f%%\n",
               c.l, t20, t43, t43 / t20, t43 / peak * 100.0);

        cudaFree(dQ);
        cudaFree(dK);
        cudaFree(dV);
        cudaFree(dO);
    }
}

int main()
{
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("=== FlashAttention v43 (SW Pipeline SMEM->Reg) ===\n");
    printf("GPU: %s (%d SMs)\n\n", p.name, p.multiProcessorCount);
    test_correctness();
    bench();
    return 0;
}
