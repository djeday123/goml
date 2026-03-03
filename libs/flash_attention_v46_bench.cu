// flash_attention_v46_bench.cu - test v46 (variable Bc) vs v20
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C"
{
    int flash_attention_v20_forward(const void *, const void *, const void *, void *,
                                    int, int, int, int, void *);
    int flash_attention_v46_forward(const void *, const void *, const void *, void *,
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
        CK(cudaEventCreate(&t0));
        CK(cudaEventCreate(&t1));
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
    ~Timer()
    {
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }
};

static inline uint16_t f2h(float f)
{
    __half hv = __float2half(f);
    uint16_t r;
    memcpy(&r, &hv, 2);
    return r;
}
void fill_random(uint16_t *d, int n)
{
    uint16_t *h = (uint16_t *)malloc(n * 2);
    for (int i = 0; i < n; i++)
        h[i] = f2h(((float)(rand() % 2001) - 1000.f) / 1000.f);
    CK(cudaMemcpy(d, h, n * 2, cudaMemcpyHostToDevice));
    free(h);
}

int main()
{
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("=== FlashAttention v46 — Variable Bc + FP16 ex2 ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n", p.name, p.multiProcessorCount, p.clockRate / 1000);
    printf("Dispatch: s<=512 → Bc=32, s<=2048 → Bc=64, s>2048 → Bc=128\n\n");

    // Correctness
    printf("--- Correctness ---\n");
    struct
    {
        int h, s, d;
    } cfgs[] = {
        {1, 32, 128},
        {1, 64, 128},
        {2, 128, 128},
        {1, 256, 128},
        {1, 512, 128},
        {32, 1024, 128},
        {32, 2048, 128},
        {32, 4096, 128},
    };
    for (auto &c : cfgs)
    {
        size_t n = (size_t)c.h * c.s * c.d;
        size_t bytes = n * 2;
        __half *dQ, *dK, *dV, *dO_ref, *dO20, *dO46;
        CK(cudaMalloc(&dQ, bytes));
        CK(cudaMalloc(&dK, bytes));
        CK(cudaMalloc(&dV, bytes));
        CK(cudaMalloc(&dO_ref, bytes));
        CK(cudaMalloc(&dO20, bytes));
        CK(cudaMalloc(&dO46, bytes));

        std::vector<__half> hv(n);
        srand(42);
        for (size_t i = 0; i < n; i++)
            hv[i] = __float2half((float)(rand() % 1000 - 500) / 1000.f);
        CK(cudaMemcpy(dQ, hv.data(), bytes, cudaMemcpyHostToDevice));
        for (size_t i = 0; i < n; i++)
            hv[i] = __float2half((float)(rand() % 1000 - 500) / 1000.f);
        CK(cudaMemcpy(dK, hv.data(), bytes, cudaMemcpyHostToDevice));
        for (size_t i = 0; i < n; i++)
            hv[i] = __float2half((float)(rand() % 1000 - 500) / 1000.f);
        CK(cudaMemcpy(dV, hv.data(), bytes, cudaMemcpyHostToDevice));

        CK(cudaMemset(dO_ref, 0, bytes));
        CK(cudaMemset(dO20, 0, bytes));
        CK(cudaMemset(dO46, 0, bytes));
        flash_attention_forward(dQ, dK, dV, dO_ref, c.h, c.s, c.d, 1, nullptr);
        flash_attention_v20_forward(dQ, dK, dV, dO20, c.h, c.s, c.d, 1, nullptr);
        flash_attention_v46_forward(dQ, dK, dV, dO46, c.h, c.s, c.d, 1, nullptr);
        CK(cudaDeviceSynchronize());

        std::vector<__half> hr(n), h20(n), h46(n);
        CK(cudaMemcpy(hr.data(), dO_ref, bytes, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h20.data(), dO20, bytes, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h46.data(), dO46, bytes, cudaMemcpyDeviceToHost));

        float max46 = 0, max46_20 = 0;
        int err46 = 0;
        for (size_t i = 0; i < n; i++)
        {
            float rv = __half2float(hr[i]);
            float v20 = __half2float(h20[i]);
            float v46 = __half2float(h46[i]);
            float ae = fabsf(rv - v46);
            float ae_cross = fabsf(v20 - v46);
            if (ae > max46)
                max46 = ae;
            if (ae_cross > max46_20)
                max46_20 = ae_cross;
            float thr = fmaxf(0.003f, fabsf(rv) * 0.08f);
            if (ae > thr)
                err46++;
        }
        const char *bc_tag = (c.s <= 512) ? "Bc32" : (c.s <= 2048) ? "Bc64"
                                                                   : "Bc128";
        printf("  %dh x %4ds [%s]: ref=%.4f err=%d %s  vs_v20=%.4f\n",
               c.h, c.s, bc_tag, max46, err46, err46 == 0 ? "PASS" : "FAIL", max46_20);

        cudaFree(dQ);
        cudaFree(dK);
        cudaFree(dV);
        cudaFree(dO_ref);
        cudaFree(dO20);
        cudaFree(dO46);
    }

    // Performance
    printf("\n--- Performance: v20 vs v46 ---\n");
    printf("%-12s %5s %10s %10s %8s %7s\n", "Config", "Bc", "v20 (T)", "v46 (T)", "delta", "peak%");
    printf("------------------------------------------------------------\n");
    Timer t;
    struct
    {
        int h, s;
        const char *l;
    } pcfgs[] = {
        {32, 256, "7B-256"},
        {32, 512, "7B-512"},
        {32, 1024, "7B-1K"},
        {32, 2048, "7B-2K"},
        {32, 4096, "7B-4K"},
        {32, 8192, "7B-8K"},
        {40, 512, "13B-512"},
        {64, 512, "70B-512"},
        {64, 2048, "70B-2K"},
        {64, 4096, "70B-4K"},
    };
    for (auto &c : pcfgs)
    {
        int n = c.h * c.s * 128;
        double flops = c.h * (4.0 * c.s * (double)c.s * 128) / 2.0;
        __half *dQ, *dK, *dV, *dO;
        CK(cudaMalloc(&dQ, (size_t)n * 2));
        CK(cudaMalloc(&dK, (size_t)n * 2));
        CK(cudaMalloc(&dV, (size_t)n * 2));
        CK(cudaMalloc(&dO, (size_t)n * 2));
        fill_random((uint16_t *)dQ, n);
        fill_random((uint16_t *)dK, n);
        fill_random((uint16_t *)dV, n);
        int it = (c.s <= 1024) ? 100 : (c.s <= 4096 ? 20 : 10);

        for (int i = 0; i < 3; i++)
            flash_attention_v20_forward(dQ, dK, dV, dO, c.h, c.s, 128, 1, nullptr);
        CK(cudaDeviceSynchronize());
        t.start();
        for (int i = 0; i < it; i++)
            flash_attention_v20_forward(dQ, dK, dV, dO, c.h, c.s, 128, 1, nullptr);
        float ms20 = t.stop();

        for (int i = 0; i < 3; i++)
            flash_attention_v46_forward(dQ, dK, dV, dO, c.h, c.s, 128, 1, nullptr);
        CK(cudaDeviceSynchronize());
        t.start();
        for (int i = 0; i < it; i++)
            flash_attention_v46_forward(dQ, dK, dV, dO, c.h, c.s, 128, 1, nullptr);
        float ms46 = t.stop();

        double t20 = flops / (ms20 / it / 1000.0) / 1e12;
        double t46 = flops / (ms46 / it / 1000.0) / 1e12;
        int bc = (c.s <= 512) ? 32 : (c.s <= 2048) ? 64
                                                   : 128;
        printf("%-12s %5d %8.2f T %8.2f T %+7.2f %6.1f%%\n",
               c.l, bc, t20, t46, t46 - t20, t46 / 165.2 * 100);
        cudaFree(dQ);
        cudaFree(dK);
        cudaFree(dV);
        cudaFree(dO);
    }
    printf("\nPeak = 165.2 T | v20: 151T | v54: 153T | FA2: 159T @ 7B-8K\n");
}
