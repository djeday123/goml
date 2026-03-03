// debug_v44c.cu - test v44c (scale first + exp2f) vs v20 vs ref
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
    int flash_attention_v44c_forward(const void *, const void *, const void *, void *,
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

int main()
{
    printf("=== v44c: scale first + exp2f ===\n\n");

    // Correctness
    struct
    {
        int h, s, d;
    } cfgs[] = {
        {1, 64, 128},
        {1, 128, 128},
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

        __half *dQ, *dK, *dV, *dO_ref, *dO20, *dOc;
        CK(cudaMalloc(&dQ, bytes));
        CK(cudaMalloc(&dK, bytes));
        CK(cudaMalloc(&dV, bytes));
        CK(cudaMalloc(&dO_ref, bytes));
        CK(cudaMalloc(&dO20, bytes));
        CK(cudaMalloc(&dOc, bytes));

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
        CK(cudaMemset(dOc, 0, bytes));

        flash_attention_forward(dQ, dK, dV, dO_ref, c.h, c.s, c.d, 1, nullptr);
        flash_attention_v20_forward(dQ, dK, dV, dO20, c.h, c.s, c.d, 1, nullptr);
        flash_attention_v44c_forward(dQ, dK, dV, dOc, c.h, c.s, c.d, 1, nullptr);
        CK(cudaDeviceSynchronize());

        std::vector<__half> hr(n), h20(n), hc(n);
        CK(cudaMemcpy(hr.data(), dO_ref, bytes, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h20.data(), dO20, bytes, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hc.data(), dOc, bytes, cudaMemcpyDeviceToHost));

        float max_c_ref = 0, max_c_20 = 0;
        int err_c = 0;
        for (size_t i = 0; i < n; i++)
        {
            float rv = __half2float(hr[i]);
            float v20 = __half2float(h20[i]);
            float vc = __half2float(hc[i]);
            float ae_cr = fabsf(rv - vc);
            float ae_c20 = fabsf(v20 - vc);
            if (ae_cr > max_c_ref)
                max_c_ref = ae_cr;
            if (ae_c20 > max_c_20)
                max_c_20 = ae_c20;
            float re = (fabsf(rv) > 1e-6f) ? ae_cr / fabsf(rv) : 0;
            if (ae_cr > 0.01f && re > 0.1f)
                err_c++;
        }
        printf("%dh x %ds: v44c_vs_ref=%.4f(%d) v44c_vs_v20=%.6f -> %s\n",
               c.h, c.s, max_c_ref, err_c, max_c_20, err_c == 0 ? "PASS" : "FAIL");

        cudaFree(dQ);
        cudaFree(dK);
        cudaFree(dV);
        cudaFree(dO_ref);
        cudaFree(dO20);
        cudaFree(dOc);
    }

    // Quick perf
    printf("\n--- Performance ---\n");
    Timer t;
    float peak = 165.2f;
    struct
    {
        const char *l;
        int h, s;
    } pcfgs[] = {
        {"7B-2K", 32, 2048},
        {"7B-4K", 32, 4096},
        {"7B-8K", 32, 8192},
    };
    for (auto &c : pcfgs)
    {
        size_t n = (size_t)c.h * c.s * 128;
        size_t bytes = n * 2;
        double flops = (double)c.h * (4.0 * c.s * c.s * 128) / 2.0;
        __half *dQ, *dK, *dV, *dO;
        CK(cudaMalloc(&dQ, bytes));
        CK(cudaMalloc(&dK, bytes));
        CK(cudaMalloc(&dV, bytes));
        CK(cudaMalloc(&dO, bytes));
        CK(cudaMemset(dQ, 0x3C, bytes));
        CK(cudaMemset(dK, 0x3C, bytes));
        CK(cudaMemset(dV, 0x3C, bytes));
        int it = 20;

        for (int i = 0; i < 3; i++)
            flash_attention_v20_forward(dQ, dK, dV, dO, c.h, c.s, 128, 1, nullptr);
        CK(cudaDeviceSynchronize());
        t.start();
        for (int i = 0; i < it; i++)
            flash_attention_v20_forward(dQ, dK, dV, dO, c.h, c.s, 128, 1, nullptr);
        float ms20 = t.stop();

        for (int i = 0; i < 3; i++)
            flash_attention_v44c_forward(dQ, dK, dV, dO, c.h, c.s, 128, 1, nullptr);
        CK(cudaDeviceSynchronize());
        t.start();
        for (int i = 0; i < it; i++)
            flash_attention_v44c_forward(dQ, dK, dV, dO, c.h, c.s, 128, 1, nullptr);
        float msc = t.stop();

        double t20 = flops / (ms20 / it / 1000.0) / 1e12;
        double tc = flops / (msc / it / 1000.0) / 1e12;
        printf("%-8s  v20=%.1fT  v44c=%.1fT  ratio=%.2fx  peak=%.1f%%\n",
               c.l, t20, tc, tc / t20, tc / peak * 100);
        cudaFree(dQ);
        cudaFree(dK);
        cudaFree(dV);
        cudaFree(dO);
    }
}
