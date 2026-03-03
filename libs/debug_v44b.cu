// debug_v44b.cu - v20 vs v44b (minimal exp2f) vs scalar ref
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
    int flash_attention_v44b_forward(const void *, const void *, const void *, void *,
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

int main()
{
    printf("=== v44b: minimal exp2f change from v20 ===\n\n");

    struct
    {
        int h, s, d;
    } cfgs[] = {
        {1, 64, 128},
        {1, 128, 128},
        {4, 128, 128},
        {2, 256, 128},
        {32, 1024, 128},
        {32, 2048, 128},
    };

    for (auto &c : cfgs)
    {
        size_t n = (size_t)c.h * c.s * c.d;
        size_t bytes = n * 2;

        __half *dQ, *dK, *dV, *dO_ref, *dO20, *dO44b;
        CK(cudaMalloc(&dQ, bytes));
        CK(cudaMalloc(&dK, bytes));
        CK(cudaMalloc(&dV, bytes));
        CK(cudaMalloc(&dO_ref, bytes));
        CK(cudaMalloc(&dO20, bytes));
        CK(cudaMalloc(&dO44b, bytes));

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

        // causal
        CK(cudaMemset(dO_ref, 0, bytes));
        CK(cudaMemset(dO20, 0, bytes));
        CK(cudaMemset(dO44b, 0, bytes));

        flash_attention_forward(dQ, dK, dV, dO_ref, c.h, c.s, c.d, 1, nullptr);
        flash_attention_v20_forward(dQ, dK, dV, dO20, c.h, c.s, c.d, 1, nullptr);
        flash_attention_v44b_forward(dQ, dK, dV, dO44b, c.h, c.s, c.d, 1, nullptr);
        CK(cudaDeviceSynchronize());

        std::vector<__half> hr(n), h20(n), h44b(n);
        CK(cudaMemcpy(hr.data(), dO_ref, bytes, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h20.data(), dO20, bytes, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h44b.data(), dO44b, bytes, cudaMemcpyDeviceToHost));

        float max20 = 0, max44b = 0, max20_44b = 0;
        int err20 = 0, err44b = 0;
        for (size_t i = 0; i < n; i++)
        {
            float rv = __half2float(hr[i]);
            float v20 = __half2float(h20[i]);
            float v44b = __half2float(h44b[i]);
            float ae20 = fabsf(rv - v20);
            float ae44b = fabsf(rv - v44b);
            float ae_cross = fabsf(v20 - v44b);
            if (ae20 > max20)
                max20 = ae20;
            if (ae44b > max44b)
                max44b = ae44b;
            if (ae_cross > max20_44b)
                max20_44b = ae_cross;
            float re20 = (fabsf(rv) > 1e-6f) ? ae20 / fabsf(rv) : 0;
            float re44b = (fabsf(rv) > 1e-6f) ? ae44b / fabsf(rv) : 0;
            if (ae20 > 0.01f && re20 > 0.1f)
                err20++;
            if (ae44b > 0.01f && re44b > 0.1f)
                err44b++;
        }
        printf("causal %dh x %ds: v20_vs_ref=%.4f(%d) v44b_vs_ref=%.4f(%d) v20_vs_v44b=%.4f\n",
               c.h, c.s, max20, err20, max44b, err44b, max20_44b);

        cudaFree(dQ);
        cudaFree(dK);
        cudaFree(dV);
        cudaFree(dO_ref);
        cudaFree(dO20);
        cudaFree(dO44b);
    }
}
