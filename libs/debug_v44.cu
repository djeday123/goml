// debug_v44.cu - compare v20 vs v44 element-by-element for s=128
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
    int flash_attention_v44_forward(const void *, const void *, const void *, void *,
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
    int h = 1, s = 128, d = 128;
    size_t n = (size_t)h * s * d;
    size_t bytes = n * 2;

    __half *dQ, *dK, *dV, *dO20, *dO44;
    CK(cudaMalloc(&dQ, bytes));
    CK(cudaMalloc(&dK, bytes));
    CK(cudaMalloc(&dV, bytes));
    CK(cudaMalloc(&dO20, bytes));
    CK(cudaMalloc(&dO44, bytes));

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

    CK(cudaMemset(dO20, 0, bytes));
    CK(cudaMemset(dO44, 0, bytes));

    flash_attention_v20_forward(dQ, dK, dV, dO20, h, s, d, 1, nullptr);
    flash_attention_v44_forward(dQ, dK, dV, dO44, h, s, d, 1, nullptr);
    CK(cudaDeviceSynchronize());

    std::vector<__half> h20(n), h44(n);
    CK(cudaMemcpy(h20.data(), dO20, bytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h44.data(), dO44, bytes, cudaMemcpyDeviceToHost));

    printf("First 10 errors (row, col, v20, v44, diff):\n");
    int cnt = 0;
    for (int row = 0; row < s && cnt < 20; row++)
    {
        for (int col = 0; col < d && cnt < 20; col++)
        {
            float a = __half2float(h20[row * d + col]);
            float b = __half2float(h44[row * d + col]);
            float ae = fabsf(a - b);
            if (ae > 0.005f)
            {
                printf("  [%3d][%3d]  v20=%.6f  v44=%.6f  diff=%.6f\n", row, col, a, b, ae);
                cnt++;
            }
        }
    }
    if (cnt == 0)
        printf("  No errors > 0.005\n");

    // Also test non-causal
    CK(cudaMemset(dO20, 0, bytes));
    CK(cudaMemset(dO44, 0, bytes));
    flash_attention_v20_forward(dQ, dK, dV, dO20, h, s, d, 0, nullptr);
    flash_attention_v44_forward(dQ, dK, dV, dO44, h, s, d, 0, nullptr);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h20.data(), dO20, bytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h44.data(), dO44, bytes, cudaMemcpyDeviceToHost));

    float max_ae_nc = 0;
    for (size_t i = 0; i < n; i++)
    {
        float ae = fabsf(__half2float(h20[i]) - __half2float(h44[i]));
        if (ae > max_ae_nc)
            max_ae_nc = ae;
    }
    printf("\nNon-causal s=128: max_ae=%.6f -> %s\n", max_ae_nc, max_ae_nc < 0.005f ? "PASS" : "FAIL");

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO20);
    cudaFree(dO44);
}
