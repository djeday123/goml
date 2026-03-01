// =============================================================================
// FlashAttention v2 DB-Swizzle Benchmark
// =============================================================================
// Compares: v2-DB (160T) vs v2-DB-SW (swizzled KV_s)
//
// Build:
//   nvcc -O3 -arch=sm_89 -std=c++17 \
//     libs/flash_attention_v2_dbsw.cu libs/flash_attention_v2_db.cu \
//     libs/flash_attention_v2.cu libs/flash_attention.cu \
//     libs/transformer_kernels.cu libs/flash_attention_v2_dbsw_bench.cu \
//     -o runs/flash_dbsw_bench -lcudart && ./runs/flash_dbsw_bench
// =============================================================================

#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C"
{
    int flash_attention_v2_db_forward(
        const void *Q, const void *K, const void *V, void *O,
        int total_heads, int seq_len, int head_dim,
        int causal, void *stream);

    int flash_attention_v2_dbsw_forward(
        const void *Q, const void *K, const void *V, void *O,
        int total_heads, int seq_len, int head_dim,
        int causal, void *stream);
}

#define CK(c)                                                    \
    do                                                           \
    {                                                            \
        cudaError_t e = (c);                                     \
        if (e != cudaSuccess)                                    \
        {                                                        \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                       \
            exit(1);                                             \
        }                                                        \
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

// =============================================================================
// Correctness: v2-DB-SW vs v2-DB reference
// =============================================================================

void test_correctness()
{
    printf("--- Correctness (v2-DB-SW vs v2-DB reference) ---\n");

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
        size_t bytes = n * sizeof(__half);

        __half *dQ, *dK, *dV, *dO_ref, *dO_sw;
        CK(cudaMalloc(&dQ, bytes));
        CK(cudaMalloc(&dK, bytes));
        CK(cudaMalloc(&dV, bytes));
        CK(cudaMalloc(&dO_ref, bytes));
        CK(cudaMalloc(&dO_sw, bytes));

        std::vector<__half> host(n);
        for (size_t i = 0; i < n; i++)
            host[i] = __float2half((float)(rand() % 1000 - 500) / 1000.0f);
        CK(cudaMemcpy(dQ, host.data(), bytes, cudaMemcpyHostToDevice));
        for (size_t i = 0; i < n; i++)
            host[i] = __float2half((float)(rand() % 1000 - 500) / 1000.0f);
        CK(cudaMemcpy(dK, host.data(), bytes, cudaMemcpyHostToDevice));
        for (size_t i = 0; i < n; i++)
            host[i] = __float2half((float)(rand() % 1000 - 500) / 1000.0f);
        CK(cudaMemcpy(dV, host.data(), bytes, cudaMemcpyHostToDevice));

        // v2-DB reference
        CK(cudaMemset(dO_ref, 0, bytes));
        int rc1 = flash_attention_v2_db_forward(dQ, dK, dV, dO_ref, c.h, c.s, c.d, 1, nullptr);
        CK(cudaDeviceSynchronize());

        // v2-DB-SW
        CK(cudaMemset(dO_sw, 0, bytes));
        int rc2 = flash_attention_v2_dbsw_forward(dQ, dK, dV, dO_sw, c.h, c.s, c.d, 1, nullptr);
        CK(cudaDeviceSynchronize());

        if (rc1 != 0 || rc2 != 0)
        {
            printf("  %dh×%ds×%dd  LAUNCH FAILED (rc1=%d, rc2=%d)\n", c.h, c.s, c.d, rc1, rc2);
            cudaFree(dQ);
            cudaFree(dK);
            cudaFree(dV);
            cudaFree(dO_ref);
            cudaFree(dO_sw);
            continue;
        }

        std::vector<__half> h_ref(n), h_sw(n);
        CK(cudaMemcpy(h_ref.data(), dO_ref, bytes, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_sw.data(), dO_sw, bytes, cudaMemcpyDeviceToHost));

        float max_ae = 0, max_re = 0;
        int errors = 0;
        for (size_t i = 0; i < n; i++)
        {
            float ref_v = __half2float(h_ref[i]);
            float sw_v = __half2float(h_sw[i]);
            float ae = fabsf(ref_v - sw_v);
            float re = (fabsf(ref_v) > 1e-6f) ? ae / fabsf(ref_v) : 0;
            if (ae > max_ae)
                max_ae = ae;
            if (re > max_re)
                max_re = re;
            if (ae > 0.005f && re > 0.1f)
                errors++;
        }

        printf("  %dh×%ds×%dd       abs=%.4f rel=%.4f err=%d → %s\n",
               c.h, c.s, c.d, max_ae, max_re, errors,
               errors == 0 ? "PASS" : "FAIL");

        cudaFree(dQ);
        cudaFree(dK);
        cudaFree(dV);
        cudaFree(dO_ref);
        cudaFree(dO_sw);
    }
}

// =============================================================================
// Performance
// =============================================================================

void bench()
{
    printf("--- Performance: v2-DB (160T) vs v2-DB-SW (swizzle) ---\n");
    printf("%-12s %10s %10s  %6s\n",
           "Config", "v2-DB", "v2-DB-SW", "SW/DB");
    printf("---------------------------------------------\n");

    struct
    {
        const char *label;
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
        double flops = (double)heads * (4.0 * c.s * c.s * c.d);

        __half *dQ, *dK, *dV, *dO;
        CK(cudaMalloc(&dQ, bytes));
        CK(cudaMalloc(&dK, bytes));
        CK(cudaMalloc(&dV, bytes));
        CK(cudaMalloc(&dO, bytes));
        CK(cudaMemset(dQ, 0x3C, bytes));
        CK(cudaMemset(dK, 0x3C, bytes));
        CK(cudaMemset(dV, 0x3C, bytes));

        int it = (c.s <= 1024) ? 100 : 20;

        // v2-DB
        CK(cudaMemset(dO, 0, bytes));
        for (int i = 0; i < 3; i++)
            flash_attention_v2_db_forward(dQ, dK, dV, dO, heads, c.s, c.d, 1, nullptr);
        CK(cudaDeviceSynchronize());
        t.start();
        for (int i = 0; i < it; i++)
            flash_attention_v2_db_forward(dQ, dK, dV, dO, heads, c.s, c.d, 1, nullptr);
        float ms_db = t.stop();
        CK(cudaGetLastError());
        double db_tflops = flops / (ms_db / it / 1000.0) / 1e12;

        // v2-DB-SW
        CK(cudaMemset(dO, 0, bytes));
        for (int i = 0; i < 3; i++)
            flash_attention_v2_dbsw_forward(dQ, dK, dV, dO, heads, c.s, c.d, 1, nullptr);
        CK(cudaDeviceSynchronize());
        t.start();
        for (int i = 0; i < it; i++)
            flash_attention_v2_dbsw_forward(dQ, dK, dV, dO, heads, c.s, c.d, 1, nullptr);
        float ms_sw = t.stop();
        CK(cudaGetLastError());
        double sw_tflops = flops / (ms_sw / it / 1000.0) / 1e12;

        printf("%-12s %8.2f T %8.2f T  %5.2fx\n",
               c.label, db_tflops, sw_tflops, sw_tflops / db_tflops);

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
    printf("=== FlashAttention v2 DB-Swizzle Benchmark ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n\n", p.name, p.multiProcessorCount, p.clockRate / 1000);

    test_correctness();
    printf("\n");
    bench();

    return 0;
}
