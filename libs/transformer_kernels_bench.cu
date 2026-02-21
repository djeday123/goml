// =============================================================================
// Transformer Kernels Benchmark
// =============================================================================
// Build: nvcc -O3 -arch=sm_89 -std=c++17 \
//        transformer_kernels.cu transformer_kernels_bench.cu \
//        -o transformer_bench -lcudart
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C"
{
    int rmsnorm_forward(const void *x, const void *weight, void *y,
                        int rows, int hidden, float eps, void *stream);
    int swiglu_forward(const void *gate, const void *up, void *y,
                       int n, void *stream);
    int rope_forward(const void *x, const void *pos, void *y,
                     int batch, int seq_len, int num_heads, int head_dim,
                     float theta_base, void *stream);
    int attention_forward(const void *Q, const void *K, const void *V, void *O,
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

// Fill device memory with random FP16 values in [-1, 1]
void fill_random_fp16(uint16_t *d_ptr, int n)
{
    uint16_t *h = (uint16_t *)malloc(n * 2);
    for (int i = 0; i < n; i++)
        h[i] = f2h(((float)(rand() % 2001) - 1000.0f) / 1000.0f);
    CK(cudaMemcpy(d_ptr, h, n * 2, cudaMemcpyHostToDevice));
    free(h);
}

// ---- Timer helper ----
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
// RMSNorm test
// =============================================================================
void test_rmsnorm()
{
    printf("--- RMSNorm ---\n");
    int rows = 4, hidden = 128;
    float eps = 1e-6f;
    int n = rows * hidden;

    uint16_t *hx = (uint16_t *)malloc(n * 2);
    uint16_t *hw = (uint16_t *)malloc(hidden * 2);
    uint16_t *hy = (uint16_t *)malloc(n * 2);

    srand(42);
    for (int i = 0; i < n; i++)
        hx[i] = f2h(((float)(rand() % 201) - 100.0f) / 100.0f);
    for (int i = 0; i < hidden; i++)
        hw[i] = f2h(((float)(rand() % 201) - 100.0f) / 100.0f + 1.0f);

    // CPU reference
    float *ref = (float *)malloc(n * 4);
    for (int r = 0; r < rows; r++)
    {
        float sum_sq = 0;
        for (int j = 0; j < hidden; j++)
        {
            float v = fp16f(hx[r * hidden + j]);
            sum_sq += v * v;
        }
        float rms_inv = 1.0f / sqrtf(sum_sq / hidden + eps);
        for (int j = 0; j < hidden; j++)
        {
            float v = fp16f(hx[r * hidden + j]);
            float w = fp16f(hw[j]);
            ref[r * hidden + j] = v * rms_inv * w;
        }
    }

    void *dx, *dw, *dy;
    CK(cudaMalloc(&dx, n * 2));
    CK(cudaMalloc(&dw, hidden * 2));
    CK(cudaMalloc(&dy, n * 2));
    CK(cudaMemcpy(dx, hx, n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw, hw, hidden * 2, cudaMemcpyHostToDevice));

    int rc = rmsnorm_forward(dx, dw, dy, rows, hidden, eps, nullptr);
    CK(cudaDeviceSynchronize());
    printf("  launch: %s\n", rc == 0 ? "OK" : "FAIL");

    CK(cudaMemcpy(hy, dy, n * 2, cudaMemcpyDeviceToHost));
    float max_err = 0;
    for (int i = 0; i < n; i++)
    {
        float ae = fabsf(fp16f(hy[i]) - ref[i]);
        if (ae > max_err)
            max_err = ae;
    }
    printf("  correctness %dx%d: max_err=%.6f → %s\n",
           rows, hidden, max_err, max_err < 0.02f ? "PASS" : "FAIL");

    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dy);
    free(hx);
    free(hw);
    free(hy);
    free(ref);
}

void bench_rmsnorm()
{
    int configs[][2] = {{2048, 4096}, {2048, 8192}, {4096, 4096}, {8192, 4096}};
    Timer t;
    for (auto &c : configs)
    {
        int rows = c[0], hidden = c[1];
        int n = rows * hidden;
        void *dx, *dw, *dy;
        CK(cudaMalloc(&dx, n * 2));
        CK(cudaMalloc(&dw, hidden * 2));
        CK(cudaMalloc(&dy, n * 2));
        fill_random_fp16((uint16_t *)dx, n);
        fill_random_fp16((uint16_t *)dw, hidden);

        // Warmup
        for (int i = 0; i < 10; i++)
            rmsnorm_forward(dx, dw, dy, rows, hidden, 1e-6f, nullptr);
        CK(cudaDeviceSynchronize());

        int it = 1000;
        t.start();
        for (int i = 0; i < it; i++)
            rmsnorm_forward(dx, dw, dy, rows, hidden, 1e-6f, nullptr);
        float ms = t.stop();

        double gb = 2.0 * n * 2.0 * 2 / 1e9; // read x + write y, skip weight (cached)
        double bw = gb / (ms / it / 1000.0);
        printf("  %4dx%4d  %7.3f us  %6.1f GB/s\n", rows, hidden, ms / it * 1000, bw);
        cudaFree(dx);
        cudaFree(dw);
        cudaFree(dy);
    }
}

// =============================================================================
// SwiGLU test
// =============================================================================
void test_swiglu()
{
    printf("--- SwiGLU ---\n");
    int n = 1024;
    uint16_t *hg = (uint16_t *)malloc(n * 2);
    uint16_t *hu = (uint16_t *)malloc(n * 2);
    uint16_t *hy = (uint16_t *)malloc(n * 2);

    srand(42);
    for (int i = 0; i < n; i++)
    {
        hg[i] = f2h(((float)(rand() % 401) - 200.0f) / 100.0f);
        hu[i] = f2h(((float)(rand() % 401) - 200.0f) / 100.0f);
    }

    float *ref = (float *)malloc(n * 4);
    for (int i = 0; i < n; i++)
    {
        float g = fp16f(hg[i]);
        float u = fp16f(hu[i]);
        float silu = g / (1.0f + expf(-g));
        ref[i] = silu * u;
    }

    void *dg, *du, *dy;
    CK(cudaMalloc(&dg, n * 2));
    CK(cudaMalloc(&du, n * 2));
    CK(cudaMalloc(&dy, n * 2));
    CK(cudaMemcpy(dg, hg, n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(du, hu, n * 2, cudaMemcpyHostToDevice));

    int rc = swiglu_forward(dg, du, dy, n, nullptr);
    CK(cudaDeviceSynchronize());
    printf("  launch: %s\n", rc == 0 ? "OK" : "FAIL");

    CK(cudaMemcpy(hy, dy, n * 2, cudaMemcpyDeviceToHost));
    float max_err = 0;
    for (int i = 0; i < n; i++)
    {
        float ae = fabsf(fp16f(hy[i]) - ref[i]);
        if (ae > max_err)
            max_err = ae;
    }
    printf("  correctness n=%d: max_err=%.6f → %s\n",
           n, max_err, max_err < 0.02f ? "PASS" : "FAIL");

    cudaFree(dg);
    cudaFree(du);
    cudaFree(dy);
    free(hg);
    free(hu);
    free(hy);
    free(ref);
}

void bench_swiglu()
{
    int sizes[] = {4096 * 2048, 11008 * 2048, 4096 * 4096, 11008 * 4096};
    const char *labels[] = {"4Kx2K", "11Kx2K", "4Kx4K", "11Kx4K"};
    Timer t;
    for (int si = 0; si < 4; si++)
    {
        int n = sizes[si];
        void *dg, *du, *dy;
        CK(cudaMalloc(&dg, (size_t)n * 2));
        CK(cudaMalloc(&du, (size_t)n * 2));
        CK(cudaMalloc(&dy, (size_t)n * 2));
        fill_random_fp16((uint16_t *)dg, n);
        fill_random_fp16((uint16_t *)du, n);

        for (int i = 0; i < 10; i++)
            swiglu_forward(dg, du, dy, n, nullptr);
        CK(cudaDeviceSynchronize());

        int it = 1000;
        t.start();
        for (int i = 0; i < it; i++)
            swiglu_forward(dg, du, dy, n, nullptr);
        float ms = t.stop();

        double gb = (double)n * 2.0 * 3 / 1e9; // read gate + up + write y
        double bw = gb / (ms / it / 1000.0);
        printf("  %-8s n=%9d  %7.3f us  %6.1f GB/s\n", labels[si], n, ms / it * 1000, bw);
        cudaFree(dg);
        cudaFree(du);
        cudaFree(dy);
    }
}

// =============================================================================
// RoPE test
// =============================================================================
void test_rope()
{
    printf("--- RoPE ---\n");
    int batch = 1, seq = 4, heads = 2, hdim = 8;
    int n = batch * seq * heads * hdim;

    uint16_t *hx = (uint16_t *)malloc(n * 2);
    uint16_t *hy = (uint16_t *)malloc(n * 2);
    int *hpos = (int *)malloc(seq * 4);

    srand(42);
    for (int i = 0; i < n; i++)
        hx[i] = f2h(((float)(rand() % 201) - 100.0f) / 100.0f);
    for (int s = 0; s < seq; s++)
        hpos[s] = s;

    // CPU reference
    float *ref = (float *)malloc(n * 4);
    float base = 10000.0f;
    for (int b = 0; b < batch; b++)
        for (int s = 0; s < seq; s++)
            for (int h = 0; h < heads; h++)
                for (int i = 0; i < hdim / 2; i++)
                {
                    int off = ((b * seq + s) * heads + h) * hdim + 2 * i;
                    float freq = 1.0f / powf(base, (float)(2 * i) / (float)hdim);
                    float theta = (float)hpos[s] * freq;
                    float x0 = fp16f(hx[off]), x1 = fp16f(hx[off + 1]);
                    ref[off] = x0 * cosf(theta) - x1 * sinf(theta);
                    ref[off + 1] = x0 * sinf(theta) + x1 * cosf(theta);
                }

    void *dx, *dy;
    int *dpos;
    CK(cudaMalloc(&dx, n * 2));
    CK(cudaMalloc(&dy, n * 2));
    CK(cudaMalloc(&dpos, seq * 4));
    CK(cudaMemcpy(dx, hx, n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dpos, hpos, seq * 4, cudaMemcpyHostToDevice));

    int rc = rope_forward(dx, dpos, dy, batch, seq, heads, hdim, base, nullptr);
    CK(cudaDeviceSynchronize());
    printf("  launch: %s\n", rc == 0 ? "OK" : "FAIL");

    CK(cudaMemcpy(hy, dy, n * 2, cudaMemcpyDeviceToHost));
    float max_err = 0;
    for (int i = 0; i < n; i++)
    {
        float ae = fabsf(fp16f(hy[i]) - ref[i]);
        if (ae > max_err)
            max_err = ae;
    }
    printf("  correctness b=%d s=%d h=%d d=%d: max_err=%.6f → %s\n",
           batch, seq, heads, hdim, max_err, max_err < 0.01f ? "PASS" : "FAIL");

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dpos);
    free(hx);
    free(hy);
    free(hpos);
    free(ref);
}

void bench_rope()
{
    struct
    {
        int b, s, h, d;
        const char *label;
    } configs[] = {
        {1, 2048, 32, 128, "7B-2K"},
        {1, 4096, 32, 128, "7B-4K"},
        {1, 2048, 40, 128, "13B-2K"},
        {1, 2048, 64, 128, "70B-2K"},
    };
    Timer t;
    for (auto &c : configs)
    {
        int n = c.b * c.s * c.h * c.d;
        void *dx, *dy;
        int *dpos;
        CK(cudaMalloc(&dx, n * 2));
        CK(cudaMalloc(&dy, n * 2));
        CK(cudaMalloc(&dpos, c.s * 4));
        fill_random_fp16((uint16_t *)dx, n);
        // Fill pos = 0,1,2,...
        int *hpos = (int *)malloc(c.s * 4);
        for (int i = 0; i < c.s; i++)
            hpos[i] = i;
        CK(cudaMemcpy(dpos, hpos, c.s * 4, cudaMemcpyHostToDevice));

        for (int i = 0; i < 10; i++)
            rope_forward(dx, dpos, dy, c.b, c.s, c.h, c.d, 10000.0f, nullptr);
        CK(cudaDeviceSynchronize());

        int it = 1000;
        t.start();
        for (int i = 0; i < it; i++)
            rope_forward(dx, dpos, dy, c.b, c.s, c.h, c.d, 10000.0f, nullptr);
        float ms = t.stop();

        double gb = (double)n * 2.0 * 2 / 1e9; // read + write
        double bw = gb / (ms / it / 1000.0);
        printf("  %-8s [%d,%d,%d,%d]  %7.3f us  %6.1f GB/s\n",
               c.label, c.b, c.s, c.h, c.d, ms / it * 1000, bw);
        cudaFree(dx);
        cudaFree(dy);
        cudaFree(dpos);
        free(hpos);
    }
}

// =============================================================================
// Attention test
// =============================================================================
void test_attention()
{
    printf("--- Attention ---\n");
    int batch = 1, heads = 2, seq = 4, hdim = 8;
    int n = batch * heads * seq * hdim;

    uint16_t *hQ = (uint16_t *)malloc(n * 2);
    uint16_t *hK = (uint16_t *)malloc(n * 2);
    uint16_t *hV = (uint16_t *)malloc(n * 2);
    uint16_t *hO = (uint16_t *)malloc(n * 2);

    srand(42);
    for (int i = 0; i < n; i++)
    {
        hQ[i] = f2h(((float)(rand() % 201) - 100.0f) / 100.0f);
        hK[i] = f2h(((float)(rand() % 201) - 100.0f) / 100.0f);
        hV[i] = f2h(((float)(rand() % 201) - 100.0f) / 100.0f);
    }

    // CPU reference (causal)
    float *ref = (float *)malloc(n * 4);
    float scale = 1.0f / sqrtf((float)hdim);
    for (int b = 0; b < batch; b++)
        for (int h = 0; h < heads; h++)
        {
            int base = (b * heads + h) * seq * hdim;
            for (int i = 0; i < seq; i++)
            {
                // Compute scores
                float scores[64]; // max seq=64 for test
                float max_s = -1e30f;
                for (int j = 0; j <= i; j++)
                { // causal
                    float dot = 0;
                    for (int d = 0; d < hdim; d++)
                        dot += fp16f(hQ[base + i * hdim + d]) * fp16f(hK[base + j * hdim + d]);
                    scores[j] = dot * scale;
                    if (scores[j] > max_s)
                        max_s = scores[j];
                }
                // Softmax
                float sum = 0;
                for (int j = 0; j <= i; j++)
                {
                    scores[j] = expf(scores[j] - max_s);
                    sum += scores[j];
                }
                for (int j = 0; j <= i; j++)
                    scores[j] /= sum;
                // Output
                for (int d = 0; d < hdim; d++)
                {
                    float acc = 0;
                    for (int j = 0; j <= i; j++)
                        acc += scores[j] * fp16f(hV[base + j * hdim + d]);
                    ref[base + i * hdim + d] = acc;
                }
            }
        }

    void *dQ, *dK, *dV, *dO;
    CK(cudaMalloc(&dQ, n * 2));
    CK(cudaMalloc(&dK, n * 2));
    CK(cudaMalloc(&dV, n * 2));
    CK(cudaMalloc(&dO, n * 2));
    CK(cudaMemcpy(dQ, hQ, n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, hK, n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, hV, n * 2, cudaMemcpyHostToDevice));

    int rc = attention_forward(dQ, dK, dV, dO, batch, heads, seq, hdim, 1, nullptr);
    CK(cudaDeviceSynchronize());
    printf("  launch: %s\n", rc == 0 ? "OK" : "FAIL");

    CK(cudaMemcpy(hO, dO, n * 2, cudaMemcpyDeviceToHost));
    float max_err = 0;
    for (int i = 0; i < n; i++)
    {
        float ae = fabsf(fp16f(hO[i]) - ref[i]);
        if (ae > max_err)
            max_err = ae;
    }
    printf("  correctness causal b=%d h=%d s=%d d=%d: max_err=%.6f → %s\n",
           batch, heads, seq, hdim, max_err, max_err < 0.02f ? "PASS" : "FAIL");

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

void bench_attention()
{
    struct
    {
        int b, h, s, d;
        const char *label;
    } configs[] = {
        {1, 32, 256, 128, "7B-256"},
        {1, 32, 512, 128, "7B-512"},
        {1, 32, 1024, 128, "7B-1K"},
        {1, 32, 2048, 128, "7B-2K"},
        {1, 40, 512, 128, "13B-512"},
        {1, 64, 512, 128, "70B-512"},
    };
    Timer t;
    for (auto &c : configs)
    {
        int n = c.b * c.h * c.s * c.d;
        int smem_needed = (256 / 32 + c.s) * 4;
        if (smem_needed > 100 * 1024)
        {
            printf("  %-9s [%d,%d,%d,%d]  SKIP (seq too long, need FlashAttn)\n",
                   c.label, c.b, c.h, c.s, c.d);
            continue;
        }

        void *dQ, *dK, *dV, *dO;
        CK(cudaMalloc(&dQ, (size_t)n * 2));
        CK(cudaMalloc(&dK, (size_t)n * 2));
        CK(cudaMalloc(&dV, (size_t)n * 2));
        CK(cudaMalloc(&dO, (size_t)n * 2));
        fill_random_fp16((uint16_t *)dQ, n);
        fill_random_fp16((uint16_t *)dK, n);
        fill_random_fp16((uint16_t *)dV, n);

        for (int i = 0; i < 5; i++)
            attention_forward(dQ, dK, dV, dO, c.b, c.h, c.s, c.d, 1, nullptr);
        CK(cudaDeviceSynchronize());

        int it = 100;
        t.start();
        for (int i = 0; i < it; i++)
            attention_forward(dQ, dK, dV, dO, c.b, c.h, c.s, c.d, 1, nullptr);
        float ms = t.stop();

        // FLOPs: 2*b*h*s*s*d (QK^T) + 2*b*h*s*s*d (attn@V) = 4*b*h*s²*d
        double flops = 4.0 * c.b * c.h * (double)c.s * c.s * c.d;
        double tflops = flops / (ms / it / 1000.0) / 1e12;
        printf("  %-9s [%d,%d,%d,%d]  %8.3f ms  %6.2f TFLOPS\n",
               c.label, c.b, c.h, c.s, c.d, ms / it, tflops);
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
    printf("=== Transformer Kernels Benchmark ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n\n", p.name, p.multiProcessorCount, p.clockRate / 1000);

    // Correctness
    test_rmsnorm();
    test_swiglu();
    test_rope();
    test_attention();

    // Performance
    printf("\n=== Performance ===\n\n");

    printf("--- RMSNorm (bandwidth-bound) ---\n");
    bench_rmsnorm();

    printf("\n--- SwiGLU (bandwidth-bound) ---\n");
    bench_swiglu();

    printf("\n--- RoPE (bandwidth-bound) ---\n");
    bench_rope();

    printf("\n--- Attention (compute-bound, basic — NOT FlashAttention) ---\n");
    bench_attention();

    printf("\nNote: Attention is O(s²) basic impl. FlashAttention needed for seq > 2K.\n");
    return 0;
}
