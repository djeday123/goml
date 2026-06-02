// =============================================================================
// FlashAttention v54 — Backward pass (minimal viable)
//
// Computes dQ, dK, dV given Q, K, V, O, dO and the per-row LSE produced by
// fa54_kernel's `launch_v54_with_lse`.
//
// Math:
//   Forward:  S = Q · Kᵀ · scale,  P = exp(S − LSE),  O = P · V
//   Backward (with D_i = Σ_d dO_i · O_i):
//       dV = Pᵀ · dO
//       dP = dO · Vᵀ
//       dS = P · (dP − D)
//       dQ = dS · K · scale
//       dK = dSᵀ · Q · scale
//
// This is a correctness-first implementation:
//   • One thread per Q-row.
//   • Q-row data lives in registers (per-thread arrays of size ≤ head_dim).
//   • K, V, dO, O are read from global memory each iteration (no tiling).
//   • dV, dK are written via atomicAdd because every Q-row contributes to
//     every K-row. dQ is written directly (one writer per row).
//   • dQ_fp32, dK_fp32, dV_fp32 are intermediate FP32 buffers; a separate
//     cast kernel converts them to FP16 for the public outputs.
//
// Expected to be ~3-10× slower than v54 forward on the same configuration.
// A tiled tensor-core version (v55_backward?) is the natural follow-up
// once this is verified.
// =============================================================================

#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#ifndef BWD_MAX_HEADDIM
#define BWD_MAX_HEADDIM 128
#endif

// -----------------------------------------------------------------------------
// Backward kernel — naive, register-resident Q-row
// -----------------------------------------------------------------------------
__global__ void fa54_backward_kernel(
    const __half *__restrict__ Q, const __half *__restrict__ K,
    const __half *__restrict__ V, const __half *__restrict__ O,
    const __half *__restrict__ dO, const float *__restrict__ LSE,
    float *__restrict__ dQ, float *__restrict__ dK, float *__restrict__ dV,
    int seq_len, int head_dim, int causal, float scale)
{
    int bh = blockIdx.x;
    int q_row = blockIdx.y * blockDim.x + threadIdx.x;
    if (q_row >= seq_len)
        return;
    if (head_dim > BWD_MAX_HEADDIM)
        return; // safety guard

    int hs = seq_len * head_dim;
    const __half *Qh = Q + bh * hs;
    const __half *Kh = K + bh * hs;
    const __half *Vh = V + bh * hs;
    const __half *Oh = O + bh * hs;
    const __half *dOh = dO + bh * hs;
    const float *LSEh = LSE + bh * seq_len;
    float *dQh = dQ + bh * hs;
    float *dKh = dK + bh * hs;
    float *dVh = dV + bh * hs;

    // Load Q[q_row], dO[q_row], O[q_row] into register-resident arrays.
    float q_local[BWD_MAX_HEADDIM];
    float dO_local[BWD_MAX_HEADDIM];
    float dQ_local[BWD_MAX_HEADDIM];

#pragma unroll 16
    for (int d = 0; d < BWD_MAX_HEADDIM; d++)
    {
        if (d < head_dim)
        {
            q_local[d] = __half2float(Qh[q_row * head_dim + d]);
            dO_local[d] = __half2float(dOh[q_row * head_dim + d]);
            dQ_local[d] = 0.0f;
        }
    }

    // D_i = Σ_d  dO_i · O_i   (per row scalar)
    float D = 0.0f;
#pragma unroll 16
    for (int d = 0; d < BWD_MAX_HEADDIM; d++)
    {
        if (d < head_dim)
        {
            D += dO_local[d] * __half2float(Oh[q_row * head_dim + d]);
        }
    }

    float lse = LSEh[q_row];
    int kv_max = causal ? (q_row + 1) : seq_len;

    for (int kv = 0; kv < kv_max; kv++)
    {
        // S = q · k · scale  (single scalar per (q,kv) pair)
        float S = 0.0f;
#pragma unroll 16
        for (int d = 0; d < BWD_MAX_HEADDIM; d++)
        {
            if (d < head_dim)
                S += q_local[d] * __half2float(Kh[kv * head_dim + d]);
        }
        S *= scale;

        // P = exp(S − LSE)
        float P = __expf(S - lse);

        // dP = dO · v  (single scalar)
        float dP = 0.0f;
#pragma unroll 16
        for (int d = 0; d < BWD_MAX_HEADDIM; d++)
        {
            if (d < head_dim)
                dP += dO_local[d] * __half2float(Vh[kv * head_dim + d]);
        }

        // dS = P · (dP − D)
        float dS = P * (dP - D);
        float dS_scaled = dS * scale;

#pragma unroll 16
        for (int d = 0; d < BWD_MAX_HEADDIM; d++)
        {
            if (d < head_dim)
            {
                float kv_val = __half2float(Kh[kv * head_dim + d]);
                // dQ accumulates locally — one writer per Q-row, no atomic needed.
                dQ_local[d] += dS_scaled * kv_val;

                // dV[kv][d] += P · dO_i[d]   — race over all Q rows → atomic.
                atomicAdd(&dVh[kv * head_dim + d], P * dO_local[d]);

                // dK[kv][d] += dS · scale · Q_i[d] — race over all Q rows → atomic.
                atomicAdd(&dKh[kv * head_dim + d], dS_scaled * q_local[d]);
            }
        }
    }

    // Flush register dQ to global memory.
#pragma unroll 16
    for (int d = 0; d < BWD_MAX_HEADDIM; d++)
    {
        if (d < head_dim)
            dQh[q_row * head_dim + d] = dQ_local[d];
    }
}

// -----------------------------------------------------------------------------
// Cast FP32 → FP16 utility kernel
// -----------------------------------------------------------------------------
__global__ void cast_fp32_to_fp16(const float *src, __half *dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = __float2half(src[i]);
}

// -----------------------------------------------------------------------------
// Public launcher
//   Allocates intermediate FP32 buffers, runs backward, casts to FP16.
//   th = batch * num_heads  (matches forward's `th`).
//   dQ/dK/dV must be FP16 buffers of size [th * seq_len * head_dim].
// -----------------------------------------------------------------------------
extern "C" void launch_v54_backward(
    const __half *Q, const __half *K, const __half *V,
    const __half *O, const __half *dO, const float *LSE,
    __half *dQ, __half *dK, __half *dV,
    int th, int sl, int hd, int ca)
{
    float scale = 1.0f / sqrtf((float)hd);
    size_t n_elems = (size_t)th * sl * hd;

    float *dQ_fp32 = nullptr, *dK_fp32 = nullptr, *dV_fp32 = nullptr;
    cudaMalloc(&dQ_fp32, n_elems * sizeof(float));
    cudaMalloc(&dK_fp32, n_elems * sizeof(float));
    cudaMalloc(&dV_fp32, n_elems * sizeof(float));
    cudaMemset(dQ_fp32, 0, n_elems * sizeof(float));
    cudaMemset(dK_fp32, 0, n_elems * sizeof(float));
    cudaMemset(dV_fp32, 0, n_elems * sizeof(float));

    const int THREADS = 64;
    dim3 grid(th, (sl + THREADS - 1) / THREADS);
    dim3 block(THREADS);
    fa54_backward_kernel<<<grid, block>>>(
        Q, K, V, O, dO, LSE,
        dQ_fp32, dK_fp32, dV_fp32,
        sl, hd, ca, scale);

    // Cast to FP16
    int cast_threads = 256;
    int cast_blocks = (int)((n_elems + cast_threads - 1) / cast_threads);
    cast_fp32_to_fp16<<<cast_blocks, cast_threads>>>(dQ_fp32, dQ, (int)n_elems);
    cast_fp32_to_fp16<<<cast_blocks, cast_threads>>>(dK_fp32, dK, (int)n_elems);
    cast_fp32_to_fp16<<<cast_blocks, cast_threads>>>(dV_fp32, dV, (int)n_elems);

    cudaFree(dQ_fp32);
    cudaFree(dK_fp32);
    cudaFree(dV_fp32);
}

// =============================================================================
// CPU reference + correctness test + benchmark (standalone main)
// Compiled only when building as an executable (omit -DBUILD_AS_LIB).
// =============================================================================
#ifndef BUILD_AS_LIB

#define CK(c)                                                       \
    do                                                              \
    {                                                               \
        cudaError_t e = (c);                                        \
        if (e != cudaSuccess)                                       \
        {                                                           \
            fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(e));                         \
            exit(1);                                                \
        }                                                           \
    } while (0)

static inline float h2f(uint16_t h)
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

// CPU reference: computes O and (dQ, dK, dV) given Q, K, V, dO.
// Uses FP32 throughout. Returns LSE per row for cross-check too.
void cpu_attention_forward_backward(
    const float *Q, const float *K, const float *V, const float *dO,
    float *O_out, float *LSE_out, float *dQ_out, float *dK_out, float *dV_out,
    int th, int sl, int hd, int causal)
{
    float scale = 1.0f / sqrtf((float)hd);
    int hs = sl * hd;
    memset(O_out, 0, sizeof(float) * th * hs);
    memset(dQ_out, 0, sizeof(float) * th * hs);
    memset(dK_out, 0, sizeof(float) * th * hs);
    memset(dV_out, 0, sizeof(float) * th * hs);

    // Per (batch_head)
    for (int bh = 0; bh < th; bh++)
    {
        const float *Qh = Q + bh * hs;
        const float *Kh = K + bh * hs;
        const float *Vh = V + bh * hs;
        const float *dOh = dO + bh * hs;
        float *Oh = O_out + bh * hs;
        float *LSEh = LSE_out + bh * sl;
        float *dQh = dQ_out + bh * hs;
        float *dKh = dK_out + bh * hs;
        float *dVh = dV_out + bh * hs;

        // Forward: O and LSE
        // P[q][k] = exp(S[q][k] - rowmax) / rowsum
        float *P = (float *)malloc(sizeof(float) * sl * sl);
        for (int q = 0; q < sl; q++)
        {
            int kv_max = causal ? (q + 1) : sl;
            float rmax = -1e30f;
            for (int k = 0; k < kv_max; k++)
            {
                float s = 0.0f;
                for (int d = 0; d < hd; d++)
                    s += Qh[q * hd + d] * Kh[k * hd + d];
                s *= scale;
                P[q * sl + k] = s;
                if (s > rmax)
                    rmax = s;
            }
            for (int k = kv_max; k < sl; k++)
                P[q * sl + k] = -1e30f;
            float rsum = 0.0f;
            for (int k = 0; k < kv_max; k++)
            {
                P[q * sl + k] = expf(P[q * sl + k] - rmax);
                rsum += P[q * sl + k];
            }
            float inv = (rsum > 0.0f) ? 1.0f / rsum : 0.0f;
            for (int k = 0; k < kv_max; k++)
                P[q * sl + k] *= inv;
            LSEh[q] = logf(fmaxf(rsum, 1e-30f)) + rmax;
            // O[q][d] = Σ_k P[q][k] V[k][d]
            for (int d = 0; d < hd; d++)
            {
                float o = 0.0f;
                for (int k = 0; k < kv_max; k++)
                    o += P[q * sl + k] * Vh[k * hd + d];
                Oh[q * hd + d] = o;
            }
        }
        // Backward
        // D_q = Σ_d dO[q][d] * O[q][d]
        for (int q = 0; q < sl; q++)
        {
            int kv_max = causal ? (q + 1) : sl;
            float D = 0.0f;
            for (int d = 0; d < hd; d++)
                D += dOh[q * hd + d] * Oh[q * hd + d];
            for (int k = 0; k < kv_max; k++)
            {
                float Pqk = P[q * sl + k];
                // dV[k][d] += P[q][k] * dO[q][d]
                for (int d = 0; d < hd; d++)
                    dVh[k * hd + d] += Pqk * dOh[q * hd + d];
                // dP[q][k] = Σ_d dO[q][d] * V[k][d]
                float dP = 0.0f;
                for (int d = 0; d < hd; d++)
                    dP += dOh[q * hd + d] * Vh[k * hd + d];
                float dS = Pqk * (dP - D);
                float dS_scaled = dS * scale;
                // dQ[q] += dS_scaled * K[k]; dK[k] += dS_scaled * Q[q]
                for (int d = 0; d < hd; d++)
                {
                    dQh[q * hd + d] += dS_scaled * Kh[k * hd + d];
                    dKh[k * hd + d] += dS_scaled * Qh[q * hd + d];
                }
            }
        }
        free(P);
    }
}

void fill_random_fp16(uint16_t *d_ptr, int n)
{
    uint16_t *h = (uint16_t *)malloc(n * 2);
    for (int i = 0; i < n; i++)
        h[i] = f2h(((float)(rand() % 2001) - 1000.0f) / 1000.0f);
    CK(cudaMemcpy(d_ptr, h, n * 2, cudaMemcpyHostToDevice));
    free(h);
}

float maxabs_diff(const uint16_t *a_h, const float *b, int n)
{
    float m = 0.0f;
    for (int i = 0; i < n; i++)
    {
        float d = fabsf(h2f(a_h[i]) - b[i]);
        if (d > m)
            m = d;
    }
    return m;
}

void test_correctness()
{
    printf("--- Correctness (GPU backward vs CPU reference) ---\n");
    int configs[][4] = {
        // {th, seq_len, head_dim, causal}
        {1, 32, 64, 1},
        {1, 64, 128, 1},
        {2, 128, 128, 1},
        {1, 256, 128, 1},
    };
    for (auto &c : configs)
    {
        int th = c[0], sl = c[1], hd = c[2], ca = c[3];
        size_t n_elems = (size_t)th * sl * hd;
        size_t lse_elems = (size_t)th * sl;

        // CPU side: random FP32 data
        float *Qf = (float *)malloc(sizeof(float) * n_elems);
        float *Kf = (float *)malloc(sizeof(float) * n_elems);
        float *Vf = (float *)malloc(sizeof(float) * n_elems);
        float *dOf = (float *)malloc(sizeof(float) * n_elems);
        float *Of_ref = (float *)malloc(sizeof(float) * n_elems);
        float *LSEf_ref = (float *)malloc(sizeof(float) * lse_elems);
        float *dQf_ref = (float *)malloc(sizeof(float) * n_elems);
        float *dKf_ref = (float *)malloc(sizeof(float) * n_elems);
        float *dVf_ref = (float *)malloc(sizeof(float) * n_elems);

        for (size_t i = 0; i < n_elems; i++)
        {
            Qf[i] = ((float)(rand() % 2001) - 1000.0f) / 1000.0f;
            Kf[i] = ((float)(rand() % 2001) - 1000.0f) / 1000.0f;
            Vf[i] = ((float)(rand() % 2001) - 1000.0f) / 1000.0f;
            dOf[i] = ((float)(rand() % 2001) - 1000.0f) / 1000.0f;
        }

        cpu_attention_forward_backward(Qf, Kf, Vf, dOf,
                                       Of_ref, LSEf_ref, dQf_ref, dKf_ref, dVf_ref,
                                       th, sl, hd, ca);

        // GPU: convert inputs to FP16. Use CPU-computed O and LSE as inputs
        // to backward — this isolates the backward test from any forward bug.
        uint16_t *Qh_cpu = (uint16_t *)malloc(n_elems * 2);
        uint16_t *Kh_cpu = (uint16_t *)malloc(n_elems * 2);
        uint16_t *Vh_cpu = (uint16_t *)malloc(n_elems * 2);
        uint16_t *dOh_cpu = (uint16_t *)malloc(n_elems * 2);
        uint16_t *Oh_cpu = (uint16_t *)malloc(n_elems * 2);
        for (size_t i = 0; i < n_elems; i++)
        {
            Qh_cpu[i] = f2h(Qf[i]);
            Kh_cpu[i] = f2h(Kf[i]);
            Vh_cpu[i] = f2h(Vf[i]);
            dOh_cpu[i] = f2h(dOf[i]);
            Oh_cpu[i] = f2h(Of_ref[i]);
        }

        __half *dQ_d, *dK_d, *dV_d, *Q_d, *K_d, *V_d, *O_d, *dO_d;
        float *LSE_d;
        CK(cudaMalloc(&Q_d, n_elems * 2));
        CK(cudaMalloc(&K_d, n_elems * 2));
        CK(cudaMalloc(&V_d, n_elems * 2));
        CK(cudaMalloc(&O_d, n_elems * 2));
        CK(cudaMalloc(&dO_d, n_elems * 2));
        CK(cudaMalloc(&dQ_d, n_elems * 2));
        CK(cudaMalloc(&dK_d, n_elems * 2));
        CK(cudaMalloc(&dV_d, n_elems * 2));
        CK(cudaMalloc(&LSE_d, lse_elems * 4));

        CK(cudaMemcpy(Q_d, Qh_cpu, n_elems * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(K_d, Kh_cpu, n_elems * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(V_d, Vh_cpu, n_elems * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dO_d, dOh_cpu, n_elems * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(O_d, Oh_cpu, n_elems * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(LSE_d, LSEf_ref, lse_elems * 4, cudaMemcpyHostToDevice));

        // Backward (using CPU-computed O & LSE)
        launch_v54_backward(Q_d, K_d, V_d, O_d, dO_d, LSE_d, dQ_d, dK_d, dV_d, th, sl, hd, ca);
        CK(cudaDeviceSynchronize());

        // Read GPU results
        uint16_t *dQ_cpu = (uint16_t *)malloc(n_elems * 2);
        uint16_t *dK_cpu = (uint16_t *)malloc(n_elems * 2);
        uint16_t *dV_cpu = (uint16_t *)malloc(n_elems * 2);
        CK(cudaMemcpy(dQ_cpu, dQ_d, n_elems * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(dK_cpu, dK_d, n_elems * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(dV_cpu, dV_d, n_elems * 2, cudaMemcpyDeviceToHost));

        float mQ = maxabs_diff(dQ_cpu, dQf_ref, (int)n_elems);
        float mK = maxabs_diff(dK_cpu, dKf_ref, (int)n_elems);
        float mV = maxabs_diff(dV_cpu, dVf_ref, (int)n_elems);

        const float tol = 0.05f; // FP16 + atomicAdd ordering → tolerate up to 5e-2
        const char *st = (mQ < tol && mK < tol && mV < tol) ? "PASS" : "FAIL";

        printf("  th=%d sl=%d hd=%d ca=%d  dQ=%.4f  dK=%.4f  dV=%.4f  %s\n",
               th, sl, hd, ca, mQ, mK, mV, st);

        free(Qf);
        free(Kf);
        free(Vf);
        free(dOf);
        free(Of_ref);
        free(LSEf_ref);
        free(dQf_ref);
        free(dKf_ref);
        free(dVf_ref);
        free(Qh_cpu);
        free(Kh_cpu);
        free(Vh_cpu);
        free(dOh_cpu);
        free(Oh_cpu);
        free(dQ_cpu);
        free(dK_cpu);
        free(dV_cpu);
        CK(cudaFree(Q_d));
        CK(cudaFree(K_d));
        CK(cudaFree(V_d));
        CK(cudaFree(O_d));
        CK(cudaFree(dO_d));
        CK(cudaFree(dQ_d));
        CK(cudaFree(dK_d));
        CK(cudaFree(dV_d));
        CK(cudaFree(LSE_d));
    }
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

void bench()
{
    printf("\n--- Performance (forward + backward) ---\n");
    // (description, th, seq_len, head_dim)
    struct Cfg
    {
        const char *name;
        int th, sl, hd;
    };
    Cfg cfgs[] = {
        {"7B-256 ", 32, 256, 128},
        {"7B-512 ", 32, 512, 128},
        {"7B-1K  ", 32, 1024, 128},
        {"7B-2K  ", 32, 2048, 128},
    };

    Timer tm;
    for (auto &c : cfgs)
    {
        int th = c.th, sl = c.sl, hd = c.hd, ca = 1;
        size_t n_elems = (size_t)th * sl * hd;
        size_t lse_elems = (size_t)th * sl;

        __half *Q_d, *K_d, *V_d, *O_d, *dO_d, *dQ_d, *dK_d, *dV_d;
        float *LSE_d;
        CK(cudaMalloc(&Q_d, n_elems * 2));
        CK(cudaMalloc(&K_d, n_elems * 2));
        CK(cudaMalloc(&V_d, n_elems * 2));
        CK(cudaMalloc(&O_d, n_elems * 2));
        CK(cudaMalloc(&dO_d, n_elems * 2));
        CK(cudaMalloc(&dQ_d, n_elems * 2));
        CK(cudaMalloc(&dK_d, n_elems * 2));
        CK(cudaMalloc(&dV_d, n_elems * 2));
        CK(cudaMalloc(&LSE_d, lse_elems * 4));
        fill_random_fp16((uint16_t *)Q_d, (int)n_elems);
        fill_random_fp16((uint16_t *)K_d, (int)n_elems);
        fill_random_fp16((uint16_t *)V_d, (int)n_elems);
        fill_random_fp16((uint16_t *)dO_d, (int)n_elems);
        fill_random_fp16((uint16_t *)O_d, (int)n_elems);
        // LSE random fill (just for timing — values don't need to be valid)
        float *lse_h = (float *)malloc(lse_elems * 4);
        for (size_t i = 0; i < lse_elems; i++)
            lse_h[i] = (float)(rand() % 100) / 100.0f;
        CK(cudaMemcpy(LSE_d, lse_h, lse_elems * 4, cudaMemcpyHostToDevice));
        free(lse_h);

        // Warmup
        launch_v54_backward(Q_d, K_d, V_d, O_d, dO_d, LSE_d, dQ_d, dK_d, dV_d, th, sl, hd, ca);
        CK(cudaDeviceSynchronize());

        // Time backward
        tm.start();
        for (int i = 0; i < 5; i++)
            launch_v54_backward(Q_d, K_d, V_d, O_d, dO_d, LSE_d, dQ_d, dK_d, dV_d, th, sl, hd, ca);
        float bwd_ms = tm.stop() / 5.0f;

        // Approximate FLOPs for causal attention backward:
        //   forward:  2 · th · sl² · hd / 2 (causal)
        //   backward: ≈ 5 · forward FLOPs (dV + dP + dS + dQ + dK each ~ matmul cost)
        double fwd_flops = 2.0 * th * (double)sl * sl * hd / 2.0;
        double bwd_flops = fwd_flops * 5.0;
        double bwd_tflops = bwd_flops / bwd_ms / 1e9;

        printf("  %s  bwd=%.2f ms (%.1f T)\n",
               c.name, bwd_ms, bwd_tflops);

        CK(cudaFree(Q_d));
        CK(cudaFree(K_d));
        CK(cudaFree(V_d));
        CK(cudaFree(O_d));
        CK(cudaFree(dO_d));
        CK(cudaFree(dQ_d));
        CK(cudaFree(dK_d));
        CK(cudaFree(dV_d));
        CK(cudaFree(LSE_d));
    }
}

int main()
{
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    printf("=== FlashAttention v54 Backward — minimal viable ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n\n", p.name, p.multiProcessorCount, clock_khz / 1000);
    srand(42);
    test_correctness();
    bench();
    return 0;
}

#endif // BUILD_AS_LIB

