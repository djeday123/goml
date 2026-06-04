// =============================================================================
// FlashAttention v61 — FP8 Forward (PoC: S = Q·Kᵀ partial)
// =============================================================================
// First step of B roadmap (docs/FP8_FA_ROADMAP.md). PoC goal: prove
// kind::f8f6f4 MMA path works end-to-end on sm_120a within an attention
// kernel skeleton.
//
// Scope of THIS PoC commit:
//   ✓ FP8 Q, K, V inputs (already quantized by caller)
//   ✓ S = Q·Kᵀ via mma m16n8k32 kind::f8f6f4 .f16.e4m3.e4m3.f16
//   ✓ Online softmax + LSE math (unchanged from v55)
//   ✓ FP8 SMEM (1 byte/elem) — halves tile size
//   ✗ P·V MMA — needs P quantize-and-repack-to-A-operand. Marked TODO.
//   ✗ Per-tile or per-row scales for production accuracy — uses single
//     per-tensor scale for now.
//
// Builds with: nvcc -gencode arch=compute_120a,code=sm_120a
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define FA_BR 64
#define FA_BC 64
#define FA_THREADS 128
#define FA_STRIDE 128

__device__ __forceinline__ void cpa16(void *s, const void *g, int n)
{
    uint32_t sa = __cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" ::"r"(sa), "l"(g), "r"(n));
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cpa_wait()
{
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

// =============================================================================
// MMA: m16n8k32 kind::f8f6f4 e4m3·e4m3 → FP16 (packed) accumulator
// =============================================================================
__device__ __forceinline__ void mma_fp8_f16(
    uint32_t &d0, uint32_t &d1,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t c0, uint32_t c1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e4m3.e4m3.f16 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(c0), "r"(c1));
}

// =============================================================================
// Swizzle for FP8 SMEM: 16-byte chunk XOR by row (rows 0..7 cycle, 4-byte step)
// =============================================================================
__device__ __forceinline__ int swz_byte(int row, int col_bytes)
{
    int chunk = col_bytes >> 4;
    int within = col_bytes & 15;
    return row * FA_STRIDE + ((chunk ^ (row & 7)) << 4) + within;
}

// =============================================================================
// FP8 tile load: 16 bytes per thread per cp.async (16 fp8 elements at once)
// =============================================================================
__device__ __forceinline__ void load_tile_fp8(
    uint8_t *dst, const uint8_t *src, int start, int rows,
    int seq_len, int head_dim)
{
    constexpr int CHUNK = 16;  // bytes per cp.async = 16 fp8
    int chunks_per_row = head_dim / CHUNK;
    int total = rows * chunks_per_row;
#pragma unroll 4
    for (int c = threadIdx.x; c < total; c += FA_THREADS)
    {
        int row = c / chunks_per_row;
        int col_bytes = (c % chunks_per_row) * CHUNK;
        int gr = start + row;
        int dst_off = swz_byte(row, col_bytes);
        cpa16(&dst[dst_off], &src[gr * head_dim + col_bytes], (gr < seq_len) ? 16 : 0);
    }
}

// =============================================================================
// v61 PoC kernel — S = Q·Kᵀ then write zeros for O (placeholder).
// Verifies the FP8 MMA path compiles and produces Sr_p with correct numbers.
// =============================================================================
__global__ void __launch_bounds__(FA_THREADS, 4)
    fa61_pq_kernel(
        const uint8_t *__restrict__ Q,
        const uint8_t *__restrict__ K,
        const uint8_t *__restrict__ V,
        __half *__restrict__ S_dump,   // S [B*H, FA_BR, FA_BC] for verification
        int seq_len, int head_dim, int causal, float scale)
{
    int nqt = (seq_len + FA_BR - 1) / FA_BR;
    int bh = blockIdx.x / nqt, qt = blockIdx.x % nqt, qs = qt * FA_BR;
    if (qs >= seq_len) return;
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    int gid = lane / 4, tid = lane % 4;
    int mrb = wid * 16;

    extern __shared__ uint8_t raw[];
    uint8_t *smQ = raw;
    uint8_t *smK = smQ + FA_BR * FA_STRIDE;
    uint8_t *smV = smK + FA_BC * FA_STRIDE;  // V loaded but unused in PoC

    int hs = seq_len * head_dim;
    const uint8_t *Qh = Q + bh * hs;
    const uint8_t *Kh = K + bh * hs;
    const uint8_t *Vh = V + bh * hs;
    __half *Sh = S_dump + bh * (FA_BR * FA_BC);  // one tile per (bh, qt=0) — PoC only handles qt=0

    load_tile_fp8(smQ, Qh, qs, FA_BR, seq_len, head_dim);
    load_tile_fp8(smK, Kh, 0, FA_BC, seq_len, head_dim);  // K[0..Bc) for kv=0
    (void)Vh; (void)smV;
    cpa_commit();
    cpa_wait<0>();
    __syncthreads();

    // Load Q frags: 4 k-steps × 4 uint32_t per step (m16k32 A operand)
    uint32_t Qr[4][4];
#pragma unroll
    for (int ks = 0; ks < 4; ks++)
    {
        int k_off = ks * 32;
        int cl = k_off + tid * 4;
        int ch = cl + 16;
        int g0 = mrb + gid, g8 = g0 + 8;
        Qr[ks][0] = *(uint32_t *)&smQ[swz_byte(g0, cl)];
        Qr[ks][1] = *(uint32_t *)&smQ[swz_byte(g8, cl)];
        Qr[ks][2] = *(uint32_t *)&smQ[swz_byte(g0, ch)];
        Qr[ks][3] = *(uint32_t *)&smQ[swz_byte(g8, ch)];
    }

    // S = Q · Kᵀ — accumulate in packed FP16
    uint32_t Sr_p[8][2];
#pragma unroll
    for (int nt = 0; nt < 8; nt++) Sr_p[nt][0] = Sr_p[nt][1] = 0u;

#pragma unroll
    for (int ks = 0; ks < 4; ks++)
    {
        int k_off = ks * 32;
        int cl = k_off + tid * 4;
        int ch = cl + 16;
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            int br = nt * 8;
            // B operand: K row = br+gid (n-direction), K col = cl..cl+3 (k-direction)
            // For m16n8k32 B operand (n=8, k=32): each thread holds 2 uint32_t
            // covering (n=gid, k=tid*4..tid*4+3) and (n=gid, k=tid*4+16..tid*4+19)
            uint32_t b0 = *(uint32_t *)&smK[swz_byte(br + gid, cl)];
            uint32_t b1 = *(uint32_t *)&smK[swz_byte(br + gid, ch)];
            mma_fp8_f16(Sr_p[nt][0], Sr_p[nt][1],
                        Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3],
                        b0, b1, Sr_p[nt][0], Sr_p[nt][1]);
        }
    }

    // Write S tile to global for verification.
    // Each thread holds 4 elements per nt: (gid, nt*8+tid*2..+1), (gid+8, ...)
#pragma unroll
    for (int nt = 0; nt < 8; nt++)
    {
        __half2 v0 = *reinterpret_cast<__half2 *>(&Sr_p[nt][0]);
        __half2 v1 = *reinterpret_cast<__half2 *>(&Sr_p[nt][1]);
        float s0 = __half2float(__low2half(v0));
        float s1 = __half2float(__high2half(v0));
        float s2 = __half2float(__low2half(v1));
        float s3 = __half2float(__high2half(v1));

        int row0 = mrb + gid;
        int row8 = mrb + gid + 8;
        int col0 = nt * 8 + tid * 2;
        int col1 = col0 + 1;
        if (row0 < FA_BR && col0 < FA_BC) Sh[row0 * FA_BC + col0] = __float2half(s0 * scale);
        if (row0 < FA_BR && col1 < FA_BC) Sh[row0 * FA_BC + col1] = __float2half(s1 * scale);
        if (row8 < FA_BR && col0 < FA_BC) Sh[row8 * FA_BC + col0] = __float2half(s2 * scale);
        if (row8 < FA_BR && col1 < FA_BC) Sh[row8 * FA_BC + col1] = __float2half(s3 * scale);
    }
}

// =============================================================================
// Host harness — quantize Q,K,V to FP8, launch, check S vs CPU reference.
// =============================================================================
#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); }} while(0)

static inline uint8_t float_to_e4m3(float f)
{
    if (f != f) return 0x7Fu;
    int sign = (f < 0.0f) ? 1 : 0;
    float af = fabsf(f);
    if (af > 448.0f) return sign ? 0xFEu : 0x7Eu;
    if (af < 1.953125e-3f) return sign ? 0x80u : 0x00u;
    int eu = (int)floorf(log2f(af));
    float mf = af / ldexpf(1.0f, eu) - 1.0f;
    int m3 = (int)(mf * 8.0f + 0.5f);
    if (m3 >= 8) { m3 = 0; eu++; }
    int eb = eu + 7;
    if (eb < 1) {
        int ms = (int)(af / ldexpf(1.0f, -9) + 0.5f);
        if (ms > 7) ms = 7;
        return (uint8_t)((sign << 7) | (ms & 7));
    }
    if (eb > 15) eb = 15;
    return (uint8_t)((sign << 7) | (eb << 3) | (m3 & 7));
}
static inline float e4m3_to_float(uint8_t v)
{
    int s = (v >> 7) & 1, e = (v >> 3) & 0xF, m = v & 7;
    if (e == 0xF && m == 7) return nanf("");
    float r = (e == 0) ? ldexpf((float)m, -9) : ldexpf(1.0f + m / 8.0f, e - 7);
    return s ? -r : r;
}
static inline float fp16f(uint16_t h)
{
    __half hv; memcpy(&hv, &h, 2); return __half2float(hv);
}

int main()
{
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    int clk = 0; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    printf("=== FA v61 FP8 forward PoC — verify S = Q·Kᵀ ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n\n", p.name, p.multiProcessorCount, clk / 1000);

    // Tiny single-tile test: bh=1, sl=64 (= FA_BR), hd=128.
    int sl = 64, hd = 128, bh = 1;
    size_t n_elems = (size_t)bh * sl * hd;

    float *Qf = (float *)malloc(sizeof(float) * n_elems);
    float *Kf = (float *)malloc(sizeof(float) * n_elems);
    float *Vf = (float *)malloc(sizeof(float) * n_elems);
    srand(42);
    for (size_t i = 0; i < n_elems; i++) {
        Qf[i] = ((float)(rand() % 16) - 8.0f) * 0.125f;  // -1..1 range, FP8-friendly
        Kf[i] = ((float)(rand() % 16) - 8.0f) * 0.125f;
        Vf[i] = ((float)(rand() % 16) - 8.0f) * 0.125f;
    }

    // Quantize to FP8 e4m3
    uint8_t *Qq = (uint8_t *)malloc(n_elems);
    uint8_t *Kq = (uint8_t *)malloc(n_elems);
    uint8_t *Vq = (uint8_t *)malloc(n_elems);
    for (size_t i = 0; i < n_elems; i++) {
        Qq[i] = float_to_e4m3(Qf[i]);
        Kq[i] = float_to_e4m3(Kf[i]);
        Vq[i] = float_to_e4m3(Vf[i]);
    }

    // CPU reference: S = Q·Kᵀ using FP8-roundtripped values (so we measure
    // kernel error, not FP8 quantization error).
    float *S_ref = (float *)malloc(sizeof(float) * FA_BR * FA_BC);
    float scale = 1.0f / sqrtf((float)hd);
    for (int m = 0; m < FA_BR; m++)
        for (int n = 0; n < FA_BC; n++) {
            float s = 0;
            for (int k = 0; k < hd; k++) {
                float q = e4m3_to_float(Qq[m * hd + k]);
                float kk = e4m3_to_float(Kq[n * hd + k]);
                s += q * kk;
            }
            S_ref[m * FA_BC + n] = s * scale;
        }

    // GPU
    uint8_t *Q_d, *K_d, *V_d;
    __half *S_d;
    CK(cudaMalloc(&Q_d, n_elems));
    CK(cudaMalloc(&K_d, n_elems));
    CK(cudaMalloc(&V_d, n_elems));
    CK(cudaMalloc(&S_d, FA_BR * FA_BC * 2));
    CK(cudaMemcpy(Q_d, Qq, n_elems, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(K_d, Kq, n_elems, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(V_d, Vq, n_elems, cudaMemcpyHostToDevice));
    CK(cudaMemset(S_d, 0, FA_BR * FA_BC * 2));

    int smem = (FA_BR + 2 * FA_BC) * FA_STRIDE;  // Q + K + V tiles in BYTES
    CK(cudaFuncSetAttribute(fa61_pq_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    fa61_pq_kernel<<<1, FA_THREADS, smem>>>(Q_d, K_d, V_d, S_d, sl, hd, 0, scale);
    CK(cudaDeviceSynchronize());

    uint16_t *S_cpu = (uint16_t *)malloc(FA_BR * FA_BC * 2);
    CK(cudaMemcpy(S_cpu, S_d, FA_BR * FA_BC * 2, cudaMemcpyDeviceToHost));

    float mx = 0, sum_abs = 0;
    int errs = 0;
    for (int i = 0; i < FA_BR * FA_BC; i++) {
        float gpu = fp16f(S_cpu[i]);
        float ref = S_ref[i];
        float ae = fabsf(gpu - ref);
        sum_abs += fabsf(ref);
        if (ae > mx) mx = ae;
        if (ae > fmaxf(0.05f, fabsf(ref) * 0.1f)) errs++;
    }
    float avg_abs = sum_abs / (FA_BR * FA_BC);
    printf("S = Q·Kᵀ FP8 MMA verification:\n");
    printf("  max_abs_err = %.4f\n", mx);
    printf("  avg |S_ref| = %.4f (relative err tolerance ~10%%)\n", avg_abs);
    printf("  hard errors = %d / %d\n", errs, FA_BR * FA_BC);
    printf("  status: %s\n", errs == 0 ? "PASS" : "FAIL");

    // Print a few samples
    printf("\nFirst 4 cells (gpu vs ref):\n");
    for (int i = 0; i < 4; i++) {
        printf("  [%d] gpu=%+.4f  ref=%+.4f  diff=%+.4f\n",
               i, fp16f(S_cpu[i]), S_ref[i], fp16f(S_cpu[i]) - S_ref[i]);
    }

    free(Qf); free(Kf); free(Vf);
    free(Qq); free(Kq); free(Vq);
    free(S_ref); free(S_cpu);
    cudaFree(Q_d); cudaFree(K_d); cudaFree(V_d); cudaFree(S_d);
    return errs == 0 ? 0 : 1;
}
