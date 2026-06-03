// =============================================================================
// FP8 GEMM v24 — kind::f8f6f4 unified path with FP32 accumulator
// =============================================================================
// C[M,N] = A[M,K] × B[N,K]^T   (B row-major, transposed via MMA)
//
// Data types: A,B = FP8 (E4M3), C = FP16, all row-major GPU memory
//
// Change vs v23:
//   v23: mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16
//   v24: mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e4m3.e4m3.f16
//
//   • KEEPS FP16 accumulator — empirically verified that the kind::f8f6f4
//     family on sm_120a accepts .f16 accum (not documented in PTX 9.0 spec
//     which only shows .f32, but ptxas accepts it). Same trick as the 4090
//     f16 accum discovery on the old mma path.
//   • Routes through the unified Blackwell 5th-gen kind::f8f6f4 path —
//     this opens the door to mxf8f6f4 block-scaled variants without
//     changing tile layout or register layout.
//   • Identical register/SMEM/threadblock geometry to v23 — drop-in.
//
// Two kernels:
//   mode=0: original   — v10b baseline structure, simpler code
//   mode=1: singlesync — paired K-step fragment loading, +2-3% on medium sizes
//
// Build shared library (Blackwell-only — kind::f8f6f4 requires sm_100a/sm_120a):
//   nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
//        --shared -Xcompiler -fPIC fp8_gemm_v24.cu -o libfp8gemm_v24.so
// =============================================================================

#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define BM 128
#define BN 128
#define BK 128
#define SMEM_STRIDE 128
#define BLOCK_THREADS 256
#define MMA_M 16
#define MMA_N 8
#define MMA_K 32
#define WARPS_N 4
#define WM 64
#define WN 32
#define M_TILES 4
#define N_TILES 4
#define K_STEPS 4
#define SMEM_PER_MAT (BM * SMEM_STRIDE)
#define SMEM_PER_BLOCK (2 * SMEM_PER_MAT)

// =============================================================================
// Swizzle + MMA helpers
// =============================================================================

__device__ __forceinline__ int swizzle16(int row, int col16)
{
    int chunk = col16 >> 4;
    return row * SMEM_STRIDE + ((chunk ^ (row & 7)) << 4);
}

__device__ __forceinline__ int swizzle4(int row, int col)
{
    int chunk = col >> 4;
    int within = col & 15;
    return row * SMEM_STRIDE + ((chunk ^ (row & 7)) << 4) + within;
}

// v24: kind::f8f6f4 with FP16 accumulator. Same register layout as v23 (2x
// uint32_t = 4 packed FP16 per m16n8 tile) but going through the unified
// Blackwell 5th-gen kind::f8f6f4 instruction family. This gives us the
// kind:: modifier (essential for future mxf8f6f4 block-scaled variant)
// while keeping v23's low register pressure.
__device__ __forceinline__ void mma_fp8(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t &c0, uint32_t &c1)
{
    uint32_t d0, d1;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e4m3.e4m3.f16 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(c0), "r"(c1));
    c0 = d0;
    c1 = d1;
}

// =============================================================================
// GMEM → SMEM tile load (shared by both kernels)
// =============================================================================

__device__ __forceinline__ void load_tile(
    uint8_t *smem, const uint8_t *gmem,
    int g_base, int stride_g, int g_limit, int bk_offset)
{
    const int thr_per_row = BK / 16;
    const int rows_per_pass = BLOCK_THREADS / thr_per_row;
    const int load_row = threadIdx.x / thr_per_row;
    const int load_col = (threadIdx.x % thr_per_row) * 16;

#pragma unroll
    for (int pass = 0; pass < 4; pass++)
    {
        int row = pass * rows_per_pass + load_row;
        int gm = g_base + row;
        int gk = bk_offset + load_col;
        uint4 val = make_uint4(0u, 0u, 0u, 0u);
        if (gm < g_limit && gk + 16 <= stride_g)
            val = __ldg((const uint4 *)&gmem[gm * stride_g + gk]);
        *(uint4 *)&smem[swizzle16(row, load_col)] = val;
    }
}

// =============================================================================
// Store C (shared by both kernels)
// =============================================================================

// v24: acc is uint32_t[M_TILES][N_TILES][2] (packed FP16, same as v23).
__device__ __forceinline__ void store_c(
    uint32_t acc[M_TILES][N_TILES][2],
    uint16_t *C, int bm, int wm, int bn, int wn,
    int group_id, int tid, int M, int N)
{
#pragma unroll
    for (int mi = 0; mi < M_TILES; mi++)
#pragma unroll
        for (int ni = 0; ni < N_TILES; ni++)
        {
            int row0 = bm + wm + mi * MMA_M + group_id;
            int row1 = row0 + 8;
            int col = bn + wn + ni * MMA_N + tid * 2;
            if (row0 < M && col + 1 < N)
                *(uint32_t *)&C[row0 * N + col] = acc[mi][ni][0];
            if (row1 < M && col + 1 < N)
                *(uint32_t *)&C[row1 * N + col] = acc[mi][ni][1];
        }
}

// =============================================================================
// Kernel 0: Original (v10b)
// =============================================================================

extern __shared__ uint8_t dyn_smem_orig[];

__global__ void __launch_bounds__(BLOCK_THREADS, 3)
    kernel_original(
        const uint8_t *__restrict__ A,
        const uint8_t *__restrict__ B,
        uint16_t *__restrict__ C,
        int M, int N, int K)
{
    uint8_t *smem_A = dyn_smem_orig;
    uint8_t *smem_B = dyn_smem_orig + SMEM_PER_MAT;

    const int bm = blockIdx.x * BM, bn = blockIdx.y * BN;
    const int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32;
    const int wm = (warp_id / WARPS_N) * WM;
    const int wn = (warp_id % WARPS_N) * WN;
    const int group_id = lane_id / 4, tid = lane_id % 4;

    uint32_t acc[M_TILES][N_TILES][2];
#pragma unroll
    for (int mi = 0; mi < M_TILES; mi++)
#pragma unroll
        for (int ni = 0; ni < N_TILES; ni++)
        {
            acc[mi][ni][0] = 0u;
            acc[mi][ni][1] = 0u;
        }

    for (int bk = 0; bk < K; bk += BK)
    {
        load_tile(smem_A, A, bm, K, M, bk);
        load_tile(smem_B, B, bn, K, N, bk);
        __syncthreads();

#pragma unroll
        for (int ki = 0; ki < K_STEPS; ki++)
        {
            int k_off = ki * MMA_K;

            uint32_t a_frag[M_TILES][4];
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                int ar = wm + mi * MMA_M;
                int cl = k_off + tid * 4, ch = cl + 16;
                a_frag[mi][0] = *(uint32_t *)&smem_A[swizzle4(ar + group_id, cl)];
                a_frag[mi][1] = *(uint32_t *)&smem_A[swizzle4(ar + group_id + 8, cl)];
                a_frag[mi][2] = *(uint32_t *)&smem_A[swizzle4(ar + group_id, ch)];
                a_frag[mi][3] = *(uint32_t *)&smem_A[swizzle4(ar + group_id + 8, ch)];
            }

            uint32_t b_frag[N_TILES][2];
#pragma unroll
            for (int ni = 0; ni < N_TILES; ni++)
            {
                int br = wn + ni * MMA_N;
                int cl = k_off + tid * 4, ch = cl + 16;
                b_frag[ni][0] = *(uint32_t *)&smem_B[swizzle4(br + group_id, cl)];
                b_frag[ni][1] = *(uint32_t *)&smem_B[swizzle4(br + group_id, ch)];
            }

#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
#pragma unroll
                for (int ni = 0; ni < N_TILES; ni++)
                    mma_fp8(a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                            b_frag[ni][0], b_frag[ni][1], acc[mi][ni][0], acc[mi][ni][1]);
        }
        __syncthreads();
    }
    store_c(acc, C, bm, wm, bn, wn, group_id, tid, M, N);
}

// =============================================================================
// Kernel 1: Singlesync (paired K-step fragment loading)
// =============================================================================

extern __shared__ uint8_t dyn_smem_ss[];

__global__ void __launch_bounds__(BLOCK_THREADS, 3)
    kernel_singlesync(
        const uint8_t *__restrict__ A,
        const uint8_t *__restrict__ B,
        uint16_t *__restrict__ C,
        int M, int N, int K)
{
    uint8_t *smem_A = dyn_smem_ss;
    uint8_t *smem_B = dyn_smem_ss + SMEM_PER_MAT;

    const int bm = blockIdx.x * BM, bn = blockIdx.y * BN;
    const int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32;
    const int wm = (warp_id / WARPS_N) * WM;
    const int wn = (warp_id % WARPS_N) * WN;
    const int group_id = lane_id / 4, tid = lane_id % 4;

    uint32_t acc[M_TILES][N_TILES][2];
#pragma unroll
    for (int mi = 0; mi < M_TILES; mi++)
#pragma unroll
        for (int ni = 0; ni < N_TILES; ni++)
        {
            acc[mi][ni][0] = 0u;
            acc[mi][ni][1] = 0u;
        }

    for (int bk = 0; bk < K; bk += BK)
    {
        load_tile(smem_A, A, bm, K, M, bk);
        load_tile(smem_B, B, bn, K, N, bk);
        __syncthreads();

#pragma unroll
        for (int kp = 0; kp < K_STEPS; kp += 2)
        {
            int k0 = kp * MMA_K, k1 = (kp + 1) * MMA_K;

            uint32_t a0[M_TILES][4], a1[M_TILES][4];
            uint32_t b0[N_TILES][2], b1[N_TILES][2];

#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                int ar = wm + mi * MMA_M;
                int cl0 = k0 + tid * 4, ch0 = cl0 + 16;
                int cl1 = k1 + tid * 4, ch1 = cl1 + 16;
                a0[mi][0] = *(uint32_t *)&smem_A[swizzle4(ar + group_id, cl0)];
                a0[mi][1] = *(uint32_t *)&smem_A[swizzle4(ar + group_id + 8, cl0)];
                a0[mi][2] = *(uint32_t *)&smem_A[swizzle4(ar + group_id, ch0)];
                a0[mi][3] = *(uint32_t *)&smem_A[swizzle4(ar + group_id + 8, ch0)];
                a1[mi][0] = *(uint32_t *)&smem_A[swizzle4(ar + group_id, cl1)];
                a1[mi][1] = *(uint32_t *)&smem_A[swizzle4(ar + group_id + 8, cl1)];
                a1[mi][2] = *(uint32_t *)&smem_A[swizzle4(ar + group_id, ch1)];
                a1[mi][3] = *(uint32_t *)&smem_A[swizzle4(ar + group_id + 8, ch1)];
            }

#pragma unroll
            for (int ni = 0; ni < N_TILES; ni++)
            {
                int br = wn + ni * MMA_N;
                int cl0 = k0 + tid * 4, ch0 = cl0 + 16;
                int cl1 = k1 + tid * 4, ch1 = cl1 + 16;
                b0[ni][0] = *(uint32_t *)&smem_B[swizzle4(br + group_id, cl0)];
                b0[ni][1] = *(uint32_t *)&smem_B[swizzle4(br + group_id, ch0)];
                b1[ni][0] = *(uint32_t *)&smem_B[swizzle4(br + group_id, cl1)];
                b1[ni][1] = *(uint32_t *)&smem_B[swizzle4(br + group_id, ch1)];
            }

#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
#pragma unroll
                for (int ni = 0; ni < N_TILES; ni++)
                    mma_fp8(a0[mi][0], a0[mi][1], a0[mi][2], a0[mi][3],
                            b0[ni][0], b0[ni][1], acc[mi][ni][0], acc[mi][ni][1]);

#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
#pragma unroll
                for (int ni = 0; ni < N_TILES; ni++)
                    mma_fp8(a1[mi][0], a1[mi][1], a1[mi][2], a1[mi][3],
                            b1[ni][0], b1[ni][1], acc[mi][ni][0], acc[mi][ni][1]);
        }
        __syncthreads();
    }
    store_c(acc, C, bm, wm, bn, wn, group_id, tid, M, N);
}

// =============================================================================
// C API — purego compatible (no CGo)
// =============================================================================

static bool g_smem_configured[2] = {false, false};

typedef void (*kernel_fn_t)(const uint8_t *, const uint8_t *, uint16_t *, int, int, int);
static kernel_fn_t g_kernels[2] = {kernel_original, kernel_singlesync};

extern "C"
{

    // fp8_gemm — unified entry point
    // mode: 0 = original, 1 = singlesync
    // stream: CUDA stream (0/NULL for default)
    // Returns: 0 on success, CUDA error code on failure
    int fp8_gemm(
        int M, int N, int K,
        const void *A, const void *B, void *C,
        int mode, void *stream)
    {
        if (mode < 0 || mode > 1)
            mode = 1;

        if (!g_smem_configured[mode])
        {
            cudaError_t err = cudaFuncSetAttribute(
                g_kernels[mode],
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                SMEM_PER_BLOCK);
            if (err != cudaSuccess)
                return (int)err;
            g_smem_configured[mode] = true;
        }

        dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

        g_kernels[mode]<<<grid, BLOCK_THREADS, SMEM_PER_BLOCK, (cudaStream_t)stream>>>(
            (const uint8_t *)A, (const uint8_t *)B, (uint16_t *)C, M, N, K);

        return (int)cudaGetLastError();
    }

    // Direct symbol entry points for purego.RegisterLibFunc
    int fp8_gemm_original(int M, int N, int K,
                          const void *A, const void *B, void *C, void *stream)
    {
        return fp8_gemm(M, N, K, A, B, C, 0, stream);
    }

    int fp8_gemm_singlesync(int M, int N, int K,
                            const void *A, const void *B, void *C, void *stream)
    {
        return fp8_gemm(M, N, K, A, B, C, 1, stream);
    }

} // extern "C"

// =============================================================================
// Standalone test + benchmark (built only when BUILD_AS_LIB not defined)
// =============================================================================
#ifndef BUILD_AS_LIB

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); }} while(0)

static inline uint8_t float_to_e4m3(float f) {
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
static inline float e4m3_to_float(uint8_t v) {
    int s = (v >> 7) & 1, e = (v >> 3) & 0xF, m = v & 7;
    if (e == 0xF && m == 7) return nanf("");
    float r = (e == 0) ? ldexpf((float)m, -9) : ldexpf(1.0f + m / 8.0f, e - 7);
    return s ? -r : r;
}
static inline float fp16f(uint16_t h) {
    __half hv; memcpy(&hv, &h, 2); return __half2float(hv);
}

bool test_correctness(int mode, const char *name) {
    int M = 512, N = 512, K = 512;
    size_t sA = (size_t)M * K, sB = (size_t)N * K, sC = (size_t)M * N * 2;
    uint8_t *hA = (uint8_t *)malloc(sA), *hB = (uint8_t *)malloc(sB);
    uint16_t *hC = (uint16_t *)malloc(sC);
    float *ref = (float *)malloc((size_t)M * N * 4);
    srand(42);
    for (size_t i = 0; i < sA; i++) hA[i] = float_to_e4m3(((float)(rand() % 16) - 8.0f) * 0.25f);
    for (size_t i = 0; i < sB; i++) hB[i] = float_to_e4m3(((float)(rand() % 16) - 8.0f) * 0.25f);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float s = 0;
            for (int k = 0; k < K; k++)
                s += e4m3_to_float(hA[m * K + k]) * e4m3_to_float(hB[n * K + k]);
            ref[m * N + n] = s;
        }
    void *dA, *dB; uint16_t *dC;
    CK(cudaMalloc(&dA, sA));
    CK(cudaMalloc(&dB, sB));
    CK(cudaMalloc(&dC, sC));
    CK(cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice));
    CK(cudaMemset(dC, 0, sC));

    fp8_gemm(M, N, K, dA, dB, dC, mode, 0);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hC, dC, sC, cudaMemcpyDeviceToHost));

    int err = 0; float mx = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float ae = fabsf(fp16f(hC[m * N + n]) - ref[m * N + n]);
            if (ae > mx) mx = ae;
            if (ae > fmaxf(1.0f, fabsf(ref[m * N + n]) * 0.05f)) err++;
        }
    printf("  %-12s 512³: max_err=%.4f errors=%d → %s\n", name, mx, err, err == 0 ? "PASS" : "FAIL");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC); free(ref);
    return err == 0;
}

double bench(int mode, int M, int N, int K) {
    void *dA, *dB, *dC;
    CK(cudaMalloc(&dA, (size_t)M * K));
    CK(cudaMalloc(&dB, (size_t)N * K));
    CK(cudaMalloc(&dC, (size_t)M * N * 2));
    CK(cudaMemset(dA, 0x38, (size_t)M * K));
    CK(cudaMemset(dB, 0x38, (size_t)N * K));

    for (int i = 0; i < 10; i++) fp8_gemm(M, N, K, dA, dB, dC, mode, 0);
    CK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    int it = 100;
    cudaEventRecord(t0);
    for (int i = 0; i < it; i++) fp8_gemm(M, N, K, dA, dB, dC, mode, 0);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    double tf = 2.0 * (double)M * (double)N * (double)K / (ms / it / 1000.0) / 1e12;
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return tf;
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    printf("=== FP8 GEMM v24 — kind::f8f6f4 + FP16 accumulator (drop-in over v23) ===\n");
    printf("GPU: %s (%d SMs, %d MHz)\n\n", p.name, p.multiProcessorCount, clock_khz / 1000);

    printf("--- Correctness ---\n");
    test_correctness(0, "original");
    test_correctness(1, "singlesync");

    printf("\n--- Performance (TFLOPS) ---\n");
    struct { int M, N, K; const char *lbl; } sizes[] = {
        {1024, 1024, 1024, "1K³"},
        {2048, 2048, 2048, "2K³"},
        {4096, 4096, 4096, "4K³"},
        {8192, 8192, 8192, "8K³"},
        {4096, 11008, 4096, "4K×11K×4K"},
        {8192, 11008, 4096, "8K×11K×4K"},
    };
    printf("%-14s %10s %12s\n", "Size", "original", "singlesync");
    for (auto &s : sizes) {
        double r0 = bench(0, s.M, s.N, s.K);
        double r1 = bench(1, s.M, s.N, s.K);
        printf("%-14s %10.1f %12.1f\n", s.lbl, r0, r1);
    }
    return 0;
}

#endif // !BUILD_AS_LIB
