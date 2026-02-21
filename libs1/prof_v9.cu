// v9 profiling harness — 4096³ only
// BK=128, stride=144, no swizzle, no launch_bounds hint
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define BM 128
#define BN 128
#define BK 128
#define SMEM_STRIDE 144
#define BLOCK_THREADS 256
#define WARPS_M 2
#define WARPS_N 4
#define WM 64
#define WN 32
#define MMA_M 16
#define MMA_N 8
#define MMA_K 32
#define M_TILES 4
#define N_TILES 4
#define K_STEPS 4

__device__ __forceinline__ void mma_fp8(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, uint32_t &c0, uint32_t &c1)
{
    uint32_t d0, d1;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1));
    c0 = d0;
    c1 = d1;
}

__global__ void __launch_bounds__(BLOCK_THREADS)
    fp8_gemm_f16acc_kernel(const uint8_t *__restrict__ A, const uint8_t *__restrict__ B,
                           uint16_t *__restrict__ C, int M, int N, int K)
{
    __shared__ uint8_t smem_A[BM * SMEM_STRIDE];
    __shared__ uint8_t smem_B[BN * SMEM_STRIDE];

    const int bm = blockIdx.x * BM, bn = blockIdx.y * BN;
    const int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32;
    const int warp_m = warp_id / WARPS_N, warp_n = warp_id % WARPS_N;
    const int wm = warp_m * WM, wn = warp_n * WN;
    const int group_id = lane_id / 4, tid = lane_id % 4;
    const int thr_per_row = BK / 16;
    const int rows_per_pass = BLOCK_THREADS / thr_per_row;
    const int load_row_in_pass = threadIdx.x / thr_per_row;
    const int load_col = (threadIdx.x % thr_per_row) * 16;

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
#pragma unroll
        for (int pass = 0; pass < 4; pass++)
        {
            int row = pass * rows_per_pass + load_row_in_pass;
            int gm = bm + row, gk = bk + load_col;
            uint4 val = make_uint4(0u, 0u, 0u, 0u);
            if (gm < M && gk + 16 <= K)
                val = __ldg((const uint4 *)&A[gm * K + gk]);
            *(uint4 *)&smem_A[row * SMEM_STRIDE + load_col] = val;
        }
#pragma unroll
        for (int pass = 0; pass < 4; pass++)
        {
            int row = pass * rows_per_pass + load_row_in_pass;
            int gn = bn + row, gk = bk + load_col;
            uint4 val = make_uint4(0u, 0u, 0u, 0u);
            if (gn < N && gk + 16 <= K)
                val = __ldg((const uint4 *)&B[gn * K + gk]);
            *(uint4 *)&smem_B[row * SMEM_STRIDE + load_col] = val;
        }
        __syncthreads();

#pragma unroll
        for (int ki = 0; ki < K_STEPS; ki++)
        {
            int k_off = ki * MMA_K;
            uint32_t a_frag[M_TILES][4];
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                int a_row = wm + mi * MMA_M;
                a_frag[mi][0] = *(uint32_t *)&smem_A[(a_row + group_id) * SMEM_STRIDE + k_off + tid * 4];
                a_frag[mi][1] = *(uint32_t *)&smem_A[(a_row + group_id + 8) * SMEM_STRIDE + k_off + tid * 4];
                a_frag[mi][2] = *(uint32_t *)&smem_A[(a_row + group_id) * SMEM_STRIDE + k_off + tid * 4 + 16];
                a_frag[mi][3] = *(uint32_t *)&smem_A[(a_row + group_id + 8) * SMEM_STRIDE + k_off + tid * 4 + 16];
            }
            uint32_t b_frag[N_TILES][2];
#pragma unroll
            for (int ni = 0; ni < N_TILES; ni++)
            {
                int b_row = wn + ni * MMA_N;
                b_frag[ni][0] = *(uint32_t *)&smem_B[(b_row + group_id) * SMEM_STRIDE + k_off + tid * 4];
                b_frag[ni][1] = *(uint32_t *)&smem_B[(b_row + group_id) * SMEM_STRIDE + k_off + tid * 4 + 16];
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

#pragma unroll
    for (int mi = 0; mi < M_TILES; mi++)
#pragma unroll
        for (int ni = 0; ni < N_TILES; ni++)
        {
            int row0 = bm + wm + mi * MMA_M + group_id, row1 = row0 + 8;
            int col = bn + wn + ni * MMA_N + tid * 2;
            if (row0 < M && col + 1 < N)
                *(uint32_t *)&C[row0 * N + col] = acc[mi][ni][0];
            if (row1 < M && col + 1 < N)
                *(uint32_t *)&C[row1 * N + col] = acc[mi][ni][1];
        }
}

int main()
{
    int M = 4096, N = 4096, K = 4096;
    void *dA, *dB, *dC;
    cudaMalloc(&dA, (size_t)M * K);
    cudaMalloc(&dB, (size_t)N * K);
    cudaMalloc(&dC, (size_t)M * N * 2);
    cudaMemset(dA, 0x38, (size_t)M * K);
    cudaMemset(dB, 0x38, (size_t)N * K);

    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN), block(BLOCK_THREADS);

    for (int i = 0; i < 10; i++)
        fp8_gemm_f16acc_kernel<<<grid, block>>>((uint8_t *)dA, (uint8_t *)dB, (uint16_t *)dC, M, N, K);
    cudaDeviceSynchronize();

    fp8_gemm_f16acc_kernel<<<grid, block>>>((uint8_t *)dA, (uint8_t *)dB, (uint16_t *)dC, M, N, K);
    cudaDeviceSynchronize();

    printf("v9 4096³ done\n");
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
