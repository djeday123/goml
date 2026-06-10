// Strict A/B: kind::f8f6f4 with f16 acc vs f32 acc, real GEMM workload.
//
// Both kernels are generated from the same template. The ONLY difference
// is the accumulator token in the inline PTX mma instruction (.f16 vs .f32)
// and the corresponding register count for the C frag (2 vs 4 .b32).
//
// Per-warp output tile: 64 M × 32 N = 4 M-tiles × 4 N-tiles = 16 m16n8 tiles.
// K-step = 32 (one m16n8k32 mma per tile per K iteration).
//
// Reg pressure per thread:
//   f32 acc:  16 tiles × 4 b32 = 64 b32 C-regs  + A,B frags (16 b32) ≈ 80 regs
//   f16 acc:  16 tiles × 2 b32 = 32 b32 C-regs  + A,B frags (16 b32) ≈ 48 regs
//
// We do NOT bother with proper warp lane layout for A/B fragments — output
// correctness is not the measurement target. Throughput is. Both variants
// run identical memory traffic patterns so any divergence is acc-only.

#include <cstdint>

constexpr int N_M_MMA = 4;
constexpr int N_N_MMA = 4;
constexpr int M_TILE  = 16 * N_M_MMA;   // 64
constexpr int N_TILE  = 8  * N_N_MMA;   // 32
constexpr int K_TILE  = 32;

template <bool USE_F32_ACC>
__device__ __forceinline__ void fp8_gemm_inner(
    const uint8_t * __restrict__ A,
    const uint8_t * __restrict__ B,
    uint8_t       * __restrict__ C,
    int M, int N, int K)
{
    int lane = threadIdx.x & 31;
    int block_m = blockIdx.y * M_TILE;
    int block_n = blockIdx.x * N_TILE;
    if (block_m >= M || block_n >= N) return;

    // C accumulator
    constexpr int C_REGS = USE_F32_ACC ? 4 : 2;
    uint32_t c[N_M_MMA][N_N_MMA][C_REGS];
    #pragma unroll
    for (int mi = 0; mi < N_M_MMA; ++mi)
        #pragma unroll
        for (int ni = 0; ni < N_N_MMA; ++ni)
            #pragma unroll
            for (int r = 0; r < C_REGS; ++r)
                c[mi][ni][r] = 0;

    for (int k0 = 0; k0 < K; k0 += K_TILE) {
        // Load A: per M-tile, frag = 16M × 32K FP8 = 512B. Each thread → 4 b32.
        uint32_t a[N_M_MMA][4];
        #pragma unroll
        for (int mi = 0; mi < N_M_MMA; ++mi) {
            int row_base = block_m + mi * 16 + (lane / 4);
            int k_base   = k0 + (lane % 4) * 4;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int row = row_base + (j & 1) * 8;
                int kk  = k_base   + ((j >> 1) & 1) * 16;
                row = row < M ? row : 0;
                kk  = kk  < K ? kk  : 0;
                a[mi][j] = *reinterpret_cast<const uint32_t*>(&A[row * K + kk]);
            }
        }
        // Load B: per N-tile, frag = 8N × 32K FP8 = 256B. Each thread → 2 b32.
        uint32_t b[N_N_MMA][2];
        #pragma unroll
        for (int ni = 0; ni < N_N_MMA; ++ni) {
            int col_base = block_n + ni * 8 + (lane / 4);
            int k_base   = k0 + (lane % 4) * 4;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int col = col_base;
                int kk  = k_base + j * 16;
                col = col < N ? col : 0;
                kk  = kk  < K ? kk  : 0;
                b[ni][j] = *reinterpret_cast<const uint32_t*>(&B[col * K + kk]);
            }
        }

        // 16 MMA instructions.
        #pragma unroll
        for (int mi = 0; mi < N_M_MMA; ++mi) {
            #pragma unroll
            for (int ni = 0; ni < N_N_MMA; ++ni) {
                if constexpr (USE_F32_ACC) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(c[mi][ni][0]), "+r"(c[mi][ni][1]),
                          "+r"(c[mi][ni][2]), "+r"(c[mi][ni][3])
                        : "r"(a[mi][0]), "r"(a[mi][1]),
                          "r"(a[mi][2]), "r"(a[mi][3]),
                          "r"(b[ni][0]), "r"(b[ni][1]));
                } else {
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e4m3.e4m3.f16 "
                        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                        : "+r"(c[mi][ni][0]), "+r"(c[mi][ni][1])
                        : "r"(a[mi][0]), "r"(a[mi][1]),
                          "r"(a[mi][2]), "r"(a[mi][3]),
                          "r"(b[ni][0]), "r"(b[ni][1]));
                }
            }
        }
    }

    // Sink C to gmem so compiler can't DCE the accumulator.
    // Stride matches per-thread C frag (4 b32 for f32, 2 b32 for f16) per tile.
    uint8_t *Cbase = C + (size_t)block_m * N * (USE_F32_ACC ? 4 : 2)
                       + (size_t)block_n * (USE_F32_ACC ? 4 : 2);
    #pragma unroll
    for (int mi = 0; mi < N_M_MMA; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < N_N_MMA; ++ni) {
            size_t off = ((size_t)mi * N * 16 + (size_t)ni * 8) * (USE_F32_ACC ? 4 : 2)
                       + lane * sizeof(uint32_t);
            #pragma unroll
            for (int r = 0; r < C_REGS; ++r) {
                size_t total_off = off + (size_t)r * 128;  // arbitrary spread
                // Bounds check: we allocated enough host-side, see host code.
                *reinterpret_cast<uint32_t*>(Cbase + total_off) = c[mi][ni][r];
            }
        }
    }
}

extern "C" __global__ void fp8_gemm_f32acc(
    const uint8_t * __restrict__ A,
    const uint8_t * __restrict__ B,
    uint8_t       * __restrict__ C,
    int M, int N, int K)
{
    fp8_gemm_inner<true>(A, B, C, M, N, K);
}

extern "C" __global__ void fp8_gemm_f16acc(
    const uint8_t * __restrict__ A,
    const uint8_t * __restrict__ B,
    uint8_t       * __restrict__ C,
    int M, int N, int K)
{
    fp8_gemm_inner<false>(A, B, C, M, N, K);
}
