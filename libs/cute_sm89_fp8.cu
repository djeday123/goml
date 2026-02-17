#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

using namespace cute;

using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = cutlass::half_t;

template <typename TiledMMA>
__global__ void cute_fp8_kernel(const ElementA *A, const ElementB *B, ElementC *C,
                                int M, int N, int K)
{
    using namespace cute;

    // 1. Параметры тайлинга (Статические)
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<64>{};

    // 2. Аллокация Shared Memory
    __shared__ ElementA sA[size(Shape<Int<128>, Int<64>>{})];
    __shared__ ElementB sB[size(Shape<Int<128>, Int<64>>{})];

    // 3. Глобальные тензоры
    Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, _1{}));
    Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), make_stride(K, _1{}));
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, _1{}));

    // Локальные тайлы для блока
    Tensor gA = local_tile(mA, make_shape(bM, bK), make_coord(blockIdx.x, _));
    Tensor gB = local_tile(mB, make_shape(bN, bK), make_coord(blockIdx.y, _));
    Tensor gC = local_tile(mC, make_shape(bM, bN), make_coord(blockIdx.x, blockIdx.y));

    // Тензоры в Shared Memory
    Tensor sA_t = make_tensor(make_smem_ptr(sA), make_shape(bM, bK), make_stride(bK, _1{}));
    Tensor sB_t = make_tensor(make_smem_ptr(sB), make_shape(bN, bK), make_stride(bK, _1{}));

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);

    // 4. Разбиение для MMA и Копирования
    auto tAgA = thr_mma.partition_A(gA);
    auto tAsA = thr_mma.partition_A(sA_t);
    auto tBgB = thr_mma.partition_B(gB);
    auto tBsB = thr_mma.partition_B(sB_t);

    auto tCrC = thr_mma.partition_fragment_C(gC);
    clear(tCrC);

    // 5. Фрагменты для вычислений (уже из Smem)
    auto tArA = thr_mma.make_fragment_A(tAsA);
    auto tBrB = thr_mma.make_fragment_B(tBsB);

    int k_tiles = size<2>(gA);
    for (int k = 0; k < k_tiles; ++k)
    {
        // Копируем из GMEM в SMEM (используем все потоки для скорости)
        // Для простоты используем copy(), в продакшене тут cp.async
        copy(gA(_, _, k), sA_t);
        copy(gB(_, _, k), sB_t);
        cp_async_wait<0>();
        __syncthreads();

        // GEMM из Smem в Регистры
        gemm(tiled_mma, tArA, tBrB, tCrC);
        __syncthreads();
    }

    auto tCgC = thr_mma.partition_C(gC);
    copy(tCrC, tCgC);
}

extern "C" int run_cute_fp8_sm89(int M, int N, int K, const void *A, const void *B, void *C)
{
    using MMA_Op = SM89_16x8x32_F16E4M3E4M3F16_TN;
    using TiledMMA = TiledMMA<MMA_Atom<MMA_Op>, Layout<Shape<_2, _2, _1>>, Tile<Int<128>, Int<128>, Int<64>>>;

    dim3 grid((M + 127) / 128, (N + 127) / 128);
    dim3 block(size(TiledMMA{}));

    cute_fp8_kernel<TiledMMA><<<grid, block>>>((const ElementA *)A, (const ElementB *)B, (ElementC *)C, M, N, K);
    return (int)cudaDeviceSynchronize();
}