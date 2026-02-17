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

    // Смещения блока
    int m_idx = blockIdx.x * 128;
    int n_idx = blockIdx.y * 128;

    // 1. Глобальные тензоры
    Tensor mA = make_tensor(make_gmem_ptr(A + m_idx * K), make_shape(Int<128>{}, K), make_stride(K, _1{}));
    Tensor mB = make_tensor(make_gmem_ptr(B + n_idx * K), make_shape(Int<128>{}, K), make_stride(K, _1{}));
    Tensor mC = make_tensor(make_gmem_ptr(C + m_idx * N + n_idx), make_shape(Int<128>{}, Int<128>{}), make_stride(N, _1{}));

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);

    // 2. Разбиение GMEM
    auto tAgA = thr_mma.partition_A(mA); // (MMA, MMA_K, k_tiles)
    auto tBgB = thr_mma.partition_B(mB); // (MMA, MMA_K, k_tiles)
    auto tCgC = thr_mma.partition_C(mC); // (MMA, MMA_N)

    // 3. СТАТИЧЕСКАЯ АЛЛОКАЦИЯ (РЕШЕНИЕ)
    // Мы создаем фрагменты на основе ПЕРВОГО СТАЙСА, но приводим его лейаут к статическому
    auto tArA = make_fragment_like(tAgA(_, _, 0));
    auto tBrB = make_fragment_like(tBgB(_, _, 0));
    auto tCrC = make_fragment_like(tCgC);

    cute::clear(tCrC);

    // 4. Цикл по K
    int k_tiles = K / 64;

#pragma unroll 1
    for (int k = 0; k < k_tiles; ++k)
    {
        // Копируем данные из GMEM в регистры
        cute::copy(tAgA(_, _, k), tArA);
        cute::copy(tBgB(_, _, k), tBrB);

        // Tensor Core MMA (Инструкция SM89)
        cute::gemm(tiled_mma, tArA, tBrB, tCrC);
    }

    // 5. Выгрузка результата
    cute::copy(tCrC, tCgC);
}

extern "C" int run_cute_fp8_sm89(int M, int N, int K, const void *A, const void *B, void *C)
{
    // Та самая инструкция SM89 для Ada Lovelace (4090)
    using MMA_Op = SM89_16x8x32_F16E4M3E4M3F16_TN;
    using MMA_Atom_SM89 = MMA_Atom<MMA_Op>;

    using TiledMMA = TiledMMA<
        MMA_Atom_SM89,
        Layout<Shape<_2, _2, _1>>,
        Tile<Int<128>, Int<128>, Int<64>>>;

    dim3 grid((M + 127) / 128, (N + 127) / 128);
    dim3 block(size(TiledMMA{}));

    cute_fp8_kernel<TiledMMA><<<grid, block>>>(
        (const ElementA *)A, (const ElementB *)B, (ElementC *)C, M, N, K);

    return (int)cudaDeviceSynchronize();
}

// int main()
// {
//     printf("CuTe 4.4: SM89 FP16-Accum Kernel - Compilation Test SUCCESS\n");
//     return 0;
// }