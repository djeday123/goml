#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/arch.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_blockwise.h>
#include <cutlass/layout/matrix.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

typedef cutlass::float_e4m3_t FP8;
typedef cutlass::half_t FP16;
typedef cutlass::layout::RowMajor RM;
typedef cutlass::layout::ColumnMajor CM;
typedef float Acc;
typedef float Scale;
typedef cutlass::epilogue::thread::LinearCombination<FP16, 8, Acc, Acc> EpilogueOp;
typedef cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<> Swizzle;

// Config 1: 64x128x128 (from example 94)
typedef cutlass::gemm::device::GemmBlockwise<FP8, RM, FP8, CM, FP16, RM, Acc, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89, cutlass::gemm::GemmShape<64, 128, 128>, cutlass::gemm::GemmShape<64, 64, 128>, cutlass::gemm::GemmShape<16, 8, 32>, Scale, RM, 128, EpilogueOp, Swizzle, 3, 16, 16, false, cutlass::arch::OpMultiplyAdd> Gemm1;

// Config 2: 128x128x128
typedef cutlass::gemm::device::GemmBlockwise<FP8, RM, FP8, CM, FP16, RM, Acc, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89, cutlass::gemm::GemmShape<128, 128, 128>, cutlass::gemm::GemmShape<64, 64, 128>, cutlass::gemm::GemmShape<16, 8, 32>, Scale, RM, 128, EpilogueOp, Swizzle, 3, 16, 16, false, cutlass::arch::OpMultiplyAdd> Gemm2;

// Config 3: 64x256x128
typedef cutlass::gemm::device::GemmBlockwise<FP8, RM, FP8, CM, FP16, RM, Acc, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89, cutlass::gemm::GemmShape<64, 256, 128>, cutlass::gemm::GemmShape<64, 64, 128>, cutlass::gemm::GemmShape<16, 8, 32>, Scale, RM, 128, EpilogueOp, Swizzle, 3, 16, 16, false, cutlass::arch::OpMultiplyAdd> Gemm3;

struct Fp8Args
{
    void *handle;
    int M, N, K;
    int _pad;
    const void *A;
    const void *B;
    void *C;
    const float *alpha;
    const float *beta;
};

static float *s_scaleA = nullptr;
static float *s_scaleB = nullptr;
static int s_scA_sz = 0;
static int s_scB_sz = 0;
static void *s_ws = nullptr;
static size_t s_ws_sz = 0;

static int cdiv(int a, int b) { return (a + b - 1) / b; }

static void ensure_scales(int M, int N, int K)
{
    int kB = cdiv(K, 128), mB = cdiv(M, 128), nB = cdiv(N, 128);
    int needA = mB * kB, needB = nB * kB;
    if (needA > s_scA_sz)
    {
        if (s_scaleA)
            cudaFree(s_scaleA);
        cudaMalloc(&s_scaleA, needA * sizeof(float));
        s_scA_sz = needA;
        std::vector<float> ones(needA, 1.0f);
        cudaMemcpy(s_scaleA, ones.data(), needA * sizeof(float), cudaMemcpyHostToDevice);
    }
    if (needB > s_scB_sz)
    {
        if (s_scaleB)
            cudaFree(s_scaleB);
        cudaMalloc(&s_scaleB, needB * sizeof(float));
        s_scB_sz = needB;
        std::vector<float> ones(needB, 1.0f);
        cudaMemcpy(s_scaleB, ones.data(), needB * sizeof(float), cudaMemcpyHostToDevice);
    }
}

template <typename G>
static int run_blockwise(Fp8Args *a)
{
    ensure_scales(a->M, a->N, a->K);
    int kB = cdiv(a->K, 128);
    typename G::EpilogueOutputOp::Params epi(*(a->alpha), *(a->beta));
    RM layA(kB), layB(kB);
    typename G::TensorRefScale rA(s_scaleA, layA), rB(s_scaleB, layB);
    typename G::Arguments args({a->M, a->N, a->K}, {(FP8 const *)a->A, a->K}, {(FP8 const *)a->B, a->K}, {(FP16 const *)a->C, a->N}, {(FP16 *)a->C, a->N}, rA, rB, epi, 1, nullptr, nullptr, nullptr);
    G op;
    cutlass::Status st = op.can_implement(args);
    if (st != cutlass::Status::kSuccess)
        return -1;
    size_t ws = G::get_workspace_size(args);
    if (ws > s_ws_sz)
    {
        if (s_ws)
            cudaFree(s_ws);
        cudaMalloc(&s_ws, ws);
        s_ws_sz = ws;
    }
    st = op.initialize(args, s_ws);
    if (st != cutlass::Status::kSuccess)
        return -2;
    st = op();
    return (st == cutlass::Status::kSuccess) ? 0 : -3;
}

extern "C"
{

    int cutlass_fp8_gemm(Fp8Args *a)
    {
        return run_blockwise<Gemm1>(a);
    }

    int cutlass_fp8_gemm_bench(Fp8Args *a)
    {
        const char *names[] = {"64x128x128", "128x128x128", "64x256x128"};
        float best_ms = 1e30f;
        int best = -1;

        // warmup
        run_blockwise<Gemm1>(a);
        cudaDeviceSynchronize();

        for (int cfg = 0; cfg < 3; cfg++)
        {
            int r;
            switch (cfg)
            {
            case 0:
                r = run_blockwise<Gemm1>(a);
                break;
            case 1:
                r = run_blockwise<Gemm2>(a);
                break;
            default:
                r = run_blockwise<Gemm3>(a);
                break;
            }
            if (r != 0)
            {
                fprintf(stderr, "[CUTLASS] %s: SKIP (err=%d)\n", names[cfg], r);
                continue;
            }
            cudaDeviceSynchronize();
            cudaEvent_t t0, t1;
            cudaEventCreate(&t0);
            cudaEventCreate(&t1);
            cudaEventRecord(t0);
            for (int i = 0; i < 20; i++)
            {
                switch (cfg)
                {
                case 0:
                    run_blockwise<Gemm1>(a);
                    break;
                case 1:
                    run_blockwise<Gemm2>(a);
                    break;
                default:
                    run_blockwise<Gemm3>(a);
                    break;
                }
            }
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            float ms;
            cudaEventElapsedTime(&ms, t0, t1);
            cudaEventDestroy(t0);
            cudaEventDestroy(t1);
            float avg = ms / 20.0f;
            double flops = 2.0 * (double)a->M * (double)a->N * (double)a->K;
            double tflops = (flops / (avg / 1000.0)) / 1e12;
            fprintf(stderr, "[CUTLASS] %s: %.3f ms (%.1f TFLOPS)\n", names[cfg], avg, tflops);
            if (ms < best_ms)
            {
                best_ms = ms;
                best = cfg;
            }
        }
        fprintf(stderr, "[CUTLASS] best: %s\n", best >= 0 ? names[best] : "NONE");
        return best;
    }

    int cuda_device_sync(void) { return (int)cudaDeviceSynchronize(); }
}