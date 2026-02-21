// =============================================================================
// FP8 MMA Throughput Test — Direct PTX, no CUTLASS type system nonsense
// =============================================================================
// This kernel directly emits the PTX instruction:
//   mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16
//
// If this instruction is hardware-supported on 4090 (AD102/SM89),
// we'll see TFLOPS close to theoretical peak.
// If NVIDIA blocked it in silicon, we'll get an illegal instruction error.
//
// Build:
//   nvcc -O3 -arch=sm_89 -std=c++17 fp8_throughput.cu -o fp8_throughput
//
// Run:
//   ./fp8_throughput
//
// Check SASS (to verify actual instruction):
//   cuobjdump -sass fp8_throughput | grep -i "mma\|HMMA"
// =============================================================================

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// -----------------------------------------------------------------------------
// Test 1: FP8 with FP16 accumulator (the one we want to test)
// PTX: mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16
// Each instruction does: M=16, N=8, K=32 → 16*8*32*2 = 8192 FLOPs per warp
// -----------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
    fp8_mma_f16acc_kernel(uint32_t *__restrict__ out, int num_iters)
{
    // Initialize fake operands (0x3C in e4m3 ≈ 1.0)
    uint32_t a0 = 0x3C3C3C3Cu, a1 = 0x3C3C3C3Cu, a2 = 0x3C3C3C3Cu, a3 = 0x3C3C3C3Cu;
    uint32_t b0 = 0x3C3C3C3Cu, b1 = 0x3C3C3C3Cu;
    uint32_t c0 = 0u, c1 = 0u;
    uint32_t d0, d1;

    for (int i = 0; i < num_iters; i++)
    {
// 8 back-to-back MMA instructions per iteration to reduce loop overhead
#define MMA_F16ACC()                                           \
    asm volatile(                                              \
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 " \
        "{%0, %1}, "                                           \
        "{%2, %3, %4, %5}, "                                   \
        "{%6, %7}, "                                           \
        "{%8, %9};\n"                                          \
        : "=r"(d0), "=r"(d1)                                   \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),                  \
          "r"(b0), "r"(b1),                                    \
          "r"(c0), "r"(c1));                                   \
    c0 = d0;                                                   \
    c1 = d1;

        MMA_F16ACC()
        MMA_F16ACC()
        MMA_F16ACC()
        MMA_F16ACC()
        MMA_F16ACC()
        MMA_F16ACC()
        MMA_F16ACC()
        MMA_F16ACC()
#undef MMA_F16ACC
    }

    // Prevent dead-code elimination
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        out[0] = d0;
        out[1] = d1;
    }
}

// -----------------------------------------------------------------------------
// Test 2: FP8 with FP32 accumulator (baseline for comparison)
// PTX: mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
// Same shape, but FP32 accumulation — this is the "known working" path
// -----------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
    fp8_mma_f32acc_kernel(uint32_t *__restrict__ out, int num_iters)
{
    uint32_t a0 = 0x3C3C3C3Cu, a1 = 0x3C3C3C3Cu, a2 = 0x3C3C3C3Cu, a3 = 0x3C3C3C3Cu;
    uint32_t b0 = 0x3C3C3C3Cu, b1 = 0x3C3C3C3Cu;
    // FP32 accumulator uses 4 registers (4 floats for m16n8 output)
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    float d0, d1, d2, d3;

    for (int i = 0; i < num_iters; i++)
    {
#define MMA_F32ACC()                                           \
    asm volatile(                                              \
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
        "{%0, %1, %2, %3}, "                                   \
        "{%4, %5, %6, %7}, "                                   \
        "{%8, %9}, "                                           \
        "{%10, %11, %12, %13};\n"                              \
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)               \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),                  \
          "r"(b0), "r"(b1),                                    \
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));                 \
    c0 = d0;                                                   \
    c1 = d1;                                                   \
    c2 = d2;                                                   \
    c3 = d3;

        MMA_F32ACC()
        MMA_F32ACC()
        MMA_F32ACC()
        MMA_F32ACC()
        MMA_F32ACC()
        MMA_F32ACC()
        MMA_F32ACC()
        MMA_F32ACC()
#undef MMA_F32ACC
    }

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        out[0] = __float_as_uint(d0);
        out[1] = __float_as_uint(d1);
    }
}

// -----------------------------------------------------------------------------
// Test 3: FP16 MMA (reference — known peak ~330 TFLOPS on 4090)
// PTX: mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
// Shape: M=16, N=8, K=16 → 16*8*16*2 = 4096 FLOPs per warp per instruction
// -----------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
    fp16_mma_kernel(uint32_t *__restrict__ out, int num_iters)
{
    uint32_t a0 = 0x3C003C00u, a1 = 0x3C003C00u, a2 = 0x3C003C00u, a3 = 0x3C003C00u;
    uint32_t b0 = 0x3C003C00u, b1 = 0x3C003C00u;
    uint32_t c0 = 0u, c1 = 0u;
    uint32_t d0, d1;

    for (int i = 0; i < num_iters; i++)
    {
#define MMA_FP16()                                           \
    asm volatile(                                            \
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 " \
        "{%0, %1}, "                                         \
        "{%2, %3, %4, %5}, "                                 \
        "{%6, %7}, "                                         \
        "{%8, %9};\n"                                        \
        : "=r"(d0), "=r"(d1)                                 \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),                \
          "r"(b0), "r"(b1),                                  \
          "r"(c0), "r"(c1));                                 \
    c0 = d0;                                                 \
    c1 = d1;

        MMA_FP16()
        MMA_FP16()
        MMA_FP16()
        MMA_FP16()
        MMA_FP16()
        MMA_FP16()
        MMA_FP16()
        MMA_FP16()
#undef MMA_FP16
    }

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        out[0] = d0;
        out[1] = d1;
    }
}

// =============================================================================
// Benchmark runner
// =============================================================================
struct BenchResult
{
    double tflops;
    bool ok;
    const char *error;
};

template <typename KernelFunc>
BenchResult run_bench(const char *name, KernelFunc kernel,
                      int flops_per_mma, int mmas_per_iter,
                      int num_blocks, int threads_per_block,
                      int warmup_iters, int bench_iters)
{
    BenchResult result = {0.0, false, nullptr};

    uint32_t *d_out;
    CHECK_CUDA(cudaMalloc(&d_out, 8));

    // Warmup
    kernel<<<num_blocks, threads_per_block>>>(d_out, warmup_iters);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        result.error = cudaGetErrorString(err);
        cudaFree(d_out);
        return result;
    }

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    kernel<<<num_blocks, threads_per_block>>>(d_out, bench_iters);
    CHECK_CUDA(cudaEventRecord(stop));

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess)
    {
        result.error = cudaGetErrorString(err);
        cudaFree(d_out);
        return result;
    }

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // Calculate TFLOPS
    // warps = total_threads / 32
    int total_warps = (num_blocks * threads_per_block) / 32;
    double total_mma_ops = (double)total_warps * bench_iters * mmas_per_iter;
    double total_flops = total_mma_ops * flops_per_mma;
    result.tflops = total_flops / (ms / 1000.0) / 1e12;
    result.ok = true;

    printf("  %-40s : %8.2f TFLOPS  (%.3f ms)\n", name, result.tflops, ms);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    cudaFree(d_out);
    return result;
}

int main()
{
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d, %d SMs, %.0f MHz)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.clockRate / 1000.0);
    printf("============================================================\n");

    int num_sms = prop.multiProcessorCount; // 128 for 4090
    int num_blocks = num_sms;               // 1 block per SM
    int threads_per_block = 256;            // 8 warps per block
    int warmup_iters = 200;
    int bench_iters = 2000;
    int mmas_per_iter = 8; // we unrolled 8 MMAs per loop iteration

    printf("\nConfig: %d blocks x %d threads, %d iters x %d MMAs/iter\n\n",
           num_blocks, threads_per_block, bench_iters, mmas_per_iter);

    // FP8 e4m3 x e4m3, FP16 accumulator — THE TEST
    // m16n8k32: 16*8*32*2 = 8192 FLOPs per MMA per warp
    {
        auto r = run_bench("FP8 e4m3 MMA, FP16 accumulator",
                           fp8_mma_f16acc_kernel,
                           8192, mmas_per_iter,
                           num_blocks, threads_per_block,
                           warmup_iters, bench_iters);
        if (!r.ok)
        {
            printf("  >>> FAILED: %s\n", r.error);
            printf("  >>> FP16 accumulator is BLOCKED on this GPU!\n");
        }
    }

    // FP8 e4m3 x e4m3, FP32 accumulator — baseline
    // m16n8k32: 16*8*32*2 = 8192 FLOPs per MMA per warp
    {
        auto r = run_bench("FP8 e4m3 MMA, FP32 accumulator",
                           fp8_mma_f32acc_kernel,
                           8192, mmas_per_iter,
                           num_blocks, threads_per_block,
                           warmup_iters, bench_iters);
        if (!r.ok)
        {
            printf("  >>> FAILED: %s\n", r.error);
        }
    }

    // FP16, FP16 accumulator — reference
    // m16n8k16: 16*8*16*2 = 4096 FLOPs per MMA per warp
    {
        auto r = run_bench("FP16 MMA, FP16 accumulator (reference)",
                           fp16_mma_kernel,
                           4096, mmas_per_iter,
                           num_blocks, threads_per_block,
                           warmup_iters, bench_iters);
        if (!r.ok)
        {
            printf("  >>> FAILED: %s\n", r.error);
        }
    }

    printf("\n============================================================\n");
    printf("Interpretation:\n");
    printf("  FP8+F32acc ~330 TFLOPS  = expected baseline for 4090\n");
    printf("  FP8+F16acc ~330 TFLOPS  = same speed, F16 path works but no speedup\n");
    printf("  FP8+F16acc ~600 TFLOPS  = JACKPOT - F16 acc doubles throughput\n");
    printf("  FP8+F16acc FAILED       = instruction blocked in hardware/driver\n");
    printf("  FP16 ref   ~330 TFLOPS  = standard FP16 Tensor Core peak\n");
    printf("============================================================\n");

    return 0;
}
