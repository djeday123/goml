// Host runner: loads fp8_acc_strict.cubin, runs fp8_gemm_f16acc and
// fp8_gemm_f32acc across 12 GEMM shapes, times each with CUDA events,
// reports median over N_REP launches.

#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <string>

#define CK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { \
    const char *s = nullptr; cuGetErrorString(r, &s); \
    fprintf(stderr, "CU err %d (%s) at %s:%d\n", r, s ? s : "?", __FILE__, __LINE__); \
    std::exit(1); }} while (0)

struct Shape { int M, N, K; };

static const Shape shapes[] = {
    { 512,  512,  512},
    {1024, 1024, 1024},
    {1536, 1536, 1536},
    {2048, 2048, 2048},
    {2560, 2560, 2560},
    {3072, 3072, 3072},
    {3584, 3584, 3584},
    {4096, 4096, 4096},
    {2048, 2048,  512},
    {2048, 4096, 1024},
    {4096, 2048, 1024},
    {1024, 4096, 4096},
};

static double median(std::vector<float> &v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n & 1 ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
}

static void launch_once(CUfunction fn, void **args, int M, int N) {
    int gx = (N + 32 - 1) / 32;   // N_TILE = 32
    int gy = (M + 64 - 1) / 64;   // M_TILE = 64
    CK(cuLaunchKernel(fn, gx, gy, 1, 32, 1, 1, 0, 0, args, nullptr));
}

int main(int argc, char **argv) {
    const char *cubin_path = (argc > 1) ? argv[1] : "fp8_acc_strict.cubin";
    const int N_WARMUP = 3;
    const int N_REP = 20;

    CK(cuInit(0));
    CUdevice dev; CK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CK(cuCtxSetCurrent(ctx));

    CUmodule mod;
    CUresult r = cuModuleLoad(&mod, cubin_path);
    if (r != CUDA_SUCCESS) {
        const char *s = nullptr; cuGetErrorString(r, &s);
        fprintf(stderr, "cuModuleLoad('%s') -> %d (%s)\n", cubin_path, r, s ? s : "?");
        return 2;
    }
    CUfunction fn_f16, fn_f32;
    CK(cuModuleGetFunction(&fn_f16, mod, "fp8_gemm_f16acc"));
    CK(cuModuleGetFunction(&fn_f32, mod, "fp8_gemm_f32acc"));

    CUevent ev_start, ev_stop;
    CK(cuEventCreate(&ev_start, CU_EVENT_DEFAULT));
    CK(cuEventCreate(&ev_stop,  CU_EVENT_DEFAULT));

    printf("%5s %5s %5s | %12s %12s | %12s %12s | %7s\n",
           "M", "N", "K",
           "f16 ms (med)", "f16 TFLOPS",
           "f32 ms (med)", "f32 TFLOPS",
           "ratio");
    printf("--------------------------------------------------------------------------------------------\n");

    for (auto sh : shapes) {
        int M = sh.M, N = sh.N, K = sh.K;
        // Allocate device buffers — sized generously to absorb our deliberately
        // non-tight store pattern (the C sink in the kernel spreads writes
        // by 128 bytes per acc reg).
        size_t Abytes = (size_t)M * K;
        size_t Bbytes = (size_t)N * K;
        size_t Cbytes = (size_t)M * N * 16; // huge over-allocation, safe sink

        CUdeviceptr dA, dB, dC;
        CK(cuMemAlloc(&dA, Abytes));
        CK(cuMemAlloc(&dB, Bbytes));
        CK(cuMemAlloc(&dC, Cbytes));
        CK(cuMemsetD8(dA, 0x3a, Abytes));
        CK(cuMemsetD8(dB, 0x4b, Bbytes));

        void *args[] = { &dA, &dB, &dC, &M, &N, &K };

        // Warmup both
        for (int w = 0; w < N_WARMUP; ++w) { launch_once(fn_f16, args, M, N); }
        for (int w = 0; w < N_WARMUP; ++w) { launch_once(fn_f32, args, M, N); }
        CK(cuCtxSynchronize());

        auto time_kernel = [&](CUfunction fn) {
            std::vector<float> times;
            for (int rep = 0; rep < N_REP; ++rep) {
                CK(cuEventRecord(ev_start, 0));
                launch_once(fn, args, M, N);
                CK(cuEventRecord(ev_stop, 0));
                CK(cuEventSynchronize(ev_stop));
                float ms = 0;
                CK(cuEventElapsedTime(&ms, ev_start, ev_stop));
                times.push_back(ms);
            }
            return median(times);
        };

        double med_f16 = time_kernel(fn_f16);
        double med_f32 = time_kernel(fn_f32);

        double flops = 2.0 * (double)M * (double)N * (double)K;
        double tflops_f16 = flops / (med_f16 * 1e-3) / 1e12;
        double tflops_f32 = flops / (med_f32 * 1e-3) / 1e12;

        printf("%5d %5d %5d | %12.4f %12.2f | %12.4f %12.2f | %7.3f\n",
               M, N, K,
               med_f16, tflops_f16,
               med_f32, tflops_f32,
               tflops_f16 / tflops_f32);

        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
    }

    cuModuleUnload(mod);
    cuEventDestroy(ev_start);
    cuEventDestroy(ev_stop);
    cuDevicePrimaryCtxRelease(dev);
    return 0;
}
