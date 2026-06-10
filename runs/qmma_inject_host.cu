// Host driver: loads a (possibly patched) cubin via the CUDA Driver API
// and runs `qmma_baseline(A, B, C, iters)`. Reports:
//   * whether launch succeeded or hit cudaErrorInvalidInstruction
//   * the first few elements of C (so we can compare patched vs baseline)
//   * wall-clock time per launch (rough throughput indicator)
//
// Compile:
//   nvcc -O2 qmma_inject_host.cu -lcuda -o qmma_host
// Usage:
//   ./qmma_host <cubin_path>   [iters]   [warps_per_block]   [blocks]

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <chrono>

#define CK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { \
    const char *s = nullptr; cuGetErrorString(r, &s); \
    fprintf(stderr, "CU err %d (%s) at %s:%d\n", r, s ? s : "?", __FILE__, __LINE__); \
    std::exit(1); }} while (0)

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <cubin> [iters] [warps] [blocks]\n", argv[0]); return 1; }
    const char *cubin_path = argv[1];
    int iters  = (argc > 2) ? atoi(argv[2]) : 4;
    int warps  = (argc > 3) ? atoi(argv[3]) : 1;
    int blocks = (argc > 4) ? atoi(argv[4]) : 1;

    CK(cuInit(0));
    CUdevice dev; CK(cuDeviceGet(&dev, 0));
    // Use primary context to avoid cuCtxCreate signature drift across CUDA versions.
    CUcontext ctx; CK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CK(cuCtxSetCurrent(ctx));

    CUmodule mod; CUresult r = cuModuleLoad(&mod, cubin_path);
    if (r != CUDA_SUCCESS) {
        const char *s = nullptr; cuGetErrorString(r, &s);
        fprintf(stderr, "cuModuleLoad('%s') -> %d (%s)\n", cubin_path, r, s ? s : "?");
        return 2;
    }
    CUfunction fn; CK(cuModuleGetFunction(&fn, mod, "qmma_baseline"));

    // m=16, n=8, k=16: A = 16*16 FP8 = 256 B, B = 8*16 FP8 = 128 B, C = 16*8 FP16 = 256 B.
    // We allocate enough for one tile per block (×blocks).
    size_t Abytes = 256 * blocks;
    size_t Bbytes = 128 * blocks;
    size_t Cbytes = 256 * blocks;

    CUdeviceptr dA, dB, dC;
    CK(cuMemAlloc(&dA, Abytes));
    CK(cuMemAlloc(&dB, Bbytes));
    CK(cuMemAlloc(&dC, Cbytes));

    // Deterministic, distinguishable input pattern
    std::vector<uint8_t> hA(Abytes), hB(Bbytes), hC(Cbytes, 0);
    for (size_t i = 0; i < Abytes; ++i) hA[i] = uint8_t((i * 7 + 3) & 0xff);
    for (size_t i = 0; i < Bbytes; ++i) hB[i] = uint8_t((i * 11 + 5) & 0xff);
    CK(cuMemcpyHtoD(dA, hA.data(), Abytes));
    CK(cuMemcpyHtoD(dB, hB.data(), Bbytes));
    CK(cuMemcpyHtoD(dC, hC.data(), Cbytes));

    void *args[] = { &dA, &dB, &dC, &iters };
    int threadsPerBlock = 32 * warps;

    // Warmup
    r = cuLaunchKernel(fn, blocks,1,1, threadsPerBlock,1,1, 0, 0, args, nullptr);
    if (r != CUDA_SUCCESS) {
        const char *s = nullptr; cuGetErrorString(r, &s);
        fprintf(stderr, "warmup launch -> %d (%s)\n", r, s ? s : "?");
        return 3;
    }
    r = cuCtxSynchronize();
    if (r != CUDA_SUCCESS) {
        const char *s = nullptr; cuGetErrorString(r, &s);
        fprintf(stderr, "warmup sync -> %d (%s)\n", r, s ? s : "?");
        return 4;
    }

    // Timed loop
    const int N_REP = 32;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int rep = 0; rep < N_REP; ++rep) {
        CK(cuLaunchKernel(fn, blocks,1,1, threadsPerBlock,1,1, 0, 0, args, nullptr));
    }
    CK(cuCtxSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    double total_mma = double(N_REP) * iters * blocks * warps;
    printf("LAUNCH OK\n");
    printf("  reps=%d  iters/rep=%d  blocks=%d  warps/block=%d\n", N_REP, iters, blocks, warps);
    printf("  wall=%.6f s   mma_issues=%.3e   issues/sec=%.3e\n",
           secs, total_mma, total_mma / secs);

    // Read back first 16 bytes of C, print as FP16 values so we can compare
    // baseline vs patched.
    CK(cuMemcpyDtoH(hC.data(), dC, Cbytes));
    const uint16_t *u16 = reinterpret_cast<const uint16_t*>(hC.data());
    printf("  C[0..7] as raw u16: ");
    for (int i = 0; i < 8; ++i) printf("%04x ", u16[i]);
    printf("\n");

    cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
    cuModuleUnload(mod);
    cuDevicePrimaryCtxRelease(dev);
    return 0;
}
