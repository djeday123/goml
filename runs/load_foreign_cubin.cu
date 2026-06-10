// load_foreign_cubin.cu — try to load sm_100a tcgen05 cubin onto sm_120
// hardware via the driver API. Goal: get the exact CUresult code.
//
// Possible outcomes:
//   CUDA_ERROR_NO_BINARY_FOR_GPU (209) — driver refuses by arch mismatch BEFORE
//      hardware ever sees the bytes. Tells us nothing about the decoder.
//   CUDA_ERROR_INVALID_PTX (218) — for PTX path only.
//   Launch returns 715 (CUDA_ERROR_ILLEGAL_INSTRUCTION) on sync — hardware
//      decoder actually rejected the opcode. This is the strongest evidence
//      of hardware absence.
//   Launch returns 700 (CUDA_ERROR_LAUNCH_FAILED) — generic, less specific
//      but still suggests the dispatcher couldn't handle it.

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>
#include <fstream>

#define CKD(c) do { CUresult _r = (c); if (_r != CUDA_SUCCESS) { \
    const char *s = nullptr; cuGetErrorString(_r, &s); \
    printf("DRIVER %s:%d → %d (%s)\n", __FILE__, __LINE__, (int)_r, s ? s : "?"); }} while(0)

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: %s <cubin_path>\n", argv[0]); return 1; }
    cuInit(0);
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    char name[256] = {0};
    int major = 0, minor = 0;
    cuDeviceGetName(name, 256, dev);
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    printf("Device: %s (sm_%d%d)\n", name, major, minor);

    CUcontext ctx;
    // Use primary context — avoids CUDA 13's cuCtxCreate signature flux.
    cuDevicePrimaryCtxRetain(&ctx, dev);
    cuCtxSetCurrent(ctx);

    std::ifstream f(argv[1], std::ios::binary);
    if (!f) { printf("can't open %s\n", argv[1]); return 1; }
    std::vector<char> bin((std::istreambuf_iterator<char>(f)), {});
    printf("loaded %zu bytes from %s\n", bin.size(), argv[1]);

    CUmodule mod = nullptr;
    printf("--- cuModuleLoadData ---\n");
    CUresult lr = cuModuleLoadData(&mod, bin.data());
    const char *ls = nullptr;
    cuGetErrorString(lr, &ls);
    printf("  → %d (%s)\n", (int)lr, ls ? ls : "?");

    if (lr == CUDA_SUCCESS) {
        CUfunction fn = nullptr;
        CKD(cuModuleGetFunction(&fn, mod, "_Z14tcgen05_kernelv"));
        if (fn) {
            printf("--- cuLaunchKernel ---\n");
            CUresult lk = cuLaunchKernel(fn, 1,1,1, 32,1,1, 0, 0, nullptr, nullptr);
            cuGetErrorString(lk, &ls);
            printf("  → %d (%s)\n", (int)lk, ls ? ls : "?");
            printf("--- cuCtxSynchronize ---\n");
            CUresult sr = cuCtxSynchronize();
            cuGetErrorString(sr, &ls);
            printf("  → %d (%s)\n", (int)sr, ls ? ls : "?");
        }
        cuModuleUnload(mod);
    }
    cuDevicePrimaryCtxRelease(dev);
    return 0;
}
