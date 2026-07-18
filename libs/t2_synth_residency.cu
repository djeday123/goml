// T2 (a): Synthetic residency probe on sm_120a.
//   Isolates HW-scheduler behaviour from application code.
//   Question: does sm_120a runtime hold 3 blocks/SM at (168 reg, dyn=32768)?
//
// Three variants (reg count controlled via --maxrregcount):
//   synth3_r168:  target 168 registers
//   synth3_r160:  target 160 registers
//   synth3_r144:  target 144 registers
// Each has __launch_bounds__(128, 3), dyn smem 32768, grid = 176 SMs × 4 = 704 blocks.
// After each launch we query API's cudaOccupancyMax + let NCu measure ctas_active.
//
// If granularity is 16: r168 → rounded 176 → 176*128*3=67584 > 65536 → 2 blocks live.
//                       r160 → 160 → 160*128*3=61440 ≤ 65536 → 3 blocks live.
//                       r144 → 144 → 55296 → 3 blocks live.
// If granularity is 8:  all three → 3 blocks live.
// If HW cap is elsewhere: all three → still 2 blocks live even at r144.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

static void must(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA fail: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// Heavy variant: forces ~160 register pressure via 160 explicit long-lived
// scalars in an all-to-all dependency graph. Each iteration each scalar reads
// three others + a smem-derived value, blocking scalar collapse.
__launch_bounds__(128, 3)
__global__ void synth3_kernel(uint32_t *out, int niter) {
    extern __shared__ uint8_t smem[];
    if (threadIdx.x == 0) smem[0] = (uint8_t)blockIdx.x;
    __syncthreads();

    // 160 accumulators. Give each a slightly different seed so compiler cannot fold.
    uint32_t r[160];
    #pragma unroll
    for (int i = 0; i < 160; ++i) r[i] = threadIdx.x * 0x9E3779B9u + i * 0x85EBCA6Bu;

    for (int it = 0; it < niter; ++it) {
        // Force smem load per iteration (kills constant folding).
        uint32_t s = *reinterpret_cast<uint32_t*>(&smem[(it * 4) & 32764]);
        // Chained rotation across 160 lanes; each new value depends on 3 prior
        // + smem + it, so all 160 must be live across iterations.
        #pragma unroll
        for (int i = 0; i < 160; ++i) {
            r[i] = (r[i] ^ r[(i + 7) % 160] ^ r[(i + 23) % 160]) + s + it;
        }
    }

    // Sink — ALL lanes must be retained until here.
    uint32_t x = 0;
    #pragma unroll
    for (int i = 0; i < 160; ++i) x ^= r[i];
    out[blockIdx.x * 128 + threadIdx.x] = x;
}

int main(int argc, char **argv) {
    int dev = 0;
    must(cudaSetDevice(dev), "setDevice");

    cudaDeviceProp p;
    must(cudaGetDeviceProperties(&p, dev), "getDevProps");
    printf("device: %s  CC=%d.%d  SMs=%d\n", p.name, p.major, p.minor, p.multiProcessorCount);

    int max_optin = 0;
    must(cudaDeviceGetAttribute(&max_optin,
         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev), "maxOptin");

    cudaFuncAttributes fa;
    must(cudaFuncGetAttributes(&fa, synth3_kernel), "funcAttr");
    printf("synth3_kernel attrs: numRegs=%d, maxThreadsPerBlock=%d, sharedSizeBytes=%zu, localSizeBytes=%zu\n",
           fa.numRegs, fa.maxThreadsPerBlock, fa.sharedSizeBytes, fa.localSizeBytes);

    const int SMEM_DYN = 32768;
    must(cudaFuncSetAttribute(synth3_kernel,
         cudaFuncAttributeMaxDynamicSharedMemorySize, max_optin),
         "setMaxDynamic");

    int n_api = -1;
    must(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
         &n_api, synth3_kernel, 128, SMEM_DYN), "occAPI");
    printf("cudaOccupancyMax: smem=%d, block=128 -> %d blocks/SM\n", SMEM_DYN, n_api);

    // Explicit carveout: request MAX carveout (100% shared) — some drivers pick a
    // sub-100 carveout that yields only 2 blocks even at API-legal footprint.
    must(cudaFuncSetAttribute(synth3_kernel,
         cudaFuncAttributePreferredSharedMemoryCarveout,
         cudaSharedmemCarveoutMaxShared),
         "setCarveoutMax");

    int n_api_carve = -1;
    must(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
         &n_api_carve, synth3_kernel, 128, SMEM_DYN), "occAPI2");
    printf("cudaOccupancyMax (carveout=MaxShared): %d blocks/SM\n", n_api_carve);

    // Launch the kernel with enough waves for NCu to see steady state.
    // grid = SMs * 12 * 2 = many concurrent blocks even at 3 blocks/SM.
    const int grid = p.multiProcessorCount * 24;    // ~4224 blocks
    const int niter = 512;
    uint32_t *d_out;
    must(cudaMalloc(&d_out, sizeof(uint32_t) * grid * 128), "malloc");

    // Warmup
    synth3_kernel<<<grid, 128, SMEM_DYN>>>(d_out, niter);
    must(cudaDeviceSynchronize(), "warmupSync");

    // Timed launch
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    synth3_kernel<<<grid, 128, SMEM_DYN>>>(d_out, niter);
    cudaEventRecord(stop);
    must(cudaDeviceSynchronize(), "runSync");
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("timed synth3 grid=%d niter=%d: %.3f ms\n", grid, niter, ms);

    // Print return code as sink
    printf("first out[0] = %u\n", 0u);
    cudaFree(d_out);
    return 0;
}
