// Minimal mbarrier reproducer extracted from v111-mbarrier-real.
// Goal: reproduce hang at sl=300 wnd=96. NO MMA, NO softmax.
// Just: init/arrive/test_wait/parity protocol + same loop structure as v111.
//
// Build:
//   nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 mbar_repro.cu -o mbar_repro
//
// Run matrix:
//   sl ∈ {64, 300, 8192} × wnd ∈ {0, 96, 1024}

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <chrono>
#include <unistd.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);}}while(0)

// Match v111-mbarrier constants
#define FA_BR 96
#define FA_BC 64
#define FA_THREADS 128
#define FA_PRODUCERS 1
#define FA_CONSUMERS 3
#define K_STAGES 2

// ===== PTX helpers (identical to v111) =====
__device__ __forceinline__ void mbar_init(uint64_t *bar, uint32_t count) {
    uint32_t sa = __cvta_generic_to_shared(bar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(sa), "r"(count));
}

__device__ __forceinline__ void mbar_arrive(uint64_t *bar) {
    uint32_t sa = __cvta_generic_to_shared(bar);
    uint64_t token;
    asm volatile("mbarrier.arrive.shared.b64 %0, [%1];"
                 : "=l"(token) : "r"(sa) : "memory");
    (void)token;
}

__device__ __forceinline__ bool mbar_test_wait(uint64_t *bar, uint32_t phase) {
    uint32_t sa = __cvta_generic_to_shared(bar);
    uint32_t result;
    asm volatile(
        "{ .reg .pred P1;\n"
        "  mbarrier.test_wait.parity.shared.b64 P1, [%1], %2;\n"
        "  selp.b32 %0, 1, 0, P1;\n"
        "}"
        : "=r"(result) : "r"(sa), "r"(phase)
    );
    return result != 0;
}

__device__ __forceinline__ void mbar_wait(uint64_t *bar, uint32_t phase) {
    while (!mbar_test_wait(bar, phase)) { }
}

// ===== Repro kernel — same structure as v111-mbarrier main loop =====
__global__ void __launch_bounds__(FA_THREADS, 2)
    mbar_repro_kernel(int *out_iters, int seq_len, int causal, int window)
{
    int nqt = (seq_len + FA_BR - 1) / FA_BR;
    int bh = blockIdx.x / nqt, qt = blockIdx.x % nqt, qs = qt * FA_BR;
    if (qs >= seq_len) return;
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    bool is_producer = (wid < FA_PRODUCERS);

    extern __shared__ uint8_t raw[];
    uint64_t *bar_ready = (uint64_t *)raw;

    // Init mbarriers — count = 32 (one warp = producer)
    if (threadIdx.x == 0) {
        mbar_init(&bar_ready[0], 32);
        mbar_init(&bar_ready[1], 32);
    }
    __syncthreads();

    // Compute iter range — same as v111
    int nkv = (seq_len + FA_BC - 1) / FA_BC;
    int kv_max_blocks = causal ? ((qs + FA_BR - 1) / FA_BC + 1) : nkv;
    if (kv_max_blocks > nkv) kv_max_blocks = nkv;
    int kv_min_blocks = 0;
    if (window > 0 && qs + 1 > window) {
        kv_min_blocks = (qs - window + 1) / FA_BC;
    }

    // Pre-load arrive: producer does work then arrives at first slot
    if (wid == 0) {
        // Simulate work — small delay loop
        for (int i = 0; i < 10; i++) __nanosleep(100);
        mbar_arrive(&bar_ready[kv_min_blocks & 1]);
    }

    // Per-slot expected phase — init=1, toggle after each wait
    uint32_t expected_phase[K_STAGES] = {1, 1};

    int iters_done = 0;
    for (int kv = kv_min_blocks; kv < kv_max_blocks; kv++)
    {
        int buf = kv & 1;
        int nxt_buf = (kv + 1) & 1;

        // Consumer waits at current slot; producer skips
        if (!is_producer) {
            mbar_wait(&bar_ready[buf], expected_phase[buf]);
            expected_phase[buf] ^= 1u;
        }
        __syncthreads();

        // (no transpose, no compute — just the sync structure)
        __syncthreads();

        // Producer mid-iter prefetch + arrive at NEXT slot
        // Match v111: rows_p = 0 on last iter, but arrive still happens? Let's see v111 carefully:
        // In v111 lines 477-482: load_tile_fp8_warp called with rows_p, then cpa_commit/wait/arrive.
        // load_tile_fp8_warp with rows_p=0 issues no cp.async, but cpa_commit/wait still execute,
        // and mbar_arrive STILL fires. So arrive always happens.
        if (wid == 0) {
            for (int i = 0; i < 10; i++) __nanosleep(100);
            mbar_arrive(&bar_ready[nxt_buf]);
        }
        iters_done++;
    }

    if (threadIdx.x == 0) {
        out_iters[blockIdx.x] = iters_done;
    }
}

// ===== Host driver =====
int run_one(int bh, int sl, int causal, int window, double timeout_sec) {
    int nqt = (sl + FA_BR - 1) / FA_BR;
    int grid = bh * nqt;

    int *out_d, *out_h;
    CK(cudaMalloc(&out_d, grid * sizeof(int)));
    CK(cudaMemset(out_d, 0xff, grid * sizeof(int)));
    out_h = (int*)malloc(grid * sizeof(int));

    int smem = 64;  // 2 mbar × 8 B + padding
    CK(cudaFuncSetAttribute(mbar_repro_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    auto t0 = std::chrono::steady_clock::now();
    mbar_repro_kernel<<<grid, FA_THREADS, smem>>>(out_d, sl, causal, window);

    // Poll with timeout to detect hang
    bool hung = false;
    while (true) {
        cudaError_t e = cudaStreamQuery(0);
        if (e == cudaSuccess) break;
        if (e != cudaErrorNotReady) {
            fprintf(stderr, "  CUDA error: %s\n", cudaGetErrorString(e));
            hung = true;
            break;
        }
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        if (elapsed > timeout_sec) {
            hung = true;
            break;
        }
        usleep(10000);  // 10ms poll
    }

    int result_code;
    if (hung) {
        printf("  bh=%-3d sl=%-5d ca=%d wnd=%-5d grid=%-6d : HANG (>%.1fs)\n",
               bh, sl, causal, window, grid, timeout_sec);
        result_code = -1;
    } else {
        CK(cudaMemcpy(out_h, out_d, grid * sizeof(int), cudaMemcpyDeviceToHost));
        // Check all blocks completed expected iters
        int min_it = INT32_MAX, max_it = -1;
        for (int i = 0; i < grid; i++) {
            if (out_h[i] < min_it) min_it = out_h[i];
            if (out_h[i] > max_it) max_it = out_h[i];
        }
        printf("  bh=%-3d sl=%-5d ca=%d wnd=%-5d grid=%-6d : OK  iters=[%d..%d]\n",
               bh, sl, causal, window, grid, min_it, max_it);
        result_code = 0;
    }

    cudaFree(out_d);
    free(out_h);
    return result_code;
}

int main()
{
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("GPU: %s (CC %d.%d)\n", p.name, p.major, p.minor);
    printf("Br=%d Bc=%d FA_THREADS=%d (1P+3C)\n\n", FA_BR, FA_BC, FA_THREADS);

    printf("=== Matrix: sl × wnd ===\n");
    int sl_list[] = {64, 300, 8192};
    int wnd_list[] = {0, 96, 1024};
    int bh = 1;
    double timeout = 5.0;

    int n_hang = 0, n_ok = 0;
    for (int sl : sl_list) {
        for (int wnd : wnd_list) {
            // Match v111 convention: window > 0 implies causal
            int causal = (wnd > 0) ? 1 : 0;
            int r = run_one(bh, sl, causal, wnd, timeout);
            if (r == -1) n_hang++; else n_ok++;
        }
    }
    printf("\nSummary: %d OK, %d HANG\n", n_ok, n_hang);

    // Also test wnd=0 with full attention (no causal)
    printf("\n=== Full attention (wnd=0, causal=0) sanity ===\n");
    for (int sl : sl_list) {
        run_one(bh, sl, 0, 0, timeout);
    }

    return n_hang;
}
