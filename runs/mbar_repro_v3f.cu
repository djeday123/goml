// mbar_repro_v3f: v3e + watchdog progress tracking.
// Each warp writes progress[block*8 + warp] = iter*10 + checkpoint_id at key points.
// Host thread polls and prints state every 500ms.
// On hang: see final state — which block × warp stuck where.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <chrono>
#include <unistd.h>
#include <thread>
#include <atomic>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);}}while(0)

#define FA_BR 96
#define FA_BC 64
#define FA_THREADS 128
#define FA_PRODUCERS 1
#define FA_CONSUMERS 3
#define K_STAGES 2
#define FA_STRIDE 128

// Checkpoint IDs (single digit, so iter*10 + cp readable):
//   0 = enter iter
//   1 = after mbar_wait (consumer)  /  producer skip
//   2 = after sync L1
//   3 = after sync L2
//   4 = after mid-iter producer cp.async + mbar_arrive
//   5 = before bar.sync 1, 96 (consumer)
//   6 = after bar.sync 1, 96
//   9 = exit loop (after all iters)

__device__ __forceinline__ void cpa16(void *s, const void *g, int n) {
    uint32_t sa = __cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" ::"r"(sa), "l"(g), "r"(n));
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }

__device__ __forceinline__ int swz_byte(int row, int col_bytes) {
    int chunk = col_bytes >> 4;
    int within = col_bytes & 15;
    return row * FA_STRIDE + ((chunk ^ (row & 7)) << 4) + within;
}

__device__ __forceinline__ void load_tile_fp8_warp(
    uint8_t *dst, const uint8_t *src, int start, int rows,
    int seq_len, int head_dim, int lane)
{
    constexpr int CHUNK = 16;
    int chunks_per_row = head_dim / CHUNK;
    int total = rows * chunks_per_row;
#pragma unroll
    for (int c = lane; c < total; c += 32) {
        int row = c / chunks_per_row;
        int col_bytes = (c % chunks_per_row) * CHUNK;
        int gr = start + row;
        int dst_off = swz_byte(row, col_bytes);
        cpa16(&dst[dst_off], &src[gr * head_dim + col_bytes], (gr < seq_len) ? 16 : 0);
    }
}

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

// Progress write — only lane 0 of each warp.
__device__ __forceinline__ void mark(volatile int *progress, int wid, int lane, int val) {
    if (lane == 0) {
        progress[blockIdx.x * 8 + wid] = val;
        __threadfence_system();   // make visible to host via managed memory
    }
}

__global__ void __launch_bounds__(FA_THREADS, 2)
    mbar_repro_kernel(volatile int *progress, int *out_iters,
                      const uint8_t *K, const uint8_t *V,
                      int seq_len, int head_dim, int causal, int window)
{
    int nqt = (seq_len + FA_BR - 1) / FA_BR;
    int bh = blockIdx.x / nqt, qt = blockIdx.x % nqt, qs = qt * FA_BR;
    if (qs >= seq_len) return;
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    bool is_producer = (wid < FA_PRODUCERS);

    int hs = seq_len * head_dim;
    const uint8_t *Kh = K + bh * hs;
    const uint8_t *Vh = V + bh * hs;

    extern __shared__ uint8_t raw[];
    uint8_t *smK[K_STAGES] = { raw, raw + FA_BC * FA_STRIDE };
    uint8_t *smV = smK[1] + FA_BC * FA_STRIDE;
    uint64_t *bar_ready = (uint64_t *)((uintptr_t)(smV + FA_BC * FA_STRIDE + 7) & ~7ULL);

    if (threadIdx.x == 0) {
        mbar_init(&bar_ready[0], 32);
        mbar_init(&bar_ready[1], 32);
    }
    __syncthreads();

    int nkv = (seq_len + FA_BC - 1) / FA_BC;
    int kv_max_blocks = causal ? ((qs + FA_BR - 1) / FA_BC + 1) : nkv;
    if (kv_max_blocks > nkv) kv_max_blocks = nkv;
    int kv_min_blocks = 0;
    if (window > 0 && qs + 1 > window) {
        kv_min_blocks = (qs - window + 1) / FA_BC;
    }

    if (wid == 0) {
        load_tile_fp8_warp(smK[kv_min_blocks & 1], Kh, kv_min_blocks * FA_BC, FA_BC,
                           seq_len, head_dim, lane);
        load_tile_fp8_warp(smV, Vh, kv_min_blocks * FA_BC, FA_BC,
                           seq_len, head_dim, lane);
        cpa_commit();
        cpa_wait<0>();
        mbar_arrive(&bar_ready[kv_min_blocks & 1]);
    }

    uint32_t expected_phase[K_STAGES] = {1, 1};

    int iters_done = 0;
    for (int kv = kv_min_blocks; kv < kv_max_blocks; kv++)
    {
        mark(progress, wid, lane, kv * 10 + 0);

        int buf = kv & 1;
        int nxt_buf = (kv + 1) & 1;

        if (!is_producer) {
            mbar_wait(&bar_ready[buf], expected_phase[buf]);
            expected_phase[buf] ^= 1u;
        }
        mark(progress, wid, lane, kv * 10 + 1);
        __syncthreads();
        mark(progress, wid, lane, kv * 10 + 2);

        for (int i = 0; i < 5; i++) __nanosleep(50);
        __syncthreads();
        mark(progress, wid, lane, kv * 10 + 3);

        int kv_p = kv + 1;
        int rows_p = (kv_p < kv_max_blocks) ? FA_BC : 0;
        if (wid == 0) {
            load_tile_fp8_warp(smK[nxt_buf], Kh, kv_p * FA_BC, rows_p,
                               seq_len, head_dim, lane);
            load_tile_fp8_warp(smV, Vh, kv_p * FA_BC, rows_p,
                               seq_len, head_dim, lane);
            cpa_commit();
            cpa_wait<0>();
            mbar_arrive(&bar_ready[nxt_buf]);
        } else {
            cpa_commit();
        }
        mark(progress, wid, lane, kv * 10 + 4);

        for (int i = 0; i < 20; i++) __nanosleep(50);

        // v3e: single bar.sync 1, 96 consumer-only — MINIMUM TRIGGER
        mark(progress, wid, lane, kv * 10 + 5);
        if (!is_producer) {
            asm volatile("bar.sync 1, 96;");
        }
        mark(progress, wid, lane, kv * 10 + 6);

        for (int i = 0; i < 25; i++) __nanosleep(50);  // PV+smP sim

        iters_done++;
    }
    mark(progress, wid, lane, 999);  // 999 = clean exit

    if (threadIdx.x == 0) {
        out_iters[blockIdx.x] = iters_done;
    }
}

static std::atomic<bool> stop_watchdog{false};

void watchdog_thread(volatile int *progress, int n_blocks) {
    int iter = 0;
    while (!stop_watchdog.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        int last[32 * 8] = {0};
        for (int b = 0; b < n_blocks && b < 32; b++) {
            for (int w = 0; w < 4; w++) {
                last[b*8 + w] = progress[b*8 + w];
            }
        }
        fprintf(stderr, "[wd t=%.1fs]", (iter+1)*0.5);
        for (int b = 0; b < n_blocks && b < 4; b++) {
            fprintf(stderr, " B%d:[", b);
            for (int w = 0; w < 4; w++) {
                int v = last[b*8 + w];
                if (v == 999) fprintf(stderr, "DONE");
                else fprintf(stderr, "%d.%d", v/10, v%10);
                if (w < 3) fprintf(stderr, " ");
            }
            fprintf(stderr, "]");
        }
        fprintf(stderr, "\n");
        fflush(stderr);
        iter++;
    }
}

int main()
{
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("GPU: %s\n", p.name);
    printf("=== v3f watchdog dump ===\n");
    printf("Checkpoint legend: iter.cp where cp = 0=enter,1=after_mbar,2=after_syncL1,3=after_syncL2,4=after_mid_arrive,5=before_bar.sync_1,6=after_bar.sync_1,9=DONE\n");
    printf("Format: B<n>:[wid0 wid1 wid2 wid3]  — wid0 = producer, wid1/2/3 = consumers\n\n");
    setvbuf(stdout, NULL, _IONBF, 0);

    int sl = 300, hd = 128, ca = 0, wnd = 0;
    int bh = 1;
    int nqt = (sl + FA_BR - 1) / FA_BR;
    int grid = bh * nqt;
    printf("Config: sl=%d ca=%d wnd=%d grid=%d\n\n", sl, ca, wnd, grid);

    size_t kv_bytes = (size_t)bh * sl * hd;
    uint8_t *K_d, *V_d;
    CK(cudaMalloc(&K_d, kv_bytes));
    CK(cudaMalloc(&V_d, kv_bytes));
    CK(cudaMemset(K_d, 0x42, kv_bytes));
    CK(cudaMemset(V_d, 0x37, kv_bytes));

    // Managed memory for progress + iters
    int *progress;
    int *out_iters;
    CK(cudaMallocManaged(&progress, 32 * 8 * sizeof(int)));
    CK(cudaMallocManaged(&out_iters, grid * sizeof(int)));
    memset(progress, 0, 32 * 8 * sizeof(int));
    memset(out_iters, 0xff, grid * sizeof(int));

    int smem = 2 * FA_BC * FA_STRIDE + FA_BC * FA_STRIDE + 32;
    CK(cudaFuncSetAttribute(mbar_repro_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    // Start watchdog
    std::thread wd(watchdog_thread, progress, grid);

    auto t0 = std::chrono::steady_clock::now();
    mbar_repro_kernel<<<grid, FA_THREADS, smem>>>(progress, out_iters, K_d, V_d, sl, hd, ca, wnd);

    // Wait up to 8 seconds for kernel to finish
    bool hung = false;
    while (true) {
        cudaError_t e = cudaStreamQuery(0);
        if (e == cudaSuccess) break;
        if (e != cudaErrorNotReady) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));
            hung = true; break;
        }
        auto t1 = std::chrono::steady_clock::now();
        double el = std::chrono::duration<double>(t1 - t0).count();
        if (el > 8.0) { hung = true; break; }
        usleep(100000);
    }

    stop_watchdog.store(true);
    wd.join();

    printf("\n=== Final progress state (%s) ===\n", hung ? "HUNG" : "COMPLETED");
    for (int b = 0; b < grid; b++) {
        printf("Block %d: ", b);
        for (int w = 0; w < 4; w++) {
            int v = progress[b*8 + w];
            const char *role = (w == 0) ? "P" : "C";
            if (v == 999) printf("%s%d=DONE  ", role, w);
            else printf("%s%d=iter%d.cp%d  ", role, w, v/10, v%10);
        }
        printf("\n");
    }

    if (!hung) {
        printf("\nout_iters: ");
        for (int b = 0; b < grid; b++) printf("%d ", out_iters[b]);
        printf("\n");
    }

    cudaFree(K_d); cudaFree(V_d);
    cudaFree(progress); cudaFree(out_iters);
    return hung ? 1 : 0;
}
