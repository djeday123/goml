// mbar_repro_v2: ADD cp.async to mbar_repro.
// Step from M1 → M2: real cp.async.cg loads with K_STAGES=2 double-buffer,
// cpa_commit/cpa_wait in the SAME positions relative to arrive/wait as v111-mbarrier.
// NO MMA, NO transpose, NO smP — just mbarrier + cp.async.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <chrono>
#include <unistd.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);}}while(0)

// Match v111-mbarrier constants EXACTLY
#define FA_BR 96
#define FA_BC 64
#define FA_THREADS 128
#define FA_PRODUCERS 1
#define FA_CONSUMERS 3
#define K_STAGES 2
#define FA_STRIDE 128       // padded SMEM row stride
#define SMV_T_STRIDE 68

// ===== PTX: cp.async (identical to v111) =====
__device__ __forceinline__ void cpa16(void *s, const void *g, int n) {
    uint32_t sa = __cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" ::"r"(sa), "l"(g), "r"(n));
}
__device__ __forceinline__ void cpa_commit() {
    asm volatile("cp.async.commit_group;");
}
template <int N>
__device__ __forceinline__ void cpa_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

// ===== Same SMEM swizzle as v111 =====
__device__ __forceinline__ int swz_byte(int row, int col_bytes) {
    int chunk = col_bytes >> 4;
    int within = col_bytes & 15;
    return row * FA_STRIDE + ((chunk ^ (row & 7)) << 4) + within;
}

// ===== Single-warp loader (lane 0..31 only) — identical to v111 =====
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

// ===== mbarrier (identical to v111) =====
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

// ===== Kernel — same structure as v111-mbarrier main loop =====
__global__ void __launch_bounds__(FA_THREADS, 2)
    mbar_repro_kernel(int *out_iters,
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

    // SMEM layout: smK[2] + smV + bar_ready[2]
    extern __shared__ uint8_t raw[];
    uint8_t *smK[K_STAGES] = {
        raw,
        raw + FA_BC * FA_STRIDE,
    };
    uint8_t *smV = smK[1] + FA_BC * FA_STRIDE;
    uint64_t *bar_ready = (uint64_t *)((uintptr_t)(smV + FA_BC * FA_STRIDE + 7) & ~7ULL);

    // Init mbarriers — count=32 (producer warp)
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

    // PRE-LOAD — identical pattern to v111-mbarrier lines 432-440
    if (wid == 0) {
        load_tile_fp8_warp(smK[kv_min_blocks & 1], Kh, kv_min_blocks * FA_BC, FA_BC,
                           seq_len, head_dim, lane);
        load_tile_fp8_warp(smV, Vh, kv_min_blocks * FA_BC, FA_BC,
                           seq_len, head_dim, lane);
        cpa_commit();
        cpa_wait<0>();                              // wait own cp.async
        mbar_arrive(&bar_ready[kv_min_blocks & 1]); // regular arrive
    }

    uint32_t expected_phase[K_STAGES] = {1, 1};

    int iters_done = 0;
    for (int kv = kv_min_blocks; kv < kv_max_blocks; kv++)
    {
        int buf = kv & 1;
        int nxt_buf = (kv + 1) & 1;

        // Consumer waits (producer skips top sync) — same as v111
        if (!is_producer) {
            mbar_wait(&bar_ready[buf], expected_phase[buf]);
            expected_phase[buf] ^= 1u;
        }
        __syncthreads();

        // NO transpose_v here (v111 has it). NO compute.
        __syncthreads();

        // Producer mid-iter prefetch — IDENTICAL to v111 lines 476-486
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
        iters_done++;
    }

    if (threadIdx.x == 0) {
        out_iters[blockIdx.x] = iters_done;
    }
}

// ===== Host driver =====
int run_one(int bh, int sl, int causal, int window, double timeout_sec) {
    int hd = 128;
    int nqt = (sl + FA_BR - 1) / FA_BR;
    int grid = bh * nqt;

    size_t kv_bytes = (size_t)bh * sl * hd;
    uint8_t *K_d, *V_d;
    CK(cudaMalloc(&K_d, kv_bytes));
    CK(cudaMalloc(&V_d, kv_bytes));
    CK(cudaMemset(K_d, 0x42, kv_bytes));
    CK(cudaMemset(V_d, 0x37, kv_bytes));

    int *out_d, *out_h;
    CK(cudaMalloc(&out_d, grid * sizeof(int)));
    CK(cudaMemset(out_d, 0xff, grid * sizeof(int)));
    out_h = (int*)malloc(grid * sizeof(int));

    int smem = 2 * FA_BC * FA_STRIDE + FA_BC * FA_STRIDE + 32;
    CK(cudaFuncSetAttribute(mbar_repro_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    auto t0 = std::chrono::steady_clock::now();
    mbar_repro_kernel<<<grid, FA_THREADS, smem>>>(out_d, K_d, V_d, sl, hd, causal, window);

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
        usleep(10000);
    }

    int rc;
    if (hung) {
        rc = -1;
    } else {
        CK(cudaMemcpy(out_h, out_d, grid * sizeof(int), cudaMemcpyDeviceToHost));
        int min_it = INT32_MAX, max_it = -1;
        for (int i = 0; i < grid; i++) {
            if (out_h[i] < min_it) min_it = out_h[i];
            if (out_h[i] > max_it) max_it = out_h[i];
        }
        rc = 0;
        (void)min_it; (void)max_it;
    }

    cudaFree(K_d); cudaFree(V_d);
    cudaFree(out_d); free(out_h);
    return rc;
}

int main(int argc, char **argv)
{
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("GPU: %s (CC %d.%d)\n", p.name, p.major, p.minor);
    printf("Br=%d Bc=%d FA_THREADS=%d (1P+3C) K_STAGES=%d\n\n",
           FA_BR, FA_BC, FA_THREADS, K_STAGES);

    // Phase 1: full matrix, single run each
    printf("=== Phase 1: Matrix sl × wnd (single run each) ===\n");
    int sl_list[] = {64, 300, 8192};
    int wnd_list[] = {0, 96, 1024};
    int bh = 1;
    double timeout = 5.0;
    int n_hang_p1 = 0;
    for (int sl : sl_list) {
        for (int wnd : wnd_list) {
            int causal = (wnd > 0) ? 1 : 0;
            int r = run_one(bh, sl, causal, wnd, timeout);
            printf("  bh=%d sl=%-5d ca=%d wnd=%-5d : %s\n",
                   bh, sl, causal, wnd, (r==0) ? "OK" : "HANG");
            if (r == -1) n_hang_p1++;
        }
    }
    printf("\nPhase 1 summary: %d HANG / 9 configs\n\n", n_hang_p1);

    // Phase 2: 100× rerun of sl=300 wnd=96 (target config, hang may be rare)
    printf("=== Phase 2: sl=300 wnd=96 × 100 runs (hang may be rare) ===\n");
    int n_hang_p2 = 0;
    int n_ok_p2 = 0;
    for (int i = 0; i < 100; i++) {
        int r = run_one(bh, 300, 1, 96, timeout);
        if (r == -1) {
            n_hang_p2++;
            printf("  run %3d: HANG\n", i+1);
            // Reset CUDA context after hang
            cudaDeviceReset();
        } else {
            n_ok_p2++;
        }
        if ((i+1) % 20 == 0) {
            printf("  [%d/100] OK=%d HANG=%d\n", i+1, n_ok_p2, n_hang_p2);
        }
    }
    printf("\nPhase 2 summary: %d OK, %d HANG out of 100 runs on sl=300 wnd=96\n",
           n_ok_p2, n_hang_p2);

    int total_hang = n_hang_p1 + n_hang_p2;
    printf("\n=== TOTAL: %d HANG across all tests ===\n", total_hang);
    return total_hang > 0 ? 1 : 0;
}
