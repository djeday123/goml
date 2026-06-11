// mbar_repro_v3k: трассировка фаз mbarrier. v3h база + инструментация.
//
// Цель M7: ПРЯМОЕ наблюдение состояния mbarrier-слова в shared, не гипотезы.
// Каждое событие пишет 32-байтную запись DbgEvent в pinned-mapped GMEM
// (cudaHostAllocMapped — host видит данные ДАЖЕ при зависшем kernel,
// потому что GPU пишет напрямую через PCIe SYS_GLOBAL store).
//
// События:
//   PRE_ARRIVE  = 1 — producer выполнил mbar_arrive в pre-load (warp 0, lane 0)
//   BEFORE_WAIT = 2 — consumer ВХОДИТ в mbar_wait (warp 1, lane 0)
//   AFTER_WAIT  = 3 — consumer ВЫХОДИТ из mbar_wait (warp 1, lane 0)
//   MID_ARRIVE  = 4 — producer выполнил mid-iter mbar_arrive (warp 0, lane 0)
//   SPIN_ALIVE  = 5 — heartbeat в spin loop (каждые 2^18 итераций)

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <chrono>
#include <unistd.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);}}while(0)

#define FA_BR 96
#define FA_BC 64
#define FA_THREADS 128
#define FA_PRODUCERS 1
#define FA_CONSUMERS 3
#define K_STAGES 2
#define FA_STRIDE 128
#define SMV_T_STRIDE 68

#define MAX_DBG_EVENTS 96

#define EV_PRE_ARRIVE  1
#define EV_BEFORE_WAIT 2
#define EV_AFTER_WAIT  3
#define EV_MID_ARRIVE  4
#define EV_SPIN_ALIVE  5

struct DbgEvent {
    uint64_t marker;        // [63:48]=0xDEAD, [47:32]=iter, [31:16]=code, [15:0]=warp
    uint64_t slot_expected; // [63:32]=slot, [31:0]=expected_phase
    uint64_t raw_word_0;    // ld.shared.b64 от bar_ready[0]
    uint64_t raw_word_1;    // ld.shared.b64 от bar_ready[1]
};

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
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(sa), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint64_t *bar) {
    uint32_t sa = __cvta_generic_to_shared(bar);
    uint64_t token;
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 %0, [%1];"
                 : "=l"(token) : "r"(sa) : "memory");
    (void)token;
}
__device__ __forceinline__ bool mbar_test_wait(uint64_t *bar, uint32_t phase) {
    uint32_t sa = __cvta_generic_to_shared(bar);
    uint32_t result;
    asm volatile(
        "{ .reg .pred P1;\n"
        "  mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 P1, [%1], %2;\n"
        "  selp.b32 %0, 1, 0, P1;\n"
        "}"
        : "=r"(result) : "r"(sa), "r"(phase)
    );
    return result != 0;
}

__device__ __forceinline__ uint64_t raw_read_bar(uint64_t *bar) {
    uint32_t sa = __cvta_generic_to_shared(bar);
    uint64_t w;
    asm volatile("ld.shared.b64 %0, [%1];" : "=l"(w) : "r"(sa) : "memory");
    return w;
}

__device__ __forceinline__ void dbg_log(
    DbgEvent *blk, int *p_idx, uint32_t code, uint32_t iter,
    uint32_t warp, uint32_t slot, uint32_t expected, uint64_t *bars)
{
    int idx = atomicAdd(p_idx, 1);
    if (idx >= MAX_DBG_EVENTS) return;
    DbgEvent e;
    e.marker = ((uint64_t)0xDEADull << 48)
             | ((uint64_t)(iter & 0xFFFF) << 32)
             | ((uint64_t)(code & 0xFFFF) << 16)
             | (warp & 0xFFFF);
    e.slot_expected = ((uint64_t)slot << 32) | expected;
    e.raw_word_0 = raw_read_bar(&bars[0]);
    e.raw_word_1 = raw_read_bar(&bars[1]);
    blk[idx] = e;
    __threadfence_system();  // flush pinned-mapped в host через PCIe
}

__global__ void __launch_bounds__(FA_THREADS, 2)
    mbar_repro_kernel(int *out_iters, DbgEvent *dbg_buf_all,
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
    uint8_t *smK[K_STAGES] = {
        raw,
        raw + FA_BC * FA_STRIDE,
    };
    uint8_t *smV = smK[1] + FA_BC * FA_STRIDE;
    uint64_t *bar_ready = (uint64_t *)((uintptr_t)(smV + FA_BC * FA_STRIDE + 7) & ~7ULL);

    __shared__ int s_dbg_idx;
    if (threadIdx.x == 0) s_dbg_idx = 0;

    DbgEvent *blk_dbg = dbg_buf_all + blockIdx.x * MAX_DBG_EVENTS;

    if (threadIdx.x == 0) {
        mbar_init(&bar_ready[0], 32);
        mbar_init(&bar_ready[1], 32);
    }
    __syncthreads();                              // [SYNC P1]

    int nkv = (seq_len + FA_BC - 1) / FA_BC;
    int kv_max_blocks = causal ? ((qs + FA_BR - 1) / FA_BC + 1) : nkv;
    if (kv_max_blocks > nkv) kv_max_blocks = nkv;
    int kv_min_blocks = 0;
    if (window > 0 && qs + 1 > window) {
        kv_min_blocks = (qs - window + 1) / FA_BC;
    }

    // PRE-LOAD K+V
    if (wid == 0) {
        load_tile_fp8_warp(smK[kv_min_blocks & 1], Kh, kv_min_blocks * FA_BC, FA_BC,
                           seq_len, head_dim, lane);
        load_tile_fp8_warp(smV, Vh, kv_min_blocks * FA_BC, FA_BC,
                           seq_len, head_dim, lane);
        cpa_commit();
        cpa_wait<0>();
        mbar_arrive(&bar_ready[kv_min_blocks & 1]);
        if (lane == 0) {
            dbg_log(blk_dbg, &s_dbg_idx, EV_PRE_ARRIVE, 0u, 0u,
                    (uint32_t)(kv_min_blocks & 1), 0, bar_ready);
        }
    }

    uint32_t expected_phase[K_STAGES] = {1, 1};

    int iters_done = 0;
    for (int kv = kv_min_blocks; kv < kv_max_blocks; kv++)
    {
        int buf = kv & 1;
        int nxt_buf = (kv + 1) & 1;

        if (!is_producer) {
            if (wid == 1 && lane == 0) {
                dbg_log(blk_dbg, &s_dbg_idx, EV_BEFORE_WAIT, (uint32_t)kv, (uint32_t)wid,
                        (uint32_t)buf, expected_phase[buf], bar_ready);
            }
            uint32_t spin_cnt = 0;
            while (!mbar_test_wait(&bar_ready[buf], expected_phase[buf])) {
                spin_cnt++;
                if (wid == 1 && lane == 0 && ((spin_cnt & 0x3FFFFu) == 0)) {
                    dbg_log(blk_dbg, &s_dbg_idx, EV_SPIN_ALIVE, (uint32_t)kv, (uint32_t)wid,
                            (uint32_t)buf, expected_phase[buf], bar_ready);
                }
            }
            if (wid == 1 && lane == 0) {
                dbg_log(blk_dbg, &s_dbg_idx, EV_AFTER_WAIT, (uint32_t)kv, (uint32_t)wid,
                        (uint32_t)buf, expected_phase[buf], bar_ready);
            }
            expected_phase[buf] ^= 1u;
        }
        __syncthreads();                          // [SYNC L1]

        for (int i = 0; i < 5; i++) __nanosleep(50);
        __syncthreads();                          // [SYNC L2]

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
            if (lane == 0) {
                dbg_log(blk_dbg, &s_dbg_idx, EV_MID_ARRIVE, (uint32_t)kv, 0u,
                        (uint32_t)nxt_buf, 0, bar_ready);
            }
        } else {
            cpa_commit();
        }

        for (int i = 0; i < 20; i++) __nanosleep(50);

        if (!is_producer) {
            asm volatile("bar.sync 1, 96;");      // [SYNC L3]
        }

        for (int i = 0; i < 5; i++) __nanosleep(50);
        for (int i = 0; i < 20; i++) __nanosleep(50);

        iters_done++;
    }

    if (threadIdx.x == 0) {
        out_iters[blockIdx.x] = iters_done;
    }
}

void flush_stdout() { fflush(stdout); fflush(stderr); }

const char *ev_name(uint32_t code) {
    switch (code) {
        case EV_PRE_ARRIVE:  return "PRE_ARRIVE ";
        case EV_BEFORE_WAIT: return "BEFORE_WAIT";
        case EV_AFTER_WAIT:  return "AFTER_WAIT ";
        case EV_MID_ARRIVE:  return "MID_ARRIVE ";
        case EV_SPIN_ALIVE:  return "SPIN_ALIVE ";
        default:             return "??         ";
    }
}

void dump_dbg(DbgEvent *dbg_h, int grid, int max_blocks_to_dump) {
    int blocks = (grid < max_blocks_to_dump) ? grid : max_blocks_to_dump;
    for (int b = 0; b < blocks; b++) {
        DbgEvent *blk = dbg_h + b * MAX_DBG_EVENTS;
        printf("---- block %d ----\n", b);
        printf("  idx | event       | iter | warp | slot | expected | raw_word_0           | raw_word_1\n");
        int printed = 0;
        for (int i = 0; i < MAX_DBG_EVENTS; i++) {
            uint64_t m = blk[i].marker;
            if ((m >> 48) != 0xDEAD) break;
            uint32_t iter = (uint32_t)((m >> 32) & 0xFFFF);
            uint32_t code = (uint32_t)((m >> 16) & 0xFFFF);
            uint32_t warp = (uint32_t)(m & 0xFFFF);
            uint64_t se = blk[i].slot_expected;
            uint32_t slot = (uint32_t)(se >> 32);
            uint32_t expected = (uint32_t)(se & 0xFFFFFFFF);
            printf("  %3d | %s | %4u | %4u | %4u | %8u | 0x%016lx | 0x%016lx\n",
                   i, ev_name(code), iter, warp, slot, expected,
                   (unsigned long)blk[i].raw_word_0,
                   (unsigned long)blk[i].raw_word_1);
            printed++;
        }
        if (printed == 0) printf("  (no events captured for this block)\n");
    }
}

int run_one_with_dbg(int bh, int sl, int causal, int window, double timeout_sec) {
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

    // PINNED-MAPPED debug buffer
    size_t dbg_bytes = (size_t)grid * MAX_DBG_EVENTS * sizeof(DbgEvent);
    DbgEvent *dbg_h = nullptr;
    DbgEvent *dbg_d = nullptr;
    CK(cudaHostAlloc(&dbg_h, dbg_bytes, cudaHostAllocMapped));
    memset(dbg_h, 0, dbg_bytes);
    CK(cudaHostGetDevicePointer((void**)&dbg_d, dbg_h, 0));

    int smem = 2 * FA_BC * FA_STRIDE + FA_BC * FA_STRIDE + 32;
    CK(cudaFuncSetAttribute(mbar_repro_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    auto t0 = std::chrono::steady_clock::now();
    mbar_repro_kernel<<<grid, FA_THREADS, smem>>>(out_d, dbg_d, K_d, V_d, sl, hd, causal, window);

    bool hung = false;
    while (true) {
        cudaError_t e = cudaStreamQuery(0);
        if (e == cudaSuccess) break;
        if (e != cudaErrorNotReady) {
            fprintf(stderr, "  CUDA error: %s\n", cudaGetErrorString(e));
            hung = true; break;
        }
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        if (elapsed > timeout_sec) { hung = true; break; }
        usleep(10000);
    }

    int rc;
    if (hung) {
        printf("  ## HANG detected — dumping debug buffer (pinned-mapped, alive) ##\n");
        flush_stdout();
        dump_dbg(dbg_h, grid, 1);
        rc = -1;
    } else {
        CK(cudaMemcpy(out_h, out_d, grid * sizeof(int), cudaMemcpyDeviceToHost));
        printf("  ## OK — dumping debug buffer для блока 0 ##\n");
        flush_stdout();
        dump_dbg(dbg_h, grid, 1);
        rc = 0;
    }

    if (!hung) {
        cudaFree(K_d); cudaFree(V_d);
        cudaFree(out_d); free(out_h);
        cudaFreeHost(dbg_h);
    }
    return rc;
}

int main(int argc, char **argv)
{
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("GPU: %s (CC %d.%d)\n", p.name, p.major, p.minor);
    printf("Br=%d Bc=%d FA_THREADS=%d (1P+3C) K_STAGES=%d\n", FA_BR, FA_BC, FA_THREADS, K_STAGES);
    printf("v3k: трассировка фаз mbarrier через pinned-mapped GMEM\n\n");
    setvbuf(stdout, NULL, _IONBF, 0);

    int sl = (argc > 1) ? atoi(argv[1]) : 300;
    int causal = (argc > 2) ? atoi(argv[2]) : 0;
    int window = (argc > 3) ? atoi(argv[3]) : 0;
    printf("=== Run: bh=1 sl=%d ca=%d wnd=%d timeout=5s ===\n", sl, causal, window);
    int r = run_one_with_dbg(1, sl, causal, window, 5.0);
    printf("\n=== Result: %s ===\n", (r == 0) ? "OK" : "HANG");
    return r == 0 ? 0 : 1;
}
