// =====================================================================
//  bench_30run_dv.cu — production same-thermal A/B P1 vs B2.1 для dV.
//
//  Protocol per L-patch lesson (B2_HANDOFF §9):
//      - 40 iter warmup alternating (B2.1, P1, B2.1, P1, ...)
//      - 30 main measurement rounds, each = (5-iter B2.1 + 5-iter P1)
//      - median-of-5 per kernel per round
//      - alternating order in warmup AND main → same thermal block
//
//  Output: per-kernel mean, sigma, min, max, median, 95% CI; A/B ratio,
//          t-statistic, condition log (driver, sm_clock, temp, date).
//
//  Form: bh=128 sl=8192 hd=128 causal=0 window=0 (production training shape).
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "fa_bwd_common.cuh"

#define CK_(c) do { cudaError_t e = (c); if (e != cudaSuccess) {                \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                     \
            cudaGetErrorString(e)); std::exit(1); }} while (0)

namespace fa_bwd_dv_mma    { void launch(const uint8_t*, const uint8_t*, const __half*,
                                          const float*, float*, int, int, int, int, int,
                                          float, cudaStream_t); }
namespace fa_bwd_dv_mma_p1 { void launch(const uint8_t*, const uint8_t*, const __half*,
                                          const float*, float*, int, int, int, int, int,
                                          float, cudaStream_t); }

struct Stats {
    double mean, sd, min_v, max_v, median;
    double ci95_half;   // half-width of 95% CI (mean ± ci95_half)
    int n;
};

static Stats compute_stats(std::vector<double> v) {
    Stats s{};
    s.n = (int)v.size();
    if (s.n == 0) return s;
    std::sort(v.begin(), v.end());
    s.min_v = v.front();
    s.max_v = v.back();
    s.median = (s.n & 1) ? v[s.n / 2]
                         : 0.5 * (v[s.n/2 - 1] + v[s.n/2]);
    double sum = 0;
    for (double x : v) sum += x;
    s.mean = sum / s.n;
    double var = 0;
    for (double x : v) var += (x - s.mean) * (x - s.mean);
    s.sd = std::sqrt(var / std::max(1, s.n - 1));
    // 95% CI half-width ≈ 1.96 * sd / sqrt(n) for large n; t-distrib for small.
    // Use 2.045 (df=29) for n=30, 2.262 for n=10, 2.776 for n=5 — close enough.
    double t95 = (s.n >= 30) ? 2.045 :
                 (s.n >= 10) ? 2.262 : 2.776;
    s.ci95_half = t95 * s.sd / std::sqrt((double)s.n);
    return s;
}

static double t_statistic(const Stats &a, const Stats &b) {
    // Welch's t (unequal variance):
    // t = (mean_a - mean_b) / sqrt(sd_a^2/n_a + sd_b^2/n_b)
    double se = std::sqrt(a.sd*a.sd/a.n + b.sd*b.sd/b.n);
    return (a.mean - b.mean) / (se + 1e-30);
}

static void run_one(int which, const uint8_t *dQ, const uint8_t *dK_g,
                    const __half *ddO, const float *dL, float *ddV,
                    int bh, int sl, int hd, int causal, int window,
                    float scale)
{
    if (which == 0)
        fa_bwd_dv_mma::launch(dQ, dK_g, ddO, dL, ddV,
                              bh, sl, hd, causal, window, scale, 0);
    else
        fa_bwd_dv_mma_p1::launch(dQ, dK_g, ddO, dL, ddV,
                                 bh, sl, hd, causal, window, scale, 0);
}

// Returns ms of median-of-5 iterations for kernel `which`.
static double median_of_5(int which, const uint8_t *dQ, const uint8_t *dK_g,
                          const __half *ddO, const float *dL, float *ddV,
                          int bh, int sl, int hd, int causal, int window,
                          float scale)
{
    cudaEvent_t t0, t1;
    CK_(cudaEventCreate(&t0));
    CK_(cudaEventCreate(&t1));
    std::vector<double> times;
    times.reserve(5);
    for (int i = 0; i < 5; ++i) {
        CK_(cudaMemset(ddV, 0, (size_t)bh * sl * hd * sizeof(float)));
        CK_(cudaEventRecord(t0));
        run_one(which, dQ, dK_g, ddO, dL, ddV, bh, sl, hd, causal, window, scale);
        CK_(cudaEventRecord(t1));
        CK_(cudaEventSynchronize(t1));
        float ms = 0.0f;
        CK_(cudaEventElapsedTime(&ms, t0, t1));
        times.push_back(ms);
    }
    CK_(cudaEventDestroy(t0));
    CK_(cudaEventDestroy(t1));
    std::sort(times.begin(), times.end());
    return times[2];   // median
}

static void log_nvidia_smi(FILE *f, const char *label) {
    fprintf(f, "[%s] nvidia-smi snapshot:\n", label);
    fflush(f);
    int rc = std::system(
        "nvidia-smi --query-gpu=name,driver_version,clocks.current.sm,"
        "clocks.current.memory,temperature.gpu,power.draw,utilization.gpu "
        "--format=csv >> /tmp/_bench_smi.tmp");
    (void)rc;
    FILE *tmp = std::fopen("/tmp/_bench_smi.tmp", "r");
    if (tmp) {
        char buf[1024];
        while (std::fgets(buf, sizeof(buf), tmp)) {
            std::fputs(buf, f);
        }
        std::fclose(tmp);
        std::remove("/tmp/_bench_smi.tmp");
    }
    fflush(f);
}

int main(int argc, char **argv)
{
    int bh        = (argc >= 2) ? std::atoi(argv[1]) : 128;
    int sl        = (argc >= 3) ? std::atoi(argv[2]) : 8192;
    int causal    = (argc >= 4) ? std::atoi(argv[3]) : 0;
    int window    = (argc >= 5) ? std::atoi(argv[4]) : 0;
    int warmup    = (argc >= 6) ? std::atoi(argv[5]) : 40;
    int rounds    = (argc >= 7) ? std::atoi(argv[6]) : 30;
    int hd        = 128;
    const char *log_path = (argc >= 8) ? argv[7]
                                       : "/data/lib/podman-data/projects/goml/runs/30run_p1_vs_b21.log";

    const size_t sz  = (size_t)bh * sl * hd;
    const size_t lsz = (size_t)bh * sl;

    // Open log file (append; append-mode to keep historical runs).
    FILE *flog = std::fopen(log_path, "w");
    if (!flog) {
        fprintf(stderr, "cannot open log %s\n", log_path);
        return 1;
    }

    std::time_t now = std::time(nullptr);
    char ts[64];
    std::strftime(ts, sizeof(ts), "%F %T %Z", std::localtime(&now));

    fprintf(flog, "==================================================\n");
    fprintf(flog, "30-run same-thermal A/B: P1 vs B2.1, dV-MMA\n");
    fprintf(flog, "==================================================\n");
    fprintf(flog, "Date     : %s\n", ts);
    fprintf(flog, "Form     : bh=%d sl=%d hd=%d causal=%d window=%d\n",
            bh, sl, hd, causal, window);
    fprintf(flog, "Warmup   : %d iter (alternating B2.1↔P1)\n", warmup);
    fprintf(flog, "Rounds   : %d (each = median-of-5 per kernel, alternating)\n", rounds);
    fprintf(flog, "Total    : %d kernel launches per kernel\n", warmup/2 + rounds*5);
    log_nvidia_smi(flog, "before-bench");
    fprintf(flog, "\n");

    printf("30-run A/B: bh=%d sl=%d hd=%d causal=%d window=%d warmup=%d rounds=%d\n",
           bh, sl, hd, causal, window, warmup, rounds);
    printf("Log: %s\n", log_path);

    // Generate inputs (FP8/FP16/FP32) once — reused across all iters.
    std::vector<uint8_t> Q8(sz), K8(sz);
    std::vector<__half>  dO16(sz);
    std::vector<float>   L32(lsz);
    {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.6f);
        for (size_t i = 0; i < sz; ++i) {
            Q8 [i] = float_to_e4m3_host(dist(rng));
            K8 [i] = float_to_e4m3_host(dist(rng));
            dO16[i] = __float2half_rn(dist(rng));
        }
        for (size_t i = 0; i < lsz; ++i) L32[i] = dist(rng);
    }

    uint8_t *dQ = nullptr, *dK_g = nullptr;
    __half  *ddO = nullptr;
    float   *dL = nullptr, *ddV = nullptr;
    CK_(cudaMalloc(&dQ,   sz  * sizeof(uint8_t)));
    CK_(cudaMalloc(&dK_g, sz  * sizeof(uint8_t)));
    CK_(cudaMalloc(&ddO,  sz  * sizeof(__half)));
    CK_(cudaMalloc(&dL,   lsz * sizeof(float)));
    CK_(cudaMalloc(&ddV,  sz  * sizeof(float)));
    CK_(cudaMemcpy(dQ,   Q8.data(),   sz  * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CK_(cudaMemcpy(dK_g, K8.data(),   sz  * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CK_(cudaMemcpy(ddO,  dO16.data(), sz  * sizeof(__half),  cudaMemcpyHostToDevice));
    CK_(cudaMemcpy(dL,   L32.data(),  lsz * sizeof(float),   cudaMemcpyHostToDevice));

    float scale = 1.0f / std::sqrt((float)hd);

    // Warmup — alternating, same thermal block as measurement.
    fprintf(flog, "Warmup (alternating B2.1↔P1, %d iter)...\n", warmup);
    for (int i = 0; i < warmup; ++i) {
        int which = i & 1;   // 0=B2.1, 1=P1
        run_one(which, dQ, dK_g, ddO, dL, ddV, bh, sl, hd, causal, window, scale);
    }
    CK_(cudaDeviceSynchronize());

    // Main measurement — alternating per-round
    std::vector<double> times_b21, times_p1;
    times_b21.reserve(rounds);
    times_p1.reserve(rounds);

    fprintf(flog, "Main rounds:\n");
    fprintf(flog, "%-6s  %12s  %12s  %12s\n",
            "round", "B2.1 ms", "P1 ms", "ratio (B2.1/P1)");
    for (int r = 0; r < rounds; ++r) {
        // Order within round: alternate parity to break sympathy
        if (r & 1) {
            double p1 = median_of_5(1, dQ, dK_g, ddO, dL, ddV, bh, sl, hd, causal, window, scale);
            double b21= median_of_5(0, dQ, dK_g, ddO, dL, ddV, bh, sl, hd, causal, window, scale);
            times_p1.push_back(p1);
            times_b21.push_back(b21);
            fprintf(flog, "%-6d  %12.4f  %12.4f  %12.4f\n", r, b21, p1, b21/p1);
        } else {
            double b21= median_of_5(0, dQ, dK_g, ddO, dL, ddV, bh, sl, hd, causal, window, scale);
            double p1 = median_of_5(1, dQ, dK_g, ddO, dL, ddV, bh, sl, hd, causal, window, scale);
            times_b21.push_back(b21);
            times_p1.push_back(p1);
            fprintf(flog, "%-6d  %12.4f  %12.4f  %12.4f\n", r, b21, p1, b21/p1);
        }
        fflush(flog);
        if (r % 5 == 4) {
            printf("  round %d/%d done\n", r+1, rounds);
        }
    }

    log_nvidia_smi(flog, "after-bench");

    Stats s_b21 = compute_stats(times_b21);
    Stats s_p1  = compute_stats(times_p1);

    auto tflops_from_ms = [&](double ms){
        // dV FLOPs ≈ 2 * QK + 2 * PdO = 4 bh sl² hd (non-causal).
        // Causal halves QK.
        double cf = causal ? 0.5 : 1.0;
        double flops = 2.0 * bh * (double)sl * sl * hd * cf
                     + 2.0 * bh * (double)sl * sl * hd;
        return flops / (ms * 1e-3) / 1e12;
    };

    fprintf(flog, "\n==================================================\n");
    fprintf(flog, "Summary (over %d rounds, median-of-5 each):\n", rounds);
    fprintf(flog, "==================================================\n");
    fprintf(flog, "Kernel    mean (ms)   sd     min      max      median   95%%CI±\n");
    fprintf(flog, "B2.1      %8.4f  %6.4f  %7.4f  %7.4f  %7.4f  %6.4f\n",
            s_b21.mean, s_b21.sd, s_b21.min_v, s_b21.max_v, s_b21.median, s_b21.ci95_half);
    fprintf(flog, "P1        %8.4f  %6.4f  %7.4f  %7.4f  %7.4f  %6.4f\n",
            s_p1.mean,  s_p1.sd,  s_p1.min_v,  s_p1.max_v,  s_p1.median,  s_p1.ci95_half);
    fprintf(flog, "\n");
    fprintf(flog, "TFLOPS    B2.1 = %.2f   P1 = %.2f\n",
            tflops_from_ms(s_b21.mean), tflops_from_ms(s_p1.mean));
    fprintf(flog, "Speedup   P1 / B2.1 = %.4f×\n", s_b21.mean / s_p1.mean);
    fprintf(flog, "Welch t   = %.2f (df≈%d)\n", t_statistic(s_b21, s_p1), 2*rounds - 2);

    // Echo to stdout too.
    printf("\n==== RESULT ====\n");
    printf("B2.1: mean %.3f ms ± %.3f (95%% CI), median %.3f, TFLOPS %.1f\n",
           s_b21.mean, s_b21.ci95_half, s_b21.median, tflops_from_ms(s_b21.mean));
    printf("P1  : mean %.3f ms ± %.3f (95%% CI), median %.3f, TFLOPS %.1f\n",
           s_p1.mean, s_p1.ci95_half, s_p1.median, tflops_from_ms(s_p1.mean));
    printf("Speedup P1/B2.1 = %.3f×, Welch t = %.1f\n",
           s_b21.mean / s_p1.mean, t_statistic(s_b21, s_p1));
    printf("Log saved: %s\n", log_path);

    std::fclose(flog);
    CK_(cudaFree(dQ));   CK_(cudaFree(dK_g));
    CK_(cudaFree(ddO));  CK_(cudaFree(dL));
    CK_(cudaFree(ddV));
    return 0;
}
