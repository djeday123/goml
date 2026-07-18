// 047 §2: bench_headwise_e2e — 5 плечей (ARM1/2S/3S/2P/3P)
// Каноническая форма bh=128 sl=8192 hd=128.
// Wall-вердикт (NCu запрещён в этом ТЗ).

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "fa_bwd_common.cuh"

#define CKR(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); }} while (0)

namespace fa_bwd_dk {
void launch_d_precompute(const __half *, const __half *, float *, int, int, int, cudaStream_t);
__global__ void kernel_d_precompute(const __half *, const __half *, float *, int, int, int);
}
namespace fa_bwd_merged_v1 {
void launch_merged(const uint8_t *, const uint8_t *, const uint8_t *, const __half *, const float *, const float *,
                    uint8_t *, uint8_t *, float *, int, int, int, int, int, float, cudaStream_t);
__global__ void kernel_merged_v1(const uint8_t *, const uint8_t *, const uint8_t *, const __half *, const float *, const float *,
                                 uint8_t *, uint8_t *, float *, int, int, int, int, int, float);
}
namespace fa_bwd_dk_new {
void launch_dk_new(const uint8_t *, const uint8_t *, float *, int, int, int, int, int, float, cudaStream_t);
__global__ void kernel_dk_new(const uint8_t *, const uint8_t *, float *, int, int, int, int, int, float);
}
namespace fa_bwd_dq_new {
void launch_dq_new(const uint8_t *, const uint8_t *, float *, int, int, int, int, int, float, cudaStream_t);
__global__ void kernel_dq_new(const uint8_t *, const uint8_t *, float *, int, int, int, int, int, float);
}

struct Buffers {
    uint8_t *dQ, *dK, *dV8;
    __half *dOO, *dOG;
    float *dL, *dD;
    uint8_t *dS_nat, *dS_T;
    float *ddV, *ddK, *ddQ;
    size_t sz, lsz, dsz;
    int bh, sl, hd, stride_ds;
};

void alloc_buffers(Buffers &b, int bh, int sl, int hd) {
    b.bh = bh; b.sl = sl; b.hd = hd;
    b.stride_ds = (sl + 15) & ~15;
    b.sz  = (size_t)bh * sl * hd;
    b.lsz = (size_t)bh * sl;
    b.dsz = (size_t)bh * sl * b.stride_ds;

    CKR(cudaMalloc(&b.dQ, b.sz));CKR(cudaMalloc(&b.dK, b.sz));CKR(cudaMalloc(&b.dV8, b.sz));
    CKR(cudaMalloc(&b.dOO, b.sz*sizeof(__half)));CKR(cudaMalloc(&b.dOG, b.sz*sizeof(__half)));
    CKR(cudaMalloc(&b.dL, b.lsz*sizeof(float)));CKR(cudaMalloc(&b.dD, b.lsz*sizeof(float)));
    CKR(cudaMalloc(&b.ddV, b.sz*sizeof(float)));CKR(cudaMalloc(&b.ddK, b.sz*sizeof(float)));CKR(cudaMalloc(&b.ddQ, b.sz*sizeof(float)));
    CKR(cudaMalloc(&b.dS_nat, b.dsz));CKR(cudaMalloc(&b.dS_T, b.dsz));

    std::vector<uint8_t> Q8(b.sz), K8(b.sz), V8(b.sz);
    std::vector<__half>  O16(b.sz), dO16(b.sz);
    std::vector<float>   L32(b.lsz);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.6f);
    for (size_t i=0;i<b.sz;++i){Q8[i]=float_to_e4m3_host(dist(rng));K8[i]=float_to_e4m3_host(dist(rng));V8[i]=float_to_e4m3_host(dist(rng));O16[i]=__float2half_rn(dist(rng));dO16[i]=__float2half_rn(dist(rng));}
    for (size_t i=0;i<b.lsz;++i) L32[i] = dist(rng);
    CKR(cudaMemcpy(b.dQ, Q8.data(), b.sz, cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(b.dK, K8.data(), b.sz, cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(b.dV8, V8.data(), b.sz, cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(b.dOO, O16.data(), b.sz*sizeof(__half), cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(b.dOG, dO16.data(), b.sz*sizeof(__half), cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(b.dL, L32.data(), b.lsz*sizeof(float), cudaMemcpyHostToDevice));
}

void reset_outputs(Buffers &b) {
    CKR(cudaMemset(b.ddV, 0, b.sz*sizeof(float)));
    CKR(cudaMemset(b.ddK, 0, b.sz*sizeof(float)));
    CKR(cudaMemset(b.ddQ, 0, b.sz*sizeof(float)));
}

// --- Per-arm implementations ---

// ARM1 monolith production order
float arm1_monolith(Buffers &b) {
    float scale = 1.0f/sqrtf((float)b.hd);
    reset_outputs(b);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, 0);
    fa_bwd_dk::launch_d_precompute(b.dOO, b.dOG, b.dD, b.bh, b.sl, b.hd, 0);
    fa_bwd_merged_v1::launch_merged(b.dQ, b.dK, b.dV8, b.dOG, b.dL, b.dD, b.dS_nat, b.dS_T, b.ddV, b.bh, b.sl, b.hd, 0, 0, scale, 0);
    fa_bwd_dk_new::launch_dk_new(b.dQ, b.dS_nat, b.ddK, b.bh, b.sl, b.hd, 0, 0, scale, 0);
    fa_bwd_dq_new::launch_dq_new(b.dK, b.dS_nat, b.ddQ, b.bh, b.sl, b.hd, 0, 0, scale, 0);
    cudaEventRecord(e1, 0);
    cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms;
}

// ARM2S dense (no persist): merged(h) -> [dk(h) || dq(h) в двух потоках] -> синк-событие -> h+1
float arm2s_dense_nopersist(Buffers &b, cudaStream_t sM, cudaStream_t sK, cudaStream_t sQ, cudaEvent_t eM, cudaEvent_t eK, cudaEvent_t eQ) {
    float scale = 1.0f/sqrtf((float)b.hd);
    reset_outputs(b);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, 0);
    // D-precompute на всех головах разом (небольшая работа) для чистоты, потом per-head chain
    fa_bwd_dk::launch_d_precompute(b.dOO, b.dOG, b.dD, b.bh, b.sl, b.hd, sM);
    cudaEventRecord(eM, sM);
    cudaStreamWaitEvent(sK, eM, 0);
    cudaStreamWaitEvent(sQ, eM, 0);

    size_t hd_stride    = (size_t)b.sl * b.hd;
    size_t l_stride     = (size_t)b.sl;
    size_t ds_stride    = (size_t)b.sl * b.stride_ds;
    for (int h = 0; h < b.bh; ++h) {
        // merged(h) на sM
        fa_bwd_merged_v1::launch_merged(
            b.dQ + h*hd_stride, b.dK + h*hd_stride, b.dV8 + h*hd_stride,
            b.dOG + h*hd_stride, b.dL + h*l_stride, b.dD + h*l_stride,
            b.dS_nat + h*ds_stride, b.dS_T + h*ds_stride, b.ddV + h*hd_stride,
            1, b.sl, b.hd, 0, 0, scale, sM);
        cudaEventRecord(eM, sM);
        cudaStreamWaitEvent(sK, eM, 0);
        cudaStreamWaitEvent(sQ, eM, 0);
        // dk(h) на sK, dq(h) на sQ
        fa_bwd_dk_new::launch_dk_new(b.dQ + h*hd_stride, b.dS_nat + h*ds_stride, b.ddK + h*hd_stride,
                                      1, b.sl, b.hd, 0, 0, scale, sK);
        fa_bwd_dq_new::launch_dq_new(b.dK + h*hd_stride, b.dS_nat + h*ds_stride, b.ddQ + h*hd_stride,
                                      1, b.sl, b.hd, 0, 0, scale, sQ);
        cudaEventRecord(eK, sK);
        cudaEventRecord(eQ, sQ);
        // merged(h+1) должен ждать consumer'ов dk/dq(h) чтобы не перепиcать dS_nat
        cudaStreamWaitEvent(sM, eK, 0);
        cudaStreamWaitEvent(sM, eQ, 0);
    }
    cudaStreamSynchronize(sM);
    cudaStreamSynchronize(sK);
    cudaStreamSynchronize(sQ);
    cudaEventRecord(e1, 0);
    cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms;
}

// ARM3S dense + persist
float arm3s_dense_persist(Buffers &b, cudaStream_t sM, cudaStream_t sK, cudaStream_t sQ, cudaEvent_t eM, cudaEvent_t eK, cudaEvent_t eQ) {
    float scale = 1.0f/sqrtf((float)b.hd);
    reset_outputs(b);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, 0);

    fa_bwd_dk::launch_d_precompute(b.dOO, b.dOG, b.dD, b.bh, b.sl, b.hd, sM);
    cudaEventRecord(eM, sM);
    cudaStreamWaitEvent(sK, eM, 0);
    cudaStreamWaitEvent(sQ, eM, 0);

    size_t hd_stride = (size_t)b.sl * b.hd;
    size_t l_stride  = (size_t)b.sl;
    size_t ds_stride = (size_t)b.sl * b.stride_ds;
    size_t headBytes = ds_stride;  // 8192 * 8192 = 64 MiB

    for (int h = 0; h < b.bh; ++h) {
        // Set access policy window на все три потока для dS(h)
        cudaStreamAttrValue attr = {};
        attr.accessPolicyWindow.base_ptr    = (void*)(b.dS_nat + h*ds_stride);
        attr.accessPolicyWindow.num_bytes   = headBytes;
        attr.accessPolicyWindow.hitRatio    = 1.0f;
        attr.accessPolicyWindow.hitProp     = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp    = cudaAccessPropertyStreaming;
        cudaStreamSetAttribute(sM, cudaStreamAttributeAccessPolicyWindow, &attr);
        cudaStreamSetAttribute(sK, cudaStreamAttributeAccessPolicyWindow, &attr);
        cudaStreamSetAttribute(sQ, cudaStreamAttributeAccessPolicyWindow, &attr);

        // merged(h) на sM
        fa_bwd_merged_v1::launch_merged(
            b.dQ + h*hd_stride, b.dK + h*hd_stride, b.dV8 + h*hd_stride,
            b.dOG + h*hd_stride, b.dL + h*l_stride, b.dD + h*l_stride,
            b.dS_nat + h*ds_stride, b.dS_T + h*ds_stride, b.ddV + h*hd_stride,
            1, b.sl, b.hd, 0, 0, scale, sM);
        cudaEventRecord(eM, sM);
        cudaStreamWaitEvent(sK, eM, 0);
        cudaStreamWaitEvent(sQ, eM, 0);

        fa_bwd_dk_new::launch_dk_new(b.dQ + h*hd_stride, b.dS_nat + h*ds_stride, b.ddK + h*hd_stride,
                                      1, b.sl, b.hd, 0, 0, scale, sK);
        fa_bwd_dq_new::launch_dq_new(b.dK + h*hd_stride, b.dS_nat + h*ds_stride, b.ddQ + h*hd_stride,
                                      1, b.sl, b.hd, 0, 0, scale, sQ);
        cudaEventRecord(eK, sK);
        cudaEventRecord(eQ, sQ);
        cudaStreamWaitEvent(sM, eK, 0);
        cudaStreamWaitEvent(sM, eQ, 0);

        // Reset persisting L2 после консюмеров
        cudaStreamSynchronize(sK);
        cudaStreamSynchronize(sQ);
        cudaCtxResetPersistingL2Cache();
    }
    cudaStreamSynchronize(sM);
    cudaEventRecord(e1, 0);
    cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms;
}

// ARM2P pipeline depth 2 без брoни: merged(h+1) стартует || [dk(h) || dq(h)]
float arm2p_pipeline_nopersist(Buffers &b, cudaStream_t sM, cudaStream_t sK, cudaStream_t sQ, cudaEvent_t eM, cudaEvent_t eK, cudaEvent_t eQ) {
    float scale = 1.0f/sqrtf((float)b.hd);
    reset_outputs(b);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, 0);

    fa_bwd_dk::launch_d_precompute(b.dOO, b.dOG, b.dD, b.bh, b.sl, b.hd, sM);
    cudaEventRecord(eM, sM);

    size_t hd_stride = (size_t)b.sl * b.hd;
    size_t l_stride  = (size_t)b.sl;
    size_t ds_stride = (size_t)b.sl * b.stride_ds;

    // Start head 0 merged
    cudaStreamWaitEvent(sM, eM, 0);
    fa_bwd_merged_v1::launch_merged(
        b.dQ, b.dK, b.dV8, b.dOG, b.dL, b.dD, b.dS_nat, b.dS_T, b.ddV,
        1, b.sl, b.hd, 0, 0, scale, sM);
    cudaEventRecord(eM, sM);

    for (int h = 0; h < b.bh; ++h) {
        // Start merged(h+1) in parallel with dk/dq(h) if h+1 exists
        // But merged(h+1) должен ждать consumer(h) для dS_nat reuse
        // dk(h)/dq(h) должны ждать eM (merged(h) done)
        cudaStreamWaitEvent(sK, eM, 0);
        cudaStreamWaitEvent(sQ, eM, 0);
        fa_bwd_dk_new::launch_dk_new(b.dQ + h*hd_stride, b.dS_nat + h*ds_stride, b.ddK + h*hd_stride,
                                      1, b.sl, b.hd, 0, 0, scale, sK);
        fa_bwd_dq_new::launch_dq_new(b.dK + h*hd_stride, b.dS_nat + h*ds_stride, b.ddQ + h*hd_stride,
                                      1, b.sl, b.hd, 0, 0, scale, sQ);
        cudaEventRecord(eK, sK);
        cudaEventRecord(eQ, sQ);
        if (h+1 < b.bh) {
            int hn = h+1;
            cudaStreamWaitEvent(sM, eK, 0);
            cudaStreamWaitEvent(sM, eQ, 0);
            fa_bwd_merged_v1::launch_merged(
                b.dQ + hn*hd_stride, b.dK + hn*hd_stride, b.dV8 + hn*hd_stride,
                b.dOG + hn*hd_stride, b.dL + hn*l_stride, b.dD + hn*l_stride,
                b.dS_nat + hn*ds_stride, b.dS_T + hn*ds_stride, b.ddV + hn*hd_stride,
                1, b.sl, b.hd, 0, 0, scale, sM);
            cudaEventRecord(eM, sM);
        }
    }
    cudaStreamSynchronize(sM);
    cudaStreamSynchronize(sK);
    cudaStreamSynchronize(sQ);
    cudaEventRecord(e1, 0);
    cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms;
}

// ARM3P pipeline + persist
float arm3p_pipeline_persist(Buffers &b, cudaStream_t sM, cudaStream_t sK, cudaStream_t sQ, cudaEvent_t eM, cudaEvent_t eK, cudaEvent_t eQ) {
    float scale = 1.0f/sqrtf((float)b.hd);
    reset_outputs(b);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, 0);

    fa_bwd_dk::launch_d_precompute(b.dOO, b.dOG, b.dD, b.bh, b.sl, b.hd, sM);
    cudaEventRecord(eM, sM);

    size_t hd_stride = (size_t)b.sl * b.hd;
    size_t l_stride  = (size_t)b.sl;
    size_t ds_stride = (size_t)b.sl * b.stride_ds;
    size_t headBytes = ds_stride;

    auto set_persist = [&](cudaStream_t s, void *base, cudaAccessProperty hp) {
        cudaStreamAttrValue attr = {};
        attr.accessPolicyWindow.base_ptr = base;
        attr.accessPolicyWindow.num_bytes = headBytes;
        attr.accessPolicyWindow.hitRatio = 1.0f;
        attr.accessPolicyWindow.hitProp = hp;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &attr);
    };

    // head 0 merged
    set_persist(sM, b.dS_nat, cudaAccessPropertyStreaming);  // writer streaming
    cudaStreamWaitEvent(sM, eM, 0);
    fa_bwd_merged_v1::launch_merged(
        b.dQ, b.dK, b.dV8, b.dOG, b.dL, b.dD, b.dS_nat, b.dS_T, b.ddV,
        1, b.sl, b.hd, 0, 0, scale, sM);
    cudaEventRecord(eM, sM);

    for (int h = 0; h < b.bh; ++h) {
        // dk/dq(h) reader potoки: Persisting на dS(h)
        set_persist(sK, b.dS_nat + h*ds_stride, cudaAccessPropertyPersisting);
        set_persist(sQ, b.dS_nat + h*ds_stride, cudaAccessPropertyPersisting);
        cudaStreamWaitEvent(sK, eM, 0);
        cudaStreamWaitEvent(sQ, eM, 0);
        fa_bwd_dk_new::launch_dk_new(b.dQ + h*hd_stride, b.dS_nat + h*ds_stride, b.ddK + h*hd_stride,
                                      1, b.sl, b.hd, 0, 0, scale, sK);
        fa_bwd_dq_new::launch_dq_new(b.dK + h*hd_stride, b.dS_nat + h*ds_stride, b.ddQ + h*hd_stride,
                                      1, b.sl, b.hd, 0, 0, scale, sQ);
        cudaEventRecord(eK, sK);
        cudaEventRecord(eQ, sQ);

        if (h+1 < b.bh) {
            int hn = h+1;
            set_persist(sM, b.dS_nat + hn*ds_stride, cudaAccessPropertyStreaming);
            cudaStreamWaitEvent(sM, eK, 0);
            cudaStreamWaitEvent(sM, eQ, 0);
            fa_bwd_merged_v1::launch_merged(
                b.dQ + hn*hd_stride, b.dK + hn*hd_stride, b.dV8 + hn*hd_stride,
                b.dOG + hn*hd_stride, b.dL + hn*l_stride, b.dD + hn*l_stride,
                b.dS_nat + hn*ds_stride, b.dS_T + hn*ds_stride, b.ddV + hn*hd_stride,
                1, b.sl, b.hd, 0, 0, scale, sM);
            cudaEventRecord(eM, sM);
        }
        // Reset persist after consumers
        cudaStreamSynchronize(sK);
        cudaStreamSynchronize(sQ);
        cudaCtxResetPersistingL2Cache();
    }
    cudaStreamSynchronize(sM);
    cudaEventRecord(e1, 0);
    cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms;
}

int main(int argc, char **argv) {
    const int bh = 128, sl = 8192, hd = 128;
    const int warmup = 4, cycles = 8;

    // Fingerprint gate
    cudaFuncAttributes fa;
    struct { const char *n; const void *fp; int e; } gate[] = {
        {"D",       (const void*)fa_bwd_dk::kernel_d_precompute,      38},
        {"merged",  (const void*)fa_bwd_merged_v1::kernel_merged_v1, 252},
        {"dk_new",  (const void*)fa_bwd_dk_new::kernel_dk_new,       128},
        {"dq_new",  (const void*)fa_bwd_dq_new::kernel_dq_new,        69},
    };
    for (int i=0;i<4;++i) {
        cudaFuncGetAttributes(&fa, gate[i].fp);
        printf("FINGERPRINT %-8s numRegs=%d (expected %d) %s\n", gate[i].n, fa.numRegs, gate[i].e, (fa.numRegs==gate[i].e)?"OK":"MISMATCH");
    }
    printf("\n047 bh=%d sl=%d hd=%d, warmup=%d, cycles=%d (interleaved round-robin ARM1->2S->3S->2P->3P)\n\n", bh, sl, hd, warmup, cycles);

    Buffers B;
    alloc_buffers(B, bh, sl, hd);

    cudaStream_t sM, sK, sQ;
    CKR(cudaStreamCreate(&sM)); CKR(cudaStreamCreate(&sK)); CKR(cudaStreamCreate(&sQ));
    cudaEvent_t eM, eK, eQ;
    cudaEventCreate(&eM); cudaEventCreate(&eK); cudaEventCreate(&eQ);

    // Warmup
    for (int i=0;i<warmup;++i) arm1_monolith(B);
    CKR(cudaDeviceSynchronize());

    // Interleaved
    std::vector<float> t1(cycles), t2s(cycles), t3s(cycles), t2p(cycles), t3p(cycles);
    for (int c=0; c<cycles; ++c) {
        t1[c]  = arm1_monolith(B);
        t2s[c] = arm2s_dense_nopersist(B, sM, sK, sQ, eM, eK, eQ);
        t3s[c] = arm3s_dense_persist(B, sM, sK, sQ, eM, eK, eQ);
        t2p[c] = arm2p_pipeline_nopersist(B, sM, sK, sQ, eM, eK, eQ);
        t3p[c] = arm3p_pipeline_persist(B, sM, sK, sQ, eM, eK, eQ);
        printf("cycle %d: ARM1=%.3f  2S=%.3f  3S=%.3f  2P=%.3f  3P=%.3f\n",
               c, t1[c], t2s[c], t3s[c], t2p[c], t3p[c]);
    }

    auto median = [](std::vector<float> v){ std::sort(v.begin(), v.end()); return v[v.size()/2]; };
    printf("\nMedians (ms E2E, bh=128, one session):\n");
    printf("  ARM1 monolith  = %.3f\n", median(t1));
    printf("  ARM2S dense    = %.3f\n", median(t2s));
    printf("  ARM3S d+persist= %.3f\n", median(t3s));
    printf("  ARM2P pipe     = %.3f\n", median(t2p));
    printf("  ARM3P p+persist= %.3f\n", median(t3p));

    // Pair deltas (interest)
    printf("\nPair deltas by cycle (positive = second arm slower):\n");
    std::vector<float> d3s2s(cycles), d2s1(cycles), d3p2p(cycles), d2p1(cycles);
    for (int c=0;c<cycles;++c) {
        d3s2s[c] = t3s[c] - t2s[c];
        d2s1[c]  = t2s[c] - t1[c];
        d3p2p[c] = t3p[c] - t2p[c];
        d2p1[c]  = t2p[c] - t1[c];
        printf("  c%d: (3S-2S)=%+.3f  (2S-1)=%+.3f  (3P-2P)=%+.3f  (2P-1)=%+.3f\n",
               c, d3s2s[c], d2s1[c], d3p2p[c], d2p1[c]);
    }
    printf("\nMedian deltas:\n");
    printf("  Delta(3S-2S) = %+.3f ms  [прогноз: жив = -(4..6), мертв = ~0]\n", median(d3s2s));
    printf("  Delta(2S-1)  = %+.3f ms  [прогноз: +3..+12]\n", median(d2s1));
    printf("  Delta(3P-2P) = %+.3f ms  [прогноз: жив = -(1.5..2.5), мертв = ~0]\n", median(d3p2p));
    printf("  Delta(2P-1)  = %+.3f ms  [прогноз: +0.5..+2]\n", median(d2p1));

    cudaStreamDestroy(sM); cudaStreamDestroy(sK); cudaStreamDestroy(sQ);
    cudaEventDestroy(eM); cudaEventDestroy(eK); cudaEventDestroy(eQ);
    return 0;
}
