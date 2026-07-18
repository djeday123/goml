// 047 §1.b: грошовая проба head-offset launch.
// h=7 offset launch (bh=1, ptr shifted) vs monolith bh=8 → byte-equivalence h=7 outputs.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "fa_bwd_common.cuh"

#define CKR(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); }} while (0)

namespace fa_bwd_dk {
void launch_d_precompute(const __half *, const __half *, float *, int, int, int, cudaStream_t);
}
namespace fa_bwd_merged_v1 {
void launch_merged(const uint8_t *, const uint8_t *, const uint8_t *, const __half *, const float *, const float *,
                    uint8_t *, uint8_t *, float *, int, int, int, int, int, float, cudaStream_t);
}
namespace fa_bwd_dk_new {
void launch_dk_new(const uint8_t *, const uint8_t *, float *, int, int, int, int, int, float, cudaStream_t);
}
namespace fa_bwd_dq_new {
void launch_dq_new(const uint8_t *, const uint8_t *, float *, int, int, int, int, int, float, cudaStream_t);
}

int main() {
    const int bh = 8, sl = 8192, hd = 128, causal = 0, window = 0;
    const int H_TEST = 7;
    int stride_ds = (sl + 15) & ~15;
    float scale = 1.0f / sqrtf((float)hd);

    size_t sz  = (size_t)bh * sl * hd;
    size_t lsz = (size_t)bh * sl;
    size_t dsz = (size_t)bh * sl * stride_ds;

    // Fill random
    std::vector<uint8_t> Q8(sz), K8(sz), V8(sz);
    std::vector<__half>  O16(sz), dO16(sz);
    std::vector<float>   L32(lsz);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.6f);
    for (size_t i=0;i<sz;++i){Q8[i]=float_to_e4m3_host(dist(rng));K8[i]=float_to_e4m3_host(dist(rng));V8[i]=float_to_e4m3_host(dist(rng));O16[i]=__float2half_rn(dist(rng));dO16[i]=__float2half_rn(dist(rng));}
    for (size_t i=0;i<lsz;++i) L32[i] = dist(rng);

    uint8_t *dQ,*dK,*dV8; __half *dOO,*dOG; float *dL,*dD;
    uint8_t *dS_nat_mono, *dS_T_mono, *dS_nat_off, *dS_T_off;
    float *ddV_mono, *ddK_mono, *ddQ_mono;
    float *ddV_off, *ddK_off, *ddQ_off;
    CKR(cudaMalloc(&dQ,sz));CKR(cudaMalloc(&dK,sz));CKR(cudaMalloc(&dV8,sz));
    CKR(cudaMalloc(&dOO,sz*sizeof(__half)));CKR(cudaMalloc(&dOG,sz*sizeof(__half)));
    CKR(cudaMalloc(&dL,lsz*sizeof(float)));CKR(cudaMalloc(&dD,lsz*sizeof(float)));
    CKR(cudaMalloc(&ddV_mono,sz*sizeof(float)));CKR(cudaMalloc(&ddK_mono,sz*sizeof(float)));CKR(cudaMalloc(&ddQ_mono,sz*sizeof(float)));
    CKR(cudaMalloc(&ddV_off,sz*sizeof(float)));CKR(cudaMalloc(&ddK_off,sz*sizeof(float)));CKR(cudaMalloc(&ddQ_off,sz*sizeof(float)));
    CKR(cudaMalloc(&dS_nat_mono,dsz));CKR(cudaMalloc(&dS_T_mono,dsz));
    CKR(cudaMalloc(&dS_nat_off,dsz));CKR(cudaMalloc(&dS_T_off,dsz));
    CKR(cudaMemcpy(dQ,Q8.data(),sz,cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(dK,K8.data(),sz,cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(dV8,V8.data(),sz,cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(dOO,O16.data(),sz*sizeof(__half),cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(dOG,dO16.data(),sz*sizeof(__half),cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(dL,L32.data(),lsz*sizeof(float),cudaMemcpyHostToDevice));

    // MONOLITH bh=8 chain
    CKR(cudaMemset(ddV_mono, 0, sz*sizeof(float)));
    CKR(cudaMemset(ddK_mono, 0, sz*sizeof(float)));
    CKR(cudaMemset(ddQ_mono, 0, sz*sizeof(float)));
    fa_bwd_dk::launch_d_precompute(dOO, dOG, dD, bh, sl, hd, 0);
    fa_bwd_merged_v1::launch_merged(dQ, dK, dV8, dOG, dL, dD, dS_nat_mono, dS_T_mono, ddV_mono, bh, sl, hd, causal, window, scale, 0);
    fa_bwd_dk_new::launch_dk_new(dQ, dS_nat_mono, ddK_mono, bh, sl, hd, causal, window, scale, 0);
    fa_bwd_dq_new::launch_dq_new(dK, dS_nat_mono, ddQ_mono, bh, sl, hd, causal, window, scale, 0);
    CKR(cudaDeviceSynchronize());

    // PER-HEAD OFFSET bh=1, h=H_TEST
    size_t off_seq   = (size_t)H_TEST * sl * hd;
    size_t off_l     = (size_t)H_TEST * sl;
    size_t off_ds    = (size_t)H_TEST * sl * stride_ds;

    CKR(cudaMemset(ddV_off, 0, sz*sizeof(float)));
    CKR(cudaMemset(ddK_off, 0, sz*sizeof(float)));
    CKR(cudaMemset(ddQ_off, 0, sz*sizeof(float)));

    // NB: D precompute for h=7 needs offset dOO/dOG/dD too
    fa_bwd_dk::launch_d_precompute(dOO + off_seq, dOG + off_seq, dD + off_l, 1, sl, hd, 0);
    fa_bwd_merged_v1::launch_merged(
        dQ + off_seq, dK + off_seq, dV8 + off_seq,
        dOG + off_seq, dL + off_l, dD + off_l,
        dS_nat_off + off_ds, dS_T_off + off_ds,   // Output offsets — kernel будет писать b=0 = shifted позиция
        ddV_off + off_seq,
        1, sl, hd, causal, window, scale, 0);
    fa_bwd_dk_new::launch_dk_new(
        dQ + off_seq, dS_nat_off + off_ds, ddK_off + off_seq,
        1, sl, hd, causal, window, scale, 0);
    fa_bwd_dq_new::launch_dq_new(
        dK + off_seq, dS_nat_off + off_ds, ddQ_off + off_seq,
        1, sl, hd, causal, window, scale, 0);
    CKR(cudaDeviceSynchronize());

    // Compare h=H_TEST slice byte-by-byte
    auto cmp = [&](const char *tag, float *a, float *b, size_t offs, size_t n) -> size_t {
        std::vector<float> ha(n), hb(n);
        cudaMemcpy(ha.data(), a + offs, n*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hb.data(), b + offs, n*sizeof(float), cudaMemcpyDeviceToHost);
        size_t mism=0; float mx=0;
        for (size_t i=0;i<n;++i){ float d=fabsf(ha[i]-hb[i]); if(d>0)mism++; if(d>mx)mx=d; }
        printf("  %s mism=%zu / %zu max_abs_diff=%.3e %s\n", tag, mism, n, mx, mism==0?"BYTE-EQUIVALENT":"DIVERGENT");
        return mism;
    };

    printf("047 §1.b probe: byte-equivalence h=%d offset-launch vs monolith bh=%d slice\n\n", H_TEST, bh);
    size_t head_sz = (size_t)sl * hd;
    size_t m_dv = cmp("dV[h=7]", ddV_mono, ddV_off, off_seq, head_sz);
    size_t m_dk = cmp("dK[h=7]", ddK_mono, ddK_off, off_seq, head_sz);
    size_t m_dq = cmp("dQ[h=7]", ddQ_mono, ddQ_off, off_seq, head_sz);
    printf("\nVerdict: %s\n", (m_dv+m_dk+m_dq==0) ? "PASS — head-offset launch БАЙТ-ЭКВИВАЛЕНТНО" : "FAIL — orchestration ошибка");
    return (m_dv+m_dk+m_dq==0) ? 0 : 1;
}
