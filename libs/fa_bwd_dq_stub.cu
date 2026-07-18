// =====================================================================
//  fa_bwd_dq_stub.cu — B4.1 stub launch_dq that writes zeros.
//
//  Purpose: lets fa_bwd_dq_test.cu build and run BEFORE B4.2 kernel exists.
//  Test will report 0% PASS until real kernel replaces this stub.
//  Real fa_bwd_dq.cu (B4.2) provides launch_dq with same signature.
// =====================================================================

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace fa_bwd_dq {

void launch_dq(
    const uint8_t * /*Q*/, const uint8_t * /*K*/, const uint8_t * /*V*/,
    const __half *  /*dO_g*/, const float * /*L*/, const float * /*D*/,
    float *dQ,
    int bh, int sl, int hd,
    int /*causal*/, int /*window*/,
    float /*scale*/, cudaStream_t stream)
{
    // Stub: zero output. Real kernel pending B4.2.
    cudaMemsetAsync(dQ, 0, (size_t)bh * sl * hd * sizeof(float), stream);
}

} // namespace fa_bwd_dq
