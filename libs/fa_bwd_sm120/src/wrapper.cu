/*
 * wrapper.cu — thin extern-C wrappers over v0.2.0 backward launchers.
 * A-LLM-1 G2 (2026-07-25).
 *
 * Тонкие extern-C entries — только развёртка аргументов, никакой логики.
 * Сигнатуры зеркалят launcher-контракты 1:1 (включая stride_ds=(sl+15)&~15
 * и dual-write dS_nat/dS_T — это ABI, не деталь).
 *
 * fingerprint gate (cudaFuncGetAttributes) — проверить numRegs после сборки
 * против cert reference (38/252/124/69). Расхождение = СТОП, не «примерно».
 */
#include "../include/fa_bwd_sm120.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

/* ---- Kernel forward declarations (для cudaFuncGetAttributes) ---- */
namespace fa_bwd_dk {
    void launch_d_precompute(
        const __half *O, const __half *dO, float *D,
        int bh, int sl, int hd, cudaStream_t stream);

    __global__ void kernel_d_precompute(const __half*, const __half*, float*, int, int, int);
}
namespace fa_bwd_merged_v1 {
    void launch_merged(
        const uint8_t *Q, const uint8_t *K, const uint8_t *V,
        const __half *dO_g, const float *L, const float *D,
        uint8_t *dS_nat, uint8_t *dS_T, float *dV,
        int bh, int sl, int hd,
        int causal, int window,
        float scale, cudaStream_t stream);
}
namespace fa_bwd_dk_new {
    void launch_dk_new(
        const uint8_t *Q, const uint8_t *dS_T,
        float *dK,
        int bh, int sl, int hd,
        int causal, int window,
        float scale, cudaStream_t stream);
}
namespace fa_bwd_dq_new {
    void launch_dq_new(
        const uint8_t *K, const uint8_t *dS_nat,
        float *dQ,
        int bh, int sl, int hd,
        int causal, int window,
        float scale, cudaStream_t stream);
}

/* ---- extern-C wrappers ---- */

extern "C" int32_t gt_fa_bwd_d_precompute(
    const void *o_half, const void *do_half, void *d_f32,
    int bh, int sl, int hd, void *stream)
{
    fa_bwd_dk::launch_d_precompute(
        (const __half*)o_half, (const __half*)do_half, (float*)d_f32,
        bh, sl, hd, (cudaStream_t)stream);
    return (int32_t)cudaGetLastError();
}

extern "C" int32_t gt_fa_bwd_merged(
    const void *q_fp8, const void *k_fp8, const void *v_fp8,
    const void *do_half, const void *l_f32, const void *d_f32,
    void *dS_nat_fp8, void *dS_T_fp8, void *dV_f32,
    int bh, int sl, int hd,
    int causal, int window,
    float scale, void *stream)
{
    fa_bwd_merged_v1::launch_merged(
        (const uint8_t*)q_fp8, (const uint8_t*)k_fp8, (const uint8_t*)v_fp8,
        (const __half*)do_half, (const float*)l_f32, (const float*)d_f32,
        (uint8_t*)dS_nat_fp8, (uint8_t*)dS_T_fp8, (float*)dV_f32,
        bh, sl, hd, causal, window, scale,
        (cudaStream_t)stream);
    return (int32_t)cudaGetLastError();
}

extern "C" int32_t gt_fa_bwd_dk(
    const void *q_fp8, const void *dS_T_fp8,
    void *dK_f32,
    int bh, int sl, int hd,
    int causal, int window,
    float scale, void *stream)
{
    fa_bwd_dk_new::launch_dk_new(
        (const uint8_t*)q_fp8, (const uint8_t*)dS_T_fp8, (float*)dK_f32,
        bh, sl, hd, causal, window, scale,
        (cudaStream_t)stream);
    return (int32_t)cudaGetLastError();
}

extern "C" int32_t gt_fa_bwd_dq(
    const void *k_fp8, const void *dS_nat_fp8,
    void *dQ_f32,
    int bh, int sl, int hd,
    int causal, int window,
    float scale, void *stream)
{
    fa_bwd_dq_new::launch_dq_new(
        (const uint8_t*)k_fp8, (const uint8_t*)dS_nat_fp8, (float*)dQ_f32,
        bh, sl, hd, causal, window, scale,
        (cudaStream_t)stream);
    return (int32_t)cudaGetLastError();
}

/* Kernel forward declarations for fingerprint query.
 * These symbols exist in the linked v0.2.0 objects.
 * We take their addresses to feed cudaFuncGetAttributes. */
namespace fa_bwd_merged_v1 { __global__ void kernel_merged_v1(
    const uint8_t*, const uint8_t*, const uint8_t*, const __half*,
    const float*, const float*, uint8_t*, uint8_t*, float*,
    int, int, int, int, int, float); }
namespace fa_bwd_dk_new { __global__ void kernel_dk_new(
    const uint8_t*, const uint8_t*, float*,
    int, int, int, int, int, float); }
namespace fa_bwd_dq_new { __global__ void kernel_dq_new(
    const uint8_t*, const uint8_t*, float*,
    int, int, int, int, int, float); }

extern "C" int32_t gt_fa_bwd_kernel_regs(int kernel_id)
{
    cudaFuncAttributes attr;
    cudaError_t e;
    switch (kernel_id) {
    case 0:
        e = cudaFuncGetAttributes(&attr, (const void*)fa_bwd_dk::kernel_d_precompute);
        break;
    case 1:
        e = cudaFuncGetAttributes(&attr, (const void*)fa_bwd_merged_v1::kernel_merged_v1);
        break;
    case 2:
        e = cudaFuncGetAttributes(&attr, (const void*)fa_bwd_dk_new::kernel_dk_new);
        break;
    case 3:
        e = cudaFuncGetAttributes(&attr, (const void*)fa_bwd_dq_new::kernel_dq_new);
        break;
    default:
        return -1;
    }
    if (e != cudaSuccess) return -1;
    return (int32_t)attr.numRegs;
}
