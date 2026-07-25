/*
 * libfa_bwd_sm120 — FlashAttention backward chain wrappers для sm_120a.
 *
 * A-LLM-1 G2 (2026-07-25). Тонкие extern-C обёртки поверх v0.2.0 namespace
 * launchers (release_v0.2.0/src/*.cu). Только развёртка аргументов, никакой
 * логики. Сигнатуры зеркалят launcher-контракты 1:1.
 *
 * ABI/контракт (общий, см. v0.2.0/README):
 *   Q,K,V:   FP8 e4m3 (uint8), row-major [bh, sl, hd].
 *   dO,O:    FP16 (uint16 buffers, little-endian), row-major [bh, sl, hd].
 *   L,D:     FP32, [bh, sl] row-major.
 *   dQ,dK,dV: FP32, [bh, sl, hd] row-major. **Caller must zero-init** (dV,dK,dQ).
 *   dS_nat,dS_T: FP8 (uint8), padded stride_ds = (sl + 15) & ~15 per row.
 *                dual-write: merged writes both, dk_new reads dS_T, dq_new reads dS_nat.
 *   causal:  {0, 1}. window: [0, sl].
 *   scale:   softmax scale (typically 1/sqrt(hd), composed with FP8 dequant if needed).
 *
 * Возвращают int32_t = cudaError_t (0 = cudaSuccess).
 *
 * Ограничения (проверяются launcher'ом с exit(1)):
 *   hd == 128 only (все три kernel'а).
 *   sm_120a.
 */
#ifndef FA_BWD_SM120_H_
#define FA_BWD_SM120_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
  #define FA_BWD_API __attribute__((visibility("default")))
#else
  #define FA_BWD_API
#endif

/*
 * gt_fa_bwd_d_precompute: D[i] = sum_j (dO[i,j] * O[i,j])
 *   O, dO: FP16 [bh, sl, hd]
 *   D:     FP32 [bh, sl]
 * Wraps fa_bwd_dk::launch_d_precompute.
 */
FA_BWD_API int32_t gt_fa_bwd_d_precompute(
    const void *o_half, const void *do_half, void *d_f32,
    int bh, int sl, int hd, void *stream);

/*
 * gt_fa_bwd_merged: fused ds_gen + dV_p1.
 *   Q,K,V:      FP8 [bh, sl, hd]
 *   dO_half:    FP16 [bh, sl, hd]
 *   L, D:       FP32 [bh, sl]
 *   dS_nat:     FP8 padded [bh, sl_i, sl_j(stride_ds)] — for dq_new
 *   dS_T:       FP8 padded [bh, sl_j, sl_i(stride_ds)] — for dk_new
 *   dV:         FP32 [bh, sl, hd] (must be zero-init by caller)
 * Wraps fa_bwd_merged_v1::launch_merged.
 */
FA_BWD_API int32_t gt_fa_bwd_merged(
    const void *q_fp8, const void *k_fp8, const void *v_fp8,
    const void *do_half, const void *l_f32, const void *d_f32,
    void *dS_nat_fp8, void *dS_T_fp8, void *dV_f32,
    int bh, int sl, int hd,
    int causal, int window,
    float scale, void *stream);

/*
 * gt_fa_bwd_dk: dK = dS_T @ Q (essentially).
 *   Q:      FP8 [bh, sl, hd]
 *   dS_T:   FP8 padded (from merged output)
 *   dK:     FP32 [bh, sl, hd] (must be zero-init by caller)
 * Wraps fa_bwd_dk_new::launch_dk_new.
 */
FA_BWD_API int32_t gt_fa_bwd_dk(
    const void *q_fp8, const void *dS_T_fp8,
    void *dK_f32,
    int bh, int sl, int hd,
    int causal, int window,
    float scale, void *stream);

/*
 * gt_fa_bwd_dq: dQ = dS_nat @ K (essentially).
 *   K:       FP8 [bh, sl, hd]
 *   dS_nat:  FP8 padded (from merged output)
 *   dQ:      FP32 [bh, sl, hd] (must be zero-init by caller)
 * Wraps fa_bwd_dq_new::launch_dq_new.
 */
FA_BWD_API int32_t gt_fa_bwd_dq(
    const void *k_fp8, const void *dS_nat_fp8,
    void *dQ_f32,
    int bh, int sl, int hd,
    int causal, int window,
    float scale, void *stream);

/*
 * Diagnostic: query kernel numRegs via cudaFuncGetAttributes.
 * Fingerprint gate for build validation (cert reference: d_precompute=38,
 * merged=252, dk_new=124, dq_new=69).
 *
 * kernel_id: 0=d_precompute, 1=merged, 2=dk_new, 3=dq_new.
 * Returns numRegs on success, -1 on failure.
 */
FA_BWD_API int32_t gt_fa_bwd_kernel_regs(int kernel_id);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FA_BWD_SM120_H_ */
