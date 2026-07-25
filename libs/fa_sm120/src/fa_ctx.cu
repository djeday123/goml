/*
 * fa_ctx.cu — context lifecycle, arch probe, kernel dispatch entry.
 *
 * v121r kernel is fully wired (linked from libs/_v121r_kernel.cu).
 * Other kernels return FA_ERR_INTERNAL with diagnostic until SHIP-2
 * incrementally links them. Dispatcher already returns correct id;
 * wrapper rejects the call with clear error.
 */
#include "../include/fa_sm120.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <new>

#define FA_VERSION_STRING "0.1.0+652T-sm120a"

/* ---- v121r kernel (linked-in, extracted from production source) ---- */
namespace fa_sm120_v121r {
    extern void launch(
        const uint8_t* Q, const uint8_t* K, const uint8_t* V, __half* O,
        int bh, int sl, int hd, int causal, int window,
        float scale, cudaStream_t stream);
}

/* ---- v121r-TRAIN kernel (A-LLM-1 G1: fwd + LSE writeback for backward) ---- */
namespace fa_sm120_v121r_train {
    extern void launch(
        const uint8_t* Q, const uint8_t* K, const uint8_t* V, __half* O,
        float* L_out,        /* [bh, sl] LSE; nullptr = skip L */
        int bh, int sl, int hd, int causal, int window,
        float scale, cudaStream_t stream);
}

struct fa_ctx {
    int device_id;
    int cc_major, cc_minor;     /* compute capability */
    cudaDeviceProp prop;
    char last_err[256];
};

static void set_err(fa_ctx_t* ctx, const char* msg) {
    if (!ctx) return;
    std::snprintf(ctx->last_err, sizeof(ctx->last_err), "%s", msg);
}

static void set_cuda_err(fa_ctx_t* ctx, cudaError_t e, const char* where) {
    if (!ctx) return;
    std::snprintf(ctx->last_err, sizeof(ctx->last_err), "%s: %s",
                  where, cudaGetErrorString(e));
}

extern "C" fa_status_t fa_create(fa_ctx_t** ctx_out)
{
    if (!ctx_out) return FA_ERR_INVALID_ARG;
    *ctx_out = nullptr;

    int dev;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return FA_ERR_CUDA;

    fa_ctx_t* ctx = new (std::nothrow) fa_ctx_t();
    if (!ctx) return FA_ERR_OOM;
    ctx->device_id = dev;
    ctx->last_err[0] = '\0';

    e = cudaGetDeviceProperties(&ctx->prop, dev);
    if (e != cudaSuccess) { delete ctx; return FA_ERR_CUDA; }
    ctx->cc_major = ctx->prop.major;
    ctx->cc_minor = ctx->prop.minor;

    /* Strict sm_120a check (consumer Blackwell). */
    if (ctx->cc_major != 12 || ctx->cc_minor != 0) {
        std::snprintf(ctx->last_err, sizeof(ctx->last_err),
                      "GPU compute capability %d.%d is not sm_120a (need 12.0).",
                      ctx->cc_major, ctx->cc_minor);
        /* Store ctx so client can read last_err, then destroy. */
        *ctx_out = ctx;
        return FA_ERR_UNSUPPORTED_ARCH;
    }

    *ctx_out = ctx;
    return FA_OK;
}

extern "C" fa_status_t fa_destroy(fa_ctx_t* ctx)
{
    if (ctx) delete ctx;
    return FA_OK;
}

extern "C" const char* fa_version(void) { return FA_VERSION_STRING; }

extern "C" const char* fa_status_str(fa_status_t s)
{
    switch (s) {
    case FA_OK:                       return "OK";
    case FA_ERR_INVALID_ARG:          return "invalid argument";
    case FA_ERR_UNSUPPORTED_ARCH:     return "unsupported GPU arch (need sm_120a)";
    case FA_ERR_UNSUPPORTED_HD:       return "unsupported head_dim (need 64 or 128)";
    case FA_ERR_UNSUPPORTED_SHAPE:    return "unsupported shape";
    case FA_ERR_CUDA:                 return "CUDA runtime error";
    case FA_ERR_OOM:                  return "out of memory";
    case FA_ERR_INTERNAL:             return "internal error";
    }
    return "unknown";
}

extern "C" const char* fa_last_cuda_error(fa_ctx_t* ctx)
{
    if (!ctx) return "";
    return ctx->last_err;
}

extern "C" fa_status_t fa_forward(fa_ctx_t* ctx,
                                  const void* q, const void* k, const void* v,
                                  void* o,
                                  int bh, int sl, int hd,
                                  int causal, int window,
                                  float scale, fa_stream_t stream)
{
    if (!ctx) return FA_ERR_INVALID_ARG;
    if (!q || !k || !v || !o) {
        set_err(ctx, "null pointer in q/k/v/o");
        return FA_ERR_INVALID_ARG;
    }
    if (bh <= 0 || sl <= 0) {
        set_err(ctx, "bh/sl must be > 0");
        return FA_ERR_INVALID_ARG;
    }
    if (hd != 64 && hd != 128) {
        set_err(ctx, "head_dim must be 64 or 128");
        return FA_ERR_UNSUPPORTED_HD;
    }
    if (window < 0 || window > sl) {
        set_err(ctx, "window must be in [0, seq_len]");
        return FA_ERR_INVALID_ARG;
    }
    if (causal != 0 && causal != 1) {
        set_err(ctx, "causal must be 0 or 1");
        return FA_ERR_INVALID_ARG;
    }

    fa_kernel_id_t kid = fa_dispatch_select(bh, sl, hd, causal, window);
    if (kid == FA_KERNEL_NONE) {
        set_err(ctx, "no kernel matches this configuration");
        return FA_ERR_UNSUPPORTED_SHAPE;
    }

    /* Wire v121r — production peak kernel. Others stubbed for SHIP-2 incremental wiring. */
    if (kid == FA_KERNEL_V121R) {
        fa_sm120_v121r::launch(
            (const uint8_t*)q, (const uint8_t*)k, (const uint8_t*)v,
            (__half*)o, bh, sl, hd, causal, window, scale,
            (cudaStream_t)stream);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            set_cuda_err(ctx, e, "v121r launch");
            return FA_ERR_CUDA;
        }
        return FA_OK;
    }

    char msg[200];
    std::snprintf(msg, sizeof(msg),
                  "kernel %s (id=%d) selected but not yet linked (SHIP-2). "
                  "Set FA_SM120_DEBUG=1 to see dispatcher trace.",
                  fa_kernel_name(kid), (int)kid);
    set_err(ctx, msg);
    return FA_ERR_INTERNAL;
}

/* fa_forward_train — forward with L (LSE) output for backward path.
 * A-LLM-1 G1 (2026-07-25). Separate function from fa_forward — production
 * fa_forward ABI/behavior UNCHANGED (боевой символ не трогаем).
 *
 * Same Q/K/V/O semantics as fa_forward:
 *   Q,K,V: FP8 e4m3 device ptrs, [bh, sl, hd] row-major
 *   O:     FP16 device ptr, [bh, sl, hd]
 *   scale: pre-composed = softmax_scale * scale_Q * scale_K (launcher
 *          hardcodes qk_descale=1.0, v_descale=1.0 in v121r-train — свёртка
 *          scale·scale_Q·scale_K тождественна: все три множителя одного
 *          скаляра до softmax, A-LLM-1 решение #2).
 * Additional:
 *   l_out: F32 device ptr, [bh, sl] LSE = m_i + log(sum(exp(q·k - m_i))).
 *          MAY be NULL — kernel skips L writeback.
 *
 * Currently hd=128 only (train kernel FA_STRIDE=128 hardcoded).
 * Uses fa_sm120_v121r_train::launch directly (no dispatcher — only v121r-train
 * wired at this stage).
 */
extern "C" fa_status_t fa_forward_train(fa_ctx_t* ctx,
                                        const void* q, const void* k, const void* v,
                                        void* o, void* l_out,
                                        int bh, int sl, int hd,
                                        int causal, int window,
                                        float scale, fa_stream_t stream)
{
    if (!ctx) return FA_ERR_INVALID_ARG;
    if (!q || !k || !v || !o) {
        set_err(ctx, "fa_forward_train: null pointer in q/k/v/o");
        return FA_ERR_INVALID_ARG;
    }
    /* l_out may be NULL (matches kernel semantics). */
    if (bh <= 0 || sl <= 0) {
        set_err(ctx, "fa_forward_train: bh/sl must be > 0");
        return FA_ERR_INVALID_ARG;
    }
    if (hd != 128) {
        set_err(ctx, "fa_forward_train: head_dim must be 128 (v121r-train FA_STRIDE hardcoded)");
        return FA_ERR_UNSUPPORTED_HD;
    }
    if (window < 0 || window > sl) {
        set_err(ctx, "fa_forward_train: window must be in [0, seq_len]");
        return FA_ERR_INVALID_ARG;
    }
    if (causal != 0 && causal != 1) {
        set_err(ctx, "fa_forward_train: causal must be 0 or 1");
        return FA_ERR_INVALID_ARG;
    }

    fa_sm120_v121r_train::launch(
        (const uint8_t*)q, (const uint8_t*)k, (const uint8_t*)v,
        (__half*)o, (float*)l_out,
        bh, sl, hd, causal, window, scale,
        (cudaStream_t)stream);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        set_cuda_err(ctx, e, "v121r-train launch");
        return FA_ERR_CUDA;
    }
    return FA_OK;
}
