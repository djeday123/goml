package gotorch

// B-impl-4: adapter extension for mixed-precision MatMul (F16, F8E4M3) +
// F16 / F8 quantize helpers. Extension API on *gotorch.Backend через
// type-assertion (тот же паттерн что RMSNorm/Embedding/RoPE).
//
// Requires libgotorch_blas_wrapper.so (F16 через gt_gemm_ex,
// F8 через gt_lt_matmul_fp8_e4m3).

import (
	"fmt"

	"github.com/djeday123/goml/backend"
)

// MatMulF16 -- FP16 IO + F32 out. Формы row-major [m,k]×[k,n].
func (b *Backend) MatMulF16(a, bb, c backend.Storage, m, n, k int) error {
	if m <= 0 || n <= 0 || k <= 0 {
		return fmt.Errorf("gotorch adapter MatMulF16: m/n/k must be > 0")
	}
	return b.gt.MatMulF16(wrapForeign(a), wrapForeign(bb), wrapForeign(c), m, n, k)
}

// MatMulF8E4M3 -- FP8 E4M3 IO + FP16 out (NVIDIA cublasLt path).
// scaleA/B/C -- device float* per-tensor scales.
// amaxD -- optional device float* (nil = не устанавливать).
func (b *Backend) MatMulF8E4M3(a, bb, c, scaleA, scaleB, scaleC, amaxD backend.Storage, m, n, k int) error {
	if m <= 0 || n <= 0 || k <= 0 {
		return fmt.Errorf("gotorch adapter MatMulF8E4M3: m/n/k must be > 0")
	}
	// amaxD может быть nil.
	if amaxD == nil {
		return b.gt.MatMulF8E4M3(
			wrapForeign(a), wrapForeign(bb), wrapForeign(c),
			wrapForeign(scaleA), wrapForeign(scaleB), wrapForeign(scaleC),
			nil, m, n, k,
		)
	}
	return b.gt.MatMulF8E4M3(
		wrapForeign(a), wrapForeign(bb), wrapForeign(c),
		wrapForeign(scaleA), wrapForeign(scaleB), wrapForeign(scaleC),
		wrapForeign(amaxD), m, n, k,
	)
}

// CastF32ToF16, CastF16ToF32, QuantizeF32ToF8E4M3, CastF8E4M3ToF32 -- passthrough.
func (b *Backend) CastF32ToF16(src, dst backend.Storage, n int) error {
	return b.gt.CastF32ToF16(wrapForeign(src), wrapForeign(dst), n)
}
func (b *Backend) CastF16ToF32(src, dst backend.Storage, n int) error {
	return b.gt.CastF16ToF32(wrapForeign(src), wrapForeign(dst), n)
}
func (b *Backend) QuantizeF32ToF8E4M3(src, dst, scale, amax backend.Storage, n int) error {
	return b.gt.QuantizeF32ToF8E4M3(wrapForeign(src), wrapForeign(dst),
		wrapForeign(scale), wrapForeign(amax), n)
}
func (b *Backend) CastF8E4M3ToF32(src, dst, scale backend.Storage, n int) error {
	return b.gt.CastF8E4M3ToF32(wrapForeign(src), wrapForeign(dst),
		wrapForeign(scale), n)
}
