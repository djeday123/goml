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

// MatMulF32Ex -- F32 GEMM с per-call trans-флагами (A-0, 2026-07-24).
// Row-major: C[m,n] = op(A)[m,k] · op(B)[k,n], где op зависит от transA/transB.
// Compute: full FP32 accumulator (не TF32).
// Требует libgotorch_blas_wrapper.so.
// Назначение: gradOW = normed^T @ gradLogits на GPU (заменяет CPU host-loop бэкворда B2).
func (b *Backend) MatMulF32Ex(a, bb, c backend.Storage, m, n, k int, transA, transB bool) error {
	if m <= 0 || n <= 0 || k <= 0 {
		return fmt.Errorf("gotorch adapter MatMulF32Ex: m/n/k must be > 0")
	}
	return b.gt.MatMulF32Ex(wrapForeign(a), wrapForeign(bb), wrapForeign(c), m, n, k, transA, transB)
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

// QuantizeF32ToF8E4M3Unit -- A-LLM-5 квант-контракт O(1): scale = amax,
// decoded |dst| <= 1 (контракт FP16-акков FA-ядер, Н4).
func (b *Backend) QuantizeF32ToF8E4M3Unit(src, dst, scale, amax backend.Storage, n int) error {
	return b.gt.QuantizeF32ToF8E4M3Unit(wrapForeign(src), wrapForeign(dst),
		wrapForeign(scale), wrapForeign(amax), n)
}
func (b *Backend) CastF8E4M3ToF32(src, dst, scale backend.Storage, n int) error {
	return b.gt.CastF8E4M3ToF32(wrapForeign(src), wrapForeign(dst),
		wrapForeign(scale), n)
}
