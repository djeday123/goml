package main

// FP32 vs FP16 vs FP8 MatMul Benchmark
//
// Compares:
//   cublasSgemm_v2 (FP32 IO + TF32 compute) -- ~82 TFLOPS theoretical
//   cublasGemmEx   (FP16 IO + TF32 compute) -- ~165 TFLOPS theoretical
//   cublasGemmEx   (FP8 IO + FP32 compute)  -- ~330 TFLOPS theoretical (SM 8.9+)
//
// Usage: go run cmd/fp16bench/main.go

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"time"
	"unsafe"

	"github.com/djeday123/goml/backend"
	_ "github.com/djeday123/goml/backend/cpu"
	_ "github.com/djeday123/goml/backend/cuda"
	"github.com/djeday123/goml/core"
)

func main() {
	runtime.LockOSThread()
	fmt.Println("=== FP32 vs FP16 vs FP8 MatMul Benchmark ===")
	fmt.Println()

	gpu, err := backend.Get(backend.CUDA)
	if err != nil {
		fmt.Printf("CUDA not available: %v\n", err)
		return
	}

	// Force init
	s, _ := gpu.Alloc(4)
	gpu.Free(s)
	fmt.Println()

	// Check GemmEx availability
	type gemmExChecker interface {
		HasGemmEx() bool
	}
	type cublasBacker interface {
		CuBLAS() interface{}
	}

	sizes := [][3]int{
		{512, 512, 512},
		{1024, 1024, 1024},
		{2048, 2048, 2048},
		{4096, 4096, 4096},
	}

	fmt.Printf("%-20s | %-15s | %-15s | %-15s | FP16/32 | FP8/32\n",
		"Size", "FP32 (TFLOPS)", "FP16 (TFLOPS)", "FP8 (TFLOPS)")
	fmt.Println("---------------------+-----------------+-----------------+-----------------+---------+-------")

	for _, sz := range sizes {
		M, K, N := sz[0], sz[1], sz[2]
		flops := float64(2) * float64(M) * float64(K) * float64(N)

		// --- FP32 benchmark ---
		dataA := randomF32(M * K)
		dataB := randomF32(K * N)

		gpuA32 := hostToGPU(gpu, f32ToBytes(dataA))
		gpuB32 := hostToGPU(gpu, f32ToBytes(dataB))
		gpuC32, _ := gpu.Alloc(M * N * 4)

		// Warmup
		gpu.MatMul(gpuC32, gpuA32, gpuB32, core.Shape{M, K}, core.Shape{K, N}, core.Float32)
		syncGPU(gpu)

		iters := 50
		if M >= 4096 {
			iters = 20
		}

		start := time.Now()
		for i := 0; i < iters; i++ {
			gpu.MatMul(gpuC32, gpuA32, gpuB32, core.Shape{M, K}, core.Shape{K, N}, core.Float32)
		}
		syncGPU(gpu)
		fp32Time := time.Since(start).Seconds() / float64(iters)
		fp32Tflops := flops / fp32Time / 1e12

		// Read FP32 result for correctness check later
		c32Host := gpuToHostF32(gpu, gpuC32, M*N)

		gpu.Free(gpuA32)
		gpu.Free(gpuB32)
		gpu.Free(gpuC32)

		// --- FP16 benchmark ---
		// Convert to FP16 on CPU, upload
		dataAf16 := f32SliceToF16Bytes(dataA)
		dataBf16 := f32SliceToF16Bytes(dataB)

		gpuA16 := hostToGPU(gpu, dataAf16)
		gpuB16 := hostToGPU(gpu, dataBf16)
		gpuC16, _ := gpu.Alloc(M * N * 4) // output is FP32

		// Use MatMulF16 via direct cublas access
		aPtr := devPtr(gpuA16)
		bPtr := devPtr(gpuB16)
		cPtr := devPtr(gpuC16)

		err := matMulF16(gpu, cPtr, aPtr, bPtr, M, K, N)
		if err != nil {
			fmt.Printf("  [%4dx%4dx%4d] FP16 not available: %v\n", M, K, N, err)
			gpu.Free(gpuA16)
			gpu.Free(gpuB16)
			gpu.Free(gpuC16)
			continue
		}
		syncGPU(gpu)

		// Correctness check
		c16Host := gpuToHostF32(gpu, gpuC16, M*N)
		maxDiff := float32(0)
		for i := range c32Host {
			d := float32(math.Abs(float64(c32Host[i] - c16Host[i])))
			if d > maxDiff {
				maxDiff = d
			}
		}
		// FP16 has less precision, expect larger diffs
		relDiff := maxDiff / absMax(c32Host)

		// Warmup
		matMulF16(gpu, cPtr, aPtr, bPtr, M, K, N)
		syncGPU(gpu)

		start = time.Now()
		for i := 0; i < iters; i++ {
			matMulF16(gpu, cPtr, aPtr, bPtr, M, K, N)
		}
		syncGPU(gpu)
		fp16Time := time.Since(start).Seconds() / float64(iters)
		fp16Tflops := flops / fp16Time / 1e12

		speedup16 := fp16Tflops / fp32Tflops

		// --- FP8 E4M3 benchmark ---
		dataAf8 := f32SliceToF8E4M3Bytes(dataA)
		dataBf8 := f32SliceToF8E4M3Bytes(dataB)

		gpuA8 := hostToGPU(gpu, dataAf8)
		gpuB8 := hostToGPU(gpu, dataBf8)
		gpuC8, _ := gpu.Alloc(M * N * 4) // output FP32

		a8Ptr := devPtr(gpuA8)
		b8Ptr := devPtr(gpuB8)
		c8Ptr := devPtr(gpuC8)

		var fp8Tflops float64
		var speedup8 float64
		var fp8Err string

		err8 := matMulF8E4M3(gpu, c8Ptr, a8Ptr, b8Ptr, M, K, N)
		if err8 != nil {
			fp8Err = fmt.Sprintf("(%v)", err8)
		} else {
			syncGPU(gpu)

			// Correctness
			c8Host := gpuToHostF32(gpu, gpuC8, M*N)
			maxDiff8 := float32(0)
			for i := range c32Host {
				d := float32(math.Abs(float64(c32Host[i] - c8Host[i])))
				if d > maxDiff8 {
					maxDiff8 = d
				}
			}
			relDiff8 := maxDiff8 / absMax(c32Host)
			_ = relDiff8

			// Warmup
			matMulF8E4M3(gpu, c8Ptr, a8Ptr, b8Ptr, M, K, N)
			syncGPU(gpu)

			start = time.Now()
			for i := 0; i < iters; i++ {
				matMulF8E4M3(gpu, c8Ptr, a8Ptr, b8Ptr, M, K, N)
			}
			syncGPU(gpu)
			fp8Time := time.Since(start).Seconds() / float64(iters)
			fp8Tflops = flops / fp8Time / 1e12
			speedup8 = fp8Tflops / fp32Tflops
		}

		if fp8Err != "" {
			fmt.Printf("[%4dx%4dx%4d] | %7.1f TFLOPS | %7.1f TFLOPS |   N/A %-10s| %.2fx  | N/A\n",
				M, K, N, fp32Tflops, fp16Tflops, fp8Err, speedup16)
		} else {
			fmt.Printf("[%4dx%4dx%4d] | %7.1f TFLOPS | %7.1f TFLOPS | %7.1f TFLOPS | %.2fx  | %.2fx\n",
				M, K, N, fp32Tflops, fp16Tflops, fp8Tflops, speedup16, speedup8)
		}

		gpu.Free(gpuA16)
		gpu.Free(gpuB16)
		gpu.Free(gpuC16)
		gpu.Free(gpuA8)
		gpu.Free(gpuB8)
		gpu.Free(gpuC8)
	}

	fmt.Println()
	fmt.Println("=== Benchmark Complete ===")
}

// matMulF16 calls cublasGemmEx through the backend's CuBLAS handle.
// Uses the Launch interface to access MatMulF16.
func matMulF16(b backend.Backend, dstPtr, aPtr, bPtr uintptr, M, K, N int) error {
	type f16Matmuler interface {
		MatMulF16(dstPtr, aPtr, bPtr uintptr, M, K, N int) error
	}
	type cublasAccessor interface {
		CuBLASHandle() interface{}
	}

	// Try direct method on backend
	if mm, ok := b.(f16Matmuler); ok {
		return mm.MatMulF16(dstPtr, aPtr, bPtr, M, K, N)
	}

	// Try through cublas handle
	if ca, ok := b.(cublasAccessor); ok {
		h := ca.CuBLASHandle()
		if mm, ok := h.(f16Matmuler); ok {
			return mm.MatMulF16(dstPtr, aPtr, bPtr, M, K, N)
		}
	}

	return fmt.Errorf("FP16 MatMul not available on this backend")
}

// === FP16 conversion ===

// f32ToF16 converts a float32 to IEEE 754 half-precision (uint16).
func f32ToF16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := (b >> 31) & 1
	exp := int((b>>23)&0xFF) - 127
	frac := b & 0x7FFFFF

	if exp > 15 {
		// Overflow -> Inf
		return uint16(sign<<15 | 0x7C00)
	}
	if exp < -14 {
		// Underflow -> 0 (or subnormal, skip for simplicity)
		return uint16(sign << 15)
	}

	// Normal number
	hexp := uint16(exp+15) & 0x1F
	hfrac := uint16(frac >> 13) // keep top 10 mantissa bits
	return uint16(sign<<15) | (hexp << 10) | hfrac
}

func f32SliceToF16Bytes(data []float32) []byte {
	out := make([]byte, len(data)*2)
	for i, v := range data {
		h := f32ToF16(v)
		binary.LittleEndian.PutUint16(out[i*2:], h)
	}
	return out
}

// f32ToF8E4M3 converts float32 to FP8 E4M3 (1 sign, 4 exp, 3 mantissa, bias=7).
// Range: ±448, precision ~0.125. Clamps to max representable.
func f32ToF8E4M3(f float32) byte {
	b := math.Float32bits(f)
	sign := (b >> 31) & 1
	exp := int((b>>23)&0xFF) - 127
	frac := b & 0x7FFFFF

	// Handle special cases
	if exp > 8 { // overflow -> max value (not inf, E4M3 has no inf)
		return byte(sign<<7 | 0x7E) // ±448
	}
	if exp < -6 { // underflow -> zero
		return byte(sign << 7)
	}
	if (b & 0x7FFFFFFF) == 0 { // zero
		return byte(sign << 7)
	}

	// Normal: rebias exponent (bias 127 -> bias 7)
	hexp := byte(exp+7) & 0x0F
	// Keep top 3 mantissa bits with rounding
	hfrac := byte((frac + (1 << 19)) >> 20) // round to nearest
	if hfrac > 7 {
		hfrac = 0
		hexp++
		if hexp > 15 { // overflow after rounding
			return byte(sign<<7 | 0x7E)
		}
	}
	return byte(sign<<7) | (hexp << 3) | hfrac
}

func f32SliceToF8E4M3Bytes(data []float32) []byte {
	out := make([]byte, len(data))
	for i, v := range data {
		out[i] = f32ToF8E4M3(v)
	}
	return out
}

// matMulF8E4M3 calls cublasGemmEx with FP8 E4M3 inputs.
func matMulF8E4M3(b backend.Backend, dstPtr, aPtr, bPtr uintptr, M, K, N int) error {
	type f8Matmuler interface {
		MatMulF8E4M3(dstPtr, aPtr, bPtr uintptr, M, K, N int) error
	}
	if mm, ok := b.(f8Matmuler); ok {
		return mm.MatMulF8E4M3(dstPtr, aPtr, bPtr, M, K, N)
	}
	return fmt.Errorf("FP8 MatMul not available")
}

// === Helpers ===

func must(err error, msg string) {
	if err != nil {
		panic(fmt.Sprintf("%s: %v", msg, err))
	}
}

func randomF32(n int) []float32 {
	data := make([]float32, n)
	for i := range data {
		data[i] = rand.Float32()*2 - 1
	}
	return data
}

func f32ToBytes(data []float32) []byte {
	b := make([]byte, len(data)*4)
	for i, v := range data {
		bits := math.Float32bits(v)
		b[i*4] = byte(bits)
		b[i*4+1] = byte(bits >> 8)
		b[i*4+2] = byte(bits >> 16)
		b[i*4+3] = byte(bits >> 24)
	}
	return b
}

func bytesToF32(b []byte) []float32 {
	n := len(b) / 4
	data := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := uint32(b[i*4]) | uint32(b[i*4+1])<<8 | uint32(b[i*4+2])<<16 | uint32(b[i*4+3])<<24
		data[i] = math.Float32frombits(bits)
	}
	return data
}

func hostToGPU(b backend.Backend, data []byte) backend.Storage {
	s, err := b.ToDevice(backend.CUDADevice(0), &cpuStorage{data: data})
	must(err, "hostToGPU")
	return s
}

func gpuToHostF32(b backend.Backend, s backend.Storage, n int) []float32 {
	cpuS, err := b.ToDevice(backend.CPU0, s)
	must(err, "gpuToHost")
	return bytesToF32(cpuS.Bytes()[:n*4])
}

func syncGPU(b backend.Backend) {
	type syncer interface{ Sync() error }
	if s, ok := b.(syncer); ok {
		must(s.Sync(), "Sync")
	}
}

func devPtr(s backend.Storage) uintptr {
	type devPtrer interface{ DevicePtr() uintptr }
	if dp, ok := s.(devPtrer); ok {
		return dp.DevicePtr()
	}
	return uintptr(s.Ptr())
}

func absMax(data []float32) float32 {
	m := float32(0)
	for _, v := range data {
		a := float32(math.Abs(float64(v)))
		if a > m {
			m = a
		}
	}
	if m == 0 {
		return 1
	}
	return m
}

type cpuStorage struct{ data []byte }

func (s *cpuStorage) Device() backend.Device { return backend.CPU0 }
func (s *cpuStorage) Ptr() unsafe.Pointer {
	if len(s.data) == 0 {
		return nil
	}
	return unsafe.Pointer(&s.data[0])
}
func (s *cpuStorage) Bytes() []byte { return s.data }
func (s *cpuStorage) ByteLen() int  { return len(s.data) }
func (s *cpuStorage) Free()         { s.data = nil }
