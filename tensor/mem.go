package tensor

import "unsafe"

// copySliceToPtr copies a Go slice into raw memory at ptr.
func copySliceToPtr[T any](data []T, ptr uintptr, byteLen int) {
	if len(data) == 0 {
		return
	}
	src := unsafe.Pointer(&data[0])
	dst := unsafe.Pointer(ptr)
	// Use byte-level copy
	srcBytes := unsafe.Slice((*byte)(src), byteLen)
	dstBytes := unsafe.Slice((*byte)(dst), byteLen)
	copy(dstBytes, srcBytes)
}

// SliceFromPtr interprets raw memory as a Go slice. The caller must ensure
// the pointer is valid and the lifetime is managed.
func SliceFromPtr[T any](ptr uintptr, n int) []T {
	if n == 0 {
		return nil
	}
	return unsafe.Slice((*T)(unsafe.Pointer(ptr)), n)
}

// ToFloat32Slice returns the tensor data as []float32.
// Only valid for Float32 tensors on CPU.
func (t *Tensor) ToFloat32Slice() []float32 {
	return SliceFromPtr[float32](t.storage.Ptr(), t.NumElements())
}

// ToFloat64Slice returns the tensor data as []float64.
func (t *Tensor) ToFloat64Slice() []float64 {
	return SliceFromPtr[float64](t.storage.Ptr(), t.NumElements())
}

// ToInt32Slice returns the tensor data as []int32.
func (t *Tensor) ToInt32Slice() []int32 {
	return SliceFromPtr[int32](t.storage.Ptr(), t.NumElements())
}

// ToInt64Slice returns the tensor data as []int64.
func (t *Tensor) ToInt64Slice() []int64 {
	return SliceFromPtr[int64](t.storage.Ptr(), t.NumElements())
}
