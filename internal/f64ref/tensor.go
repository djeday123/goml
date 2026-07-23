// Package f64ref — F64 референс-реализация для судьи impl-5.
//
// Роль: **эталонная линейка** для валидации F32-путей goml/gotorch. НЕ боевой
// код, НЕ производственный backend. Не должен использоваться из train/nn.
// После валидации импл-5 остаётся постоянным инструментом для последующих
// портов ядер в gotorch (impl-6+).
//
// Разделение вычислений:
//   - MatMul → GPU F64 через gotorch.MatMulF64 (быстро, точно, покрытие есть)
//   - Активации (Sigmoid/Tanh/ReLU/SiLU/GeLU) → GPU F64 через gotorch (fdlibm ulp-точность)
//   - Softmax, Sum, Mean по последней оси → GPU F64
//   - LayerNorm, Embedding, RoPE → CPU FP64 (дыры в gotorch F64 покрытии)
//   - Attention SDPA → композиция MatMul(GPU) + Softmax(GPU) + маска(CPU) + MatMul(GPU)
//   - AdamW → CPU (host loop над []float64)
//
// F64Tensor хранит данные как host-side []float64 + shape. Перед GPU-вызовом
// делает Upload (H2D), после — Download (D2H). Не тратим оптимизации, здесь
// точность важнее скорости.
//
// В ARCHITECTURE.md строка про роль f64ref добавляется в impl-5c финальном
// коммите. Правило read-only после импл-5: судья не оптимизируется, только
// расширяется при добавлении новых слоёв (если понадобится).
package f64ref

import (
	"fmt"
	"unsafe"

	gtcuda "github.com/djeday123/gotorch/cuda"
)

// F64Tensor — host-side F64 тензор с shape/dtype метаданными.
type F64Tensor struct {
	Data  []float64
	Shape []int
}

// NewF64Tensor создаёт тензор из данных и shape (проверяет длину).
func NewF64Tensor(data []float64, shape []int) *F64Tensor {
	n := numElements(shape)
	if len(data) != n {
		panic(fmt.Sprintf("f64ref.NewF64Tensor: data length %d != shape elements %d",
			len(data), n))
	}
	// Копируем чтобы владение чисто.
	d := make([]float64, len(data))
	copy(d, data)
	s := make([]int, len(shape))
	copy(s, shape)
	return &F64Tensor{Data: d, Shape: s}
}

// Zeros создаёт нулевой тензор.
func Zeros(shape ...int) *F64Tensor {
	n := numElements(shape)
	s := make([]int, len(shape))
	copy(s, shape)
	return &F64Tensor{Data: make([]float64, n), Shape: s}
}

func (t *F64Tensor) NumElements() int { return numElements(t.Shape) }

// Clone возвращает независимую копию.
func (t *F64Tensor) Clone() *F64Tensor {
	return NewF64Tensor(t.Data, t.Shape)
}

func numElements(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// f64Bytes / bytesF64 — сериализация [float64]↔[byte] little-endian.
// Используется для гибрида gotorch CopyH2D/D2H которые работают с []byte.
func f64Bytes(v []float64) []byte {
	out := make([]byte, 8*len(v))
	for i, x := range v {
		bits := *(*uint64)(unsafe.Pointer(&x))
		out[i*8+0] = byte(bits)
		out[i*8+1] = byte(bits >> 8)
		out[i*8+2] = byte(bits >> 16)
		out[i*8+3] = byte(bits >> 24)
		out[i*8+4] = byte(bits >> 32)
		out[i*8+5] = byte(bits >> 40)
		out[i*8+6] = byte(bits >> 48)
		out[i*8+7] = byte(bits >> 56)
	}
	return out
}

func bytesF64(b []byte) []float64 {
	n := len(b) / 8
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		bits := uint64(b[i*8+0]) |
			uint64(b[i*8+1])<<8 |
			uint64(b[i*8+2])<<16 |
			uint64(b[i*8+3])<<24 |
			uint64(b[i*8+4])<<32 |
			uint64(b[i*8+5])<<40 |
			uint64(b[i*8+6])<<48 |
			uint64(b[i*8+7])<<56
		out[i] = *(*float64)(unsafe.Pointer(&bits))
	}
	return out
}

// ─────────────────────────────────────────────────────────────────
// GPU-мост: одиночная инициализация gotorch backend для f64ref.
// Один backend на процесс — судья не многопоточен, не делит state.
// ─────────────────────────────────────────────────────────────────

var gtBackend gtcuda.Backend

// initGPU лениво инициализирует gotorch backend для F64-GPU операций.
func initGPU() error {
	if gtBackend != nil {
		return nil
	}
	b, err := gtcuda.NewBackend(0)
	if err != nil {
		return fmt.Errorf("f64ref initGPU: %w", err)
	}
	gtBackend = b
	return nil
}

// upload — H2D через gotorch storage. Возвращает Storage; caller обязан Free.
func upload(host []float64) (gtcuda.Storage, error) {
	if err := initGPU(); err != nil {
		return gtcuda.Storage{}, err
	}
	s, err := gtBackend.Alloc(len(host) * 8)
	if err != nil {
		return gtcuda.Storage{}, err
	}
	if err := gtBackend.CopyH2D(s, f64Bytes(host)); err != nil {
		gtBackend.Free(s)
		return gtcuda.Storage{}, err
	}
	return s, nil
}

// download — D2H в новый []float64.
func download(s gtcuda.Storage, n int) ([]float64, error) {
	if err := gtBackend.Sync(); err != nil {
		return nil, err
	}
	buf := make([]byte, n*8)
	if err := gtBackend.CopyD2H(buf, s); err != nil {
		return nil, err
	}
	return bytesF64(buf), nil
}

// MatMulF64GPU — a[M,K] @ b[K,N] = c[M,N] через gotorch.MatMulF64.
// Host↔GPU перенос на каждый вызов; для судьи скорость не критична.
func MatMulF64GPU(a, b *F64Tensor, M, K, N int) *F64Tensor {
	if len(a.Data) != M*K {
		panic(fmt.Sprintf("MatMulF64GPU: A shape mismatch, want %d, got %d", M*K, len(a.Data)))
	}
	if len(b.Data) != K*N {
		panic(fmt.Sprintf("MatMulF64GPU: B shape mismatch, want %d, got %d", K*N, len(b.Data)))
	}
	aS, err := upload(a.Data)
	if err != nil {
		panic(err)
	}
	defer gtBackend.Free(aS)
	bS, err := upload(b.Data)
	if err != nil {
		panic(err)
	}
	defer gtBackend.Free(bS)
	cS, err := gtBackend.Alloc(M * N * 8)
	if err != nil {
		panic(err)
	}
	defer gtBackend.Free(cS)
	if err := gtBackend.MatMulF64(aS, bS, cS, M, N, K); err != nil {
		panic(err)
	}
	data, err := download(cS, M*N)
	if err != nil {
		panic(err)
	}
	return &F64Tensor{Data: data, Shape: []int{M, N}}
}
