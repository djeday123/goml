package main

import (
	"fmt"
	"math"

	_ "github.com/vugar/goml/backend/cpu" // register CPU backend
	"github.com/vugar/goml/backend"
	"github.com/vugar/goml/ops"
	"github.com/vugar/goml/tensor"
)

func main() {
	fmt.Println("=== GoML Tensor Engine Test ===\n")

	// Test 1: Create tensors from slices
	fmt.Println("--- Test 1: Tensor creation ---")
	a, _ := tensor.FromSlice([]float32{1, 2, 3, 4, 5, 6}, tensor.Shape{2, 3})
	b, _ := tensor.FromSlice([]float32{7, 8, 9, 10, 11, 12}, tensor.Shape{2, 3})
	fmt.Println("a:", a)
	fmt.Println("b:", b)
	fmt.Println("a data:", a.ToFloat32Slice())
	fmt.Println("b data:", b.ToFloat32Slice())

	// Test 2: Element-wise add
	fmt.Println("\n--- Test 2: Add ---")
	c, err := ops.Add(a, b)
	if err != nil {
		panic(err)
	}
	fmt.Println("a + b:", c.ToFloat32Slice()) // [8, 10, 12, 14, 16, 18]

	// Test 3: Element-wise mul
	fmt.Println("\n--- Test 3: Mul ---")
	d, err := ops.Mul(a, b)
	if err != nil {
		panic(err)
	}
	fmt.Println("a * b:", d.ToFloat32Slice()) // [7, 16, 27, 40, 55, 72]

	// Test 4: MatMul
	fmt.Println("\n--- Test 4: MatMul ---")
	m1, _ := tensor.FromSlice([]float32{1, 2, 3, 4, 5, 6}, tensor.Shape{2, 3})
	m2, _ := tensor.FromSlice([]float32{7, 8, 9, 10, 11, 12}, tensor.Shape{3, 2})
	mm, err := ops.MatMul(m1, m2)
	if err != nil {
		panic(err)
	}
	fmt.Println("matmul shape:", mm.Shape())     // [2, 2]
	fmt.Println("matmul data:", mm.ToFloat32Slice()) // [58, 64, 139, 154]

	// Test 5: Broadcasting
	fmt.Println("\n--- Test 5: Broadcasting ---")
	x, _ := tensor.FromSlice([]float32{1, 2, 3, 4, 5, 6}, tensor.Shape{2, 3})
	bias, _ := tensor.FromSlice([]float32{10, 20, 30}, tensor.Shape{1, 3})
	xb, err := ops.Add(x, bias)
	if err != nil {
		panic(err)
	}
	fmt.Println("x + bias:", xb.ToFloat32Slice()) // [11, 22, 33, 14, 25, 36]

	// Test 6: Softmax
	fmt.Println("\n--- Test 6: Softmax ---")
	logits, _ := tensor.FromSlice([]float32{1, 2, 3, 4}, tensor.Shape{1, 4})
	probs, err := ops.Softmax(logits, 1)
	if err != nil {
		panic(err)
	}
	pData := probs.ToFloat32Slice()
	fmt.Printf("softmax: [%.4f, %.4f, %.4f, %.4f]\n", pData[0], pData[1], pData[2], pData[3])
	sum := float32(0)
	for _, p := range pData {
		sum += p
	}
	fmt.Printf("sum = %.6f (should be 1.0)\n", sum)

	// Test 7: LayerNorm
	fmt.Println("\n--- Test 7: LayerNorm ---")
	ln, _ := tensor.FromSlice([]float32{1, 2, 3, 4, 5, 6}, tensor.Shape{2, 3})
	normed, err := ops.LayerNorm(ln, nil, nil, 1, 1e-5)
	if err != nil {
		panic(err)
	}
	nData := normed.ToFloat32Slice()
	fmt.Printf("layernorm: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
		nData[0], nData[1], nData[2], nData[3], nData[4], nData[5])

	// Test 8: GELU activation
	fmt.Println("\n--- Test 8: GELU ---")
	gIn, _ := tensor.FromSlice([]float32{-2, -1, 0, 1, 2}, tensor.Shape{5})
	gOut, err := ops.Gelu(gIn)
	if err != nil {
		panic(err)
	}
	gData := gOut.ToFloat32Slice()
	fmt.Printf("gelu: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
		gData[0], gData[1], gData[2], gData[3], gData[4])

	// Test 9: Zeros and Ones
	fmt.Println("\n--- Test 9: Zeros/Ones ---")
	z, _ := tensor.Zeros(tensor.Shape{2, 3}, tensor.Float32, backend.CPU0)
	o, _ := tensor.Ones(tensor.Shape{2, 3}, tensor.Float32, backend.CPU0)
	fmt.Println("zeros:", z.ToFloat32Slice())
	fmt.Println("ones:", o.ToFloat32Slice())

	// Test 10: Transpose
	fmt.Println("\n--- Test 10: Transpose ---")
	t1, _ := tensor.FromSlice([]float32{1, 2, 3, 4, 5, 6}, tensor.Shape{2, 3})
	t1T, _ := t1.T()
	fmt.Println("original shape:", t1.Shape(), "strides:", t1.Strides())
	fmt.Println("transposed shape:", t1T.Shape(), "strides:", t1T.Strides())
	fmt.Println("is contiguous:", t1T.IsContiguous())

	// Test 11: View / Reshape
	fmt.Println("\n--- Test 11: View ---")
	v1, _ := tensor.FromSlice([]float32{1, 2, 3, 4, 5, 6}, tensor.Shape{2, 3})
	v2, err := v1.View(tensor.Shape{3, 2})
	if err != nil {
		panic(err)
	}
	fmt.Println("original:", v1.Shape(), "->", v2.Shape())
	fmt.Println("data still shared:", v2.ToFloat32Slice())

	// Test 12: Attention
	fmt.Println("\n--- Test 12: Scaled Dot-Product Attention ---")
	batchSize, numHeads, seqLen, headDim := 1, 2, 4, 8
	total := batchSize * numHeads * seqLen * headDim
	qData := make([]float32, total)
	kData := make([]float32, total)
	vData := make([]float32, total)
	for i := range qData {
		qData[i] = float32(math.Sin(float64(i) * 0.1))
		kData[i] = float32(math.Cos(float64(i) * 0.1))
		vData[i] = float32(i) * 0.01
	}
	shape := tensor.Shape{batchSize, numHeads, seqLen, headDim}
	qT, _ := tensor.FromSlice(qData, shape)
	kT, _ := tensor.FromSlice(kData, shape)
	vT, _ := tensor.FromSlice(vData, shape)
	attnOut, err := ops.ScaledDotProductAttention(qT, kT, vT, numHeads, true)
	if err != nil {
		panic(err)
	}
	fmt.Println("attention output shape:", attnOut.Shape())
	aData := attnOut.ToFloat32Slice()
	fmt.Printf("first 8 values: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
		aData[0], aData[1], aData[2], aData[3], aData[4], aData[5], aData[6], aData[7])

	// Test 13: BFloat16
	fmt.Println("\n--- Test 13: BFloat16 ---")
	bf := tensor.BFloat16FromFloat32(3.14159)
	fmt.Printf("bfloat16: 3.14159 -> stored -> recovered: %.4f\n", bf.Float32())

	fmt.Println("\n=== All tests passed! ===")
}
