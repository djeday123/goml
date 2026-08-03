package abjexam

// A-LLM-6 П.1а: диагноз slice-канала ДО фикса (хвост-б A-LLM-5).
// Подозрение (записано до пробы): adapter ToDevice(CPU0, foreign-slice)
// теряет данные — класс Н1 (стык Storage-типов).
// Три операции с отдельным вердиктом каждой.

import (
	"math"
	"math/rand"
	"testing"

	"github.com/djeday123/goml/backend"
	_ "github.com/djeday123/goml/backend/cpu"
	_ "github.com/djeday123/goml/backend/cuda"
	adapter "github.com/djeday123/goml/backend/gotorch"
	"github.com/djeday123/goml/core"
)

func TestALLM_SliceChannelDiag(t *testing.T) {
	gomlB, err := backend.Get(backend.CUDA)
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	if s, err := gomlB.Alloc(4); err == nil {
		gomlB.Free(s)
	}
	if err := adapter.Enable(); err != nil {
		t.Fatalf("adapter Enable: %v", err)
	}
	adB, _ := backend.Get(backend.CUDA)

	// Полный буфер [4][8x8] f32 с паттерном; срез = батч 2.
	const B, R, C = 4, 8, 8
	n := B * R * C
	r := rand.New(rand.NewSource(42))
	host := make([]float32, n)
	for i := range host {
		host[i] = float32(r.NormFloat64())
	}
	full, err := adB.ToDevice(backend.CUDADevice(0), &cpuStorage{data: f32ToBytes(host)})
	if err != nil {
		t.Fatalf("upload full: %v", err)
	}
	defer adB.Free(full)
	off := 2 * R * C // батч 2
	sl := &sliceStore{ptr: devPtr(full) + uintptr(off*4), byteLen: R * C * 4}
	want := host[off : off+R*C]

	// (i) gpuToHost(adapter, slice)
	got := gpuToHost(adB, sl, R*C)
	var w1 float64
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d > w1 {
			w1 = d
		}
	}
	t.Logf("П.1а(i) gpuToHost(slice): max|Δ|=%.3e -> %s", w1, map[bool]string{true: "ПАТТЕРН (жив)", false: "МЁРТВ"}[w1 == 0])

	// (ii) b.MatMul со slice-входом: I @ slice == slice.
	eye := make([]float32, R*R)
	for i := 0; i < R; i++ {
		eye[i*R+i] = 1.0
	}
	eyeG, _ := adB.ToDevice(backend.CUDADevice(0), &cpuStorage{data: f32ToBytes(eye)})
	defer adB.Free(eyeG)
	out, _ := adB.Alloc(R * C * 4)
	defer adB.Free(out)
	if err := adB.MatMul(out, eyeG, sl, core.Shape{R, R}, core.Shape{R, C}, core.Float32); err != nil {
		t.Logf("П.1а(ii) b.MatMul(slice): ERROR %v", err)
	} else {
		if s, ok := adB.(interface{ Sync() error }); ok {
			s.Sync()
		}
		got2 := gpuToHost(adB, out, R*C)
		var w2 float64
		for i := range want {
			if d := math.Abs(float64(got2[i] - want[i])); d > w2 {
				w2 = d
			}
		}
		t.Logf("П.1а(ii) b.MatMul(I, slice): max|Δ|=%.3e -> %s", w2, map[bool]string{true: "ПРОИЗВЕДЕНИЕ (жив)", false: "МЁРТВ"}[w2 < 1e-6])
	}

	// (iii) b.Copy(tmp, slice) + gpuToHost(tmp)
	tmp, _ := adB.Alloc(R * C * 4)
	defer adB.Free(tmp)
	if err := adB.Copy(tmp, sl, R*C*4); err != nil {
		t.Logf("П.1а(iii) b.Copy(slice): ERROR %v", err)
	} else {
		if s, ok := adB.(interface{ Sync() error }); ok {
			s.Sync()
		}
		got3 := gpuToHost(adB, tmp, R*C)
		var w3 float64
		for i := range want {
			if d := math.Abs(float64(got3[i] - want[i])); d > w3 {
				w3 = d
			}
		}
		t.Logf("П.1а(iii) b.Copy(tmp,slice)+gpuToHost(tmp): max|Δ|=%.3e -> %s", w3, map[bool]string{true: "ЖИВ", false: "МЁРТВ"}[w3 == 0])
	}
}
