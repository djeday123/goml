package abjexam

// A-LLM-6 П.2а: репро amax-гонки ЧИСЛОМ (хвост-а A-LLM-5).
// 10x цикл [GPU-активность -> zero-upload amax -> Unit-quantize -> чтение amax]
// в нагруженном контексте. ПРОГНОЗ (до прогона): >=1/10 amax==0
// (в A-LLM-5 стреляло ~50% прогонов процесса при живом chain-вызове).
// После фикса (если потребуется): 10/10 живых.

import (
	"math/rand"
	"testing"

	"github.com/djeday123/goml/backend"
	_ "github.com/djeday123/goml/backend/cpu"
	_ "github.com/djeday123/goml/backend/cuda"
	adapter "github.com/djeday123/goml/backend/gotorch"
	"github.com/djeday123/goml/core"
)

func TestALLM_AmaxRaceRepro(t *testing.T) {
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
	gtB := adB.(*adapter.Backend)

	const n = 4 * 2048 * 128 // форма A/B блока
	r := rand.New(rand.NewSource(11))
	host := make([]float32, n)
	for i := range host {
		host[i] = float32(r.NormFloat64() * 0.5)
	}
	src, err := adB.ToDevice(backend.CUDADevice(0), &cpuStorage{data: f32ToBytes(host)})
	if err != nil {
		t.Fatalf("upload: %v", err)
	}
	defer adB.Free(src)
	dst, _ := adB.Alloc(n)
	defer adB.Free(dst)
	scale, _ := adB.Alloc(4)
	defer adB.Free(scale)
	amax, _ := adB.Alloc(4)
	defer adB.Free(amax)
	// Нагрузочные матрицы для фоновой активности (класс Stage1-контекста).
	mA, _ := adB.Alloc(512 * 512 * 4)
	defer adB.Free(mA)
	mB, _ := adB.Alloc(512 * 512 * 4)
	defer adB.Free(mB)
	mC, _ := adB.Alloc(512 * 512 * 4)
	defer adB.Free(mC)

	sync := func() {
		if s, ok := adB.(interface{ Sync() error }); ok {
			s.Sync()
		}
	}
	zeroF := []byte{0, 0, 0, 0}
	dead := 0
	for round := 1; round <= 10; round++ {
		// Фоновая активность БЕЗ sync (как в bwd-цепочке перед блоком).
		for k := 0; k < 5; k++ {
			adB.MatMul(mC, mA, mB, core.Shape{512, 512}, core.Shape{512, 512}, core.Float32)
		}
		// Точный паттерн attnFABwdBlock (a): zero-upload x3 -> Sync -> quantize.
		uploadInto(adB, amax, zeroF)
		uploadInto(adB, scale, zeroF)
		sync() // рейс-фикс A-LLM-5 (тот же, что в блоке)
		if err := gtB.QuantizeF32ToF8E4M3Unit(src, dst, scale, amax, n); err != nil {
			t.Fatalf("round %d quantize: %v", round, err)
		}
		sync()
		a := gpuToHost(adB, amax, 1)[0]
		status := "ЖИВ"
		if a == 0 {
			status = "МЁРТВ (amax=0)"
			dead++
		}
		t.Logf("П.2а round %2d: amax=%.6e host-ref=%.6e -> %s", round, a, hostAbsMax(host), status)
	}
	t.Logf("П.2а итог: мёртвых %d/10 (прогноз до прогона: >=1 при живой гонке; 0 = гонка закрыта Н3-Sync'ом A-LLM-5)", dead)
	if dead > 0 {
		t.Errorf("amax-гонка воспроизведена: %d/10", dead)
	}
}

func hostAbsMax(h []float32) float32 {
	var mx float32
	for _, v := range h {
		if v < 0 {
			v = -v
		}
		if v > mx {
			mx = v
		}
	}
	return mx
}
