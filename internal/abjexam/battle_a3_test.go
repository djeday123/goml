package abjexam

// A-3 (2026-07-25): prod-Step без test-only gradOW D2H.
// Финальный ход battle-цепочки. SGD verified on GPU (adamw_f32 kernels.go:610).
//
// Тесты:
//   TestA3_Battle_LossOnly     — sanity: 50 шагов, loss траектория (A-3 не возвращает gradOW).
//                                Loss должен bit-exact vs A-2 (та же математика, +1 skip D2H).
//   TestA3_Battle_Speed        — 30-run FA-класс с CV-гейтом (CV>1% = сноска, разбор).
//   TestA3_Battle_CPUMap       — 4-я карта: gradOW D2H устранён, доминанта?

import (
	"math"
	"math/rand"
	"os/exec"
	"sort"
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/djeday123/goml/backend"
	_ "github.com/djeday123/goml/backend/cpu"
	_ "github.com/djeday123/goml/backend/cuda"
	adapter "github.com/djeday123/goml/backend/gotorch"
	"github.com/djeday123/goml/core"
)

func TestA3_Battle_LossOnly(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	cfg := DefaultBattleCfg()
	const steps = 50

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

	// Path A = A-2 (with gradOW D2H).
	lossA, _, _ := runBattleA2(t, adB, cfg, steps)
	t.Logf("Path A (A-2 test-D2H) losses: step1=%.6f step%d=%.6f (Δ=%.4f)",
		lossA[0], steps, lossA[steps-1], lossA[steps-1]-lossA[0])

	// Path B = A-3 (no gradOW D2H).
	rInit := rand.New(rand.NewSource(battleSeedInit))
	stB, err := NewBattleState(cfg, rInit, adB)
	if err != nil {
		t.Fatal(err)
	}
	scratch, err := NewBattleScratch(cfg, adB)
	if err != nil {
		t.Fatal(err)
	}
	defer scratch.FreeAll(adB)
	rBatch := rand.New(rand.NewSource(battleSeedBatches))
	lossB := make([]float64, steps)
	for s := 1; s <= steps; s++ {
		inp, tgt := battleBatch(rBatch, cfg)
		l, err := trainStepBattleA3(adB, stB, scratch, inp, tgt, s)
		if err != nil {
			t.Fatalf("A3 step %d: %v", s, err)
		}
		lossB[s-1] = l
	}
	t.Logf("Path B (A-3 prod no-D2H) losses: step1=%.6f step%d=%.6f (Δ=%.4f)",
		lossB[0], steps, lossB[steps-1], lossB[steps-1]-lossB[0])

	// Bit-exact expected (same math + same gradOW on GPU).
	var worstLoss float64
	for i := 0; i < steps; i++ {
		d := math.Abs(lossA[i] - lossB[i])
		if d > worstLoss {
			worstLoss = d
		}
		if i%10 == 0 || i == steps-1 {
			t.Logf("step %2d: A=%.6f B=%.6f |A-B|=%.3e", i+1, lossA[i], lossB[i], d)
		}
	}
	t.Logf("Loss criterion: worst |A-B| = %.3e (floor 1e-8, bit-exact expected)", worstLoss)
	if worstLoss > 1e-8 {
		t.Errorf("A-3 НЕ bit-exact vs A-2: %.3e > 1e-8", worstLoss)
	}
	if lossB[steps-1] >= lossB[0] {
		t.Errorf("B not descending: %.4f → %.4f", lossB[0], lossB[steps-1])
	}
}

// TestA3_Battle_Speed -- prod-режим 30-run.
// CV-гейт: если > 1% -- сноска в отчёте, диагностика.
func TestA3_Battle_Speed(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	if out, err := exec.Command("nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits").Output(); err == nil {
		t.Logf("nvidia-smi gate: %s", strings.TrimSpace(string(out)))
	}
	if out, err := exec.Command("nvidia-smi", "--query-gpu=clocks.current.sm,temperature.gpu,power.draw", "--format=csv,noheader,nounits").Output(); err == nil {
		t.Logf("clocks before: %s", strings.TrimSpace(string(out)))
	}
	defer func() {
		if out, err := exec.Command("nvidia-smi", "--query-gpu=clocks.current.sm,temperature.gpu,power.draw", "--format=csv,noheader,nounits").Output(); err == nil {
			t.Logf("clocks after: %s", strings.TrimSpace(string(out)))
		}
	}()

	cfg := DefaultBattleCfg()
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

	rInit := rand.New(rand.NewSource(battleSeedInit))
	st, err := NewBattleState(cfg, rInit, adB)
	if err != nil {
		t.Fatal(err)
	}
	scratch, err := NewBattleScratch(cfg, adB)
	if err != nil {
		t.Fatal(err)
	}
	defer scratch.FreeAll(adB)
	rBatch := rand.New(rand.NewSource(battleSeedBatches))

	// Warmup 5
	for s := 1; s <= 5; s++ {
		inp, tgt := battleBatch(rBatch, cfg)
		if _, err := trainStepBattleA3(adB, st, scratch, inp, tgt, s); err != nil {
			t.Fatal(err)
		}
	}
	// Measure 30
	times := make([]float64, 30)
	for i := 0; i < 30; i++ {
		inp, tgt := battleBatch(rBatch, cfg)
		t0 := time.Now()
		if _, err := trainStepBattleA3(adB, st, scratch, inp, tgt, i+6); err != nil {
			t.Fatal(err)
		}
		times[i] = float64(time.Since(t0).Nanoseconds()) / 1e6
	}
	sort.Float64s(times)
	median := times[15]
	var sum, sumsq float64
	for _, v := range times {
		sum += v
		sumsq += v * v
	}
	mean := sum / 30
	variance := sumsq/30 - mean*mean
	cv := math.Sqrt(math.Max(variance, 0)) / mean * 100

	t.Log("=== SPEED A-3 (30-run median, ms/step) ===")
	t.Logf("A3 prod no-D2H           | median=%.3f ms | CV=%.2f%% | speedup vs B2 (4791.5 ms): %.1f×, vs A-2 (18.735 ms): %.2f×",
		median, cv, 4791.5/median, 18.735/median)
	if cv > 1.0 {
		t.Logf("(CV-gate: %.2f%% > 1%% — feedback-cv-gate-strict, число условно-сертифицировано)", cv)
	}
}

// TestA3_Battle_CPUMap -- 4-я карта после устранения gradOW D2H.
func TestA3_Battle_CPUMap(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	cfg := DefaultBattleCfg()
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

	rInit := rand.New(rand.NewSource(battleSeedInit))
	st, err := NewBattleState(cfg, rInit, adB)
	if err != nil {
		t.Fatal(err)
	}
	scratch, err := NewBattleScratch(cfg, adB)
	if err != nil {
		t.Fatal(err)
	}
	defer scratch.FreeAll(adB)
	rBatch := rand.New(rand.NewSource(battleSeedBatches))
	m := cfg.Batch * cfg.Seq

	// warmup 5
	for s := 1; s <= 5; s++ {
		inp, tgt := battleBatch(rBatch, cfg)
		if _, err := trainStepBattleA3(adB, st, scratch, inp, tgt, s); err != nil {
			t.Fatal(err)
		}
	}

	const N = 10
	blockNames := []string{
		"1.  tokens+targets H2D",
		"2.  Embedding GPU",
		"3.  LayerNorm GPU",
		"4.  MatMul-fwd GPU",
		"5.  cross_entropy_f32 GPU",
		"6.  MatMul-bwd GPU (F32Ex)",
		"7.  adamw_f32 GPU",
		"8.  loss[m] D2H (4 KB)",
		"9.  residual (no scratch alloc, no gradOW D2H)",
	}
	blockTimes := make([][]float64, len(blockNames))
	for i := range blockTimes {
		blockTimes[i] = make([]float64, N)
	}
	syncFn := func() {
		if s, ok := adB.(interface{ Sync() error }); ok {
			s.Sync()
		}
	}

	for iter := 0; iter < N; iter++ {
		inp, tgt := battleBatch(rBatch, cfg)
		wallStart := time.Now()

		t0 := time.Now()
		inputGPU, _ := adB.ToDevice(backend.CUDADevice(0), &cpuStorage{data: int64ToBytes(inp)})
		targetsI32 := make([]int32, m)
		for i, v := range tgt {
			targetsI32[i] = int32(v)
		}
		targetsGPU, _ := adB.ToDevice(backend.CUDADevice(0), &cpuStorage{data: int32ToBytesLE(targetsI32)})
		syncFn()
		blockTimes[0][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		t0 = time.Now()
		adB.Embedding(scratch.Embedded, st.EmbW, inputGPU, cfg.Vocab, cfg.Embed, m, core.Float32)
		syncFn()
		blockTimes[1][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		t0 = time.Now()
		adB.LayerNorm(scratch.Normed, scratch.Embedded, st.LNG, st.LNB,
			core.Shape{m, cfg.Embed}, 1, 1e-5, core.Float32)
		syncFn()
		blockTimes[2][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		t0 = time.Now()
		adB.MatMul(scratch.Logits, scratch.Normed, st.OutW,
			core.Shape{m, cfg.Embed}, core.Shape{cfg.Embed, cfg.Vocab}, core.Float32)
		syncFn()
		blockTimes[3][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		t0 = time.Now()
		launchCEF32(t, adB, scratch.Logits, targetsGPU, scratch.Loss, scratch.GradLogits, m, cfg.Vocab)
		blockTimes[4][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		t0 = time.Now()
		if gtB, ok := adB.(*adapter.Backend); ok {
			_ = gtB.MatMulF32Ex(scratch.Normed, scratch.GradLogits, scratch.GradOW,
				cfg.Embed, cfg.Vocab, m, true, false)
		}
		syncFn()
		blockTimes[5][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		t0 = time.Now()
		b1corr := float32(1.0 - math.Pow(float64(Beta1), float64(iter+1)))
		b2corr := float32(1.0 - math.Pow(float64(Beta2), float64(iter+1)))
		nOW := uint32(cfg.Embed * cfg.Vocab)
		owPtr := devPtr(st.OutW)
		gowPtr := devPtr(scratch.GradOW)
		momPtr := devPtr(st.OutMomM)
		vomPtr := devPtr(st.OutMomV)
		lrLoc := LR
		b1Loc := Beta1
		b2Loc := Beta2
		epsLoc := EPS
		wdLoc := WD
		adamParams := []unsafe.Pointer{
			unsafe.Pointer(&owPtr), unsafe.Pointer(&gowPtr),
			unsafe.Pointer(&momPtr), unsafe.Pointer(&vomPtr),
			unsafe.Pointer(&nOW),
			unsafe.Pointer(&lrLoc), unsafe.Pointer(&b1Loc), unsafe.Pointer(&b2Loc),
			unsafe.Pointer(&epsLoc), unsafe.Pointer(&wdLoc),
			unsafe.Pointer(&b1corr), unsafe.Pointer(&b2corr),
		}
		if l, ok := adB.(interface {
			Launch(name string, gx, gy, gz, bx, by, bz uint32, params []unsafe.Pointer) error
		}); ok {
			_ = l.Launch("adamw_f32", gridSize(int(nOW), 256), 1, 1, 256, 1, 1, adamParams)
		}
		syncFn()
		blockTimes[6][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		t0 = time.Now()
		_ = gpuToHost(adB, scratch.Loss, m)
		blockTimes[7][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		adB.Free(inputGPU)
		adB.Free(targetsGPU)

		wallMs := float64(time.Since(wallStart).Nanoseconds()) / 1e6
		sumOther := 0.0
		for i := 0; i < 8; i++ {
			sumOther += blockTimes[i][iter]
		}
		blockTimes[8][iter] = wallMs - sumOther
	}

	mid := func(a []float64) float64 {
		b := make([]float64, len(a))
		copy(b, a)
		sort.Float64s(b)
		return b[len(b)/2]
	}
	total := 0.0
	medians := make([]float64, len(blockNames))
	for i := range blockNames {
		medians[i] = mid(blockTimes[i])
		total += medians[i]
	}

	// sum(blocks) vs wall convergence — публикуем как rule per feedback-cpumap-anomaly.
	sumMeasured := 0.0
	for i := 0; i < 8; i++ {
		sumMeasured += medians[i]
	}
	residual := medians[8]

	t.Log("=== A-3 CPU-MAP — компоненты trainStepBattleA3 prod-Step (median 10 iter) ===")
	t.Logf("%-52s | %10s | %6s", "block", "ms", "share")
	t.Logf("%s", strings.Repeat("-", 80))
	for i, name := range blockNames {
		share := 100.0 * medians[i] / total
		t.Logf("%-52s | %10.3f | %5.1f%%", name, medians[i], share)
	}
	t.Logf("%s", strings.Repeat("-", 80))
	t.Logf("%-52s | %10.3f | 100.0%%", "TOTAL", total)
	t.Logf("Sum(blocks 1-8)=%.3f, residual (block 9)=%.3f, residual/total=%.2f%%",
		sumMeasured, residual, 100.0*residual/total)
}
