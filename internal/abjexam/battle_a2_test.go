package abjexam

// A-2 (2026-07-24): buffer cache — устраняет alloc/free 6 больших буферов из hot loop.
//
// Тесты:
//   TestA2_Battle_Accuracy — A/B 50 шагов A=A-1, B=A-2. Прогноз bit-exact (та же
//     математика, только стабильные адреса; НЕ bit-exact = скрытая зависимость от
//     свежести памяти, СТОП разбор).
//   TestA2_Battle_Speed    — 30-run FA-класс. Прогноз ~18.8 мс (alloc 0.93 мс уходит).
//   TestA2_Battle_CPUMap   — новая карта, ожидаем alloc→0.

import (
	"fmt"
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

// runBattleA2 -- 50 шагов через trainStepBattleA2 с pre-allocated scratch.
func runBattleA2(t *testing.T, b backend.Backend, cfg BattleCfg, steps int) (losses []float64, grad1, gradLast []float32) {
	t.Helper()
	rInit := rand.New(rand.NewSource(battleSeedInit))
	st, err := NewBattleState(cfg, rInit, b)
	if err != nil {
		t.Fatalf("A2 NewBattleState: %v", err)
	}
	scratch, err := NewBattleScratch(cfg, b)
	if err != nil {
		t.Fatalf("A2 NewBattleScratch: %v", err)
	}
	defer scratch.FreeAll(b)
	rBatch := rand.New(rand.NewSource(battleSeedBatches))
	losses = make([]float64, steps)
	grad1 = make([]float32, cfg.Embed*cfg.Vocab)
	gradLast = make([]float32, cfg.Embed*cfg.Vocab)
	for s := 1; s <= steps; s++ {
		inp, tgt := battleBatch(rBatch, cfg)
		l, g, err := trainStepBattleA2(b, st, scratch, inp, tgt, s)
		if err != nil {
			t.Fatalf("A2 step %d: %v", s, err)
		}
		losses[s-1] = l
		if s == 1 {
			copy(grad1, g)
		}
		if s == steps {
			copy(gradLast, g)
		}
	}
	return
}

func TestA2_Battle_Accuracy(t *testing.T) {
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

	t.Logf("A-2 accuracy exam: Vocab=%d Embed=%d Seq=%d Batch=%d, steps=%d",
		cfg.Vocab, cfg.Embed, cfg.Seq, cfg.Batch, steps)
	// Path A = A-1 CE-fused (baseline).
	lossA, gradA1, gradALast := runBattleA1(t, adB, cfg, steps)
	t.Logf("Path A (A-1 CE-fused) losses: step1=%.6f step%d=%.6f (Δ=%.4f)",
		lossA[0], steps, lossA[steps-1], lossA[steps-1]-lossA[0])
	// Path B = A-2 buffer cache.
	lossB, gradB1, gradBLast := runBattleA2(t, adB, cfg, steps)
	t.Logf("Path B (A-2 buffer-cache) losses: step1=%.6f step%d=%.6f (Δ=%.4f)",
		lossB[0], steps, lossB[steps-1], lossB[steps-1]-lossB[0])

	// Bit-exact expected.
	t.Log("=== Loss comparison (samples every 5 steps) ===")
	t.Log("step | A A-1       | B A-2       | |A-B|")
	var worstLoss float64
	for i := 0; i < steps; i++ {
		d := math.Abs(lossA[i] - lossB[i])
		if d > worstLoss {
			worstLoss = d
		}
		if i%5 == 0 || i == steps-1 {
			t.Logf("%4d | %11.6f | %11.6f | %.3e", i+1, lossA[i], lossB[i], d)
		}
	}
	if lossA[steps-1] >= lossA[0] {
		t.Errorf("A not descending: %.4f → %.4f", lossA[0], lossA[steps-1])
	}
	if lossB[steps-1] >= lossB[0] {
		t.Errorf("B not descending: %.4f → %.4f", lossB[0], lossB[steps-1])
	}

	// Pre-registered floor 0 (bit-exact prognosis).
	// Realistically FP arithmetic gives 0 diff since same math + same buffer contents
	// after zero-init (all overwrites). Allow tiny FP noise 1e-8 as safety.
	const lossFloor = 1e-8
	t.Logf("Loss criterion: worst |A-B| = %.3e (floor %g, bit-exact expected)", worstLoss, lossFloor)
	if worstLoss > lossFloor {
		t.Errorf("(!!) A-2 НЕ bit-exact vs A-1: worst=%.3e > floor %g. Возможна скрытая зависимость от свежести памяти -- РАЗБОР.", worstLoss, lossFloor)
	}

	// Grad hybrid (should be exact zeros or tiny).
	auditGradA2 := func(tag string, got, ref []float32) {
		var maxAbs float64
		nonZeroDiffs := 0
		for i := range got {
			d := math.Abs(float64(got[i]) - float64(ref[i]))
			if d > maxAbs {
				maxAbs = d
			}
			if d > 0 {
				nonZeroDiffs++
			}
		}
		t.Logf("grad %s: maxAbs=%.3e non-zero-diffs=%d/%d", tag, maxAbs, nonZeroDiffs, len(got))
		if maxAbs > 1e-6 {
			t.Errorf("grad %s not bit-exact: maxAbs=%.3e", tag, maxAbs)
		}
	}
	auditGradA2("step 1", gradB1, gradA1)
	auditGradA2(fmt.Sprintf("step %d", steps), gradBLast, gradALast)
}

func TestA2_Battle_Speed(t *testing.T) {
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
		if _, _, err := trainStepBattleA2(adB, st, scratch, inp, tgt, s); err != nil {
			t.Fatal(err)
		}
	}
	// Measure 30
	times := make([]float64, 30)
	for i := 0; i < 30; i++ {
		inp, tgt := battleBatch(rBatch, cfg)
		t0 := time.Now()
		if _, _, err := trainStepBattleA2(adB, st, scratch, inp, tgt, i+6); err != nil {
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

	t.Log("=== SPEED A-2 (30-run median, ms/step) ===")
	t.Logf("A2 buffer-cache          | median=%.3f ms | CV=%.2f%% | speedup vs B2 (4791.5 ms): %.1f×, vs A-1 (19.4 ms): %.2f×",
		median, cv, 4791.5/median, 19.414/median)
	if cv > 5.0 {
		t.Logf("(warn) A2 CV=%.2f%% > 5%%", cv)
	}
}

// TestA2_Battle_CPUMap -- новая карта после устранения 6 allocs/frees.
func TestA2_Battle_CPUMap(t *testing.T) {
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
		if _, _, err := trainStepBattleA2(adB, st, scratch, inp, tgt, s); err != nil {
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
		"9.  gradOW D2H (32 MB, test-only)",
		"10. residual overhead (no scratch alloc/free)",
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

		// 1. tokens + targets H2D (per-step small)
		t0 := time.Now()
		inputGPU, _ := adB.ToDevice(backend.CUDADevice(0), &cpuStorage{data: int64ToBytes(inp)})
		targetsI32 := make([]int32, m)
		for i, v := range tgt {
			targetsI32[i] = int32(v)
		}
		targetsGPU, _ := adB.ToDevice(backend.CUDADevice(0), &cpuStorage{data: int32ToBytesLE(targetsI32)})
		syncFn()
		blockTimes[0][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		// 2. Embedding (scratch)
		t0 = time.Now()
		adB.Embedding(scratch.Embedded, st.EmbW, inputGPU, cfg.Vocab, cfg.Embed, m, core.Float32)
		syncFn()
		blockTimes[1][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		// 3. LayerNorm (scratch)
		t0 = time.Now()
		adB.LayerNorm(scratch.Normed, scratch.Embedded, st.LNG, st.LNB,
			core.Shape{m, cfg.Embed}, 1, 1e-5, core.Float32)
		syncFn()
		blockTimes[2][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		// 4. MatMul fwd (scratch)
		t0 = time.Now()
		adB.MatMul(scratch.Logits, scratch.Normed, st.OutW,
			core.Shape{m, cfg.Embed}, core.Shape{cfg.Embed, cfg.Vocab}, core.Float32)
		syncFn()
		blockTimes[3][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		// 5. CE fused (scratch)
		t0 = time.Now()
		launchCEF32(t, adB, scratch.Logits, targetsGPU, scratch.Loss, scratch.GradLogits, m, cfg.Vocab)
		blockTimes[4][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		// 6. MatMul bwd (scratch)
		t0 = time.Now()
		if gtB, ok := adB.(*adapter.Backend); ok {
			_ = gtB.MatMulF32Ex(scratch.Normed, scratch.GradLogits, scratch.GradOW,
				cfg.Embed, cfg.Vocab, m, true, false)
		}
		syncFn()
		blockTimes[5][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		// 7. adamw
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

		// 8. loss D2H
		t0 = time.Now()
		_ = gpuToHost(adB, scratch.Loss, m)
		blockTimes[7][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		// 9. gradOW D2H
		t0 = time.Now()
		_ = gpuToHost(adB, scratch.GradOW, cfg.Embed*cfg.Vocab)
		blockTimes[8][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		// Free per-step allocs
		adB.Free(inputGPU)
		adB.Free(targetsGPU)

		wallMs := float64(time.Since(wallStart).Nanoseconds()) / 1e6
		sumOther := 0.0
		for i := 0; i < 9; i++ {
			sumOther += blockTimes[i][iter]
		}
		blockTimes[9][iter] = wallMs - sumOther
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

	t.Log("=== A-2 CPU-MAP — компоненты trainStepBattleA2 (scratch, median 10 iter) ===")
	t.Logf("%-46s | %10s | %6s", "block", "ms", "share")
	t.Logf("%s", strings.Repeat("-", 74))
	for i, name := range blockNames {
		share := 100.0 * medians[i] / total
		t.Logf("%-46s | %10.3f | %5.1f%%", name, medians[i], share)
	}
	t.Logf("%s", strings.Repeat("-", 74))
	t.Logf("%-46s | %10.3f | 100.0%%", "TOTAL", total)
}
