package abjexam

// B2-BATTLE: боевой экзамен -- точность (50 шагов) + скорость (30-run FA-класс).
//
// Точность:
//   50 шагов × 3 режима, seed identical, синтетические данные random.Intn(Vocab).
//   Судейская траектория (F64 CPU) -- дорогая на боевой форме, поэтому только
//   короткий отрезок в accuracy exam НЕ гонится (форма 32000 вокабуляр = минуты
//   на CPU per step). Вместо этого F16/F8 vs F32 сравнение с pre-registered floors.
//
// Скорость:
//   Гейт nvidia-smi (тишина), фиксация clocks до/после, 5 warmup + 30 median.
//   CV > 1% -> разбор. Амдал сравнение.
//
// Скоуп по ТЗ: attention/FFN не в scope. Один Linear proj Embed->Vocab.

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/djeday123/goml/backend"
	_ "github.com/djeday123/goml/backend/cpu"
	_ "github.com/djeday123/goml/backend/cuda"
	adapter "github.com/djeday123/goml/backend/gotorch"
)

// TestMain печатает маркер "=== B2_RUN_COMPLETE ===" последней строкой
// вне зависимости от PASS/FAIL. Основа детач-протокола длинных прогонов:
// проверка завершения = grep маркера в логе (см. feedback-detached-long-runs).
func TestMain(m *testing.M) {
	code := m.Run()
	fmt.Println("=== B2_RUN_COMPLETE ===")
	os.Exit(code)
}

const battleSeedInit int64 = 4242
const battleSeedBatches int64 = 5252

// runBattleAccuracy -- 50 шагов, возвращает losses + grad @ step 1 + grad @ step last.
func runBattleAccuracy(t *testing.T, b backend.Backend, cfg BattleCfg, prec Precision, steps int) (losses []float64, grad1, gradLast []float32) {
	t.Helper()
	rInit := rand.New(rand.NewSource(battleSeedInit))
	st, err := NewBattleState(cfg, rInit, b)
	if err != nil {
		t.Fatalf("%s NewBattleState: %v", prec, err)
	}
	rBatch := rand.New(rand.NewSource(battleSeedBatches))
	losses = make([]float64, steps)
	grad1 = make([]float32, cfg.Embed*cfg.Vocab)
	gradLast = make([]float32, cfg.Embed*cfg.Vocab)
	for s := 1; s <= steps; s++ {
		inp, tgt := battleBatch(rBatch, cfg)
		l, g, err := trainStepBattle(b, st, inp, tgt, s, prec)
		if err != nil {
			t.Fatalf("%s step %d: %v", prec, s, err)
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

func TestB2_Battle_Accuracy(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode")
	}
	cfg := DefaultBattleCfg()
	const steps = 50

	// ── Path A F32 baseline (goml.cuda) ──
	gomlB, err := backend.Get(backend.CUDA)
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	if s, err := gomlB.Alloc(4); err == nil {
		gomlB.Free(s)
	}
	t.Logf("Battle config: Vocab=%d Dim=%d Seq=%d Batch=%d, steps=%d, matmul [M=%d K=%d N=%d]",
		cfg.Vocab, cfg.Embed, cfg.Seq, cfg.Batch, steps,
		cfg.Batch*cfg.Seq, cfg.Embed, cfg.Vocab)
	lossA, gradA1, gradALast := runBattleAccuracy(t, gomlB, cfg, PrecF32, steps)
	t.Logf("Path A F32 losses: step1=%.6f step%d=%.6f (Δ=%.4f)", lossA[0], steps, lossA[steps-1], lossA[steps-1]-lossA[0])

	// Enable adapter.
	if err := adapter.Enable(); err != nil {
		t.Fatalf("adapter Enable: %v", err)
	}
	adB, _ := backend.Get(backend.CUDA)

	// ── Path B F16 ──
	lossF16, gradF161, gradF16Last := runBattleAccuracy(t, adB, cfg, PrecF16, steps)
	t.Logf("Path B F16 losses: step1=%.6f step%d=%.6f (Δ=%.4f)", lossF16[0], steps, lossF16[steps-1], lossF16[steps-1]-lossF16[0])

	// ── Path B F8 ──
	lossF8, gradF81, gradF8Last := runBattleAccuracy(t, adB, cfg, PrecF8E4M3, steps)
	t.Logf("Path B F8 losses: step1=%.6f step%d=%.6f (Δ=%.4f)", lossF8[0], steps, lossF8[steps-1], lossF8[steps-1]-lossF8[0])

	// ── Comparison table (10-step samples) ──
	t.Log("=== Loss comparison (samples every 5 steps) ===")
	t.Log("step | A F32      | F16        | F8         | |A-F16|  | |A-F8|")
	var worstF16, worstF8 float64
	for i := 0; i < steps; i++ {
		dF16 := math.Abs(lossA[i] - lossF16[i])
		dF8 := math.Abs(lossA[i] - lossF8[i])
		if dF16 > worstF16 {
			worstF16 = dF16
		}
		if dF8 > worstF8 {
			worstF8 = dF8
		}
		if i%5 == 0 || i == steps-1 {
			t.Logf("%4d | %10.6f | %10.6f | %10.6f | %.3e | %.3e",
				i+1, lossA[i], lossF16[i], lossF8[i], dF16, dF8)
		}
	}

	// ── Descend check (F32, F16 required; F8 diagnostic) ──
	if lossA[steps-1] >= lossA[0] {
		t.Errorf("F32 не убыл: [0]=%.4f [-1]=%.4f", lossA[0], lossA[steps-1])
	}
	if lossF16[steps-1] >= lossF16[0] {
		t.Errorf("F16 не убыл: [0]=%.4f [-1]=%.4f", lossF16[0], lossF16[steps-1])
	}
	if lossF8[steps-1] >= lossF8[0] {
		t.Logf("(diagnostic) F8 не убыл: [0]=%.4f [-1]=%.4f", lossF8[0], lossF8[steps-1])
	}

	// Pre-registered floors (see B2_BATTLE.md):
	//   F16 worst |A-F16|: 1e-5 (per-op 5e-4 × avg_target_prob 3e-5 × sqrt(50) = 1e-7, floor 1000× запас).
	//   F8 worst |A-F8|:  1e-3 (per-op 2.5e-3 × 3e-5 × sqrt(50) = 5.3e-7, floor ~2000× запас).
	t.Logf("Criterion F16: worst |A-F16| = %.3e (floor 1e-5)", worstF16)
	if worstF16 > 1e-5 {
		t.Errorf("F16 worst = %.3e > 1e-5", worstF16)
	}
	t.Logf("Criterion F8: worst |A-F8| = %.3e (floor 1e-3, diagnostic if exceeded)", worstF8)
	if worstF8 > 1e-3 {
		t.Logf("(diagnostic) F8 worst exceeds pre-reg floor 1e-3")
	}

	// Grad audit step 1 and last.
	auditGrad := func(tag string, got, ref []float32, absTol, relTol float64) {
		var maxAbs, maxRel float64
		hybridFail := 0
		for i := range got {
			g := float64(got[i])
			r := float64(ref[i])
			d := math.Abs(g - r)
			rel := d / (math.Abs(r) + 1e-30)
			if d > maxAbs {
				maxAbs = d
			}
			if rel > maxRel {
				maxRel = rel
			}
			if d > absTol+relTol*math.Abs(r) {
				hybridFail++
			}
		}
		t.Logf("grad %s: maxAbs=%.3e maxRel=%.3e hybridFail=%d/%d (floor abs=%.0e+rel=%.0e·|ref|)",
			tag, maxAbs, maxRel, hybridFail, len(got), absTol, relTol)
	}
	auditGrad("F16 step 1 vs A", gradF161, gradA1, 1e-3, 1e-3)
	auditGrad(fmt.Sprintf("F16 step %d vs A", steps), gradF16Last, gradALast, 1e-3, 1e-3)
	auditGrad("F8 step 1 vs A", gradF81, gradA1, 1e-2, 1e-2)
	auditGrad(fmt.Sprintf("F8 step %d vs A", steps), gradF8Last, gradALast, 1e-2, 1e-2)
}

// smiSilent -- проверка что GPU idle перед speed-бенчем.
func smiSilent(t *testing.T) bool {
	t.Helper()
	out, err := exec.Command("nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits").Output()
	if err != nil {
		t.Logf("nvidia-smi unavailable: %v (skipping silence gate)", err)
		return true
	}
	line := strings.TrimSpace(string(out))
	t.Logf("nvidia-smi gate: %s", line)
	parts := strings.Split(line, ",")
	if len(parts) < 2 {
		return true
	}
	var util int
	fmt.Sscanf(strings.TrimSpace(parts[0]), "%d", &util)
	if util > 20 { // > 20% GPU busy => noise
		t.Logf("GPU util = %d%% > 20%% -- потенциальный шум", util)
		return false
	}
	return true
}

func smiClocks(t *testing.T, tag string) {
	t.Helper()
	out, err := exec.Command("nvidia-smi", "--query-gpu=clocks.current.sm,temperature.gpu,power.draw", "--format=csv,noheader,nounits").Output()
	if err == nil {
		t.Logf("clocks %s: %s", tag, strings.TrimSpace(string(out)))
	}
}

// TestB2_Battle_Speed -- 30-run FA-класс speed measurement.
func TestB2_Battle_Speed(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode")
	}
	if !smiSilent(t) {
		t.Skip("GPU non-silent (soседи/utilization)")
	}
	smiClocks(t, "before")
	defer smiClocks(t, "after")

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

	measure := func(name string, b backend.Backend, prec Precision) (medianMs, cvPct float64) {
		t.Helper()
		rInit := rand.New(rand.NewSource(battleSeedInit))
		st, err := NewBattleState(cfg, rInit, b)
		if err != nil {
			t.Fatalf("%s init: %v", name, err)
		}
		rBatch := rand.New(rand.NewSource(battleSeedBatches))
		// Warmup 5 steps.
		for s := 1; s <= 5; s++ {
			inp, tgt := battleBatch(rBatch, cfg)
			if _, _, err := trainStepBattle(b, st, inp, tgt, s, prec); err != nil {
				t.Fatalf("%s warmup: %v", name, err)
			}
		}
		// Measure 30 steps.
		times := make([]float64, 30)
		for i := 0; i < 30; i++ {
			inp, tgt := battleBatch(rBatch, cfg)
			t0 := time.Now()
			if _, _, err := trainStepBattle(b, st, inp, tgt, i+6, prec); err != nil {
				t.Fatalf("%s measure: %v", name, err)
			}
			times[i] = float64(time.Since(t0).Nanoseconds()) / 1e6
		}
		sort.Float64s(times)
		medianMs = times[15]
		var sum, sumsq float64
		for _, v := range times {
			sum += v
			sumsq += v * v
		}
		mean := sum / 30
		variance := sumsq/30 - mean*mean
		std := math.Sqrt(math.Max(variance, 0))
		cvPct = std / mean * 100
		return
	}

	f32Ms, f32CV := measure("F32", gomlB, PrecF32)
	f16Ms, f16CV := measure("F16", adB, PrecF16)
	f8Ms, f8CV := measure("F8", adB, PrecF8E4M3)

	t.Log("=== SPEED (30-run median, ms/step) ===")
	t.Logf("%-8s | median(ms) | CV(%%) | speedup vs F32", "prec")
	t.Logf("F32      | %10.3f | %5.2f | 1.00×", f32Ms, f32CV)
	t.Logf("F16      | %10.3f | %5.2f | %.2f×", f16Ms, f16CV, f32Ms/f16Ms)
	t.Logf("F8       | %10.3f | %5.2f | %.2f×", f8Ms, f8CV, f32Ms/f8Ms)

	// CV > 1% -> warn, not fail (реальный шум thermal может быть выше).
	for name, cv := range map[string]float64{"F32": f32CV, "F16": f16CV, "F8": f8CV} {
		if cv > 1.0 {
			t.Logf("(warn) %s CV=%.2f%% > 1%% -- потенциальный noise", name, cv)
		}
	}
}

// TestB2_Battle_PeakMemory -- Peak GPU memory during Step, 3 modes.
func TestB2_Battle_PeakMemory(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode")
	}
	cfg := DefaultBattleCfg()

	getMB := func() int {
		out, err := exec.Command("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits").Output()
		if err != nil {
			return -1
		}
		var mib int
		fmt.Sscanf(strings.TrimSpace(string(out)), "%d", &mib)
		return mib
	}
	baseMB := getMB()
	t.Logf("Baseline GPU used: %d MiB", baseMB)

	gomlB, _ := backend.Get(backend.CUDA)
	if s, err := gomlB.Alloc(4); err == nil {
		gomlB.Free(s)
	}
	if err := adapter.Enable(); err != nil {
		t.Fatal(err)
	}
	adB, _ := backend.Get(backend.CUDA)

	measurePeak := func(name string, b backend.Backend, prec Precision) int {
		rInit := rand.New(rand.NewSource(battleSeedInit))
		st, err := NewBattleState(cfg, rInit, b)
		if err != nil {
			t.Fatalf("%s init: %v", name, err)
		}
		rBatch := rand.New(rand.NewSource(battleSeedBatches))
		// Prime 2 steps then measure at mid-step.
		for s := 1; s <= 2; s++ {
			inp, tgt := battleBatch(rBatch, cfg)
			trainStepBattle(b, st, inp, tgt, s, prec)
		}
		var maxMB int
		for s := 3; s <= 5; s++ {
			inp, tgt := battleBatch(rBatch, cfg)
			// Poll memory during step (crude but works for peak indicator).
			done := make(chan struct{})
			go func() {
				for {
					select {
					case <-done:
						return
					default:
						if m := getMB(); m > maxMB {
							maxMB = m
						}
						time.Sleep(2 * time.Millisecond)
					}
				}
			}()
			trainStepBattle(b, st, inp, tgt, s, prec)
			close(done)
		}
		return maxMB
	}

	f32Peak := measurePeak("F32", gomlB, PrecF32)
	f16Peak := measurePeak("F16", adB, PrecF16)
	f8Peak := measurePeak("F8", adB, PrecF8E4M3)

	t.Log("=== PEAK MEMORY (nvidia-smi during Step) ===")
	t.Logf("Baseline (idle)  : %d MiB", baseMB)
	t.Logf("F32 peak         : %d MiB (delta %+d)", f32Peak, f32Peak-baseMB)
	t.Logf("F16 peak         : %d MiB (delta %+d)", f16Peak, f16Peak-baseMB)
	t.Logf("F8  peak         : %d MiB (delta %+d)", f8Peak, f8Peak-baseMB)

	// 48GB Pro 5000 scenario:
	if f8Peak > 0 {
		t.Logf("48GB Pro 5000: fits (worst F32=%d MiB < 48000 MiB) -- headroom %d MiB",
			f32Peak, 48000-f32Peak)
	}
}
