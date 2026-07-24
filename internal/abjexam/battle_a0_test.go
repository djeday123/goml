package abjexam

// A-0 (2026-07-24): matmul-backward на GPU через wrapper (gt_gemm_ex trans-flags).
//
// Три теста:
//   TestA0_Battle_Accuracy — A/B 50 шагов, A=F32 CPU-bwd (B2 baseline), B=F32 GPU-bwd.
//   TestA0_Battle_Speed    — 30-run FA-класс на F32 GPU-bwd, сравнение с B2.
//   TestA0_Battle_CPUMap   — профиль остаточных host-компонент (probs D2H, CE, gradLogits H2D).
//
// Pre-registered floors:
//   Loss diff floor: 1e-5 (accumulated 50 шагов через softmax).
//   Grad hybrid: abs=1e-3 + rel=1e-3·|ref| (SGEMM F32 accum vs CPU sequential accum diff ≈ K·eps).
//   Speed прогноз: median 10-15 мс (probs D2H+H2D copy-bound), speedup ~300× vs B2.

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

// runBattleA0 -- 50 шагов F32 GPU-bwd через adapter.
func runBattleA0(t *testing.T, b backend.Backend, cfg BattleCfg, steps int) (losses []float64, grad1, gradLast []float32) {
	t.Helper()
	rInit := rand.New(rand.NewSource(battleSeedInit))
	st, err := NewBattleState(cfg, rInit, b)
	if err != nil {
		t.Fatalf("A0 NewBattleState: %v", err)
	}
	rBatch := rand.New(rand.NewSource(battleSeedBatches))
	losses = make([]float64, steps)
	grad1 = make([]float32, cfg.Embed*cfg.Vocab)
	gradLast = make([]float32, cfg.Embed*cfg.Vocab)
	for s := 1; s <= steps; s++ {
		inp, tgt := battleBatch(rBatch, cfg)
		l, g, err := trainStepBattleA0(b, st, inp, tgt, s)
		if err != nil {
			t.Fatalf("A0 step %d: %v", s, err)
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

func TestA0_Battle_Accuracy(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode")
	}
	cfg := DefaultBattleCfg()
	const steps = 50

	// ── Path A: F32 CPU-bwd (B2 baseline) ─
	gomlB, err := backend.Get(backend.CUDA)
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	if s, err := gomlB.Alloc(4); err == nil {
		gomlB.Free(s)
	}
	t.Logf("A-0 accuracy exam: Vocab=%d Embed=%d Seq=%d Batch=%d, steps=%d",
		cfg.Vocab, cfg.Embed, cfg.Seq, cfg.Batch, steps)
	lossA, gradA1, gradALast := runBattleAccuracy(t, gomlB, cfg, PrecF32, steps)
	t.Logf("Path A (F32 CPU-bwd) losses: step1=%.6f step%d=%.6f (Δ=%.4f)",
		lossA[0], steps, lossA[steps-1], lossA[steps-1]-lossA[0])

	// ── Enable adapter ─
	if err := adapter.Enable(); err != nil {
		t.Fatalf("adapter Enable: %v", err)
	}
	adB, _ := backend.Get(backend.CUDA)

	// ── Path B: F32 GPU-bwd (A-0) ─
	lossB, gradB1, gradBLast := runBattleA0(t, adB, cfg, steps)
	t.Logf("Path B (F32 GPU-bwd) losses: step1=%.6f step%d=%.6f (Δ=%.4f)",
		lossB[0], steps, lossB[steps-1], lossB[steps-1]-lossB[0])

	// ── Loss comparison ─
	t.Log("=== Loss comparison (samples every 5 steps) ===")
	t.Log("step | A CPU-bwd   | B GPU-bwd   | |A-B|")
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

	// ── Descend checks ─
	if lossA[steps-1] >= lossA[0] {
		t.Errorf("A F32 CPU-bwd не убыл: [0]=%.4f [-1]=%.4f", lossA[0], lossA[steps-1])
	}
	if lossB[steps-1] >= lossB[0] {
		t.Errorf("B F32 GPU-bwd не убыл: [0]=%.4f [-1]=%.4f", lossB[0], lossB[steps-1])
	}

	// ── Loss floor 1e-5 (accumulated через softmax) ─
	t.Logf("Loss criterion: worst |A-B| = %.3e (floor 1e-5)", worstLoss)
	if worstLoss > 1e-5 {
		t.Errorf("Loss diff exceeds floor: %.3e > 1e-5", worstLoss)
	}

	// ── Grad audit hybrid abs=1e-3 + rel=1e-3 ─
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
		if hybridFail > 0 {
			t.Errorf("grad %s hybrid fail: %d/%d", tag, hybridFail, len(got))
		}
	}
	auditGrad("step 1 B vs A", gradB1, gradA1, 1e-3, 1e-3)
	auditGrad(fmt.Sprintf("step %d B vs A", steps), gradBLast, gradALast, 1e-3, 1e-3)
}

// TestA0_Battle_Speed — FA-класс 30-run для F32 GPU-bwd.
func TestA0_Battle_Speed(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode")
	}
	// silence gate
	out, err := exec.Command("nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits").Output()
	if err == nil {
		t.Logf("nvidia-smi gate: %s", strings.TrimSpace(string(out)))
	}
	// clocks before
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

	measure := func(name string, runStep func(step int) error) (medianMs, cvPct float64) {
		t.Helper()
		// warmup 5
		for s := 1; s <= 5; s++ {
			if err := runStep(s); err != nil {
				t.Fatalf("%s warmup: %v", name, err)
			}
		}
		times := make([]float64, 30)
		for i := 0; i < 30; i++ {
			t0 := time.Now()
			if err := runStep(i + 6); err != nil {
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
		cvPct = math.Sqrt(math.Max(variance, 0)) / mean * 100
		return
	}

	// A-0 GPU-bwd path
	rInit := rand.New(rand.NewSource(battleSeedInit))
	stA0, err := NewBattleState(cfg, rInit, adB)
	if err != nil {
		t.Fatal(err)
	}
	rBatch := rand.New(rand.NewSource(battleSeedBatches))
	medA0, cvA0 := measure("A0 F32 GPU-bwd", func(step int) error {
		inp, tgt := battleBatch(rBatch, cfg)
		_, _, err := trainStepBattleA0(adB, stA0, inp, tgt, step)
		return err
	})

	t.Log("=== SPEED A-0 (30-run median, ms/step) ===")
	t.Logf("%-24s | median(ms) | CV(%%) | speedup vs B2 F32 baseline (4791.5 ms)", "config")
	t.Logf("A0 F32 GPU-bwd           | %10.3f | %5.2f | %.1f×", medA0, cvA0, 4791.5/medA0)

	if cvA0 > 5.0 {
		t.Logf("(warn) A0 CV=%.2f%% > 5%% -- copy-bound overhead", cvA0)
	}
	if 4791.5/medA0 < 100 {
		t.Logf("(warn) A0 speedup %.1fx < 100× -- ниже прогноза (10-15 мс)", 4791.5/medA0)
	}
}

// TestA0_Battle_CPUMap — компонентный профиль ВСЕХ блоков trainStepBattleA0.
// Один инструментированный прогон 10 шагов; медианы + доли шага.
// GPU-блоки с per-op Sync (искажает overall wall, ok для diagnostic map).
func TestA0_Battle_CPUMap(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode")
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
	rBatch := rand.New(rand.NewSource(battleSeedBatches))
	m := cfg.Batch * cfg.Seq

	// warmup
	for s := 1; s <= 5; s++ {
		inp, tgt := battleBatch(rBatch, cfg)
		if _, _, err := trainStepBattleA0(adB, st, inp, tgt, s); err != nil {
			t.Fatal(err)
		}
	}

	// 10 инструментированных прогонов, per-block time.Since с Sync-границей.
	// ВСЕ 12 блоков trainStepBattleA0. Sync после каждого GPU-op искажает
	// overall wall (обычная сборка перекрывает latency) — здесь дominant priority
	// = compositional map, не absolute speed.
	const N = 10
	blockNames := []string{
		"1.  tokens H2D",
		"2.  Embedding GPU",
		"3.  LayerNorm GPU",
		"4.  MatMul-fwd GPU",
		"5.  Softmax GPU",
		"6.  probs D2H (128 MB)",
		"7.  CE + gradLogits host",
		"8.  gradLogits H2D (128 MB)",
		"9.  MatMul-bwd GPU (F32Ex T,N)",
		"10. adamw_f32 GPU launch+Sync",
		"11. gradOW D2H (32 MB)",
		"12. buffer allocs+free overhead",
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

		// 12. Alloc overhead — измеряем как хвост.
		allocT0 := time.Now()

		// 1. tokens H2D
		t0 := time.Now()
		inputGPU, _ := adB.ToDevice(backend.CUDADevice(0), &cpuStorage{data: int64ToBytes(inp)})
		syncFn()
		blockTimes[0][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		// Alloc buffers upfront (амортизировано)
		embedded, _ := adB.Alloc(m * cfg.Embed * 4)
		normed, _ := adB.Alloc(m * cfg.Embed * 4)
		logits, _ := adB.Alloc(m * cfg.Vocab * 4)
		probs, _ := adB.Alloc(m * cfg.Vocab * 4)
		gradOWGPU, _ := adB.Alloc(cfg.Embed * cfg.Vocab * 4)

		// 2. Embedding
		t0 = time.Now()
		adB.Embedding(embedded, st.EmbW, inputGPU, cfg.Vocab, cfg.Embed, m, core.Float32)
		syncFn()
		blockTimes[1][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		// 3. LayerNorm
		t0 = time.Now()
		adB.LayerNorm(normed, embedded, st.LNG, st.LNB,
			core.Shape{m, cfg.Embed}, 1, 1e-5, core.Float32)
		syncFn()
		blockTimes[2][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		// 4. MatMul forward
		t0 = time.Now()
		adB.MatMul(logits, normed, st.OutW,
			core.Shape{m, cfg.Embed}, core.Shape{cfg.Embed, cfg.Vocab}, core.Float32)
		syncFn()
		blockTimes[3][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		// 5. Softmax
		t0 = time.Now()
		adB.Softmax(probs, logits, core.Shape{m, cfg.Vocab}, 1, core.Float32)
		syncFn()
		blockTimes[4][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		// 6. probs D2H
		t0 = time.Now()
		probsHost := gpuToHost(adB, probs, m*cfg.Vocab)
		blockTimes[5][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		// 7. CE + gradLogits build (host)
		gradLogits := make([]float32, m*cfg.Vocab)
		t0 = time.Now()
		copy(gradLogits, probsHost)
		for i := 0; i < m; i++ {
			tgti := int(tgt[i])
			gradLogits[i*cfg.Vocab+tgti] -= 1.0
		}
		invM := float32(1.0 / float32(m))
		for i := range gradLogits {
			gradLogits[i] *= invM
		}
		blockTimes[6][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		// 8. gradLogits H2D
		t0 = time.Now()
		gradLogitsGPU, _ := adB.ToDevice(backend.CUDADevice(0), &cpuStorage{data: f32ToBytes(gradLogits)})
		blockTimes[7][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		// 9. MatMul backward (F32Ex trans A)
		t0 = time.Now()
		if gtB, ok := adB.(*adapter.Backend); ok {
			_ = gtB.MatMulF32Ex(normed, gradLogitsGPU, gradOWGPU, cfg.Embed, cfg.Vocab, m, true, false)
		}
		syncFn()
		blockTimes[8][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		// 10. adamw_f32 launch+Sync
		t0 = time.Now()
		b1corr := float32(1.0 - math.Pow(float64(Beta1), float64(iter+1)))
		b2corr := float32(1.0 - math.Pow(float64(Beta2), float64(iter+1)))
		nOW := uint32(cfg.Embed * cfg.Vocab)
		owPtr := devPtr(st.OutW)
		gowPtr := devPtr(gradOWGPU)
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
		blockTimes[9][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		// 11. gradOW D2H (return path)
		t0 = time.Now()
		_ = gpuToHost(adB, gradOWGPU, cfg.Embed*cfg.Vocab)
		blockTimes[10][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		adB.Free(inputGPU)
		adB.Free(embedded)
		adB.Free(normed)
		adB.Free(logits)
		adB.Free(probs)
		adB.Free(gradLogitsGPU)
		adB.Free(gradOWGPU)

		// 12. Общий wall - сумма всех блоков = overhead alloc/free + прочее
		wallMs := float64(time.Since(allocT0).Nanoseconds()) / 1e6
		measuredSum := 0.0
		for i := 0; i < 11; i++ {
			measuredSum += blockTimes[i][iter]
		}
		blockTimes[11][iter] = wallMs - measuredSum
	}

	median := func(a []float64) float64 {
		b := make([]float64, len(a))
		copy(b, a)
		sort.Float64s(b)
		return b[len(b)/2]
	}

	totalMs := 0.0
	medianPerBlock := make([]float64, len(blockNames))
	for i := range blockNames {
		medianPerBlock[i] = median(blockTimes[i])
		totalMs += medianPerBlock[i]
	}

	t.Log("=== CPU-MAP — компоненты trainStepBattleA0 (median 10 iter, per-block Sync) ===")
	t.Logf("%-38s | %10s | %6s", "block", "ms", "share")
	t.Logf("%s", strings.Repeat("-", 66))
	for i, name := range blockNames {
		share := 100.0 * medianPerBlock[i] / totalMs
		t.Logf("%-38s | %10.3f | %5.1f%%", name, medianPerBlock[i], share)
	}
	t.Logf("%s", strings.Repeat("-", 66))
	t.Logf("%-38s | %10.3f | 100.0%%", "TOTAL (per-block sum)", totalMs)
}
