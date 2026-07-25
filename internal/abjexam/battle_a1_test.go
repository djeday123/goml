package abjexam

// A-1 (2026-07-24): fused CrossEntropy GPU kernel — устраняет probs D2H + CE host + gradLogits H2D
// (81% wall'а по CPU-карте A-0). Тесты уровня R02b:
//   TestA1_CE_Correctness_* — форма [1,8], [8,32000], [1024,32000]
//   TestA1_CE_Gradcheck    — numerical grad F64 1e-8 на паре логитов
//   TestA1_CE_Edges        — target=0, target=V-1, uniform, extreme outlier
//   TestA1_Battle_Accuracy — 50-шаг A/B (A=A-0 F32, B=A-1 CE-fused)
//   TestA1_Battle_Speed    — 30-run FA-класс
//   TestA1_Battle_CPUMap   — новая карта после CE-scatter
//
// Pre-registered floors:
//   Loss per-row vs Go F64: abs ≤ 1e-4 (sqrt(V)·eps модель, V=32000, +peak margin)
//     sqrt(32000)·eps = 179·6e-8 = 1.07e-5 per row; ×10 запас для approx PTX (lg2/ex2 1-3 ULP)
//   Grad hybrid: abs=1e-6 + rel=1e-4
//     Non-target: mag~1/V=3e-5, error ~sqrt(V)·eps·softmax ~3e-10 abs; ×3000 запас
//     Target: mag~1, after invM(=1e-3) mag~1e-3, error ~1e-8 abs; ×100 запас
//   Battle траектория 50-step A vs B: floor 1e-4 (accumulated approx PTX)

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

// ceReferenceF64 — CPU Go math ref (IEEE 754 double). Возвращает per-row loss и grad
// с масштабированием invM = 1/m (как kernel).
func ceReferenceF64(logits []float32, targets []int32, m, V int) (loss []float64, grad []float64) {
	loss = make([]float64, m)
	grad = make([]float64, m*V)
	invM := 1.0 / float64(m)
	for i := 0; i < m; i++ {
		// row max
		maxVal := math.Inf(-1)
		for j := 0; j < V; j++ {
			v := float64(logits[i*V+j])
			if v > maxVal {
				maxVal = v
			}
		}
		// sum exp
		sum := 0.0
		for j := 0; j < V; j++ {
			sum += math.Exp(float64(logits[i*V+j]) - maxVal)
		}
		logZ := maxVal + math.Log(sum)
		tgt := int(targets[i])
		tgtLogit := float64(logits[i*V+tgt])
		loss[i] = logZ - tgtLogit
		// grad
		invSum := 1.0 / sum
		for j := 0; j < V; j++ {
			p := math.Exp(float64(logits[i*V+j])-maxVal) * invSum
			if j == tgt {
				p -= 1.0
			}
			grad[i*V+j] = p * invM
		}
	}
	return
}

// launchCEF32 — helper: запускает cross_entropy_f32 kernel через backend.Launch.
func launchCEF32(t *testing.T, b backend.Backend, logitsGPU, targetsGPU, lossGPU, gradGPU backend.Storage, m, V int) {
	t.Helper()
	logitsPtr := devPtr(logitsGPU)
	targetsPtr := devPtr(targetsGPU)
	lossPtr := devPtr(lossGPU)
	gradPtr := devPtr(gradGPU)
	nRows := uint32(m)
	vocab := uint32(V)
	invBs := float32(1.0 / float32(m))
	params := []unsafe.Pointer{
		unsafe.Pointer(&logitsPtr),
		unsafe.Pointer(&targetsPtr),
		unsafe.Pointer(&lossPtr),
		unsafe.Pointer(&gradPtr),
		unsafe.Pointer(&nRows),
		unsafe.Pointer(&vocab),
		unsafe.Pointer(&invBs),
	}
	if l, ok := b.(interface {
		Launch(name string, gx, gy, gz, bx, by, bz uint32, params []unsafe.Pointer) error
	}); ok {
		if err := l.Launch("cross_entropy_f32", uint32(m), 1, 1, 256, 1, 1, params); err != nil {
			t.Fatalf("cross_entropy_f32 launch: %v", err)
		}
	}
	if s, ok := b.(interface{ Sync() error }); ok {
		s.Sync()
	}
}

// int32ToBytesLE — little-endian encoding.
func int32ToBytesLE(a []int32) []byte {
	b := make([]byte, len(a)*4)
	for i, v := range a {
		u := uint32(v)
		b[i*4] = byte(u)
		b[i*4+1] = byte(u >> 8)
		b[i*4+2] = byte(u >> 16)
		b[i*4+3] = byte(u >> 24)
	}
	return b
}

// runCECorrectness — общий блок: генерит form, запускает kernel, сравнивает с ceReferenceF64.
func runCECorrectness(t *testing.T, tag string, m, V int, seed int64, lossFloor, gradAbsTol, gradRelTol float64) {
	t.Helper()

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
	b, _ := backend.Get(backend.CUDA)

	// Generate logits (F32) and targets (int32).
	r := rand.New(rand.NewSource(seed))
	logitsHost := make([]float32, m*V)
	for i := range logitsHost {
		logitsHost[i] = float32(r.NormFloat64() * 1.0)
	}
	targetsHost := make([]int32, m)
	for i := range targetsHost {
		targetsHost[i] = int32(r.Intn(V))
	}

	// Upload.
	logitsGPU, _ := b.ToDevice(backend.CUDADevice(0), &cpuStorage{data: f32ToBytes(logitsHost)})
	defer b.Free(logitsGPU)
	targetsGPU, _ := b.ToDevice(backend.CUDADevice(0), &cpuStorage{data: int32ToBytesLE(targetsHost)})
	defer b.Free(targetsGPU)
	lossGPU, _ := b.Alloc(m * 4)
	defer b.Free(lossGPU)
	gradGPU, _ := b.Alloc(m * V * 4)
	defer b.Free(gradGPU)

	// GPU CE F32.
	launchCEF32(t, b, logitsGPU, targetsGPU, lossGPU, gradGPU, m, V)

	// Download outputs.
	lossHost := gpuToHost(b, lossGPU, m)
	gradHost := gpuToHost(b, gradGPU, m*V)

	// CPU F64 reference.
	lossRef, gradRef := ceReferenceF64(logitsHost, targetsHost, m, V)

	// Compare loss.
	var maxLossAbs, maxLossRel float64
	for i := 0; i < m; i++ {
		d := math.Abs(float64(lossHost[i]) - lossRef[i])
		rel := d / (math.Abs(lossRef[i]) + 1e-30)
		if d > maxLossAbs {
			maxLossAbs = d
		}
		if rel > maxLossRel {
			maxLossRel = rel
		}
	}
	t.Logf("[%s m=%d V=%d] loss: maxAbs=%.3e maxRel=%.3e (floor abs=%.0e)", tag, m, V, maxLossAbs, maxLossRel, lossFloor)
	if maxLossAbs > lossFloor {
		t.Errorf("[%s] loss abs %.3e > floor %.0e", tag, maxLossAbs, lossFloor)
	}

	// Compare grad hybrid abs+rel·|ref|.
	var maxGradAbs, maxGradRel float64
	hybridFail := 0
	for i := range gradHost {
		g := float64(gradHost[i])
		r := gradRef[i]
		d := math.Abs(g - r)
		rel := d / (math.Abs(r) + 1e-30)
		if d > maxGradAbs {
			maxGradAbs = d
		}
		if rel > maxGradRel {
			maxGradRel = rel
		}
		if d > gradAbsTol+gradRelTol*math.Abs(r) {
			hybridFail++
		}
	}
	t.Logf("[%s m=%d V=%d] grad: maxAbs=%.3e maxRel=%.3e hybridFail=%d/%d (floor abs=%.0e+rel=%.0e·|ref|)",
		tag, m, V, maxGradAbs, maxGradRel, hybridFail, len(gradHost), gradAbsTol, gradRelTol)
	if hybridFail > 0 {
		t.Errorf("[%s] grad hybridFail: %d/%d", tag, hybridFail, len(gradHost))
	}
}

func TestA1_CE_Correctness_Small(t *testing.T) {
	// Small form: m=1, V=8. sqrt(8)·eps=1.7e-7. Floor 1e-5.
	runCECorrectness(t, "small", 1, 8, 42, 1e-5, 1e-6, 1e-4)
}

func TestA1_CE_Correctness_Medium(t *testing.T) {
	// Medium form: m=8, V=32000. sqrt(32000)·eps=1.07e-5. Floor 1e-4.
	runCECorrectness(t, "medium", 8, 32000, 43, 1e-4, 1e-6, 1e-4)
}

func TestA1_CE_Correctness_Battle(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	// Battle form: m=1024, V=32000. Same sqrt(V)·eps model. Floor 1e-4.
	runCECorrectness(t, "battle", 1024, 32000, 44, 1e-4, 1e-6, 1e-4)
}

// TestA1_CE_Gradcheck — numerical grad F64 через perturbation logits.
// Строит small form [1, 32], проверяет grad_analytic vs grad_numerical
// на несколько логитов. Точность F64 CPU ref: 1e-6 (numerical eps ~ 1e-4, F32 ULP ~ 6e-8).
func TestA1_CE_Gradcheck(t *testing.T) {
	m, V := 1, 32
	gomlB, err := backend.Get(backend.CUDA)
	if err != nil {
		t.Skip("CUDA unavailable")
	}
	if s, err := gomlB.Alloc(4); err == nil {
		gomlB.Free(s)
	}
	if err := adapter.Enable(); err != nil {
		t.Fatal(err)
	}
	b, _ := backend.Get(backend.CUDA)

	r := rand.New(rand.NewSource(1234))
	logits := make([]float32, m*V)
	for i := range logits {
		logits[i] = float32(r.NormFloat64())
	}
	targets := []int32{7} // arbitrary

	// GPU CE grad (F32).
	logitsGPU, _ := b.ToDevice(backend.CUDADevice(0), &cpuStorage{data: f32ToBytes(logits)})
	defer b.Free(logitsGPU)
	targetsGPU, _ := b.ToDevice(backend.CUDADevice(0), &cpuStorage{data: int32ToBytesLE(targets)})
	defer b.Free(targetsGPU)
	lossGPU, _ := b.Alloc(m * 4)
	defer b.Free(lossGPU)
	gradGPU, _ := b.Alloc(m * V * 4)
	defer b.Free(gradGPU)
	launchCEF32(t, b, logitsGPU, targetsGPU, lossGPU, gradGPU, m, V)
	gpuGrad := gpuToHost(b, gradGPU, m*V)

	// Numerical grad F64 via central difference on CPU ref.
	// dL/dlogit_j = (L(logit_j+h) - L(logit_j-h)) / (2h)  with h = 1e-4.
	h := 1e-4
	invM := 1.0 / float64(m)
	baseLossRef, _ := ceReferenceF64(logits, targets, m, V)
	_ = baseLossRef

	checkIndices := []int{0, 7, 15, 31} // включая tgt=7
	for _, j := range checkIndices {
		lp := make([]float32, m*V)
		copy(lp, logits)
		lp[j] = logits[j] + float32(h)
		lossPlus, _ := ceReferenceF64(lp, targets, m, V)
		lp[j] = logits[j] - float32(h)
		lossMinus, _ := ceReferenceF64(lp, targets, m, V)
		numGrad := (lossPlus[0] - lossMinus[0]) / (2.0 * h) * invM
		analytic := float64(gpuGrad[j])
		diff := math.Abs(numGrad - analytic)
		rel := diff / (math.Abs(numGrad) + 1e-30)
		t.Logf("gradcheck j=%2d: analytic=%+.6e numerical=%+.6e diff=%.3e rel=%.3e",
			j, analytic, numGrad, diff, rel)
		// Floor: F32 CE ~ sqrt(V)·eps ~ 3e-8, plus numerical eps for h=1e-4 ~ 1e-4·h*3rd_derivative.
		// Consolidated tol 1e-4 relative or 1e-6 abs.
		if diff > 1e-6 && rel > 1e-3 {
			t.Errorf("gradcheck j=%d: diff=%.3e rel=%.3e out of tol", j, diff, rel)
		}
	}
}

func TestA1_CE_Edges(t *testing.T) {
	m, V := 4, 128
	gomlB, err := backend.Get(backend.CUDA)
	if err != nil {
		t.Skip("CUDA unavailable")
	}
	if s, err := gomlB.Alloc(4); err == nil {
		gomlB.Free(s)
	}
	if err := adapter.Enable(); err != nil {
		t.Fatal(err)
	}
	b, _ := backend.Get(backend.CUDA)

	// Edge cases:
	//   row 0: target = 0 (first)
	//   row 1: target = V-1 (last)
	//   row 2: uniform logits (all 0.5)
	//   row 3: extreme outlier (one logit huge, target != outlier)
	logits := make([]float32, m*V)
	targets := []int32{0, int32(V - 1), 63, 100}
	// row 0: gradient random-ish
	r := rand.New(rand.NewSource(99))
	for j := 0; j < V; j++ {
		logits[0*V+j] = float32(r.NormFloat64())
	}
	// row 1: random-ish
	for j := 0; j < V; j++ {
		logits[1*V+j] = float32(r.NormFloat64())
	}
	// row 2: uniform
	for j := 0; j < V; j++ {
		logits[2*V+j] = 0.5
	}
	// row 3: extreme outlier at j=50, target at 100
	for j := 0; j < V; j++ {
		logits[3*V+j] = float32(r.NormFloat64() * 0.1)
	}
	logits[3*V+50] = 50.0 // huge outlier

	logitsGPU, _ := b.ToDevice(backend.CUDADevice(0), &cpuStorage{data: f32ToBytes(logits)})
	defer b.Free(logitsGPU)
	targetsGPU, _ := b.ToDevice(backend.CUDADevice(0), &cpuStorage{data: int32ToBytesLE(targets)})
	defer b.Free(targetsGPU)
	lossGPU, _ := b.Alloc(m * 4)
	defer b.Free(lossGPU)
	gradGPU, _ := b.Alloc(m * V * 4)
	defer b.Free(gradGPU)

	launchCEF32(t, b, logitsGPU, targetsGPU, lossGPU, gradGPU, m, V)
	lossHost := gpuToHost(b, lossGPU, m)
	gradHost := gpuToHost(b, gradGPU, m*V)

	lossRef, gradRef := ceReferenceF64(logits, targets, m, V)

	// Per-row loss compare.
	for i := 0; i < m; i++ {
		d := math.Abs(float64(lossHost[i]) - lossRef[i])
		t.Logf("edge row %d target=%d: loss GPU=%.6f Go F64=%.6f diff=%.3e", i, targets[i], lossHost[i], lossRef[i], d)
		if d > 1e-4 {
			t.Errorf("edge row %d loss diff %.3e > 1e-4", i, d)
		}
	}

	// Uniform-row sanity: loss should be ~ln(V) = ln(128) = 4.852
	expectedUniformLoss := math.Log(float64(V))
	if math.Abs(float64(lossHost[2])-expectedUniformLoss) > 1e-4 {
		t.Errorf("uniform row loss expected ~ln(V)=%.4f, got %.4f", expectedUniformLoss, lossHost[2])
	}

	// Extreme outlier sanity: row 3, target=100, outlier at 50.
	// softmax has almost all mass at j=50, target=100 → probs[100]≈0 → loss ≈ 50 (large).
	if lossHost[3] < 40 {
		t.Errorf("extreme outlier row loss expected ~50, got %.4f", lossHost[3])
	}

	// Grad hybrid check per row.
	for i := 0; i < m; i++ {
		hybridFail := 0
		for j := 0; j < V; j++ {
			g := float64(gradHost[i*V+j])
			ref := gradRef[i*V+j]
			d := math.Abs(g - ref)
			if d > 1e-6+1e-4*math.Abs(ref) {
				hybridFail++
			}
		}
		if hybridFail > 0 {
			t.Errorf("edge row %d grad hybridFail=%d/%d", i, hybridFail, V)
		}
	}
}

// runBattleA1 -- 50 шагов через trainStepBattleA1.
func runBattleA1(t *testing.T, b backend.Backend, cfg BattleCfg, steps int) (losses []float64, grad1, gradLast []float32) {
	t.Helper()
	rInit := rand.New(rand.NewSource(battleSeedInit))
	st, err := NewBattleState(cfg, rInit, b)
	if err != nil {
		t.Fatalf("A1 NewBattleState: %v", err)
	}
	rBatch := rand.New(rand.NewSource(battleSeedBatches))
	losses = make([]float64, steps)
	grad1 = make([]float32, cfg.Embed*cfg.Vocab)
	gradLast = make([]float32, cfg.Embed*cfg.Vocab)
	for s := 1; s <= steps; s++ {
		inp, tgt := battleBatch(rBatch, cfg)
		l, g, err := trainStepBattleA1(b, st, inp, tgt, s)
		if err != nil {
			t.Fatalf("A1 step %d: %v", s, err)
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

func TestA1_Battle_Accuracy(t *testing.T) {
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

	t.Logf("A-1 accuracy exam: Vocab=%d Embed=%d Seq=%d Batch=%d, steps=%d",
		cfg.Vocab, cfg.Embed, cfg.Seq, cfg.Batch, steps)
	// Path A = A-0 F32 GPU-bwd (baseline).
	lossA, gradA1, gradALast := runBattleA0(t, adB, cfg, steps)
	t.Logf("Path A (A-0 F32 GPU-bwd) losses: step1=%.6f step%d=%.6f (Δ=%.4f)",
		lossA[0], steps, lossA[steps-1], lossA[steps-1]-lossA[0])
	// Path B = A-1 CE-fused.
	lossB, gradB1, gradBLast := runBattleA1(t, adB, cfg, steps)
	t.Logf("Path B (A-1 CE-fused) losses: step1=%.6f step%d=%.6f (Δ=%.4f)",
		lossB[0], steps, lossB[steps-1], lossB[steps-1]-lossB[0])

	t.Log("=== Loss comparison (samples every 5 steps) ===")
	t.Log("step | A A-0       | B A-1       | |A-B|")
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
	// Pre-registered floor 1e-4 (approx PTX 1-3 ULP × sqrt(50) × AdamW-boost).
	const lossFloor = 1e-4
	t.Logf("Loss criterion: worst |A-B| = %.3e (floor %g)", worstLoss, lossFloor)
	if worstLoss > lossFloor {
		t.Errorf("loss diff exceeds floor: %.3e > %g", worstLoss, lossFloor)
	}
	// Grad hybrid abs=1e-3 + rel=1e-3.
	auditGradA1 := func(tag string, got, ref []float32, absTol, relTol float64) {
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
	auditGradA1("step 1 B vs A", gradB1, gradA1, 1e-3, 1e-3)
	auditGradA1(fmt.Sprintf("step %d B vs A", steps), gradBLast, gradALast, 1e-3, 1e-3)
}

func TestA1_Battle_Speed(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	// silence gate + clocks
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
	rBatch := rand.New(rand.NewSource(battleSeedBatches))
	// Warmup 5
	for s := 1; s <= 5; s++ {
		inp, tgt := battleBatch(rBatch, cfg)
		if _, _, err := trainStepBattleA1(adB, st, inp, tgt, s); err != nil {
			t.Fatal(err)
		}
	}
	// Measure 30
	times := make([]float64, 30)
	for i := 0; i < 30; i++ {
		inp, tgt := battleBatch(rBatch, cfg)
		t0 := time.Now()
		if _, _, err := trainStepBattleA1(adB, st, inp, tgt, i+6); err != nil {
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

	t.Log("=== SPEED A-1 (30-run median, ms/step) ===")
	t.Logf("A1 CE-fused              | median=%.3f ms | CV=%.2f%% | speedup vs B2 (4791.5 ms): %.1f×, vs A-0 (136.2 ms): %.2f×",
		median, cv, 4791.5/median, 136.2/median)
	if cv > 5.0 {
		t.Logf("(warn) A1 CV=%.2f%% > 5%%", cv)
	}
}

// TestA1_Battle_CPUMap -- новая карта 12 блоков после устранения probs D2H + CE host + gradLogits H2D.
func TestA1_Battle_CPUMap(t *testing.T) {
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
	rBatch := rand.New(rand.NewSource(battleSeedBatches))
	m := cfg.Batch * cfg.Seq

	// warmup 5
	for s := 1; s <= 5; s++ {
		inp, tgt := battleBatch(rBatch, cfg)
		if _, _, err := trainStepBattleA1(adB, st, inp, tgt, s); err != nil {
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
		"9.  gradOW D2H (32 MB, для сверки в тесте)",
		"10. buffer allocs+free overhead",
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

		// 1. tokens + targets H2D
		t0 := time.Now()
		inputGPU, _ := adB.ToDevice(backend.CUDADevice(0), &cpuStorage{data: int64ToBytes(inp)})
		targetsI32 := make([]int32, m)
		for i, v := range tg32Slice(tgt) {
			targetsI32[i] = v
		}
		targetsGPU, _ := adB.ToDevice(backend.CUDADevice(0), &cpuStorage{data: int32ToBytesLE(targetsI32)})
		syncFn()
		blockTimes[0][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		embedded, _ := adB.Alloc(m * cfg.Embed * 4)
		normed, _ := adB.Alloc(m * cfg.Embed * 4)
		logits, _ := adB.Alloc(m * cfg.Vocab * 4)
		lossGPU, _ := adB.Alloc(m * 4)
		gradLogitsGPU, _ := adB.Alloc(m * cfg.Vocab * 4)
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

		// 4. MatMul fwd
		t0 = time.Now()
		adB.MatMul(logits, normed, st.OutW,
			core.Shape{m, cfg.Embed}, core.Shape{cfg.Embed, cfg.Vocab}, core.Float32)
		syncFn()
		blockTimes[3][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		// 5. CE fused
		t0 = time.Now()
		launchCEF32(t, adB, logits, targetsGPU, lossGPU, gradLogitsGPU, m, cfg.Vocab)
		blockTimes[4][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		// 6. MatMul bwd
		t0 = time.Now()
		if gtB, ok := adB.(*adapter.Backend); ok {
			_ = gtB.MatMulF32Ex(normed, gradLogitsGPU, gradOWGPU, cfg.Embed, cfg.Vocab, m, true, false)
		}
		syncFn()
		blockTimes[5][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		// 7. adamw
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
		blockTimes[6][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		// 8. loss D2H (4 KB tiny)
		t0 = time.Now()
		_ = gpuToHost(adB, lossGPU, m)
		blockTimes[7][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		// 9. gradOW D2H (32 MB)
		t0 = time.Now()
		_ = gpuToHost(adB, gradOWGPU, cfg.Embed*cfg.Vocab)
		blockTimes[8][iter] = float64(time.Since(t0).Nanoseconds()) / 1e6

		adB.Free(inputGPU)
		adB.Free(targetsGPU)
		adB.Free(embedded)
		adB.Free(normed)
		adB.Free(logits)
		adB.Free(lossGPU)
		adB.Free(gradLogitsGPU)
		adB.Free(gradOWGPU)

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

	t.Log("=== A-1 CPU-MAP — компоненты trainStepBattleA1 (median 10 iter, per-block Sync) ===")
	t.Logf("%-42s | %10s | %6s", "block", "ms", "share")
	t.Logf("%s", strings.Repeat("-", 70))
	for i, name := range blockNames {
		share := 100.0 * medians[i] / total
		t.Logf("%-42s | %10.3f | %5.1f%%", name, medians[i], share)
	}
	t.Logf("%s", strings.Repeat("-", 70))
	t.Logf("%-42s | %10.3f | 100.0%%", "TOTAL", total)
}

// tg32Slice: for CPU-map, targets int64 → int32.
func tg32Slice(tgt []int64) []int32 {
	out := make([]int32, len(tgt))
	for i, v := range tgt {
		out[i] = int32(v)
	}
	return out
}
