package abjexam

// A-LLM-2 sign-of-life: 5-step training на маленькой форме.
// Гейт: loss ДОЛЖНА двигаться (не заклинить на ln(V)) — критический life sign.
// Attention path: F32-recon (эталон). FA-bwd путь -- отдельным тестом когда встроен.

import (
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/djeday123/goml/backend"
	_ "github.com/djeday123/goml/backend/cpu"
	_ "github.com/djeday123/goml/backend/cuda"
	gomlcuda "github.com/djeday123/goml/backend/cuda"
	adapter "github.com/djeday123/goml/backend/gotorch"
)

func TestALLM_TrainStep_SignOfLife_F32Recon(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	if err := gomlcuda.FALoad(); err != nil {
		t.Skipf("libfa_sm120.so unavailable: %v", err)
	}
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

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	faCtx, err := gomlcuda.FACreate()
	if err != nil {
		t.Fatalf("FACreate: %v", err)
	}
	defer faCtx.Destroy()

	// Small form: L=1, B=1, S=128, HD=128, D=128 (H=1), V=256, FFN=256.
	cfg := BattleACfg{
		V: 256, D: 128, H: 1, HD: 128, L: 1, S: 128, B: 1, FFN: 256,
		Base: 10000.0, Eps: 1e-5,
	}
	rInit := rand.New(rand.NewSource(42))
	st, err := NewBattleAState(cfg, rInit, adB)
	if err != nil {
		t.Fatalf("NewBattleAState: %v", err)
	}
	defer st.FreeAll(adB)
	sc, err := NewBattleAScratch(cfg, adB)
	if err != nil {
		t.Fatalf("NewBattleAScratch: %v", err)
	}
	defer sc.FreeAll(adB)
	bs, err := NewBattleABwdScratch(cfg, adB)
	if err != nil {
		t.Fatalf("NewBattleABwdScratch: %v", err)
	}
	defer bs.FreeAll(adB)
	grads, err := NewBattleAGrads(cfg, adB)
	if err != nil {
		t.Fatalf("NewBattleAGrads: %v", err)
	}
	defer grads.FreeAll(adB)

	// Fixed batch (learn to predict shifted-by-one).
	rTok := rand.New(rand.NewSource(99))
	M := cfg.B * cfg.S
	inp := make([]int64, M)
	tgt := make([]int32, M)
	for i := 0; i < M; i++ {
		inp[i] = int64(rTok.Intn(cfg.V))
		tgt[i] = int32(rTok.Intn(cfg.V))
	}

	expectedLoss := math.Log(float64(cfg.V))
	t.Logf("Small form: V=%d D=%d H=%d hd=%d L=%d S=%d B=%d FFN=%d, ln(V)=%.4f",
		cfg.V, cfg.D, cfg.H, cfg.HD, cfg.L, cfg.S, cfg.B, cfg.FFN, expectedLoss)

	const nSteps = 10
	const lr = 1e-4

	losses := make([]float64, nSteps)
	for step := 0; step < nSteps; step++ {
		loss, err := trainStepBattleA(adB, st, sc, bs, grads, faCtx, inp, tgt, lr, AttnBwdF32Recon)
		if err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
		if math.IsNaN(loss) || math.IsInf(loss, 0) {
			t.Fatalf("step %d: loss NaN/Inf = %v", step, loss)
		}
		losses[step] = loss
		t.Logf("step %d: loss=%.6f (delta from ln(V)=%+.6f)", step, loss, loss-expectedLoss)
	}
	// Sign of life: loss must move (not stuck at ln(V)).
	delta := losses[nSteps-1] - losses[0]
	if delta > -1e-4 { // Must DECREASE (delta < -1e-4)
		t.Errorf("SIGN OF LIFE FAIL: loss did not decrease (delta over %d steps = %+.6e, need < -1e-4)", nSteps, delta)
	} else {
		t.Logf("SIGN OF LIFE PASS: loss decreased by %+.4f over %d steps (monotonic decrement each step)", delta, nSteps)
	}
}
