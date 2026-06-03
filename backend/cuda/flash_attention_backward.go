package cuda

import (
	"fmt"
	"sync"

	"github.com/ebitengine/purego"
)

// FlashAttention backward kernel bindings (Tri Dao-style two-pass).
//
// Resolves to v56 (vectorized writeback) when available, falls back to v55.
// Both ABIs are identical — the .so is selected via libflash_attention_backward.so
// symlink set up by scripts/build_cuda.sh.
//
// All tensors are FP16 device pointers in [batch_head, seq_len, head_dim]
// layout (collapsed batch×heads dim). LSE and D are float32 [batch_head, seq_len].
//
// Two-pass split eliminates atomicAdd contention:
//   Pass 1 (dQ kernel)   — block per Q-tile, iterates K-tiles
//   Pass 2 (dKdV kernel) — block per K-tile, iterates Q-tiles
//
// Measured on RTX PRO 6000 Blackwell (v56):
//   th=4 sl=4096 hd=128 ca=1  → 113.9 TFLOPS
//   th=8 sl=2048 hd=128 ca=1  → 109.6 TFLOPS
// Correctness: max FP16 diff < 0.0015 vs FP32 CPU reference.

var faBackward struct {
	once sync.Once
	err  error
	lib  uintptr

	computeD func(dO, O, D uintptr, th, sl, hd int32)
	dq       func(Q, K, V, dO, LSE, D, dQ uintptr, th, sl, hd, ca int32)
	dkdv     func(Q, K, V, dO, LSE, D, dK, dV uintptr, th, sl, hd, ca int32)
	combined func(Q, K, V, dO, O, LSE, dQ, dK, dV, D uintptr, th, sl, hd, ca int32)
}

func initFlashAttentionBackward() error {
	faBackward.once.Do(func() {
		// Prefer the unversioned symlink (v56 latest), fall back to v55.
		candidates := []string{
			"libflash_attention_backward.so",
			"libflash_attention_v56_backward.so",
			"libflash_attention_v55_backward.so",
		}
		var lib uintptr
		var lastErr error
		var picked string
		for _, name := range candidates {
			l, err := purego.Dlopen(resolveLib(name), purego.RTLD_LAZY|purego.RTLD_GLOBAL)
			if err == nil {
				lib = l
				picked = name
				break
			}
			lastErr = err
		}
		if lib == 0 {
			faBackward.err = fmt.Errorf("flash_attention_backward: dlopen: %w", lastErr)
			return
		}
		faBackward.lib = lib

		// Symbol names: v56 and v55 both export launch_v55_backward_* (kept for ABI
		// stability) plus their own versioned launch_v5N_backward symbols.
		dqSym := "launch_v55_backward_dq"
		dkdvSym := "launch_v55_backward_dkdv"
		combinedSym := "launch_v55_backward"
		if picked == "libflash_attention_v56_backward.so" || picked == "libflash_attention_backward.so" {
			// v56 currently uses launch_v56_* names; try those first.
			dqSym = "launch_v56_backward_dq"
			dkdvSym = "launch_v56_backward_dkdv"
			combinedSym = "launch_v56_backward"
		}

		purego.RegisterLibFunc(&faBackward.computeD, lib, "launch_compute_D")
		purego.RegisterLibFunc(&faBackward.dq, lib, dqSym)
		purego.RegisterLibFunc(&faBackward.dkdv, lib, dkdvSym)
		purego.RegisterLibFunc(&faBackward.combined, lib, combinedSym)
	})
	return faBackward.err
}

// FlashAttentionComputeD precomputes D_i = Σ_d dO_i[d] · O_i[d] per row.
// Stored as float32 [batchHeads × seqLen]. Required input for the dQ and dKdV
// kernels.
func FlashAttentionComputeD(dO, O, D uintptr, batchHeads, seqLen, headDim int) error {
	if err := initFlashAttentionBackward(); err != nil {
		return err
	}
	faBackward.computeD(dO, O, D, int32(batchHeads), int32(seqLen), int32(headDim))
	return nil
}

// FlashAttentionBackwardDQ runs Pass 1 of the FA backward: dQ accumulation.
//
// Inputs (all FP16 device ptrs unless noted):
//
//	Q, K, V: [batchHeads, seqLen, headDim]
//	dO:      [batchHeads, seqLen, headDim] — output gradient
//	LSE:     [batchHeads, seqLen]          float32 — log-sum-exp from forward
//	D:       [batchHeads, seqLen]          float32 — from FlashAttentionComputeD
//
// Output:
//
//	dQ_out: [batchHeads, seqLen, headDim]  FP16 — must be zeroed by caller
func FlashAttentionBackwardDQ(Q, K, V, dO, LSE, D, dQ uintptr, batchHeads, seqLen, headDim int, causal bool) error {
	if err := initFlashAttentionBackward(); err != nil {
		return err
	}
	ca := int32(0)
	if causal {
		ca = 1
	}
	faBackward.dq(Q, K, V, dO, LSE, D, dQ, int32(batchHeads), int32(seqLen), int32(headDim), ca)
	return nil
}

// FlashAttentionBackwardDKDV runs Pass 2 of the FA backward: dK, dV accumulation.
// Same inputs as Pass 1 plus dK_out, dV_out (FP16, must be zeroed by caller).
func FlashAttentionBackwardDKDV(Q, K, V, dO, LSE, D, dK, dV uintptr, batchHeads, seqLen, headDim int, causal bool) error {
	if err := initFlashAttentionBackward(); err != nil {
		return err
	}
	ca := int32(0)
	if causal {
		ca = 1
	}
	faBackward.dkdv(Q, K, V, dO, LSE, D, dK, dV, int32(batchHeads), int32(seqLen), int32(headDim), ca)
	return nil
}

// FlashAttentionBackward runs the full FA backward (compute_D + dQ + dKdV).
//
// dQ, dK, dV must be zeroed by the caller. D is a scratch buffer of size
// batchHeads × seqLen × float32, owned by the caller (typically reused
// across iterations).
func FlashAttentionBackward(Q, K, V, dO, O, LSE, dQ, dK, dV, D uintptr, batchHeads, seqLen, headDim int, causal bool) error {
	if err := initFlashAttentionBackward(); err != nil {
		return err
	}
	ca := int32(0)
	if causal {
		ca = 1
	}
	faBackward.combined(Q, K, V, dO, O, LSE, dQ, dK, dV, D, int32(batchHeads), int32(seqLen), int32(headDim), ca)
	return nil
}
