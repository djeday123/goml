package cuda

import (
	"fmt"
	"sync"

	"github.com/ebitengine/purego"
)

// FlashAttention v55 backward kernel bindings (Tri Dao-style two-pass).
//
// All tensors are FP16 device pointers in [batch_head, seq_len, head_dim]
// layout (collapsed batch×heads dim). LSE and D are float32 [batch_head, seq_len].
//
// Two-pass split eliminates atomicAdd contention:
//   Pass 1 (dQ kernel)   — block per Q-tile, iterates K-tiles
//   Pass 2 (dKdV kernel) — block per K-tile, iterates Q-tiles
//
// Measured on RTX PRO 6000 Blackwell: ~110 TFLOPS at sl=4096, ~105 TFLOPS at
// th=8 sl=2048. Correctness: max FP16 diff < 0.0015 vs FP32 CPU reference.

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
		lib, err := purego.Dlopen(resolveLib("libflash_attention_v55_backward.so"), purego.RTLD_LAZY|purego.RTLD_GLOBAL)
		if err != nil {
			faBackward.err = fmt.Errorf("flash_attention_backward: dlopen: %w", err)
			return
		}
		faBackward.lib = lib

		purego.RegisterLibFunc(&faBackward.computeD, lib, "launch_compute_D")
		purego.RegisterLibFunc(&faBackward.dq, lib, "launch_v55_backward_dq")
		purego.RegisterLibFunc(&faBackward.dkdv, lib, "launch_v55_backward_dkdv")
		purego.RegisterLibFunc(&faBackward.combined, lib, "launch_v55_backward")
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
