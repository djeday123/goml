package abjexam

// A-LLM-2 (2026-07-26): trainStepBattleA = fwd + bwd + SGD update.
// Host-side SGD (простой, correctness-first). GPU-optim -- отдельное звено.

import (
	"fmt"

	"github.com/djeday123/goml/backend"
	gomlcuda "github.com/djeday123/goml/backend/cuda"
)

// sgdUpdateHost — читает W и dW на хост, применяет W -= lr * dW, upload'ит обратно.
// Простой и медленный. Correctness verification.
func sgdUpdateHost(b backend.Backend, W, dW backend.Storage, n int, lr float32) error {
	wH := gpuToHost(b, W, n)
	dwH := gpuToHost(b, dW, n)
	for i := 0; i < n; i++ {
		wH[i] -= lr * dwH[i]
	}
	if _, err := uploadInto(b, W, f32ToBytes(wH)); err != nil {
		return fmt.Errorf("sgd upload: %w", err)
	}
	return nil
}

// applySGD -- host-side SGD across all weights.
func applySGD(b backend.Backend, st *BattleAState, grads *BattleAGrads, lr float32) error {
	cfg := st.Cfg
	if err := sgdUpdateHost(b, st.Embed, grads.DEmbed, cfg.V*cfg.D, lr); err != nil {
		return fmt.Errorf("sgd Embed: %w", err)
	}
	for l := 0; l < cfg.L; l++ {
		lw := &st.Layers[l]
		lg := &grads.Layers[l]
		if err := sgdUpdateHost(b, lw.Norm1, lg.DNorm1, cfg.D, lr); err != nil {
			return fmt.Errorf("sgd L%d Norm1: %w", l, err)
		}
		if err := sgdUpdateHost(b, lw.Wq, lg.DWq, cfg.D*cfg.D, lr); err != nil {
			return fmt.Errorf("sgd L%d Wq: %w", l, err)
		}
		if err := sgdUpdateHost(b, lw.Wk, lg.DWk, cfg.D*cfg.D, lr); err != nil {
			return fmt.Errorf("sgd L%d Wk: %w", l, err)
		}
		if err := sgdUpdateHost(b, lw.Wv, lg.DWv, cfg.D*cfg.D, lr); err != nil {
			return fmt.Errorf("sgd L%d Wv: %w", l, err)
		}
		if err := sgdUpdateHost(b, lw.Wo, lg.DWo, cfg.D*cfg.D, lr); err != nil {
			return fmt.Errorf("sgd L%d Wo: %w", l, err)
		}
		if err := sgdUpdateHost(b, lw.Norm2, lg.DNorm2, cfg.D, lr); err != nil {
			return fmt.Errorf("sgd L%d Norm2: %w", l, err)
		}
		if err := sgdUpdateHost(b, lw.W1, lg.DW1, cfg.D*cfg.FFN, lr); err != nil {
			return fmt.Errorf("sgd L%d W1: %w", l, err)
		}
		if err := sgdUpdateHost(b, lw.W2, lg.DW2, cfg.FFN*cfg.D, lr); err != nil {
			return fmt.Errorf("sgd L%d W2: %w", l, err)
		}
	}
	if err := sgdUpdateHost(b, st.NormOut, grads.DNormOut, cfg.D, lr); err != nil {
		return fmt.Errorf("sgd NormOut: %w", err)
	}
	if err := sgdUpdateHost(b, st.Wout, grads.DWout, cfg.D*cfg.V, lr); err != nil {
		return fmt.Errorf("sgd Wout: %w", err)
	}
	return nil
}

// trainStepBattleA -- fwd + bwd + SGD. Returns loss.
func trainStepBattleA(b backend.Backend, st *BattleAState, sc *BattleAScratch,
	bs *BattleABwdScratch, grads *BattleAGrads,
	faCtx *gomlcuda.FAContext, inputTokens []int64, targetTokens []int32,
	lr float32, attnPath AttnBwdPath,
	snap *BattleASnapScratch, fb *faBlockBufs) (float64, error) {
	loss, err := fwdBattleA(b, st, sc, faCtx, inputTokens, targetTokens, snap)
	if err != nil {
		return 0, fmt.Errorf("fwd: %w", err)
	}
	if err := zeroGrads(b, grads); err != nil {
		return 0, fmt.Errorf("zero grads: %w", err)
	}
	if err := bwdBattleA(b, st, sc, bs, grads, faCtx, inputTokens, attnPath, snap, fb); err != nil {
		return 0, fmt.Errorf("bwd: %w", err)
	}
	if err := applySGD(b, st, grads, lr); err != nil {
		return 0, fmt.Errorf("sgd: %w", err)
	}
	return loss, nil
}
