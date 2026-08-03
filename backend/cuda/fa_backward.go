package cuda

// A-LLM-1 G2 (2026-07-25): purego binding для libfa_bwd_sm120.so —
// backward chain wrappers (D + merged + dk + dq).
// Тонкие обёртки поверх v0.2.0 launchers. Только развёртка аргументов, без логики.
//
// МАГНИТУДНЫЙ КОНТРАКТ (A-LLM-5, 2026-08-03, Н4): decoded-магнитуда FP8-входов
// (Q/K/V) обязана быть O(1) — квантизация scale = amax (НЕ amax/448).
// Карта аккумуляторов (П.0 A-LLM-5): merged — F32-акк (f32.f16.f16.f32,
// fa_bwd_merged_v1.cu:43); dk — F32-акки (:51,:66); dq — F16-акк
// (f16.e4m3.e4m3.f16, fa_bwd_dq_new.cu:47, packed): требует
// sum_j |dS[i,j]|*|Kd| < 65504 — при O(1)-входах и P-нормировке
// (sum_j P_ij = 1) запас >= 12x наивный / ~1e4x фактический.
// dS в dSnat/dST квантизуется direct-cast e4m3 БЕЗ amax-скейла: порог
// стирания в kernel-units 2^-9 (субнорм) / 2^-6 (норм).

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/ebitengine/purego"
)

var (
	faBwdOnce     sync.Once
	faBwdLib      uintptr
	faBwdErr      error
	gtFaBwdDPre   func(o, do, d uintptr, bh, sl, hd int32, stream uintptr) int32
	gtFaBwdMerged func(q, k, v, do, l, d, dSnat, dST, dV uintptr,
		bh, sl, hd, causal, window int32,
		scale float32, stream uintptr) int32
	gtFaBwdDK func(q, dST, dK uintptr,
		bh, sl, hd, causal, window int32,
		scale float32, stream uintptr) int32
	gtFaBwdDQ func(k, dSnat, dQ uintptr,
		bh, sl, hd, causal, window int32,
		scale float32, stream uintptr) int32
	gtFaBwdRegs func(kernelID int32) int32
)

// FABwdLoad — dlopen libfa_bwd_sm120.so + register symbols.
func FABwdLoad() error {
	faBwdOnce.Do(func() {
		candidates := []string{}
		if p := os.Getenv("GOML_FA_BWD_LIB"); p != "" {
			candidates = append(candidates, p)
		}
		candidates = append(candidates,
			"/data/lib/podman-data/projects/goml/libs/fa_bwd_sm120/libfa_bwd_sm120.so",
			"libs/fa_bwd_sm120/libfa_bwd_sm120.so",
			"./libfa_bwd_sm120.so",
			"/usr/local/lib/libfa_bwd_sm120.so",
		)
		var err error
		for _, path := range candidates {
			if _, statErr := os.Stat(path); statErr != nil {
				continue
			}
			abs, _ := filepath.Abs(path)
			lib, e := purego.Dlopen(abs, purego.RTLD_LAZY)
			if e == nil {
				faBwdLib = lib
				break
			}
			err = e
		}
		if faBwdLib == 0 {
			if err == nil {
				faBwdErr = fmt.Errorf("libfa_bwd_sm120.so not found (set GOML_FA_BWD_LIB)")
			} else {
				faBwdErr = fmt.Errorf("libfa_bwd_sm120.so load: %w", err)
			}
			return
		}
		purego.RegisterLibFunc(&gtFaBwdDPre, faBwdLib, "gt_fa_bwd_d_precompute")
		purego.RegisterLibFunc(&gtFaBwdMerged, faBwdLib, "gt_fa_bwd_merged")
		purego.RegisterLibFunc(&gtFaBwdDK, faBwdLib, "gt_fa_bwd_dk")
		purego.RegisterLibFunc(&gtFaBwdDQ, faBwdLib, "gt_fa_bwd_dq")
		purego.RegisterLibFunc(&gtFaBwdRegs, faBwdLib, "gt_fa_bwd_kernel_regs")
	})
	return faBwdErr
}

// FABwdKernelRegs — query numRegs for kernel_id (0=d_precompute, 1=merged, 2=dk_new, 3=dq_new).
func FABwdKernelRegs(kernelID int) (int, error) {
	if err := FABwdLoad(); err != nil {
		return 0, err
	}
	regs := gtFaBwdRegs(int32(kernelID))
	if regs < 0 {
		return 0, fmt.Errorf("cudaFuncGetAttributes(kernel_id=%d) failed", kernelID)
	}
	return int(regs), nil
}

// FABwdDPrecompute — call gt_fa_bwd_d_precompute.
//   O, dO: FP16 [bh, sl, hd] device ptrs
//   D:     FP32 [bh, sl] device ptr
// Мостовая дисциплина: caller must LockOSThread if bracket'ing multiple calls.
func FABwdDPrecompute(o, dO, d uintptr, bh, sl, hd int, stream uintptr) error {
	if err := FABwdLoad(); err != nil {
		return err
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	rc := gtFaBwdDPre(o, dO, d, int32(bh), int32(sl), int32(hd), stream)
	if rc != 0 {
		return fmt.Errorf("gt_fa_bwd_d_precompute: cudaError %d", rc)
	}
	return nil
}

// FABwdMerged — call gt_fa_bwd_merged (fused ds_gen + dV_p1).
//   Q,K,V: FP8 [bh, sl, hd]
//   dO:    FP16 [bh, sl, hd]
//   L, D:  FP32 [bh, sl]
//   dSnat, dST: FP8 padded stride_ds=(sl+15)&~15 per row [bh, sl_i, sl_j_pad]
//   dV:    FP32 [bh, sl, hd] — must be zero-init
func FABwdMerged(q, k, v, dO, l, d, dSnat, dST, dV uintptr,
	bh, sl, hd, causal, window int, scale float32, stream uintptr) error {
	if err := FABwdLoad(); err != nil {
		return err
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	rc := gtFaBwdMerged(q, k, v, dO, l, d, dSnat, dST, dV,
		int32(bh), int32(sl), int32(hd),
		int32(causal), int32(window),
		scale, stream)
	if rc != 0 {
		return fmt.Errorf("gt_fa_bwd_merged: cudaError %d", rc)
	}
	return nil
}

// FABwdDK — call gt_fa_bwd_dk (essentially dS_T @ Q).
//   Q:    FP8 [bh, sl, hd]
//   dST:  FP8 padded (from FABwdMerged output)
//   dK:   FP32 [bh, sl, hd] — must be zero-init
func FABwdDK(q, dST, dK uintptr, bh, sl, hd, causal, window int,
	scale float32, stream uintptr) error {
	if err := FABwdLoad(); err != nil {
		return err
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	rc := gtFaBwdDK(q, dST, dK, int32(bh), int32(sl), int32(hd),
		int32(causal), int32(window), scale, stream)
	if rc != 0 {
		return fmt.Errorf("gt_fa_bwd_dk: cudaError %d", rc)
	}
	return nil
}

// FABwdDQ — call gt_fa_bwd_dq (essentially dS_nat @ K).
//   K:      FP8 [bh, sl, hd]
//   dSnat:  FP8 padded (from FABwdMerged output)
//   dQ:     FP32 [bh, sl, hd] — must be zero-init
func FABwdDQ(k, dSnat, dQ uintptr, bh, sl, hd, causal, window int,
	scale float32, stream uintptr) error {
	if err := FABwdLoad(); err != nil {
		return err
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	rc := gtFaBwdDQ(k, dSnat, dQ, int32(bh), int32(sl), int32(hd),
		int32(causal), int32(window), scale, stream)
	if rc != 0 {
		return fmt.Errorf("gt_fa_bwd_dq: cudaError %d", rc)
	}
	return nil
}
