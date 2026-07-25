package cuda

// A-LLM-1 G1 (2026-07-25): purego binding для libfa_sm120.so — fa_forward_train
// с LSE выходом для будущего backward. Boевой fa_forward здесь НЕ биндим —
// используется вне goml (fa-blackwell-fp8 репо); наш скоп — только train-путь.
//
// Мостовая дисциплина: FA использует cudaGetDevice (primary context),
// goml.cuda тоже primary через cuDevicePrimaryCtxRetain (Fix A) — контексты
// совпадают. LockOSThread накладывается вызывающим (trainStep).

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// FA status codes (see include/fa_sm120.h).
type FAStatus int32

const (
	FA_OK                    FAStatus = 0
	FA_ERR_INVALID_ARG       FAStatus = 1
	FA_ERR_UNSUPPORTED_ARCH  FAStatus = 2
	FA_ERR_UNSUPPORTED_HD    FAStatus = 3
	FA_ERR_UNSUPPORTED_SHAPE FAStatus = 4
	FA_ERR_CUDA              FAStatus = 5
	FA_ERR_OOM               FAStatus = 6
	FA_ERR_INTERNAL          FAStatus = 7
)

func (s FAStatus) String() string {
	switch s {
	case FA_OK:
		return "OK"
	case FA_ERR_INVALID_ARG:
		return "invalid argument"
	case FA_ERR_UNSUPPORTED_ARCH:
		return "unsupported GPU arch (need sm_120a)"
	case FA_ERR_UNSUPPORTED_HD:
		return "unsupported head_dim"
	case FA_ERR_UNSUPPORTED_SHAPE:
		return "unsupported shape"
	case FA_ERR_CUDA:
		return "CUDA runtime error"
	case FA_ERR_OOM:
		return "out of memory"
	case FA_ERR_INTERNAL:
		return "internal error"
	}
	return fmt.Sprintf("unknown(%d)", int(s))
}

var (
	faLoadOnce   sync.Once
	faLib        uintptr
	faLibErr     error
	faCreate     func(ctxOut *uintptr) int32
	faDestroy    func(ctx uintptr) int32
	faVersion    func() uintptr // returns const char*
	faStatusStr  func(s int32) uintptr
	faLastErr    func(ctx uintptr) uintptr
	faForwardTrn func(ctx, q, k, v, o, lOut uintptr,
		bh, sl, hd, causal, window int32,
		scale float32, stream uintptr) int32
)

// FALoad — one-time dlopen libfa_sm120.so + register symbols.
// Search paths: $GOML_FA_LIB, ./libs/fa_sm120/libfa_sm120.so, /usr/local/lib/libfa_sm120.so.
func FALoad() error {
	faLoadOnce.Do(func() {
		candidates := []string{}
		if p := os.Getenv("GOML_FA_LIB"); p != "" {
			candidates = append(candidates, p)
		}
		candidates = append(candidates,
			"/data/lib/podman-data/projects/goml/libs/fa_sm120/libfa_sm120.so",
			"libs/fa_sm120/libfa_sm120.so",
			"./libfa_sm120.so",
			"/usr/local/lib/libfa_sm120.so",
		)
		var err error
		for _, path := range candidates {
			if _, statErr := os.Stat(path); statErr != nil {
				continue
			}
			abs, _ := filepath.Abs(path)
			lib, e := purego.Dlopen(abs, purego.RTLD_LAZY)
			if e == nil {
				faLib = lib
				break
			}
			err = e
		}
		if faLib == 0 {
			if err == nil {
				faLibErr = fmt.Errorf("libfa_sm120.so not found (set GOML_FA_LIB)")
			} else {
				faLibErr = fmt.Errorf("libfa_sm120.so load: %w", err)
			}
			return
		}
		purego.RegisterLibFunc(&faCreate, faLib, "fa_create")
		purego.RegisterLibFunc(&faDestroy, faLib, "fa_destroy")
		purego.RegisterLibFunc(&faVersion, faLib, "fa_version")
		purego.RegisterLibFunc(&faStatusStr, faLib, "fa_status_str")
		purego.RegisterLibFunc(&faLastErr, faLib, "fa_last_cuda_error")
		purego.RegisterLibFunc(&faForwardTrn, faLib, "fa_forward_train")
	})
	return faLibErr
}

// FAContext — opaque handle.
type FAContext struct {
	ptr uintptr
}

// FACreate — create FA context (probes device for sm_120a).
func FACreate() (*FAContext, error) {
	if err := FALoad(); err != nil {
		return nil, err
	}
	var ptr uintptr
	st := FAStatus(faCreate(&ptr))
	if st != FA_OK {
		if ptr != 0 {
			// Context created with error diagnostic — free it.
			defer faDestroy(ptr)
			msg := ""
			if faLastErr != nil {
				msg = cString(faLastErr(ptr))
			}
			return nil, fmt.Errorf("fa_create: %s (%s)", st, msg)
		}
		return nil, fmt.Errorf("fa_create: %s", st)
	}
	return &FAContext{ptr: ptr}, nil
}

// Destroy — release FA context.
func (c *FAContext) Destroy() {
	if c == nil || c.ptr == 0 {
		return
	}
	faDestroy(c.ptr)
	c.ptr = 0
}

// ForwardTrain — call fa_forward_train.
//   q,k,v: FP8 e4m3 device pointers, [bh, sl, hd] row-major.
//   o:     FP16 device pointer, [bh, sl, hd].
//   lOut:  F32 device pointer, [bh, sl] LSE. Pass 0 to skip L writeback.
//   scale: composed = softmax_scale * scale_Q * scale_K.
//   stream: CUDA stream (0 = default).
//
// LockOSThread should be held by caller (mostовая дисциплина).
func (c *FAContext) ForwardTrain(q, k, v, o, lOut uintptr,
	bh, sl, hd, causal, window int32,
	scale float32, stream uintptr) error {
	if c == nil || c.ptr == 0 {
		return fmt.Errorf("FAContext.ForwardTrain: nil ctx")
	}
	// Prevent Go scheduler migration mid-call (FA uses cudaGetDevice on
	// current thread; primary-ctx sharing with goml.cuda works only when
	// pinned).
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	st := FAStatus(faForwardTrn(c.ptr, q, k, v, o, lOut,
		bh, sl, hd, causal, window, scale, stream))
	if st != FA_OK {
		msg := ""
		if faLastErr != nil {
			msg = cString(faLastErr(c.ptr))
		}
		return fmt.Errorf("fa_forward_train: %s (%s)", st, msg)
	}
	return nil
}

// FAVersion — returns library version string.
func FAVersion() (string, error) {
	if err := FALoad(); err != nil {
		return "", err
	}
	return cString(faVersion()), nil
}

// cString — copy null-terminated C string from uintptr into Go string.
func cString(p uintptr) string {
	if p == 0 {
		return ""
	}
	var n int
	for {
		b := *(*byte)(unsafe.Pointer(p + uintptr(n)))
		if b == 0 {
			break
		}
		n++
		if n > 1024 {
			break
		}
	}
	if n == 0 {
		return ""
	}
	return string(unsafe.Slice((*byte)(unsafe.Pointer(p)), n))
}
