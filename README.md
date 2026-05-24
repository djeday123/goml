# goml

Compound LLM system written in pure Go with no-CGo CUDA support via
[purego](https://github.com/ebitengine/purego). Targets are tensors,
autograd, transformer training, and hand-tuned CUDA kernels (FlashAttention,
FP8 GEMM, transformer kernels).

```go
import "github.com/djeday123/goml"
```

## GPU compatibility

The CUDA backend ships as a **fat binary** that targets every NVIDIA
architecture from Ampere onward, plus PTX for forward-compatibility:

| Family | Compute capability | Cards | Notes |
|---|---|---|---|
| Ampere     | `sm_80`, `sm_86` | A100, RTX 3090, A40 | native SASS |
| Ada        | `sm_89`          | RTX 4090, L40       | native SASS |
| Hopper     | `sm_90`          | H100, H200          | native SASS |
| Blackwell  | `sm_100`         | B100, B200          | native SASS (requires CUDA 12.8+) |
| Blackwell  | `sm_120`         | RTX PRO 6000 Blackwell, RTX 5090 / 5080 | native SASS (requires CUDA 12.8+) |
| future     | (PTX JIT)        | sm_120+             | JIT-compiled by the driver |

If you build on a machine without CUDA 12.8+, the script falls back to
PTX-only forward compatibility — the resulting `.so` files still work on
Blackwell GPUs via runtime JIT (first-call delay of ~hundreds of ms).

## Quick start

```bash
git clone https://github.com/djeday123/goml.git
cd goml

# Build CUDA libs (fat binary covering all supported architectures)
make cuda

# Or only the architecture you care about — faster build:
make cuda-blackwell    # sm_120 only
make cuda-ampere       # sm_80 + sm_89 only

# Build Go code
make build
```

Then add `libs/` to the dynamic loader path or set `GOML_LIBS_DIR`:

```bash
export GOML_LIBS_DIR="$(pwd)/libs"
# or:
export LD_LIBRARY_PATH="$(pwd)/libs:$LD_LIBRARY_PATH"
```

The Go code also automatically searches `./libs`, `<exe-dir>/libs`, and
`/usr/local/lib/goml` if `$GOML_LIBS_DIR` is unset — see `backend/cuda/arch.go`.

## Requirements

| | Minimum | Recommended |
|---|---|---|
| Go            | 1.22                | 1.22+ |
| CUDA Toolkit  | 12.0 (for nvcc)     | 12.8+ for Blackwell native SASS |
| NVIDIA driver | 525 (CUDA 12 base)  | 570+ for Blackwell |
| GPU           | Ampere `sm_80`+     | any of the above |

No CGo, no Python. CUDA is loaded entirely at runtime via `dlopen` of
`libcuda.so` plus our prebuilt fat-binary `.so` files in `libs/`.

## Installing as a dependency

```bash
go get github.com/djeday123/goml@latest
```

You then need to either:
1. **Clone and `make cuda`** in a path the target machine can read, and set
   `GOML_LIBS_DIR` to that directory; or
2. **Download a prebuilt release tarball** from
   https://github.com/djeday123/goml/releases (when available) and unpack
   to `/usr/local/lib/goml`.

The Go module proper does not embed binary `.so` files — they are
GPU-architecture specific and may be 100s of MB once fattened across
five SMs.

## Architecture overview

```
github.com/djeday123/goml/
├── core/         primitive types (DType, Shape)
├── tensor/       tensor type, strides, autograd hooks
├── autograd/     reverse-mode autograd graph
├── ops/          element-wise / matmul / softmax / loss
├── nn/           Linear, MHA (with RoPE + KV cache), FFN (SwiGLU), TransformerBlock, LLM
├── optim/        AdamW with decoupled weight decay, gradient clipping, cosine LR schedule
├── train/        Trainer wraps Forward → Loss → Backward → step
├── tokenizer/    byte tokenizer + BPE with shared special-token IDs
├── backend/
│   ├── cpu/      naive CPU fallback
│   └── cuda/     purego-bound CUDA Driver API + custom PTX kernels + libcublas glue
├── libs/         .cu sources + built .so fat binaries (see below)
├── libs1/        cuBLAS/cuBLASLt C wrappers + FP8-GEMM experiments
├── libs2/        FP8 GEMM v10+ (production singlesync kernel)
├── cuda-future/  research: SASS analysis, register comparisons (sm_89)
└── scripts/      build_cuda.sh (multi-arch fatbin builder)
```

### What ships in `libs/`

After `make cuda` you get a flat directory of fat-binary `.so` files:

- `libfp8gemm.so`           — FP8 GEMM (singlesync production kernel, 587 T on RTX 4090)
- `libtransformer.so`       — fused RMSNorm, SwiGLU, RoPE, basic attention
- `libflash_attention_v54.so` — Flash Attention forward (153 T on RTX 4090, 96.2% FA2)
- `libcublas_wrapper.so`    — single-pointer entry to `cublasGemmEx`
- `libcublaslt_wrapper.so`  — single-pointer entry to FP8 `cublasLtMatmul`

The PTX kernels in `backend/cuda/kernels.go` and `kernels_b.go` target
`sm_80`; they JIT-compile to whatever SM is present at runtime. No
recompile is needed there even when you move to Blackwell.

## Useful Makefile targets

```bash
make            # cuda + build
make cuda       # fat-binary CUDA build (default — all supported SMs)
make build      # go build ./...
make test       # go test ./...
make race       # go test -race ./...
make bench      # benchmarks
make info       # show detected GPU + nvcc version + libs status
make clean      # remove .so and Go build cache
```

## Verifying installation

```bash
make info
```

…should print the GPU name, compute capability, nvcc version, and which
`.so` files are present. If you want a runtime sanity check from Go:

```go
import "github.com/djeday123/goml/backend/cuda"

// QueryDevice(0) etc. — backend/cuda/driver.go for details
```

## Status

| Component | Notes |
|---|---|
| Forward training | Working end-to-end (Shakespeare in `cmd/simpletrain`) |
| FlashAttention forward | v54 production, 153 TFLOPS on RTX 4090 |
| FP8 GEMM | singlesync, ~587 TFLOPS on RTX 4090 |
| Backward FlashAttention | not yet — uses manual `nn/backward*.go` path |
| Blackwell native targets | supported via fat binary when built with CUDA 12.8+ |

See `GOML_FIXES.md` for the audit + bug-fix history.

## License

(set this when ready)
