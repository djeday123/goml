# Changelog

All notable changes to goml are documented here. Format roughly follows
[Keep a Changelog](https://keepachangelog.com/). Versions use
[Semantic Versioning](https://semver.org/).

## [Unreleased]

— v0.1.0 cut on 2026-05-24; no unreleased changes at this time.

---

## [v0.1.0] — 2026-05-24

First tagged release. Production-ready FlashAttention forward kernel and
FP8 GEMM, no-CGo CUDA stack, and a fat-binary build system covering every
NVIDIA architecture from Ampere through Blackwell.

### Added — multi-architecture CUDA build (commit `8854360`)

* **`scripts/build_cuda.sh`** — universal builder that detects the
  installed CUDA toolkit and produces fat-binary `.so` files containing
  native SASS for `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100`, `sm_120`
  plus embedded PTX (`compute_120,compute_120`) for forward-compatibility
  with future architectures. Older toolkits gracefully drop the Blackwell
  targets while keeping PTX-only forward compat — the result still runs on
  Blackwell via the driver's JIT compiler.
* **`Makefile`** with targets `cuda`, `cuda-blackwell`, `cuda-ampere`,
  `build`, `test`, `race`, `bench`, `info`, `clean`.
* **`backend/cuda/arch.go`** — `archName(major, minor)`, `SMTag`,
  `LibsDir()` that searches `$GOML_LIBS_DIR → ./libs → <exe>/libs →
  /usr/local/lib/goml`, and `resolveLib()` for all `purego.Dlopen` call
  sites. The init log now prints e.g.
  `[GoML] CUDA backend initialized: NVIDIA GeForce RTX 4090 (Ada Lovelace, sm_89)`.
* **`README.md`** — full rewrite with a GPU compatibility matrix, install
  instructions for `go get`, per-architecture build targets, and a
  project-layout overview.

### Fixed — ops/ autograd path completion (commit `2b384b7`)

The `ops/` package had several activations marked as differentiable but
without a registered `GradFn`. These now compute correct gradients:

* **`ReLU.Backward`** — applies the `x > 0` mask (previously returned the
  upstream gradient unchanged with a `TODO` comment).
* **`Softmax`** — new `softmaxGradFn` implementing the Jacobian
  `dx = s · (g − sum(g · s, axis))` for any rank.
* **`Gelu`** — `geluGradFn` for the tanh-approximation that the CUDA
  `gelu_f32` kernel uses.
* **`Silu`** — `siluGradFn` implementing `dy/dx = σ(x) · (1 + x · (1 − σ(x)))`.

A full audit was performed and documented in `GOML_FIXES.md`. Most
findings from the automated audit turned out to be false positives on
manual review (softmax double-scale, AdamW coupling, CrossEntropy pad
mask, final-norm reconstruction, RoPE backward order, CUDA stride
swap, embedding gradient accumulation — all verified correct).

### Known state of the FlashAttention research line

* **Production:** `libs/flash_attention_v54.cu` →
  `libs/libflash_attention_v54.so`. 153 TFLOPS on 7B-8K on RTX 4090,
  92.5 % of peak, 96.2 % of Tri Dao's FA2.
* The remaining 3.8 % gap to FA2 is at the SASS scheduling level
  (clustered LDSM/HMMA vs. interleaved LDSM × HMMA). Closing it would
  require hand-written inline PTX, ~200-300 lines of fragile assembly —
  intentionally out of scope for v0.1.0.
* The v54-over-v20 win comes from a single technique: replacing
  FP32 `__expf` with `ex2.approx.f16x2` (native `MUFU.EX2.F16` on SM89+).
  Source of the idea: the open-source FlashInfer project.

### Known state of FP8 GEMM

* **Production:** `libs/fp8_gemm.cu` → `libs/libfp8gemm.so`, dual-mode
  kernel (original + singlesync). ~587 TFLOPS stock on RTX 4090, up to
  ~590 with OC at 3045 MHz.

### Known limitations in v0.1.0

* **Backward FlashAttention** is not yet a CUDA kernel; the manual
  `nn/backward_attn.go` path is used during training. Forward kernel v54
  is the only attention path with hand-tuned CUDA.
* **No FP8 attention** kernel. On Blackwell the FP8 tensor cores could
  deliver ~600 TFLOPS for attention but this path is not wired up.
* No prebuilt `.so` files are shipped — users build them locally with
  `make cuda`. (A future release may publish prebuilt binaries for
  common GPU classes via GitHub Releases.)

[Unreleased]: https://github.com/djeday123/goml/compare/v0.1.0...HEAD
[v0.1.0]: https://github.com/djeday123/goml/releases/tag/v0.1.0
