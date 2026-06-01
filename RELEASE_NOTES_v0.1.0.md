# goml v0.1.0

First tagged release of goml — a compound LLM training stack written in
Go, with a CUDA backend that uses **no CGo**. CUDA Driver is loaded at
runtime via `purego`, and the compute kernels ship as fat-binary `.so`
files covering every NVIDIA architecture from Ampere through Blackwell.

## Highlights

* **Hand-tuned FlashAttention forward (`libs/flash_attention_v54.cu`)**
  reaching **153 TFLOPS on a 7B-8K workload on RTX 4090** — 92.5 % of FP16
  peak and 96.2 % of Tri Dao's FA2 reference. The full optimisation journey
  (52 numbered iterations, what worked, what didn't, why) is documented in
  `cuda-future/` along with SASS dumps comparing v54 with FA2.
* **FP8 GEMM** (`libs/fp8_gemm.cu`) at ~587 TFLOPS stock and ~590 TFLOPS
  at OC 3045 MHz on RTX 4090.
* **Pure-Go ML stack:** tensors, reverse-mode autograd, AdamW with decoupled
  weight decay and cosine schedule, BPE + byte tokenizer, transformer
  blocks with RoPE + KV cache, end-to-end trainer (`cmd/simpletrain`
  trains on Shakespeare).
* **No CGo.** CUDA Driver API and cuBLAS are loaded at runtime through
  `github.com/ebitengine/purego`; the only required system library is
  `libcuda.so` from the NVIDIA driver.
* **Multi-arch fat-binary build.** One `make cuda` produces `.so` files
  with native SASS for `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100`,
  `sm_120`, plus embedded PTX (`compute_120`) for forward-compatibility.
  Same library runs on RTX 3090, RTX 4090, H100, RTX 6000 Blackwell, and
  any future SM the driver can JIT.
* **Init banner** confirms which GPU and architecture were detected:
  `[GoML] CUDA backend initialized: NVIDIA GeForce RTX 4090 (Ada Lovelace, sm_89)`.

## GPU compatibility

| Family | Compute capability | Cards |
|---|---|---|
| Ampere | `sm_80`, `sm_86` | A100, RTX 3090, A40 |
| Ada | `sm_89` | RTX 4090, L40 |
| Hopper | `sm_90` | H100, H200 |
| Blackwell DC | `sm_100` | B100, B200 |
| Blackwell consumer/workstation | `sm_120` | RTX PRO 6000 Blackwell, RTX 5090 / 5080 |
| Future | (PTX JIT) | sm_120+ via driver |

Requires CUDA 12.8+ at build time for native Blackwell SASS. Older
toolkits fall back to PTX-only forward compat — code still runs on
Blackwell GPUs via runtime JIT.

## Install

```bash
git clone https://github.com/djeday123/goml.git
cd goml
make cuda            # fat-binary build, all SMs the toolkit supports
export GOML_LIBS_DIR="$(pwd)/libs"
make build           # go build ./...
```

For finer control:

```bash
make cuda-blackwell   # Blackwell-only build (fastest)
make cuda-ampere      # Ampere/Ada only
make info             # show detected GPU + nvcc + libs status
```

## What's NOT in v0.1.0

* **CUDA backward FlashAttention** — the production training loop uses
  the manual `nn/backward_attn.go` path. A custom backward kernel is
  next on the roadmap.
* **FP8 attention** — Blackwell's FP8 tensor cores could push attention
  to ~600 TFLOPS but the kernel is not yet written.
* **Prebuilt `.so` binaries** — users build locally because architecture
  -specific fat binaries can balloon to hundreds of MB. A future release
  may publish per-GPU-class tarballs via GitHub Releases.
* **Distributed / multi-GPU training.**

## How the FlashAttention v54 win was found

The single optimisation that beat v20 (151 TFLOPS) into v54 (153 TFLOPS)
was replacing `__expf` (FP32 SFU) with `ex2.approx.f16x2` (native
`MUFU.EX2.F16` on SM89+). Source: the open-source FlashInfer project.
The throughput gain isn't from the SFU itself — both end up emitting
the same number of `MUFU` ops — but from moving the F2F float→half
convert off the critical path, which gives nvcc more freedom to
interleave LDSM and HMMA. Full analysis with `cuobjdump -sass` diffs
is in `cuda-future/`.

See `CHANGELOG.md` for the per-commit breakdown, and `GOML_FIXES.md` for
the audit report (with the false positives that the audit raised but
that turned out to be correct on close reading — those are documented
too, on principle).
