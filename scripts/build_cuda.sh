#!/usr/bin/env bash
# build_cuda.sh — universal multi-architecture CUDA build for goml.
#
# Produces fat-binary .so files in libs/ that work on:
#   - Ampere   (sm_80, sm_86)  — A100, RTX 3090, A40
#   - Ada      (sm_89)         — RTX 4090, L40
#   - Hopper   (sm_90)         — H100, H200
#   - Blackwell (sm_120)       — RTX PRO 6000 Blackwell Workstation, RTX 5090
#   - any future SM            — via embedded PTX, JIT-compiled by the driver
#
# Targets requiring CUDA 12.8+ (sm_120) are skipped automatically when an
# older toolkit is detected, with a clear warning.
#
# Usage:
#   ./scripts/build_cuda.sh              # build all libs
#   ./scripts/build_cuda.sh fp8gemm      # build just the fp8 GEMM lib
#   GOML_GENCODE_EXTRA="..." ./build_cuda.sh   # pass extra gencode flags
#
# Output:
#   libs/libfp8gemm.so
#   libs/libtransformer.so
#   libs/libflash_attention_v54.so   (FA forward, production)
#   libs/libcublas_wrapper.so        (cuBLAS shim, from libs1/cublas_wrapper.c)

set -e

cd "$(dirname "$0")/.."

# ─── 1. Detect CUDA version ─────────────────────────────────────────────────

if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not found in PATH. Install CUDA Toolkit 12.0+ (12.8+ for Blackwell)."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)
CUDA_MAJOR=${CUDA_VERSION%.*}
CUDA_MINOR=${CUDA_VERSION#*.}

echo "=== Building goml CUDA libraries ==="
echo "nvcc:    $(nvcc --version | head -4 | tail -1)"
echo "CUDA:    ${CUDA_VERSION}"

# ─── 2. Compose --gencode flags by toolkit capability ───────────────────────

GENCODE_FLAGS=""
add_gencode() {
    GENCODE_FLAGS+=" -gencode arch=$1,code=$2"
}

# Always available (CUDA 11.0+)
add_gencode compute_80 sm_80    # Ampere A100
add_gencode compute_86 sm_86    # Ampere RTX 3090, A40

# CUDA 11.8+
if [ "$CUDA_MAJOR" -gt 11 ] || ([ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]); then
    add_gencode compute_89 sm_89    # Ada Lovelace RTX 4090, L40
else
    echo "WARN: CUDA $CUDA_VERSION < 11.8 — skipping sm_89 (Ada/RTX 4090)"
fi

# CUDA 12.0+
if [ "$CUDA_MAJOR" -ge 12 ]; then
    add_gencode compute_90 sm_90    # Hopper H100, H200
fi

# CUDA 12.8+ — Blackwell
HAS_BLACKWELL=0
if [ "$CUDA_MAJOR" -gt 12 ] || ([ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 8 ]); then
    add_gencode compute_100 sm_100  # Blackwell datacenter (B100/B200)
    add_gencode compute_120 sm_120  # Blackwell workstation/consumer (RTX PRO 6000, RTX 5090)
    # PTX for forward compat with sm_120+
    add_gencode compute_120 compute_120
    HAS_BLACKWELL=1
else
    echo "WARN: CUDA $CUDA_VERSION < 12.8 — skipping sm_100 + sm_120 (Blackwell)"
    echo "      Code will still run on Blackwell via JIT'd PTX, but build with CUDA 12.8+ for native SASS."
    # Fall back to PTX-only forward-compat from sm_90
    if [ "$CUDA_MAJOR" -ge 12 ]; then
        add_gencode compute_90 compute_90
    else
        add_gencode compute_86 compute_86
    fi
fi

# User-supplied additions
GENCODE_FLAGS+=" ${GOML_GENCODE_EXTRA:-}"

echo "gencode: $(echo $GENCODE_FLAGS | tr -s ' ')"
echo

# ─── 3. nvcc / gcc flag defaults ────────────────────────────────────────────

NVCC_BASE="-O3 -std=c++17 --shared -Xcompiler -fPIC ${GENCODE_FLAGS}"

# Build targets — each (source, output, extra_flags)
build_so() {
    local src="$1"
    local out="$2"
    local extra="${3:-}"
    if [ ! -f "$src" ]; then
        echo "  SKIP: $src not found"
        return 0
    fi
    echo "  --> $out"
    nvcc ${NVCC_BASE} ${extra} "$src" -o "$out"
}

build_c_so() {
    local src="$1"
    local out="$2"
    local extra="${3:-}"
    if [ ! -f "$src" ]; then
        echo "  SKIP: $src not found"
        return 0
    fi
    echo "  --> $out (C wrapper)"
    gcc -O3 -shared -fPIC -I/usr/local/cuda/include ${extra} "$src" -o "$out"
}

TARGET="${1:-all}"

case "$TARGET" in
    all|fp8gemm)
        echo "=== fp8 GEMM ==="
        build_so libs/fp8_gemm.cu libs/libfp8gemm.so "-lcublas"
        ;;
esac

case "$TARGET" in
    all|transformer)
        echo "=== Transformer kernels ==="
        build_so libs/transformer_kernels.cu libs/libtransformer.so
        ;;
esac

case "$TARGET" in
    all|fa|flash_attention)
        echo "=== Flash Attention (production v54) ==="
        build_so libs/flash_attention_v54.cu libs/libflash_attention_v54.so
        # Symlink to a stable name expected by Go code if needed
        if [ -f libs/libflash_attention_v54.so ]; then
            ln -sf libflash_attention_v54.so libs/libflash_attention.so
        fi
        ;;
esac

case "$TARGET" in
    all|cublas_wrapper)
        echo "=== cuBLAS wrapper ==="
        build_c_so libs1/cublas_wrapper.c libs/libcublas_wrapper.so "-L/usr/local/cuda/lib64 -lcublas"
        ;;
esac

case "$TARGET" in
    all|cublaslt_wrapper)
        echo "=== cuBLASLt wrapper ==="
        if [ -f libs1/cublas_lt_wrapper_v3.c ]; then
            build_c_so libs1/cublas_lt_wrapper_v3.c libs/libcublaslt_wrapper.so "-L/usr/local/cuda/lib64 -lcublasLt"
        elif [ -f libs1/cublas_lt_wrapper_v2.c ]; then
            build_c_so libs1/cublas_lt_wrapper_v2.c libs/libcublaslt_wrapper.so "-L/usr/local/cuda/lib64 -lcublasLt"
        elif [ -f libs1/cublas_lt_wrapper.c ]; then
            build_c_so libs1/cublas_lt_wrapper.c libs/libcublaslt_wrapper.so "-L/usr/local/cuda/lib64 -lcublasLt"
        fi
        ;;
esac

echo
echo "=== Built libraries ==="
ls -lh libs/*.so 2>/dev/null || echo "(no .so files produced)"

if [ "$HAS_BLACKWELL" -eq 1 ]; then
    echo
    echo "✓ Blackwell (sm_120) targets included natively."
else
    echo
    echo "ℹ Built without native Blackwell support. Will JIT-compile from PTX on Blackwell GPUs."
fi

echo
echo "=== Install hint ==="
echo "  export LD_LIBRARY_PATH=\"\$(pwd)/libs:\$LD_LIBRARY_PATH\""
echo "  # or set GOML_LIBS_DIR=\$(pwd)/libs   (recognised by the cuda backend loader)"
