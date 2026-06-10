#!/bin/bash
# v79c = v79b + P-in-registers. Eliminates smP STS + 2 syncs + smP LDS.
# Replaces with __shfl_sync gather + JIT FP16→FP8 quantize + b32 pack.
# Hypothesis: this is the 4090 winning trick not yet tried on Br=128.
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v79c with -Xptxas=-v ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v \
    libs/flash_attention_v79c_fp8_forward.cu \
    -o runs/fa_v79c_fp8 -lcudart 2>&1 | tail -8

echo ""
echo "=== Regs (target: ≤255 for 2 blocks/SM, v79b baseline=249) ==="
echo ""
echo "=== v79c correctness + bench ==="
runs/fa_v79c_fp8

echo ""
echo "=== v79b baseline (Br=128 + smP SMEM round-trip) ==="
runs/fa_v79b_fp8 2>&1 | grep -E "perf=" | head -11

echo ""
echo "=== v80b (Br=64 small-grid kernel) ==="
runs/fa_v80b_fp8 2>&1 | grep -E "perf=" | head -11
