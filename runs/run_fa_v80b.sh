#!/bin/bash
# v80b = v72 base + cp.async V + transpose_v + V mid-iter overlap + 3 blocks/SM.
# Honest test of "Br=64 + all v79b patches".
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v80b with -Xptxas=-v ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v \
    libs/flash_attention_v80b_fp8_forward.cu \
    -o runs/fa_v80b_fp8 -lcudart 2>&1 | tail -8

echo ""
echo "=== Regs (target: ≤170 for 3 blocks/SM) ==="
echo ""
echo "=== v80b correctness + bench ==="
runs/fa_v80b_fp8

echo ""
echo "=== Baselines for comparison ==="
echo ""
echo "--- v80a (4 blocks, no V overlap) ---"
runs/fa_v80a_fp8 2>&1 | grep -E "perf=" | head -11

echo ""
echo "--- v79b (Br=128, 2 blocks, full patches) ---"
runs/fa_v79b_fp8 2>&1 | grep -E "perf=" | head -11
