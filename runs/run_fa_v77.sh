#!/bin/bash
# v77: v69 + cooperative transpose_v fixes 62.5% SMEM store bank conflict.
# Build + correctness + bench + side-by-side with v69 binary.
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v77 ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    libs/flash_attention_v77_fp8_forward.cu \
    -o runs/fa_v77_fp8 -lcudart -Xptxas=-v 2>&1 | tail -8

echo ""
echo "=== v77 correctness + bench ==="
runs/fa_v77_fp8

echo ""
echo "=== v69 baseline (for side-by-side) ==="
runs/fa_v69_fp8 2>&1 | grep -E "perf=|max_diff" | head -20
