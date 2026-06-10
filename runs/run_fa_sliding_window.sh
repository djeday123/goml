#!/bin/bash
# Phase A: sliding window early-skip added to v69 (production) and v72 (small-grid).
# Quick check: causal sliding window on representative bench configs.
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v69 (with sliding window) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    libs/flash_attention_v69_fp8_forward.cu \
    -o runs/fa_v69_fp8 -lcudart -Xptxas=-v 2>&1 | tail -6

echo ""
echo "=== Build v72 (with sliding window) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    libs/flash_attention_v72_fp8_forward.cu \
    -o runs/fa_v72_fp8 -lcudart -Xptxas=-v 2>&1 | tail -6

echo ""
echo "=== v69 (Br=128 production) — correctness + bench ==="
runs/fa_v69_fp8

echo ""
echo "=== v72 (Br=64 small-grid optimizer) — correctness + bench ==="
runs/fa_v72_fp8
