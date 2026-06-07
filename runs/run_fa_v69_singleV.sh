#!/bin/bash
# v69_singleV: Door 1 — single-buffer V (drop smV[1]), smV_T intact, occupancy ×2.
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v69_singleV ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    libs/flash_attention_v69_singleV_fp8_forward.cu \
    -o runs/fa_v69_singleV_fp8 -lcudart -Xptxas=-v 2>&1 | tail -10

echo ""
echo "=== v68 baseline (1 block/SM, 8.33% occ) ==="
runs/fa_v68_fp8 2>&1 | tail -16

echo ""
echo "=== v69_singleV (2 blocks/SM, 16.67% occ) ==="
runs/fa_v69_singleV_fp8
