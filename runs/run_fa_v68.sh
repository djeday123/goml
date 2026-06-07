#!/bin/bash
# v68 = Variant C: smV_T stride 64→68 padding to break 32-way write conflict.
# Compares vs v66 baseline.

set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v68 ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    libs/flash_attention_v68_fp8_forward.cu \
    -o runs/fa_v68_fp8 -lcudart -Xptxas=-v 2>&1 | tail -15

echo ""
echo "=== v66 baseline (regression) ==="
runs/fa_v66_fp8 2>&1 | tail -16

echo ""
echo "=== v68 padded smV_T ==="
runs/fa_v68_fp8
