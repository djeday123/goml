#!/bin/bash
# Build and run production FA v69 FP8 forward (single-buffer V, 2 blocks/SM).
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v69 ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    libs/flash_attention_v69_fp8_forward.cu \
    -o runs/fa_v69_fp8 -lcudart -Xptxas=-v 2>&1 | tail -8

echo ""
echo "=== v69 production ==="
runs/fa_v69_fp8
