#!/bin/bash
# v72 EXPERIMENT: 4 blocks/SM via Br=64 + door 2 + single-K + launch_bounds(4).
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v72 ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    libs/flash_attention_v72_fp8_forward.cu \
    -o runs/fa_v72_fp8 -lcudart -Xptxas=-v 2>&1 | tail -10

echo ""
echo "=== v69 production baseline ==="
runs/fa_v69_fp8 2>&1 | tail -20

echo ""
echo "=== v72 (4 blocks/SM target) ==="
runs/fa_v72_fp8
