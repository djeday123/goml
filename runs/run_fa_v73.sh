#!/bin/bash
# v73 EXPERIMENT: 8 blocks/SM via Br=32 + Bc=32 + 2-warp + launch_bounds(8).
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v73 ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    libs/flash_attention_v73_fp8_forward.cu \
    -o runs/fa_v73_fp8 -lcudart -Xptxas=-v 2>&1 | tail -10

echo ""
echo "=== v69 production baseline ==="
runs/fa_v69_fp8 2>&1 | tail -20

echo ""
echo "=== v72 (4 blocks/SM baseline) ==="
runs/fa_v72_fp8 2>&1 | tail -16

echo ""
echo "=== v73 (8 blocks/SM target) ==="
runs/fa_v73_fp8
