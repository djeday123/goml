#!/bin/bash
# v75 STRESS: v73 base + launch_bounds(_, 12). Reg cap 85, was 128 in v73.
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v75 ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    libs/flash_attention_v75_fp8_forward.cu \
    -o runs/fa_v75_fp8 -lcudart -Xptxas=-v 2>&1 | tail -10

echo ""
echo "=== v69 production ==="
runs/fa_v69_fp8 2>&1 | tail -12

echo ""
echo "=== v73 (8 blocks/SM, 123 regs) ==="
runs/fa_v73_fp8 2>&1 | tail -12

echo ""
echo "=== v75 (12 launch_bounds, ~85 reg cap) ==="
runs/fa_v75_fp8
