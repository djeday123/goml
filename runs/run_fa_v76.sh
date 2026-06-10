#!/bin/bash
# v76 REAL 12 blocks/SM via smQ↔smV_T overlap → 8 KB/block.
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v76 ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    libs/flash_attention_v76_fp8_forward.cu \
    -o runs/fa_v76_fp8 -lcudart -Xptxas=-v 2>&1 | tail -10

echo ""
echo "=== v69 production (2 blocks/SM, 338T peak) ==="
runs/fa_v69_fp8 2>&1 | tail -12

echo ""
echo "=== v72 (4 blocks/SM, +22% small grids) ==="
runs/fa_v72_fp8 2>&1 | tail -12

echo ""
echo "=== v76 (real 12 blocks/SM target) ==="
runs/fa_v76_fp8
