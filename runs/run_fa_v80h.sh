#!/bin/bash
# v80h: Br=80, 1-producer + 5-consumer, FA_THREADS=192.
# Real warp spec on non-sweet-spot Br=80 geometry.
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v80h ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v \
    libs/flash_attention_v80h_fp8_forward.cu \
    -o runs/fa_v80h_fp8 -lcudart 2>&1 | tail -8

echo ""
echo "=== v80h correctness + bench ==="
runs/fa_v80h_fp8

echo ""
echo "=== v80g baseline (Br=80, 5 warps, no warp spec) ==="
runs/fa_v80g_fp8 2>&1 | grep -E "perf=" | head -11
