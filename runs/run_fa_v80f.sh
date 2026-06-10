#!/bin/bash
# v80f: Br=32 + 2-producer + 2-consumer warp specialization.
# Real warp spec architecture: 50% producer ratio. Unexplored regime.
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v80f ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v \
    libs/flash_attention_v80f_fp8_forward.cu \
    -o runs/fa_v80f_fp8 -lcudart 2>&1 | tail -8

echo ""
echo "=== v80f correctness + bench ==="
runs/fa_v80f_fp8

echo ""
echo "=== v80b baseline (Br=64, no warp spec) ==="
runs/fa_v80b_fp8 2>&1 | grep -E "perf=" | head -11

echo ""
echo "=== v80c Phase 3 (Br=96, 1-prod + 3-cons warp spec) ==="
runs/fa_v80c_fp8 2>&1 | grep -E "perf=" | head -11
