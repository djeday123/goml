#!/bin/bash
# v80d = v79b base (Br=128, 4 compute warps) + concentrated cp.async to warp 0.
# Test: does cp.async-concentration mechanism help WITHOUT Br shrink?
# All 4 warps still compute (cover Br=128). Warp 0 ALSO issues cp.async.
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v80d ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v \
    libs/flash_attention_v80d_fp8_forward.cu \
    -o runs/fa_v80d_fp8 -lcudart 2>&1 | tail -8

echo ""
echo "=== v80d correctness + bench ==="
runs/fa_v80d_fp8

echo ""
echo "=== v79b baseline ==="
runs/fa_v79b_fp8 2>&1 | grep -E "perf=" | head -11
