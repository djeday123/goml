#!/bin/bash
# v80e = v80b base (Br=64, 3 blocks/SM, V cp.async + transpose_v + mid-iter overlap)
#       + concentrated cp.async to warp 0 only. All 4 warps still compute.
# Test if hybrid warp 0 (compute + 4× cp.async work) is less catastrophic on Br=64
# with 12 warps/SM than on Br=128 with 8 warps/SM (v80d −25%).
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v80e ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v \
    libs/flash_attention_v80e_fp8_forward.cu \
    -o runs/fa_v80e_fp8 -lcudart 2>&1 | tail -8

echo ""
echo "=== v80e correctness + bench ==="
runs/fa_v80e_fp8

echo ""
echo "=== v80b baseline (Br=64 without warp spec) ==="
runs/fa_v80b_fp8 2>&1 | grep -E "perf=" | head -11

echo ""
echo "=== v79b baseline (Br=128 peak champion) ==="
runs/fa_v79b_fp8 2>&1 | grep -E "perf=" | head -11
