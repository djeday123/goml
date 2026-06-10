#!/bin/bash
# v80g: Br=80, 5 compute warps (FA_THREADS=160), no warp spec.
# Data point — Br=80 wave math without warp spec.
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v80g ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v \
    libs/flash_attention_v80g_fp8_forward.cu \
    -o runs/fa_v80g_fp8 -lcudart 2>&1 | tail -8

echo ""
echo "=== v80g correctness + bench ==="
runs/fa_v80g_fp8

echo ""
echo "=== Comparison baselines ==="
echo ""
echo "v80b (Br=64, 4 warps, 3 bl/SM):"
runs/fa_v80b_fp8 2>&1 | grep -E "perf=" | head -11

echo ""
echo "v80c P3 (Br=96, 4 warps + warp spec, 2 bl/SM):"
runs/fa_v80c_fp8 2>&1 | grep -E "perf=" | head -11

echo ""
echo "v79b (Br=128, 4 warps, 2 bl/SM, 401T peak):"
runs/fa_v79b_fp8 2>&1 | grep -E "perf=" | head -11
