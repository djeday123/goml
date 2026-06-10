#!/bin/bash
# v80c Phase 1: Br=96 with consumer-only compute paths.
# Phase 1: warp 0 idle on compute (no producer role yet), warps 1-3 do all compute.
# Goal: build clean + correctness 8/8 PASS to validate Br=96 + warp gating works.
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v80c Phase 1 with -Xptxas=-v ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v \
    libs/flash_attention_v80c_fp8_forward.cu \
    -o runs/fa_v80c_fp8 -lcudart 2>&1 | tail -12

echo ""
echo "=== v80c Phase 1 correctness + bench ==="
runs/fa_v80c_fp8

echo ""
echo "=== v79b baseline (Br=128, 4 consumer warps) ==="
runs/fa_v79b_fp8 2>&1 | grep -E "perf=" | head -11
