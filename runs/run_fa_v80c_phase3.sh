#!/bin/bash
# v80c Phase 3: producer-warp-only cp.async (no mbarrier yet, just concentrated cp.async).
# Test: does dedicating cp.async issuance to 1 warp matter?
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v80c Phase 3 ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v \
    libs/flash_attention_v80c_fp8_forward.cu \
    -o runs/fa_v80c_fp8 -lcudart 2>&1 | tail -10

echo ""
echo "=== v80c Phase 3 correctness + bench ==="
runs/fa_v80c_fp8

echo ""
echo "=== v79b baseline (for direct compare) ==="
runs/fa_v79b_fp8 2>&1 | grep -E "perf=" | head -11
