#!/bin/bash
# v78: V[N+1] prefetch moved INTO mid-iter, targeting dead smK[buf] slot.
# Overlaps V load with softmax + PV (~80% iter time).
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v78 ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    libs/flash_attention_v78_fp8_forward.cu \
    -o runs/fa_v78_fp8 -lcudart -Xptxas=-v 2>&1 | tail -8

echo ""
echo "=== v78 correctness + bench ==="
runs/fa_v78_fp8

echo ""
echo "=== v69 baseline (side-by-side) ==="
runs/fa_v69_fp8 2>&1 | grep -E "perf=|max_diff" | head -20
