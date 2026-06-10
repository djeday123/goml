#!/bin/bash
# v79e = v79b with ONE __syncthreads replaced by cuda::barrier (mbarrier).
# Toolchain validation: tests mbarrier works on sm_120a, measures perf overhead.
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v79e with -Xptxas=-v ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v \
    libs/flash_attention_v79e_fp8_forward.cu \
    -o runs/fa_v79e_fp8 -lcudart 2>&1 | tail -8

echo ""
echo "=== v79e correctness + bench ==="
runs/fa_v79e_fp8

echo ""
echo "=== v79b baseline ==="
runs/fa_v79b_fp8 2>&1 | grep -E "perf=" | head -11
