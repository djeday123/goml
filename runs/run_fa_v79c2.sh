#!/bin/bash
# v79c2 = v79b + CORRECT P-in-registers via uniform-nt shfl pattern.
# Tests honest claim: did v79c "fail" because of spill (architectural ceiling)
# or because of shfl bug (broken code)?
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v79c2 with -Xptxas=-v ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v \
    libs/flash_attention_v79c2_fp8_forward.cu \
    -o runs/fa_v79c2_fp8 -lcudart 2>&1 | tail -8

echo ""
echo "=== Regs check ==="
echo "v79b baseline: 249 regs, 0 spill, 16 stack"
echo "v79c (broken): 239 regs, 0 spill, 144 stack"
echo "v79c2 (correct): see above"
echo ""

echo "=== v79c2 correctness + bench ==="
runs/fa_v79c2_fp8

echo ""
echo "=== v79b baseline ==="
runs/fa_v79b_fp8 2>&1 | grep -E "perf=" | head -11
