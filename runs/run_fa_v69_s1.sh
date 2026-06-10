#!/bin/bash
# v69_s1 (Stage 1): drop smV_T, byte-gather V from cur_V. smP gets own slot.
# Net SMEM ~unchanged. Isolated test of byte-gather V correctness + conflict.
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v69_s1 ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    libs/flash_attention_v69_s1_fp8_forward.cu \
    -o runs/fa_v69_s1_fp8 -lcudart -Xptxas=-v 2>&1 | tail -10

echo ""
echo "=== v68 baseline ==="
runs/fa_v68_fp8 2>&1 | tail -16

echo ""
echo "=== v69_s1 byte-gather V ==="
runs/fa_v69_s1_fp8
