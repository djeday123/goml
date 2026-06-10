#!/bin/bash
# v80a = v72 base + branch-free K/V prefetch (Lever 3 from v79 only).
# Minimum-risk pilot: zero reg cost, must keep ≤128 regs / 4 blocks/SM.
# Goal: does branch-free alone move the needle on bh=16 sl=2048/4096?
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v80a with -Xptxas=-v ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v \
    libs/flash_attention_v80a_fp8_forward.cu \
    -o runs/fa_v80a_fp8 -lcudart 2>&1 | tail -8

echo ""
echo "=== Regs (target: ≤128 for 4 blocks/SM) ==="
echo ""
echo "=== v80a correctness + bench ==="
runs/fa_v80a_fp8

echo ""
echo "=== v72 baseline (for side-by-side) ==="
runs/fa_v72_fp8 2>&1 | grep -E "perf=" | head -15
