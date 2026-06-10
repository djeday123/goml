#!/bin/bash
# v79b = v79 with f16 P_local refactor. P stored as __half2[8][M_TILES] (16 b32)
# instead of float[8][M_TILES][4] (64 b32). Save ~48 regs/thread if ptxas was
# holding P_local in regs. Measure regs before/after, correctness, perf.
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v79b with -Xptxas=-v ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v \
    libs/flash_attention_v79b_fp8_forward.cu \
    -o runs/fa_v79b_fp8 -lcudart 2>&1 | tail -8

echo ""
echo "=== Compare regs: v79=245 → v79b=? ==="
echo ""
echo "=== v79b correctness + bench ==="
runs/fa_v79b_fp8

echo ""
echo "=== v79 baseline (side-by-side) ==="
runs/fa_v79_fp8 2>&1 | grep -E "perf=|max_diff" | head -20
