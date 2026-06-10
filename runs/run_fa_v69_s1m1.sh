#!/bin/bash
# v69_s1m1: Stage 1 (byte-gather V) on M_TILES=1 / 256-thread base.
# Hypothesis: halving per-lane reg state eliminates spill from byte-gather V.
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v69_s1m1 (look for spill stores/loads in ptxas) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    libs/flash_attention_v69_s1m1_fp8_forward.cu \
    -o runs/fa_v69_s1m1_fp8 -lcudart -Xptxas=-v 2>&1 | tail -10

echo ""
echo "=== v68 baseline (M_TILES=2, 4 warps) ==="
runs/fa_v68_fp8 2>&1 | tail -16

echo ""
echo "=== v69_s1 (M_TILES=2 + byte-gather V — спилл) ==="
runs/fa_v69_s1_fp8 2>&1 | tail -16

echo ""
echo "=== v69_s1m1 (M_TILES=1 + byte-gather V) ==="
runs/fa_v69_s1m1_fp8
