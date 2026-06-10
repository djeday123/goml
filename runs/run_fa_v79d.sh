#!/bin/bash
# v79d = v79b with single-K + double-V layout swap. Same total SMEM 48.5KB.
# Tests: does explicit double-V beat v79b's K-buf-aliasing trick for V?
# K loses early-prefetch (was line 286 BEFORE QK, now line 332 AFTER QK).
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v79d with -Xptxas=-v ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v \
    libs/flash_attention_v79d_fp8_forward.cu \
    -o runs/fa_v79d_fp8 -lcudart 2>&1 | tail -8

echo ""
echo "=== Regs (target: ≤255 for 2 blocks/SM, v79b=249) ==="
echo ""
echo "=== v79d correctness + bench ==="
runs/fa_v79d_fp8

echo ""
echo "=== v79b baseline (Br=128 + double-K + single-V + K-buf-aliasing) ==="
runs/fa_v79b_fp8 2>&1 | grep -E "perf=" | head -11
