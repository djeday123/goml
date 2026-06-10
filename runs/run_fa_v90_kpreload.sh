#!/bin/bash
# v90 = v89 + K-preload (b0/b1 for all 8 nt's into 16 regs before QK MMA loop).
# Targets short_scoreboard 5.50% in v89 by decoupling smK reads from MMA chain.
# Risk: +16 regs/thread → ptxas may exceed LB=3 budget (170 max). Measure first!

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v90 K-preload ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v90_kpreload_fp8_forward.cu \
    -o runs/fa_v90_kpreload -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -40

echo ""
echo "=== v89 reference: 168 regs LB=3, 0 spill, 413T (small) / 466T (large peak) ==="
echo "=== v90 expected: regs may grow to ~180+ → check if 3 blocks still fit (≤170) ==="
echo ""
echo "=== Run: attrs + correctness + bench (small + large grid coverage) ==="
runs/fa_v90_kpreload
