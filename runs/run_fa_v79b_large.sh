#!/bin/bash
# v79b hd=128 large-grid validation.
# v79b currently measured peak ~401T on smaller bench. Add bh=64-128 configs
# (parity with v89 hd=64 large bench) + variance ×3 to see if hd=128 also
# reveals a higher peak on large grids — as hd=64 did exposing 466T at bh=64 sl=8192.
#
# IMPORTANT: hd=128 ≠ hd=64 work load. TFLOPS NOT directly comparable across hd.
# This probe answers ONLY: does v79b peak at 401, or higher on larger grids?
#
# Setup: v79b is launch_bounds(_, 2) — fixed 2 blocks/SM (249 reg-bound, no LB=3).

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v79b (with extended large-grid bench + variance ×3) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v79b_fp8_forward.cu \
    -o runs/fa_v79b_large -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -40

echo ""
echo "=== Reference points ==="
echo "  v79b ref: ~245 regs, 2 blocks/SM, 401T peak (small bench)"
echo "  hd=128 is 2× work per head vs hd=64. NOT comparable across hd."
echo "  Question: does v79b show higher peak on bh=64-128 large grids?"
echo ""
echo "=== Run: correctness + bench (small + large + sliding window) ==="
runs/fa_v79b_large
