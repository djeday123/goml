#!/bin/bash
# v86 hd=64 + tree-reduced softmax (Option C, no extra accumulators).
# Phase B rmax: sequential fmaxf chain (depth 8) → log2 tree (depth 3)
# Phase E2 ns sum: sequential add chain → log2 tree
# Target: wait stall 27% drop via softmax dep chain reduction.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v86 tree-reduced softmax (LB=2 and LB=3, -lineinfo) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v86_hd64_treered_fp8_forward.cu \
    -o runs/fa_v86_treered_fp8 -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -40

echo ""
echo "=== Run: attrs + correctness ×8×2 + A/B perf ×11×2 ×variance 3 ==="
runs/fa_v86_treered_fp8
