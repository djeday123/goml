#!/bin/bash
# v103 = v102 + LB=4 hint. SMEM still 32 KB/block — caps at 3 blocks via SMEM limit
# (4 × 32 = 128 KB > 100 KB cap). Real 4 blocks needs Br=64 structural change.
# This probe sees if ptxas squeezes regs further with LB=4 budget (target ≤128 regs).

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v103 hd=128 LB=4 hint ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v103_lb4_hd128_fp8_forward.cu \
    -o runs/fa_v103_lb4 -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -20

echo ""
echo "=== Reference (LB=3 v102): regs 168, spill 212B/192B, stack 112B, 3 blocks/SM loaded ==="
echo "=== v103 LB=4 prediction: more aggressive reg squeeze (target 128), MORE spill ==="
echo ""
echo "=== Run: just bench (corrupt correctness, perf only) ==="
runs/fa_v103_lb4
