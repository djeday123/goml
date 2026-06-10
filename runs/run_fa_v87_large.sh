#!/bin/bash
# v87 large-grid validation — close data gap vs v89.
# v87 bench previously only had bh≤32 configs. Adding bh=64/128 grids that v89
# was validated on (462/466/453T peak) to enable direct comparison.
#
# Goal: determine whether v89's +3.8% advantage on mid grids holds, shrinks,
# or reverses on large grids. If v87 ≈ v89 on large → v87 (160 regs, 8 closer
# to 127 target) becomes better base for future 4-block algorithmic redesign.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v87 (with extended large-grid bench configs) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v87_hd64_ksbatched_fp8_forward.cu \
    -o runs/fa_v87_large -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -40

echo ""
echo "=== Reference (v89 measured on these same configs) ==="
echo "  bh=64  sl=4096  LB=3: ~462T"
echo "  bh=64  sl=8192  LB=3: ~466T (absolute peak)"
echo "  bh=128 sl=2048  LB=3: ~437T"
echo "  bh=128 sl=4096  LB=3: ~453T"
echo ""
echo "  v87 LB=3: 160 regs, 8 closer to 127 (4-block target)"
echo "  v89 LB=3: 168 regs (P-in-regs added 8 for Pf_pair shfl machinery)"
echo ""
echo "=== Run: attrs + correctness + bench (small + large + sliding window) ==="
runs/fa_v87_large
