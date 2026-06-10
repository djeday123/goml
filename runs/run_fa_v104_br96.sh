#!/bin/bash
# v104 = Br=96 + FA_THREADS=192 (6 warps × 32) + M_TILES=1.
# 6 warps × 1 m16-tile × 16 rows = Br=96.
# Per-block SMEM: smQ(12K) + smK(8K) + smV(8K) = 28 KB. 2 blocks × 28 = 56 KB ≤ cap.
# Reg budget at 2 blocks: 65536/(192×2) = 170/thread → loose, no spill cliff.
# 12 warps/SM (vs v96's 8, vs v102's 12) — same warp count as v102 LB=3 but in 2 blocks.
#
# HYPOTHESIS: less SMEM bank contention (2 blocks vs 3) → math_pipe/short_scb lower
# → net Eligible gain over v102.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v104 hd=128 Br=96 6-warp ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v104_br96_6warps_fp8_forward.cu \
    -o runs/fa_v104_br96 -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -30

echo ""
echo "=== Reference baseline ==="
echo "  v96: 242 regs, 48.5KB, 2 blocks/SM (8 warps), 568T peak"
echo "  v102: 168 regs + 212B spill, 32KB, 3 blocks/SM (12 warps), 553T (-2.7%)"
echo "  v104 target: ~150-170 regs/thread no spill, 28KB, 2 blocks/SM (12 warps)"
echo ""
echo "=== Run: correctness will fail (smV_T removed inherited from v102), perf only ==="
runs/fa_v104_br96

echo ""
echo "=== Watch ==="
echo "  ptxas regs: ≤ 170 with little spill — 2 blocks/SM viable"
echo "  Peak vs v102 553T:"
echo "    > v102 → less SMEM contention paid off"
echo "    ≈ v102 → same trade-off pattern"
echo "    < v102 → 6-warp overhead worse than 3-block contention"
