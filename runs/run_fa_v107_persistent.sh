#!/bin/bash
# v107 = v96 + persistent (grid-stride) kernel for wave-tail equalization.
# Launch min(total_tiles, 2 × SMs) blocks. Each block grid-strides over Q-tiles.
# Expected: +3-10% on mid grids with wave-tail loss. Risk: outer loop may
# inflate regs and break 2 blocks/SM budget.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v107 hd=128 persistent ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v107_persistent_hd128_fp8_forward.cu \
    -o runs/fa_v107_persistent -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -30

echo ""
echo "=== Reference: v96 = 242 regs no spill, 49.7 KB SMEM, 568T peak / 564T sustained ==="
echo "=== v107 reg budget: keep ≤ 256 to maintain 2 blocks/SM (4 warps × 256 = 1024, 2 × 1024 = 2048 < 65536 ✓) ==="
echo "    But effective constraint is launch_bounds(128, 2) → 256 max @ 2 blocks."
echo "    If regs > 242, outer-loop overhead degrades scheduling — watch carefully."
echo ""
echo "=== Run: full correctness + bench ==="
runs/fa_v107_persistent
