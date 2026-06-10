#!/bin/bash
# v106 = v105 + smV_T + transpose_v RESTORED for correctness.
# v102/v104/v105 all silently FAILED correctness (PV read from smV with wrong layout).
# Goal: pass 8/8 correctness AND retain v105's stall-reduction wins (vs v104).

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v106 hd=128 Br=96 6-warp P-in-regs +smV_T (correctness) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v106_correct_br96_fp8_forward.cu \
    -o runs/fa_v106_correct -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -30

echo ""
echo "=== References ==="
echo "  v96  : 242 regs, 49.7 KB, 2 blocks/SM (8 warps), 568T peak (production champion)"
echo "  v104 : 156 regs no spill, 28 KB,  2 blocks (12 warps), 511T peak — correctness FAIL"
echo "  v105 : 151 regs no spill, 28 KB,  2 blocks (12 warps), 549T peak — correctness FAIL"
echo "  v106 target: ~155-165 regs no spill, 36.7 KB, 2 blocks/SM, correctness 8/8 PASS"
echo ""
echo "=== Run: full correctness + bench ==="
runs/fa_v106_correct
