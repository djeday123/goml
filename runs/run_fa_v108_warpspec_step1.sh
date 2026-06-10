#!/bin/bash
# v108 STEP 1 of warp-specialization probe for hd=128 sm_120a.
# Br=64, 2 producer + 2 consumer warps, K=2 V=2 stages, smV_T preserved.
# Goal: correctness 8/8 PASS + measure baseline. Perf may be below v96 (overhead
# of role coordination, 2 fewer MMA warps per block).
#
# Diagnostic interpretation:
#   wait↓ + perf close to v96 → warp-spec overlap working, iterate (step 2+)
#   wait↓ + perf <90% v96    → role overhead too high, tune balance
#   wait unchanged           → producer not overlapping → debug pipeline

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v108 STEP 1 warp-spec hd=128 Br=64 ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v108_warpspec_step1_hd128_fp8_forward.cu \
    -o runs/fa_v108_warpspec_step1 -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -30

echo ""
echo "=== Reference: v96 = 242 regs no spill, 48.5 KB SMEM, 568 peak / 564 mean ==="
echo "    v96 NCu PEAK: wait 37.77% / Eligible 32.96%"
echo ""
echo "=== v108 expected: same 48.5 KB SMEM, ~245-260 regs (extra logic for role split) ==="
echo "    LB=(128, 2). If regs > 256 → won't fit 2 blocks/SM."
echo ""
echo "=== Run: full correctness + bench ==="
runs/fa_v108_warpspec_step1
