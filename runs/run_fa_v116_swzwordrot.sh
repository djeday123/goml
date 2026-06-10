#!/bin/bash
# v116 = v111 + word-position-rotation swizzle in swz_byte_smvt.
# Mathematical 4-way write conflict → 0-way fix.
# Reads: identical conflict count under uniform per-nt word_rot.
# Target: short_scoreboard ↓, uncoalesced ↓, perf neutral-to-positive.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v116 word-rot swizzle ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v116_swzwordrot_hd128_fp8_forward.cu \
    -o runs/fa_v116_swzwordrot -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -30

echo ""
echo "=== References ==="
echo "  v111 baseline: 240 regs, 484T mean PEAK, UncoalescedShared 40.2% (STS transpose 30%)"
echo "  v116 target:   ≤+2 regs, UncoalescedShared <30% AND perf ≥ v111"
echo ""
echo "=== Run: full correctness + bench ==="
runs/fa_v116_swzwordrot
