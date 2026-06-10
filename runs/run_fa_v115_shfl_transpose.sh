#!/bin/bash
# v115 = v96 + SHFL-based cooperative transpose_v.
# 4-thread groups transpose one 4x4 tile each. Each thread writes 1 row of smV_T.
# Goal: eliminate 4-way write bank conflict (4 threads write 4 adjacent rows
# instead of 1 thread writing 4 rows with stride 4 → bank stride 17, gcd 1).
# Risk: SHFL byte permutation logic may have bug; reads may have new conflicts.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v115 SHFL transpose ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v115_shfl_transpose_hd128_fp8_forward.cu \
    -o runs/fa_v115_shfl_transpose -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -30

echo ""
echo "=== References ==="
echo "  v96  : 242 regs, 568 mean PEAK, UncoalescedSharedAccess 40.0%"
echo "  v114 : 242 regs, 540 mean (-5%), failed simple mapping swap"
echo "  v115 target: same regs, UncoalescedSharedAccess <30% AND perf ≥ v96"
echo ""
echo "=== Run: full correctness + bench ==="
runs/fa_v115_shfl_transpose
