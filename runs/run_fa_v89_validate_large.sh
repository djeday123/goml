#!/bin/bash
# v89 validation on large production grids (bh=64, bh=128).
# Confirms 413T champion + LB=3 +27% wave-reduction hold at production scale.
# Tests HBM bandwidth ceiling vs MMA pipe — if perf scales linearly with batch×heads,
# kernel is compute-bound. If perf saturates, HBM is becoming the limit.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v89 validation variant (large grids) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v89_validate_large_grids.cu \
    -o runs/fa_v89_validate_large -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -40

echo ""
echo "=== Run: full correctness + extended bench (bh=64, bh=128 added) ==="
runs/fa_v89_validate_large
