#!/bin/bash
# NCu v69 on bh=16 sl=4096 (512 blocks = production-shape) to verify the
# +51% gain mechanism. KEY: does Achieved Occupancy hit 16.67%, confirming
# 2 blocks/SM as the mechanism, or stay at 8.33% (different cause)?
set -euo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml
OUT=$GOML/runs/v69_bh16_ncu.txt

export HOME=/tmp

cd "$GOML"

# Launch sequence (from main): 5 correctness + 8 bench × 55 launches each.
# bench config order: {4,1024}, {4,2048}, {8,2048}, {4,4096},
#                     {8,4096}, {16,2048}, {16,4096}, {32,2048}.
# bh=16 sl=4096 is config index 6.
# Launches before it: 5 correctness + 6 × 55 = 335.
# Skip first warmup of config 6 (5 launches), profile next measured.
SKIP=340

echo "=== NCu v69 @ bh=16 sl=4096 (launch-skip=$SKIP, the +51% config) ==="
"$NCU" \
    --target-processes all \
    --kernel-name regex:fa69_kernel \
    --launch-skip "$SKIP" \
    --launch-count 1 \
    --section Occupancy \
    --section SchedulerStats \
    --section WarpStateStats \
    --section LaunchStats \
    runs/fa_v69_fp8 \
    2>&1 | tee "$OUT"
