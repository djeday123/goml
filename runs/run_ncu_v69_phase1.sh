#!/bin/bash
# Phase 1 NCu: classify warp stalls in v69 at the peak config (bh=16 sl=4096).
# Goal: distinguish softmax-bound (stall_math + stall_short_scoreboard)
# from MMA-issue-bound (no_eligible + stall_wait) from memory-bound (stall_membar / stall_long_scoreboard).
#
# Launch index math:
#   8 correctness launches +
#   bench configs 0..5 (4x1024, 4x2048, 8x2048, 4x4096, 8x4096, 16x2048) × 55 (5 warmup+50 timed) = 330 launches
#   = 338 total prior. bh=16 sl=4096 warmups = launches 338..342. First timed = 343.
# Pick a mid-timing launch (350) so cache state is steady.
set -euo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/runs/fa_v69_fp8
OUT=/data/lib/podman-data/projects/goml/runs/ncu_v69_phase1.log

"$NCU" \
    --target-processes all \
    --launch-skip 350 --launch-count 1 \
    --section WarpStateStats \
    --section SchedulerStats \
    --section ComputeWorkloadAnalysis \
    --section MemoryWorkloadAnalysis \
    --section SpeedOfLight \
    --section Occupancy \
    --csv --page details \
    "$BIN" > "$OUT" 2>&1 || true

echo "--- last 80 lines of $OUT ---"
tail -80 "$OUT"
