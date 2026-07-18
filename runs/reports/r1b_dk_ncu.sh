#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/libs/r1b_dk_wall
OUT=/data/lib/podman-data/projects/goml/runs/reports/004_r1b_fix_dk_ncu.txt

"$NCU" --launch-count 1 --launch-skip 5 \
       --section LaunchStats \
       --section Occupancy \
       --section MemoryWorkloadAnalysis \
       --section SchedulerStats \
       --kernel-name kernel_dk_new \
       "$BIN" > "$OUT" 2>&1
echo "saved: $OUT"
