#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/libs/r1a_wall
OUT=/data/lib/podman-data/projects/goml/runs/reports/001_r1a_ncu.txt

"$NCU" --launch-count 1 --launch-skip 5 \
       --section LaunchStats \
       --section Occupancy \
       --section MemoryWorkloadAnalysis \
       --section SchedulerStats \
       "$BIN" > "$OUT" 2>&1
echo "saved: $OUT"
