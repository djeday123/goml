#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/libs/bench_dv
OUT=/data/lib/podman-data/projects/goml/runs/r2_seal/dv_baseline_launch_occ.txt

"$NCU" --launch-count 1 --launch-skip 5 \
       --section LaunchStats \
       --section Occupancy \
       --section MemoryWorkloadAnalysis \
       --section SchedulerStats \
       "$BIN" > "$OUT" 2>&1
echo "saved: $OUT"
