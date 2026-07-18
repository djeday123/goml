#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/libs/bench_dq
OUT=/data/lib/podman-data/projects/goml/runs/r2_seal/d_launch_occupancy.txt

"$NCU" --launch-count 1 --launch-skip 5 \
       --section LaunchStats \
       --section Occupancy \
       --section SchedulerStats \
       "$BIN" > "$OUT" 2>&1
echo "saved: $OUT"
