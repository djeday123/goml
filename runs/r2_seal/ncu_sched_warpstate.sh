#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=$1
OUT=$2

"$NCU" --launch-count 1 --launch-skip 5 \
       --section SchedulerStats \
       --section WarpStateStats \
       --section Occupancy \
       --section LaunchStats \
       --metrics smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__inst_issued.avg.per_cycle_active,smsp__warps_eligible.avg.per_cycle_active,smsp__cycles_active.avg,smsp__warps_active.avg.per_cycle_active,smsp__issue_inst0.avg.per_cycle_active,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio \
       "$BIN" > "$OUT" 2>&1
echo "saved: $OUT"
