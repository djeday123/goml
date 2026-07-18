#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
METRICS="smsp__inst_issued.sum,\
smsp__inst_issued.avg.per_cycle_active,\
smsp__inst_issued.avg.per_cycle_active.pct_of_peak_sustained_active,\
gpc__cycles_elapsed.max,\
sm__cycles_active.avg,\
smsp__cycles_active.avg,\
smsp__average_warps_issue_stalled_barrier_per_issue_active,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active,\
smsp__average_warps_issue_stalled_wait_per_issue_active,\
sm__warps_active.avg.per_cycle_active,\
smsp__warps_eligible.avg.per_cycle_active,\
smsp__issue_active.avg.per_cycle_active"

for TAG in A_sealed B_pack_pi C_d5lite; do
  echo "===== $TAG ====="
  "$NCU" --kernel-name kernel_dq_new --launch-count 1 \
      --metrics "$METRICS" \
      /data/lib/podman-data/projects/goml/libs/r1c_dq_$TAG 2>&1 | tail -22
done
