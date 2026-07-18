#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/libs/r1c_dq_wall
"$NCU" --kernel-name kernel_dq_new --launch-count 1 \
    --metrics \
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__warps_active.avg.per_cycle_active,\
sm__warps_active.sum,\
sm__maximum_warps_per_active_cycle_pct,\
sm__ctas_active.avg.pct_of_peak_sustained_active,\
sm__ctas_active.avg.per_cycle_active,\
smsp__warps_eligible.avg.per_cycle_active,\
smsp__warps_eligible.avg.pct_of_peak_sustained_active,\
smsp__warps_issue_stalled_no_instruction_per_warp_active.pct,\
smsp__issue_active.avg.per_cycle_active,\
smsp__warp_cycles_per_issue_active.avg \
    "$BIN" 2>&1 | tail -25
