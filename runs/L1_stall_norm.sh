#!/bin/bash
# L1: honest stall breakdown from raw PC samples (dV) with per-class %.
# Pulls all pcsamp counters from NCu on dV.
env HOME=/data/lib/podman-data/projects/goml/ncu_home /usr/local/cuda-13.1/bin/ncu \
  --csv --print-metric-instances none \
  --launch-skip 6 --launch-count 1 --kernel-name kernel_dv_mma_p1 \
  --metrics smsp__pcsamp_warps_issue_stalled_barrier,smsp__pcsamp_warps_issue_stalled_short_scoreboard,smsp__pcsamp_warps_issue_stalled_long_scoreboard,smsp__pcsamp_warps_issue_stalled_mio_throttle,smsp__pcsamp_warps_issue_stalled_wait,smsp__pcsamp_warps_issue_stalled_selected,smsp__pcsamp_warps_issue_stalled_not_selected,smsp__pcsamp_warps_issue_stalled_math_pipe_throttle,smsp__pcsamp_warps_issue_stalled_lg_throttle,smsp__pcsamp_warps_issue_stalled_drain,smsp__pcsamp_warps_issue_stalled_dispatch_stall,smsp__pcsamp_warps_issue_stalled_no_instruction,smsp__pcsamp_warps_issue_stalled_branch_resolving,smsp__pcsamp_warps_issue_stalled_membar,smsp__pcsamp_warps_issue_stalled_imc_miss,smsp__pcsamp_warps_issue_stalled_tex_throttle,smsp__pcsamp_warps_issue_stalled_sleeping \
  --log-file /data/lib/podman-data/projects/goml/runs/dv_L1_stalls.txt \
  /data/lib/podman-data/projects/goml/libs/bench_dv 128 8192 0 0 5 20
