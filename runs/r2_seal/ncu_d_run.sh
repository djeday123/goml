#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/libs/bench_dq
OUT=/data/lib/podman-data/projects/goml/runs/r2_seal/d_ncu.csv

METRICS="\
l1tex__throughput.avg.pct_of_peak_sustained_active,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__warps_active.avg.per_cycle_active,\
sm__ctas_active.avg.pct_of_peak_sustained_active,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
l1tex__data_bank_conflicts_pipe_lsu.sum,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio"

"$NCU" --csv --launch-count 1 --launch-skip 5 --metrics "$METRICS" "$BIN" > "$OUT" 2>&1
echo "saved: $OUT"
