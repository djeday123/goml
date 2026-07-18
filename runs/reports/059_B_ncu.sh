#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BASE=/data/lib/podman-data/projects/goml/runs/archive/059_pre/r2c_merged_wall_base
CAND=/data/lib/podman-data/projects/goml/runs/archive/056_pre/r2c_merged_wall_cand_B
LOG=/data/lib/podman-data/projects/goml/runs/reports/059_B_ncu_data.txt
> "$LOG"
STALLS="smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct"
LSU="l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,dram__bytes.sum,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread"
for label in BASE CAND; do
    if [ "$label" = "BASE" ]; then BIN=$BASE; else BIN=$CAND; fi
    echo "===== $label =====" | tee -a "$LOG"
    "$NCU" --kernel-name kernel_merged_v1 --launch-count 1 --metrics "$STALLS" "$BIN" 2>&1 | tail -14 | tee -a "$LOG"
    "$NCU" --kernel-name kernel_merged_v1 --launch-count 1 --metrics "$LSU" "$BIN" 2>&1 | tail -12 | tee -a "$LOG"
done
