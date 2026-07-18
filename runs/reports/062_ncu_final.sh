#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/libs/bench_r2c_e2e
LOG=/data/lib/podman-data/projects/goml/runs/reports/062_ncu_final.txt
> "$LOG"
STALLS="smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct"
LSU="l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,dram__bytes.sum,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread"
for KERNEL in kernel_merged_v1 kernel_dk_new kernel_dq_new kernel_d_precompute; do
    echo "===== $KERNEL =====" | tee -a "$LOG"
    "$NCU" --kernel-name $KERNEL --launch-count 1 --metrics "$STALLS" "$BIN" 2>&1 | tail -13 | tee -a "$LOG"
    "$NCU" --kernel-name $KERNEL --launch-count 1 --metrics "$LSU" "$BIN" 2>&1 | tail -11 | tee -a "$LOG"
    echo "" | tee -a "$LOG"
done
