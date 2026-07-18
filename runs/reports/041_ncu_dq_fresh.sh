#!/bin/bash
# 041 II.6: fresh NCu profile dq_new production
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/libs/bench_r2c_e2e
OUT=/data/lib/podman-data/projects/goml/runs/reports/041_ncu_dq_fresh_data.txt
> "$OUT"

STALLS="\
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_wait_per_warp_active.pct,\
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_selected_per_warp_active.pct,\
smsp__warp_issue_stalled_not_selected_per_warp_active.pct,\
smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct"

echo "=== 041 II.6: NCu dq_new production (fresh) [режим: NCu-mode] ===" | tee -a "$OUT"

echo "-- Stalls --" | tee -a "$OUT"
"$NCU" --kernel-name kernel_dq_new --launch-count 1 --metrics "$STALLS" "$BIN" 2>&1 | tail -18 | tee -a "$OUT"

echo "-- LSU/conflicts/L2/DRAM --" | tee -a "$OUT"
"$NCU" --kernel-name kernel_dq_new --launch-count 1 \
    --metrics "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,lts__t_sector_hit_rate.pct,dram__bytes.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active" \
    "$BIN" 2>&1 | tail -15 | tee -a "$OUT"
