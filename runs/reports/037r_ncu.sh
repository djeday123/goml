#!/bin/bash
# 037-r NCu пересъёмка (post-cut merged, gate-verified 254r)
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/libs/r2c_merged_wall
OUT=/data/lib/podman-data/projects/goml/runs/reports/037r_ncu_data.txt
> "$OUT"

STALLS="\
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_wait_per_warp_active.pct,\
smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct,\
smsp__warp_issue_stalled_no_instruction_per_warp_active.pct,\
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_drain_per_warp_active.pct,\
smsp__warp_issue_stalled_imc_miss_per_warp_active.pct,\
smsp__warp_issue_stalled_membar_per_warp_active.pct,\
smsp__warp_issue_stalled_misc_per_warp_active.pct,\
smsp__warp_issue_stalled_not_selected_per_warp_active.pct,\
smsp__warp_issue_stalled_selected_per_warp_active.pct"

LSU="\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
smsp__inst_executed_shared_loads.sum,\
smsp__inst_executed_shared_stores.sum"

GLOBAL="\
lts__t_sector_hit_rate.pct,\
l1tex__t_sector_hit_rate.pct,\
dram__bytes.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__cycles_active.avg,\
gpc__cycles_elapsed.avg.per_second"

echo "=== STALL FULL (post-cut merged, fresh bin, gate 254r) ===" | tee -a "$OUT"
"$NCU" --kernel-name kernel_merged_v1 --launch-count 1 \
    --metrics "$STALLS" "$BIN" 2>&1 | tail -30 | tee -a "$OUT"

echo "" | tee -a "$OUT"
echo "=== LSU / CONFLICTS ===" | tee -a "$OUT"
"$NCU" --kernel-name kernel_merged_v1 --launch-count 1 \
    --metrics "$LSU" "$BIN" 2>&1 | tail -15 | tee -a "$OUT"

echo "" | tee -a "$OUT"
echo "=== L2 / DRAM / OCC ===" | tee -a "$OUT"
"$NCU" --kernel-name kernel_merged_v1 --launch-count 1 \
    --metrics "$GLOBAL" "$BIN" 2>&1 | tail -20 | tee -a "$OUT"
