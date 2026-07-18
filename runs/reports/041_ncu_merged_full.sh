#!/bin/bash
# 041 I.1: полный NCu-профиль merged post-040
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/libs/r2c_merged_wall
GATE=/data/lib/podman-data/projects/goml/runs/reports/037r_gate.sh
OUT=/data/lib/podman-data/projects/goml/runs/reports/041_ncu_merged_full_data.txt
> "$OUT"
"$GATE" 2>&1 | tee -a "$OUT"

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

echo "" | tee -a "$OUT"
echo "=== 041 I.1a: Полная stall-таблица (Σ ~100%) [режим: NCu-mode, wall+3-6% typical] ===" | tee -a "$OUT"
"$NCU" --kernel-name kernel_merged_v1 --launch-count 1 --metrics "$STALLS" "$BIN" 2>&1 | tail -30 | tee -a "$OUT"

LSU="\
smsp__inst_executed_shared_loads.sum,\
smsp__inst_executed_shared_stores.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"

echo "" | tee -a "$OUT"
echo "=== 041 I.1b: LSU/conflicts (для распутывания 4-way ярлыка) ===" | tee -a "$OUT"
"$NCU" --kernel-name kernel_merged_v1 --launch-count 1 --metrics "$LSU" "$BIN" 2>&1 | tail -20 | tee -a "$OUT"

SOL="\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct,\
dram__bytes.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__cycles_active.avg,\
gpc__cycles_elapsed.avg.per_second"

echo "" | tee -a "$OUT"
echo "=== 041 I.1c: SOL L1/L2/DRAM + occupancy [режим: NCu-mode] ===" | tee -a "$OUT"
"$NCU" --kernel-name kernel_merged_v1 --launch-count 1 --metrics "$SOL" "$BIN" 2>&1 | tail -18 | tee -a "$OUT"
