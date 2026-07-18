#!/bin/bash
# 038 NCu post-E: сверка LD/ST conflicts, DRAM с 037-r fresh
GATE=/data/lib/podman-data/projects/goml/runs/reports/037r_gate.sh
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/libs/r2c_merged_wall
OUT=/data/lib/podman-data/projects/goml/runs/reports/038_ncu_post_E_data.txt
> "$OUT"
"$GATE" >> "$OUT" 2>&1 || { echo "gate failed"; exit 1; }

echo "=== NCu post-E: conflicts/DRAM/occupancy сверка ===" | tee -a "$OUT"

echo "-- LSU / conflicts --" | tee -a "$OUT"
$NCU --kernel-name kernel_merged_v1 --launch-count 1 \
     --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum \
     "$BIN" 2>&1 | tail -12 | tee -a "$OUT"

echo "-- DRAM / occupancy --" | tee -a "$OUT"
$NCU --kernel-name kernel_merged_v1 --launch-count 1 \
     --metrics dram__bytes.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,lts__t_sector_hit_rate.pct \
     "$BIN" 2>&1 | tail -15 | tee -a "$OUT"

echo "-- Stalls (barrier/mio/short_sb для дрейф-контроля) --" | tee -a "$OUT"
$NCU --kernel-name kernel_merged_v1 --launch-count 1 \
     --metrics smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct \
     "$BIN" 2>&1 | tail -10 | tee -a "$OUT"
