#!/bin/bash
# 040 NCu post-правка: SASS check + wavefronts + DRAM + stalls
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/libs/r2c_merged_wall
OUT=/data/lib/podman-data/projects/goml/runs/reports/040_ncu_post_data.txt
> "$OUT"

echo "=== 040 NCu post — предсказания vs факт ===" | tee -a "$OUT"

# 1. Stalls (predict: mio DOWN, short_sb ?)
echo "-- Stalls --" | tee -a "$OUT"
"$NCU" --kernel-name kernel_merged_v1 --launch-count 1 \
    --metrics smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct \
    "$BIN" 2>&1 | tail -10 | tee -a "$OUT"

# 2. LSU / conflicts / wavefronts
echo "-- LSU / conflicts / wavefronts (predict LD_conflict events=0 additional, wavefronts #7: 256→128) --" | tee -a "$OUT"
"$NCU" --kernel-name kernel_merged_v1 --launch-count 1 \
    --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum \
    "$BIN" 2>&1 | tail -10 | tee -a "$OUT"

# 3. DRAM / occupancy (predict DRAM 9.79 GB unchanged)
echo "-- DRAM / occupancy (predict 9.79 GB, 2 blk × 4 warps = 16.58%) --" | tee -a "$OUT"
"$NCU" --kernel-name kernel_merged_v1 --launch-count 1 \
    --metrics dram__bytes.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,lts__t_sector_hit_rate.pct \
    "$BIN" 2>&1 | tail -10 | tee -a "$OUT"
