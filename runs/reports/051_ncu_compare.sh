#!/bin/bash
# 051 §2 NCu compare S2v3 vs baseline (production 033)
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
S2v3=/data/lib/podman-data/projects/goml/libs/bench_r2c_e2e_s2v3
BASE=/data/lib/podman-data/projects/goml/runs/archive/050_pre/bench_r2c_e2e
OUT=/data/lib/podman-data/projects/goml/runs/reports/051_ncu_compare_data.txt
> "$OUT"

STALLS="smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct"
LSU="l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,dram__bytes.sum,sm__warps_active.avg.pct_of_peak_sustained_active"

for label in BASELINE S2v3; do
    if [ "$label" = "BASELINE" ]; then BIN=$BASE; else BIN=$S2v3; fi
    echo "===== $label ($BIN) =====" | tee -a "$OUT"
    echo "-- Stalls kernel_dk_new --" | tee -a "$OUT"
    "$NCU" --kernel-name kernel_dk_new --launch-count 1 --metrics "$STALLS" "$BIN" 2>&1 | tail -12 | tee -a "$OUT"
    echo "-- LSU/DRAM/Occupancy --" | tee -a "$OUT"
    "$NCU" --kernel-name kernel_dk_new --launch-count 1 --metrics "$LSU" "$BIN" 2>&1 | tail -10 | tee -a "$OUT"
    echo "" | tee -a "$OUT"
done
