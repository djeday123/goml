#!/bin/bash
# 042 хвост 041-0.d: NCu dq_new post-041 (sealed d7a11a3d)
# Сверка с 029 killers: mio +5.97 / barrier +2.36 pp vs 024 baseline
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/libs/bench_r2c_e2e
OUT=/data/lib/podman-data/projects/goml/runs/reports/042_ncu_dq_post041_data.txt
> "$OUT"

echo "=== 042 хвост 041-0.d: NCu dq_new post-041 (dq_new=d7a11a3d 69r) [NCu-mode] ===" | tee -a "$OUT"

STALLS="smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct"

"$NCU" --kernel-name kernel_dq_new --launch-count 1 --metrics "$STALLS" "$BIN" 2>&1 | tail -12 | tee -a "$OUT"
