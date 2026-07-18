#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=$1
"$NCU" --kernel-name kernel_dk_new --launch-count 1 \
       --metrics smsp__pcsamp_warps_issue_stalled_short_scoreboard.sum \
       --set full --import-source yes \
       --print-details all --print-source per-inst \
       "$BIN" 2>&1 | head -20

echo "===== try filter ====="
"$NCU" --kernel-name kernel_dk_new --launch-count 1 \
       --metrics group:memory__shared_table --replay-mode kernel \
       --print-source no \
       "$BIN" 2>&1 | tail -10
