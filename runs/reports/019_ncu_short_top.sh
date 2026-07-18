#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=$1
"$NCU" --kernel-name kernel_dk_new --launch-count 1 \
       --metrics smsp__pcsamp_warps_issue_stalled_short_scoreboard.sum,derived__memory_l1_conflicts_shared_nway \
       --set full --import-source yes \
       "$BIN" 2>&1 | grep -iE "short_scoreboard|dk_new\.cu" | head -20

echo "===== source-line top-short_sb via CSV parse ====="
"$NCU" --import /tmp/dk_source.ncu-rep --page source --print-source function 2>&1 | head -5
