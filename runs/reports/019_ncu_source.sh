#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=$1
"$NCU" --kernel-name kernel_dk_new --launch-count 1 \
    --section SourceCounters --import-source yes \
    --print-summary per-kernel --print-details all \
    "$BIN" 2>&1 | grep -E "smsp_average_warp_issue_stalled|smsp__pcsamp|Source|fa_bwd_dk_new\.cu:.*(short|long|barrier|scoreboard|throttle|wait)" | head -80
