#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=$1
OUT=/tmp/dk_source.ncu-rep
rm -f "$OUT"
"$NCU" --kernel-name kernel_dk_new --launch-count 1 \
    --set full --import-source yes \
    --export "$OUT" "$BIN" > /dev/null 2>&1
echo "=== source page (top-loaded lines) ==="
"$NCU" --import "$OUT" --page raw --csv 2>&1 | head -3
echo "=== stall breakdown per SASS block: parse .ncu-rep ==="
"$NCU" --import "$OUT" --page details --print-details all --print-source per-instruction --print-metric-attribution instruction 2>&1 | grep -iE "short_scoreboard|fa_bwd_dk_new\.cu|kernel_dk_new" | head -40
