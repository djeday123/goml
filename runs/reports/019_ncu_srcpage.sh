#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=$1
OUT=/tmp/dk_source2.ncu-rep
rm -f "$OUT"
"$NCU" --kernel-name kernel_dk_new --launch-count 1 \
    --set full --import-source yes \
    --export "$OUT" "$BIN" > /dev/null 2>&1

echo "=== stall_short_scoreboard per-inst top-50 (via ncu-ui-style report) ==="
"$NCU" --import "$OUT" \
       --page raw --csv | head -1 | tr ',' '\n' | grep -in "short_scoreboard\|source_file\|source_line" | head -10
