#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=$1
OUT=$2

METRICS="\
l1tex__throughput.avg.pct_of_peak_sustained_active,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__warps_active.avg.per_cycle_active,\
sm__ctas_active.avg.pct_of_peak_sustained_active,\
sm__ctas_active.avg.per_cycle_active"

"$NCU" --csv --launch-count 1 --launch-skip 0 --metrics "$METRICS" "$BIN" > "$OUT" 2>&1
echo "saved: $OUT"
