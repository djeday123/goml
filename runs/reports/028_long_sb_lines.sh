#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/libs/r1c_dq_wall
OUT=/tmp/dq_source_lineinfo.ncu-rep
rm -f "$OUT"
"$NCU" --kernel-name kernel_dq_new --launch-count 1 \
    --set full --import-source yes \
    --export "$OUT" "$BIN" > /dev/null 2>&1

# Print occupancy + wave info + scheduler
"$NCU" --import "$OUT" --page details 2>&1 | grep -iE "Achieved Occupancy|Warps Per Block|CTAs Per SM|Achieved Active|Waves Per SM|Estimated" | head -20

echo ""
echo "=== SASS source page (top-instructions by long_scoreboard) ==="
"$NCU" --import "$OUT" --page source 2>&1 | head -200
