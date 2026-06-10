#!/bin/bash
# PC sampling for v111 PEAK — per-source-line stall reasons.

set -uo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

OUT="$GOML/runs/ncu_v111_pcsamp.csv"

echo "=== PC-sampling NCu for v111 PEAK cfg=9 ==="
"$NCU" \
    --target-processes all \
    --launch-skip 1 --launch-count 1 \
    --section SourceCounters \
    --section InstructionStats \
    --section WarpStateStats \
    --import-source on \
    --csv \
    "$GOML/runs/fa_v111_producer_skip" --ncu 9 > "$OUT" 2>&1

echo "  Raw output: $OUT"
echo ""

# First check what's in it
echo "--- First 30 lines of output (diagnostic) ---"
head -30 "$OUT"
echo ""
echo "--- Header rows + a few data rows ---"
grep -E '^"ID"|^Section|^Metric' "$OUT" | head -10
echo ""

# Count metrics
echo "--- Metrics present ---"
grep -E '"smsp__|"sm__' "$OUT" | awk -F',' '{print $4}' | sort -u | head -30
