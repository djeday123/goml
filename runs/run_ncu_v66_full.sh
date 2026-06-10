#!/bin/bash
# Deeper NCu profile of v66 — full stall breakdown via PC sampling.
# Slower (15-30s per kernel × 10 passes ≈ minutes) but gives the exact
# stall reason distribution we need to pick the next lever precisely.

set -euo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml
OUT=$GOML/runs/v66_ncu_full.txt

export HOME=/tmp

cd "$GOML"

echo "=== NCu full profile of fa66_kernel (this will take a few minutes) ==="

"$NCU" \
    --target-processes all \
    --kernel-name regex:fa66_kernel \
    --launch-skip 200 \
    --launch-count 1 \
    --set full \
    runs/fa_v66_fp8 \
    2>&1 | tee "$OUT"

echo ""
echo "=== Report saved to $OUT ==="
echo ""
echo "=== Key sections to look at: ==="
echo "    - 'Warp State Statistics' table (full stall reason breakdown)"
echo "    - 'Stall Long Scoreboard' / 'Stall Short Scoreboard' / 'Stall Wait' lines"
