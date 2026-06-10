#!/bin/bash
# Step 2a (v3): source-attributed NCu profile of v66.
# Goal: locate the 91% store bank conflict line.
# Strategy: capture full set to .ncu-rep, then re-print with source page.

set -euo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml
REP=$GOML/runs/v66_source.ncu-rep
OUT=$GOML/runs/v66_source_ncu.txt

export HOME=/tmp

cd "$GOML"

echo "=== Phase 1: capture .ncu-rep with --set full ==="
"$NCU" \
    -o runs/v66_source \
    --force-overwrite \
    --target-processes all \
    --kernel-name regex:fa66_kernel \
    --launch-skip 200 \
    --launch-count 1 \
    --set full \
    runs/fa_v66_fp8 2>&1 | tail -10

echo ""
echo "=== Phase 2: print source page with SASS attribution ==="
"$NCU" \
    --import "$REP" \
    --page source \
    --print-source sass 2>&1 | tee "$OUT"
