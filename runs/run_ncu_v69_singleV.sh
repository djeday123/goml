#!/bin/bash
# v69_singleV NCu — verify actual occupancy and check what changed.
set -euo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml
OUT=$GOML/runs/v69_singleV_ncu.txt

export HOME=/tmp

cd "$GOML"

echo "=== NCu v69_singleV ==="
"$NCU" \
    --target-processes all \
    --kernel-name regex:fa69_singleV_kernel \
    --launch-skip 200 \
    --launch-count 1 \
    --section Occupancy \
    --section SchedulerStats \
    --section WarpStateStats \
    --section MemoryWorkloadAnalysis_Tables \
    runs/fa_v69_singleV_fp8 \
    2>&1 | tee "$OUT"
