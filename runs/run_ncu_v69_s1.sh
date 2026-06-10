#!/bin/bash
# NCu profile of v69_s1 to determine root cause of -23% perf vs v68.
# Hypothesis: reg pressure + spills, NOT V read bank conflicts.
set -euo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml
OUT=$GOML/runs/v69_s1_ncu.txt

export HOME=/tmp

cd "$GOML"

echo "=== Quick NCu (memory + scheduler only) of fa69_s1_kernel ==="
"$NCU" \
    --target-processes all \
    --kernel-name regex:fa69_s1_kernel \
    --launch-skip 200 \
    --launch-count 1 \
    --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Tables \
    --section SchedulerStats \
    --section WarpStateStats \
    --section Occupancy \
    runs/fa_v69_s1_fp8 \
    2>&1 | tee "$OUT"
