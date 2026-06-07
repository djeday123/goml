#!/bin/bash
# Re-extract key sections from v68_ncu.ncu-rep.
set -euo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
REP=/data/lib/podman-data/projects/goml/runs/v68_ncu.ncu-rep
OUT=/data/lib/podman-data/projects/goml/runs/v68_ncu_sections.txt

export HOME=/tmp

echo "=== v68 NCu: key sections ==="
"$NCU" --import "$REP" \
       --page details \
       --section MemoryWorkloadAnalysis_Tables \
       --section SchedulerStats \
       --section WarpStateStats \
       --section Occupancy \
       --section SourceCounters \
       2>&1 | tee "$OUT"
echo ""
echo "Saved to $OUT"
