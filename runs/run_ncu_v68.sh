#!/bin/bash
# Verify v68 conflict-fix via NCu re-profile.
# Two key questions:
#   1. Did write-conflict drop from 91% → near 0?
#   2. Did read-conflict stay low or grow?

set -euo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml
REP=$GOML/runs/v68_ncu.ncu-rep
OUT=$GOML/runs/v68_ncu.txt

export HOME=/tmp

cd "$GOML"

echo "=== NCu full profile of fa68_kernel (same config as v66 profile) ==="

"$NCU" \
    -o runs/v68_ncu \
    --force-overwrite \
    --target-processes all \
    --kernel-name regex:fa68_kernel \
    --launch-skip 200 \
    --launch-count 1 \
    --set full \
    runs/fa_v68_fp8 \
    2>&1 | tee "$OUT"

echo ""
echo "=== Key sections to compare with v66: ==="
echo "    1. Memory Workload Analysis Tables: shared store/load bank conflict % "
echo "    2. Scheduler Statistics: 'No Eligible' %"
echo "    3. Source Counters: uncoalesced shared accesses %"
echo "    4. Occupancy: still 8.33% (expected, didn't change SMEM threshold)"
