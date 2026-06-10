#!/bin/bash
# Step 1: Source-attributed UncoalescedSharedAccess on v111.
# Compare hot lines with v96 to verify v111 (96-thread consumer transpose)
# did NOT introduce new conflicts (uncoalesced % almost equal: 40.0 vs 40.2,
# but per-line attribution will say where exactly).

set -uo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml
BIN="$GOML/runs/fa_v111_producer_skip"

cd "$GOML"

if [ ! -x "$BIN" ]; then echo "ERROR: $BIN not built" >&2; exit 1; fi

REP="$GOML/runs/ncu_v111_uncoal_source.ncu-rep"

echo "=== Collect SourceCounters on v111 cfg=9 ==="
"$NCU" \
    --target-processes all \
    --launch-skip 1 --launch-count 1 \
    --section SourceCounters \
    --import-source on \
    --export "$REP" \
    "$BIN" --ncu 9 2>&1 | tail -10

echo ""
echo "=== Source page dump ==="
SRC_TXT="$GOML/runs/ncu_v111_uncoal_source.txt"
"$NCU" --import "$REP" --page source > "$SRC_TXT" 2>&1 || true
echo "Wrote $SRC_TXT ($(wc -l < $SRC_TXT) lines)"
