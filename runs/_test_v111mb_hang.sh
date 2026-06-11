#!/bin/bash
# Verify the original v111-mbarrier still hangs at sl=300 wnd=96.
GOML=/data/lib/podman-data/projects/goml
BIN="$GOML/runs/fa_v111_warpspec_mbarrier"
echo "=== Running original v111-mbarrier with 30s timeout on sl=300 wnd=96 ==="
timeout 30 "$BIN" 2>&1 | tail -25
echo ""
echo "Exit code: $?"
echo "(124 = timeout aka hang, 0 = completed)"
