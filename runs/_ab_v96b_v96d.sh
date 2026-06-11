#!/bin/bash
GOML=/data/lib/podman-data/projects/goml
for bin in fa_v96b_localfix fa_v96d_ffma_minimal fa_v96b_localfix fa_v96d_ffma_minimal; do
    echo "================================================================"
    echo "=== $bin ==="
    "$GOML/runs/$bin" 2>&1 | grep -E "^  bh=" | head -23
done
