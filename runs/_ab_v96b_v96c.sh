#!/bin/bash
# Same-thermal A/B v96b vs v96c
GOML=/data/lib/podman-data/projects/goml
for bin in fa_v96b_localfix fa_v96c_ffma fa_v96b_localfix fa_v96c_ffma; do
    echo "================================================================"
    echo "=== $bin ==="
    "$GOML/runs/$bin" 2>&1 | grep -E "^  bh=" | head -23
done
