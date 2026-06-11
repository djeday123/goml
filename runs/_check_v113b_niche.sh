#!/bin/bash
# Same-thermal A/B v96 vs v96b vs v113 vs v113b on large-bh mid grids (former v113 niche)
GOML=/data/lib/podman-data/projects/goml

for bin in fa_v96_ksbatched fa_v96b_localfix fa_v113_producer_arrive fa_v113b_localfix; do
    echo "================================================================"
    echo "=== $bin ==="
    echo "================================================================"
    "$GOML/runs/$bin" 2>&1 | grep -E "bh=128 *sl=(2048|4096) wnd=0|bh=64 *sl=(2048|4096) wnd=0"
done
