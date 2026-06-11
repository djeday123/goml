#!/bin/bash
# Check LDL/STL count across all production-candidate kernels
GOML=/data/lib/podman-data/projects/goml
CUDA=/usr/local/cuda-13.1

for b in v96_ksbatched v113_producer_arrive v111_producer_skip v118_localfix; do
    bin="$GOML/runs/fa_${b}"
    if [ -x "$bin" ]; then
        echo "================================================================"
        echo "== fa_${b} =="
        echo "================================================================"
        # SASS local memory count
        ldl=$("$CUDA/bin/cuobjdump" -sass "$bin" 2>/dev/null | grep -cE '\bLDL\b')
        stl=$("$CUDA/bin/cuobjdump" -sass "$bin" 2>/dev/null | grep -cE '\bSTL\b')
        echo "  LDL count: $ldl"
        echo "  STL count: $stl"
        echo "  Total local: $((ldl + stl))"
    fi
done
