#!/bin/bash
CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"
echo "=== Build v3j (canonical mbarrier + __nanosleep(32) backoff) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v runs/mbar_repro_v3j.cu -o runs/mbar_repro_v3j -lcudart 2>&1 | grep -E "register|barriers|stack|error|warning|fatal"
echo ""
echo "=== Verify SASS: NANOSLEEP near PHASECHK ==="
"$CUDA/bin/cuobjdump" -sass runs/mbar_repro_v3j | grep -E 'SYNCS\.|NANOSLEEP|YIELD' | head -20
echo ""
echo "=== SYNCS + NANOSLEEP opcode counts ==="
"$CUDA/bin/cuobjdump" -sass runs/mbar_repro_v3j | grep -oE '(SYNCS|NANOSLEEP)\.[A-Z._0-9]*' | sort | uniq -c
