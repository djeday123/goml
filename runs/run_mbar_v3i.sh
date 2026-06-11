#!/bin/bash
CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"
echo "=== Build v3i (try_wait.parity.acquire warp-park) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v runs/mbar_repro_v3i.cu -o runs/mbar_repro_v3i -lcudart 2>&1 | grep -E "register|barriers|stack|error|warning|fatal"
echo ""
echo "=== Verify SASS: SYNCS.* opcodes ==="
"$CUDA/bin/cuobjdump" -sass runs/mbar_repro_v3i | grep -E 'SYNCS\.|MEMBAR|FENCE' | head -10
echo ""
echo "=== SYNCS opcode counts ==="
"$CUDA/bin/cuobjdump" -sass runs/mbar_repro_v3i | grep -oE 'SYNCS\.[A-Z._0-9]+' | sort | uniq -c
