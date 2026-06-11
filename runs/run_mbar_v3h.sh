#!/bin/bash
CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"
echo "=== Build v3h (canonical sm_90+ mbarrier forms) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v runs/mbar_repro_v3h.cu -o runs/mbar_repro_v3h -lcudart 2>&1 | grep -E "register|barriers|stack|error|warning|fatal"
echo ""
echo "=== Verify SASS: SYNCS.* opcodes ==="
"$CUDA/bin/cuobjdump" -sass runs/mbar_repro_v3h | grep -E 'SYNCS\.|MEMBAR|FENCE' | head -20
echo ""
echo "=== SYNCS opcode counts ==="
"$CUDA/bin/cuobjdump" -sass runs/mbar_repro_v3h | grep -oE 'SYNCS\.[A-Z._0-9]+' | sort | uniq -c
echo ""
echo "=== BAR.SYNC count ==="
"$CUDA/bin/cuobjdump" -sass runs/mbar_repro_v3h | grep -cE 'BAR\.SYNC'
