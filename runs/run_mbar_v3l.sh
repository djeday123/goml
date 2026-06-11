#!/bin/bash
CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"
echo "=== Build v3l (M7 fix: expected_phase = {0,0}) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v runs/mbar_repro_v3l.cu -o runs/mbar_repro_v3l -lcudart 2>&1 | grep -E "register|barriers|stack|error|warning|fatal"
echo ""
echo "=== Run v3l (Phase1 + Phase2 + Phase3) ==="
runs/mbar_repro_v3l
echo ""
echo "=== rc=$? ==="
