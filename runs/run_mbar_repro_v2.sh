#!/bin/bash
CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"

echo "=== Build mbar_repro_v2 (+cp.async) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v runs/mbar_repro_v2.cu -o runs/mbar_repro_v2 -lcudart 2>&1 \
    | grep -E "register|barriers|stack|error"

echo ""
echo "=== Run ==="
runs/mbar_repro_v2
