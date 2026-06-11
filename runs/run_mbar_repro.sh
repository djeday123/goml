#!/bin/bash
CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"

echo "=== Build mbar_repro ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v runs/mbar_repro.cu -o runs/mbar_repro -lcudart 2>&1 \
    | grep -E "register|barriers|error|stack"

echo ""
echo "=== Run repro matrix ==="
runs/mbar_repro
