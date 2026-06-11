#!/bin/bash
CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"

echo "=== Build mbar_repro_v3 (+sync mix) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v runs/mbar_repro_v3.cu -o runs/mbar_repro_v3 -lcudart 2>&1 \
    | grep -E "register|barriers|stack|error"

echo ""
echo "=== Run ==="
runs/mbar_repro_v3 2>&1 | tee /data/lib/podman-data/projects/goml/runs/mbar_v3_out.txt
echo "---DONE---"
