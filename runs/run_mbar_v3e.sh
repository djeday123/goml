#!/bin/bash
CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"
echo "=== Build v3e (only bar.sync 1, 96 added vs v2) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v runs/mbar_repro_v3e.cu -o runs/mbar_repro_v3e -lcudart 2>&1 | grep -E "register|barriers|stack|error"
echo "=== Run v3e ==="
runs/mbar_repro_v3e 2>&1 | tee /data/lib/podman-data/projects/goml/runs/mbar_v3e_out.txt
