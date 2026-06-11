#!/bin/bash
CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"
echo "=== Build v3d (v3 minus Q-prelude sync) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v runs/mbar_repro_v3d.cu -o runs/mbar_repro_v3d -lcudart 2>&1 | grep -E "register|barriers|stack|error"
echo "=== Run v3d ==="
runs/mbar_repro_v3d 2>&1 | tee /data/lib/podman-data/projects/goml/runs/mbar_v3d_out.txt
echo "---DONE---"
