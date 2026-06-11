#!/bin/bash
CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"
echo "=== Build v3f (watchdog) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v runs/mbar_repro_v3f.cu -o runs/mbar_repro_v3f -lcudart -lpthread 2>&1 | grep -E "register|barriers|stack|error"
echo ""
echo "=== Run v3f (watchdog logs every 0.5s) ==="
timeout 15 runs/mbar_repro_v3f 2>&1 | tee /data/lib/podman-data/projects/goml/runs/mbar_v3f_out.txt
echo "---DONE---"
