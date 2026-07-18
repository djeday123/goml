#!/bin/bash
CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v \
    runs/probe_e1c_ldmatrix_perf.cu -o runs/probe_e1c_perf -lcudart 2>&1 \
    | grep -E "register|spill|stack|warning|error" | head -10
