#!/bin/bash
CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    runs/smid_probe.cu -o runs/smid_probe -lcudart 2>&1 | grep -E "error|warning"
runs/smid_probe
