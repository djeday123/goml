#!/bin/bash
CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"
"$CUDA/bin/nvcc" -O3 -std=c++17 \
    libs/fa_bwd_cpu_reference.cu \
    -o runs/fa_bwd_cpu_reference 2>&1 | grep -E "error|warning" | head -10
echo "---"
runs/fa_bwd_cpu_reference
