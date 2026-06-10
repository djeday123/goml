#!/bin/bash
# Build and run mma_probe_sm120.cu — verifies QMMA m16n8k32 and HMMA m16n8k16
# fragment layouts on sm_120a empirically. Required before any new FA kernel.
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build mma_probe_sm120 ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    libs/mma_probe_sm120.cu -o runs/mma_probe_sm120 -lcudart

echo ""
echo "=== Run ==="
runs/mma_probe_sm120
