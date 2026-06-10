#!/bin/bash
# Build & run FA forward FP8 v67 (TMA conveyor). Compares with v66 baseline.
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v67 ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    libs/flash_attention_v67_fp8_forward.cu \
    -o runs/fa_v67_fp8 -lcudart -lcuda 2>&1 | tail -20

echo ""
echo "=== v66 baseline (regression) ==="
runs/fa_v66_fp8 2>&1 | tail -20

echo ""
echo "=== v67 TMA ==="
runs/fa_v67_fp8
