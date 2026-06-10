#!/bin/bash
# Isolated ldmatrix.trans correctness probe for FP8.
# Tests if ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 gives correct
# transposed layout when reading K-major smV as packed FP8 pairs.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build ldmatrix.trans FP8 probe ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v \
    libs/ldmatrix_trans_fp8_probe.cu \
    -o runs/ldmatrix_trans_probe -lcudart 2>&1 | grep -E "(register|spill|stack|smem|warning|error)" | head -20

echo ""
echo "=== Run probe ==="
runs/ldmatrix_trans_probe
