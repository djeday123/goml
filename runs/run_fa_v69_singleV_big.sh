#!/bin/bash
# v69_singleV with EXTENDED bench (incl. configs with 256+ blocks where 2 blocks/SM helps).
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Rebuild v68 with extended bench ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    libs/flash_attention_v68_fp8_forward.cu \
    -o runs/fa_v68_fp8 -lcudart 2>&1 | tail -3

echo ""
echo "=== Rebuild v69_singleV with extended bench ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    libs/flash_attention_v69_singleV_fp8_forward.cu \
    -o runs/fa_v69_singleV_fp8 -lcudart 2>&1 | tail -3

echo ""
echo "=== v68 (1 block/SM) — extended bench ==="
runs/fa_v68_fp8 2>&1 | tail -20

echo ""
echo "=== v69_singleV (2 blocks/SM theoretical) — extended bench ==="
runs/fa_v69_singleV_fp8 2>&1 | tail -20
