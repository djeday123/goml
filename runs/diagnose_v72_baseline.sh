#!/bin/bash
# Diagnose: is v72 (Br=64, 4 blocks/SM) baseline already competitive with v79 on
# the wave-tail config (bh=16 sl=2048) we want to attack? If v72 raw < v79 already,
# Path B (port v78/v79 patches into v72) is unlikely to win.
#
# Also rebuild v72 to confirm ptxas reg count (was 127, need ≤128 for 4 blocks/SM).
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Rebuild v72 with -Xptxas=-v ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v \
    libs/flash_attention_v72_fp8_forward.cu \
    -o runs/fa_v72_fp8 -lcudart 2>&1 | tail -6

echo ""
echo "=== v72 bench (current state) ==="
runs/fa_v72_fp8 2>&1 | grep -E "perf=|GPU:" | head -15

echo ""
echo "=== v79 bench (production target) ==="
runs/fa_v79_fp8 2>&1 | grep -E "perf=" | head -15
