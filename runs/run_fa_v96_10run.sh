#!/bin/bash
# v96 champion — LARGE-variance bench (10 runs per config on full sweep).
# ~20 sec total runtime.
# Safer source-edit + restore via trap.

set -uo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

V96_SRC="libs/flash_attention_v96_ksbatched_hd128_fp8_forward.cu"

cp "$V96_SRC" "$V96_SRC.bak"
trap 'mv "$V96_SRC.bak" "$V96_SRC"' EXIT

echo "=== Step 1: VARIANCE_RUNS = 3 → 10 ==="
sed -i 's/const int VARIANCE_RUNS = 3;/const int VARIANCE_RUNS = 10;/' "$V96_SRC"

echo "=== Step 2: build v96 ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    "$V96_SRC" -o runs/fa_v96_10run -lcudart 2>&1 | grep -E "(register|spill|error)" | head -5

echo ""
echo "=== Full bench (16 configs × 10 runs each) ==="
echo "  Output format per config:"
echo "    best/med/worst from sorted 10 runs + mean + sd + RAW: v1 ... v10"
echo ""
runs/fa_v96_10run
