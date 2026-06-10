#!/bin/bash
# Fresh build of v96 champion + 3-run bench on full config sweep.
# Same as original v96 default bench (VARIANCE_RUNS = 3 already in source).

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v96 (champion hd=128) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v96_ksbatched_hd128_fp8_forward.cu \
    -o runs/fa_v96_ksbatched -lcudart 2>&1 | grep -E "(register|spill|error)" | head -5

echo ""
echo "=== Confirm VARIANCE_RUNS = 3 ==="
grep "VARIANCE_RUNS = " libs/flash_attention_v96_ksbatched_hd128_fp8_forward.cu | head -1

echo ""
echo "=== Full bench (16 configs × 3 runs each) ==="
runs/fa_v96_ksbatched
