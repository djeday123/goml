#!/bin/bash
CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"
echo "=== Build v3k (трассировка фаз) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v runs/mbar_repro_v3k.cu -o runs/mbar_repro_v3k -lcudart 2>&1 | grep -E "register|barriers|stack|error|warning|fatal"
echo ""
echo "=== Run sl=300 ca=0 wnd=0 (известно: viset) ==="
timeout 12 runs/mbar_repro_v3k 300 0 0
rc=$?
echo ""
echo "=== rc=$rc ==="
