#!/bin/bash
CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v118b_addrhoist_hd128_fp8_forward.cu \
    -o runs/fa_v118b_addrhoist -lcudart 2>&1 | grep -E "(register|spill|stack|warning|error)" | head -10
