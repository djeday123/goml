#!/bin/bash
CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v111_warpspec_mbarrier_real_hd128_fp8_forward.cu \
    -o runs/fa_v111_warpspec_mbarrier -lcudart 2>&1 | grep -E "(register|spill|stack|smem|warning|error)" | head -10
echo "Build done. Running smoke bh=1 sl=300 wnd=96 × 100..."
runs/fa_v111_warpspec_mbarrier --smoke 1 300 96 100
echo "smoke rc=$?"
