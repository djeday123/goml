#!/bin/bash
# clock64 probe on v111 — measure per-warp arrival skew at top-of-iter sync.
# Test the "barrier 8.6% = arrival skew" hypothesis.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v111 clockprobe ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v111_clockprobe_hd128_fp8_forward.cu \
    -o runs/fa_v111_clockprobe -lcudart 2>&1 | grep -E "(register|spill|stack|smem|error)" | head -10

echo ""
echo "=== Run clkprobe cfg=9 (bh=64 sl=8192) probe_iter=5 ==="
runs/fa_v111_clockprobe --clkprobe 9 5

echo ""
echo "=== Run clkprobe cfg=9 probe_iter=20 (later iter, deeper steady state) ==="
runs/fa_v111_clockprobe --clkprobe 9 20

echo ""
echo "=== Run clkprobe cfg=2 (bh=8 sl=2048) probe_iter=5 ==="
runs/fa_v111_clockprobe --clkprobe 2 5
