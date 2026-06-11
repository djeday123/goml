#!/bin/bash
# Build v111 in TWO variants:
#   1. fa_v111_producer_skip       — with -lineinfo (for NCu source attribution)
#   2. fa_v111_producer_skip_bench — without -lineinfo (clean optimized for bench)
# Per recommendation: profile lineinfo build, bench clean build.
# Note: -lineinfo is NOT -G; doesn't affect optimization. Both should perform
# identically. Dual build is for measurement hygiene + bench-noise sensitivity.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

SRC=libs/flash_attention_v111_producer_skip_hd128_fp8_forward.cu
COMMON_FLAGS="-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -lcudart"

echo "=== Build #1: PROFILE build (with -lineinfo) ==="
"$CUDA/bin/nvcc" $COMMON_FLAGS -Xptxas=-v -lineinfo \
    "$SRC" -o runs/fa_v111_producer_skip 2>&1 \
    | grep -E "(register|spill|stack|smem|error)" | head -5

echo ""
echo "=== Build #2: BENCH build (clean, no -lineinfo) ==="
"$CUDA/bin/nvcc" $COMMON_FLAGS -Xptxas=-v \
    "$SRC" -o runs/fa_v111_producer_skip_bench 2>&1 \
    | grep -E "(register|spill|stack|smem|error)" | head -5

echo ""
echo "=== Binary sizes ==="
ls -la runs/fa_v111_producer_skip runs/fa_v111_producer_skip_bench

echo ""
echo "=== Quick perf parity check (PEAK config bh=64 sl=8192 only) ==="
echo "--- PROFILE build (lineinfo) ---"
runs/fa_v111_producer_skip --loop 9 50 2>&1 | tail -2
echo "--- BENCH build (clean) ---"
runs/fa_v111_producer_skip_bench --loop 9 50 2>&1 | tail -2

echo ""
echo "=== Usage ==="
echo "  For NCu source attribution:  fa_v111_producer_skip"
echo "  For wall-clock benches:      fa_v111_producer_skip_bench"
