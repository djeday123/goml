#!/bin/bash
# v113 = v111 + top-of-iter via bar.arrive (non-blocking producer) + bar.sync 6,128.
# Producer cpa_wait + bar.arrive (32 arrivals, NO wait) → races to mid-iter.
# Consumer cpa_wait (no-op) + bar.sync 6,128 → waits for 32+96=128 arrivals.
# Pattern guarantees producer's cp.async lands before consumer reads, WITHOUT
# producer blocking at block-wide sync.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v113 PRODUCER bar.arrive top sync ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v113_producer_arrive_hd128_fp8_forward.cu \
    -o runs/fa_v113_producer_arrive -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -30

echo ""
echo "=== References ==="
echo "  v110 : 454 mean PEAK, barrier 10.39"
echo "  v111 : 484 mean PEAK, barrier 8.61"
echo "  v112 (race): 485 PEAK + 421-475 mid (+8-10 mid) but FAIL correctness"
echo "  v113 target: v112's +8-10 mid AND correctness 8/8"
echo ""
echo "=== Run: full correctness + bench ==="
runs/fa_v113_producer_arrive
