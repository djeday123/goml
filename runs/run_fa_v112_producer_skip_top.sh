#!/bin/bash
# v112 = v111 + producer-skip TOP-OF-ITER sync (in addition to v111's post-transpose).
# Producer does cpa_wait + __threadfence_block + races to mid-iter.
# Consumer does cpa_wait + bar.sync 4, 96 (cross-warp PV-done sync).
# Risk: K visibility race if producer's fence doesn't complete before consumer reads.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v112 PRODUCER-SKIP top+post-transpose ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v112_producer_skip_top_hd128_fp8_forward.cu \
    -o runs/fa_v112_producer_skip_top -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -30

echo ""
echo "=== References ==="
echo "  v96  : 568 mean, wait 37.78 / barrier 2.00  / Eligible 32.90"
echo "  v110 : 454 mean, wait 35.78 / barrier 10.39 / Eligible 28.49"
echo "  v111 : 484 mean, wait 38.66 / barrier 8.61  / Eligible 29.59"
echo "  v112 target: barrier ↓ further (~5-6pp?) — if correctness holds"
echo ""
echo "=== Run: full correctness + bench ==="
runs/fa_v112_producer_skip_top
