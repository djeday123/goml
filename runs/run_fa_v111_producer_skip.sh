#!/bin/bash
# v111 PRODUCER-SKIP transpose: transpose_v only by consumer warps (96 threads).
# Producer skips transpose + post-transpose sync → producer races ahead.
# K stays as v110 (raw cp.async, top-of-iter block-wide __syncthreads).
# Goal: kill producer's post-transpose barrier wait (was contributing to v110's 10.4% barrier).

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v111 PRODUCER-SKIP transpose hd=128 ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v111_producer_skip_hd128_fp8_forward.cu \
    -o runs/fa_v111_producer_skip -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -30

echo ""
echo "=== References ==="
echo "  v96  : 242 regs, 568 mean, wait 37.79 / barrier 2.00  / Eligible 32.90"
echo "  v110 : 243 regs, 454 mean, wait 35.78 / barrier 10.38 / Eligible 28.49"
echo "  v111 target: barrier ↓ (producer skips post-transpose sync), correctness 8/8 PASS"
echo "  Risk: V race if cp.async lands too fast (consumer transpose ~100-200 cyc, cp.async ~500+ cyc)"
echo ""
echo "=== Run: full correctness + bench ==="
runs/fa_v111_producer_skip
