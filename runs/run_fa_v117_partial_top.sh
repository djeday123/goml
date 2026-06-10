#!/bin/bash
# v117 = v111 + partial top sync: consumer-only bar.sync 5, 96 instead of
# block-wide __syncthreads. Producer skips top, issues K immediately, waits
# at mid-iter bar.sync 4, 128 (replaces bar.sync 3, 96), then issues V.
# Expected: barrier % ↓ (producer no longer idle at top sync), wall-clock ±0.5-2%.
# Time-box: one shot, result of any sign is final.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v117 partial top sync ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v117_partial_top_sync_hd128_fp8_forward.cu \
    -o runs/fa_v117_partial_top -lcudart 2>&1 | grep -E "(register|spill|stack|smem|barrier|Function|warning|error)" | head -30

echo ""
echo "=== References ==="
echo "  v111 baseline: 240 regs, 4 barriers, 484T mean PEAK"
echo "  v117 target:   ≤+2 regs, ≤4 barriers, perf ±0.5-2%, barrier % < 8.6%"
echo ""
echo "=== Run: full correctness + bench ==="
runs/fa_v117_partial_top
