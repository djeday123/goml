#!/bin/bash
# v99 = v96 + cp.async.ca (Cache All) instead of cp.async.cg (Cache Global).
#
# CHANGE: single line in cpa16() helper. cp.async.cg = bypass L1, only L2.
# cp.async.ca = cache in both L1 and L2.
#
# HYPOTHESIS:
#   .ca might reduce long_scoreboard 1.77% if L1 cache hit on cp.async path
#   helps later reads. PROBABLY NULL — HBM latency unchanged, just cache hint.
#
# COUNTER-RISKS:
#   1. L1 pollution may slow regular SMEM access (mio_throttle ↑?)
#   2. cp.async.ca for 16B may have different completion model
#   3. Could regress perf if L1 evictions happen
#
# CRITERION:
#   long_scb 1.77 → 1.0% + perf +0.3%: surprise, .ca helps
#   long_scb unchanged + perf null:     confirms HBM-bound, structural
#   long_scb unchanged + perf negative: .ca hurts (L1 pollution), revert

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v99 hd=128 cp.async.ca probe ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v99_cpasyncca_hd128_fp8_forward.cu \
    -o runs/fa_v99_cpasyncca -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -40

echo ""
echo "=== Reference (v96 30-run sustained) ==="
echo "  v96 mean=564.36T sd=1.74 median=563.76 worst=562.39 best=570.86"
echo "  v96 long_scb peak: 1.77%"
echo ""
echo "=== Run: correctness + bench ==="
runs/fa_v99_cpasyncca

echo ""
echo "=== Watch ==="
echo "  v99 bh=64 sl=8192 perf:"
echo "    > v96 → .ca helped (surprise)"
echo "    ≈ v96 → null (HBM-bound confirmed)"
echo "    < v96 → .ca hurt (L1 pollution)"
