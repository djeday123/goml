#!/bin/bash
# v96 = v79b hd=128 + ks-batched MMA reorder (port v87 mechanism).
#
# Hypothesis: v87 freed 8 regs + 1.7-2.9% perf on hd=64 by explicit ks=0/1
# batching. hd=128 has 4 ks-steps in QK (vs hd=64's 2) and 2 ks-steps in PV
# (same). Reorder makes scheduler see explicit phase boundaries between
# same-accumulator MMA pairs.
#
# Code changes:
#   QK: 4 explicit batches replace `for ks` outer. Each batch: 8 nt × 2 mi = 16
#       MMAs with fixed Qr[ks] and corresponding K bytes at cl/ch = ks*32+...
#   PV: 2 explicit batches replace `for ks` outer. Each batch: 16 nt × 2 mi = 32
#       MMAs with fresh Pr0/Pr1 loaded from smP at ks*32+... offset.
#
# Expected outcomes:
#   regs 249→241 (−8 as v87) + perf +1-3%: hypothesis confirmed
#   regs unchanged, perf ±0:                compiler already optimal at scheduling
#   regs unchanged, perf −2%+:              v91-style scheduler interference (too tight)

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v96 hd=128 ks-batched ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v96_ksbatched_hd128_fp8_forward.cu \
    -o runs/fa_v96_ksbatched -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -40

echo ""
echo "=== Reference points ==="
echo "  v79b: 249 regs, 2 blocks/SM, NEW peak 534T/541T at bh=64 sl=8192 (large bench)"
echo "  v87 hd=64 (analog): −8 regs (168→160) + 1.7-2.9% perf wide"
echo ""
echo "=== Run: correctness + bench (small + large + sliding window) ==="
runs/fa_v96_ksbatched
