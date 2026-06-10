#!/bin/bash
# v81 hd=64 honest A/B — LB=2 (2 blocks/SM) vs LB=3 (3 blocks/SM) on same kernel.
# Phase 1: 8 correctness configs × both LBs (causal, sliding window, edge cases).
# Phase 2: 11 perf configs × both LBs, variance ×3 → best/median/worst → ratio LB3/LB2.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v81 (LB=2 and LB=3 instantiations) ==="
# -lineinfo: embed .cu file:line mapping into SASS for NCu source attribution.
# Zero perf cost (metadata only). REQUIRED for run_ncu_v81_source.sh to work.
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v81_hd64_fp8_forward.cu \
    -o runs/fa_v81_hd64_fp8 -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -40

echo ""
echo "=== Run: attrs + correctness ×8×2 + A/B perf ×11×2 ×variance 3 ==="
runs/fa_v81_hd64_fp8
