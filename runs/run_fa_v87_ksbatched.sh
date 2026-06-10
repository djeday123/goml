#!/bin/bash
# v87 hd=64 + Option C MMA reorder — explicit ks=0 batch then ks=1 batch.
# QK: 16 MMAs ks=0 (all nt × all mi) → 16 MMAs ks=1 (same accumulators)
# PV: 16 MMAs ks=0 → 16 MMAs ks=1
# No extra accumulators. Same Sr_p/Or_p. Gap between same-acc MMAs = 15 ops.
# Test if compiler was already optimal at scheduling 16-op gap across ks.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v87 ks-batched MMA (LB=2 and LB=3, -lineinfo) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v87_hd64_ksbatched_fp8_forward.cu \
    -o runs/fa_v87_ksbatched_fp8 -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -40

echo ""
echo "=== Run: attrs + correctness ×8×2 + A/B perf ×11×2 ×variance 3 ==="
runs/fa_v87_ksbatched_fp8
