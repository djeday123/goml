#!/bin/bash
# v83 hd=64 + TMA-K — K loaded via cp.async.bulk.tensor in kv-loop (per-iter).
# Q stays cp.async.cg (v82 closed Q-TMA as wasted), V stays cp.async.cg (isolated K).
# Lessons from v82 applied: __align__(128) raw, __grid_constant__ k_tmap,
# single mbarrier with alternating parity.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v83 TMA-K (LB=2 and LB=3) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v \
    libs/flash_attention_v83_hd64_tmak_fp8_forward.cu \
    -o runs/fa_v83_tmak_fp8 -lcudart -lcuda 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -40

echo ""
echo "=== Run: attrs + correctness ×8×2 + A/B perf ×11×2 ×variance 3 ==="
runs/fa_v83_tmak_fp8
