#!/bin/bash
# v84 hd=64 + TMA-K + SWIZZLE_64B — isolate bank-conflict effect vs mbarrier overhead.
# v83 used SWIZZLE_NONE + plain K reads → 4-way bank conflicts → No Eligible +6pp WORSE.
# v84: SWIZZLE_64B (HW chunk^(row&3) matches manual swz_byte) + restore swizzled reads.
# Three outcomes:
#   correctness PASS + No Eligible back to ~43% + perf back to v81 → bank conflicts were the driver
#   correctness PASS + No Eligible still high → mbarrier/single-lane is the driver, close TMA
#   correctness FAIL → HW SWIZZLE_64B pattern differs from manual swz_byte, hypothesis wrong

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v84 TMA-K + SWIZZLE_64B (LB=2 and LB=3) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v \
    libs/flash_attention_v84_hd64_tmak_swz_fp8_forward.cu \
    -o runs/fa_v84_tmak_swz_fp8 -lcudart -lcuda 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -40

echo ""
echo "=== Run: attrs + correctness ×8×2 + A/B perf ×11×2 ×variance 3 ==="
runs/fa_v84_tmak_swz_fp8
