#!/bin/bash
# v111 STEP 4: v110 + real mbarrier (replaces cpa_wait + block-wide __syncthreads
# at top of iter with consumer-only mbarrier.wait + smaller __syncthreads).
# Goal: reduce barrier excess +8pp from v110 → close gap to v96.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v111 STEP 4 warp-spec + real mbarrier ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v111_warpspec_mbarrier_real_hd128_fp8_forward.cu \
    -o runs/fa_v111_warpspec_mbarrier -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -30

echo ""
echo "=== References ==="
echo "  v96  : 242 regs, 568 mean, wait 37.79 / barrier 2.00  / math 8.87 / Eligible 32.90"
echo "  v110 : 243 regs, 454 mean, wait 35.78 / barrier 10.38 / math 10.95 / Eligible 28.49"
echo "  v111 target: barrier ↓, math_pipe sustained, wait stable → Eligible up"
echo ""
echo "=== Run: full correctness + bench ==="
runs/fa_v111_warpspec_mbarrier
