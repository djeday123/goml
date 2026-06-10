#!/bin/bash
# v109 STEP 2: v108 + named-bar consumer-only sync (replaces block-wide __syncthreads).
# Goal: kill barrier 19.48% stall from v108. Producer no longer waits at
# consumer-only barriers → pipelines cp.async ahead.
#
# Diagnostic interpretation:
#   barrier drops 19→3 AND wait stays 39   → wait is math-latency, warp-spec
#                                           can't help. CLOSE warp-spec path.
#   barrier drops AND wait drops 39→33     → real overlap, continue (step 3
#                                           = role balance)
#   barrier doesn't drop                    → my named-bar usage wrong, debug

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v109 STEP 2 warp-spec + named-bar ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v109_warpspec_mbarrier_hd128_fp8_forward.cu \
    -o runs/fa_v109_warpspec_mbarrier -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -30

echo ""
echo "=== References ==="
echo "  v96       : 242 regs, 568 peak / 564 mean. NCu: wait 37.79 / barrier 2.00 / Eligible 32.90"
echo "  v108 step1: 228 regs, 362 mean (-36%).      NCu: wait 39.13 / barrier 19.48 / Eligible 26.74"
echo "  v109 target: barrier should drop to ~3-5%. Other stalls?"
echo ""
echo "=== Run: full correctness + bench ==="
runs/fa_v109_warpspec_mbarrier
