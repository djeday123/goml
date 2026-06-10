#!/bin/bash
# v110 STEP 3: 1 producer + 3 consumer, Br=96, K=2 V=1.
# Closes Eligible gap from v108/v109 (only 2 consumer warps did MMA).
# Hypothesis: more MMA warps → math_pipe utilization up → perf approaches v96.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v110 STEP 3 warp-spec 1P+3C Br=96 ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v110_warpspec_1p3c_br96_hd128_fp8_forward.cu \
    -o runs/fa_v110_warpspec_1p3c -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -30

echo ""
echo "=== References ==="
echo "  v96  : 242 regs, 48.5 KB, 564 mean, Eligible 32.90% (math_pipe 8.87%)"
echo "  v108 : 228 regs, 48.5 KB, 362 mean, Eligible 26.74% (math_pipe 6.59% — MMA under)"
echo "  v109 : 241 regs, 48.5 KB, 363 mean, Eligible 26.88% (barrier moved, not killed)"
echo "  v110 target: ~245-260 regs, 44.5 KB SMEM, 2 blocks/SM"
echo "    Expected perf 420-470T if 3 consumers close MMA gap"
echo ""
echo "=== Run: full correctness + bench ==="
runs/fa_v110_warpspec_1p3c
