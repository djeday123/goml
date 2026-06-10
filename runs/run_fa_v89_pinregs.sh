#!/bin/bash
# v89 hd=64 + P-in-regs (shfl-based gather, skip smP).
# Targets: short_scoreboard 7.48% (smP round-trip) + barrier 3.57% (2 syncs removed)
# Trade-off: 32 shfls per kv-iter vs 48 SMEM ops + 2 __syncthreads
# Expected: +1-3% perf if shfl is faster than SMEM round-trip + sync wait

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v89 P-in-regs (LB=2 and LB=3, -lineinfo) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v89_pinregs_fp8_forward.cu \
    -o runs/fa_v89_pinregs_fp8 -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -40

echo ""
echo "=== Run: attrs + correctness ×8×2 + A/B perf ×11×2 ×variance 3 ==="
runs/fa_v89_pinregs_fp8
