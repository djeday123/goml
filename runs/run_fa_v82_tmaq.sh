#!/bin/bash
# v82 hd=64 + TMA-Q — Q-only TMA-tensor descriptor replaces cp.async.cg for Q.
# Measure: numRegs delta (target ≤127 to open 4 blocks/SM at hd=64) + correctness.
# Two LB variants compared as in v81. Perf only on WIN configs (grid>376) so the
# +27% wave-reduction regime where TMA-Q effect is visible.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v82 TMA-Q (LB=2 and LB=3) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v \
    libs/flash_attention_v82_hd64_tmaq_fp8_forward.cu \
    -o runs/fa_v82_tmaq_fp8 -lcudart -lcuda 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -40

echo ""
echo "=== Run: attrs + correctness ×8×2 + A/B perf ×11×2 ×variance 3 ==="
runs/fa_v82_tmaq_fp8
