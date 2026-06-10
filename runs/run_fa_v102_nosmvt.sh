#!/bin/bash
# v102 = v101 + drop smV_T + drop transpose_v + LB=3 attempt.
# PERF-ONLY PROBE. Correctness WILL FAIL (smV K-major bytes interpreted N-major).
# Goal: measure SMEM/reg/perf impact IF 3 blocks/SM achievable.
#
# Outcomes:
#   regs > 170 → ptxas spills or 1 block/SM → catastrophe → close 3-block path
#   regs ≤ 170 AND 3 blocks load → measure perf vs v96 568T
#     +5-10% peak → invest in real N-major V path (external transpose API)
#     null/negative → 3 blocks doesn't help, occupancy NOT the only bottleneck

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v102 hd=128 no-smV_T LB=3 attempt ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v102_nosmvt_hd128_fp8_forward.cu \
    -o runs/fa_v102_nosmvt -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -20

echo ""
echo "=== Reference ==="
echo "  v96: 242 regs, 48.5KB, 2 blocks/SM, 568T peak (3-run)"
echo "  v101: 251 regs, 40.5KB, 2 blocks/SM, 553T peak (-2.6%)"
echo "  v102: target 31.8KB → 3 blocks SMEM-fits. Need regs ≤170 for 3 blocks reg-fit."
echo "  Reg budget 256/thread × 128 threads × 3 blocks = 65536 reg file (full SM cap)"
echo ""
echo "=== Run: skip correctness validation, perf only ==="
runs/fa_v102_nosmvt

echo ""
echo "=== Watch ==="
echo "  ptxas regs: must be ≤ 170 for 3 blocks to actually load."
echo "  If 251 (v101 level) → only 1 block/SM (catastrophe). Spill check."
echo "  3-block actual load → see if peak perf changes vs v96 568T."
echo "  bench correctness will REPORT mismatches — IGNORE (this is perf probe)."
