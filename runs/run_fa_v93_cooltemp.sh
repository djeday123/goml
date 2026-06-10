#!/bin/bash
# v93 cool-temp probe: falsify the 4-block plan ONE-SHOT experiment.
#
# Hypothesis (analyze_v87_regs_v3 output):
#   v87 used 158 unique regs in kv-loop; gradient analysis showed 37 "cool" regs
#   (cold + cool-no-MMA: in-loop uses ≤ 3, no MMA touch). If these 37 occupied
#   REAL independent slots → freeing them all could drop us toward the 127-reg
#   target for 4 blocks/SM. If they shared slots via reuse → mirage.
#
# This probe removes ONE slice of cool-temps:
#   gid_lane_base / src_low / src_high (3 source vars, ~4-5 reg slots maybe)
# Replaces with inline (threadIdx.x & 0x1c) | ... at each shfl_sync site.
#
# Possible outcomes:
#   ptxas drops 2-5 regs → cool slots were real → SOME 4-block path may exist
#   ptxas unchanged       → compiler CSE'd back to same SASS → mirage confirmed
#   ptxas INCREASES       → forced recompute held more live values → also mirage
#
# Correctness MUST be 8/8 (identical bit-arithmetic). If correctness fails →
# bug in inline expansion, not a fundamental issue.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v93 cool-temp probe ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v93_cooltemp_fp8_forward.cu \
    -o runs/fa_v93_cooltemp -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -40

echo ""
echo "=== Reference points ==="
echo "=== v89 ref: 168 regs LB=3, 0 spill, 413T (small) / 466T (large peak bh=64 sl=8192) ==="
echo "=== v93 prediction: regs drop 2-5 (cool slots real) OR unchanged (CSE mirage) ==="
echo "=== Need 127 regs for 4 blocks/SM. 168→163 marginal; 168→127 not achievable here ==="
echo ""
echo "=== Run: attrs + correctness + bench (small grid only — perf delta is noise here) ==="
runs/fa_v93_cooltemp
