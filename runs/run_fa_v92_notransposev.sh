#!/bin/bash
# v92 = v89 + eliminate transpose_v entirely.
# PV reads V via byte-gather from prev_V_slot (K×N row-major layout) instead of
# uint32-stride from smV_T (N×K layout). Saves: 1 __syncthreads + transpose_v
# function call. Cost: 4× SMEM byte-loads per b0/b1 in PV (8 nt × 2 batches × 4
# byte-loads × 2 b-operands = 128 byte-loads/iter vs 16 uint32-loads/iter).
#
# Expected outcomes:
#   1. Catastrophe (-30 to -50%): SMEM throughput becomes the floor → like v70.
#   2. Mild regression (-5 to -15%): byte-gather slower than uint32 but freed regs.
#   3. Surprise (-2 to +2%): compiler emits PRMT-pack from LDS.32 instead of 4×LDS.U8.
#
# Risk: NO HW damage. Worst case = perf regression or correctness fail.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v92 no-transpose_v + byte-gather V ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v92_notransposev_fp8_forward.cu \
    -o runs/fa_v92_notransposev -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -40

echo ""
echo "=== Reference points ==="
echo "=== v89 ref: 168 regs LB=3, 0 spill, 413T (small) / 466T (large peak bh=64 sl=8192) ==="
echo "=== v91 K+V preload: 164 regs LB=3, perf -0.2 to -2.2% (compound scheduler interference) ==="
echo "=== v92 prediction: regs may drop (smV_T DCE'd, no transpose_v); SMEM ops 8× → likely -5 to -15% ==="
echo ""
echo "=== Run: attrs + correctness + bench (small + large grid for 466T zone) ==="
runs/fa_v92_notransposev
