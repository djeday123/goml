#!/bin/bash
# v95 = v89 with softmax inner loop and Pf_pair quantize loop MERGED.
#
# Hypothesis (TRUE computation reorder, like v87 −8):
#   v89 has two consecutive loops in softmax phase:
#     loop A (lines 484-503): for nt,mi: compute P_top[nt][mi], P_bot[nt][mi]
#                                       (as __half2 arrays, 16+16 = 32 fp16x2 vals)
#                                       + accumulate ns
#     loop B (lines 519-529): for nt,mi: read P_top/P_bot → convert fp16x2 → fp8x2,
#                                       pack into Pf_pair[nt][mi]
#
#   Between A and B, the 32 fp16x2 values must live in 16-32 registers.
#   Merging A and B: scalar p_top_u, p_bot_u live ONLY one iter, killed when
#   Pf_pair[nt][mi] is assigned. Real instruction-stream reorder.
#
# Expected outcomes:
#   regs LB=3 168 → 160 or lower: hypothesis confirmed, P_top/P_bot were held
#   regs LB=3 168 → 168:          compiler already merged via dead-store elim,
#                                  arrays weren't actually held in regs
#   regs LB=3 168 → 165-167:      partial merge benefit
#
# Correctness MUST be 8/8 (identical math: ex2 → ns accumulate → fp16→fp8 pack).

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v95 merged softmax+quantize ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v95_merged_fp8_forward.cu \
    -o runs/fa_v95_merged -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -40

echo ""
echo "=== Reference points ==="
echo "=== v89 ref: 168 regs LB=3, 0 spill, 413T (small) / 466T (large peak) ==="
echo "=== v87 historical: 160 regs LB=3 (before P-in-regs added 8 for Pf_pair shfl machinery) ==="
echo "=== v95 hypothesis: merge of 2 loops kills 16-32 reg P_top/P_bot intermediate ==="
echo ""
echo "=== Run: attrs + correctness + bench ==="
runs/fa_v95_merged
