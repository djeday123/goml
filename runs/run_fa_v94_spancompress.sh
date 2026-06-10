#!/bin/bash
# v94 = v89 with Oh declaration moved from pre-kv-loop to post-kv-loop.
#
# Hypothesis (NEW mechanism: span compression, not extraction):
#   v89 SASS LB=3 shows R8 born at 0x0340 (kernel start), used as STG.E.U16 base
#   at 0xb710 (epilogue write). R8/R9 pair (64-bit Oh ptr) lives through ENTIRE
#   kv-loop without being touched. That's 2 reg slots wasted at peak pressure.
#
#   By moving "__half *Oh = O + bh * hs;" from line ~234 (pre-loop) to line ~632
#   (post-loop), the compiler should defer this address computation until needed
#   in epilogue, freeing R8/R9 during the kv-loop body.
#
# Mechanism evidence: v87 −8 regs from MMA reorder showed register allocator
# IS sensitive to operation order. Span compression = same dimension.
#
# Expected outcomes:
#   regs LB=3 168 → 166 (-2): hypothesis confirmed, R8/R9 freed
#   regs LB=3 168 → 167 (-1): partial — compiler already had some lazy eval
#   regs LB=3 168 → 168 (0):  compiler ALREADY deferred Oh CSE; no source-level
#                              effect because LICM/sinking is automatic
#   regs LB=3 168 → 165+ (-3+): bonus — Oh was holding additional dependent values
#
# Correctness MUST be 8/8 (identical arithmetic, just sinked).

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v94 span-compress (Oh sink to epilogue) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v94_spancompress_fp8_forward.cu \
    -o runs/fa_v94_spancompress -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -40

echo ""
echo "=== Reference points ==="
echo "=== v89 ref: 168 regs LB=3, 0 spill, 413T (small) / 466T (large peak) ==="
echo "=== v93 cool-temp probe: 168 regs LB=3 (CSE mirage, extraction failed) ==="
echo "=== v94 hypothesis: span compression different mechanism, may drop 1-2 regs ==="
echo ""
echo "=== Run: attrs + correctness + bench ==="
runs/fa_v94_spancompress
