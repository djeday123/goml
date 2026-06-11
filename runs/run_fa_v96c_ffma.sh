#!/bin/bash
# v96c = v96b + FFMA fusion in softmax (Sr raw + fmaf at diff).
# Acceptance criteria (per task spec):
#   1. SASS FADD+FMUL drop significantly
#   2. Regs ≤ 247, stack frame 0, LDL/STL 0 (don't break localfix)
#   3. Correctness 8/8 PASS
#   4. Wall-clock ≥ v96b (target +0.5-1.5%)

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"

SRC=libs/flash_attention_v96c_ffma_hd128_fp8_forward.cu
BIN=runs/fa_v96c_ffma

echo "=== Build v96c ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo "$SRC" -o "$BIN" -lcudart 2>&1 \
    | grep -E "stack|register|spill|barriers|error"

echo ""
echo "=== Criterion 1: SASS instruction census ==="
N_FADD=$("$CUDA/bin/cuobjdump" -sass "$BIN" | grep -cE '\bFADD\b')
N_FMUL=$("$CUDA/bin/cuobjdump" -sass "$BIN" | grep -cE '\bFMUL\b')
N_FFMA=$("$CUDA/bin/cuobjdump" -sass "$BIN" | grep -cE '\bFFMA\b')
N_LDL=$("$CUDA/bin/cuobjdump" -sass "$BIN" | grep -cE '\bLDL\b|\bSTL\b')
echo "  v96b baseline: FADD=142 FMUL=200 FFMA=36 LDL/STL=0"
echo "  v96c:          FADD=$N_FADD FMUL=$N_FMUL FFMA=$N_FFMA LDL/STL=$N_LDL"

echo ""
echo "=== Criterion 3+4: Correctness + perf ==="
"$BIN"
