#!/bin/bash
# v118 = v111 + local memory fix: replace `uint8_t *smK[2]` array (forces stack
# frame + LDL.64 on dynamic index in hot loop, 22.5M local sectors per launch)
# with arithmetic-stride base pointer.
# Acceptance criteria (per task spec, in order):
#   1. ptxas: "0 bytes stack frame" + regs ≤243
#   2. SASS: zero LDL/STL instructions
#   3. Correctness 8/8 PASS
#   4. Wall-clock not regressing on production grids

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

SRC=libs/flash_attention_v118_localfix_hd128_fp8_forward.cu
BIN=runs/fa_v118_localfix
COMMON="-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -lcudart"

echo "=== Build v118 (lineinfo for profile) ==="
"$CUDA/bin/nvcc" $COMMON -Xptxas=-v -lineinfo "$SRC" -o "$BIN" 2>&1 | grep -E "(stack|register|spill|smem|barrier|error)" | head -10

echo ""
echo "=== Criterion 1: ptxas stack frame check ==="
"$CUDA/bin/nvcc" $COMMON -Xptxas=-v "$SRC" -o /tmp/v118_check 2>&1 | grep -E "stack|register|barriers"

echo ""
echo "=== Criterion 2: SASS LDL/STL count ==="
LDL_COUNT=$("$CUDA/bin/cuobjdump" -sass "$BIN" 2>/dev/null | grep -cE 'LDL|STL')
echo "  LDL/STL instructions in SASS: $LDL_COUNT  (expect 0)"

echo ""
echo "=== Criterion 3: Correctness + perf ==="
"$BIN"
