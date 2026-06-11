#!/bin/bash
# v96d = v96b + ONLY Point 3 (rsexp = fmaf(rsexp, rsc, ns)). Zero new live values.
# IRON RULE: regs ≤ 247 strict. If +1 reg → revert immediately.
CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"

SRC=libs/flash_attention_v96d_ffma_minimal_hd128_fp8_forward.cu
BIN=runs/fa_v96d_ffma_minimal

echo "=== Build v96d ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo "$SRC" -o "$BIN" -lcudart 2>&1 \
    | grep -E "stack|register|spill|barriers|error"

echo ""
echo "=== Criterion 0 (HARD STOP): registers ≤ 247? ==="
REGS=$("$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v "$SRC" -o /tmp/v96d_check -lcudart 2>&1 \
    | grep -oE "Used [0-9]+ registers" | grep -oE "[0-9]+" | head -1)
echo "  v96b baseline regs: 247"
echo "  v96d regs:          $REGS"
if [ "$REGS" -gt 247 ]; then
    echo "  ❌ ABORT: regs > 247, reverting"
    exit 1
else
    echo "  ✅ OK: regs ≤ 247"
fi

echo ""
echo "=== SASS census ==="
N_FADD=$("$CUDA/bin/cuobjdump" -sass "$BIN" | grep -cE '\bFADD\b')
N_FMUL=$("$CUDA/bin/cuobjdump" -sass "$BIN" | grep -cE '\bFMUL\b')
N_FFMA=$("$CUDA/bin/cuobjdump" -sass "$BIN" | grep -cE '\bFFMA\b')
N_LDL=$("$CUDA/bin/cuobjdump" -sass "$BIN" | grep -cE '\bLDL\b|\bSTL\b')
echo "  v96b baseline: FADD=142 FMUL=200 FFMA=36 LDL/STL=0"
echo "  v96d:          FADD=$N_FADD FMUL=$N_FMUL FFMA=$N_FFMA LDL/STL=$N_LDL"

echo ""
echo "=== Correctness + perf ==="
"$BIN"
