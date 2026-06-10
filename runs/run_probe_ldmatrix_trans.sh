#!/bin/bash
# Standalone ldmatrix.x4.trans.shared.b16 probe for sm_120a.
# Verifies: (1) compiles, (2) runs without trap, (3) dumps per-thread layout.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build probe ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    libs/probe_ldmatrix_trans.cu \
    -o runs/probe_ldmatrix_trans -lcudart 2>&1 | grep -E "(error|warning)" | head -5

if [ ! -x runs/probe_ldmatrix_trans ]; then
    echo "BUILD FAILED — ldmatrix.x4.trans likely not supported in this form"
    exit 1
fi

echo ""
echo "=== Run probe ==="
runs/probe_ldmatrix_trans

echo ""
echo "================================================================"
echo "EXPECTED layout for MMA m16n8k32 kind::f8f6f4 B-operand:"
echo "  Each thread holds 8 bytes total: b0 (4 bytes k=0..3) + b1 (4 bytes k=16..19)"
echo "  for ONE fixed n (column). Lane → n mapping: tid 0..3 → n=0, 4..7 → n=1, ..."
echo ""
echo "ldmatrix.x4 returns 16 bytes per thread (4 b32 registers)."
echo "Adapts to MMA layout if each register matches expected b0/b1 of 2 different tiles."
echo "================================================================"
