#!/bin/bash
# QMMA microbench L/T measurement: FP8 e4m3 m16n8k32 + FP4 e2m1 m16n8k64

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

SRC=runs/qmma_microbench_sm120a.cu
BIN=runs/qmma_microbench

echo "=== Build ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    "$SRC" -o "$BIN" -lcudart 2>&1 | tail -20

if [ ! -x "$BIN" ]; then
    echo "BUILD FAILED — checking PTXAS errors..."
    "$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
        "$SRC" -o "$BIN" -lcudart 2>&1 | tail -40
    exit 1
fi

echo ""
echo "=== SASS QMMA count check ==="
QMMA_COUNT=$("$CUDA/bin/cuobjdump" -sass "$BIN" 2>/dev/null | grep -cE 'QMMA|MMA|HMMA' )
echo "  Total MMA-family instructions in SASS: $QMMA_COUNT"

echo ""
echo "=== GPU clock during run (lock with nvidia-smi) ==="
nvidia-smi -q -d CLOCK 2>&1 | grep -A2 "Current Clocks" | head -8

echo ""
echo "=== Run ==="
"$BIN"
