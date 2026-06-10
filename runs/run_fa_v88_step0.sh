#!/bin/bash
# v88 step 0 probe — Or_p extracted from registers to static SMEM.
# Measures: ptxas reg count delta vs v87 (160) + actual SMEM growth.
# If reg delta ≥ 33 → 4 blocks reachable by reg budget (need ≤127)
# If SMEM grows by ~16 KB → confirms predicted cost, blocks must drop to fit cap

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v88_step0 Or_p-in-SMEM probe ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v88_step0_orpsmem_fp8_forward.cu \
    -o runs/fa_v88_step0_fp8 -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -40

echo ""
echo "=== v87 baseline reference: 160 regs LB=3, 168 regs v81, 0 spill ==="
echo "=== v88 expected: regs ~128 (down 32 if Or_p fully out), smem +16 KB ==="
echo ""
echo "=== Run: probe correctness check + attrs ==="
runs/fa_v88_step0_fp8
