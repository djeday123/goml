#!/bin/bash
# v91 = v89 + K-preload (QK) + V-preload (PV) — combined K+V preload.
# Compound experiment: K alone was null (v90), V alone untested.
# Risk: +16 regs in QK + +16 regs in PV could compound to exceed LB=3 budget.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v91 KV-preload combined ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v91_kvpreload_fp8_forward.cu \
    -o runs/fa_v91_kvpreload -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -40

echo ""
echo "=== v89 ref: 168 regs LB=3, 413T small / 466T peak ==="
echo "=== v90 K-only: 168 regs LB=3, perf == v89 (null) ==="
echo "=== v91 K+V combined: regs? perf vs v89 (predicted ≈ null based on v90) ==="
echo ""
echo "=== Run: attrs + correctness + bench (small + large grid) ==="
runs/fa_v91_kvpreload
