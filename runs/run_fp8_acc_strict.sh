#!/bin/bash
# Strict A/B: kind::f8f6f4.f16 vs kind::f8f6f4.f32 on real GEMM workload.
set -euo pipefail

CUDA=/usr/local/cuda-13.1
RUNS=/data/lib/podman-data/projects/goml/runs
cd "$RUNS"

echo "=== Compile kernel cubin (sm_120a) ==="
"$CUDA/bin/nvcc" -arch=sm_120a -cubin fp8_acc_strict_kernel.cu \
    -o fp8_acc_strict.cubin -Xptxas=-v 2>&1 | tail -20

echo ""
echo "=== Confirm both kernels use kind::f8f6f4 with expected acc ==="
"$CUDA/bin/cuobjdump" --dump-sass fp8_acc_strict.cubin \
    | grep -E "QMMA" | sort | uniq -c | sort -rn

echo ""
echo "=== Compile host driver ==="
"$CUDA/bin/nvcc" -O2 fp8_acc_strict_host.cu -lcuda -o fp8_acc_strict_host

echo ""
echo "=== Run A/B ==="
./fp8_acc_strict_host fp8_acc_strict.cubin
