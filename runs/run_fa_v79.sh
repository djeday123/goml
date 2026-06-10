#!/bin/bash
# v79 = v78 + 3 levers:
#   L2: end-of-iter sync 518 removed (redundant after V cp.async moved mid-iter in v78)
#   L3: branch-free K/V prefetch (ternary row count, no `if` guards)
#   L4: ex2.approx.f16x2 for softmax exp (Sr/rmax in log2 space via fs *= log2(e))
# L1 (K-before-transpose) was BLOCKED — race with prev_V_slot==nxt_K aliasing.
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v79 ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    libs/flash_attention_v79_fp8_forward.cu \
    -o runs/fa_v79_fp8 -lcudart -Xptxas=-v 2>&1 | tail -8

echo ""
echo "=== v79 correctness + bench ==="
runs/fa_v79_fp8

echo ""
echo "=== v78 baseline (for side-by-side) ==="
runs/fa_v78_fp8 2>&1 | grep -E "perf=|max_diff" | head -20
