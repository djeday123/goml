#!/bin/bash
# v80c Phase 3 NCu — measure No Eligible on bh=16 sl=4096 (peak).
# Compare to v79b 58.05% / v78 55.52%.
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
NCU="$CUDA/bin/ncu"
SRC="$GOML/libs/flash_attention_v80c_fp8_forward.cu"
BIN_LI="$GOML/runs/fa_v80c_fp8_li"
OUT="$GOML/runs/ncu_v80c_full.log"

cd "$GOML"

echo "=== Rebuild v80c with -lineinfo ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -lineinfo \
    "$SRC" \
    -o "$BIN_LI" -lcudart 2>&1 | tail -6

echo ""
echo "=== NCu --set full on launch 350 (bh=16 sl=4096 mid-timing) ==="
"$NCU" \
    --target-processes all \
    --launch-skip 350 --launch-count 1 \
    --set full \
    --csv --page details \
    "$BIN_LI" > "$OUT" 2>&1 || true

echo ""
echo "=== v80c Phase 3 KEY METRICS ==="
grep -E "No Eligible|Active Warps Per Scheduler|Eligible Warps Per Scheduler|Warp Cycles Per Issued|Issued Warp Per Scheduler|One or More Eligible" "$OUT" | grep -v "limited by" | head -10

echo ""
grep -E "Compute \(SM\)|Tensor.*highest|DRAM|Mem Pipes|Duration" "$OUT" | head -10

echo ""
echo "=== BASELINES ==="
echo "v69     | v78     | v79     | v79b    | v80c expected"
echo "59.55%  | 55.52%  | 58.05%  | (~)     | ? (target: <55%)"
