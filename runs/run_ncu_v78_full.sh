#!/bin/bash
# Phase 5: NCu --set full on v78 bh=16 sl=4096 (peak config).
# Goal: measure No Eligible % vs v69's baseline 59.55% to confirm V-overlap mechanism.
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
NCU="$CUDA/bin/ncu"
SRC="$GOML/libs/flash_attention_v78_fp8_forward.cu"
BIN_LI="$GOML/runs/fa_v78_fp8_li"
OUT="$GOML/runs/ncu_v78_full.log"
RPT="$GOML/runs/ncu_v78_full.ncu-rep"

cd "$GOML"

echo "=== Rebuild v78 with -lineinfo for source attribution ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -lineinfo \
    "$SRC" \
    -o "$BIN_LI" -lcudart -Xptxas=-v 2>&1 | tail -6

echo ""
echo "=== NCu --set full on launch 350 (bh=16 sl=4096) ==="
"$NCU" \
    --target-processes all \
    --launch-skip 350 --launch-count 1 \
    --set full \
    --import-source yes \
    --export "$RPT" \
    --csv --page details \
    "$BIN_LI" > "$OUT" 2>&1 || true

echo ""
echo "=== v78 KEY METRICS ==="
echo ""
echo "--- Scheduler / Warp State ---"
grep -E "No Eligible|Active Warps Per Scheduler|Eligible Warps Per Scheduler|Warp Cycles Per Issued|Issued Warp Per Scheduler|One or More Eligible" "$OUT" | grep -v "limited by" | head -15

echo ""
echo "--- SOL / Pipeline ---"
grep -E "Compute \(SM\)|Memory Throughput|Tensor.*highest|DRAM Throughput|L1/TEX|L2 Hit|Mem Pipes Busy|Elapsed Cycles|Duration" "$OUT" | head -15

echo ""
echo "--- Occupancy ---"
grep -E "Theoretical Occupancy|Achieved Occupancy|Block Limit Registers|Block Limit Shared|Theoretical Active Warps" "$OUT" | head -8

echo ""
echo "=== v69 BASELINE (for delta) ==="
echo "No Eligible:              59.55%"
echo "Active Warps/Sched:        1.64"
echo "Eligible Warps/Sched:      0.50"
echo "Warp Cycles/Issued:        4.05"
echo "SOL Compute:              35.32%"
echo "SOL Memory:               33.00%"
echo "Tensor pipe utilization:  34.5%"
echo "DRAM Throughput:           2.26%"
echo "Theoretical Occupancy:    16.67%"
echo "Achieved Occupancy:       13.63%"
