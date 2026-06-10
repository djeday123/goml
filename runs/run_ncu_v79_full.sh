#!/bin/bash
# NCu --set full on v79 bh=16 sl=4096 (peak 397T config).
# Compare No Eligible%, Issued/Sched, Tensor pipe vs v69 (59.55%/0.40/34.5%) and v78 (55.52%/0.44/36.6%).
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
NCU="$CUDA/bin/ncu"
SRC="$GOML/libs/flash_attention_v79_fp8_forward.cu"
BIN_LI="$GOML/runs/fa_v79_fp8_li"
OUT="$GOML/runs/ncu_v79_full.log"
RPT="$GOML/runs/ncu_v79_full.ncu-rep"

cd "$GOML"

echo "=== Rebuild v79 with -lineinfo ==="
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
echo "=== v79 KEY METRICS ==="
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
echo "=== BASELINES (for delta) ==="
echo ""
echo "        | v69     | v78     | v79 expected"
echo "No Eligible:      59.55%  | 55.52%  | ?"
echo "Issued Warp/Sched: 0.40   | 0.44    | ?"
echo "Warp Cycles/Issued: 4.05  | 3.74    | ?"
echo "Tensor pipe:       34.5%  | 36.6%   | ?"
echo "SOL Compute:       35.32% | 38.94%  | ?"
echo "Duration (ns):    654432  | 616800  | ?"
echo "Achieved Occ:      13.63% | 13.87%  | ?"
