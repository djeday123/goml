#!/bin/bash
# NCu profile v79 on bh=16 sl=2048 (wave-tail config — was 316T on v78, jumped to 342T on v79).
# Goal: understand the +8% mechanism. Compare to v79 peak profile (bh=16 sl=4096 = 397T = +0.3%).
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
NCU="$CUDA/bin/ncu"
BIN_LI="$GOML/runs/fa_v79_fp8_li"
OUT="$GOML/runs/ncu_v79_wavetail.log"
RPT="$GOML/runs/ncu_v79_wavetail.ncu-rep"

cd "$GOML"

# Launch index math:
#   8 correctness + 5×55 (configs 0..4 = 4×1024, 4×2048, 8×2048, 4×4096, 8×4096) = 283.
#   Config 5 (bh=16 sl=2048): launches 283..337. Mid-timing: 310.
echo "=== NCu --set full on launch 310 (bh=16 sl=2048) ==="
"$NCU" \
    --target-processes all \
    --launch-skip 310 --launch-count 1 \
    --set full \
    --import-source yes \
    --export "$RPT" \
    --csv --page details \
    "$BIN_LI" > "$OUT" 2>&1 || true

echo ""
echo "=== v79 wave-tail (bh=16 sl=2048) KEY METRICS ==="
echo ""
echo "--- Scheduler / Warp State ---"
grep -E "No Eligible|Active Warps Per Scheduler|Eligible Warps Per Scheduler|Warp Cycles Per Issued|Issued Warp Per Scheduler|One or More Eligible" "$OUT" | grep -v "limited by" | head -15

echo ""
echo "--- SOL / Pipeline ---"
grep -E "Compute \(SM\)|Memory Throughput|Tensor.*highest|DRAM Throughput|L1/TEX|L2 Hit|Mem Pipes Busy|Elapsed Cycles|Duration" "$OUT" | head -15

echo ""
echo "--- Occupancy ---"
grep -E "Theoretical Occupancy|Achieved Occupancy|Block Limit|Theoretical Active Warps" "$OUT" | head -8

echo ""
echo "=== Compare: v79 PEAK (bh=16 sl=4096) vs v79 WAVE-TAIL (bh=16 sl=2048) ==="
echo ""
echo "                    | PEAK     | WAVE-TAIL (expect)"
echo "No Eligible:           58.05%   | ?"
echo "Issued Warp/Sched:      0.42    | ?"
echo "Warp Cycles/Issued:     3.90    | ?"
echo "Tensor pipe:           37.2%    | ?"
echo "SOL Compute:           37.2%    | ?"
echo "Duration (ns):       607,520    | ?"
echo "Achieved Occ:         13.64%    | ?"
echo "Grid:               (512, 1, 1) | (256, 1, 1)"
