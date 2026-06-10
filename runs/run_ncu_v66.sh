#!/bin/bash
# Diagnostic NCu profile of FA forward FP8 v66.
# Goal: identify dominant warp-stall reason to pick next optimization lever.
#
# We profile only ONE kernel invocation (the largest, most representative)
# at bh=4 sl=4096 — that's where v66 hits 190T = compute-saturated regime.
#
# Sections requested:
#   WarpStateStats     — % time warps stalled per reason (the key diagnostic)
#   SchedulerStats     — issue rate, "no instruction issued" cycles
#   InstructionStats   — instruction mix (count of MMA vs LDG vs SMEM)
#   Occupancy          — sanity check, expected ~25% (2 blocks/SM × 4 warps)
#
# We launch v66 binary, NCu attaches and profiles. v66 runs ~50 kernel
# invocations per config; --launch-skip 200 --launch-count 1 picks just one
# kernel after warmup is settled.

set -euo pipefail

# /usr/bin/ncu = 2021.3.1, не поддерживает sm_120a. Берём свежий из CUDA 13.1.
NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml
OUT=$GOML/runs/v66_ncu_report.txt

# NCu пишет cached секции в HOME — указываем writable.
export HOME=/tmp
mkdir -p /tmp/ncu_cache

cd "$GOML"

echo "=== NCu version ==="
"$NCU" --version | head -3

echo ""
echo "=== Profile fa66_kernel (largest config: bh=4 sl=4096) ==="
echo "    skip 200 invocations (correctness + first 3 bench configs),"
echo "    then profile 1 launch of the bh=4 sl=4096 timer loop."

"$NCU" \
    --target-processes all \
    --kernel-name regex:fa66_kernel \
    --launch-skip 200 \
    --launch-count 1 \
    --section WarpStateStats \
    --section SchedulerStats \
    --section InstructionStats \
    --section Occupancy \
    runs/fa_v66_fp8 \
    2>&1 | tee "$OUT"

echo ""
echo "=== Report saved to $OUT ==="
