#!/bin/bash
# v78 variance check: 3 back-to-back runs to confirm the +5-8% gain is real.
# Focus configs: bh=16 sl=4096 (new peak claim), bh=32 sl=2048 (+6% claim), bh=4 sl=2048 (+5% claim).
set -euo pipefail

NVSMI=/usr/bin/nvidia-smi
GOML=/data/lib/podman-data/projects/goml

echo "=== GPU clocks baseline ==="
"$NVSMI" -q -d CLOCK -i 0 | grep -E "Graphics|SM" | head -4

echo ""
echo "######################################"
echo "# 3× v78 — full bench (looking for σ)"
echo "######################################"
for run in 1 2 3; do
    echo ""
    echo "--- v78 run $run ---"
    "$NVSMI" -q -d CLOCK -i 0 | grep -E "Graphics" | head -2 | sed 's/^/    PRE: /'
    "$GOML/runs/fa_v78_fp8" 2>&1 | grep -E "bh=(4|8|16|32).*sl=(2048|4096).*wnd=0|bh=4.*sl=8192" | grep -v "ca=1.*wnd=0"
    "$NVSMI" -q -d CLOCK -i 0 | grep -E "Graphics" | head -2 | sed 's/^/    POST: /'
    sleep 1
done

echo ""
echo "######################################"
echo "# 3× v69 — same configs (control)"
echo "######################################"
for run in 1 2 3; do
    echo ""
    echo "--- v69 run $run ---"
    "$GOML/runs/fa_v69_fp8" 2>&1 | grep -E "bh=(4|8|16|32).*sl=(2048|4096).*wnd=0|bh=4.*sl=8192" | grep -v "ca=1.*wnd=0"
    sleep 1
done
