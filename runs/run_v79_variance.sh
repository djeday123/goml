#!/bin/bash
# v79 variance: confirm wave-tail gains (bh=8 sl=4096, bh=16 sl=2048) and check
# the small regressions (bh=4 sl=2048, bh=32 sl=2048) are noise not signal.
set -euo pipefail

NVSMI=/usr/bin/nvidia-smi
GOML=/data/lib/podman-data/projects/goml

echo "=== GPU baseline ==="
"$NVSMI" -q -d CLOCK -i 0 | grep -E "Graphics" | head -2

echo ""
echo "######################################"
echo "# 3× v79 full bench"
echo "######################################"
for run in 1 2 3; do
    echo ""
    echo "--- v79 run $run ---"
    "$GOML/runs/fa_v79_fp8" 2>&1 | grep -E "bh=(4|8|16|32).*sl=(2048|4096).*wnd=0|bh=(4|8).*sl=8192" | grep -v "ca=1.*wnd=0"
    sleep 1
done

echo ""
echo "######################################"
echo "# 3× v78 (control)"
echo "######################################"
for run in 1 2 3; do
    echo ""
    echo "--- v78 run $run ---"
    "$GOML/runs/fa_v78_fp8" 2>&1 | grep -E "bh=(4|8|16|32).*sl=(2048|4096).*wnd=0|bh=(4|8).*sl=8192" | grep -v "ca=1.*wnd=0"
    sleep 1
done
