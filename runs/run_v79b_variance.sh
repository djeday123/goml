#!/bin/bash
# v79b variance check: confirm new peak 401.4T and +2.6% on bh=8 sl=2048 are real.
set -euo pipefail

GOML=/data/lib/podman-data/projects/goml

echo "######################################"
echo "# 3× v79b"
echo "######################################"
for run in 1 2 3; do
    echo ""
    echo "--- v79b run $run ---"
    "$GOML/runs/fa_v79b_fp8" 2>&1 | grep -E "perf=" | head -11
    sleep 1
done

echo ""
echo "######################################"
echo "# 3× v79 (control)"
echo "######################################"
for run in 1 2 3; do
    echo ""
    echo "--- v79 run $run ---"
    "$GOML/runs/fa_v79_fp8" 2>&1 | grep -E "perf=" | head -11
    sleep 1
done
