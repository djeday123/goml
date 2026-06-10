#!/bin/bash
# v80b variance: confirm small-grid wins +18-40% and bound large-grid losses.
set -euo pipefail

GOML=/data/lib/podman-data/projects/goml

echo "######################################"
echo "# 3× v80b (full bench)"
echo "######################################"
for run in 1 2 3; do
    echo ""
    echo "--- v80b run $run ---"
    "$GOML/runs/fa_v80b_fp8" 2>&1 | grep -E "perf=" | head -11
    sleep 1
done

echo ""
echo "######################################"
echo "# 3× v79b (control)"
echo "######################################"
for run in 1 2 3; do
    echo ""
    echo "--- v79b run $run ---"
    "$GOML/runs/fa_v79b_fp8" 2>&1 | grep -E "perf=" | head -11
    sleep 1
done
