#!/bin/bash
# v80c Phase 3 variance + correctness comparison vs v79b.
set -euo pipefail

GOML=/data/lib/podman-data/projects/goml

echo "######################################"
echo "# 3× v80c Phase 3 (full bench)"
echo "######################################"
for run in 1 2 3; do
    echo ""
    echo "--- v80c Phase 3 run $run ---"
    "$GOML/runs/fa_v80c_fp8" 2>&1 | grep -E "perf=" | head -11
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
