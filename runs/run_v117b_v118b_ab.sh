#!/bin/bash
set -uo pipefail
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"

extract_bench() {
    "$1" 2>&1 | awk '/--- Performance ---/{p=1} p{print}' | tail -n +2 | head -24
}

for run in 1 2 3; do
    echo "================ RUN $run ================"
    echo "--- v118 (baseline) ---"
    extract_bench runs/fa_v118_localfix
    echo "--- v118b (addrhoist) ---"
    extract_bench runs/fa_v118b_addrhoist
    echo "--- v117 (baseline) ---"
    extract_bench runs/fa_v117_partial_top
    echo "--- v117b (localfix) ---"
    extract_bench runs/fa_v117b_localfix
done
