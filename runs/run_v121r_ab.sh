#!/bin/bash
set -uo pipefail
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"
"$GOML/runs/_build_v121r.sh"
echo ""
extract_bench() {
    "$1" 2>&1 | awk '/--- Performance ---/{p=1} p{print}' | tail -n +2 | head -24
}
for run in 1 2 3; do
    echo "================ RUN $run ================"
    echo "--- v121 (baseline) ---"
    extract_bench runs/fa_v121_addrhoist
    echo "--- v121r (R1+R2 diet, 2 blocks) ---"
    extract_bench runs/fa_v121r_diet
done
