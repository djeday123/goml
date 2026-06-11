#!/bin/bash
set -uo pipefail
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"
echo "=== Same-thermal A/B v96b / v121 / v122 (interleaved 3 runs each) ==="
extract_bench() {
    "$1" 2>&1 | awk '/--- Performance ---/{p=1} p{print}' | tail -n +2
}
for run in 1 2 3; do
    echo "================ RUN $run ================"
    echo "--- v96b ---"
    extract_bench runs/fa_v96b_localfix | head -24
    echo "--- v121 ---"
    extract_bench runs/fa_v121_addrhoist | head -24
    echo "--- v122 ---"
    extract_bench runs/fa_v122_br64_mt1 | head -24
done
