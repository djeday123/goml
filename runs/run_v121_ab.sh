#!/bin/bash
# v96b vs v121 same-thermal A/B. Чередуем запуски, чтобы thermal drift не смазал результаты.
set -uo pipefail
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"

echo "=== Same-thermal A/B v96b vs v121 (interleaved 3 runs each) ==="
extract_bench() {
    "$1" 2>&1 | awk '/--- Performance ---/{p=1} p{print}' | tail -n +2
}

for run in 1 2 3; do
    echo "================ RUN $run ================"
    echo "--- v96b ---"
    extract_bench runs/fa_v96b_localfix | head -24
    echo "--- v121 ---"
    extract_bench runs/fa_v121_addrhoist | head -24
done
