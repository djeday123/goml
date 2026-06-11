#!/bin/bash
# Полная проверка диспетчера: hd=128 wave-tail + small-grid bench.
# Сравниваем v96b / v121 / v122 / v118 / v111 / v117 на тех же конфигурациях.
set -uo pipefail
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"

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
    echo "--- v118 (bh=4 niche) ---"
    extract_bench runs/fa_v118_localfix | head -24
    echo "--- v111-warpspec-mbarrier (sliding window niche) ---"
    extract_bench runs/fa_v111_warpspec_mbarrier | head -24
    echo "--- v117 (partial top sync) ---"
    extract_bench runs/fa_v117_partial_top | head -24
done
