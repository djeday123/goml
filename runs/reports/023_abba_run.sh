#!/bin/bash
# 023 ABBA series: ABBA BAAB ABBA BAAB (16 points) + 1 warmup
# Each point = median of 5 binary launches
A_BIN=/data/lib/podman-data/projects/goml/libs/r1b_dk_wall_A_sealed
B_BIN=/data/lib/podman-data/projects/goml/libs/r1b_dk_wall_B_piv

run_point() {
    local bin=$1
    local label=$2
    local vals=""
    for k in 1 2 3 4 5; do
        v=$("$bin" 2>&1 | grep avg_ms | awk '{split($2, a, "="); print a[2]}')
        vals="$vals $v"
    done
    # median: sort, pick middle
    median=$(echo "$vals" | tr ' ' '\n' | sort -n | awk 'NR==3')
    echo "$label $median  runs:$vals"
}

echo "=== WARMUP (not counted) ==="
run_point "$A_BIN" "warm"

echo ""
echo "=== ABBA BAAB ABBA BAAB (16 points) ==="
SEQ="A B B A B A A B A B B A B A A B"
i=0
for tag in $SEQ; do
    i=$((i+1))
    if [ "$tag" = "A" ]; then bin=$A_BIN; else bin=$B_BIN; fi
    run_point "$bin" "p${i}(${tag})"
done
