#!/bin/bash
# 036-r ABBA: AB vs 0 (pack+π_V vs byte no π_V), 8 pairs = 16 runs + 4 warmup
AB=/data/lib/podman-data/projects/goml/libs/bench_r2c_e2e_AB
Z=/data/lib/podman-data/projects/goml/libs/bench_r2c_e2e_0
LOG=/data/lib/podman-data/projects/goml/runs/reports/036_r_abba_ab_0_data.txt
> "$LOG"

echo "=== WARMUP (4 runs, discarded) ==="
for i in 1 2 3 4; do
    "$AB" 2>&1 | grep "total=" | head -1 > /dev/null
done

# 8 pairs ABBA: X Y Y X X Y Y X X Y Y X X Y Y X  (X=AB, Y=0)
SEQ="X Y Y X X Y Y X X Y Y X X Y Y X"
i=0
for tag in $SEQ; do
    i=$((i+1))
    if [ "$tag" = "X" ]; then BIN=$AB; NAME=AB; else BIN=$Z; NAME=0; fi
    LINE=$("$BIN" 2>&1 | grep "total=" | head -1)
    TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    echo "run=$i tag=$NAME temp=${TEMP}C $LINE" | tee -a "$LOG"
done
