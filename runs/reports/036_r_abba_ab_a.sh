#!/bin/bash
# 036-r ABBA: AB vs A (pack+π_V vs pack no π_V), 8 pairs = 16 runs + 4 warmup
AB=/data/lib/podman-data/projects/goml/libs/bench_r2c_e2e_AB
A=/data/lib/podman-data/projects/goml/libs/bench_r2c_e2e_Aonly
LOG=/data/lib/podman-data/projects/goml/runs/reports/036_r_abba_ab_a_data.txt
> "$LOG"

echo "=== WARMUP (4 runs, discarded) ==="
for i in 1 2 3 4; do
    "$AB" 2>&1 | grep "total=" | head -1 > /dev/null
done

# 8 pairs ABBA: A B B A A B B A A B B A A B B A  (where A=AB, B=Aonly)
SEQ="X Y Y X X Y Y X X Y Y X X Y Y X"
i=0
for tag in $SEQ; do
    i=$((i+1))
    if [ "$tag" = "X" ]; then BIN=$AB; NAME=AB; else BIN=$A; NAME=A; fi
    LINE=$("$BIN" 2>&1 | grep "total=" | head -1)
    TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    echo "run=$i tag=$NAME temp=${TEMP}C $LINE" | tee -a "$LOG"
done
