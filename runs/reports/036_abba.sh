#!/bin/bash
A=/data/lib/podman-data/projects/goml/libs/bench_r2c_e2e_A
B=/data/lib/podman-data/projects/goml/libs/bench_r2c_e2e_B
LOG=/data/lib/podman-data/projects/goml/runs/reports/036_abba_data.txt
> "$LOG"

# 4 warmup runs (A×2 + B×2, discard)
echo "=== WARMUP (4 runs, discarded) ==="
for i in 1 2 3 4; do
    "$A" 2>&1 | grep "total=" | head -1 > /dev/null
done

# 8 pairs ABBA scheme: A B B A A B B A A B B A A B B A
SEQ="A B B A A B B A A B B A A B B A"
i=0
for tag in $SEQ; do
    i=$((i+1))
    if [ "$tag" = "A" ]; then BIN=$A; else BIN=$B; fi
    LINE=$("$BIN" 2>&1 | grep "total=" | head -1)
    TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    echo "run=$i tag=$tag temp=${TEMP}C $LINE" | tee -a "$LOG"
done
