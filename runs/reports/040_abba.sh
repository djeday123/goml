#!/bin/bash
# 040 ABBA 8 пар: pre-baseline (archived) vs candidate (current build)
BASELINE=/data/lib/podman-data/projects/goml/runs/archive/040_pre/r2c_merged_wall
CANDIDATE=/data/lib/podman-data/projects/goml/libs/r2c_merged_wall
GATE=/data/lib/podman-data/projects/goml/runs/reports/037r_gate.sh
LOG=/data/lib/podman-data/projects/goml/runs/reports/040_abba_data.txt
> "$LOG"

# Gate check (только candidate)
"$GATE" >> "$LOG" 2>&1 || { echo "gate failed"; exit 1; }

echo "=== WARMUP (4 runs baseline, discarded) ==="
for i in 1 2 3 4; do "$BASELINE" 2>&1 | grep avg_ms > /dev/null; done

# ABBA 8 pairs: BASE CAND CAND BASE  BASE CAND CAND BASE  ...
SEQ="B C C B B C C B B C C B B C C B"
i=0
for tag in $SEQ; do
    i=$((i+1))
    if [ "$tag" = "B" ]; then BIN=$BASELINE; NAME=BASE; else BIN=$CANDIDATE; NAME=CAND; fi
    TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    LINE=$("$BIN" 2>&1 | grep avg_ms)
    echo "run=$i tag=$NAME temp=${TEMP}C mode=isolated $LINE" | tee -a "$LOG"
done
