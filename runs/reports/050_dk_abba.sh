#!/bin/bash
# 050 §3.d ABBA 8 пар dk isolated (in-chain через bench_r2c_e2e)
BASE=/data/lib/podman-data/projects/goml/runs/archive/050_pre/bench_r2c_e2e
CAND=/data/lib/podman-data/projects/goml/libs/bench_r2c_e2e
LOG=/data/lib/podman-data/projects/goml/runs/reports/050_dk_abba_data.txt
> "$LOG"

echo "=== fingerprints ===" | tee -a "$LOG"
"$BASE" 2>&1 | head -6 | tee -a "$LOG"
"$CAND" 2>&1 | head -6 | tee -a "$LOG"

echo "=== WARMUP (4 runs baseline) ==="
for i in 1 2 3 4; do "$BASE" 2>&1 | grep "total=" > /dev/null; done

echo "=== ABBA 8 пар (B=baseline_pre, C=candidate_050) ===" | tee -a "$LOG"
SEQ="B C C B B C C B B C C B B C C B"
i=0
for tag in $SEQ; do
    i=$((i+1))
    if [ "$tag" = "B" ]; then BIN=$BASE; NAME=BASE; else BIN=$CAND; NAME=CAND; fi
    TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    LINE=$("$BIN" 2>&1 | grep "total=")
    echo "run=$i tag=$NAME temp=${TEMP}C $LINE" | tee -a "$LOG"
done
