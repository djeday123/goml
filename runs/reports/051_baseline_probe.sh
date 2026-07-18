#!/bin/bash
BIN=/data/lib/podman-data/projects/goml/libs/bench_r2c_e2e
LOG=/data/lib/podman-data/projects/goml/runs/reports/051_baseline_probe_data.txt
> "$LOG"
echo "=== baseline dk (production 033 sealed) 5 consecutive runs ==="  | tee -a "$LOG"
for i in 1 2 3 4 5; do
    T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    L=$("$BIN" 2>&1 | grep "total=")
    echo "run=$i temp=${T}C $L" | tee -a "$LOG"
done
