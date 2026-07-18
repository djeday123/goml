#!/bin/bash
LOG=/data/lib/podman-data/projects/goml/runs/reports/057_control_e2e_after_kill.txt
> "$LOG"
for i in 1 2 3 4 5 6 7 8 9; do
    T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    L=$(/data/lib/podman-data/projects/goml/libs/bench_r2c_e2e 2>&1 | grep "total=")
    echo "run=$i temp=${T}C $L" | tee -a "$LOG"
done
