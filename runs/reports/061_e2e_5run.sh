#!/bin/bash
LOG=/data/lib/podman-data/projects/goml/runs/reports/061_e2e_5run.txt
> "$LOG"
for i in 1 2 3 4 5; do
    T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    LINE=$(/data/lib/podman-data/projects/goml/libs/bench_r2c_e2e 2>&1 | grep "total=")
    echo "run=$i temp=${T}C $LINE" | tee -a "$LOG"
done
