#!/bin/bash
BIN=/data/lib/podman-data/projects/goml/libs/bench_r2c_e2e
LOG=/data/lib/podman-data/projects/goml/runs/reports/036_calib_data.txt
> "$LOG"
for i in $(seq 1 12); do
    T_ISO=$(date +%s.%3N)
    TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    CLK=$(nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv,noheader,nounits | head -1)
    LINE=$("$BIN" 2>&1 | grep "D=" | head -1)
    echo "run=$i t=$T_ISO temp=${TEMP}C clk=${CLK} $LINE" | tee -a "$LOG"
done
