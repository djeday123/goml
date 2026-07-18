#!/bin/bash
# 040 E2E 5-run in-chain (леджер)
BIN=/data/lib/podman-data/projects/goml/libs/bench_r2c_e2e
LOG=/data/lib/podman-data/projects/goml/runs/reports/040_e2e_inchain_data.txt
> "$LOG"
for i in 1 2 3 4; do "$BIN" 2>&1 | grep "total=" > /dev/null; done
echo "=== 040 E2E 5-run in-chain (isolated + full chain) ===" | tee -a "$LOG"
for i in 1 2 3 4 5; do
    TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    LINE=$("$BIN" 2>&1 | grep "total=")
    echo "run=$i temp=${TEMP}C mode=in-chain $LINE" | tee -a "$LOG"
done
