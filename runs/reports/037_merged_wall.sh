#!/bin/bash
# 037 phase 0a: merged isolated wall stand-protocol (4 warmup + 5 measured)
BIN=/data/lib/podman-data/projects/goml/libs/r2c_merged_wall
LOG=/data/lib/podman-data/projects/goml/runs/reports/037_merged_wall_data.txt
> "$LOG"
echo "=== WARMUP (4 runs, discarded) ==="
for i in 1 2 3 4; do "$BIN" 2>&1 | grep avg_ms > /dev/null; done
echo "=== MEASURED (5 runs) ==="
for i in 1 2 3 4 5; do
    TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    LINE=$("$BIN" 2>&1 | grep avg_ms)
    echo "run=$i temp=${TEMP}C $LINE" | tee -a "$LOG"
done
