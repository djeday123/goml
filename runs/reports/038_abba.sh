#!/bin/bash
# 038 ABBA: post-E (current binary) vs 037-r fresh binary from archive
# For control experiment: rebuild pre-E binary in tmp, run ABBA schedule
GATE=/data/lib/podman-data/projects/goml/runs/reports/037r_gate.sh
BIN=/data/lib/podman-data/projects/goml/libs/r2c_merged_wall
LOG=/data/lib/podman-data/projects/goml/runs/reports/038_abba_data.txt
> "$LOG"
"$GATE" >> "$LOG" 2>&1 || { echo "gate failed"; exit 1; }
# Note: current BIN is post-E (fresh 038 build).
# Compare via multiple ABBA-like independent runs, expect drift bounds.

echo "=== ABBA-schedule 8 pair post-E (drift control, single binary) ===" | tee -a "$LOG"
for i in 1 2 3 4; do "$BIN" 2>&1 | grep avg_ms > /dev/null; done  # warmup
for i in 1 2 3 4 5 6 7 8; do
    TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    LINE=$("$BIN" 2>&1 | grep avg_ms)
    echo "run=$i temp=${TEMP}C mode=isolated $LINE" | tee -a "$LOG"
done
