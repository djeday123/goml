#!/bin/bash
# 041 I.3: E2E 5-run in-chain per-kernel decomposition
BIN=/data/lib/podman-data/projects/goml/libs/bench_r2c_e2e
GATE=/data/lib/podman-data/projects/goml/runs/reports/037r_gate.sh
LOG=/data/lib/podman-data/projects/goml/runs/reports/041_e2e_decomp_data.txt
> "$LOG"
"$GATE" 2>&1 | tee -a "$LOG"

echo "=== warmup ==="
for i in 1 2 3 4; do "$BIN" 2>&1 | grep "total=" > /dev/null; done

echo "=== 041 I.3 5-run E2E in-chain (per-kernel: D/merged/dk/dq/total) ===" | tee -a "$LOG"
for i in 1 2 3 4 5; do
    TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    LINE=$("$BIN" 2>&1 | grep "total=")
    echo "run=$i temp=${TEMP}C mode=in-chain $LINE" | tee -a "$LOG"
done
