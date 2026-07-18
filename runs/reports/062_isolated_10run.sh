#!/bin/bash
LOG=/data/lib/podman-data/projects/goml/runs/reports/062_isolated_10run.txt
> "$LOG"
FOREIGN=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -vE "r2c_merged_wall|bench_r2c_e2e|r2c_merged_bit_exact|r1b_dk_bit_exact|ldmatrix_|S2v4_" | grep -v "^$")
if [ -n "$FOREIGN" ]; then echo "GATE-ABORT: $FOREIGN"; exit 2; fi

echo "=== 062 Isolated merged x10-run (r2c_merged_wall) ===" | tee -a "$LOG"
for i in $(seq 1 10); do
    T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    OUT=$(/data/lib/podman-data/projects/goml/libs/r2c_merged_wall 2>&1)
    FP=$(echo "$OUT" | grep FINGERPRINT | head -1)
    AVG=$(echo "$OUT" | grep "avg_ms" | head -1)
    echo "run=$i temp=${T}C $AVG (fp: $FP)" | tee -a "$LOG"
done
