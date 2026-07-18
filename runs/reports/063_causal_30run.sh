#!/bin/bash
LOG=/data/lib/podman-data/projects/goml/runs/reports/063_causal_30run.txt
> "$LOG"
FOREIGN=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -vE "r2c_merged_wall|bench_r2c_e2e|r2c_merged_bit_exact|r1b_dk_bit_exact|ldmatrix_|S2v4_" | grep -v "^$")
if [ -n "$FOREIGN" ]; then echo "GATE-ABORT: $FOREIGN"; exit 2; fi
echo "=== 063 E2E R2C causal=1 30-run (CAUSAL=1 env, fingerprint per run) ===" | tee -a "$LOG"
echo "-- warmup 4 --" | tee -a "$LOG"
for i in 1 2 3 4; do
    T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    OUT=$(env CAUSAL=1 /data/lib/podman-data/projects/goml/libs/bench_r2c_e2e 2>&1)
    FP=$(echo "$OUT" | grep MISMATCH | head -1)
    if [ -n "$FP" ]; then echo "warmup=$i FP FAIL: $FP"; exit 3; fi
    LINE=$(echo "$OUT" | grep "total=")
    echo "warmup=$i temp=${T}C $LINE" | tee -a "$LOG"
done
echo "-- 30 timed --" | tee -a "$LOG"
for i in $(seq 1 30); do
    T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    OUT=$(env CAUSAL=1 /data/lib/podman-data/projects/goml/libs/bench_r2c_e2e 2>&1)
    FP=$(echo "$OUT" | grep MISMATCH | head -1)
    if [ -n "$FP" ]; then echo "run=$i FP FAIL: $FP"; exit 3; fi
    LINE=$(echo "$OUT" | grep "total=")
    TF16=$(echo "$OUT" | grep "16N^2d" | tail -1 | grep -oE '[0-9]+\.[0-9]+ T')
    echo "run=$i temp=${T}C $LINE tflops16=$TF16" | tee -a "$LOG"
done
