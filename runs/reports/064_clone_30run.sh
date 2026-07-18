#!/bin/bash
BIN=/data/lib/podman-data/projects/goml/release_verify_clone/bin/bench_r2c_e2e
LOG_NC=/data/lib/podman-data/projects/goml/runs/reports/064_clone_30run_nc.txt
LOG_C=/data/lib/podman-data/projects/goml/runs/reports/064_clone_30run_causal.txt
> "$LOG_NC"; > "$LOG_C"

FOREIGN=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -vE "r2c_merged_wall|bench_r2c_e2e|r2c_merged_bit_exact|r1b_dk_bit_exact|ldmatrix_|S2v4_" | grep -v "^$")
if [ -n "$FOREIGN" ]; then echo "GATE-ABORT: $FOREIGN"; exit 2; fi

echo "=== 064 clean-clone E2E R2C 30-run nc ===" | tee -a "$LOG_NC"
echo "-- warmup 4 --" | tee -a "$LOG_NC"
for i in 1 2 3 4; do
    T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    OUT=$("$BIN" 2>&1 | grep "total=")
    echo "warmup=$i temp=${T}C $OUT" | tee -a "$LOG_NC"
done
echo "-- 30 timed --" | tee -a "$LOG_NC"
for i in $(seq 1 30); do
    T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    OUT_FULL=$("$BIN" 2>&1)
    OUT=$(echo "$OUT_FULL" | grep "total=")
    TF16=$(echo "$OUT_FULL" | grep "16N^2d" | tail -1 | grep -oE '[0-9]+\.[0-9]+ T')
    TF10=$(echo "$OUT_FULL" | grep "10N^2d" | tail -1 | grep -oE '[0-9]+\.[0-9]+ T')
    echo "run=$i temp=${T}C $OUT tflops16=$TF16 tflops10=$TF10" | tee -a "$LOG_NC"
done

echo "=== 064 clean-clone E2E R2C 30-run causal ===" | tee -a "$LOG_C"
echo "-- warmup 4 --" | tee -a "$LOG_C"
for i in 1 2 3 4; do
    T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    OUT=$(env CAUSAL=1 "$BIN" 2>&1 | grep "total=")
    echo "warmup=$i temp=${T}C $OUT" | tee -a "$LOG_C"
done
echo "-- 30 timed --" | tee -a "$LOG_C"
for i in $(seq 1 30); do
    T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    OUT_FULL=$(env CAUSAL=1 "$BIN" 2>&1)
    OUT=$(echo "$OUT_FULL" | grep "total=")
    TF16=$(echo "$OUT_FULL" | grep "16N^2d" | tail -1 | grep -oE '[0-9]+\.[0-9]+ T')
    echo "run=$i temp=${T}C $OUT tflops16=$TF16" | tee -a "$LOG_C"
done
