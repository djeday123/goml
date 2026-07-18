#!/bin/bash
# 070 gate: fingerprint 252/124/69/38 + bit-exact 11/11 x3 + canary + memcheck + E2E sanity + VRAM delta.
export HOME=/tmp
export CUDA_MODULE_LOADING=LAZY
cd /data/lib/podman-data/projects/goml/libs

LOG=/data/lib/podman-data/projects/goml/runs/reports/070_gate.txt
> "$LOG"

echo "==== §A2.1 Fingerprint + sanity nc ====" | tee -a "$LOG"
./bench_r2c_e2e 2>&1 | head -10 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "==== §A2.2 sanity causal ====" | tee -a "$LOG"
CAUSAL=1 ./bench_r2c_e2e 2>&1 | grep -E "FINGERPRINT|SEQUENTIAL|total=" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "==== §A2.3 r1b_dk_bit_exact x3 ====" | tee -a "$LOG"
for i in 1 2 3; do
    echo "--- pass $i ---" | tee -a "$LOG"
    ./r1b_dk_bit_exact 2>&1 | tail -14 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "==== §A2.4 r2c_merged_bit_exact x3 ====" | tee -a "$LOG"
for i in 1 2 3; do
    echo "--- pass $i ---" | tee -a "$LOG"
    ./r2c_merged_bit_exact 2>&1 | tail -14 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "==== §A2.5 canary ====" | tee -a "$LOG"
./r1b_dk_bit_exact --inject 2>&1 | grep -E "CANARY|BIT-EXACT|MISMATCH" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "==== §A2.6 memcheck ====" | tee -a "$LOG"
/usr/local/cuda-13.1/bin/compute-sanitizer --tool memcheck ./r1b_dk_bit_exact 2>&1 | tail -3 | tee -a "$LOG"

echo "done" | tee -a "$LOG"
