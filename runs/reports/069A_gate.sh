#!/bin/bash
# 069A gate: bit-exact 11/11 nc+causal x3 + canary + memcheck.
export HOME=/tmp
export CUDA_MODULE_LOADING=LAZY
cd /data/lib/podman-data/projects/goml/libs

LOG=/data/lib/podman-data/projects/goml/runs/reports/069A_gate.txt
> "$LOG"

echo "=== r1b_dk_bit_exact x3 (nc + causal per form via harness) ===" | tee -a "$LOG"
for i in 1 2 3; do
    echo "--- pass $i ---" | tee -a "$LOG"
    ./r1b_dk_bit_exact 2>&1 | tee -a "$LOG"
done

echo "=== r2c_merged_bit_exact x3 (nc + causal per form via harness) ===" | tee -a "$LOG"
for i in 1 2 3; do
    echo "--- pass $i ---" | tee -a "$LOG"
    ./r2c_merged_bit_exact 2>&1 | tee -a "$LOG"
done

echo "=== canary via r1b_dk_bit_exact --inject (BITFLIP catch) ===" | tee -a "$LOG"
./r1b_dk_bit_exact --inject 2>&1 | tee -a "$LOG"

echo "=== cuda-memcheck (one pass, quick form) ===" | tee -a "$LOG"
/usr/local/cuda-13.1/bin/compute-sanitizer --tool memcheck ./r1b_dk_bit_exact 2>&1 | tee -a "$LOG" | tail -30

echo "done" | tee -a "$LOG"
