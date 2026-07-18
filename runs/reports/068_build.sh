#!/bin/bash
# 068 gate step 1: rebuild all three bin — bench + 2 bit-exact.
export HOME=/tmp
cd /data/lib/podman-data/projects/goml/libs

LOG=/data/lib/podman-data/projects/goml/runs/reports/068_build.txt
> "$LOG"

for MF in Makefile.bench_r2c_e2e Makefile.r2c_merged_bit_exact Makefile.r1b_dk_bit_exact; do
    echo "=== $MF ===" | tee -a "$LOG"
    make -f "$MF" clean 2>&1 | tee -a "$LOG"
    make -f "$MF"       2>&1 | tee -a "$LOG"
done
echo "done" | tee -a "$LOG"
