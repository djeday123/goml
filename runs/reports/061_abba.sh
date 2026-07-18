#!/bin/bash
BASE=/data/lib/podman-data/projects/goml/runs/archive/061_pre/bench_r2c_e2e_base
CAND=/data/lib/podman-data/projects/goml/runs/archive/061_pre/bench_r2c_e2e_cand
LOG=/data/lib/podman-data/projects/goml/runs/reports/061_abba_data.txt
> "$LOG"
echo "=== 061 S2v4 ABBA: base(033 sealed 128r) vs cand(S2v4 124r LDSM+свизл) ===" | tee -a "$LOG"

# Gate-тишина проверка ДО ABBA
FOREIGN=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -vE "r2c_merged_wall|bench_r2c_e2e|r2c_merged_bit_exact|r1b_dk_bit_exact|ldmatrix_|S2v4_" | grep -v "^$")
if [ -n "$FOREIGN" ]; then
    echo "GATE-ТИШИНА ABORT: foreign GPU tenants:" | tee -a "$LOG"
    echo "$FOREIGN" | tee -a "$LOG"
    exit 2
fi
echo "gate-тишина OK: чужих GPU-tenants нет" | tee -a "$LOG"

echo "-- warmup 4 --" | tee -a "$LOG"
for i in 1 2 3 4; do
    T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    if [ $((i % 2)) -eq 0 ]; then BIN=$CAND; TAG=CAND; else BIN=$BASE; TAG=BASE; fi
    OUT=$("$BIN" 2>&1 | grep "total=" | head -1)
    echo "warmup=$i temp=${T}C $TAG: $OUT" | tee -a "$LOG"
done
echo "-- 16 timed (8 pairs) --" | tee -a "$LOG"
PATTERN=(BASE CAND CAND BASE BASE CAND CAND BASE BASE CAND CAND BASE BASE CAND CAND BASE)
for i in $(seq 0 15); do
    TAG=${PATTERN[$i]}
    if [ "$TAG" = "BASE" ]; then BIN=$BASE; else BIN=$CAND; fi
    T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    OUT=$("$BIN" 2>&1 | grep "total=" | head -1)
    echo "timed=$((i+1)) temp=${T}C $TAG: $OUT" | tee -a "$LOG"
done
