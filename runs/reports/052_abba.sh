#!/bin/bash
# 052 ABBA merged isolated: base (040) vs cand (052 smQ prefetch), 8 pairs
# One session, 4 warmup + 16 timed (alternating pattern A B B A A B B A ...)
BASE=/data/lib/podman-data/projects/goml/runs/archive/052_pre/r2c_merged_wall_base
CAND=/data/lib/podman-data/projects/goml/runs/archive/052_pre/r2c_merged_wall_cand
LOG=/data/lib/podman-data/projects/goml/runs/reports/052_abba_data.txt
> "$LOG"

echo "=== 052 ABBA merged isolated (base=040 252r, cand=052 250r smQ prefetch) ===" | tee -a "$LOG"

# 4 warmup runs (mix)
echo "-- warmup 4 runs --" | tee -a "$LOG"
for i in 1 2 3 4; do
    T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    if [ $((i % 2)) -eq 0 ]; then BIN=$CAND; TAG=CAND; else BIN=$BASE; TAG=BASE; fi
    OUT=$("$BIN" 2>&1 | grep -E "avg_ms" | head -1)
    echo "warmup=$i temp=${T}C $TAG: $OUT" | tee -a "$LOG"
done

echo "-- 16 timed (8 pairs) --" | tee -a "$LOG"
# ABBA pattern: A B B A A B B A A B B A A B B A (block of 4 self-inverse)
# where A=BASE, B=CAND
PATTERN=(BASE CAND CAND BASE BASE CAND CAND BASE BASE CAND CAND BASE BASE CAND CAND BASE)
for i in $(seq 0 15); do
    TAG=${PATTERN[$i]}
    if [ "$TAG" = "BASE" ]; then BIN=$BASE; else BIN=$CAND; fi
    T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    OUT=$("$BIN" 2>&1 | grep -E "avg_ms" | head -1)
    echo "timed=$((i+1)) temp=${T}C $TAG: $OUT" | tee -a "$LOG"
done
