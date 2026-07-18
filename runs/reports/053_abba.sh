#!/bin/bash
# 053 ABBA merged isolated: base (040 sealed 252r) vs cand (053 254r dO half-prefetch)
# 4 warmup + 16 timed (8 pairs) alternating BABAABBA...
BASE=/data/lib/podman-data/projects/goml/runs/archive/053_pre/r2c_merged_wall_base
CAND=/data/lib/podman-data/projects/goml/runs/archive/053_pre/r2c_merged_wall_cand
LOG=/data/lib/podman-data/projects/goml/runs/reports/053_abba_data.txt
> "$LOG"

echo "=== 053 ABBA merged isolated (base=040 252r, cand=053 254r dO half-prefetch) ===" | tee -a "$LOG"

echo "-- warmup 4 runs --" | tee -a "$LOG"
for i in 1 2 3 4; do
    T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    if [ $((i % 2)) -eq 0 ]; then BIN=$CAND; TAG=CAND; else BIN=$BASE; TAG=BASE; fi
    OUT=$("$BIN" 2>&1 | grep -E "avg_ms" | head -1)
    echo "warmup=$i temp=${T}C $TAG: $OUT" | tee -a "$LOG"
done

echo "-- 16 timed (8 pairs) --" | tee -a "$LOG"
PATTERN=(BASE CAND CAND BASE BASE CAND CAND BASE BASE CAND CAND BASE BASE CAND CAND BASE)
for i in $(seq 0 15); do
    TAG=${PATTERN[$i]}
    if [ "$TAG" = "BASE" ]; then BIN=$BASE; else BIN=$CAND; fi
    T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    OUT=$("$BIN" 2>&1 | grep -E "avg_ms" | head -1)
    echo "timed=$((i+1)) temp=${T}C $TAG: $OUT" | tee -a "$LOG"
done
