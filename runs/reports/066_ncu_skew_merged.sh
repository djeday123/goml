#!/bin/bash
# NCu per-SM cycle-active spread for kernel_merged_v1 under causal.
export CAUSAL=1
export CUDA_MODULE_LOADING=LAZY
cd /data/lib/podman-data/projects/goml

OUT=/data/lib/podman-data/projects/goml/runs/reports/066_ncu_skew_merged.csv
METRICS="sm__cycles_active.avg,sm__cycles_active.max,sm__cycles_active.min,sm__cycles_active.sum,smsp__cycles_active.avg,smsp__cycles_active.max"

/usr/bin/ncu \
    --target-processes all \
    --kernel-name kernel_merged_v1 \
    --launch-count 1 \
    --metrics "$METRICS" \
    --csv \
    libs/bench_r2c_e2e > "$OUT" 2>&1

echo "done: $OUT"
