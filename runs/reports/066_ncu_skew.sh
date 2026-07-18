#!/bin/bash
# NCu skew probe: per-SM cycle-active for all three iter-loop kernels under causal.
# Working pattern from 063r_ncu_work.sh (uses CUDA 13.1 ncu + HOME=/tmp).
# Metric: sm__cycles_active.{avg,max,min,sum} → skew ratio = max/avg quantifies wave-tail dominance.

export HOME=/tmp
export CUDA_MODULE_LOADING=LAZY
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/libs/bench_r2c_e2e
LOG=/data/lib/podman-data/projects/goml/runs/reports/066_ncu_skew.txt
> "$LOG"

METRICS="sm__cycles_active.avg,sm__cycles_active.max,sm__cycles_active.min,sm__cycles_active.sum,launch__grid_size,launch__block_size,launch__waves_per_multiprocessor"

for MODE in NC CAUSAL; do
    if [ "$MODE" = "NC" ]; then ENV_CAUSAL="0"; else ENV_CAUSAL="1"; fi
    echo "=========================================" | tee -a "$LOG"
    echo "===== MODE=$MODE (CAUSAL=$ENV_CAUSAL) =====" | tee -a "$LOG"
    echo "=========================================" | tee -a "$LOG"
    for KERNEL in kernel_merged_v1 kernel_dk_new kernel_dq_new; do
        echo "----- $KERNEL -----" | tee -a "$LOG"
        env CAUSAL="$ENV_CAUSAL" "$NCU" --kernel-name "$KERNEL" --launch-count 1 --metrics "$METRICS" "$BIN" 2>&1 | tail -18 | tee -a "$LOG"
    done
done
echo "done"
